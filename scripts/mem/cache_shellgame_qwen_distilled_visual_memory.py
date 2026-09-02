#!/usr/bin/env python3
# ruff: noqa: E402
"""Cache held-out final memories from the Qwen-distilled direct-visual MEM.

This is the first half of the direct-MEM -> action compatibility experiment.
The student sees the fixed 60-frame observation history and receives only the
initial reveal slot as its recurrent initial state.  It does not receive swap
relations or the final cup label.  The initial slot is the interface supplied
by the already validated Qwen reveal classifier in the full system.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from examples.shellgame import train_qwen_distilled_direct_visual_recurrent_memory_probe as probe
from openpi.models import model as model_lib
from openpi.training import checkpoints
from openpi.training import config_pi0_mem
from openpi.training import sharding
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as memory_recipe
from scripts.mem import train_semantic_memory as memory_trainer

DEFAULT_EXP_NAME = "qwen_distilled_direct_visual_memory250_260825"
DEFAULT_OUTPUT = Path("artifacts/shellgame_qwen_distilled_direct_visual_memory_step999_val240_260825.npz")
DEFAULT_ALL_OUTPUT = Path("artifacts/shellgame_qwen_distilled_direct_visual_memory_step999_all5000_260825.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument(
        "--student-segment-size",
        type=int,
        default=probe._semantic.SWAP_SEGMENT_SIZE,  # noqa: SLF001
        help="Rebuild the tracker with the temporal clip length used by the checkpoint.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_memory_config(args: argparse.Namespace):
    """Recreate the exact step-999 graph while retaining prompt tokens."""
    build_args = SimpleNamespace(
        exp_name=args.exp_name,
        teacher_checkpoint=probe.DEFAULT_TEACHER_CHECKPOINT,
        steps=1000,
        warmup_steps=0,
        peak_lr=4e-5,
        decay_lr=4e-6,
        memory_distill_weight=1.0,
        stage_slot_weight=0.25,
        shuffle_teacher_targets=False,
        # Keep cache/eval graph construction in sync with the training entry
        # after the recurrent student gained configurable clip length.
        student_segment_size=args.student_segment_size,
        batch_size=args.batch_size,
        num_workers=0,
        fsdp_devices=args.fsdp_devices,
        eval_interval=100,
        eval_batches=args.batches,
        save_interval=500,
        keep_period=499,
        overwrite=False,
        resume=True,
    )
    config = probe.build_config(build_args)
    child = dataclasses.replace(config.data.datasets[0], tokenize_prompt=True)
    return dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, datasets=[child]),
        num_workers=0,
    )


def predict_step(label_table, state, batch):
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, _actions = batch
    episode = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode]
    processed = model_lib.preprocess_observation(None, observation, train=False)
    outputs = model.track_history(
        processed,
        initial_slots=labels[:, 0],
        teacher_relation_ids=None,
        train=False,
    )
    final_logits = outputs["stage_logits"][:, -1]
    final_memory = outputs["student_memories"][:, -1]
    final_labels = labels[:, 1 + memory_recipe.NUM_STAGES + memory_recipe.NUM_STAGES - 1]
    return {
        "episode_index": episode,
        "initial_slot": labels[:, 0],
        "final_label": final_labels,
        "final_prediction": jnp.argmax(final_logits, axis=-1),
        "final_memory": final_memory.astype(jnp.float32),
    }


def _full_loader(config, data_sharding):
    data_config = config.data.create_all(config.assets_dirs, config.model)[0]
    child = config.data.datasets[0]
    dataset = config_pi0_mem._build_pi0_mem_dataset(  # noqa: SLF001
        data_config,
        child.video_frame_config(),
        action_horizon=config.model.action_horizon,
        skip_norm_stats=False,
    )
    dataset_size = len(dataset)
    padding = (-dataset_size) % config.batch_size
    if padding:
        # TorchDataLoader drops an incomplete final batch.  Append only enough
        # deterministic duplicates to preserve every real episode, then remove
        # duplicates by episode_index after inference.
        dataset = torch.utils.data.ConcatDataset([dataset, torch.utils.data.Subset(dataset, list(range(padding)))])
    loader = memory_trainer._make_loader(  # noqa: SLF001
        config, data_config, dataset, data_sharding, shuffle=False
    )
    return loader, dataset_size


def main() -> None:
    args = parse_args()
    if args.all_episodes and args.output == DEFAULT_OUTPUT:
        args.output = DEFAULT_ALL_OUTPUT
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    if args.batches <= 0:
        raise ValueError("--batches must be positive")

    config = make_memory_config(args)
    label_table = memory_recipe.load_episode_label_table(config)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if args.all_episodes:
        inference_loader, dataset_size = _full_loader(config, data_sharding)
        num_batches = (dataset_size + args.batch_size - 1) // args.batch_size
    else:
        _train_loader, inference_loader = memory_trainer.create_train_val_data_loaders(config, data_sharding)
        dataset_size = args.batches * args.batch_size
        num_batches = args.batches
    inference_iter = iter(inference_loader)

    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    if not resuming:
        raise FileNotFoundError(f"No checkpoint found under {config.checkpoint_dir}")
    _rng, init_rng = jax.random.split(jax.random.key(config.seed))
    state, state_sharding = memory_trainer.init_train_state(config, init_rng, mesh, resume=True)
    state = checkpoints.restore_state(manager, state, inference_loader)
    jax.block_until_ready(state)

    ppredict = jax.jit(
        predict_step,
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=replicated,
    )
    rows = []
    with sharding.set_mesh(mesh):
        for batch_index in range(num_batches):
            rows.append(jax.device_get(ppredict(label_table, state, next(inference_iter))))
            print(f"cached {batch_index + 1}/{num_batches} batches", flush=True)
    stacked = jax.tree.map(lambda *values: np.concatenate(values, axis=0), *rows)
    episode = np.asarray(stacked["episode_index"], dtype=np.int32)
    unique_episode, first_index = np.unique(episode, return_index=True)
    if len(unique_episode) != dataset_size:
        raise ValueError(f"Expected {dataset_size} unique fixed-prefix episodes, got {len(unique_episode)}")
    order = first_index[np.argsort(unique_episode)]
    payload = {key: np.asarray(value)[order] for key, value in stacked.items()}
    episode = payload["episode_index"]
    accuracy = float(np.mean(payload["final_prediction"] == payload["final_label"]))
    metadata = {
        "source_checkpoint": str(config.checkpoint_dir / "999"),
        "inference_contract": "60 history images + Qwen reveal initial slot -> final [128,64] memory",
        "uses_swap_or_final_gt_as_model_input": False,
        "initial_slot_source_in_this_cache": "simulator label as exact proxy for validated Qwen reveal output",
        "batches": num_batches,
        "episodes": len(episode),
        "final_slot_accuracy": accuracy,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        **payload,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if args.all_episodes:
        dense_episode_to_template = np.full(int(np.max(episode)) + 1, -1, dtype=np.int32)
        dense_episode_to_template[episode] = np.arange(len(episode), dtype=np.int32)
        if np.any(dense_episode_to_template < 0):
            raise ValueError("Full memory bank requires dense episode indices")
        output_payload.update(
            memory_templates=payload["final_memory"],
            episode_template_index=dense_episode_to_template,
        )
    np.savez_compressed(output, **output_payload)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
