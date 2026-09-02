#!/usr/bin/env python3
"""Test whether the EEF action expert can directly consume visual MEM tokens.

The action checkpoint, current image/state, target action chunk, and diffusion
noise are identical across conditions.  Only the external [128,64] memory is
changed: symbolic teacher, Qwen-distilled direct visual, wrong visual memory,
or zeros.  No action parameters are updated by this diagnostic.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flax.training import common_utils
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train_pi0_mem_compress as action_trainer
from openpi.training import config as training_config
from openpi.training import sharding
from openpi.training import weight_loaders
from openpi.training.mem.recipes import shellgame_qwen_event_memory_action as action_recipe
from scripts.mem import cache_shellgame_qwen_distilled_visual_memory as memory_cache
from scripts.mem import train_semantic_memory as memory_trainer


DEFAULT_ACTION_CHECKPOINT = Path(
    "checkpoints/pi0_shellgame_qwen_event_memory_action_eef7_260825/"
    "qwen_event_mem_v10init_action250_6gpu_260825/249/params"
)
DEFAULT_PREDICTED_MEMORY = memory_cache.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path(
    "evaluation/shellgame/qwen_distilled_direct_visual_memory_action_compatibility_260825.json"
)
CONDITIONS = ("teacher", "direct_visual", "wrong_visual", "zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-checkpoint", type=Path, default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument("--predicted-memory", type=Path, default=DEFAULT_PREDICTED_MEMORY)
    parser.add_argument("--teacher-memory", type=Path, default=action_recipe.DEFAULT_MEMORY_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_predicted(path: Path):
    with np.load(path, allow_pickle=False) as source:
        episodes = np.asarray(source["episode_index"], dtype=np.int32)
        memories = np.asarray(source["final_memory"], dtype=np.float32)
        labels = np.asarray(source["final_label"], dtype=np.int32)
        predictions = np.asarray(source["final_prediction"], dtype=np.int32)
        metadata = json.loads(str(np.asarray(source["metadata_json"]).reshape(())))
    if memories.shape != (len(episodes), 128, 64):
        raise ValueError(f"Invalid predicted memories: {memories.shape}")
    size = int(np.max(episodes)) + 1
    dense = np.full((size, 128, 64), np.nan, dtype=np.float32)
    final_labels = np.full((size,), -1, dtype=np.int32)
    final_predictions = np.full((size,), -1, dtype=np.int32)
    dense[episodes] = memories
    final_labels[episodes] = labels
    final_predictions[episodes] = predictions

    # Pick a deterministic cached donor with a different ground-truth final cup.
    donor = np.full((size,), -1, dtype=np.int32)
    for episode in episodes:
        candidates = episodes[labels != final_labels[episode]]
        donor[episode] = candidates[int(episode) % len(candidates)]
    return dense, donor, final_labels, final_predictions, metadata


def _load_teacher(path: Path):
    with np.load(path, allow_pickle=False) as source:
        templates = np.asarray(source["memory_templates"], dtype=np.float32)
        episode_to_template = np.asarray(source["episode_template_index"], dtype=np.int32)
    return templates, episode_to_template


def _current_only(observation):
    images = {
        key: value[:, -1:] if value.ndim == 5 else value
        for key, value in observation.images.items()
    }
    frame_valid_masks = {
        key: value[:, -1:] if value is not None and value.ndim == 2 else value
        for key, value in observation.frame_valid_masks.items()
    }
    return observation.replace(images=images, frame_valid_masks=frame_valid_masks)


def eval_step(predicted, donor, teacher_templates, teacher_indices, rng, state, batch):
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, actions = batch
    observation = _current_only(observation)
    episode = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    direct_memory = predicted[episode]
    teacher_memory = teacher_templates[teacher_indices[episode]]
    memories = {
        "teacher": teacher_memory,
        "direct_visual": direct_memory,
        "wrong_visual": predicted[donor[episode]],
        "zero": jnp.zeros_like(direct_memory),
    }
    result = {"episode_index": episode}
    for condition in CONDITIONS:
        variant = observation.replace(semantic_memory=memories[condition])
        chunk_loss, _aux = model.compute_loss_with_memory_aux(rng, variant, actions, train=False)
        per_sample = jnp.mean(chunk_loss, axis=-1)
        result[condition] = {
            "loss_sum": jnp.sum(per_sample),
            "count": jnp.asarray(per_sample.size, dtype=jnp.int32),
        }
    student = direct_memory.astype(jnp.float32)
    teacher = teacher_memory.astype(jnp.float32)
    student_unit = student / jnp.maximum(jnp.linalg.norm(student, axis=-1, keepdims=True), 1e-6)
    teacher_unit = teacher / jnp.maximum(jnp.linalg.norm(teacher, axis=-1, keepdims=True), 1e-6)
    result["memory_cosine_distance_sum"] = jnp.sum(
        jnp.mean(1.0 - jnp.sum(student_unit * teacher_unit, axis=-1), axis=-1)
    )
    result["memory_mse_sum"] = jnp.sum(jnp.mean(jnp.square(student - teacher), axis=(1, 2)))
    return result


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")

    predicted_np, donor_np, labels, predictions, predicted_metadata = _load_predicted(
        args.predicted_memory.expanduser().resolve()
    )
    teacher_templates_np, teacher_indices_np = _load_teacher(
        args.teacher_memory.expanduser().resolve()
    )

    memory_args = argparse.Namespace(
        exp_name=memory_cache.DEFAULT_EXP_NAME,
        batches=args.eval_batches,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
    )
    memory_config = memory_cache.make_memory_config(memory_args)
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    _train_loader, val_loader = memory_trainer.create_train_val_data_loaders(
        memory_config, data_sharding
    )
    val_iter = iter(val_loader)

    action_config = action_recipe.make_train_config(
        config_module=training_config,
        exp_name="direct_visual_memory_action_compatibility",
        memory_path=args.teacher_memory,
        init_checkpoint=args.action_checkpoint,
        steps=1,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=0,
        overwrite=False,
    )
    action_config = dataclasses.replace(
        action_config,
        weight_loader=weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(
            str(args.action_checkpoint.expanduser().resolve())
        ),
    )
    rng, init_rng = jax.random.split(jax.random.key(action_config.seed))
    state, state_sharding = action_trainer.init_train_state(
        action_config, init_rng, mesh, resume=False
    )
    jax.block_until_ready(state)

    predicted = jax.device_put(jnp.asarray(predicted_np), replicated)
    donor = jax.device_put(jnp.asarray(donor_np), replicated)
    teacher_templates = jax.device_put(jnp.asarray(teacher_templates_np), replicated)
    teacher_indices = jax.device_put(jnp.asarray(teacher_indices_np), replicated)
    peval = jax.jit(
        functools.partial(eval_step, predicted, donor, teacher_templates, teacher_indices),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=replicated,
    )
    infos = []
    with sharding.set_mesh(mesh):
        for batch_index in range(args.eval_batches):
            infos.append(peval(jax.random.fold_in(rng, batch_index), state, next(val_iter)))
            print(f"evaluated {batch_index + 1}/{args.eval_batches}", flush=True)
    stacked = jax.device_get(common_utils.stack_forest(infos))
    evaluated_episodes = np.asarray(stacked["episode_index"]).reshape(-1)
    if np.any(~np.isfinite(predicted_np[evaluated_episodes])):
        raise ValueError("Evaluation requested an episode absent from the predicted-memory cache")

    summary = {
        "action_checkpoint": str(args.action_checkpoint.expanduser().resolve()),
        "predicted_memory": str(args.predicted_memory.expanduser().resolve()),
        "teacher_memory": str(args.teacher_memory.expanduser().resolve()),
        "same_action_samples_and_diffusion_noise": True,
        "action_parameters_updated": False,
        "eval_batches": args.eval_batches,
        "samples": int(len(evaluated_episodes)),
        "direct_visual_final_slot_accuracy": float(
            np.mean(predictions[evaluated_episodes] == labels[evaluated_episodes])
        ),
        "memory_cosine_distance_to_teacher": float(
            np.sum(stacked["memory_cosine_distance_sum"]) / len(evaluated_episodes)
        ),
        "memory_mse_to_teacher": float(
            np.sum(stacked["memory_mse_sum"]) / len(evaluated_episodes)
        ),
        "conditions": {},
        "predicted_memory_metadata": predicted_metadata,
    }
    for condition in CONDITIONS:
        loss_sum = float(np.sum(stacked[condition]["loss_sum"]))
        count = int(np.sum(stacked[condition]["count"]))
        summary["conditions"][condition] = {
            "action_loss": loss_sum / max(count, 1),
            "count": count,
        }
    teacher_loss = summary["conditions"]["teacher"]["action_loss"]
    for condition in CONDITIONS[1:]:
        summary["conditions"][condition]["relative_to_teacher"] = (
            summary["conditions"][condition]["action_loss"] / max(teacher_loss, 1e-12)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
