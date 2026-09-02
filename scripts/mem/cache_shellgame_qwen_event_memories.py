#!/usr/bin/env python3
"""Cache the validated recurrent memory for all ShellGame event sequences.

Training uses simulator event labels as clean teacher forcing.  Deployment
uses the frozen Qwen event detector; its held-out equivalence is recorded in
the output metadata.  Only the 81 possible initial/swap sequences are run
through the neural updater, then episodes reference those compact templates.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.training import sharding
from openpi.training import weight_loaders
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as recipe
import train_semantic_memory as trainer


DEFAULT_CHECKPOINT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "shellgame_stage_slot_only_relation_recurrent_probe/"
    "stage_slot_only_random_relation_frozen_memory_1k_260821/500/params"
)
DEFAULT_OUTPUT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/artifacts/"
    "shellgame_qwen_event_final_memory_v1_260825.npz"
)
DEFAULT_QWEN_ADAPTER = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375"
)
DEFAULT_QWEN_VALIDATION = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame/"
    "qwen3vl_gt_event_lora_v1_step375_sliding_trigger_val20.summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-adapter", type=Path, default=DEFAULT_QWEN_ADAPTER)
    parser.add_argument("--qwen-validation", type=Path, default=DEFAULT_QWEN_VALIDATION)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    if args.batch_size % args.fsdp_devices != 0:
        raise ValueError("batch-size must be divisible by fsdp-devices")
    base = recipe.make_train_config()
    labels = np.asarray(recipe.load_episode_label_table(base), dtype=np.int32)
    event_sequences = labels[:, : 1 + recipe.NUM_STAGES]
    templates, episode_template_index = np.unique(event_sequences, axis=0, return_inverse=True)

    config = dataclasses.replace(
        base,
        name="shellgame_qwen_event_memory_cache",
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.checkpoint)),
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=0,
        num_train_steps=0,
        wandb_enabled=False,
    )
    rng = jax.random.key(config.seed)
    mesh = sharding.make_mesh(config.fsdp_devices)
    state, state_sharding = trainer.init_train_state(config, rng, mesh, resume=False)
    jax.block_until_ready(state)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    def apply_sequences(state, sequence_batch):
        model = nnx.merge(state.model_def, state.params)
        model.eval()
        batch = sequence_batch.shape[0]
        dummy_patches = jnp.zeros(
            (batch, model.history_frames, 256, 1152), dtype=jnp.bfloat16
        )
        outputs = model.HistoryThreeSwapVisualRelationMemoryTracker(
            dummy_patches,
            sequence_batch[:, 0].astype(jnp.int32),
            sequence_batch[:, 1:].astype(jnp.int32),
        )
        return outputs[2][:, -1]

    papply = jax.jit(
        apply_sequences,
        in_shardings=(state_sharding, data_sharding),
        out_shardings=data_sharding,
    )
    pad = (-len(templates)) % args.batch_size
    padded = np.concatenate((templates, np.repeat(templates[-1:], pad, axis=0)), axis=0) if pad else templates
    memories = []
    with sharding.set_mesh(mesh):
        for start in range(0, len(padded), args.batch_size):
            batch = jax.device_put(jnp.asarray(padded[start : start + args.batch_size]), data_sharding)
            memories.append(np.asarray(jax.device_get(papply(state, batch)), dtype=np.float16))
            print(f"cached templates {min(start + args.batch_size, len(templates))}/{len(templates)}", flush=True)
    memories = np.concatenate(memories, axis=0)[: len(templates)]
    if memories.shape != (len(templates), 128, 64) or not np.all(np.isfinite(memories)):
        raise ValueError(f"Invalid cached memories: {memories.shape}")

    validation = json.loads(args.qwen_validation.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 1,
        "event_source_train": "simulator_gt_teacher_forcing",
        "event_source_deploy": "qwen3vl_step375_sliding_trigger",
        "qwen_adapter": str(args.qwen_adapter.resolve()),
        "qwen_validation": str(args.qwen_validation.resolve()),
        "qwen_event_precision": validation["event_trigger"]["event_precision"],
        "qwen_event_recall": validation["event_trigger"]["event_recall"],
        "qwen_exact_sequence_accuracy": validation["event_trigger"]["exact_three_relation_sequence_accuracy"],
        "recurrent_checkpoint": str(args.checkpoint.resolve()),
        "episodes": int(len(labels)),
        "unique_event_sequences": int(len(templates)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        memory_templates=memories,
        event_sequences=templates.astype(np.int8),
        episode_template_index=episode_template_index.astype(np.int16),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {args.output} ({args.output.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
