#!/usr/bin/env python3
"""Fixed-noise action-loss ablation for ShellGame external semantic memory."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
from pathlib import Path

from flax.training import common_utils
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train_pi0_mem_compress as trainer
from openpi.training import config as training_config
from openpi.training import sharding
from openpi.training import weight_loaders
from openpi.training.mem.recipes import shellgame_qwen_event_memory_action as recipe


DEFAULT_CHECKPOINT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_event_memory_action_eef7_260825/"
    "qwen_event_mem_v10init_action250_6gpu_260825/249/params"
)
DEFAULT_OUTPUT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame/"
    "qwen_event_mem_v10init_action250_fixed_noise_ablation_260825.json"
)
CONDITIONS = ("correct", "wrong_episode", "zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--memory-path", type=Path, default=recipe.DEFAULT_MEMORY_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _final_slot(event_sequence: np.ndarray) -> int:
    """Apply the three canonical swaps to the initial cup slot."""
    slot = int(event_sequence[0])
    swap_pairs = ((0, 1), (0, 2), (1, 2))
    for relation_id in event_sequence[1:]:
        left, right = swap_pairs[int(relation_id)]
        if slot == left:
            slot = right
        elif slot == right:
            slot = left
    return slot


def _memory_tables(path: Path):
    with np.load(path, allow_pickle=False) as source:
        templates = np.asarray(source["memory_templates"], dtype=np.float32)
        event_sequences = np.asarray(source["event_sequences"], dtype=np.int32)
        episode_to_template = np.asarray(source["episode_template_index"], dtype=np.int32)
    final_slots = np.asarray([_final_slot(sequence) for sequence in event_sequences])
    # For each template choose a deterministic donor whose final target cup is
    # different.  This makes the counterfactual stronger than merely selecting
    # another episode, which can still end at the same cup one third of the time.
    donor_for_template = np.empty(len(event_sequences), dtype=np.int32)
    for template_index, final_slot in enumerate(final_slots):
        candidates = np.flatnonzero(final_slots != final_slot)
        donor_for_template[template_index] = candidates[template_index % len(candidates)]
    donor_episode_to_template = donor_for_template[episode_to_template]
    if np.any(final_slots[donor_episode_to_template] == final_slots[episode_to_template]):
        raise AssertionError("Wrong-memory donor unexpectedly has the same final slot")
    return jnp.asarray(templates), jnp.asarray(donor_episode_to_template)


def _masked_stats(loss_per_sample, frame_index, low, high):
    mask = (frame_index >= low) & (frame_index <= high)
    return jnp.sum(loss_per_sample * mask), jnp.sum(mask)


def ablation_step(config, templates, donor_table, rng, state, batch):
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, actions = batch
    episode = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    frame = jnp.asarray(observation.frame_index, dtype=jnp.int32)
    memories = {
        "correct": observation.semantic_memory,
        "wrong_episode": templates[donor_table[episode]],
        "zero": jnp.zeros_like(observation.semantic_memory),
    }
    result = {}
    for condition in CONDITIONS:
        variant = observation.replace(semantic_memory=memories[condition])
        chunk_loss, _aux = model.compute_loss_with_memory_aux(rng, variant, actions, train=False)
        per_sample = jnp.mean(chunk_loss, axis=-1)
        total_sum, total_count = jnp.sum(per_sample), per_sample.size
        approach_sum, approach_count = _masked_stats(per_sample, frame, 59, 90)
        grasp_sum, grasp_count = _masked_stats(per_sample, frame, 91, 153)
        result[condition] = {
            "total_sum": total_sum,
            "total_count": jnp.asarray(total_count, dtype=jnp.int32),
            "approach_sum": approach_sum,
            "approach_count": approach_count,
            "grasp_sum": grasp_sum,
            "grasp_count": grasp_count,
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    config = recipe.make_train_config(
        config_module=training_config,
        exp_name="qwen_event_action_memory_ablation",
        memory_path=args.memory_path,
        init_checkpoint=args.checkpoint,
        steps=1,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=0,
        overwrite=False,
    )
    config = dataclasses.replace(
        config,
        weight_loader=weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(str(args.checkpoint.resolve())),
        eval_batches=args.eval_batches,
    )
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng, init_rng = jax.random.split(jax.random.key(config.seed))
    state, state_sharding = trainer.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    _train_loader, val_loader = trainer.create_train_val_data_loaders(config, data_sharding)
    val_iter = iter(val_loader)
    templates, donor_table = _memory_tables(args.memory_path)
    templates = jax.device_put(templates, replicated)
    donor_table = jax.device_put(donor_table, replicated)
    peval = jax.jit(
        functools.partial(ablation_step, config, templates, donor_table),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=replicated,
    )
    infos = []
    with sharding.set_mesh(mesh):
        for batch_index in range(args.eval_batches):
            infos.append(peval(jax.random.fold_in(rng, batch_index), state, next(val_iter)))
            print(f"evaluated {batch_index + 1}/{args.eval_batches}", flush=True)
    stacked = jax.device_get(common_utils.stack_forest(infos))
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "memory_path": str(args.memory_path.resolve()),
        "same_batches_and_diffusion_noise": True,
        "wrong_memory_final_target_always_differs": True,
        "eval_batches": args.eval_batches,
        "samples": args.eval_batches * args.batch_size,
        "conditions": {},
    }
    for condition in CONDITIONS:
        values = stacked[condition]
        condition_summary = {}
        for phase in ("total", "approach", "grasp"):
            loss_sum = float(jnp.sum(values[f"{phase}_sum"]))
            count = int(jnp.sum(values[f"{phase}_count"]))
            condition_summary[f"{phase}_loss"] = loss_sum / max(count, 1)
            condition_summary[f"{phase}_count"] = count
        summary["conditions"][condition] = condition_summary
    correct = summary["conditions"]["correct"]
    for condition in ("wrong_episode", "zero"):
        for phase in ("total", "approach", "grasp"):
            baseline = correct[f"{phase}_loss"]
            value = summary["conditions"][condition][f"{phase}_loss"]
            summary["conditions"][condition][f"{phase}_relative_to_correct"] = value / max(baseline, 1e-12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
