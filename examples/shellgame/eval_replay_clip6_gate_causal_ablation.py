"""Causal gate ablation for the clean 60-frame replay tracker.

The evaluator loads one trained replay tracker, encodes the same held-out
episodes under all six clip-grid offsets, and compares four inference-only
gate interventions:

* normal: use the learned gate unchanged;
* freeze_tail: allow the third transition, then freeze all later updates;
* oracle_change_mask: retain learned gate values only on clips overlapping a
  ground-truth swap interval;
* oracle_open: force a fixed gate on swap-overlap clips and zero elsewhere.

Ground-truth timing is used only for diagnosis and never enters training.
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from examples.shellgame import train_replay_unrolled_clip6_memory_probe as _probe
from openpi.models import model as _model
from openpi.training import sharding
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as _recipe
from scripts.mem import train_semantic_memory as _trainer

DEFAULT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "shellgame_replay_unrolled_clip6_memory_probe/"
    "replay_clip6_gated_clean60_warm_bptt_1k_260825/1499/params"
)
DEFAULT_OUTPUT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame/"
    "replay_clip6_gate_causal_ablation_step1499_260825.json"
)
CONDITIONS = ("normal", "freeze_tail", "oracle_change_mask", "oracle_open")
SWAP_START_FRAMES = (20, 30, 40)


def _config_args(args: argparse.Namespace) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        exp_name="replay_clip6_gate_causal_ablation_eval",
        init_mode="warm",
        warm_checkpoint=args.checkpoint,
        steps=1,
        warmup_steps=0,
        peak_lr=1e-5,
        decay_lr=1e-5,
        final_slot_weight=1.0,
        intermediate_slot_weight=0.0,
        transition_slot_weight=1.0,
        hold_slot_weight=1.0,
        carry_gate=True,
        carry_gate_bias=-2.0,
        detach_between_clips=False,
        clip_order="normal",
        batch_size=args.batch_size,
        num_workers=0,
        fsdp_devices=args.fsdp_devices,
        eval_interval=1,
        eval_batches=args.eval_batches,
        save_interval=1,
        keep_period=1,
        overwrite=False,
        resume=False,
    )


def _clip_geometry(offsets):
    steps = jnp.arange(_probe.NUM_CLIPS, dtype=jnp.int32)[None, :]
    starts = offsets[:, None] + steps * _probe.CLIP_SIZE
    ends = jnp.minimum(starts + _probe.CLIP_SIZE - 1, _probe.REPLAY_FRAMES - 1)
    swap_starts = jnp.asarray(SWAP_START_FRAMES, dtype=jnp.int32)
    swap_ends = jnp.asarray(_probe.SWAP_END_FRAMES, dtype=jnp.int32)
    overlaps = jnp.any(
        (starts[:, :, None] <= swap_ends[None, None, :]) & (ends[:, :, None] >= swap_starts[None, None, :]),
        axis=-1,
    )
    completed = jnp.sum(ends[:, :, None] >= swap_ends[None, None, :], axis=-1)
    previous_completed = jnp.concatenate(
        [jnp.zeros((offsets.shape[0], 1), dtype=completed.dtype), completed[:, :-1]],
        axis=1,
    )
    transition = completed > previous_completed
    partial = jnp.any(
        (ends[:, :, None] >= swap_starts[None, None, :]) & (ends[:, :, None] < swap_ends[None, None, :]),
        axis=-1,
    )
    return starts, ends, overlaps, completed, previous_completed, transition, partial


def _sum_masked(values, mask):
    return jnp.sum(values.astype(jnp.float32) * mask.astype(jnp.float32))


def _condition_counts(outputs, clip_labels, completed, transition, partial, offsets):
    predictions = jnp.argmax(outputs["clip_logits"], axis=-1)
    correct = predictions == clip_labels
    final_prediction = predictions[:, -1]
    final_label = clip_labels[:, -1]
    final_correct = final_prediction == final_label

    stage_correct = []
    stage_count = []
    for stage in range(1, _recipe.NUM_STAGES + 1):
        mask = transition & (completed == stage)
        stage_correct.append(_sum_masked(correct, mask))
        stage_count.append(jnp.sum(mask))
    stage_correct = jnp.stack(stage_correct)
    stage_count = jnp.stack(stage_count)

    third_mask = transition & (completed == _recipe.NUM_STAGES)
    third_correct = jnp.sum(correct & third_mask, axis=1).astype(jnp.bool_)
    third_count = jnp.sum(third_correct)
    third_wrong = ~third_correct

    confusion = jnp.einsum(
        "bi,bj->ij",
        jax.nn.one_hot(final_label, 3, dtype=jnp.float32),
        jax.nn.one_hot(final_prediction, 3, dtype=jnp.float32),
    )
    offset_correct = jnp.stack(
        [jnp.sum(final_correct & (offsets == offset)) for offset in range(_probe.MAX_OFFSET + 1)]
    )
    offset_count = jnp.stack([jnp.sum(offsets == offset) for offset in range(_probe.MAX_OFFSET + 1)])
    hold = ~transition
    gates = outputs["gates"].astype(jnp.float32)
    return {
        "sequence_count": jnp.asarray(final_correct.size, dtype=jnp.float32),
        "final_correct": jnp.sum(final_correct),
        "clip_correct": jnp.sum(correct),
        "clip_count": jnp.asarray(correct.size, dtype=jnp.float32),
        "stage_correct": stage_correct,
        "stage_count": stage_count,
        "partial_correct": _sum_masked(correct, partial),
        "partial_count": jnp.sum(partial),
        "third_correct": third_count,
        "third_correct_final_correct": jnp.sum(third_correct & final_correct),
        "third_correct_final_wrong": jnp.sum(third_correct & ~final_correct),
        "third_wrong_final_correct": jnp.sum(third_wrong & final_correct),
        "third_wrong_count": jnp.sum(third_wrong),
        "transition_gate_sum": _sum_masked(gates, transition),
        "transition_gate_count": jnp.sum(transition),
        "hold_gate_sum": _sum_masked(gates, hold),
        "hold_gate_count": jnp.sum(hold),
        "final_gate_sum": jnp.sum(gates[:, -1]),
        "confusion": confusion,
        "offset_correct": offset_correct,
        "offset_count": offset_count,
    }


def causal_eval_step(config, label_table, oracle_open_gate, rng, state, batch):
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, _actions = batch
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode_index]
    initial = labels[:, 0]
    stage_slots = labels[:, 1 + _recipe.NUM_STAGES :]

    processed = _model.preprocess_observation(rng, observation, train=False)
    history = processed.images["base_rgb"][:, : _probe.REPLAY_FRAMES]
    _, encoder_out = model.PaliGemma.img(history, train=False)
    patches = encoder_out["with_posemb"][:, : _probe.REPLAY_FRAMES]

    batch_size = episode_index.shape[0]
    offsets = jnp.tile(jnp.arange(_probe.MAX_OFFSET + 1, dtype=jnp.int32), batch_size)
    patches = jnp.repeat(patches, _probe.MAX_OFFSET + 1, axis=0)
    initial = jnp.repeat(initial, _probe.MAX_OFFSET + 1, axis=0)
    stage_slots = jnp.repeat(stage_slots, _probe.MAX_OFFSET + 1, axis=0)

    _starts, _ends, overlap, completed, previous_completed, transition, partial = _clip_geometry(offsets)
    state_slots = jnp.concatenate([initial[:, None], stage_slots], axis=1)
    clip_labels = jnp.take_along_axis(state_slots, completed, axis=1)
    freeze_tail = (previous_completed < _recipe.NUM_STAGES).astype(jnp.float32)
    oracle_change = overlap.astype(jnp.float32)
    oracle_override = oracle_change * jnp.asarray(oracle_open_gate, dtype=jnp.float32)

    tracker = model.HistoryReplayUnrolledVisualMemoryTracker
    outputs = {
        "normal": tracker(patches, initial, offsets, train=False),
        "freeze_tail": tracker(
            patches,
            initial,
            offsets,
            train=False,
            gate_multiplier=freeze_tail,
        ),
        "oracle_change_mask": tracker(
            patches,
            initial,
            offsets,
            train=False,
            gate_multiplier=oracle_change,
        ),
        "oracle_open": tracker(
            patches,
            initial,
            offsets,
            train=False,
            gate_override=oracle_override,
        ),
    }
    return {
        condition: _condition_counts(
            condition_outputs,
            clip_labels,
            completed,
            transition,
            partial,
            offsets,
        )
        for condition, condition_outputs in outputs.items()
    }


def _add_trees(total, update):
    if total is None:
        return update
    return jax.tree.map(lambda left, right: left + right, total, update)


def _ratio(numerator, denominator):
    return float(numerator / max(float(denominator), 1.0))


def _finalize(raw):
    result = {}
    for condition in CONDITIONS:
        item = raw[condition]
        stage_accuracy = item["stage_correct"] / np.maximum(item["stage_count"], 1.0)
        result[condition] = {
            "sequence_count": int(item["sequence_count"]),
            "final_accuracy": _ratio(item["final_correct"], item["sequence_count"]),
            "clip_accuracy": _ratio(item["clip_correct"], item["clip_count"]),
            "stage_endpoint_accuracy": [float(value) for value in stage_accuracy],
            "partial_swap_hold_accuracy": _ratio(item["partial_correct"], item["partial_count"]),
            "final_given_third_correct": _ratio(item["third_correct_final_correct"], item["third_correct"]),
            "tail_flip_rate_given_third_correct": _ratio(item["third_correct_final_wrong"], item["third_correct"]),
            "final_recovery_given_third_wrong": _ratio(item["third_wrong_final_correct"], item["third_wrong_count"]),
            "third_correct_count": int(item["third_correct"]),
            "transition_gate_mean": _ratio(item["transition_gate_sum"], item["transition_gate_count"]),
            "hold_gate_mean": _ratio(item["hold_gate_sum"], item["hold_gate_count"]),
            "final_gate_mean": _ratio(item["final_gate_sum"], item["sequence_count"]),
            "offset_final_accuracy": [
                _ratio(correct, count)
                for correct, count in zip(item["offset_correct"], item["offset_count"], strict=True)
            ],
            "final_confusion_gt_rows_pred_cols": np.asarray(item["confusion"], dtype=np.int64).tolist(),
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=125)
    parser.add_argument("--oracle-open-gate", type=float, default=1.0)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size % args.fsdp_devices != 0:
        raise ValueError("batch-size must be divisible by fsdp-devices")
    eval_batches = args.smoke_batches or args.eval_batches
    config = _probe.build_config(_config_args(args))
    label_table = _recipe.load_episode_label_table(config)
    rng = jax.random.key(config.seed)
    eval_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    _train_loader, val_loader = _trainer.create_train_val_data_loaders(config, data_sharding)
    val_iter = iter(val_loader)
    state, state_sharding = _trainer.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)

    peval = jax.jit(
        functools.partial(
            causal_eval_step,
            config,
            label_table,
            args.oracle_open_gate,
        ),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=replicated,
    )
    totals = None
    for index in range(eval_batches):
        with sharding.set_mesh(mesh):
            counts = peval(jax.random.fold_in(eval_rng, index), state, next(val_iter))
        counts = jax.device_get(counts)
        totals = _add_trees(totals, counts)
        if (index + 1) % 25 == 0 or index + 1 == eval_batches:
            print(f"evaluated {index + 1}/{eval_batches} batches")

    results = _finalize(totals)
    payload = {
        "checkpoint": args.checkpoint,
        "held_out_episodes": 500,
        "evaluated_batches": eval_batches,
        "batch_size": args.batch_size,
        "offsets_per_episode": _probe.MAX_OFFSET + 1,
        "oracle_open_gate": args.oracle_open_gate,
        "conditions": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not args.smoke_batches:
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
