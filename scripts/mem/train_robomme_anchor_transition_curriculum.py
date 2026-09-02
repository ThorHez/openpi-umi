#!/usr/bin/env python3
"""Distill ceiling transitions with anchor evidence and semantic feedback curriculum."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import flax
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.tasks.robomme.anchor_conditioned_transition_memory import (  # noqa: E402
    AnchorConditionedTransitionMemory,
)
from scripts.mem import train_robomme_anchor_conditioned_decomposition as anchor_base  # noqa: E402
from scripts.mem import train_robomme_decomposed_region_distillation as base  # noqa: E402


DEFAULT_INIT = ROOT / "checkpoints/robomme_anchor_conditioned_decomposition_seed260831_260828/params.msgpack"
DEFAULT_OUTPUT = ROOT / "checkpoints/robomme_anchor_transition_curriculum_seed260901_260829"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, default=base.DEFAULT_FIXED)
    parser.add_argument("--teacher-dir", type=Path, default=base.DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=base.DEFAULT_FEATURES)
    parser.add_argument("--anchor-dir", type=Path, default=anchor_base.DEFAULT_ANCHORS)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--end-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--event-type-weight", type=float, default=1.0)
    parser.add_argument("--write-entity-weight", type=float, default=0.5)
    parser.add_argument("--write-region-weight", type=float, default=2.0)
    parser.add_argument("--swap-pair-weight", type=float, default=2.0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--no-change-weight", type=float, default=1.0)
    parser.add_argument("--delta-weight", type=float, default=2.0)
    parser.add_argument("--final-weight", type=float, default=2.0)
    parser.add_argument("--trajectory-weight", type=float, default=0.1)
    parser.add_argument("--gate-temperature", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260901)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _restore_matching(params, path: Path):
    restored = flax.serialization.msgpack_restore(path.read_bytes())
    flat = traverse_util.flatten_dict(params)
    source = traverse_util.flatten_dict(restored)
    loaded = []
    # Encoder names are unchanged; operation heads moved under a scanned cell
    # and intentionally start fresh because they now consume semantic feedback.
    for key, value in source.items():
        if key in flat and np.shape(value) == np.shape(flat[key]):
            flat[key] = jnp.asarray(value, dtype=flat[key].dtype)
            loaded.append("/".join(key))
    return traverse_util.unflatten_dict(flat), loaded


def _inputs(batch: dict[str, Any], teacher_force_mask: np.ndarray | jax.Array):
    return {
        **anchor_base._model_inputs(batch),  # noqa: SLF001
        "teacher_previous_tables": jnp.asarray(batch["table_targets"][:, :-1]),
        "teacher_force_mask": jnp.asarray(teacher_force_mask),
    }


def _curriculum_ratio(step: int, steps: int) -> float:
    """20% full teacher forcing, 50% linear to 0.3, 30% mostly free rollout."""
    first = max(1, int(round(0.20 * steps)))
    second = max(first + 1, int(round(0.70 * steps)))
    if step <= first:
        return 1.0
    if step <= second:
        progress = (step - first) / max(second - first, 1)
        return 1.0 - 0.7 * progress
    progress = (step - second) / max(steps - second, 1)
    return 0.3 * (1.0 - progress)


def _table_losses(output, batch):
    probabilities = jnp.clip(output["all_tables"][:, 1:], 1e-6, 1.0)
    targets = jnp.asarray(batch["table_targets"][:, 1:], dtype=jnp.int32)
    previous = jnp.asarray(batch["table_targets"][:, :-1], dtype=jnp.int32)
    field_mask = jnp.asarray(batch["table_mask"][:, 1:], dtype=jnp.float32)
    valid = jnp.asarray(batch["sequence_mask"], dtype=jnp.float32)
    transition = jnp.asarray(batch["state_change_mask"], dtype=jnp.float32) * valid
    hold = (1.0 - jnp.asarray(batch["state_change_mask"], dtype=jnp.float32)) * valid
    ce = -jnp.log(
        jnp.take_along_axis(probabilities, targets[..., None], axis=-1)[..., 0]
    )
    state_ce = jnp.sum(ce * field_mask, axis=-1) / jnp.maximum(
        jnp.sum(field_mask, axis=-1), 1.0
    )

    def mean_on(values, mask):
        return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    transition_loss = mean_on(state_ce, transition)
    no_change_loss = mean_on(state_ce, hold)
    trajectory_loss = mean_on(state_ce, valid)
    changed_fields = (targets != previous).astype(jnp.float32) * field_mask
    delta_loss = jnp.sum(ce * changed_fields) / jnp.maximum(
        jnp.sum(changed_fields), 1.0
    )
    lengths = jnp.sum(valid, axis=1).astype(jnp.int32) - 1
    final_ce = state_ce[jnp.arange(state_ce.shape[0]), lengths]
    final_loss = jnp.mean(final_ce)
    return {
        "transition_loss": transition_loss,
        "no_change_loss": no_change_loss,
        "delta_loss": delta_loss,
        "final_loss": final_loss,
        "trajectory_loss": trajectory_loss,
    }


def _summary(output: dict[str, np.ndarray], batch: dict[str, np.ndarray]):
    result = base._summary(output, batch)  # noqa: SLF001
    prediction = np.argmax(output["all_tables"][:, 1:], axis=-1)
    targets = batch["table_targets"][:, 1:]
    field_mask = batch["table_mask"][:, 1:]
    exact = np.all((prediction == targets) | ~field_mask, axis=-1)
    valid = batch["sequence_mask"]
    transition = batch["state_change_mask"] & valid
    hold = ~batch["state_change_mask"] & valid
    result["overall"].update(
        transition_state_exact_accuracy=float(exact[transition].mean()),
        no_change_state_exact_accuracy=float(exact[hold].mean()),
        state_exact_accuracy=float(exact[valid].mean()),
    )
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size % 3 or args.eval_batch_size % 3:
        raise ValueError("train and eval batch sizes must be divisible by three")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: anchor_base.AnchorRegionDataset(split, args)
        for split in ("train", "dev", "test")
    }
    try:
        model = AnchorConditionedTransitionMemory(
            max_steps=datasets["train"].max_steps,
            gate_temperature=args.gate_temperature,
            straight_through_hard_feedback=True,
        )
        rng = np.random.default_rng(args.seed)
        initial = datasets["train"].batch(
            datasets["train"].sample(rng, args.batch_size)
        )
        force_all = initial["sequence_mask"].copy()
        params = model.init(
            jax.random.key(args.seed), **_inputs(initial, force_all), train=False
        )["params"]
        loaded = []
        if args.init_checkpoint is not None:
            params, loaded = _restore_matching(params, args.init_checkpoint)
        schedule = optax.warmup_cosine_decay_schedule(
            0.0,
            args.learning_rate,
            min(args.warmup_steps, max(args.steps - 1, 1)),
            args.steps,
            end_value=args.end_learning_rate,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=args.weight_decay),
        )
        opt_state = optimizer.init(params)
        type_weights = jnp.asarray(datasets["train"].event_type_weights)

        def objective(current_params, batch, teacher_force_mask, *, train):
            output = model.apply(
                {"params": current_params},
                **_inputs(batch, teacher_force_mask),
                train=train,
            )
            event = base._masked_ce(
                output["event_type_logits"],
                jnp.asarray(batch["event_type"]),
                jnp.asarray(batch["micro_mask"]),
                type_weights,
            )
            entity = base._masked_ce(
                output["write_entity_logits"],
                jnp.asarray(batch["write_entity"]),
                jnp.asarray(batch["write_mask"]),
            )
            region = base._masked_ce(
                output["write_region_logits"],
                jnp.asarray(batch["write_region"]),
                jnp.asarray(batch["write_mask"]),
            )
            pair = base._masked_ce(
                output["swap_pair_logits"],
                jnp.asarray(batch["swap_pair"]),
                jnp.asarray(batch["swap_mask"]),
            )
            table = _table_losses(output, batch)
            operation_loss = (
                args.event_type_weight * event
                + args.write_entity_weight * entity
                + args.write_region_weight * region
                + args.swap_pair_weight * pair
            )
            loss = (
                operation_loss
                + args.transition_weight * table["transition_loss"]
                + args.no_change_weight * table["no_change_loss"]
                + args.delta_weight * table["delta_loss"]
                + args.final_weight * table["final_loss"]
                + args.trajectory_weight * table["trajectory_loss"]
            )
            return loss, {
                "loss": loss,
                "operation_loss": operation_loss,
                "event_type_loss": event,
                "write_entity_loss": entity,
                "write_region_loss": region,
                "swap_pair_loss": pair,
                **table,
            }

        @jax.jit
        def train_step(current_params, current_opt, batch, teacher_force_mask):
            (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(
                current_params, batch, teacher_force_mask, train=True
            )
            updates, next_opt = optimizer.update(grads, current_opt, current_params)
            return optax.apply_updates(current_params, updates), next_opt, metrics

        @jax.jit
        def infer(current_params, batch, teacher_force_mask):
            return model.apply(
                {"params": current_params},
                **_inputs(batch, teacher_force_mask),
                train=False,
            )

        def evaluate(current_params, split: str, *, teacher_forcing: bool):
            outputs: dict[str, list[np.ndarray]] = defaultdict(list)
            batches = []
            rows = datasets[split].rows
            for start in range(0, len(rows), args.eval_batch_size):
                indices = rows[start : start + args.eval_batch_size]
                valid_count = len(indices)
                if valid_count < args.eval_batch_size:
                    indices = np.pad(
                        indices, (0, args.eval_batch_size - valid_count), mode="edge"
                    )
                batch = datasets[split].batch(indices)
                force = batch["sequence_mask"] & bool(teacher_forcing)
                output = jax.device_get(infer(current_params, batch, force))
                for key, value in output.items():
                    if key != "all_memories":
                        outputs[key].append(np.asarray(value)[:valid_count])
                batches.append({key: value[:valid_count] for key, value in batch.items()})
            merged_output = {key: np.concatenate(value) for key, value in outputs.items()}
            merged_batch = {
                key: np.concatenate([batch[key] for batch in batches])
                for key in batches[0]
            }
            return _summary(merged_output, merged_batch)

        best_params = params
        best_step = 0
        best_score = (-1.0,) * 5
        history = []
        started = time.monotonic()
        for step in range(1, args.steps + 1):
            batch = datasets["train"].batch(
                datasets["train"].sample(rng, args.batch_size)
            )
            ratio = _curriculum_ratio(step, args.steps)
            teacher_force_mask = batch["sequence_mask"] & (
                rng.random(batch["sequence_mask"].shape) < ratio
            )
            params, opt_state, metrics = train_step(
                params, opt_state, batch, teacher_force_mask
            )
            if step % args.eval_every == 0 or step == args.steps:
                dev = evaluate(params, "dev", teacher_forcing=False)
                overall = dev["overall"]
                balance = min(
                    overall["transition_state_exact_accuracy"],
                    overall["no_change_state_exact_accuracy"],
                    overall["mean_task_final_query_accuracy"],
                )
                score = (
                    balance,
                    overall["mean_task_final_query_accuracy"],
                    overall["transition_state_exact_accuracy"],
                    overall["no_change_state_exact_accuracy"],
                    overall["mean_task_trajectory_accuracy"],
                )
                if score > best_score:
                    best_score = score
                    best_step = step
                    best_params = jax.device_get(params)
                row = {
                    "step": step,
                    "teacher_force_ratio": ratio,
                    **{key: float(value) for key, value in metrics.items()},
                    "selection_score": score,
                    "dev_free_rollout": dev,
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        result = {
            "schema_version": 1,
            "experiment": "anchor_conditioned_transition_delta_st_curriculum",
            "best_step": best_step,
            "best_score": best_score,
            "elapsed_seconds": time.monotonic() - started,
            "metrics": {
                split: {
                    "teacher_forced": evaluate(best_params, split, teacher_forcing=True),
                    "free_rollout": evaluate(best_params, split, teacher_forcing=False),
                }
                for split in ("train", "dev", "test")
            },
            "history": history,
        }
        (args.output_dir / "params.msgpack").write_bytes(
            flax.serialization.to_bytes(best_params)
        )
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        (args.output_dir / "training_config.json").write_text(
            json.dumps(
                {
                    **{
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "loaded_leaves": loaded,
                    "semantic_feedback": "straight_through_hard",
                    "checkpoint_selection": "min(transition,no-change,mean-final)",
                    "jax_devices": [str(device) for device in jax.devices()],
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(result["metrics"], indent=2), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
