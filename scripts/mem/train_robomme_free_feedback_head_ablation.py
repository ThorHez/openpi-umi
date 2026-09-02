#!/usr/bin/env python3
"""Ablate joint versus completion/phase heads in free-feedback RoboMME MEM."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.tasks.robomme.anchor_conditioned_transition_memory import AnchorConditionedTransitionMemory  # noqa: E402
from scripts.mem import train_robomme_anchor_conditioned_decomposition as anchor_base  # noqa: E402
from scripts.mem import train_robomme_anchor_transition_curriculum as transition  # noqa: E402
from scripts.mem import train_robomme_decomposed_region_distillation as base  # noqa: E402
from scripts.mem import train_robomme_phase_aware_visual_parser as phase_base  # noqa: E402
from scripts.mem import train_robomme_visual_operation_parser_ablation as parser_base  # noqa: E402

DEFAULT_FEATURES = ROOT / "artifacts/robomme_fixed_chunk_rgb_grid8_v1_260829"
DEFAULT_PHASES = ROOT / "artifacts/robomme_fixed_chunk_phase_labels_v1_260829"
DEFAULT_OUTPUT = ROOT / "checkpoints/robomme_free_feedback_head_ablation_260829"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("joint", "completion", "phase"), required=True)
    parser.add_argument("--fixed-dir", type=Path, default=base.DEFAULT_FIXED)
    parser.add_argument("--teacher-dir", type=Path, default=base.DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--phase-dir", type=Path, default=DEFAULT_PHASES)
    parser.add_argument("--anchor-dir", type=Path, default=anchor_base.DEFAULT_ANCHORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--operation-pretrain-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--end-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--completion-positive-weight", type=float, default=3.0)
    parser.add_argument("--phase-weight", type=float, default=0.5)
    parser.add_argument("--write-entity-weight", type=float, default=0.5)
    parser.add_argument("--write-region-weight", type=float, default=2.0)
    parser.add_argument("--swap-pair-weight", type=float, default=2.0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--no-change-weight", type=float, default=1.0)
    parser.add_argument("--delta-weight", type=float, default=2.0)
    parser.add_argument("--final-weight", type=float, default=2.0)
    parser.add_argument("--trajectory-weight", type=float, default=0.1)
    parser.add_argument("--gate-temperature", type=float, default=0.25)
    parser.add_argument("--train-commit-threshold", type=float, default=0.5)
    parser.add_argument("--target-hold-fpr", type=float, default=0.005)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260905)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _inputs(batch: dict[str, Any], teacher_force_mask: np.ndarray | jax.Array):
    return {
        **anchor_base._model_inputs(batch),  # noqa: SLF001
        "teacher_previous_tables": jnp.asarray(batch["table_targets"][:, :-1]),
        "teacher_force_mask": jnp.asarray(teacher_force_mask),
    }


def _model(args: argparse.Namespace, data: phase_base.PhaseDataset, threshold: float):
    return AnchorConditionedTransitionMemory(
        max_steps=data.max_steps,
        spatial_tokens=data.spatial_tokens,
        input_width=data.patch_width,
        gate_temperature=args.gate_temperature,
        straight_through_hard_feedback=True,
        hard_event_commit=True,
        event_head_mode="joint" if args.variant == "joint" else "completion",
        use_auxiliary_heads=True,
        commit_threshold=threshold,
    )


def _effective_output(output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = dict(output)
    result["event_type_logits"] = np.log(
        np.clip(np.asarray(output["committed_event_gates"]), 1e-8, 1.0)
    )
    return result


def _phase_metrics(output: dict[str, np.ndarray], batch: dict[str, np.ndarray]):
    prediction = np.argmax(output["phase_logits"], axis=-1)
    target = batch["phase"]
    mask = batch["sequence_mask"]
    return {"accuracy": float((prediction[mask] == target[mask]).mean())}


def _summary(output: dict[str, np.ndarray], batch: dict[str, np.ndarray]):
    effective = _effective_output(output)
    result = transition._summary(effective, batch)  # noqa: SLF001
    result["routing"] = parser_base._metrics(effective, batch)  # noqa: SLF001
    result["phase"] = _phase_metrics(output, batch)
    return result


def _score(summary: dict[str, Any]) -> tuple[float, ...]:
    overall = summary["overall"]
    routing = summary["routing"]
    balance = min(
        overall["transition_state_exact_accuracy"],
        overall["no_change_state_exact_accuracy"],
        overall["mean_task_final_query_accuracy"],
    )
    return (
        balance,
        overall["mean_task_final_query_accuracy"],
        overall["transition_state_exact_accuracy"],
        overall["no_change_state_exact_accuracy"],
        routing["full_update_recall"],
    )


def main() -> None:
    args = parse_args()
    if args.batch_size % 3 or args.eval_batch_size % 3:
        raise ValueError("train and eval batch sizes must be divisible by three")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: phase_base.PhaseDataset(split, args) for split in ("train", "dev", "test")
    }
    try:
        train_data = datasets["train"]
        training_model = _model(args, train_data, args.train_commit_threshold)
        rng = np.random.default_rng(args.seed)
        initial = train_data.parser_batch(train_data.sample(rng, args.batch_size))
        force_all = initial["sequence_mask"].copy()
        params = training_model.init(
            jax.random.key(args.seed), **_inputs(initial, force_all), train=False
        )["params"]
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
        joint_weights = np.sqrt(train_data.event_type_weights)
        joint_weights = jnp.asarray(joint_weights / joint_weights.mean())
        phase_values = train_data.phase[train_data.rows]
        phase_mask = train_data.fixed["step_mask"][train_data.rows]
        phase_counts = np.bincount(phase_values[phase_mask], minlength=4).astype(np.float32)
        phase_weights = np.sqrt(phase_counts.sum() / np.maximum(4.0 * phase_counts, 1.0))
        phase_weights = jnp.asarray(phase_weights / phase_weights.mean())

        def objective(current_params, batch, teacher_force_mask, recurrent_weight):
            output = training_model.apply(
                {"params": current_params},
                **_inputs(batch, teacher_force_mask),
                train=True,
            )
            micro_mask = jnp.asarray(batch["micro_mask"])
            update_mask = jnp.asarray(batch["event_type"] != 0) & micro_mask
            if args.variant == "joint":
                routing_loss = base._masked_ce(  # noqa: SLF001
                    output["event_type_logits"],
                    jnp.asarray(batch["event_type"]),
                    micro_mask,
                    joint_weights,
                )
                completion_loss = jnp.asarray(0.0)
                kind_loss = jnp.asarray(0.0)
            else:
                completion_target = jnp.asarray(
                    batch["event_type"] != 0, dtype=jnp.float32
                )
                values = optax.sigmoid_binary_cross_entropy(
                    output["completion_logits"], completion_target
                )
                weights = jnp.where(
                    completion_target > 0, args.completion_positive_weight, 1.0
                ) * micro_mask
                completion_loss = jnp.sum(values * weights) / jnp.maximum(
                    jnp.sum(weights), 1.0
                )
                kind_loss = base._masked_ce(  # noqa: SLF001
                    output["event_kind_logits"],
                    jnp.maximum(jnp.asarray(batch["event_type"]) - 1, 0),
                    update_mask,
                )
                routing_loss = completion_loss + kind_loss
            phase_loss = base._masked_ce(  # noqa: SLF001
                output["phase_logits"],
                jnp.asarray(batch["phase"]),
                jnp.asarray(batch["sequence_mask"]),
                phase_weights,
            )
            entity = base._masked_ce(  # noqa: SLF001
                output["write_entity_logits"],
                jnp.asarray(batch["write_entity"]),
                jnp.asarray(batch["write_mask"]),
            )
            region = base._masked_ce(  # noqa: SLF001
                output["write_region_logits"],
                jnp.asarray(batch["write_region"]),
                jnp.asarray(batch["write_mask"]),
            )
            pair = base._masked_ce(  # noqa: SLF001
                output["swap_pair_logits"],
                jnp.asarray(batch["swap_pair"]),
                jnp.asarray(batch["swap_mask"]),
            )
            table = transition._table_losses(output, batch)  # noqa: SLF001
            active_phase_weight = args.phase_weight if args.variant == "phase" else 0.0
            operation_loss = (
                routing_loss
                + active_phase_weight * phase_loss
                + args.write_entity_weight * entity
                + args.write_region_weight * region
                + args.swap_pair_weight * pair
            )
            loss = (
                operation_loss
                + recurrent_weight
                * (
                    args.transition_weight * table["transition_loss"]
                    + args.no_change_weight * table["no_change_loss"]
                    + args.delta_weight * table["delta_loss"]
                    + args.final_weight * table["final_loss"]
                    + args.trajectory_weight * table["trajectory_loss"]
                )
            )
            return loss, {
                "loss": loss,
                "routing_loss": routing_loss,
                "completion_loss": completion_loss,
                "kind_loss": kind_loss,
                "phase_loss": phase_loss,
                "write_entity_loss": entity,
                "write_region_loss": region,
                "swap_pair_loss": pair,
                **table,
            }

        @jax.jit
        def train_step(
            current_params,
            current_opt,
            batch,
            teacher_force_mask,
            recurrent_weight,
        ):
            (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(
                current_params, batch, teacher_force_mask, recurrent_weight
            )
            updates, next_opt = optimizer.update(grads, current_opt, current_params)
            return optax.apply_updates(current_params, updates), next_opt, metrics

        infer_cache = {}

        def infer_function(threshold: float):
            key = round(float(threshold), 6)
            if key not in infer_cache:
                evaluation_model = _model(args, train_data, key)

                @jax.jit
                def infer(current_params, batch, teacher_force_mask):
                    return evaluation_model.apply(
                        {"params": current_params},
                        **_inputs(batch, teacher_force_mask),
                        train=False,
                    )

                infer_cache[key] = infer
            return infer_cache[key]

        def evaluate(
            current_params,
            split: str,
            *,
            teacher_forcing: bool,
            threshold: float,
        ):
            infer = infer_function(threshold)
            outputs: dict[str, list[np.ndarray]] = defaultdict(list)
            batches = []
            data = datasets[split]
            for start in range(0, len(data.rows), args.eval_batch_size):
                indices = data.rows[start : start + args.eval_batch_size]
                valid_count = len(indices)
                if valid_count < args.eval_batch_size:
                    indices = np.pad(
                        indices,
                        (0, args.eval_batch_size - valid_count),
                        mode="edge",
                    )
                batch = data.parser_batch(indices)
                force = batch["sequence_mask"] & bool(teacher_forcing)
                output = jax.device_get(infer(current_params, batch, force))
                for key, value in output.items():
                    if key != "all_memories":
                        outputs[key].append(np.asarray(value)[:valid_count])
                batches.append(
                    {key: np.asarray(value)[:valid_count] for key, value in batch.items()}
                )
            merged_output = {
                key: np.concatenate(values) for key, values in outputs.items()
            }
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
            batch = train_data.parser_batch(train_data.sample(rng, args.batch_size))
            if step <= args.operation_pretrain_steps:
                ratio = 1.0
                recurrent_weight = 0.0
            else:
                recurrent_step = step - args.operation_pretrain_steps
                recurrent_steps = max(args.steps - args.operation_pretrain_steps, 1)
                ratio = transition._curriculum_ratio(  # noqa: SLF001
                    recurrent_step, recurrent_steps
                )
                recurrent_weight = min(
                    1.0,
                    recurrent_step / max(0.2 * recurrent_steps, 1.0),
                )
            teacher_force_mask = batch["sequence_mask"] & (
                rng.random(batch["sequence_mask"].shape) < ratio
            )
            params, opt_state, train_metrics = train_step(
                params,
                opt_state,
                batch,
                teacher_force_mask,
                jnp.asarray(recurrent_weight, dtype=jnp.float32),
            )
            if step % args.eval_every == 0 or step == args.steps:
                dev = evaluate(
                    params,
                    "dev",
                    teacher_forcing=False,
                    threshold=args.train_commit_threshold,
                )
                score = _score(dev)
                if step >= args.operation_pretrain_steps and score > best_score:
                    best_score = score
                    best_step = step
                    best_params = jax.device_get(params)
                row = {
                    "step": step,
                    "teacher_force_ratio": ratio,
                    "recurrent_weight": recurrent_weight,
                    "selection_score": score,
                    "train_batch": {
                        key: float(value) for key, value in train_metrics.items()
                    },
                    "dev_free_rollout": dev,
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        # The threshold is a static Flax field, so every candidate requires one
        # XLA compilation. These points cover the useful conservative regime
        # without turning calibration into dozens of equivalent recompiles.
        candidates = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])
        calibration = []
        for threshold in candidates:
            summary = evaluate(
                best_params,
                "dev",
                teacher_forcing=False,
                threshold=float(threshold),
            )
            routing = summary["routing"]
            feasible = routing["hold_false_positive_rate"] <= args.target_hold_fpr
            calibration.append(
                {
                    "threshold": float(threshold),
                    "feasible": feasible,
                    "selection_score": (
                        int(feasible),
                        *_score(summary),
                        routing["full_update_recall"],
                    ),
                    "summary": summary,
                }
            )
        selected = max(calibration, key=lambda row: row["selection_score"])
        selected_threshold = selected["threshold"]
        metrics = {
            split: {
                "teacher_forced": evaluate(
                    best_params,
                    split,
                    teacher_forcing=True,
                    threshold=selected_threshold,
                ),
                "free_rollout": evaluate(
                    best_params,
                    split,
                    teacher_forcing=False,
                    threshold=selected_threshold,
                ),
            }
            for split in ("train", "dev", "test")
        }
        result = {
            "schema_version": 1,
            "experiment": "robomme_free_feedback_event_head_ablation",
            "variant": args.variant,
            "best_step": best_step,
            "best_score_at_train_threshold": best_score,
            "selected_threshold": selected_threshold,
            "elapsed_seconds": time.monotonic() - started,
            "metrics": metrics,
            "calibration": calibration,
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
                    "event_head_mode": "joint" if args.variant == "joint" else "completion",
                    "active_phase_weight": args.phase_weight if args.variant == "phase" else 0.0,
                    "same_parameter_tree_across_variants": True,
                    "jax_devices": [str(device) for device in jax.devices()],
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(metrics, indent=2), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
