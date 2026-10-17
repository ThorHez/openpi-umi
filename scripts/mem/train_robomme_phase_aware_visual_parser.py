#!/usr/bin/env python3
"""Train an online phase-aware visual event parser with conservative commit."""

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

from openpi.tasks.robomme.causal_visual_operation_parser import CausalVisualOperationParser  # noqa: E402
from scripts.mem import eval_robomme_transition_causal_ablation as replay  # noqa: E402
from scripts.mem import train_robomme_anchor_conditioned_decomposition as anchor_base  # noqa: E402
from scripts.mem import train_robomme_anchor_transition_curriculum as transition  # noqa: E402
from scripts.mem import train_robomme_decomposed_region_distillation as base  # noqa: E402
from scripts.mem import train_robomme_visual_operation_parser_ablation as parser_base  # noqa: E402

DEFAULT_FEATURES = ROOT / "artifacts/robomme_fixed_chunk_rgb_grid8_v1_260829"
DEFAULT_PHASES = ROOT / "artifacts/robomme_fixed_chunk_phase_labels_v1_260829"
DEFAULT_OUTPUT = ROOT / "checkpoints/robomme_phase_aware_rgb8_seed260903_260829"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, default=base.DEFAULT_FIXED)
    parser.add_argument("--teacher-dir", type=Path, default=base.DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--phase-dir", type=Path, default=DEFAULT_PHASES)
    parser.add_argument("--anchor-dir", type=Path, default=anchor_base.DEFAULT_ANCHORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--end-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--phase-weight", type=float, default=0.5)
    parser.add_argument("--completion-weight", type=float, default=2.0)
    parser.add_argument("--completion-positive-weight", type=float, default=3.0)
    parser.add_argument("--kind-weight", type=float, default=1.0)
    parser.add_argument("--entity-weight", type=float, default=1.0)
    parser.add_argument("--region-weight", type=float, default=2.0)
    parser.add_argument("--pair-weight", type=float, default=2.0)
    parser.add_argument("--target-hold-fpr", type=float, default=0.005)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260903)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class PhaseDataset(parser_base.FixedParserDataset):
    def __init__(self, split: str, args: argparse.Namespace):
        super().__init__(split, args)
        with np.load(args.phase_dir / f"{split}.npz", allow_pickle=False) as payload:
            self.phase = np.asarray(payload["phase"], dtype=np.int32)
        if self.phase.shape != self.fixed["step_mask"].shape:
            raise ValueError(f"Phase shape mismatch on {split}: {self.phase.shape}")

    def parser_batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        result = super().parser_batch(indices)
        result["phase"] = self.phase[indices]
        return result


def _model_inputs(batch: dict[str, Any]) -> dict[str, jax.Array]:
    return parser_base._model_inputs(batch)  # noqa: SLF001


def _categorical_logits(indices: np.ndarray, classes: int) -> np.ndarray:
    return np.where(
        np.eye(classes, dtype=np.bool_)[indices],
        np.float32(20.0),
        np.float32(-20.0),
    )


def _threshold_output(
    output: dict[str, np.ndarray], threshold: float
) -> dict[str, np.ndarray]:
    completion = jax.nn.sigmoid(jnp.asarray(output["completion_logits"]))
    completion = np.asarray(completion) >= threshold
    kind = np.argmax(output["event_kind_logits"], axis=-1) + 1
    event_type = np.where(completion, kind, 0)
    result = dict(output)
    result["event_type_logits"] = _categorical_logits(event_type, 3)
    return result


def _phase_metrics(
    output: dict[str, np.ndarray], batch: dict[str, np.ndarray]
) -> dict[str, Any]:
    prediction = np.argmax(output["phase_logits"], axis=-1)
    target = batch["phase"]
    mask = batch["sequence_mask"]
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, guess in zip(target[mask], prediction[mask], strict=True):
        confusion[int(truth), int(guess)] += 1
    return {
        "accuracy": float((prediction[mask] == target[mask]).mean()),
        "confusion_gt_rows_pred_columns": confusion.tolist(),
    }


def _evaluate_threshold(
    output: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    thresholded = _threshold_output(output, threshold)
    operation = parser_base._metrics(thresholded, batch)  # noqa: SLF001
    replayed = replay._replay(  # noqa: SLF001
        thresholded,
        batch,
        oracle_event=False,
        oracle_payload=False,
    )
    rollout = transition._summary(replayed, batch)  # noqa: SLF001
    return {
        "threshold": threshold,
        "operation": operation,
        "rollout": rollout,
        "phase": _phase_metrics(output, batch),
    }


def _select_threshold(
    output: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    target_fpr: float,
) -> tuple[float, dict[str, Any]]:
    candidates = np.concatenate(
        (
            np.linspace(0.05, 0.95, 37),
            np.asarray([0.97, 0.98, 0.99, 0.995, 0.999]),
        )
    )
    rows = [_evaluate_threshold(output, batch, float(value)) for value in candidates]

    def score(row):
        operation = row["operation"]
        feasible = operation["hold_false_positive_rate"] <= target_fpr
        return (
            int(feasible),
            operation["full_update_recall"] if feasible else -operation["hold_false_positive_rate"],
            operation["update_exact_type_recall"],
            operation["update_exact_type_precision"],
            operation["update_payload_accuracy"],
        )

    selected = max(rows, key=score)
    return float(selected["threshold"]), selected


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: PhaseDataset(split, args) for split in ("train", "dev", "test")}
    try:
        train_data = datasets["train"]
        model = CausalVisualOperationParser(
            max_steps=train_data.max_parser_steps,
            spatial_tokens=train_data.spatial_tokens,
            input_width=train_data.patch_width,
            recurrent_event_state=True,
        )
        rng = np.random.default_rng(args.seed)
        initial = train_data.parser_batch(train_data.sample(rng, args.batch_size))
        params = model.init(
            jax.random.key(args.seed), **_model_inputs(initial), train=False
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
        phase_values = train_data.phase[train_data.rows]
        phase_mask = train_data.fixed["step_mask"][train_data.rows]
        phase_counts = np.bincount(phase_values[phase_mask], minlength=4).astype(np.float32)
        phase_weights = np.sqrt(phase_counts.sum() / np.maximum(4.0 * phase_counts, 1.0))
        phase_weights = jnp.asarray(phase_weights / phase_weights.mean())

        def objective(current_params, batch):
            output = model.apply(
                {"params": current_params}, **_model_inputs(batch), train=True
            )
            sequence_mask = jnp.asarray(batch["sequence_mask"])
            micro_mask = jnp.asarray(batch["micro_mask"])
            update_mask = jnp.asarray(batch["event_type"] != 0) & micro_mask
            phase_loss = base._masked_ce(  # noqa: SLF001
                output["phase_logits"],
                jnp.asarray(batch["phase"]),
                sequence_mask,
                phase_weights,
            )
            completion_target = jnp.asarray(batch["event_type"] != 0, dtype=jnp.float32)
            completion_loss_values = optax.sigmoid_binary_cross_entropy(
                output["completion_logits"], completion_target
            )
            completion_weights = jnp.where(
                completion_target > 0,
                args.completion_positive_weight,
                1.0,
            ) * micro_mask
            completion_loss = jnp.sum(completion_loss_values * completion_weights) / jnp.maximum(
                jnp.sum(completion_weights), 1.0
            )
            kind_target = jnp.maximum(jnp.asarray(batch["event_type"]) - 1, 0)
            kind_loss = base._masked_ce(  # noqa: SLF001
                output["event_kind_logits"], kind_target, update_mask
            )
            entity_loss = base._masked_ce(  # noqa: SLF001
                output["write_entity_logits"],
                jnp.asarray(batch["write_entity"]),
                jnp.asarray(batch["write_mask"]),
            )
            region_loss = base._masked_ce(  # noqa: SLF001
                output["write_region_logits"],
                jnp.asarray(batch["write_region"]),
                jnp.asarray(batch["write_mask"]),
            )
            pair_loss = base._masked_ce(  # noqa: SLF001
                output["swap_pair_logits"],
                jnp.asarray(batch["swap_pair"]),
                jnp.asarray(batch["swap_mask"]),
            )
            loss = (
                args.phase_weight * phase_loss
                + args.completion_weight * completion_loss
                + args.kind_weight * kind_loss
                + args.entity_weight * entity_loss
                + args.region_weight * region_loss
                + args.pair_weight * pair_loss
            )
            return loss, {
                "loss": loss,
                "phase_loss": phase_loss,
                "completion_loss": completion_loss,
                "kind_loss": kind_loss,
                "entity_loss": entity_loss,
                "region_loss": region_loss,
                "pair_loss": pair_loss,
            }

        @jax.jit
        def train_step(current_params, current_opt_state, batch):
            (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(
                current_params, batch
            )
            updates, next_opt_state = optimizer.update(
                grads, current_opt_state, current_params
            )
            return (
                optax.apply_updates(current_params, updates),
                next_opt_state,
                metrics,
            )

        @jax.jit
        def infer(current_params, batch):
            return model.apply(
                {"params": current_params}, **_model_inputs(batch), train=False
            )

        def infer_split(current_params, split: str):
            data = datasets[split]
            outputs: dict[str, list[np.ndarray]] = defaultdict(list)
            batches = []
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
                output = jax.device_get(infer(current_params, batch))
                for key, value in output.items():
                    outputs[key].append(np.asarray(value)[:valid_count])
                batches.append(
                    {key: np.asarray(value)[:valid_count] for key, value in batch.items()}
                )
            return (
                {key: np.concatenate(values) for key, values in outputs.items()},
                {
                    key: np.concatenate([batch[key] for batch in batches])
                    for key in batches[0]
                },
            )

        best_params = params
        best_step = 0
        best_threshold = 0.999
        best_score = (-1.0,) * 5
        history = []
        started = time.monotonic()
        for step in range(1, args.steps + 1):
            batch = train_data.parser_batch(train_data.sample(rng, args.batch_size))
            params, opt_state, train_metrics = train_step(params, opt_state, batch)
            if step % args.eval_every == 0 or step == args.steps:
                dev_output, dev_batch = infer_split(params, "dev")
                threshold, dev = _select_threshold(
                    dev_output, dev_batch, args.target_hold_fpr
                )
                operation = dev["operation"]
                score = (
                    operation["full_update_recall"],
                    operation["update_exact_type_recall"],
                    operation["update_exact_type_precision"],
                    operation["update_payload_accuracy"],
                    dev["rollout"]["overall"]["mean_task_final_query_accuracy"],
                )
                if score > best_score:
                    best_score = score
                    best_step = step
                    best_threshold = threshold
                    best_params = jax.device_get(params)
                row = {
                    "step": step,
                    "threshold": threshold,
                    "selection_score": score,
                    "train_batch": {
                        key: float(value) for key, value in train_metrics.items()
                    },
                    "dev": dev,
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        metrics = {}
        for split in ("train", "dev", "test"):
            output, batch = infer_split(best_params, split)
            metrics[split] = _evaluate_threshold(output, batch, best_threshold)
        result = {
            "schema_version": 1,
            "experiment": "robomme_phase_aware_rgb8_conservative_commit",
            "best_step": best_step,
            "best_threshold": best_threshold,
            "best_score": best_score,
            "elapsed_seconds": time.monotonic() - started,
            "metrics": metrics,
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
                    "phase_class_weights": np.asarray(phase_weights).tolist(),
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
