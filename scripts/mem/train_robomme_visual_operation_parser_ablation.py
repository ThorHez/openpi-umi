#!/usr/bin/env python3
"""Train controlled RoboMME visual operation parser ablations."""

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
from scripts.mem import train_robomme_anchor_conditioned_decomposition as anchor_base  # noqa: E402
from scripts.mem import train_robomme_decomposed_region_distillation as base  # noqa: E402

DEFAULT_EVENT_FEATURES_4 = (
    ROOT / "artifacts/robomme_four_task_visual_features_4x4_v1_260826"
)
DEFAULT_OUTPUT = ROOT / "checkpoints/robomme_visual_operation_parser_ablation_260829"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed", "event"), required=True)
    parser.add_argument("--recurrent", action="store_true")
    parser.add_argument("--fixed-dir", type=Path, default=base.DEFAULT_FIXED)
    parser.add_argument("--teacher-dir", type=Path, default=base.DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_EVENT_FEATURES_4)
    parser.add_argument("--anchor-dir", type=Path, default=anchor_base.DEFAULT_ANCHORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--end-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--event-weight", type=float, default=1.0)
    parser.add_argument("--entity-weight", type=float, default=1.0)
    parser.add_argument("--region-weight", type=float, default=2.0)
    parser.add_argument("--pair-weight", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260902)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class FixedParserDataset(anchor_base.AnchorRegionDataset):
    """Existing online fixed-chunk corpus with GT previous semantic state."""

    def __init__(self, split: str, args: argparse.Namespace):
        super().__init__(split, args)
        self.max_parser_steps = self.max_steps
        self.spatial_tokens = int(self.features.attrs["spatial_tokens"])
        self.patch_width = int(self.features.attrs["patch_width"])

    def parser_batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        batch_size = len(indices)
        patches = np.zeros(
            (
                batch_size,
                self.max_steps,
                12,
                self.spatial_tokens,
                self.patch_width,
            ),
            dtype=np.float16,
        )
        for batch_row, row in enumerate(indices):
            values = self.features[f"episode_{int(row):06d}/patch_tokens"][()]
            patches[batch_row, : len(values)] = values
        return {
            "patch_tokens": patches,
            "sequence_mask": self.fixed["step_mask"][indices],
            "task_ids": self.fixed["task_ids"][indices],
            "goal_color_ids": self.fixed["goal_color_ids"][indices],
            "queried_ordinals": self.fixed["queried_ordinals"][indices],
            "num_regions": self.fixed["num_regions"][indices],
            "anchor_yx": self.anchor_yx[indices],
            "anchor_mask": self.anchor_mask[indices],
            "previous_tables": self.table_targets[indices, :-1],
            "event_type": self.event_type[indices],
            "write_entity": self.write_entity[indices],
            "write_region": self.write_region[indices],
            "swap_pair": self.swap_pair[indices],
            "micro_mask": self.micro_mask[indices],
            "write_mask": self.write_mask[indices],
            "swap_mask": self.swap_mask[indices],
            "table_targets": self.table_targets[indices],
            "table_mask": self.table_mask[indices],
            "state_change_mask": self.fixed["state_change_mask"][indices],
        }


class EventParserDataset(anchor_base.AnchorRegionDataset):
    """Oracle-selected complete event windows; no hold examples are present."""

    def __init__(self, split: str, args: argparse.Namespace):
        super().__init__(split, args)
        self.max_parser_steps = 1
        self.spatial_tokens = int(self.features.attrs["spatial_tokens"])
        self.patch_width = int(self.features.attrs["patch_width"])
        self.samples: list[tuple[int, int, int, int]] = []
        for row_value in self.rows:
            row = int(row_value)
            event_ordinal = 0
            length = int(self.fixed["step_mask"][row].sum())
            for chunk in range(length):
                for micro in range(2):
                    if int(self.event_type[row, chunk, micro]) == 0:
                        continue
                    self.samples.append((row, event_ordinal, chunk, micro))
                    event_ordinal += 1
            cached = int(self.features[f"episode_{row:06d}/patch_tokens"].shape[0])
            if cached != event_ordinal:
                raise ValueError(
                    f"Event/cache mismatch {split}:{row}: {event_ordinal} != {cached}"
                )
        self.sample_task_ids = np.asarray(
            [self.fixed["task_ids"][row] for row, _, _, _ in self.samples],
            dtype=np.int32,
        )

    def sample(self, rng: np.random.Generator, batch_size: int) -> np.ndarray:
        if batch_size % 3:
            raise ValueError("batch size must be divisible by three")
        per_task = batch_size // 3
        selected = []
        for task_id in range(3):
            candidates = np.flatnonzero(self.sample_task_ids == task_id)
            selected.extend(rng.choice(candidates, per_task, replace=True))
        rng.shuffle(selected)
        return np.asarray(selected, dtype=np.int32)

    @property
    def parser_rows(self) -> np.ndarray:
        return np.arange(len(self.samples), dtype=np.int32)

    def parser_batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        batch_size = len(indices)
        patches = np.zeros(
            (batch_size, 1, 12, self.spatial_tokens, self.patch_width),
            dtype=np.float16,
        )
        task_ids = np.zeros(batch_size, dtype=np.int32)
        goal_color_ids = np.zeros((batch_size, 2), dtype=np.int32)
        queried_ordinals = np.zeros(batch_size, dtype=np.int32)
        num_regions = np.zeros(batch_size, dtype=np.int32)
        anchor_yx = np.zeros((batch_size, 4, 2), dtype=np.float32)
        anchor_mask = np.zeros((batch_size, 4), dtype=np.bool_)
        previous_tables = np.zeros((batch_size, 1, 7), dtype=np.int32)
        event_type = np.zeros((batch_size, 1, 2), dtype=np.int32)
        write_entity = np.zeros_like(event_type)
        write_region = np.zeros_like(event_type)
        swap_pair = np.zeros_like(event_type)
        write_mask = np.zeros_like(event_type, dtype=np.bool_)
        swap_mask = np.zeros_like(event_type, dtype=np.bool_)
        for batch_row, sample_index in enumerate(indices):
            row, event_ordinal, chunk, micro = self.samples[int(sample_index)]
            patches[batch_row, 0] = self.features[
                f"episode_{row:06d}/patch_tokens"
            ][event_ordinal]
            task_ids[batch_row] = self.fixed["task_ids"][row]
            goal_color_ids[batch_row] = self.fixed["goal_color_ids"][row]
            queried_ordinals[batch_row] = self.fixed["queried_ordinals"][row]
            num_regions[batch_row] = self.fixed["num_regions"][row]
            anchor_yx[batch_row] = self.anchor_yx[row]
            anchor_mask[batch_row] = self.anchor_mask[row]
            previous_tables[batch_row, 0] = self.table_targets[row, event_ordinal]
            event_type[batch_row, 0, 0] = self.event_type[row, chunk, micro]
            write_entity[batch_row, 0, 0] = self.write_entity[row, chunk, micro]
            write_region[batch_row, 0, 0] = self.write_region[row, chunk, micro]
            swap_pair[batch_row, 0, 0] = self.swap_pair[row, chunk, micro]
            write_mask[batch_row, 0, 0] = self.write_mask[row, chunk, micro]
            swap_mask[batch_row, 0, 0] = self.swap_mask[row, chunk, micro]
        return {
            "patch_tokens": patches,
            "sequence_mask": np.ones((batch_size, 1), dtype=np.bool_),
            "task_ids": task_ids,
            "goal_color_ids": goal_color_ids,
            "queried_ordinals": queried_ordinals,
            "num_regions": num_regions,
            "anchor_yx": anchor_yx,
            "anchor_mask": anchor_mask,
            "previous_tables": previous_tables,
            "event_type": event_type,
            "write_entity": write_entity,
            "write_region": write_region,
            "swap_pair": swap_pair,
            "micro_mask": write_mask | swap_mask,
            "write_mask": write_mask,
            "swap_mask": swap_mask,
        }


def _model_inputs(batch: dict[str, Any]) -> dict[str, jax.Array]:
    return {
        key: jnp.asarray(batch[key])
        for key in (
            "patch_tokens",
            "sequence_mask",
            "task_ids",
            "goal_color_ids",
            "queried_ordinals",
            "num_regions",
            "anchor_yx",
            "anchor_mask",
            "previous_tables",
        )
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _metrics(output: dict[str, np.ndarray], batch: dict[str, np.ndarray]) -> dict[str, Any]:
    event = np.argmax(output["event_type_logits"], axis=-1)
    entity = np.argmax(output["write_entity_logits"], axis=-1)
    region = np.argmax(output["write_region_logits"], axis=-1)
    pair = np.argmax(output["swap_pair_logits"], axis=-1)
    target = batch["event_type"]
    valid = batch["micro_mask"]
    write = batch["write_mask"]
    swap = batch["swap_mask"]
    hold = valid & (target == 0)
    gt_update = valid & (target != 0)
    pred_update = valid & (event != 0)
    exact_type = gt_update & (event == target)
    write_payload = write & (entity == batch["write_entity"]) & (
        region == batch["write_region"]
    )
    swap_payload = swap & (pair == batch["swap_pair"])
    payload_correct = write_payload | swap_payload
    full_correct = exact_type & payload_correct
    confusion = np.zeros((3, 3), dtype=np.int64)
    for truth, guess in zip(target[valid], event[valid], strict=True):
        confusion[int(truth), int(guess)] += 1
    result = {
        "examples": int(valid.sum()),
        "confusion_gt_rows_pred_columns": confusion.tolist(),
        "event_type_accuracy": float((event[valid] == target[valid]).mean()),
        "hold_false_positive_rate": _safe_ratio(
            int((hold & pred_update).sum()), int(hold.sum())
        ),
        "update_exact_type_recall": _safe_ratio(int(exact_type.sum()), int(gt_update.sum())),
        "update_exact_type_precision": _safe_ratio(int(exact_type.sum()), int(pred_update.sum())),
        "write_entity_accuracy": _safe_ratio(
            int((write & (entity == batch["write_entity"])).sum()), int(write.sum())
        ),
        "write_region_accuracy": _safe_ratio(
            int((write & (region == batch["write_region"])).sum()), int(write.sum())
        ),
        "write_payload_joint_accuracy": _safe_ratio(int(write_payload.sum()), int(write.sum())),
        "swap_pair_accuracy": _safe_ratio(int(swap_payload.sum()), int(swap.sum())),
        "update_payload_accuracy": _safe_ratio(int(payload_correct.sum()), int(gt_update.sum())),
        "full_update_recall": _safe_ratio(int(full_correct.sum()), int(gt_update.sum())),
    }
    per_task = {}
    for task_id, task_name in enumerate(base.TASKS):
        task_mask = batch["task_ids"][:, None, None] == task_id
        selected = gt_update & task_mask
        per_task[task_name] = {
            "updates": int(selected.sum()),
            "payload_accuracy": _safe_ratio(
                int((payload_correct & selected).sum()), int(selected.sum())
            ),
            "full_update_recall": _safe_ratio(
                int((full_correct & selected).sum()), int(selected.sum())
            ),
        }
    result["per_task"] = per_task
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "event" and args.recurrent:
        raise ValueError("Event-window upper bounds are intentionally local")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_type = FixedParserDataset if args.mode == "fixed" else EventParserDataset
    datasets = {split: dataset_type(split, args) for split in ("train", "dev", "test")}
    try:
        train_data = datasets["train"]
        model = CausalVisualOperationParser(
            max_steps=train_data.max_parser_steps,
            spatial_tokens=train_data.spatial_tokens,
            input_width=train_data.patch_width,
            recurrent_event_state=args.recurrent,
        )
        rng = np.random.default_rng(args.seed)
        initial_indices = train_data.sample(rng, args.batch_size)
        initial = train_data.parser_batch(initial_indices)
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
        if args.mode == "fixed":
            event_weights = np.sqrt(train_data.event_type_weights)
            event_weights = event_weights / event_weights.mean()
        else:
            counts = np.bincount(
                [train_data.event_type[row, chunk, micro] for row, _, chunk, micro in train_data.samples],
                minlength=3,
            ).astype(np.float32)
            event_weights = np.zeros(3, dtype=np.float32)
            event_weights[1:] = counts[1:].sum() / np.maximum(2.0 * counts[1:], 1.0)
        event_weights = jnp.asarray(event_weights)

        def objective(current_params, batch):
            output = model.apply(
                {"params": current_params}, **_model_inputs(batch), train=True
            )
            event_loss = base._masked_ce(  # noqa: SLF001
                output["event_type_logits"],
                jnp.asarray(batch["event_type"]),
                jnp.asarray(batch["micro_mask"]),
                event_weights,
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
                args.event_weight * event_loss
                + args.entity_weight * entity_loss
                + args.region_weight * region_loss
                + args.pair_weight * pair_loss
            )
            return loss, {
                "loss": loss,
                "event_loss": event_loss,
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

        def evaluate(current_params, split: str):
            data = datasets[split]
            rows = data.rows if args.mode == "fixed" else data.parser_rows
            outputs: dict[str, list[np.ndarray]] = defaultdict(list)
            batches = []
            for start in range(0, len(rows), args.eval_batch_size):
                indices = rows[start : start + args.eval_batch_size]
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
            merged_output = {
                key: np.concatenate(values) for key, values in outputs.items()
            }
            merged_batch = {
                key: np.concatenate([batch[key] for batch in batches])
                for key in batches[0]
            }
            return _metrics(merged_output, merged_batch)

        best_params = params
        best_step = 0
        best_score = (-1.0,) * 4
        history = []
        started = time.monotonic()
        for step in range(1, args.steps + 1):
            indices = train_data.sample(rng, args.batch_size)
            batch = train_data.parser_batch(indices)
            params, opt_state, train_metrics = train_step(params, opt_state, batch)
            if step % args.eval_every == 0 or step == args.steps:
                dev = evaluate(params, "dev")
                if args.mode == "fixed":
                    score = (
                        min(
                            1.0 - dev["hold_false_positive_rate"],
                            dev["update_exact_type_recall"],
                            dev["update_payload_accuracy"],
                        ),
                        dev["full_update_recall"],
                        dev["update_payload_accuracy"],
                        -dev["hold_false_positive_rate"],
                    )
                else:
                    score = (
                        dev["update_payload_accuracy"],
                        dev["write_payload_joint_accuracy"],
                        dev["swap_pair_accuracy"],
                        dev["full_update_recall"],
                    )
                if score > best_score:
                    best_score = score
                    best_step = step
                    best_params = jax.device_get(params)
                row = {
                    "step": step,
                    "selection_score": score,
                    "train_batch": {
                        key: float(value) for key, value in train_metrics.items()
                    },
                    "dev": dev,
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        result = {
            "schema_version": 1,
            "experiment": "robomme_visual_operation_parser_ablation",
            "mode": args.mode,
            "recurrent_event_state": args.recurrent,
            "spatial_tokens": train_data.spatial_tokens,
            "best_step": best_step,
            "best_score": best_score,
            "elapsed_seconds": time.monotonic() - started,
            "metrics": {
                split: evaluate(best_params, split)
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
                    "event_class_weights": np.asarray(event_weights).tolist(),
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
