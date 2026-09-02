#!/usr/bin/env python3
"""Attribute the visual-ceiling gap to routing, payload, or recurrent updates.

The evaluator reuses one trained anchor-conditioned transition checkpoint and
the locked dataset split.  No weights are retrained.  Oracle inputs are used
only by explicitly named diagnostic rows and never by the deployable rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.tasks.robomme.anchor_conditioned_transition_memory import AnchorConditionedTransitionMemory  # noqa: E402
from openpi.tasks.robomme.decomposed_region_recurrent_memory import SWAP_PAIRS  # noqa: E402
from scripts.mem import train_robomme_anchor_conditioned_decomposition as anchor_base  # noqa: E402
from scripts.mem import train_robomme_anchor_transition_curriculum as transition  # noqa: E402
from scripts.mem import train_robomme_decomposed_region_distillation as base  # noqa: E402

DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints/robomme_anchor_transition_curriculum_seed260901_260829/params.msgpack"
)
DEFAULT_OUTPUT = (
    ROOT
    / "checkpoints/robomme_transition_causal_ablation_seed260901_260829/result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, default=base.DEFAULT_FIXED)
    parser.add_argument("--teacher-dir", type=Path, default=base.DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=base.DEFAULT_FEATURES)
    parser.add_argument("--anchor-dir", type=Path, default=anchor_base.DEFAULT_ANCHORS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-batch-size", type=int, default=3)
    parser.add_argument("--gate-temperature", type=float, default=0.25)
    return parser.parse_args()


def _hard_apply(
    table: np.ndarray,
    event_type: int,
    write_entity: int,
    write_region: int,
    swap_pair: int,
) -> np.ndarray:
    result = table.copy()
    if event_type == 1:
        result[write_entity] = write_region + 1
    elif event_type == 2:
        region_a, region_b = SWAP_PAIRS[swap_pair]
        value_a, value_b = region_a + 1, region_b + 1
        is_a = result == value_a
        is_b = result == value_b
        result[is_a] = value_b
        result[is_b] = value_a
    return result


def _categorical_logits(indices: np.ndarray, classes: int) -> np.ndarray:
    return np.where(
        np.eye(classes, dtype=np.bool_)[indices],
        np.float32(20.0),
        np.float32(-20.0),
    )


def _replay(
    logits: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    *,
    oracle_event: bool,
    oracle_payload: bool,
) -> dict[str, np.ndarray]:
    """Replay categorical operations with independently selected oracle sources."""

    predicted_event = np.argmax(logits["event_type_logits"], axis=-1)
    predicted_entity = np.argmax(logits["write_entity_logits"], axis=-1)
    predicted_region = np.argmax(logits["write_region_logits"], axis=-1)
    predicted_pair = np.argmax(logits["swap_pair_logits"], axis=-1)
    applied_event = batch["event_type"].copy() if oracle_event else predicted_event.copy()
    applied_entity = predicted_entity.copy()
    applied_region = predicted_region.copy()
    applied_pair = predicted_pair.copy()

    if oracle_payload:
        # Payload is defined only on a real GT update.  A learned false-positive
        # event has no oracle payload and deliberately retains its learned one.
        applied_entity[batch["write_mask"]] = batch["write_entity"][batch["write_mask"]]
        applied_region[batch["write_mask"]] = batch["write_region"][batch["write_mask"]]
        applied_pair[batch["swap_mask"]] = batch["swap_pair"][batch["swap_mask"]]

    batch_size, steps, micro_events = applied_event.shape
    tables = np.zeros((batch_size, steps + 1, 7), dtype=np.int32)
    for row in range(batch_size):
        for step in range(steps):
            tables[row, step + 1] = tables[row, step]
            if not batch["sequence_mask"][row, step]:
                continue
            for micro in range(micro_events):
                tables[row, step + 1] = _hard_apply(
                    tables[row, step + 1],
                    int(applied_event[row, step, micro]),
                    int(applied_entity[row, step, micro]),
                    int(applied_region[row, step, micro]),
                    int(applied_pair[row, step, micro]),
                )

    return {
        "all_tables": np.eye(5, dtype=np.float32)[tables],
        "event_type_logits": _categorical_logits(applied_event, 3),
        "write_entity_logits": _categorical_logits(applied_entity, 7),
        "write_region_logits": _categorical_logits(applied_region, 4),
        "swap_pair_logits": _categorical_logits(applied_pair, len(SWAP_PAIRS)),
    }


def _routing_diagnostics(
    output: dict[str, np.ndarray], batch: dict[str, np.ndarray]
) -> dict[str, Any]:
    predicted = np.argmax(output["event_type_logits"], axis=-1)
    target = batch["event_type"]
    mask = batch["micro_mask"]
    confusion = np.zeros((3, 3), dtype=np.int64)
    for truth, guess in zip(target[mask], predicted[mask], strict=True):
        confusion[int(truth), int(guess)] += 1
    gt_update = (target != 0) & mask
    pred_update = (predicted != 0) & mask
    true_update = gt_update & pred_update & (target == predicted)
    hold = (target == 0) & mask
    false_update = hold & pred_update
    episode_exact = [
        bool(np.all(predicted[row][mask[row]] == target[row][mask[row]]))
        for row in range(len(mask))
    ]

    def safe_ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "confusion_gt_rows_pred_columns": confusion.tolist(),
        "hold_false_positive_rate": safe_ratio(int(false_update.sum()), int(hold.sum())),
        "update_exact_type_recall": safe_ratio(int(true_update.sum()), int(gt_update.sum())),
        "update_exact_type_precision": safe_ratio(int(true_update.sum()), int(pred_update.sum())),
        "episode_exact_event_sequence_accuracy": float(np.mean(episode_exact)),
        "valid_micro_events": int(mask.sum()),
        "gt_updates": int(gt_update.sum()),
        "predicted_updates": int(pred_update.sum()),
    }


def _payload_diagnostics(
    output: dict[str, np.ndarray], batch: dict[str, np.ndarray]
) -> dict[str, Any]:
    event = np.argmax(output["event_type_logits"], axis=-1)
    entity = np.argmax(output["write_entity_logits"], axis=-1)
    region = np.argmax(output["write_region_logits"], axis=-1)
    pair = np.argmax(output["swap_pair_logits"], axis=-1)
    write = batch["write_mask"]
    swap = batch["swap_mask"]
    write_payload = (
        (entity == batch["write_entity"])
        & (region == batch["write_region"])
        & write
    )
    swap_payload = (pair == batch["swap_pair"]) & swap
    update_payload_correct = int(write_payload.sum() + swap_payload.sum())
    update_count = int(write.sum() + swap.sum())
    routed_update_correct = int(
        (write_payload & (event == 1)).sum()
        + (swap_payload & (event == 2)).sum()
    )
    return {
        "write_payload_joint_accuracy": float(write_payload.sum() / write.sum()),
        "swap_payload_accuracy": float(swap_payload.sum() / swap.sum()),
        "update_payload_accuracy_given_oracle_event": float(
            update_payload_correct / update_count
        ),
        "full_update_recall": float(routed_update_correct / update_count),
    }


def _summarize(
    output: dict[str, np.ndarray], batch: dict[str, np.ndarray]
) -> dict[str, Any]:
    result = transition._summary(output, batch)  # noqa: SLF001
    result["routing"] = _routing_diagnostics(output, batch)
    result["payload"] = _payload_diagnostics(output, batch)
    return result


def _concat(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([item[key] for item in items], axis=0)
        for key in items[0]
    }


def main() -> None:
    args = parse_args()
    if args.eval_batch_size % 3:
        raise ValueError("eval batch size must be divisible by three")
    datasets = {
        split: anchor_base.AnchorRegionDataset(split, args)
        for split in ("train", "dev", "test")
    }
    try:
        max_steps = datasets["train"].max_steps
        soft_model = AnchorConditionedTransitionMemory(
            max_steps=max_steps,
            gate_temperature=args.gate_temperature,
            straight_through_hard_feedback=True,
            hard_event_commit=False,
        )
        hard_model = AnchorConditionedTransitionMemory(
            max_steps=max_steps,
            gate_temperature=args.gate_temperature,
            straight_through_hard_feedback=True,
            hard_event_commit=True,
        )
        params = flax.serialization.msgpack_restore(args.checkpoint.read_bytes())

        @jax.jit
        def infer_soft(batch, teacher_force_mask):
            return soft_model.apply(
                {"params": params},
                **transition._inputs(batch, teacher_force_mask),  # noqa: SLF001
                train=False,
            )

        @jax.jit
        def infer_hard(batch):
            return hard_model.apply(
                {"params": params},
                **transition._inputs(batch, jnp.zeros_like(batch["sequence_mask"])),  # noqa: SLF001
                train=False,
            )

        result: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "robomme_transition_causal_four_way_ablation",
            "checkpoint": str(args.checkpoint),
            "protocol": {
                "A_oracle_event_learned_payload": "GT-state local logits; oracle event type; learned payload; hard replay",
                "B_learned_event_oracle_payload": "GT-state local logits; learned event type; GT payload on real GT updates; hard replay",
                "C_full_learned_hard_updater": "free rollout; learned event and payload; categorical hard commit",
                "D_full_learned_soft_updater": "free rollout; learned event and payload; original temperature-0.25 soft commit",
                "diagnostic_local_full_hard": "GT-state local logits; learned event and payload; hard replay",
                "sanity_oracle_oracle": "oracle event and payload; hard replay",
            },
            "splits": {},
        }
        for split, dataset in datasets.items():
            batch_parts: list[dict[str, np.ndarray]] = []
            local_parts: list[dict[str, np.ndarray]] = []
            soft_parts: list[dict[str, np.ndarray]] = []
            hard_parts: list[dict[str, np.ndarray]] = []
            rows = dataset.rows
            for start in range(0, len(rows), args.eval_batch_size):
                indices = rows[start : start + args.eval_batch_size]
                valid_count = len(indices)
                if valid_count < args.eval_batch_size:
                    indices = np.pad(
                        indices,
                        (0, args.eval_batch_size - valid_count),
                        mode="edge",
                    )
                batch = dataset.batch(indices)
                local = jax.device_get(
                    infer_soft(batch, jnp.asarray(batch["sequence_mask"]))
                )
                soft = jax.device_get(
                    infer_soft(batch, jnp.zeros_like(batch["sequence_mask"]))
                )
                hard = jax.device_get(infer_hard(batch))
                batch_parts.append(
                    {key: np.asarray(value)[:valid_count] for key, value in batch.items()}
                )
                for source, destination in (
                    (local, local_parts),
                    (soft, soft_parts),
                    (hard, hard_parts),
                ):
                    destination.append(
                        {
                            key: np.asarray(value)[:valid_count]
                            for key, value in source.items()
                            if key != "all_memories"
                        }
                    )

            merged_batch = _concat(batch_parts)
            local = _concat(local_parts)
            soft = _concat(soft_parts)
            hard = _concat(hard_parts)
            variants = {
                "A_oracle_event_learned_payload": _replay(
                    local, merged_batch, oracle_event=True, oracle_payload=False
                ),
                "B_learned_event_oracle_payload": _replay(
                    local, merged_batch, oracle_event=False, oracle_payload=True
                ),
                "C_full_learned_hard_updater": hard,
                "D_full_learned_soft_updater": soft,
                "diagnostic_local_full_hard": _replay(
                    local, merged_batch, oracle_event=False, oracle_payload=False
                ),
                "sanity_oracle_oracle": _replay(
                    local, merged_batch, oracle_event=True, oracle_payload=True
                ),
            }
            result["splits"][split] = {
                name: _summarize(output, merged_batch)
                for name, output in variants.items()
            }
            print(json.dumps({split: result["splits"][split]}, sort_keys=True), flush=True)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"output": str(args.output)}, indent=2), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
