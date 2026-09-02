#!/usr/bin/env python3
"""Hybrid rollout ablation for anchor-conditioned RoboMME operation heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import flax
import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.tasks.robomme.anchor_conditioned_decomposed_memory import (  # noqa: E402
    AnchorConditionedDecomposedMemory,
)
from openpi.tasks.robomme.decomposed_region_recurrent_memory import (  # noqa: E402
    DecomposedRegionRecurrentMemory,
    SWAP_PAIRS,
)
from scripts.mem import train_robomme_anchor_conditioned_decomposition as anchor_train  # noqa: E402
from scripts.mem import train_robomme_decomposed_region_distillation as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, default=base.DEFAULT_FIXED)
    parser.add_argument("--teacher-dir", type=Path, default=base.DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=base.DEFAULT_FEATURES)
    parser.add_argument("--anchor-dir", type=Path, default=anchor_train.DEFAULT_ANCHORS)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=(
            ROOT
            / "checkpoints/robomme_decomposed_region_operation_only_seed260830_260828/params.msgpack"
        ),
    )
    parser.add_argument(
        "--anchor-checkpoint",
        type=Path,
        default=(
            ROOT
            / "checkpoints/robomme_anchor_conditioned_decomposition_seed260831_260828/params.msgpack"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "checkpoints/robomme_anchor_conditioned_decomposition_seed260831_260828/hybrid_ablation.json"
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=3)
    parser.add_argument("--gate-temperature", type=float, default=0.25)
    return parser.parse_args()


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    values = values.astype(np.float64) / temperature
    values = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=-1, keepdims=True)


def _rollout(
    type_logits: np.ndarray,
    entity_logits: np.ndarray,
    region_logits: np.ndarray,
    pair_logits: np.ndarray,
    sequence_mask: np.ndarray,
    *,
    temperature: float,
    oracle_type: np.ndarray | None = None,
    oracle_entity: np.ndarray | None = None,
) -> np.ndarray:
    batch, steps, micro_events = type_logits.shape[:3]
    table = np.zeros((batch, 7, 5), dtype=np.float64)
    table[:, :, 0] = 1.0
    all_tables = [table.copy()]
    for step in range(steps):
        candidate = table.copy()
        for micro in range(micro_events):
            if oracle_type is None:
                gates = _softmax(type_logits[:, step, micro], temperature)
            else:
                gates = np.eye(3)[oracle_type[:, step, micro]]
            entity_ids = (
                np.argmax(entity_logits[:, step, micro], axis=-1)
                if oracle_entity is None
                else oracle_entity[:, step, micro]
            )
            region_ids = np.argmax(region_logits[:, step, micro], axis=-1)
            pair_ids = np.argmax(pair_logits[:, step, micro], axis=-1)
            write = candidate.copy()
            for row in range(batch):
                write[row, entity_ids[row]] = 0.0
                write[row, entity_ids[row], region_ids[row] + 1] = 1.0
            swapped = np.empty_like(candidate)
            for row in range(batch):
                swapped[row] = candidate[row]
                region_a, region_b = SWAP_PAIRS[pair_ids[row]]
                swapped[row, :, region_a + 1] = candidate[row, :, region_b + 1]
                swapped[row, :, region_b + 1] = candidate[row, :, region_a + 1]
            candidate = (
                gates[:, 0, None, None] * candidate
                + gates[:, 1, None, None] * write
                + gates[:, 2, None, None] * swapped
            )
        valid = sequence_mask[:, step, None, None]
        table = np.where(valid, candidate, table)
        all_tables.append(table.copy())
    return np.stack(all_tables, axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    dataset = anchor_train.AnchorRegionDataset("test", args)
    try:
        base_model = DecomposedRegionRecurrentMemory()
        anchor_model = AnchorConditionedDecomposedMemory()
        base_params = flax.serialization.msgpack_restore(
            args.base_checkpoint.read_bytes()
        )
        anchor_params = flax.serialization.msgpack_restore(
            args.anchor_checkpoint.read_bytes()
        )

        @jax.jit
        def infer_base(batch):
            return base_model.apply(
                {"params": base_params}, **base._model_inputs(batch), train=False
            )

        @jax.jit
        def infer_anchor(batch):
            return anchor_model.apply(
                {"params": anchor_params},
                **anchor_train._model_inputs(batch),
                train=False,
            )

        batches = []
        base_outputs = []
        anchor_outputs = []
        rows = dataset.rows
        for start in range(0, len(rows), args.eval_batch_size):
            indices = rows[start : start + args.eval_batch_size]
            valid = len(indices)
            if valid < args.eval_batch_size:
                indices = np.pad(
                    indices, (0, args.eval_batch_size - valid), mode="edge"
                )
            batch = dataset.batch(indices)
            base_outputs.append(
                {
                    key: np.asarray(value)[:valid]
                    for key, value in jax.device_get(infer_base(batch)).items()
                    if key != "all_memories"
                }
            )
            anchor_outputs.append(
                {
                    key: np.asarray(value)[:valid]
                    for key, value in jax.device_get(infer_anchor(batch)).items()
                    if key != "all_memories"
                }
            )
            batches.append({key: value[:valid] for key, value in batch.items()})
        batch = {
            key: np.concatenate([item[key] for item in batches]) for key in batches[0]
        }
        base_output = {
            key: np.concatenate([item[key] for item in base_outputs])
            for key in base_outputs[0]
        }
        anchor_output = {
            key: np.concatenate([item[key] for item in anchor_outputs])
            for key in anchor_outputs[0]
        }

        variants = {
            "base_all": (
                base_output["event_type_logits"],
                base_output["write_entity_logits"],
                base_output["write_region_logits"],
                base_output["swap_pair_logits"],
                None,
                None,
            ),
            "anchor_all": (
                anchor_output["event_type_logits"],
                anchor_output["write_entity_logits"],
                anchor_output["write_region_logits"],
                anchor_output["swap_pair_logits"],
                None,
                None,
            ),
            "base_gate_entity_anchor_payload": (
                base_output["event_type_logits"],
                base_output["write_entity_logits"],
                anchor_output["write_region_logits"],
                anchor_output["swap_pair_logits"],
                None,
                None,
            ),
            "oracle_gate_base_entity_anchor_payload": (
                base_output["event_type_logits"],
                base_output["write_entity_logits"],
                anchor_output["write_region_logits"],
                anchor_output["swap_pair_logits"],
                batch["event_type"],
                None,
            ),
            "oracle_gate_entity_anchor_payload": (
                base_output["event_type_logits"],
                base_output["write_entity_logits"],
                anchor_output["write_region_logits"],
                anchor_output["swap_pair_logits"],
                batch["event_type"],
                batch["write_entity"],
            ),
        }
        results = {}
        for name, (
            type_logits,
            entity_logits,
            region_logits,
            pair_logits,
            oracle_type,
            oracle_entity,
        ) in variants.items():
            output = {
                "all_tables": _rollout(
                    type_logits,
                    entity_logits,
                    region_logits,
                    pair_logits,
                    batch["sequence_mask"],
                    temperature=args.gate_temperature,
                    oracle_type=oracle_type,
                    oracle_entity=oracle_entity,
                ),
                "event_type_logits": (
                    type_logits
                    if oracle_type is None
                    else np.eye(3, dtype=np.float32)[oracle_type]
                ),
                "write_entity_logits": (
                    entity_logits
                    if oracle_entity is None
                    else np.eye(7, dtype=np.float32)[oracle_entity]
                ),
                "write_region_logits": region_logits,
                "swap_pair_logits": pair_logits,
            }
            results[name] = base._summary(output, batch)
        payload = {
            "schema_version": 1,
            "experiment": "anchor_conditioned_hybrid_rollout_ablation",
            "split": "fixed_test",
            "variants": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()

