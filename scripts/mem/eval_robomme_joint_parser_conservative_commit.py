#!/usr/bin/env python3
"""Evaluate a joint-event visual parser under a calibrated conservative commit."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from openpi.tasks.robomme.causal_visual_operation_parser import CausalVisualOperationParser  # noqa: E402
from scripts.mem import eval_robomme_transition_causal_ablation as replay  # noqa: E402
from scripts.mem import train_robomme_anchor_conditioned_decomposition as anchor_base  # noqa: E402
from scripts.mem import train_robomme_anchor_transition_curriculum as transition  # noqa: E402
from scripts.mem import train_robomme_decomposed_region_distillation as base  # noqa: E402
from scripts.mem import train_robomme_visual_operation_parser_ablation as parser_base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, default=base.DEFAULT_FIXED)
    parser.add_argument("--teacher-dir", type=Path, default=base.DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path, default=anchor_base.DEFAULT_ANCHORS)
    parser.add_argument("--eval-batch-size", type=int, default=3)
    parser.add_argument("--target-hold-fpr", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _categorical_logits(indices: np.ndarray, classes: int) -> np.ndarray:
    return np.where(
        np.eye(classes, dtype=np.bool_)[indices],
        np.float32(20.0),
        np.float32(-20.0),
    )


def _threshold_output(
    output: dict[str, np.ndarray], threshold: float
) -> dict[str, np.ndarray]:
    probabilities = np.asarray(jax.nn.softmax(jnp.asarray(output["event_type_logits"])))
    completion = (1.0 - probabilities[..., 0]) >= threshold
    kind = np.argmax(probabilities[..., 1:], axis=-1) + 1
    event_type = np.where(completion, kind, 0)
    result = dict(output)
    result["event_type_logits"] = _categorical_logits(event_type, 3)
    return result


def _infer_split(
    model: CausalVisualOperationParser,
    params: Any,
    data: parser_base.FixedParserDataset,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    @jax.jit
    def infer(batch):
        return model.apply(
            {"params": params}, **parser_base._model_inputs(batch), train=False  # noqa: SLF001
        )

    outputs: dict[str, list[np.ndarray]] = defaultdict(list)
    batches: list[dict[str, np.ndarray]] = []
    for start in range(0, len(data.rows), batch_size):
        indices = data.rows[start : start + batch_size]
        valid_count = len(indices)
        if valid_count < batch_size:
            indices = np.pad(indices, (0, batch_size - valid_count), mode="edge")
        batch = data.parser_batch(indices)
        output = jax.device_get(infer(batch))
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


def _operation(
    output: dict[str, np.ndarray], batch: dict[str, np.ndarray], threshold: float
) -> dict[str, Any]:
    return parser_base._metrics(_threshold_output(output, threshold), batch)  # noqa: SLF001


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
    rows = [(float(value), _operation(output, batch, float(value))) for value in candidates]

    def score(row):
        metrics = row[1]
        feasible = metrics["hold_false_positive_rate"] <= target_fpr
        return (
            int(feasible),
            metrics["full_update_recall"] if feasible else -metrics["hold_false_positive_rate"],
            metrics["update_exact_type_recall"],
            metrics["update_exact_type_precision"],
        )

    return max(rows, key=score)


def main() -> None:
    args = parse_args()
    datasets = {
        split: parser_base.FixedParserDataset(split, args)
        for split in ("train", "dev", "test")
    }
    try:
        train_data = datasets["train"]
        model = CausalVisualOperationParser(
            max_steps=train_data.max_parser_steps,
            spatial_tokens=train_data.spatial_tokens,
            input_width=train_data.patch_width,
            recurrent_event_state=True,
        )
        initial = train_data.parser_batch(train_data.rows[: args.eval_batch_size])
        template = model.init(
            jax.random.key(0), **parser_base._model_inputs(initial), train=False  # noqa: SLF001
        )["params"]
        params = flax.serialization.from_bytes(
            template, (args.checkpoint_dir / "params.msgpack").read_bytes()
        )
        cached = {
            split: _infer_split(model, params, data, args.eval_batch_size)
            for split, data in datasets.items()
        }
        threshold, dev_operation = _select_threshold(
            *cached["dev"], args.target_hold_fpr
        )
        metrics = {}
        for split, (output, batch) in cached.items():
            thresholded = _threshold_output(output, threshold)
            replayed = replay._replay(  # noqa: SLF001
                thresholded, batch, oracle_event=False, oracle_payload=False
            )
            metrics[split] = {
                "operation": parser_base._metrics(thresholded, batch),  # noqa: SLF001
                "rollout": transition._summary(replayed, batch),  # noqa: SLF001
            }
        result = {
            "schema_version": 1,
            "experiment": "robomme_joint_parser_conservative_commit_eval",
            "checkpoint_dir": str(args.checkpoint_dir),
            "target_hold_fpr": args.target_hold_fpr,
            "selected_threshold": threshold,
            "dev_selection_operation": dev_operation,
            "metrics": metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
