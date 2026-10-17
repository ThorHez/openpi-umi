#!/usr/bin/env python3
"""Build the unified-contract four-task RoboMME Qwen mixture."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme import four_task_temporal_contract as temporal_contract  # noqa: E402

_UNIFIED = _ROOT / "artifacts/robomme_qwen_unified_events_seed260826"
_SOURCES = {
    "videounmask_variable_demo": 0.15,
    "videounmaskswap_local_event": 0.25,
    "videoplaceorder_local_event": 0.30,
    "pickxtimes_local_event": 0.20,
    "shellgame_unified_replay": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=_UNIFIED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "artifacts/robomme_four_task_qwen_unified_mixture_seed260826",
    )
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument(
        "--event-temperature",
        type=float,
        default=0.5,
        help="Within-task event sampling: 1=natural frequency, 0=uniform events.",
    )
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        temporal_contract.validate_teacher_frame_indices(row["frame_indices"])
    return rows


def _sample(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= len(rows):
        return rng.sample(rows, count)
    return [rng.choice(rows) for _ in range(count)]


def _event_temperature_sample(
    rows: list[dict[str, Any]], count: int, temperature: float, rng: random.Random
) -> list[dict[str, Any]]:
    if not 0.0 <= temperature <= 1.0:
        raise ValueError(f"event temperature must be in [0,1], got {temperature}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        event = str(json.loads(row["target"])["event"])
        groups.setdefault(event, []).append(row)
    weights = {event: len(group) ** temperature for event, group in groups.items()}
    exact = {event: count * weight / sum(weights.values()) for event, weight in weights.items()}
    allocation = {event: int(value) for event, value in exact.items()}
    for event in sorted(groups, key=lambda key: exact[key] - allocation[key], reverse=True):
        if sum(allocation.values()) == count:
            break
        allocation[event] += 1
    sampled = [
        row
        for event, event_count in allocation.items()
        for row in _sample(groups[event], event_count, rng)
    ]
    rng.shuffle(sampled)
    return sampled


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    allocations = {source: round(args.train_samples * weight) for source, weight in _SOURCES.items()}
    allocations["videoplaceorder_local_event"] += args.train_samples - sum(allocations.values())
    train = []
    for source, count in allocations.items():
        train.extend(
            _event_temperature_sample(
                _read(args.input_dir / source / "train.jsonl"),
                count,
                args.event_temperature,
                rng,
            )
        )
    rng.shuffle(train)
    evaluation = {}
    for split in ("dev", "test"):
        rows = []
        for source in _SOURCES:
            path = args.input_dir / source / f"{split}.jsonl"
            if path.exists():
                rows.extend(_read(path))
        rng.shuffle(rows)
        evaluation[split] = rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.output_dir / "train.jsonl",
        "dev": args.output_dir / "dev.jsonl",
        "test": args.output_dir / "test.jsonl",
    }
    if any(path.exists() for path in paths.values()) and not args.overwrite:
        raise FileExistsError(f"Mixture exists in {args.output_dir}; pass --overwrite")
    _write(paths["train"], train)
    for split, rows in evaluation.items():
        _write(paths[split], rows)
    summary = {
        "schema_version": 3,
        "contract": "unified_causal_event_v1",
        "teacher_frame_count": temporal_contract.TEACHER_FRAME_COUNT,
        "seed": args.seed,
        "weights": _SOURCES,
        "event_temperature": args.event_temperature,
        "allocations": allocations,
        "train_samples": len(train),
        "dev_samples": len(evaluation["dev"]),
        "test_samples": len(evaluation["test"]),
        "train_sources": dict(sorted(Counter(row["source"] for row in train).items())),
        "train_events": dict(
            sorted(Counter(json.loads(row["target"])["event"] for row in train).items())
        ),
        "explicit_task_id_in_prompt": False,
        "shared_output_head": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
