#!/usr/bin/env python3
"""Build a task-balanced Qwen mixture for the four-task RoboMME pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULTS = {
    "videounmask": _ROOT / "artifacts/videounmask_qwen3vl_variable_demo_seed260826",
    "videounmaskswap": _ROOT / "artifacts/videounmaskswap_qwen3vl_local_events_seed260826",
    "videoplaceorder": _ROOT / "artifacts/videoplaceorder_qwen3vl_local_events_seed260826",
    "pickxtimes": _ROOT / "artifacts/pickxtimes_qwen3vl_local_events_seed260826",
}
_WEIGHTS = {
    "videounmask": 0.15,
    "videounmaskswap": 0.25,
    "videoplaceorder": 0.30,
    "pickxtimes": 0.20,
    "shellgame": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in _DEFAULTS.items():
        parser.add_argument(f"--{name}-dir", type=Path, default=path)
    parser.add_argument(
        "--shellgame-train",
        type=Path,
        default=_ROOT / "artifacts/shellgame_qwen3vl_gt_event_sft_v1/train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "artifacts/robomme_four_task_qwen_mixture_seed260826",
    )
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def _sample(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= len(rows):
        return rng.sample(rows, count)
    return [rng.choice(rows) for _ in range(count)]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "sample_types": dict(sorted(Counter(row.get("sample_type", "unknown") for row in rows).items())),
    }


def main() -> None:
    args = parse_args()
    task_dirs = {name: getattr(args, f"{name}_dir") for name in _DEFAULTS}
    train_sources = {name: _read(path / "train.jsonl") for name, path in task_dirs.items()}
    train_sources["shellgame"] = [
        {**row, "source": "shellgame"} for row in _read(args.shellgame_train)
    ]
    dev_sources = {name: _read(path / "dev.jsonl") for name, path in task_dirs.items()}

    rng = random.Random(args.seed)
    allocations = {
        name: int(round(args.train_samples * weight)) for name, weight in _WEIGHTS.items()
    }
    allocations["videoplaceorder"] += args.train_samples - sum(allocations.values())
    train = [
        row
        for name, count in allocations.items()
        for row in _sample(train_sources[name], count, rng)
    ]
    rng.shuffle(train)
    dev = [row for rows in dev_sources.values() for row in rows]
    rng.shuffle(dev)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": args.output_dir / "train.jsonl", "dev": args.output_dir / "dev.jsonl"}
    if any(path.exists() for path in paths.values()) and not args.overwrite:
        raise FileExistsError(f"Mixture exists in {args.output_dir}; pass --overwrite")
    _write(paths["train"], train)
    _write(paths["dev"], dev)
    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "weights": _WEIGHTS,
        "allocations": allocations,
        "train": _summary(train),
        "dev": _summary(dev),
        "test_included": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
