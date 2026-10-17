#!/usr/bin/env python3
"""Oversample real-cup sequence rows to balance local and sequence SFT tasks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-repeat", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequence_repeat < 1:
        raise ValueError("--sequence-repeat must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    balanced = []
    for row in rows:
        repeats = args.sequence_repeat if row["sample_type"] == "sequence" else 1
        for repeat_index in range(repeats):
            item = dict(row)
            item["repeat_index"] = repeat_index
            balanced.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in balanced:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "input_samples": len(rows),
                "output_samples": len(balanced),
                "sample_types": dict(sorted(Counter(row["sample_type"] for row in balanced).items())),
                "sequence_repeat": args.sequence_repeat,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
