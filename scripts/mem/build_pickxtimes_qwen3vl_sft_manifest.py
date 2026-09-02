#!/usr/bin/env python3
"""Build causal sparse-prefix Qwen3-VL SFT manifests for RoboMME PickXtimes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme.pickxtimes.qwen3vl_sft_contract import compact_response  # noqa: E402

DEFAULT_H5 = _ROOT / "data/robomme_extracted/record_dataset_PickXtimes.h5"
DEFAULT_LABELS = _ROOT / "data/robomme_extracted/pickxtimes_event_labels_w10_v3_press5_gripper.json"
DEFAULT_SPLIT = _ROOT / "data/robomme_extracted/pickxtimes_split_seed260827_train70_dev15_test15.json"
DEFAULT_OUTPUT = _ROOT / "artifacts/pickxtimes_qwen3vl_sft_seed260827"
FRAME_COUNT = 12
_EVIDENCE_OFFSETS = ((-5, 1), (-3, 3), (-7, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix-variants", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _event_pair(event: dict[str, Any], variant: int, num_steps: int) -> list[int]:
    before, after = _EVIDENCE_OFFSETS[variant % len(_EVIDENCE_OFFSETS)]
    anchor = int(event["anchor"])
    if event["event_type"] == "press_complete":
        after = 0
    return [max(0, min(anchor + offset, num_steps - 1)) for offset in (before, after)]


def _prefix_frames(item: dict[str, Any], event_index: int, variant: int) -> list[int]:
    events = item["events"][:event_index]
    current = events[-1]
    selected = [event for event in events if event["event_type"] == "place_complete"]
    if current["event_type"] != "place_complete":
        selected.append(current)
    evidence = [
        frame
        for event in selected
        for frame in _event_pair(event, variant, int(item["num_steps"]))
    ]
    filler_count = FRAME_COUNT - len(evidence)
    if filler_count < 0:
        raise AssertionError(f"Too much evidence for {item['episode_name']}: {evidence}")
    first_anchor = int(item["events"][0]["anchor"])
    filler_stop = max(0, min(first_anchor - 10, min(evidence) - 1))
    filler = (
        np.linspace(0, filler_stop, filler_count).round().astype(int).tolist()
        if filler_count
        else []
    )
    frames = sorted(filler + evidence)
    if len(frames) != FRAME_COUNT or max(frames) > max(evidence):
        raise AssertionError(f"Invalid causal frames: {frames}")
    return frames


def _local_frames(event: dict[str, Any], num_steps: int) -> list[int]:
    start = max(int(event["start"]), int(event["anchor"]) - 10)
    stop = min(int(event["end"]) - 1, int(event["anchor"]) + 4, num_steps - 1)
    return np.linspace(start, stop, FRAME_COUNT).round().astype(int).tolist()


def _rows(item: dict[str, Any], h5_path: Path, variants: int) -> list[dict[str, Any]]:
    common = {
        "schema_version": 1,
        "source": "pickxtimes",
        "episode_index": int(item["episode_index"]),
        "episode_name": str(item["episode_name"]),
        "h5_path": str(h5_path.resolve()),
        "difficulty": str(item["difficulty"]),
        "target_color": str(item["target_color"]),
        "required_count": int(item["required_count"]),
    }
    rows = []
    for event_index, event in enumerate(item["events"], start=1):
        event_type = str(event["event_type"])
        completed_count = int(event["state_after"]["completed_count"])
        target = compact_response(
            "causal_prefix",
            event=event_type,
            completed_count=completed_count,
            required_count=int(item["required_count"]),
        )
        rows.extend(
            {
                **common,
                "sample_type": "causal_prefix",
                "event_index": event_index,
                "variant": variant,
                "frame_indices": _prefix_frames(item, event_index, variant),
                "target": target,
            }
            for variant in range(variants)
        )
        if event_index >= 2:
            rows.append(
                {
                    **common,
                    "sample_type": "local_only",
                    "event_index": event_index,
                    "variant": 0,
                    "frame_indices": _local_frames(event, int(item["num_steps"])),
                    "target": compact_response("local_only"),
                }
            )
    first_anchor = int(item["events"][0]["anchor"])
    for variant in range(2):
        stop = max(11, first_anchor - 10 - 8 * variant)
        rows.append(
            {
                **common,
                "sample_type": "no_event",
                "event_index": 0,
                "variant": variant,
                "frame_indices": np.linspace(0, stop, FRAME_COUNT).round().astype(int).tolist(),
                "target": compact_response("no_event"),
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len({int(row["episode_index"]) for row in rows}),
        "samples": len(rows),
        "sample_types": dict(sorted(Counter(row["sample_type"] for row in rows).items())),
        "required_count": dict(
            sorted(Counter(str(row["required_count"]) for row in rows if row["variant"] == 0).items())
        ),
    }


def main() -> None:
    args = parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    by_index = {int(item["episode_index"]): item for item in labels["episodes"]}
    ids = {
        "train": [int(value) for value in split["train_episode_indices"]],
        "val": [int(value) for value in split["val_episode_indices"]],
        "test": [int(value) for value in split["test_episode_indices"]],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split_name, episode_ids in ids.items():
        path = args.output_dir / f"{split_name}.jsonl"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Manifest exists: {path}; pass --overwrite")
        rows = [
            row
            for episode_id in episode_ids
            for row in _rows(by_index[episode_id], args.h5, args.prefix_variants)
        ]
        _write(path, rows)
        summaries[split_name] = _summary(rows)
    summary = {
        "schema_version": 1,
        "h5": str(args.h5.resolve()),
        "labels": str(args.labels.resolve()),
        "split": str(args.split.resolve()),
        "frame_count": FRAME_COUNT,
        "future_frame_leakage": False,
        "copied_video_bytes": 0,
        **summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

