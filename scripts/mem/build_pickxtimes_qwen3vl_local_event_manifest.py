#!/usr/bin/env python3
"""Build local-only PickXtimes Qwen3-VL event manifests."""

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

from openpi.tasks.robomme import four_task_temporal_contract as temporal_contract  # noqa: E402
from openpi.tasks.robomme.pickxtimes.qwen3vl_local_event_contract import compact_response  # noqa: E402

DEFAULT_H5 = _ROOT / "data/robomme_extracted/record_dataset_PickXtimes.h5"
DEFAULT_LABELS = _ROOT / "data/robomme_extracted/pickxtimes_event_labels_w10_v3_press5_gripper.json"
DEFAULT_SPLIT = _ROOT / "data/robomme_extracted/pickxtimes_split_seed260827_train70_dev15_test15.json"
DEFAULT_OUTPUT = _ROOT / "artifacts/pickxtimes_qwen3vl_local_events_seed260826"
FRAME_COUNT = temporal_contract.TEACHER_FRAME_COUNT
# Include a stable post-transition result. The former anchor+0..4 windows
# differed from incomplete clips by only about 0.3 seconds at 10 Hz.
_POSITIVE_OFFSETS = ((-7, 8), (-5, 10), (-9, 6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _linspace(start: int, stop: int, num_steps: int) -> list[int]:
    start = max(0, min(start, num_steps - 1))
    stop = max(start, min(stop, num_steps - 1))
    return np.linspace(start, stop, FRAME_COUNT).round().astype(int).tolist()


def _event_rows(item: dict[str, Any], h5_path: Path) -> list[dict[str, Any]]:
    common = {
        "schema_version": 2,
        "source": "pickxtimes_local_event",
        "episode_index": int(item["episode_index"]),
        "episode_name": str(item["episode_name"]),
        "h5_path": str(h5_path.resolve()),
        "difficulty": str(item["difficulty"]),
        "target_color": str(item["target_color"]),
        "required_count": int(item["required_count"]),
        "num_steps": int(item["num_steps"]),
    }
    rows = []
    for event_index, event in enumerate(item["events"]):
        event_type = str(event["event_type"])
        anchor = int(event["anchor"])
        for variant, (before, after) in enumerate(_POSITIVE_OFFSETS):
            rows.append(
                {
                    **common,
                    "sample_type": "completed_event",
                    "event_index": event_index,
                    "variant": variant,
                    "frame_indices": _linspace(
                        anchor + before, anchor + after, int(item["num_steps"])
                    ),
                    "target": compact_response(event_type, target_color=item["target_color"]),
                }
            )
        incomplete_stop = max(int(event["start"]), anchor - 4)
        rows.append(
            {
                **common,
                "sample_type": "incomplete_event",
                "event_index": event_index,
                "variant": 0,
                "frame_indices": _linspace(
                    max(int(event["start"]), anchor - FRAME_COUNT),
                    incomplete_stop,
                    int(item["num_steps"]),
                ),
                "target": compact_response("incomplete_event"),
            }
        )
    # Every first event starts at frame zero, so the former construction made
    # twelve copies of frame zero. Use distinct initial frames and a second
    # causal hold window after the already-completed press event.
    no_event_ranges = (
        (0, min(FRAME_COUNT - 1, int(item["events"][0]["anchor"]) - 4)),
        (
            min(int(item["num_steps"]) - 1, int(item["completion_rise"]) + 5),
            min(int(item["num_steps"]) - 1, int(item["completion_rise"]) + 16),
        ),
    )
    for variant, (start, stop) in enumerate(no_event_ranges):
        rows.append(
            {
                **common,
                "sample_type": "no_completed_event",
                "event_index": -1,
                "variant": variant,
                "frame_indices": _linspace(start, stop, int(item["num_steps"])),
                "target": compact_response("no_completed_event"),
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len({row["episode_index"] for row in rows}),
        "samples": len(rows),
        "sample_types": dict(sorted(Counter(row["sample_type"] for row in rows).items())),
        "events": dict(
            sorted(
                Counter(json.loads(row["target"])["event"] for row in rows).items()
            )
        ),
    }


def main() -> None:
    args = parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    by_index = {int(item["episode_index"]): item for item in labels["episodes"]}
    ids = {
        "train": [int(value) for value in split["train_episode_indices"]],
        "dev": [int(value) for value in split.get("dev_episode_indices", split["val_episode_indices"])],
        "test": [int(value) for value in split["test_episode_indices"]],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split_name, episode_ids in ids.items():
        path = args.output_dir / f"{split_name}.jsonl"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Manifest exists: {path}; pass --overwrite")
        rows = [row for episode_id in episode_ids for row in _event_rows(by_index[episode_id], args.h5)]
        _write(path, rows)
        summaries[split_name] = _summary(rows)
    summary = {
        "schema_version": 2,
        "contract": "local_event_only",
        "cumulative_count_in_qwen_target": False,
        "window_recipe": {
            "positive_offsets": _POSITIVE_OFFSETS,
            "incomplete_stop": "anchor_minus_4",
            "no_event": "initial_distinct_frames_and_post_completion_hold",
        },
        "frame_count": FRAME_COUNT,
        "future_frame_leakage": False,
        "h5": str(args.h5.resolve()),
        "labels": str(args.labels.resolve()),
        "split": str(args.split.resolve()),
        **summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
