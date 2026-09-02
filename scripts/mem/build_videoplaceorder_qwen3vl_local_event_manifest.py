#!/usr/bin/env python3
"""Build local place and target-relevant swap manifests for VideoPlaceOrder."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme import four_task_temporal_contract as temporal_contract  # noqa: E402
from openpi.tasks.robomme.videoplaceorder.qwen3vl_local_event_contract import compact_negative_response  # noqa: E402
from openpi.tasks.robomme.videoplaceorder.qwen3vl_local_event_contract import compact_place_response  # noqa: E402
from openpi.tasks.robomme.videoplaceorder.qwen3vl_local_event_contract import compact_swap_response  # noqa: E402
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import cell_from_yx  # noqa: E402
from scripts.mem.build_videoplaceorder_qwen3vl_sft_manifest import _episode_metadata  # noqa: E402

DEFAULT_H5 = _ROOT / "data/robomme_extracted/record_dataset_VideoPlaceOrder.h5"
DEFAULT_SPLITS = _ROOT / "artifacts/robomme_four_task_pilot_seed260826/episode_splits.json"
DEFAULT_OUTPUT = _ROOT / "artifacts/videoplaceorder_qwen3vl_local_events_seed260826"
FRAME_COUNT = temporal_contract.TEACHER_FRAME_COUNT
_POSITIVE_OFFSETS = ((-14, 1), (-10, 3), (-18, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _linspace(start: int, stop: int, upper: int) -> list[int]:
    start = max(0, min(start, upper - 1))
    stop = max(start, min(stop, upper - 1))
    return np.linspace(start, stop, FRAME_COUNT).round().astype(int).tolist()


def _static_starts(episode: h5py.Group, demo_end: int) -> list[int]:
    result = []
    for index in range(demo_end):
        info = episode[f"timestep_{index}/info"]
        if not bool(info["is_subgoal_boundary"][()]):
            continue
        raw = info["simple_subgoal"][()]
        if isinstance(raw, np.ndarray):
            raw = raw.reshape(-1)[0]
        text = raw.decode() if isinstance(raw, bytes | np.bytes_) else str(raw)
        if text == "static":
            result.append(index)
    return result


def _rows(item: dict[str, Any], episode: h5py.Group, h5_path: Path) -> tuple[list[dict[str, Any]], bool]:
    common = {
        "schema_version": 2,
        "source": "videoplaceorder_local_event",
        "episode_index": int(item["episode_index"]),
        "episode_name": str(item["episode_name"]),
        "h5_path": str(h5_path.resolve()),
        "difficulty": str(item["difficulty"]),
        "target_color": str(item["target_color"]),
        "queried_ordinal": int(item["ordinal"]),
        "num_demonstrated_targets": int(item["num_demonstrated_targets"]),
        "demo_end": int(item["demo_end"]),
    }
    rows = []
    demonstrated_cell_yx = cell_from_yx(*item["demonstrated_xy"])
    final_cell_yx = cell_from_yx(*item["target_xy"])
    for ordinal, drop in enumerate(item["drops"], start=1):
        anchor = int(drop["end"]) - 1
        target_cell = cell_from_yx(*drop["target_xy"])
        for variant, (before, after) in enumerate(_POSITIVE_OFFSETS):
            rows.append(
                {
                    **common,
                    "sample_type": "place_complete",
                    "event_index": ordinal - 1,
                    "variant": variant,
                    "frame_indices": _linspace(
                        anchor + before, anchor + after, int(item["demo_end"])
                    ),
                    "target": compact_place_response(target_cell),
                }
            )
        rows.append(
            {
                **common,
                "sample_type": "incomplete_place",
                "event_index": ordinal - 1,
                "variant": 0,
                "frame_indices": _linspace(
                    max(int(drop["index"]), anchor - FRAME_COUNT),
                    max(int(drop["index"]), anchor - 3),
                    int(item["demo_end"]),
                ),
                "target": compact_negative_response("incomplete_event"),
            }
        )

    static_starts = _static_starts(episode, int(item["demo_end"]))
    if static_starts:
        first_static = min(static_starts)
        rows.append(
            {
                **common,
                "sample_type": "no_completed_event",
                "event_index": -1,
                "variant": 0,
                "frame_indices": _linspace(
                    first_static, min(first_static + 20, int(item["demo_end"]) - 1), int(item["demo_end"])
                ),
                "target": compact_negative_response("no_completed_event"),
            }
        )

    moved_distance = float(
        np.linalg.norm(np.asarray(item["target_xy"]) - np.asarray(item["demonstrated_xy"]))
    )
    target_relevant_swap = (
        item["difficulty"] == "hard"
        and demonstrated_cell_yx != final_cell_yx
        and moved_distance > 16.0
        and len(static_starts) >= 2
    )
    if target_relevant_swap:
        swap_start = max(static_starts)
        swap_end = int(item["demo_end"]) - 1
        pair = (demonstrated_cell_yx, final_cell_yx)
        for variant, shift in enumerate((0, -3, 3)):
            rows.append(
                {
                    **common,
                    "sample_type": "target_relevant_swap_complete",
                    "event_index": len(item["drops"]),
                    "variant": variant,
                    "frame_indices": _linspace(
                        swap_start + shift, swap_end + shift, int(item["demo_end"])
                    ),
                    "target": compact_swap_response(*pair),
                }
            )
        rows.append(
            {
                **common,
                "sample_type": "incomplete_swap",
                "event_index": len(item["drops"]),
                "variant": 0,
                "frame_indices": _linspace(
                    swap_start, (swap_start + swap_end) // 2, int(item["demo_end"])
                ),
                "target": compact_negative_response("incomplete_event"),
            }
        )
    return rows, target_relevant_swap


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["tasks"]["VideoPlaceOrder"]
    key_map = {
        "train": "train_episode_indices",
        "dev": "dev_episode_indices",
        "test": "test_episode_indices",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    with h5py.File(args.h5, "r") as source:
        metadata = {
            index: _episode_metadata(source[f"episode_{index}"], index) for index in range(100)
        }
        for split_name, key in key_map.items():
            rows = []
            swap_episodes = 0
            for episode_index in splits[key]:
                episode_rows, has_swap = _rows(
                    metadata[episode_index], source[f"episode_{episode_index}"], args.h5
                )
                rows.extend(episode_rows)
                swap_episodes += int(has_swap)
            path = args.output_dir / f"{split_name}.jsonl"
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Manifest exists: {path}; pass --overwrite")
            _write(path, rows)
            summaries[split_name] = {
                "episodes": len(splits[key]),
                "samples": len(rows),
                "sample_types": dict(sorted(Counter(row["sample_type"] for row in rows).items())),
                "target_relevant_swap_episodes": swap_episodes,
            }
    total_hard = sum(item["difficulty"] == "hard" for item in metadata.values())
    total_labeled_swaps = sum(value["target_relevant_swap_episodes"] for value in summaries.values())
    summary = {
        "schema_version": 2,
        "contract": "local_place_and_target_relevant_swap",
        "full_demo_final_answer_target": False,
        "future_frame_leakage": False,
        "h5": str(args.h5.resolve()),
        "split": str(args.splits.resolve()),
        "swap_label_boundary": (
            "Only target-relevant hard swaps are labeled in this pilot. Full swap-pair labels "
            "must be recovered from simulator replay or motion tracking before full updater training."
        ),
        "hard_episodes": total_hard,
        "target_relevant_swap_labeled": total_labeled_swaps,
        **summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
