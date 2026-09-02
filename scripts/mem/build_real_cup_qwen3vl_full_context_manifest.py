#!/usr/bin/env python3
# ruff: noqa: E402
"""Build 36-frame uncut full-context real-cup Qwen3-VL supervision."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openpi.tasks.shellgame.real_cup_qwen3vl_sft_contract import SCREEN_CUPS
from openpi.tasks.shellgame.real_cup_qwen3vl_sft_contract import compact_cup_target
from openpi.tasks.shellgame.real_cup_qwen3vl_sft_contract import compact_local_target
from scripts.mem.export_cup_replay_buffer_episodes import ZarrV2Array
from scripts.mem.export_cup_replay_buffer_episodes import episode_ends
from scripts.mem.export_cup_replay_buffer_episodes import to_uint8

DEFAULT_BUFFER = ROOT / "data/cup_replay_buffer/replay_buffer.zarr"
DEFAULT_LABELS = ROOT / "data/cup_replay_buffer/labels.jsonl"
DEFAULT_SPLIT = ROOT / "artifacts/real_cup_qwen3vl_gt_sft_v1_260826/summary.json"
DEFAULT_OUTPUT = ROOT / "artifacts/real_cup_qwen3vl_full_context36_sft_v1_260826"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", type=Path, default=DEFAULT_BUFFER)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split-summary", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-frames", type=int, default=36)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _screen_cup(index: int) -> str:
    if not 0 <= int(index) < len(SCREEN_CUPS):
        raise ValueError(f"Cup index outside [0,2]: {index}")
    return SCREEN_CUPS[int(index)]


def _map_times(times: np.ndarray, label_length: int, raw_length: int) -> np.ndarray:
    return np.rint(np.asarray(times) * (raw_length - 1) / (label_length - 1)).astype(np.int64)


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "episodes": len({int(row["episode_id"]) for row in rows}),
        "sample_types": dict(sorted(Counter(row["sample_type"] for row in rows).items())),
    }


def main() -> None:
    args = parse_args()
    if args.num_frames != 36:
        raise ValueError("The full-context-first probe fixes input to 36 frames")
    output = args.output_dir.resolve()
    train_path, val_path = output / "train.jsonl", output / "val.jsonl"
    if not args.overwrite and any(path.exists() for path in (train_path, val_path)):
        raise FileExistsError(f"Output manifests already exist under {output}; pass --overwrite")
    clips = output / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    split = json.loads(args.split_summary.read_text(encoding="utf-8"))
    val_ids = {int(value) for value in split["validation_episode_ids"]}
    ends = episode_ends(args.buffer)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    if len(labels) != len(ends) or [int(row["episode_id"]) for row in labels] != list(range(len(ends))):
        raise ValueError("Label/Zarr episodes are not a contiguous one-to-one mapping")
    camera = ZarrV2Array(args.buffer / "data/camera0_rgb")
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    mappings = []

    for label in labels:
        episode = int(label["episode_id"])
        raw_start, raw_end = int(starts[episode]), int(ends[episode])
        raw_length = raw_end - raw_start
        # One uniform, chronological clip from reveal through the decision
        # point. No labeled event boundary or post-decision action is exposed.
        label_times = np.linspace(0, int(label["n_observe_frames"]) - 1, args.num_frames)
        raw_indices = _map_times(label_times, int(label["n_frames"]), raw_length)
        frames = to_uint8(camera.take_first_axis(raw_start + raw_indices))
        clip_path = clips / f"episode_{episode:03d}_full36.npz"
        np.savez_compressed(clip_path, frames=frames, raw_frame_indices=raw_indices)
        destination = val_rows if episode in val_ids else train_rows
        common = {
            "schema_version": 1,
            "source": "real_cup",
            "episode_id": episode,
            "clip_path": str(clip_path.resolve()),
        }
        destination.append(
            {
                **common,
                "sample_type": "full_initial",
                "target": compact_cup_target("initial_cup", _screen_cup(int(label["initial_cup"]))),
            }
        )
        for event_index, pair in enumerate(label["moves"]):
            destination.append(
                {
                    **common,
                    "sample_type": "full_swap",
                    "event_index": event_index,
                    "pair_indices": sorted(int(value) for value in pair),
                    "target": compact_local_target([_screen_cup(value) for value in sorted(pair)]),
                }
            )
        destination.append(
            {
                **common,
                "sample_type": "full_final",
                "target": compact_cup_target("final_cup", _screen_cup(int(label["final_cup"]))),
            }
        )
        mappings.append(
            {
                "episode_id": episode,
                "raw_frames": raw_length,
                "raw_observe_end": int(raw_indices[-1]),
                "raw_frame_indices": raw_indices.tolist(),
            }
        )
        if episode % 10 == 0 or episode == len(labels) - 1:
            print(f"materialized full context {episode + 1}/{len(labels)}", flush=True)

    _write(train_path, train_rows)
    _write(val_path, val_rows)
    summary = {
        "schema_version": 1,
        "buffer": str(args.buffer.resolve()),
        "labels": str(args.labels.resolve()),
        "split_summary": str(args.split_summary.resolve()),
        "num_input_frames": args.num_frames,
        "input_scope": "uniform reveal-to-decision full observation; excludes post-decision action",
        "uses_event_boundaries_as_input": False,
        "teacher_forces_previous_events": False,
        "validation_episode_ids": sorted(val_ids),
        "train": _counts(train_rows),
        "val": _counts(val_rows),
        "mapping_examples": mappings[:10],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
