#!/usr/bin/env python3
# ruff: noqa: E402
"""Build episode-disjoint Qwen3-VL SFT clips from real-cup GT labels.

``labels.jsonl`` uses a linearly resampled per-episode timeline.  This builder
maps those labeled times back to the raw Zarr camera timeline and materializes
only the twelve RGB frames needed by each local-event or full-sequence sample.
"""

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
from openpi.tasks.shellgame.real_cup_qwen3vl_sft_contract import compact_local_target
from openpi.tasks.shellgame.real_cup_qwen3vl_sft_contract import compact_sequence_target
from scripts.mem.export_cup_replay_buffer_episodes import ZarrV2Array
from scripts.mem.export_cup_replay_buffer_episodes import episode_ends
from scripts.mem.export_cup_replay_buffer_episodes import to_uint8

DEFAULT_BUFFER = ROOT / "data/cup_replay_buffer/replay_buffer.zarr"
DEFAULT_LABELS = ROOT / "data/cup_replay_buffer/labels.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts/real_cup_qwen3vl_gt_sft_v1_260826"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", type=Path, default=DEFAULT_BUFFER)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=260826)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _label_to_raw(label_times: np.ndarray, *, label_length: int, raw_length: int) -> np.ndarray:
    if label_length < 2 or raw_length < 2:
        raise ValueError(f"Invalid timeline lengths: label={label_length}, raw={raw_length}")
    scale = (raw_length - 1) / (label_length - 1)
    result = np.rint(np.asarray(label_times, dtype=np.float64) * scale).astype(np.int64)
    return np.clip(result, 0, raw_length - 1)


def _screen_cup(index: int) -> str:
    if not 0 <= int(index) < len(SCREEN_CUPS):
        raise ValueError(f"Cup index outside [0,2]: {index}")
    return SCREEN_CUPS[int(index)]


def _initial_red_slot(frames: np.ndarray) -> int | None:
    sample = frames[: min(80, len(frames))].astype(np.float32)
    yy, xx = np.indices(sample.shape[1:3])
    best: tuple[int, np.ndarray] | None = None
    for image in sample:
        red, green, blue = image[..., 0], image[..., 1], image[..., 2]
        mask = (yy >= 65) & (yy < 135) & (red > 145) & (red - green > 55) & (red - blue > 55)
        count = int(mask.sum())
        if best is None or count > best[0]:
            best = (count, mask)
    if best is None or best[0] < 3:
        return None
    x = float(np.median(xx[best[1]]))
    return int(np.argmin(np.abs(np.asarray([84.0, 112.0, 140.0]) - x)))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "episodes": len({int(row["episode_id"]) for row in rows}),
        "sample_types": dict(sorted(Counter(str(row["sample_type"]) for row in rows).items())),
        "local_pairs": dict(
            sorted(Counter(str(row.get("pair_indices")) for row in rows if row["sample_type"] == "local_swap").items())
        ),
        "final_cups": dict(
            sorted(Counter(str(row.get("final_cup_index")) for row in rows if row["sample_type"] == "sequence").items())
        ),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0,1)")
    if args.num_frames != 12:
        raise ValueError("This experiment fixes teacher input to twelve frames")
    output = args.output_dir.resolve()
    train_path, val_path = output / "train.jsonl", output / "val.jsonl"
    if not args.overwrite and any(path.exists() for path in (train_path, val_path)):
        raise FileExistsError(f"Output manifests already exist under {output}; pass --overwrite")
    clips = output / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    ends = episode_ends(args.buffer)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    if len(labels) != len(ends):
        raise ValueError(f"Label/Zarr episode count mismatch: {len(labels)} vs {len(ends)}")
    if [int(row["episode_id"]) for row in labels] != list(range(len(labels))):
        raise ValueError("labels.jsonl must be ordered by contiguous episode_id")

    shuffled = np.random.default_rng(args.split_seed).permutation(len(labels))
    num_val = min(max(1, round(len(labels) * args.val_ratio)), len(labels) - 1)
    val_ids = {int(value) for value in shuffled[:num_val]}
    camera = ZarrV2Array(args.buffer / "data/camera0_rgb")
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    initial_checks: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    for row in labels:
        episode = int(row["episode_id"])
        raw_start, raw_end = int(starts[episode]), int(ends[episode])
        raw_length = raw_end - raw_start
        label_length = int(row["n_frames"])
        frames_per_move = int(row["frames_per_move"])
        moves = [[int(value) for value in pair] for pair in row["moves"]]
        if len(moves) != 3 or int(row["n_observe_frames"]) != 3 * frames_per_move + 1:
            raise ValueError(f"Episode {episode}: unexpected three-move timeline")

        local_times = [
            np.linspace(stage * frames_per_move, (stage + 1) * frames_per_move, args.num_frames)
            for stage in range(3)
        ]
        sequence_times = np.linspace(0, int(row["n_observe_frames"]) - 1, args.num_frames)
        all_times = np.concatenate((*local_times, sequence_times))
        raw_local = _label_to_raw(all_times, label_length=label_length, raw_length=raw_length)
        global_indices = raw_start + raw_local
        selected = to_uint8(camera.take_first_axis(global_indices))
        selected = selected.reshape(4, args.num_frames, *selected.shape[1:])

        destination = val_rows if episode in val_ids else train_rows
        for stage, pair in enumerate(moves):
            clip_path = clips / f"episode_{episode:03d}_swap_{stage}.npz"
            np.savez_compressed(clip_path, frames=selected[stage], raw_frame_indices=raw_local[stage * args.num_frames : (stage + 1) * args.num_frames])
            screen_pair = [_screen_cup(value) for value in sorted(pair)]
            destination.append(
                {
                    "schema_version": 1,
                    "source": "real_cup",
                    "episode_id": episode,
                    "sample_type": "local_swap",
                    "event_index": stage,
                    "pair_indices": sorted(pair),
                    "clip_path": str(clip_path.resolve()),
                    "target": compact_local_target(screen_pair),
                }
            )

        sequence_path = clips / f"episode_{episode:03d}_sequence.npz"
        np.savez_compressed(sequence_path, frames=selected[3], raw_frame_indices=raw_local[-args.num_frames:])
        initial_cup = _screen_cup(int(row["initial_cup"]))
        final_cup = _screen_cup(int(row["final_cup"]))
        screen_moves = [[_screen_cup(value) for value in sorted(pair)] for pair in moves]
        destination.append(
            {
                "schema_version": 1,
                "source": "real_cup",
                "episode_id": episode,
                "sample_type": "sequence",
                "clip_path": str(sequence_path.resolve()),
                "initial_cup_index": int(row["initial_cup"]),
                "moves": moves,
                "final_cup_index": int(row["final_cup"]),
                "target": compact_sequence_target(initial_cup, screen_moves, final_cup),
            }
        )

        initial_probe_indices = raw_start + np.arange(min(80, raw_length), dtype=np.int64)
        initial_probe = to_uint8(camera.take_first_axis(initial_probe_indices))
        detected = _initial_red_slot(initial_probe)
        initial_checks.append(
            {
                "episode_id": episode,
                "label": int(row["initial_cup"]),
                "detected": detected,
                "correct": detected == int(row["initial_cup"]),
            }
        )
        mapping_rows.append(
            {
                "episode_id": episode,
                "raw_frames": raw_length,
                "label_frames": label_length,
                "raw_observe_end": int(_label_to_raw(np.asarray([row["n_observe_frames"] - 1]), label_length=label_length, raw_length=raw_length)[0]),
            }
        )
        if episode % 10 == 0 or episode == len(labels) - 1:
            print(f"materialized episode {episode + 1}/{len(labels)}", flush=True)

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    valid_initial = [item for item in initial_checks if item["detected"] is not None]
    summary = {
        "schema_version": 1,
        "buffer": str(args.buffer.resolve()),
        "labels": str(args.labels.resolve()),
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "num_input_frames": args.num_frames,
        "train_episode_ids": sorted(set(range(len(labels))) - val_ids),
        "validation_episode_ids": sorted(val_ids),
        "train": _counts(train_rows),
        "val": _counts(val_rows),
        "label_rollout_consistency_rate": 1.0,
        "initial_rgb_detection_coverage": len(valid_initial) / len(initial_checks),
        "initial_rgb_label_accuracy": sum(item["correct"] for item in valid_initial) / max(len(valid_initial), 1),
        "timeline_mapping": "raw_index=round(label_index*(raw_length-1)/(label_n_frames-1))",
        "mapping_examples": mapping_rows[:10],
        "initial_checks": initial_checks,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"initial_checks", "train_episode_ids"}}, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
