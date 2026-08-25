# ruff: noqa: E402
"""Build a disk-light Qwen3-VL SFT manifest from simulator GT events.

No images are duplicated.  Each JSONL row stores a raw NPZ path, ten causal
frame indices, a task prompt type, and one compact camera-relative JSON target.
The validation episodes exactly match the semantic-memory held-out split.
"""

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

from openpi.tasks.shellgame.qwen3vl_sft_contract import compact_response
from openpi.tasks.shellgame.qwenvl_event_adapter import screen_cup_from_world_slot
from openpi.tasks.shellgame.qwenvl_event_adapter import screen_pair_from_world_pair

DEFAULT_RAW_ROOT = Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_absolute_eef_phase_instruction_dataset"
)
DEFAULT_OUTPUT_DIR = _ROOT / "artifacts/shellgame_qwen3vl_gt_event_sft_v1"
EXPECTED_PHASES = {
    "reveal": [0, 9],
    "cover": [10, 19],
    "swap_0": [20, 29],
    "swap_1": [30, 39],
    "swap_2": [40, 49],
    "settle": [50, 59],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _episodes(raw_root: Path) -> list[int]:
    result = []
    for path in raw_root.glob("episode_*"):
        try:
            episode = int(path.name.removeprefix("episode_"))
        except ValueError:
            continue
        if (path / "metadata.json").is_file() and (path / "vla_trajectory.npz").is_file():
            result.append(episode)
    if len(result) < 2:
        raise ValueError(f"Need at least two complete episodes under {raw_root}")
    return sorted(result)


def _make_row(
    *,
    raw_root: Path,
    episode: int,
    sample_type: str,
    frame_indices: list[int],
    target: str,
    event_index: int | None = None,
) -> dict[str, Any]:
    if len(frame_indices) != 10 or frame_indices != list(range(frame_indices[0], frame_indices[0] + 10)):
        raise ValueError(f"Every SFT clip must contain ten consecutive frames: {frame_indices}")
    if min(frame_indices) < 0 or max(frame_indices) >= 60:
        raise ValueError(f"SFT clips may only use the observation prefix 0..59: {frame_indices}")
    return {
        "schema_version": 1,
        "episode_index": episode,
        "sample_type": sample_type,
        "event_index": event_index,
        "trajectory_path": str((raw_root / f"episode_{episode:06d}" / "vla_trajectory.npz").resolve()),
        "frame_indices": frame_indices,
        "target": target,
    }


def _episode_rows(raw_root: Path, episode: int) -> list[dict[str, Any]]:
    episode_dir = raw_root / f"episode_{episode:06d}"
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    for name, expected in EXPECTED_PHASES.items():
        if metadata.get("phase_ranges", {}).get(name) != expected:
            raise ValueError(f"episode {episode}: phase {name} is not {expected}")
    swaps = metadata.get("swaps")
    if not isinstance(swaps, list) or len(swaps) != 3:
        raise ValueError(f"episode {episode}: expected exactly three GT swaps")

    rows = [
        _make_row(
            raw_root=raw_root,
            episode=episode,
            sample_type="reveal",
            frame_indices=list(range(10)),
            target=compact_response("reveal", screen_cup_from_world_slot(str(metadata["initial_ball_cup"]))),
        )
    ]
    for stage, world_pair in enumerate(swaps):
        start = 20 + 10 * stage
        rows.append(
            _make_row(
                raw_root=raw_root,
                episode=episode,
                sample_type="swap",
                event_index=stage,
                frame_indices=list(range(start, start + 10)),
                target=compact_response("swap", screen_pair_from_world_pair(world_pair)),
            )
        )
    rows.append(
        _make_row(
            raw_root=raw_root,
            episode=episode,
            sample_type="no_event",
            frame_indices=list(range(50, 60)),
            target=compact_response("no_event"),
        )
    )
    # A deterministic partial window makes the model reject clips without a
    # complete before/after trajectory.  Varying the boundary across episodes
    # covers all three exchange stages without changing split membership.
    partial_stage = episode % 3
    partial_start = 15 + 10 * partial_stage
    rows.append(
        _make_row(
            raw_root=raw_root,
            episode=episode,
            sample_type="incomplete_event",
            event_index=partial_stage,
            frame_indices=list(range(partial_start, partial_start + 10)),
            target=compact_response("incomplete_event"),
        )
    )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "episodes": len({int(row["episode_index"]) for row in rows}),
        "sample_types": dict(sorted(Counter(str(row["sample_type"]) for row in rows).items())),
        "targets": dict(sorted(Counter(str(row["target"]) for row in rows).items())),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val-ratio must be in (0, 1)")
    episodes = _episodes(args.raw_root)
    shuffled = np.random.default_rng(args.split_seed).permutation(np.asarray(episodes, dtype=np.int64))
    num_val = min(max(1, round(len(episodes) * args.val_ratio)), len(episodes) - 1)
    val_ids = {int(value) for value in shuffled[:num_val]}
    train_ids = set(episodes) - val_ids
    if train_ids & val_ids or train_ids | val_ids != set(episodes):
        raise AssertionError("Episode split is not a disjoint partition")

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for order, episode in enumerate(episodes, start=1):
        rows = _episode_rows(args.raw_root, episode)
        (val_rows if episode in val_ids else train_rows).extend(rows)
        if order % 500 == 0:
            print(f"validated metadata: {order}/{len(episodes)} episodes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    existing = [path for path in (train_path, val_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}; pass --overwrite")
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    summary = {
        "schema_version": 1,
        "raw_root": str(args.raw_root.resolve()),
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "validation_episode_ids": sorted(val_ids),
        "train": _counts(train_rows),
        "val": _counts(val_rows),
        "coordinate_contract": {
            "world_left": "screen_right_cup",
            "world_middle": "screen_middle_cup",
            "world_right": "screen_left_cup",
        },
        "future_frame_leakage": False,
        "copied_video_bytes": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
