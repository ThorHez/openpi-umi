#!/usr/bin/env python3
"""Build demonstration-only Qwen3-VL SFT manifests for VideoPlaceOrder."""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme.videoplaceorder.qwen3vl_sft_contract import cell_from_xy  # noqa: E402
from openpi.tasks.robomme.videoplaceorder.qwen3vl_sft_contract import compact_response  # noqa: E402

DEFAULT_H5 = _ROOT / "data/robomme_extracted/record_dataset_VideoPlaceOrder.h5"
DEFAULT_OUTPUT = _ROOT / "artifacts/videoplaceorder_qwen3vl_sft_seed260825"
FRAME_COUNT = 12
_COORD_RE = re.compile(r"<(\d+),\s*(\d+)>")
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_COLORS = ("red", "green", "blue")
_DROP_OFFSETS = ((-12, -2), (-18, -4), (-8, -1), (-15, -3), (-10, -1), (-20, -5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--train-variants", type=int, default=6)
    parser.add_argument("--val-variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260825)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _decode(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return value.decode() if isinstance(value, bytes | np.bytes_) else str(value)


def _coord(text: str) -> tuple[int, int] | None:
    match = _COORD_RE.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _color_center(image: np.ndarray, color: str) -> tuple[int, int] | None:
    channels = {"red": 0, "green": 1, "blue": 2}
    channel = channels[color]
    other = [index for index in range(3) if index != channel]
    mask = (
        (image[..., channel] > 160)
        & (image[..., other[0]] < 110)
        & (image[..., other[1]] < 110)
    )
    y, x = np.where(mask)
    if len(x) < 20:
        return None
    return int(np.median(x)), int(np.median(y))


def _episode_metadata(episode: h5py.Group, episode_index: int) -> dict[str, Any]:
    goal = _decode(episode["setup/task_goal"][()][0])
    ordinals = [value for word, value in _ORDINALS.items() if re.search(rf"\b{word}\b", goal)]
    colors = [color for color in _COLORS if re.search(rf"\b{color}\b", goal)]
    if len(ordinals) != 1 or len(colors) != 1:
        raise ValueError(f"episode_{episode_index}: invalid goal {goal!r}, {ordinals=}, {colors=}")
    names = sorted(
        (name for name in episode if name.startswith("timestep_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    boundaries = []
    for index, name in enumerate(names):
        timestep = episode[name]
        if bool(timestep["info/is_subgoal_boundary"][()]):
            boundaries.append(
                {
                    "index": index,
                    "demo": bool(timestep["info/is_video_demo"][()]),
                    "simple": _decode(timestep["info/simple_subgoal"][()]),
                    "grounded": _decode(timestep["info/grounded_subgoal"][()]),
                }
            )
    if not boundaries or not boundaries[0]["demo"]:
        raise ValueError(f"episode_{episode_index}: missing demonstration prefix")
    demo_end = next(item["index"] for item in boundaries if not item["demo"])
    demo_boundaries = [item for item in boundaries if item["demo"]]
    drops = [item for item in demo_boundaries if item["simple"] == "drop the cube onto target"]
    for drop in drops:
        later = [item["index"] for item in boundaries if item["index"] > drop["index"]]
        drop["end"] = min(later)
        target_xy = _coord(drop["grounded"])
        if target_xy is None:
            later_pick = next(
                (
                    item
                    for item in demo_boundaries
                    if item["index"] >= drop["end"] and item["simple"] == "pick up the cube"
                ),
                None,
            )
            target_xy = _coord(later_pick["grounded"]) if later_pick is not None else None
        if target_xy is None:
            target_xy = _color_center(
                np.asarray(
                    episode[f"timestep_{int(drop['end']) - 2}/obs/front_rgb"][()],
                    dtype=np.uint8,
                ),
                colors[0],
            )
        if target_xy is None:
            raise ValueError(f"episode_{episode_index}: cannot recover target at {drop['index']}")
        drop["target_xy"] = target_xy
    execution = next(
        item
        for item in boundaries
        if not item["demo"] and item["simple"] == "place the cube onto the correct target"
    )
    target_xy = _coord(execution["grounded"])
    if target_xy is None:
        raise ValueError(f"episode_{episode_index}: missing execution target coordinate")
    if not 2 <= len(drops) <= 4:
        raise ValueError(
            f"episode_{episode_index}: invalid goal/drops {goal!r}, {ordinals=}, {colors=}, drops={len(drops)}"
        )
    ordinal = ordinals[0]
    if ordinal > len(drops):
        raise ValueError(f"episode_{episode_index}: requested drop {ordinal} of {len(drops)}")
    static = [item["index"] for item in demo_boundaries if item["simple"] == "static"]
    if not static:
        raise ValueError(f"episode_{episode_index}: missing post-demonstration static segment")
    demonstrated_xy = drops[ordinal - 1]["target_xy"]
    candidate_cells = [cell_from_xy(*drop["target_xy"]) for drop in drops]
    return {
        "episode_index": episode_index,
        "episode_name": f"episode_{episode_index}",
        "difficulty": _decode(episode["setup/difficulty"][()]),
        "goal": goal,
        "target_color": colors[0],
        "ordinal": ordinal,
        "num_demonstrated_targets": len(drops),
        "demo_end": demo_end,
        "static_start": min(static),
        "drops": drops,
        "target_xy": target_xy,
        "target_cell": cell_from_xy(*target_xy),
        "demonstrated_xy": demonstrated_xy,
        "demonstrated_cell": cell_from_xy(*demonstrated_xy) if demonstrated_xy is not None else None,
        "candidate_target_cells": candidate_cells,
        "candidate_target_xy": [list(drop["target_xy"]) for drop in drops],
    }


def _stratified_split(metadata: list[dict[str, Any]], val_count: int, seed: int) -> tuple[set[int], set[int]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for item in metadata:
        strata[(item["difficulty"], item["target_color"], item["ordinal"], item["num_demonstrated_targets"])].append(
            item["episode_index"]
        )
    for values in strata.values():
        rng.shuffle(values)
    exact = {key: len(values) * val_count / len(metadata) for key, values in strata.items()}
    allocation = {key: int(np.floor(value)) for key, value in exact.items()}
    remaining = val_count - sum(allocation.values())
    order = sorted(strata, key=lambda key: exact[key] - allocation[key], reverse=True)
    for key in order[:remaining]:
        allocation[key] += 1
    val = {episode for key, values in strata.items() for episode in values[: allocation[key]]}
    train = {item["episode_index"] for item in metadata} - val
    return train, val


def _full_demo_frames(item: dict[str, Any], variant: int) -> list[int]:
    before, after = _DROP_OFFSETS[variant % len(_DROP_OFFSETS)]
    drop_frames = [
        max(int(drop["index"]), int(drop["end"]) + offset)
        for drop in item["drops"]
        for offset in (before, after)
    ]
    filler_count = FRAME_COUNT - len(drop_frames)
    filler = (
        np.linspace(int(item["static_start"]), int(item["demo_end"]) - 1, filler_count)
        .round()
        .astype(int)
        .tolist()
    )
    frames = sorted(drop_frames + filler)
    if len(frames) != FRAME_COUNT or max(frames) >= int(item["demo_end"]):
        raise AssertionError(f"Invalid demonstration frames: {frames}")
    return frames


def _negative_frames(item: dict[str, Any], sample_type: str) -> list[int]:
    requested = item["drops"][int(item["ordinal"]) - 1]
    if sample_type == "truncated_demo":
        stop = max(11, int(requested["index"]) - 1)
        return np.linspace(0, stop, FRAME_COUNT).round().astype(int).tolist()
    start = max(int(requested["index"]), int(requested["end"]) - 24)
    stop = int(requested["end"]) - 1
    return np.linspace(start, stop, FRAME_COUNT).round().astype(int).tolist()


def _rows(item: dict[str, Any], h5_path: Path, variants: int) -> list[dict[str, Any]]:
    common = {
        "schema_version": 1,
        "source": "videoplaceorder",
        "episode_index": int(item["episode_index"]),
        "episode_name": str(item["episode_name"]),
        "h5_path": str(h5_path.resolve()),
        "difficulty": str(item["difficulty"]),
        "target_color": str(item["target_color"]),
        "ordinal": int(item["ordinal"]),
        "num_demonstrated_targets": int(item["num_demonstrated_targets"]),
        "target_cell": str(item["target_cell"]),
        "demonstrated_cell": item["demonstrated_cell"],
        "target_cell_moved": item["demonstrated_cell"] not in (None, item["target_cell"]),
        "candidate_target_cells": item["candidate_target_cells"],
        "candidate_target_xy": item["candidate_target_xy"],
        "target_xy": list(item["target_xy"]),
        "demo_end": int(item["demo_end"]),
    }
    target = compact_response(
        "full_demo",
        target_color=item["target_color"],
        ordinal=item["ordinal"],
        target_cell=item["target_cell"],
    )
    rows = [
        {
            **common,
            "sample_type": "full_demo",
            "variant": variant,
            "frame_indices": _full_demo_frames(item, variant),
            "target": target,
        }
        for variant in range(variants)
    ]
    rows.extend(
        (
            {
                **common,
                "sample_type": sample_type,
                "variant": 0,
                "frame_indices": _negative_frames(item, sample_type),
                "target": compact_response(sample_type),
            }
        )
        for sample_type in ("truncated_demo", "local_only")
    )
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len({row["episode_index"] for row in rows}),
        "samples": len(rows),
        "sample_types": dict(sorted(Counter(row["sample_type"] for row in rows).items())),
        "difficulty": dict(sorted(Counter(row["difficulty"] for row in rows if row["variant"] == 0).items())),
    }


def main() -> None:
    args = parse_args()
    with h5py.File(args.h5, "r") as source:
        metadata = [
            _episode_metadata(source[f"episode_{episode_index}"], episode_index)
            for episode_index in range(100)
        ]
    train_ids, val_ids = _stratified_split(metadata, args.val_episodes, args.seed)
    train_rows, val_rows = [], []
    for item in metadata:
        if item["episode_index"] in train_ids:
            train_rows.extend(_rows(item, args.h5, args.train_variants))
        else:
            val_rows.extend(_rows(item, args.h5, args.val_variants))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": args.output_dir / "train.jsonl", "val": args.output_dir / "val.jsonl"}
    if any(path.exists() for path in paths.values()) and not args.overwrite:
        raise FileExistsError("Manifest exists; pass --overwrite")
    _write(paths["train"], train_rows)
    _write(paths["val"], val_rows)
    moved = [
        {
            "difficulty": item["difficulty"],
            "distance_pixels": float(
                np.linalg.norm(np.asarray(item["target_xy"]) - np.asarray(item["demonstrated_xy"]))
            ),
        }
        for item in metadata
        if item["demonstrated_xy"] is not None
    ]
    summary = {
        "schema_version": 1,
        "h5": str(args.h5.resolve()),
        "seed": args.seed,
        "frame_count": FRAME_COUNT,
        "demonstration_only": True,
        "execution_frames_in_input": False,
        "future_frame_leakage": False,
        "train_episode_ids": sorted(train_ids),
        "val_episode_ids": sorted(val_ids),
        "train": _counts(train_rows),
        "val": _counts(val_rows),
        "target_cell_distribution": dict(sorted(Counter(item["target_cell"] for item in metadata).items())),
        "final_cell_in_candidate_set": int(
            sum(item["target_cell"] in item["candidate_target_cells"] for item in metadata)
        ),
        "demo_coordinate_available": len(moved),
        "target_cell_moved": int(
            sum(
                item["demonstrated_cell"] not in (None, item["target_cell"])
                for item in metadata
            )
        ),
        "demo_to_final_distance_by_difficulty": {
            difficulty: {
                "count": len(values),
                "mean_pixels": float(np.mean(values)) if values else None,
                "max_pixels": float(np.max(values)) if values else None,
            }
            for difficulty in ("easy", "medium", "hard")
            for values in [[entry["distance_pixels"] for entry in moved if entry["difficulty"] == difficulty]]
        },
        "copied_video_bytes": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
