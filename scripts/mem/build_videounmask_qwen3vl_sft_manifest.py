#!/usr/bin/env python3
"""Build disk-light, demonstration-only Qwen3-VL SFT manifests for VideoUnmask."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import COLORS  # noqa: E402
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import cell_from_yx  # noqa: E402
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import compact_response  # noqa: E402

DEFAULT_H5 = _ROOT / "data/robomme_extracted/record_dataset_VideoUnmask.h5"
DEFAULT_LABELS = _ROOT / "data/robomme_extracted/videounmask_semantic_labels_seed260823.json"
DEFAULT_OUTPUT = _ROOT / "artifacts/videounmask_qwen3vl_sft_seed260823"
DEFAULT_SHELLGAME = _ROOT / "artifacts/shellgame_qwen3vl_gt_event_sft_v1/train.jsonl"
FRAME_COUNT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shellgame-manifest", type=Path, default=DEFAULT_SHELLGAME)
    parser.add_argument("--paired-variants", type=int, default=6)
    parser.add_argument("--visible-variants", type=int, default=3)
    parser.add_argument("--masked-variants", type=int, default=3)
    parser.add_argument("--replay-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=260825)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _color_centers(image: np.ndarray) -> dict[str, tuple[float, float]]:
    masks = {
        "red": (image[..., 0] > 180) & (image[..., 1] < 70) & (image[..., 2] < 70),
        "green": (image[..., 1] > 180) & (image[..., 0] < 70) & (image[..., 2] < 70),
        "blue": (image[..., 2] > 180) & (image[..., 0] < 70) & (image[..., 1] < 70),
    }
    centers = {}
    for color, mask in masks.items():
        y, x = np.where(mask)
        if len(y) < 20:
            raise ValueError(f"Could not find a reliable {color} cube ({len(y)} pixels)")
        centers[color] = (float(np.median(y)), float(np.median(x)))
    if len({cell_from_yx(y, x) for y, x in centers.values()}) != len(COLORS):
        raise ValueError(f"Color cubes collide in the 8x8 supervision grid: {centers}")
    return centers


def _indices(rng: np.random.Generator, low: int, high: int, count: int) -> list[int]:
    return sorted(int(value) for value in rng.choice(np.arange(low, high + 1), count, replace=False))


def _video_rows(
    h5_path: Path,
    labels: dict[str, Any],
    *,
    paired_variants: int,
    visible_variants: int,
    masked_variants: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split_by_episode = {int(index): "train" for index in labels["train_episode_indices"]}
    split_by_episode.update({int(index): "val" for index in labels["val_episode_indices"]})
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    center_audit = []
    with h5py.File(h5_path, "r") as source:
        for episode_meta in labels["episodes"]:
            episode_index = int(episode_meta["episode_index"])
            episode_name = str(episode_meta["episode_name"])
            demo_count = int(episode_meta["demo_count"])
            if demo_count != 66:
                raise ValueError(f"{episode_name}: expected 66 demo frames, got {demo_count}")
            episode = source[episode_name]
            if not all(bool(episode[f"timestep_{i}/info/is_video_demo"][()]) for i in range(demo_count)):
                raise ValueError(f"{episode_name}: non-demo frame inside prefix")
            image = np.asarray(episode["timestep_0/obs/front_rgb"][()], dtype=np.uint8)
            centers = _color_centers(image)
            target_color = str(episode_meta["target_color"])
            target_y, target_x = episode_meta["target_point_yx"]
            detected_y, detected_x = centers[target_color]
            distance = float(np.hypot(detected_y - target_y, detected_x - target_x))
            if distance > 8.0:
                raise ValueError(f"{episode_name}: visual/choice target mismatch {distance:.2f}px")
            center_audit.append(distance)
            destination = train if split_by_episode[episode_index] == "train" else val
            rng = np.random.default_rng(seed + episode_index * 1009)
            patterns = [
                ("paired_memory", variant, _indices(rng, 0, 31, 6) + _indices(rng, 32, 65, 6))
                for variant in range(paired_variants)
            ]
            patterns.extend(
                ("visible_grounding", variant, _indices(rng, 0, 31, FRAME_COUNT))
                for variant in range(visible_variants)
            )
            patterns.extend(
                ("masked_only", variant, _indices(rng, 32, 65, FRAME_COUNT))
                for variant in range(masked_variants)
            )
            for color in COLORS:
                cell = cell_from_yx(*centers[color])
                for sample_type, variant, frame_indices in patterns:
                    destination.append(
                        {
                            "schema_version": 1,
                            "source": "videounmask",
                            "episode_index": episode_index,
                            "episode_name": episode_name,
                            "h5_path": str(h5_path.resolve()),
                            "sample_type": sample_type,
                            "variant": variant,
                            "target_color": color,
                            "target_cell": cell,
                            "goal_target_color": target_color,
                            "difficulty": str(episode_meta["difficulty"]),
                            "num_targets": int(episode_meta["num_targets"]),
                            "frame_indices": frame_indices,
                            "target": compact_response(sample_type, color, cell),
                        }
                    )
    audit = {
        "visual_choice_center_distance_mean": float(np.mean(center_audit)),
        "visual_choice_center_distance_max": float(np.max(center_audit)),
        "visual_choice_center_distance_within_8px": int(sum(value <= 8 for value in center_audit)),
    }
    return train, val, audit


def _shellgame_replay(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = random.Random(seed).sample(rows, count) if count <= len(rows) else random.Random(seed).choices(rows, k=count)
    return [{**row, "source": "shellgame"} for row in selected]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "episodes": len({(row["source"], int(row["episode_index"])) for row in rows}),
        "sources": dict(sorted(Counter(str(row["source"]) for row in rows).items())),
        "sample_types": dict(sorted(Counter(str(row["sample_type"]) for row in rows).items())),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.replay_fraction < 1:
        raise ValueError("replay-fraction must be in [0, 1)")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    train, val, audit = _video_rows(
        args.h5,
        labels,
        paired_variants=args.paired_variants,
        visible_variants=args.visible_variants,
        masked_variants=args.masked_variants,
        seed=args.seed,
    )
    replay_count = round(len(train) * args.replay_fraction / (1 - args.replay_fraction))
    mixed = train + _shellgame_replay(args.shellgame_manifest, replay_count, args.seed + 17)
    random.Random(args.seed).shuffle(mixed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.output_dir / "train.jsonl",
        "train_mixed": args.output_dir / "train_mixed_shellgame25.jsonl",
        "val": args.output_dir / "val.jsonl",
    }
    if any(path.exists() for path in paths.values()) and not args.overwrite:
        raise FileExistsError("Manifest exists; pass --overwrite")
    _write(paths["train"], train)
    _write(paths["train_mixed"], mixed)
    _write(paths["val"], val)
    summary = {
        "schema_version": 1,
        "h5": str(args.h5.resolve()),
        "labels": str(args.labels.resolve()),
        "demonstration_only": True,
        "execution_frames_in_input": False,
        "frame_count": FRAME_COUNT,
        "visible_range": [0, 31],
        "masked_range": [32, 65],
        "seed": args.seed,
        "replay_fraction": replay_count / len(mixed),
        "train": _counts(train),
        "train_mixed": _counts(mixed),
        "val": _counts(val),
        "center_audit": audit,
        "copied_video_bytes": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
