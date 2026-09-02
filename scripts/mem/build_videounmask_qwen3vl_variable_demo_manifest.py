#!/usr/bin/env python3
"""Build VideoUnmask manifests without assuming a fixed demonstration length."""

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
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme import four_task_temporal_contract as temporal_contract  # noqa: E402
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import COLORS  # noqa: E402
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import cell_from_yx  # noqa: E402
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import compact_response  # noqa: E402

DEFAULT_H5 = _ROOT / "data/robomme_extracted/record_dataset_VideoUnmask.h5"
DEFAULT_SPLITS = _ROOT / "artifacts/robomme_four_task_pilot_seed260826/episode_splits.json"
DEFAULT_OUTPUT = _ROOT / "artifacts/videounmask_qwen3vl_variable_demo_seed260826"
FRAME_COUNT = temporal_contract.TEACHER_FRAME_COUNT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _decode(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return value.decode() if isinstance(value, bytes | np.bytes_) else str(value)


def _color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "red": (image[..., 0] > 180) & (image[..., 1] < 70) & (image[..., 2] < 70),
        "green": (image[..., 1] > 180) & (image[..., 0] < 70) & (image[..., 2] < 70),
        "blue": (image[..., 2] > 180) & (image[..., 0] < 70) & (image[..., 1] < 70),
    }


def _centers(image: np.ndarray) -> dict[str, tuple[float, float]]:
    result = {}
    for color, mask in _color_masks(image).items():
        y, x = np.where(mask)
        if len(x) < 20:
            raise ValueError(f"Could not locate {color} cube: {len(x)} pixels")
        result[color] = (float(np.median(y)), float(np.median(x)))
    return result


def _demo_end(episode: h5py.Group) -> int:
    names = sorted(
        (name for name in episode if name.startswith("timestep_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    flags = [bool(episode[name]["info/is_video_demo"][()]) for name in names]
    indices = [index for index, flag in enumerate(flags) if flag]
    if not indices or indices != list(range(indices[-1] + 1)):
        raise ValueError("Video demonstration is not one contiguous prefix")
    return indices[-1] + 1


def _phase_bounds(episode: h5py.Group, demo_end: int) -> tuple[int, int]:
    visible = []
    fully_masked = []
    for index in range(demo_end):
        image = episode[f"timestep_{index}/obs/front_rgb"][()]
        counts = [int(mask.sum()) for mask in _color_masks(image).values()]
        visible.append(all(count >= 20 for count in counts))
        fully_masked.append(all(count < 8 for count in counts))
    visible_end = next(
        (index for index, value in enumerate(visible) if not value),
        demo_end,
    )
    masked_start = next(
        (index for index in range(visible_end, demo_end) if fully_masked[index]),
        None,
    )
    if visible_end < FRAME_COUNT or masked_start is None or demo_end - masked_start < FRAME_COUNT:
        raise ValueError(
            f"Cannot recover visible/masked phases: {visible_end=}, {masked_start=}, {demo_end=}"
        )
    return visible_end, masked_start


def _sample(
    rng: np.random.Generator, start: int, stop: int, count: int
) -> list[int]:
    values = np.arange(start, stop)
    if len(values) < count:
        return np.linspace(start, stop - 1, count).round().astype(int).tolist()
    return sorted(int(value) for value in rng.choice(values, count, replace=False))


def _episode_rows(
    episode_index: int,
    episode: h5py.Group,
    h5_path: Path,
    *,
    variants: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    demo_end = _demo_end(episode)
    visible_end, masked_start = _phase_bounds(episode, demo_end)
    centers = _centers(episode["timestep_0/obs/front_rgb"][()])
    common = {
        "schema_version": 2,
        "source": "videounmask_variable_demo",
        "episode_index": episode_index,
        "episode_name": f"episode_{episode_index}",
        "h5_path": str(h5_path.resolve()),
        "difficulty": _decode(episode["setup/difficulty"][()]),
        "goal": _decode(episode["setup/task_goal"][()]),
        "demo_end": demo_end,
        "visible_end": visible_end,
        "masked_start": masked_start,
    }
    rng = np.random.default_rng(seed + episode_index * 1009)
    rows = []
    for color in COLORS:
        cell = cell_from_yx(*centers[color])
        for variant in range(variants):
            visible_frames = _sample(rng, 0, visible_end, FRAME_COUNT)
            masked_frames = _sample(rng, masked_start, demo_end, FRAME_COUNT)
            paired_frames = sorted(
                _sample(rng, 0, visible_end, FRAME_COUNT // 2)
                + _sample(rng, masked_start, demo_end, FRAME_COUNT // 2)
            )
            for sample_type, frame_indices in (
                ("visible_grounding", visible_frames),
                ("paired_memory", paired_frames),
                ("masked_only", masked_frames),
            ):
                rows.append(
                    {
                        **common,
                        "sample_type": sample_type,
                        "variant": variant,
                        "target_color": color,
                        "target_cell": cell,
                        "frame_indices": frame_indices,
                        "target": compact_response(sample_type, color, cell),
                    }
                )
    return rows, {
        "episode_index": episode_index,
        "demo_end": demo_end,
        "visible_end": visible_end,
        "masked_start": masked_start,
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    split = json.loads(args.splits.read_text(encoding="utf-8"))["tasks"]["VideoUnmask"]
    split_by_episode = {
        int(episode): split_name
        for split_name, key in (
            ("train", "train_episode_indices"),
            ("dev", "dev_episode_indices"),
            ("test", "test_episode_indices"),
        )
        for episode in split[key]
    }
    rows = {"train": [], "dev": [], "test": []}
    audits = []
    with h5py.File(args.h5, "r") as source:
        for episode_index in range(100):
            episode_rows, audit = _episode_rows(
                episode_index,
                source[f"episode_{episode_index}"],
                args.h5,
                variants=args.variants,
                seed=args.seed,
            )
            rows[split_by_episode[episode_index]].extend(episode_rows)
            audits.append(audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split_name, split_rows in rows.items():
        path = args.output_dir / f"{split_name}.jsonl"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Manifest exists: {path}; pass --overwrite")
        _write(path, split_rows)
        summaries[split_name] = {
            "episodes": len({row["episode_index"] for row in split_rows}),
            "samples": len(split_rows),
            "sample_types": dict(
                sorted(Counter(row["sample_type"] for row in split_rows).items())
            ),
        }
    summary = {
        "schema_version": 2,
        "fixed_demo_length_assumption": False,
        "future_frame_leakage": False,
        "h5": str(args.h5.resolve()),
        "split": str(args.splits.resolve()),
        "demo_end_range": [min(x["demo_end"] for x in audits), max(x["demo_end"] for x in audits)],
        "visible_end_range": [
            min(x["visible_end"] for x in audits),
            max(x["visible_end"] for x in audits),
        ],
        "masked_start_range": [
            min(x["masked_start"] for x in audits),
            max(x["masked_start"] for x in audits),
        ],
        **summaries,
    }
    (args.output_dir / "phase_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
