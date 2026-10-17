#!/usr/bin/env python3
"""Build local target/swap manifests for variable-length VideoUnmaskSwap demos."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any

import h5py
import numpy as np
from scipy import ndimage

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme import four_task_temporal_contract as temporal_contract  # noqa: E402
from openpi.tasks.robomme.four_task_state import TargetIdentityState  # noqa: E402
from openpi.tasks.robomme.videounmaskswap.qwen3vl_local_event_contract import compact_negative_response  # noqa: E402
from openpi.tasks.robomme.videounmaskswap.qwen3vl_local_event_contract import compact_swap_response  # noqa: E402
from openpi.tasks.robomme.videounmaskswap.qwen3vl_local_event_contract import compact_target_response  # noqa: E402

DEFAULT_H5 = _ROOT / "data/robomme_extracted/record_dataset_VideoUnmaskSwap.h5"
DEFAULT_SPLITS = _ROOT / "artifacts/robomme_four_task_pilot_seed260826/episode_splits.json"
DEFAULT_OUTPUT = _ROOT / "artifacts/videounmaskswap_qwen3vl_local_events_seed260826"
FRAME_COUNT = temporal_contract.TEACHER_FRAME_COUNT
COLORS = ("red", "green", "blue")
_COORD_RE = re.compile(r"<(\d+),\s*(\d+)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _decode(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return value.decode() if isinstance(value, bytes | np.bytes_) else str(value)


def _color_centers(image: np.ndarray) -> dict[str, tuple[float, float]]:
    masks = {
        "red": (image[..., 0] > 180) & (image[..., 1] < 70) & (image[..., 2] < 70),
        "green": (image[..., 1] > 180) & (image[..., 0] < 70) & (image[..., 2] < 70),
        "blue": (image[..., 2] > 180) & (image[..., 0] < 70) & (image[..., 1] < 70),
    }
    result = {}
    for color, mask in masks.items():
        y, x = np.where(mask)
        if len(x) < 20:
            raise ValueError(f"Could not detect {color} cube: {len(x)} pixels")
        result[color] = (float(np.median(y)), float(np.median(x)))
    return result


def _gray_components(image: np.ndarray) -> list[tuple[float, float]]:
    maximum = image.max(axis=-1)
    minimum = image.min(axis=-1)
    mask = (maximum - minimum < 25) & (maximum > 75) & (maximum < 250)
    mask[:50] = False
    labels, count = ndimage.label(mask)
    components = []
    for label in range(1, count + 1):
        y, x = np.where(labels == label)
        width = int(x.max() - x.min() + 1)
        height = int(y.max() - y.min() + 1)
        if 150 <= len(x) <= 1200 and y.min() > 45 and width < 45 and height < 65:
            components.append((float(np.median(y)), float(np.median(x))))
    return components


def _position_centers(
    visible_centers: dict[str, tuple[float, float]],
    covered_image: np.ndarray,
    *,
    difficulty: str,
) -> tuple[list[tuple[float, float]], bool]:
    positions = list(visible_centers.values())
    if difficulty == "easy":
        return sorted(positions), True
    components = _gray_components(covered_image)
    ranked = sorted(
        (
            min(float(np.linalg.norm(np.subtract(component, selected))) for selected in positions),
            component,
        )
        for component in components
    )
    if not ranked or ranked[-1][0] < 10.0:
        return sorted(positions), False
    positions.append(ranked[-1][1])
    return sorted(positions), True


def _nearest_slot(
    point: tuple[float, float], position_centers: list[tuple[float, float]]
) -> str:
    index = min(
        range(len(position_centers)),
        key=lambda candidate: float(
            np.linalg.norm(np.subtract(point, position_centers[candidate]))
        ),
    )
    return f"slot_{index}"


def _target_colors(goal: str) -> tuple[str, ...]:
    colors = tuple(match.group(1) for match in re.finditer(r"(red|green|blue) cube", goal))
    if not 1 <= len(colors) <= 2:
        raise ValueError(f"Cannot parse target colors from {goal!r}")
    return colors


def _grounded_pick_slots(
    episode: h5py.Group,
    target_colors: tuple[str, ...],
    position_centers: list[tuple[float, float]],
) -> dict[str, str]:
    result = {}
    for name in sorted(
        (name for name in episode if name.startswith("timestep_")),
        key=lambda name: int(name.split("_")[-1]),
    ):
        info = episode[name]["info"]
        if bool(info["is_video_demo"][()]) or not bool(info["is_subgoal_boundary"][()]):
            continue
        simple = _decode(info["simple_subgoal"][()])
        grounded = _decode(info["grounded_subgoal"][()])
        if "hides the" not in simple:
            continue
        color = next((value for value in target_colors if f"{value} cube" in simple), None)
        match = _COORD_RE.search(grounded)
        if color is not None and match is not None:
            y, x = int(match.group(1)), int(match.group(2))
            result[color] = _nearest_slot((y, x), position_centers)
    return result


def _frames(start: int, stop: int, demo_end: int) -> list[int]:
    start = max(0, min(start, demo_end - 1))
    stop = max(start, min(stop, demo_end - 1))
    return np.linspace(start, stop, FRAME_COUNT).round().astype(int).tolist()


def _swap_pair(
    episode: h5py.Group,
    position_centers: list[tuple[float, float]],
    start: int,
    stop: int,
) -> tuple[str, str, float]:
    start_image = np.asarray(episode[f"timestep_{start}/obs/front_rgb"][()], dtype=np.float32)
    middle = (start + stop) // 2
    middle_image = np.asarray(episode[f"timestep_{middle}/obs/front_rgb"][()], dtype=np.float32)
    scores = []
    radius = 16
    for y_float, x_float in position_centers:
        y, x = int(round(y_float)), int(round(x_float))
        start_patch = start_image[y - radius : y + radius + 1, x - radius : x + radius + 1]
        middle_patch = middle_image[y - radius : y + radius + 1, x - radius : x + radius + 1]
        scores.append(float(np.mean(np.abs(start_patch - middle_patch))))
    moving = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:2]
    static_scores = [scores[index] for index in range(len(scores)) if index not in moving]
    if not static_scores or min(scores[index] for index in moving) - max(static_scores) < 10.0:
        raise ValueError(
            f"Ambiguous moving pair at {start}:{stop}: {position_centers=}, {scores=}"
        )
    container_ids = [f"slot_{index}" for index in moving]
    return container_ids[0], container_ids[1], float(max(static_scores))


def _episode_rows(
    episode_index: int, episode: h5py.Group, h5_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    goal = _decode(episode["setup/task_goal"][()])
    difficulty = _decode(episode["setup/difficulty"][()])
    target_colors = _target_colors(goal)
    step_names = [name for name in episode if name.startswith("timestep_")]
    demo_end = sum(bool(episode[name]["info/is_video_demo"][()]) for name in step_names)
    swap_count = int(round(max(0, demo_end - 64) / 50))
    if swap_count not in (1, 2, 3):
        raise ValueError(f"episode_{episode_index}: invalid {demo_end=} -> {swap_count=}")

    visible_centers = _color_centers(episode["timestep_0/obs/front_rgb"][()])
    position_centers, extra_position_valid = _position_centers(
        visible_centers,
        episode["timestep_63/obs/front_rgb"][()],
        difficulty=difficulty,
    )
    color_to_slot = {
        color: _nearest_slot(center, position_centers)
        for color, center in visible_centers.items()
    }
    state = TargetIdentityState.empty(target_colors)
    for color in target_colors:
        state = state.observe_target(color, color_to_slot[color], covered=True)

    common = {
        "schema_version": 2,
        "source": "videounmaskswap_local_event",
        "episode_index": episode_index,
        "episode_name": f"episode_{episode_index}",
        "h5_path": str(h5_path.resolve()),
        "difficulty": difficulty,
        "goal": goal,
        "target_colors": list(target_colors),
        "num_containers": len(position_centers),
        "demo_end": demo_end,
        "swap_count": swap_count,
    }
    rows = []
    for color in COLORS:
        container_id = color_to_slot[color]
        rows.append(
            {
                **common,
                "sample_type": "target_visible",
                "event_index": -2,
                "variant": 0,
                "frame_indices": _frames(0, 31, demo_end),
                "target": compact_target_response("target_visible", color, container_id),
            }
        )
        rows.append(
            {
                **common,
                "sample_type": "target_covered",
                "event_index": -1,
                "variant": 0,
                "frame_indices": _frames(20, 63, demo_end),
                "target": compact_target_response("target_covered", color, container_id),
            }
        )
    swap_pairs = []
    max_static_change_score = 0.0
    for swap_index in range(swap_count):
        start = 64 + 50 * swap_index
        stop = start + 49
        if not extra_position_valid and difficulty != "easy":
            raise ValueError("Cannot recover the fourth container's initial position")
        container_a, container_b, static_change_score = _swap_pair(
            episode, position_centers, start, stop
        )
        max_static_change_score = max(max_static_change_score, static_change_score)
        swap_pairs.append((container_a, container_b))
        for variant, shift in enumerate((0, -2, 2)):
            rows.append(
                {
                    **common,
                    "sample_type": "swap_complete",
                    "event_index": swap_index,
                    "variant": variant,
                    "frame_indices": _frames(start + shift, stop + shift, demo_end),
                    "target": compact_swap_response(container_a, container_b),
                }
            )
        rows.append(
            {
                **common,
                "sample_type": "incomplete_swap",
                "event_index": swap_index,
                "variant": 0,
                "frame_indices": _frames(start, start + 24, demo_end),
                "target": compact_negative_response("incomplete_event"),
            }
        )
        state = state.apply_swap(container_a, container_b)

    grounded = _grounded_pick_slots(episode, target_colors, position_centers)
    oracle_matches = {
        color: grounded.get(color) == state.target_cells[target_colors.index(color)]
        for color in target_colors
        if color in grounded
    }
    audit = {
        "episode_index": episode_index,
        "target_colors": list(target_colors),
        "candidate_position_yx": [list(position) for position in position_centers],
        "initial_target_slots": [color_to_slot[color] for color in target_colors],
        "swap_pairs": [list(pair) for pair in swap_pairs],
        "predicted_final_slots": list(state.target_cells),
        "grounded_pick_slots": grounded,
        "unscored_target_colors": [color for color in target_colors if color not in grounded],
        "oracle_matches": oracle_matches,
        "max_static_change_score": max_static_change_score,
    }
    return rows, audit


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["tasks"]["VideoUnmaskSwap"]
    split_by_episode = {
        int(episode): split_name
        for split_name, key in (
            ("train", "train_episode_indices"),
            ("dev", "dev_episode_indices"),
            ("test", "test_episode_indices"),
        )
        for episode in splits[key]
    }
    rows = {"train": [], "dev": [], "test": []}
    audits = []
    rejected_episodes = []
    with h5py.File(args.h5, "r") as source:
        for episode_index in range(100):
            try:
                episode_rows, audit = _episode_rows(
                    episode_index, source[f"episode_{episode_index}"], args.h5
                )
            except ValueError as error:
                rejected_episodes.append({"episode_index": episode_index, "reason": str(error)})
                continue
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
    match_values = [value for audit in audits for value in audit["oracle_matches"].values()]
    unscored_targets = sum(len(audit["unscored_target_colors"]) for audit in audits)
    summary = {
        "schema_version": 2,
        "contract": "local_target_and_swap_events",
        "fixed_demo_length_assumption": False,
        "future_frame_leakage": False,
        "h5": str(args.h5.resolve()),
        "split": str(args.splits.resolve()),
        "oracle_state_matches_grounded_pick": int(sum(match_values)),
        "oracle_state_total_targets": len(match_values),
        "oracle_state_accuracy": float(np.mean(match_values)),
        "oracle_state_unscored_targets_missing_grounded_coordinate": unscored_targets,
        "maximum_static_patch_change_score": float(
            max(audit["max_static_change_score"] for audit in audits)
        ),
        "accepted_episodes": len(audits),
        "rejected_episodes": rejected_episodes,
        **summaries,
    }
    (args.output_dir / "episode_audit.json").write_text(
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
