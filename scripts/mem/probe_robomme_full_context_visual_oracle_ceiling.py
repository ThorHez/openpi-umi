#!/usr/bin/env python3
"""Full-context RGB motion ceiling for the three RoboMME region tasks.

Unmask and UnmaskSwap are recovered from RGB only (plus the task prompt and
demonstration horizon).  PlaceOrder uses demonstration-only subgoal boundaries
and anchor coordinates to initialize the ordinal table, then infers whether and
where the queried target moved from full-context RGB patch motion.  Execution
GroundSG coordinates and canonical event/region labels are scorer-only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.tasks.robomme import unified_gt_teacher as contract  # noqa: E402
from openpi.tasks.robomme.four_task_state import TargetIdentityState  # noqa: E402
from scripts.mem.build_videounmaskswap_qwen3vl_local_event_manifest import _color_centers  # noqa: E402
from scripts.mem.build_videounmaskswap_qwen3vl_local_event_manifest import _decode  # noqa: E402
from scripts.mem.build_videounmaskswap_qwen3vl_local_event_manifest import _position_centers  # noqa: E402
from scripts.mem.build_videounmaskswap_qwen3vl_local_event_manifest import _swap_pair  # noqa: E402
from scripts.mem.build_videounmaskswap_qwen3vl_local_event_manifest import _target_colors  # noqa: E402
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import cell_from_yx  # noqa: E402


DEFAULT_TEACHER = ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
DEFAULT_DATA = ROOT / "data/robomme_extracted"
DEFAULT_OUTPUT = ROOT / "checkpoints/robomme_full_context_visual_oracle_region_ceiling_260828"
TASK_TO_ENV = {
    "videounmask_variable_demo": "VideoUnmask",
    "videounmaskswap_local_event": "VideoUnmaskSwap",
    "videoplaceorder_local_event": "VideoPlaceOrder",
}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
COORD_RE = re.compile(r"<(\d+),\s*(\d+)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _names(episode: h5py.Group) -> list[str]:
    return sorted(
        (name for name in episode if name.startswith("timestep_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )


def _final_target(
    data: dict[str, np.ndarray], row: int, field: int
) -> int:
    length = int(data["step_mask"][row].sum())
    return int(data["state_targets"][row, length, field]) - 1


def _unmask_prediction(
    episode: h5py.Group, goal_color_ids: np.ndarray
) -> dict[int, int]:
    centers = _color_centers(episode["timestep_0/obs/front_rgb"][()])
    cells = {
        color: cell_from_yx(*point) for color, point in centers.items()
    }
    ordered = sorted(
        set(cells.values()),
        key=lambda value: tuple(int(part[1:]) for part in value.split("_")),
    )
    return {
        int(color_id): ordered.index(cells[contract.COLORS[int(color_id)]])
        for color_id in goal_color_ids
        if int(color_id) > 0
    }


def _swap_prediction(episode: h5py.Group) -> dict[int, int]:
    goal = _decode(episode["setup/task_goal"][()])
    difficulty = _decode(episode["setup/difficulty"][()])
    target_colors = _target_colors(goal)
    names = _names(episode)
    demo_end = sum(bool(episode[name]["info/is_video_demo"][()]) for name in names)
    swap_count = int(round(max(0, demo_end - 64) / 50))
    visible = _color_centers(episode["timestep_0/obs/front_rgb"][()])
    positions, valid = _position_centers(
        visible, episode["timestep_63/obs/front_rgb"][()], difficulty=difficulty
    )
    if not valid:
        raise ValueError("Could not recover candidate positions")
    color_to_region = {
        color: min(
            range(len(positions)),
            key=lambda index: float(
                np.linalg.norm(np.subtract(center, positions[index]))
            ),
        )
        for color, center in visible.items()
    }
    state = TargetIdentityState.empty(target_colors)
    for color in target_colors:
        state = state.observe_target(
            color, f"region_{color_to_region[color]}", covered=True
        )
    for swap_index in range(swap_count):
        start = 64 + 50 * swap_index
        stop = start + 49
        region_a, region_b, _ = _swap_pair(episode, positions, start, stop)
        state = state.apply_swap(
            region_a.replace("slot_", "region_"),
            region_b.replace("slot_", "region_"),
        )
    return {
        contract.COLORS.index(color): int(region.rsplit("_", 1)[1])
        for color, region in zip(
            state.target_colors, state.target_cells, strict=True
        )
    }


def _coord(text: str) -> tuple[float, float] | None:
    match = COORD_RE.search(text)
    return (float(match.group(1)), float(match.group(2))) if match else None


def _place_demo(episode: h5py.Group) -> dict[str, Any]:
    goal = _decode(episode["setup/task_goal"][()])
    ordinal_values = [
        value for word, value in ORDINALS.items() if re.search(rf"\b{word}\b", goal)
    ]
    if len(ordinal_values) != 1:
        raise ValueError(f"Cannot parse ordinal from {goal!r}")
    target_colors = [
        color for color in ("red", "green", "blue") if re.search(rf"\b{color}\b", goal)
    ]
    if len(target_colors) != 1:
        raise ValueError(f"Cannot parse target color from {goal!r}")
    boundaries = []
    for index, name in enumerate(_names(episode)):
        info = episode[name]["info"]
        if bool(info["is_subgoal_boundary"][()]):
            boundaries.append(
                {
                    "index": index,
                    "demo": bool(info["is_video_demo"][()]),
                    "simple": _decode(info["simple_subgoal"][()]),
                    "grounded": _decode(info["grounded_subgoal"][()]),
                }
            )
    demo_end = next(row["index"] for row in boundaries if not row["demo"])
    demo = [row for row in boundaries if row["demo"]]
    drops = [row for row in demo if row["simple"] == "drop the cube onto target"]
    anchors = []
    for drop in drops:
        later_indices = [row["index"] for row in boundaries if row["index"] > drop["index"]]
        drop_end = min(later_indices)
        point = _coord(drop["grounded"])
        if point is None:
            later = next(
                (
                    row
                    for row in demo
                    if row["index"] > drop["index"]
                    and row["simple"] == "pick up the cube"
                    and _coord(row["grounded"]) is not None
                ),
                None,
            )
            point = _coord(later["grounded"]) if later is not None else None
        if point is None:
            image = episode[f"timestep_{drop_end - 2}/obs/front_rgb"][()]
            channel = {"red": 0, "green": 1, "blue": 2}[target_colors[0]]
            others = [index for index in range(3) if index != channel]
            mask = (
                (image[..., channel] > 160)
                & (image[..., others[0]] < 110)
                & (image[..., others[1]] < 110)
            )
            y, x = np.where(mask)
            if len(x) >= 20:
                point = (float(np.median(y)), float(np.median(x)))
        if point is None:
            raise ValueError("Missing demonstration-only placement anchor")
        anchors.append(point)
    static = [row["index"] for row in demo if row["simple"] == "static"]
    if not static:
        raise ValueError("Missing demonstration static interval")
    return {
        "difficulty": _decode(episode["setup/difficulty"][()]),
        "ordinal": ordinal_values[0],
        "demo_end": demo_end,
        "swap_start": max(static),
        "anchors": sorted(anchors),
        "demonstrated": anchors[ordinal_values[0] - 1],
    }


def _place_motion_features(episode: h5py.Group) -> dict[str, Any]:
    item = _place_demo(episode)
    anchors = item["anchors"]
    initial = min(
        range(len(anchors)),
        key=lambda index: float(
            np.linalg.norm(np.subtract(item["demonstrated"], anchors[index]))
        ),
    )
    target_motion = 0.0
    destination = initial
    scores = [0.0] * len(anchors)
    if item["difficulty"] == "hard":
        frames = np.linspace(
            item["swap_start"], item["demo_end"] - 1, 12
        ).round().astype(int)
        images = np.stack(
            [episode[f"timestep_{index}/obs/front_rgb"][()] for index in frames]
        ).astype(np.float32)
        for point in anchors:
            y, x = (int(round(value)) for value in point)
            radius = 16
            patches = images[
                :, y - radius : y + radius + 1, x - radius : x + radius + 1
            ]
            from_start = np.abs(patches - patches[0]).mean(axis=(1, 2, 3))
            scores[anchors.index(point)] = float(from_start.mean())
            if anchors.index(point) == initial:
                target_motion = float(from_start.max())
        alternatives = [index for index in range(len(anchors)) if index != initial]
        destination = max(alternatives, key=lambda index: scores[index])
    return {
        **item,
        "initial_region": initial,
        "destination_region": destination,
        "target_motion": target_motion,
        "region_motion_scores": scores,
    }


def _place_predict(features: dict[str, Any], threshold: float) -> int:
    if features["difficulty"] != "hard":
        return int(features["initial_region"])
    if float(features["target_motion"]) <= threshold:
        return int(features["initial_region"])
    return int(features["destination_region"])


def _place_rows(
    data: dict[str, np.ndarray], source: h5py.File
) -> list[dict[str, Any]]:
    result = []
    field_base = contract.STATE_FIELDS.index("ordered_cell_0")
    for row in np.flatnonzero(data["task_id"] == 2):
        row = int(row)
        episode_index = int(data["episode_index"][row])
        features = _place_motion_features(source[f"episode_{episode_index}"])
        ordinal = int(data["queried_ordinal"][row])
        result.append(
            {
                "episode_index": episode_index,
                "target": _final_target(data, row, field_base + ordinal - 1),
                "features": features,
            }
        )
    return result


def _select_threshold(
    train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]
) -> dict[str, float]:
    candidates = np.linspace(0.0, 50.0, 201)
    scored = []
    for threshold in candidates:
        train = np.mean(
            [
                _place_predict(row["features"], float(threshold)) == row["target"]
                for row in train_rows
            ]
        )
        dev = np.mean(
            [
                _place_predict(row["features"], float(threshold)) == row["target"]
                for row in dev_rows
            ]
        )
        scored.append((float(dev), float(train), -float(threshold), float(threshold)))
    dev, train, _, threshold = max(scored)
    return {"threshold": threshold, "train_accuracy": train, "dev_accuracy": dev}


def _task_metrics(predictions: list[int], targets: list[int]) -> dict[str, Any]:
    correct = np.asarray(predictions) == np.asarray(targets)
    return {
        "queries": len(correct),
        "query_accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
    }


def main() -> None:
    args = parse_args()
    h5 = {
        task: h5py.File(args.data_dir / f"record_dataset_{env}.h5", "r")
        for task, env in TASK_TO_ENV.items()
    }
    splits = {
        split: _load(args.teacher_dir / f"{split}.npz")
        for split in ("train", "dev", "test")
    }
    try:
        place = {
            split: _place_rows(splits[split], h5["videoplaceorder_local_event"])
            for split in splits
        }
        selection = _select_threshold(place["train"], place["dev"])
        results: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "full_context_visual_oracle_region_ceiling",
            "input_contract": {
                "unmask": "prompt + full demonstration RGB",
                "unmask_swap": "prompt/difficulty + full demonstration RGB; swap intervals inferred from demonstration length",
                "place_order": "prompt + demonstration-only subgoal boundaries/anchor coordinates + full swap-segment RGB",
                "forbidden_prediction_inputs": [
                    "canonical event labels",
                    "canonical region_a/region_b",
                    "state_targets",
                    "execution GroundSG coordinates",
                ],
            },
            "place_motion_threshold_selection": selection,
            "splits": {},
        }
        for split, data in splits.items():
            task_results = {}
            overall_predictions = []
            overall_targets = []
            for task_id, task in enumerate(contract.TASKS[:2]):
                predictions = []
                targets = []
                source = h5[task]
                for row in np.flatnonzero(data["task_id"] == task_id):
                    row = int(row)
                    episode = source[f"episode_{int(data['episode_index'][row])}"]
                    predicted = (
                        _unmask_prediction(episode, data["goal_color_ids"][row])
                        if task_id == 0
                        else _swap_prediction(episode)
                    )
                    for color_id in data["goal_color_ids"][row]:
                        color_id = int(color_id)
                        if color_id == 0:
                            continue
                        field = contract.STATE_FIELDS.index(
                            f"{contract.COLORS[color_id]}_cell"
                        )
                        predictions.append(predicted[color_id])
                        targets.append(_final_target(data, row, field))
                task_results[task] = _task_metrics(predictions, targets)
                overall_predictions.extend(predictions)
                overall_targets.extend(targets)
            place_predictions = [
                _place_predict(row["features"], selection["threshold"])
                for row in place[split]
            ]
            place_targets = [int(row["target"]) for row in place[split]]
            task_results["videoplaceorder_local_event"] = _task_metrics(
                place_predictions, place_targets
            )
            overall_predictions.extend(place_predictions)
            overall_targets.extend(place_targets)
            task_results["overall"] = _task_metrics(
                overall_predictions, overall_targets
            )
            results["splits"][split] = task_results
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "result.json"
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
        print(f"wrote {output}")
    finally:
        for source in h5.values():
            source.close()


if __name__ == "__main__":
    main()
