#!/usr/bin/env python3
"""Measure a causal symbolic upper bound for RoboMME region memory.

The summarizer never reads the final state target.  It receives only the goal
fields and the ordered causal event fields (event, entity, region_a, region_b),
then maintains an explicit entity/ordinal-to-region table.  Final state targets
are used only for scoring.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openpi.tasks.robomme import unified_gt_teacher as contract  # noqa: E402


DEFAULT_DATA = ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
DEFAULT_OUTPUT = ROOT / "checkpoints/robomme_region_information_ceiling_symbolic_260828"
REGION_TASKS = contract.TASKS[:3]
CELL_FIELDS = {
    color_id: contract.STATE_FIELDS.index(f"{color}_cell")
    for color_id, color in enumerate(contract.COLORS)
    if color_id > 0
}
ORDERED_FIELDS = tuple(
    contract.STATE_FIELDS.index(f"ordered_cell_{index}") for index in range(4)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _swap(value: int, region_a: int, region_b: int) -> int:
    if value == region_a:
        return region_b
    if value == region_b:
        return region_a
    return value


def _initial_region_state(task_id: int, goal_color_ids: np.ndarray) -> dict[str, Any]:
    if task_id in (0, 1):
        return {
            "color_cells": {
                int(color): 0 for color in goal_color_ids if int(color) > 0
            }
        }
    if task_id == 2:
        return {"ordered_cells": [0, 0, 0, 0], "written_count": 0}
    raise ValueError(f"Not a region task: {task_id}")


def _apply_event(
    state: dict[str, Any],
    task_id: int,
    event_id: int,
    entity_id: int,
    region_a: int,
    region_b: int,
    *,
    use_swap_regions: bool,
) -> None:
    event = contract.EVENTS[event_id]
    if task_id in (0, 1):
        if event in ("target_visible", "target_covered"):
            if entity_id in state["color_cells"]:
                state["color_cells"][entity_id] = region_a
        elif event == "swap_complete" and use_swap_regions:
            state["color_cells"] = {
                color: _swap(cell, region_a, region_b)
                for color, cell in state["color_cells"].items()
            }
        return
    if task_id == 2:
        if event == "place_complete":
            index = int(state["written_count"])
            if index >= 4:
                raise ValueError("More than four placement events")
            state["ordered_cells"][index] = region_a
            state["written_count"] = index + 1
        elif event == "swap_complete" and use_swap_regions:
            state["ordered_cells"] = [
                _swap(cell, region_a, region_b) for cell in state["ordered_cells"]
            ]
        return
    raise ValueError(task_id)


def _region_vector(state: dict[str, Any], task_id: int) -> np.ndarray:
    result = np.zeros((len(contract.STATE_FIELDS),), dtype=np.int32)
    if task_id in (0, 1):
        for color_id, region in state["color_cells"].items():
            result[CELL_FIELDS[color_id]] = region
    else:
        for index, region in enumerate(state["ordered_cells"]):
            result[ORDERED_FIELDS[index]] = region
    return result


def rollout_episode(
    data: dict[str, np.ndarray], row: int, *, use_swap_regions: bool
) -> np.ndarray:
    task_id = int(data["task_id"][row])
    state = _initial_region_state(task_id, data["goal_color_ids"][row])
    trajectory = [_region_vector(state, task_id)]
    length = int(data["step_mask"][row].sum())
    for step in range(length):
        _apply_event(
            state,
            task_id,
            int(data["event_ids"][row, step]),
            int(data["entity_ids"][row, step]),
            int(data["region_a_ids"][row, step]),
            int(data["region_b_ids"][row, step]),
            use_swap_regions=use_swap_regions,
        )
        trajectory.append(_region_vector(state, task_id))
    return np.stack(trajectory)


def _query_fields(data: dict[str, np.ndarray], row: int) -> tuple[int, ...]:
    task_id = int(data["task_id"][row])
    if task_id in (0, 1):
        return tuple(
            CELL_FIELDS[int(color)]
            for color in data["goal_color_ids"][row]
            if int(color) > 0
        )
    ordinal = int(data["queried_ordinal"][row])
    return (ORDERED_FIELDS[ordinal - 1],)


def evaluate(
    data: dict[str, np.ndarray], *, use_swap_regions: bool
) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in range(len(data["task_id"])):
        task_id = int(data["task_id"][row])
        if task_id >= len(REGION_TASKS):
            continue
        source = REGION_TASKS[task_id]
        length = int(data["step_mask"][row].sum())
        predicted = rollout_episode(data, row, use_swap_regions=use_swap_regions)
        target = data["state_targets"][row, : length + 1]
        mask = data["state_field_mask"][row, : length + 1].copy()
        region_field_mask = np.zeros((len(contract.STATE_FIELDS),), dtype=np.bool_)
        region_field_mask[list(CELL_FIELDS.values()) + list(ORDERED_FIELDS)] = True
        mask &= region_field_mask[None]
        state_valid = mask.any(axis=1)
        state_exact = ((predicted == target) | ~mask).all(axis=1)[state_valid]
        query_fields = _query_fields(data, row)
        query_correct = predicted[-1, list(query_fields)] == target[-1, list(query_fields)]

        for key in (source, "overall"):
            values = totals[key]
            values["episodes"] += 1
            values["queries"] += len(query_fields)
            values["query_correct"] += float(query_correct.sum())
            values["episode_correct"] += float(query_correct.all())
            values["complete"] += float(
                (predicted[-1, list(query_fields)] > 0).all()
            )
            values["states"] += len(state_exact)
            values["state_correct"] += float(state_exact.sum())
            values["chance_sum"] += float(1.0 / int(data["num_regions"][row])) * len(
                query_fields
            )

    result = {}
    for key, values in totals.items():
        episodes = int(values["episodes"])
        queries = int(values["queries"])
        states = int(values["states"])
        result[key] = {
            "episodes": episodes,
            "queries": queries,
            "final_query_accuracy": values["query_correct"] / queries,
            "final_episode_exact_accuracy": values["episode_correct"] / episodes,
            "final_complete_rate": values["complete"] / episodes,
            "region_state_trajectory_exact_accuracy": values["state_correct"] / states,
            "candidate_count_random_accuracy": values["chance_sum"] / queries,
        }
    return result


def main() -> None:
    args = parse_args()
    results: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "causal_symbolic_region_information_ceiling",
        "prediction_inputs": [
            "task_id",
            "goal_color_ids",
            "queried_ordinal",
            "event_ids",
            "entity_ids",
            "region_a_ids",
            "region_b_ids",
        ],
        "target_leakage": "state_targets are read only by the scorer",
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        data = _load(args.data_dir / f"{split}.npz")
        results["splits"][split] = {
            "full_causal_region_events": evaluate(data, use_swap_regions=True),
            "hold_without_swap_region_updates": evaluate(
                data, use_swap_regions=False
            ),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "result.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

