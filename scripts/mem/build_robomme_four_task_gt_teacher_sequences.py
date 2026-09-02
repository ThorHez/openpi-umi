#!/usr/bin/env python3
"""Build episode-level GT event/state sequences for the unified RoboMME teacher."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme import unified_gt_teacher as contract  # noqa: E402
from openpi.tasks.robomme.four_task_state import OrderedTargetState  # noqa: E402
from openpi.tasks.robomme.four_task_state import PickCountState  # noqa: E402
from openpi.tasks.robomme.four_task_state import TargetIdentityState  # noqa: E402
from openpi.tasks.robomme.qwen3vl_unified_event_contract import validate_compact_response  # noqa: E402

DEFAULT_INPUT = _ROOT / "artifacts/robomme_qwen_unified_events_optimized_v2_seed260826"
DEFAULT_OUTPUT = _ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _color_id(color: str | None) -> int:
    value = "none" if color is None else color.removesuffix("_cube")
    if value not in contract.COLORS:
        raise ValueError(f"Unsupported color/entity: {color!r}")
    return contract.COLORS.index(value)


def _region_id(region: str | None) -> int:
    value = "none" if region is None else region
    if value not in contract.REGIONS:
        raise ValueError(
            f"Unified teacher currently supports region_0..region_3, got {region!r}"
        )
    return contract.REGIONS.index(value)


def _temporal_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    frames = [int(value) for value in row["frame_indices"]]
    entity = str(row.get("focus_entity") or "")
    return max(frames), min(frames), int(row.get("event_index", -1)), entity


def _canonical_rows(source: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("contract") != "unified_causal_event_v1" or int(row.get("variant", 0)) != 0:
            continue
        sample_type = str(row["sample_type"])
        if source == "videounmask_variable_demo":
            keep = row.get("goal_source") == "original_instruction" and sample_type in {
                "visible_grounding",
                "paired_memory",
            }
        elif source == "videounmaskswap_local_event":
            if sample_type in {"target_visible", "target_covered"}:
                color = str(row.get("focus_entity", "")).removesuffix("_cube")
                keep = color in row["target_colors"]
            else:
                keep = sample_type == "swap_complete"
        elif source == "videoplaceorder_local_event":
            keep = sample_type in {"place_complete", "target_relevant_swap_complete"}
        elif source == "pickxtimes_local_event":
            keep = sample_type == "completed_event"
        else:
            raise ValueError(f"Unsupported source: {source}")
        if keep:
            selected.append(row)
    selected.sort(key=_temporal_key)
    return selected


def _initial_state(source: str, row: dict[str, Any]):
    if source == "videounmask_variable_demo":
        return TargetIdentityState.empty((str(row["target_color"]),))
    if source == "videounmaskswap_local_event":
        return TargetIdentityState.empty(tuple(str(value) for value in row["target_colors"]))
    if source == "videoplaceorder_local_event":
        return OrderedTargetState(str(row["target_color"]), int(row["queried_ordinal"]))
    if source == "pickxtimes_local_event":
        return PickCountState(str(row["target_color"]), int(row["required_count"]))
    raise ValueError(source)


def _apply(source: str, state, event: dict[str, Any]):
    event_type = str(event["event"])
    if source in {"videounmask_variable_demo", "videounmaskswap_local_event"}:
        if event_type in {"target_visible", "target_covered"}:
            return state.observe_target(
                str(event["entity"]).removesuffix("_cube"),
                str(event["region_a"]),
                covered=event_type == "target_covered",
            )
        if event_type == "swap_complete":
            return state.apply_swap(str(event["region_a"]), str(event["region_b"]))
    elif source == "videoplaceorder_local_event":
        if event_type == "place_complete":
            return state.place_complete(state.written_count + 1, str(event["region_a"]))
        if event_type == "swap_complete":
            return state.swap_complete(str(event["region_a"]), str(event["region_b"]))
    elif source == "pickxtimes_local_event" and event_type in {
        "pick_complete",
        "place_complete",
        "press_complete",
    }:
        return state.apply(event_type)
    raise ValueError(f"Illegal GT event for {source}: {event}")


def _goal_fields(source: str, row: dict[str, Any]) -> dict[str, Any]:
    task_id = contract.TASKS.index(source)
    if source == "videounmask_variable_demo":
        colors = (str(row["target_color"]),)
        required_count, queried_ordinal = 0, 0
    elif source == "videounmaskswap_local_event":
        colors = tuple(str(value) for value in row["target_colors"])
        required_count, queried_ordinal = 0, 0
    elif source == "videoplaceorder_local_event":
        colors = (str(row["target_color"]),)
        required_count, queried_ordinal = 0, int(row["queried_ordinal"])
    else:
        colors = (str(row["target_color"]),)
        required_count, queried_ordinal = int(row["required_count"]), 0
    padded_colors = tuple(_color_id(color) for color in colors) + (0,) * (2 - len(colors))
    return {
        "task_id": task_id,
        "goal_color_ids": padded_colors,
        "required_count": required_count,
        "queried_ordinal": queried_ordinal,
        "num_regions": int(row.get("candidate_region_count") or 0),
        "num_targets": int(row.get("num_demonstrated_targets") or len(colors)),
    }


def _state_labels(source: str, state, goal: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((len(contract.STATE_FIELDS),), dtype=np.int32)
    mask = np.zeros_like(targets, dtype=np.bool_)
    field = {name: index for index, name in enumerate(contract.STATE_FIELDS)}

    def set_field(name: str, value: int) -> None:
        targets[field[name]] = value
        mask[field[name]] = True

    set_field("task", int(goal["task_id"]))
    set_field("target_color_0", int(goal["goal_color_ids"][0]))
    set_field("target_color_1", int(goal["goal_color_ids"][1]))

    if isinstance(state, TargetIdentityState):
        cell_by_color = dict(zip(state.target_colors, state.target_cells, strict=True))
        for color in state.target_colors:
            set_field(f"{color}_cell", _region_id(cell_by_color[color]))
        set_field("covered", int(state.covered))
        set_field("completed_swap_count", int(state.completed_swap_count))
    elif isinstance(state, OrderedTargetState):
        for index in range(int(goal["num_targets"])):
            set_field(f"ordered_cell_{index}", _region_id(state.target_cells[index]))
        set_field("completed_swap_count", int(state.completed_swap_count))
        set_field("written_count", int(state.written_count))
        set_field("queried_ordinal", int(state.queried_ordinal))
    elif isinstance(state, PickCountState):
        set_field("required_count", int(state.required_count))
        set_field("completed_count", int(state.completed_count))
        set_field("holding", int(state.holding))
        set_field("ready_to_press", int(state.ready_to_press))
        set_field("done", int(state.done))
    else:
        raise TypeError(type(state))
    return targets, mask


def _complete(source: str, state) -> bool:
    if isinstance(state, TargetIdentityState):
        return all(cell is not None for cell in state.target_cells)
    if isinstance(state, OrderedTargetState):
        return state.queried_cell is not None
    if isinstance(state, PickCountState):
        return state.done and not state.holding and state.completed_count == state.required_count
    raise TypeError((source, state))


def _build_episode(source: str, episode_index: int, rows: list[dict[str, Any]], max_steps: int):
    canonical = _canonical_rows(source, rows)
    if not canonical:
        raise ValueError(f"No canonical GT events for {source} episode {episode_index}")
    if len(canonical) > max_steps:
        raise ValueError(
            f"{source} episode {episode_index} has {len(canonical)} events > max_steps={max_steps}"
        )
    goal = _goal_fields(source, canonical[0])
    state = _initial_state(source, canonical[0])
    state_targets = np.zeros((max_steps + 1, len(contract.STATE_FIELDS)), dtype=np.int32)
    state_field_mask = np.zeros_like(state_targets, dtype=np.bool_)
    state_targets[0], state_field_mask[0] = _state_labels(source, state, goal)
    event_ids = np.zeros((max_steps,), dtype=np.int32)
    entity_ids = np.zeros_like(event_ids)
    region_a_ids = np.zeros_like(event_ids)
    region_b_ids = np.zeros_like(event_ids)
    step_mask = np.zeros((max_steps,), dtype=np.bool_)
    event_records = []
    states = [asdict(state)]

    for index, row in enumerate(canonical):
        event = validate_compact_response(str(row["target"]))
        if event["event"] not in contract.EVENTS[1 : contract.STATE_CHANGING_EVENT_COUNT + 1]:
            raise ValueError(f"Canonical row is not state-changing: {event}")
        event_ids[index] = contract.EVENTS.index(str(event["event"]))
        entity_ids[index] = _color_id(event["entity"])
        region_a_ids[index] = _region_id(event["region_a"])
        region_b_ids[index] = _region_id(event["region_b"])
        step_mask[index] = True
        state = _apply(source, state, event)
        state_targets[index + 1], state_field_mask[index + 1] = _state_labels(source, state, goal)
        event_records.append(event)
        states.append(asdict(state))
    if not _complete(source, state):
        raise ValueError(f"Incomplete final GT state for {source} episode {episode_index}: {state}")
    return {
        **goal,
        "source": source,
        "episode_index": episode_index,
        "event_ids": event_ids,
        "entity_ids": entity_ids,
        "region_a_ids": region_a_ids,
        "region_b_ids": region_b_ids,
        "step_mask": step_mask,
        "state_targets": state_targets,
        "state_field_mask": state_field_mask,
        "events": event_records,
        "states": states,
    }


def _load_split(input_dir: Path, split: str, max_steps: int) -> list[dict[str, Any]]:
    episodes = []
    for source in contract.TASKS:
        path = input_dir / source / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row["episode_index"]), []).append(row)
        for episode_index, episode_rows in sorted(grouped.items()):
            episodes.append(_build_episode(source, episode_index, episode_rows, max_steps))
    return episodes


def _write_split(output_dir: Path, split: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    array_keys = (
        "task_id",
        "goal_color_ids",
        "required_count",
        "queried_ordinal",
        "num_regions",
        "num_targets",
        "episode_index",
        "event_ids",
        "entity_ids",
        "region_a_ids",
        "region_b_ids",
        "step_mask",
        "state_targets",
        "state_field_mask",
    )
    payload = {
        key: np.asarray([episode[key] for episode in episodes])
        for key in array_keys
    }
    payload["source"] = np.asarray([episode["source"] for episode in episodes])
    np.savez_compressed(output_dir / f"{split}.npz", **payload)
    with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            record = {
                "source": episode["source"],
                "episode_index": episode["episode_index"],
                "goal_color_ids": list(episode["goal_color_ids"]),
                "required_count": episode["required_count"],
                "queried_ordinal": episode["queried_ordinal"],
                "num_regions": episode["num_regions"],
                "events": episode["events"],
                "states": episode["states"],
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "episodes": len(episodes),
        "tasks": dict(sorted(Counter(episode["source"] for episode in episodes).items())),
        "event_counts": dict(
            sorted(
                Counter(event["event"] for episode in episodes for event in episode["events"]).items()
            )
        ),
        "max_events": max(len(episode["events"]) for episode in episodes),
    }


def main() -> None:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / f"{split}.{suffix}" for split in SPLITS for suffix in ("npz", "jsonl")]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Output exists under {args.output_dir}; pass --overwrite")
    split_summaries = {}
    split_episode_keys: dict[str, set[tuple[str, int]]] = {}
    for split in SPLITS:
        episodes = _load_split(args.input_dir, split, args.max_steps)
        split_summaries[split] = _write_split(args.output_dir, split, episodes)
        split_episode_keys[split] = {(episode["source"], episode["episode_index"]) for episode in episodes}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = split_episode_keys[left] & split_episode_keys[right]
            if overlap:
                raise ValueError(f"Episode leakage between {left}/{right}: {sorted(overlap)[:5]}")
    summary = {
        "schema_version": 1,
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "teacher_uses_qwen_predictions": False,
        "teacher_input": "goal categorical fields plus ordered GT event/argument sequence",
        "memory_shape": [128, 64],
        "max_steps": args.max_steps,
        "state_fields": list(contract.STATE_FIELDS),
        "field_class_counts": list(contract.FIELD_CLASS_COUNTS),
        "video_place_order_limitation": (
            "Current labels include all placements but only target-relevant hard swaps; recover full "
            "swap pairs before claiming a complete hard-scene teacher."
        ),
        "label_provenance": {
            "pickxtimes_local_event": "choice_action + subgoal boundaries + gripper edges",
            "videounmask_variable_demo": "visible color/cell labels and visible-to-covered phase",
            "videounmaskswap_local_event": "color/cell initialization + audited motion-derived swap pairs",
            "videoplaceorder_local_event": "demonstrated placement cells + target-relevant hard swaps",
        },
        "splits": split_summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

