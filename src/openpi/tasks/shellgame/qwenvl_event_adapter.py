"""ShellGame adapter for generic Qwen-VL event/state-delta proposals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from openpi.planning.qwenvl_event_schema import PlannerPatch

SLOTS = ("left", "middle", "right")
SWAP_PAIRS = (("left", "middle"), ("left", "right"), ("middle", "right"))
SCREEN_CUPS = ("screen_left_cup", "screen_middle_cup", "screen_right_cup")
_WORLD_TO_SCREEN = {
    "left": "screen_right_cup",
    "middle": "screen_middle_cup",
    "right": "screen_left_cup",
}


def normalize_cup_entity(value: str) -> str:
    """Normalize conservative aliases while rejecting ungrounded entities."""
    compact = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "left": "left",
        "left_cup": "left",
        "leftmost": "left",
        "leftmost_cup": "left",
        "middle": "middle",
        "middle_cup": "middle",
        "center": "middle",
        "center_cup": "middle",
        "central": "middle",
        "central_cup": "middle",
        "right": "right",
        "right_cup": "right",
        "rightmost": "right",
        "rightmost_cup": "right",
        # Qwen labels appearance in the opponent-camera frame.  The camera is
        # mirrored relative to the simulator's world-slot convention, so the
        # task adapter performs the calibrated coordinate transform.
        "screen_left": "right",
        "screen_left_cup": "right",
        "screen_middle": "middle",
        "screen_middle_cup": "middle",
        "screen_center": "middle",
        "screen_center_cup": "middle",
        "screen_right": "left",
        "screen_right_cup": "left",
    }
    if compact not in aliases:
        raise ValueError(f"Unknown ShellGame cup entity: {value!r}")
    return aliases[compact]


def screen_cup_from_world_slot(slot: str) -> str:
    """Convert simulator/world slots to the opponent-camera label contract."""
    try:
        return _WORLD_TO_SCREEN[slot]
    except KeyError as exc:
        raise ValueError(f"Unknown ShellGame world slot: {slot!r}") from exc


def screen_pair_from_world_pair(pair: Iterable[str]) -> tuple[str, str]:
    """Convert a world exchange pair and order it left-to-right on screen."""
    values = tuple(pair)
    if len(values) != 2 or values[0] == values[1]:
        raise ValueError(f"Invalid ShellGame world exchange pair: {values!r}")
    screen_pair = tuple(screen_cup_from_world_slot(value) for value in values)
    canonical = tuple(sorted(screen_pair, key=SCREEN_CUPS.index))
    valid_pairs = {
        ("screen_left_cup", "screen_middle_cup"),
        ("screen_left_cup", "screen_right_cup"),
        ("screen_middle_cup", "screen_right_cup"),
    }
    if canonical not in valid_pairs:
        raise ValueError(f"Invalid ShellGame screen exchange pair: {canonical!r}")
    return canonical


def initial_slot_from_patch(patch: PlannerPatch) -> str:
    delta = patch.event.state_delta
    if delta.operation != "set_relation":
        raise ValueError(f"Reveal requires set_relation, got {delta.operation!r}")
    if (delta.subject or "").strip().lower() not in {"ball", "small_ball", "yellow_ball"}:
        raise ValueError(f"Reveal subject must be ball, got {delta.subject!r}")
    if (delta.predicate or "").strip().lower() not in {
        "contained_by",
        "under",
        "hidden_by",
        "covered_by",
    }:
        raise ValueError(f"Unsupported reveal predicate: {delta.predicate!r}")
    if delta.object is None:
        raise ValueError("Reveal relation has no object cup")
    return normalize_cup_entity(delta.object)


def swap_pair_from_patch(patch: PlannerPatch) -> tuple[str, str]:
    delta = patch.event.state_delta
    if delta.operation != "exchange_entity_states":
        raise ValueError(f"Swap requires exchange_entity_states, got {delta.operation!r}")
    pair = tuple(normalize_cup_entity(value) for value in delta.subjects)
    canonical = tuple(sorted(pair, key=SLOTS.index))
    if canonical not in SWAP_PAIRS:
        raise ValueError(f"Invalid ShellGame exchange pair: {pair!r}")
    return canonical


def relation_id_from_patch(patch: PlannerPatch) -> int:
    return SWAP_PAIRS.index(swap_pair_from_patch(patch))


def apply_swap(slot: str, pair: tuple[str, str]) -> str:
    if slot == pair[0]:
        return pair[1]
    if slot == pair[1]:
        return pair[0]
    return slot


@dataclass
class ShellGameTaskLedger:
    """Deterministic, idempotent high-level task state for one episode."""

    task_id: str
    target_slot: str | None = None
    revision: int = 0
    active_subgoal_id: str = "identify_hidden_ball_container"
    committed_event_ids: tuple[str, ...] = ()
    uncertain: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "revision": self.revision,
            "active_subgoal_id": self.active_subgoal_id,
            "hidden_ball_container_slot": self.target_slot,
            "committed_event_ids": list(self.committed_event_ids),
            "uncertain": self.uncertain,
        }

    def commit_reveal(self, patch: PlannerPatch) -> bool:
        if patch.event.event_id in self.committed_event_ids:
            return False
        self.target_slot = initial_slot_from_patch(patch)
        self.revision += 1
        self.active_subgoal_id = "track_hidden_container"
        self.committed_event_ids += (patch.event.event_id,)
        self.uncertain = False
        return True

    def commit_swap(self, patch: PlannerPatch) -> bool:
        if patch.event.event_id in self.committed_event_ids:
            return False
        if self.target_slot is None:
            raise ValueError("Cannot apply a swap before the initial target slot is known")
        self.target_slot = apply_swap(self.target_slot, swap_pair_from_patch(patch))
        self.revision += 1
        self.committed_event_ids += (patch.event.event_id,)
        self.uncertain = False
        return True


def load_annotation_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def group_valid_records(records: Iterable[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        if not record.get("schema_valid", False) or not record.get("adapter_valid", False):
            continue
        episode = int(record["episode_index"])
        key = str(record["query_key"])
        if key in grouped.setdefault(episode, {}):
            raise ValueError(f"Duplicate annotation for episode={episode}, query={key}")
        grouped[episode][key] = record
    return grouped
