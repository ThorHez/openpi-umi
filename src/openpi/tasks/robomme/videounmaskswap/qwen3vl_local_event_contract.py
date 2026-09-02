"""Local target and swap event contract for VideoUnmaskSwap."""

from __future__ import annotations

import json
from typing import Any


COLORS = ("red", "green", "blue")
NEGATIVE_EVENTS = ("no_completed_event", "incomplete_event", "insufficient_evidence")
CONTAINER_IDS = ("slot_0", "slot_1", "slot_2", "slot_3")

SYSTEM_PROMPT = """You are a local visual event detector for an object permanence task. Inspect
only the supplied chronological short clip. Report a visible/covered target or one completed swap
of two opaque containers. Do not infer the final target location from missing history. Return
exactly one compact JSON object and do not explain. Candidate slots are assigned in row-major
screen order at the start of the clip."""

_USER_PROMPT = """There are {num_containers} candidate containers. Candidate ids slot_0 through
slot_N follow row-major screen order at the start of the clip. Report a visible/covered colored
target or one completed container swap. Do not infer the final target from earlier missing history.
Return no_completed_event, incomplete_event, or insufficient_evidence when appropriate."""


def prompt_for_local_event(num_containers: int) -> str:
    if num_containers not in (3, 4):
        raise ValueError(f"Expected three or four containers, got {num_containers}")
    return _USER_PROMPT.format(num_containers=num_containers)


def _check_container_id(container_id: str) -> None:
    if container_id not in CONTAINER_IDS:
        raise ValueError(f"Invalid candidate container id: {container_id!r}")


def compact_target_response(event: str, target_color: str, container_id: str) -> str:
    if event not in ("target_visible", "target_covered") or target_color not in COLORS:
        raise ValueError(f"Invalid target event: {event=}, {target_color=}")
    _check_container_id(container_id)
    return json.dumps(
        {"event": event, "target_color": target_color, "container_id": container_id},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def compact_swap_response(container_a: str, container_b: str) -> str:
    _check_container_id(container_a)
    _check_container_id(container_b)
    if container_a == container_b:
        raise ValueError("A swap requires two distinct container ids")
    return json.dumps(
        {"event": "swap_complete", "container_ids": sorted((container_a, container_b))},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def compact_negative_response(event: str) -> str:
    if event not in NEGATIVE_EVENTS:
        raise ValueError(f"Invalid negative event: {event!r}")
    return json.dumps({"event": event}, separators=(",", ":"))


def validate_compact_response(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    event = value.get("event")
    if event in NEGATIVE_EVENTS and set(value) == {"event"}:
        return value
    if event in ("target_visible", "target_covered") and set(value) == {
        "event",
        "target_color",
        "container_id",
    }:
        compact_target_response(event, value["target_color"], value["container_id"])
        return value
    if event == "swap_complete" and set(value) == {"event", "container_ids"}:
        container_ids = value["container_ids"]
        if not isinstance(container_ids, list) or len(container_ids) != 2:
            raise ValueError(f"Invalid swap container ids: {container_ids!r}")
        compact_swap_response(str(container_ids[0]), str(container_ids[1]))
        return value
    raise ValueError(f"Unsupported response: {value!r}")
