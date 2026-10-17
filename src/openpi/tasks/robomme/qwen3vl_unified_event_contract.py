"""Task-neutral causal visual-event contract shared by RoboMME tasks."""

from __future__ import annotations

import json
import re
from typing import Any


EVENTS = (
    "target_visible",
    "target_covered",
    "swap_complete",
    "pick_complete",
    "place_complete",
    "press_complete",
    "no_completed_event",
    "incomplete_event",
    "insufficient_evidence",
)
ENTITIES = ("red_cube", "green_cube", "blue_cube", "tracked_object")
_REGION_RE = re.compile(r"region_([0-7])\Z")

SYSTEM_PROMPT = """You are a general causal visual event detector. Inspect only the supplied
chronological short clip and the natural-language goal. Report the latest visible local event;
never infer an event from video history that is not supplied. Candidate spatial regions, when
present, are named region_0 through region_N in row-major screen order. Return exactly one compact
JSON object with the four keys event, entity, region_a, and region_b. Use null for every field that
does not apply, and do not explain."""

_USER_PROMPT = """Goal: {goal}
Focus entity: {focus_entity}
Candidate region count: {candidate_region_count}
When a focus entity is supplied, it is the entity queried by this sample; use the goal only as
context and never substitute another entity mentioned in the goal.
Use one event from: target_visible, target_covered, swap_complete, pick_complete, place_complete,
press_complete, no_completed_event, incomplete_event, insufficient_evidence. A swap uses region_a
and region_b. A spatial target or placement uses region_a. Pick/place manipulation events may use
an entity without a region. Return all four JSON fields."""


def entity_for_color(color: str) -> str:
    entity = f"{color}_cube"
    if entity not in ENTITIES:
        raise ValueError(f"Unsupported color: {color!r}")
    return entity


def prompt_for_goal(
    goal: str, *, focus_entity: str | None, candidate_region_count: int | None = None
) -> str:
    goal = goal.strip()
    if not goal:
        raise ValueError("A natural-language goal is required")
    if focus_entity is not None and focus_entity not in ENTITIES:
        raise ValueError(f"Unsupported focus entity: {focus_entity!r}")
    if candidate_region_count is not None and not 1 <= candidate_region_count <= 8:
        raise ValueError(f"Invalid candidate region count: {candidate_region_count}")
    return _USER_PROMPT.format(
        goal=goal,
        focus_entity=focus_entity or "none",
        candidate_region_count=candidate_region_count or "none",
    )


def _check_region(region: str | None) -> None:
    if region is not None and _REGION_RE.fullmatch(region) is None:
        raise ValueError(f"Invalid candidate region: {region!r}")


def compact_response(
    event: str,
    *,
    entity: str | None = None,
    region_a: str | None = None,
    region_b: str | None = None,
) -> str:
    if event not in EVENTS:
        raise ValueError(f"Unsupported event: {event!r}")
    if entity is not None and entity not in ENTITIES:
        raise ValueError(f"Unsupported entity: {entity!r}")
    _check_region(region_a)
    _check_region(region_b)
    if event in ("target_visible", "target_covered"):
        valid = entity is not None and region_a is not None and region_b is None
    elif event == "swap_complete":
        valid = entity is None and region_a is not None and region_b is not None and region_a != region_b
    elif event in ("pick_complete", "place_complete"):
        valid = (
            (entity is not None and region_a is None and region_b is None)
            or (entity is None and region_a is not None and region_b is None)
        )
    elif event == "press_complete":
        valid = entity is None and region_a is None and region_b is None
    else:
        valid = entity is None and region_a is None and region_b is None
    if not valid:
        raise ValueError(
            f"Invalid arguments for {event}: {entity=}, {region_a=}, {region_b=}"
        )
    return json.dumps(
        {"event": event, "entity": entity, "region_a": region_a, "region_b": region_b},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_compact_response(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    if set(value) != {"event", "entity", "region_a", "region_b"}:
        raise ValueError(f"Expected the fixed four-field schema, got {value!r}")
    canonical = compact_response(
        str(value["event"]),
        entity=value["entity"],
        region_a=value["region_a"],
        region_b=value["region_b"],
    )
    return json.loads(canonical)
