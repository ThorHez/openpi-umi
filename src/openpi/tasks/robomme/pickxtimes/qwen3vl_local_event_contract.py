"""Local-only Qwen3-VL event contract for PickXtimes."""

from __future__ import annotations

import json
from typing import Any


COLORS = ("red", "green", "blue")
EVENTS = ("pick_complete", "place_complete", "press_complete")
NEGATIVE_EVENTS = ("no_completed_event", "incomplete_event")

SYSTEM_PROMPT = """You are a local visual event detector for a robot. Inspect only the supplied
chronological short clip. Report a completed pick, place, or stop-button press for the requested
target. Do not infer cumulative counts from earlier history. Repeated views of the same transition
describe one local event. Return exactly one compact JSON object and do not explain."""

_USER_PROMPT = """The instruction target is the {target_color} cube. Decide whether this clip
completes one of pick_complete, place_complete, or press_complete. If no transition completes,
return {{\"event\":\"no_completed_event\"}}. If a transition starts but its result is not visible,
return {{\"event\":\"incomplete_event\"}}. A positive answer must be exactly
{{\"event\":\"EVENT\",\"target_color\":\"{target_color}\"}}."""


def prompt_for_task(target_color: str) -> str:
    if target_color not in COLORS:
        raise ValueError(f"Unsupported target color: {target_color!r}")
    return _USER_PROMPT.format(target_color=target_color)


def compact_response(event: str, *, target_color: str | None = None) -> str:
    if event in NEGATIVE_EVENTS:
        value: dict[str, Any] = {"event": event}
    elif event in EVENTS and target_color in COLORS:
        value = {"event": event, "target_color": target_color}
    else:
        raise ValueError(f"Invalid local event: {event=}, {target_color=}")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_compact_response(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    if value.get("event") in NEGATIVE_EVENTS and set(value) == {"event"}:
        return value
    if set(value) != {"event", "target_color"}:
        raise ValueError(f"Unsupported response: {value!r}")
    compact_response(str(value["event"]), target_color=str(value["target_color"]))
    return value
