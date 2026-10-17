"""Compact Qwen3-VL supervision contract for PickXtimes progress memory."""

from __future__ import annotations

import json
from typing import Any

EVENTS = ("pick_complete", "place_complete", "press_complete")
REJECTION_EVENTS = ("no_completed_event", "insufficient_history")

SYSTEM_PROMPT = """You are a visual progress-memory encoder for a robot. Inspect only the
supplied chronological execution frames and answer with exactly one compact JSON object. Count a
pick-place cycle only when the target cube has been released on the target. Repeated views of the
same transition are one event. Do not explain."""

_USER_PROMPT = """The instruction requires picking and placing the {target_color} cube
{required_count} time(s), followed by pressing the stop button. Infer the latest completed event
and cumulative state in this causal frame prefix. Return exactly {{"event":"EVENT",
"completed_count":C,"holding":BOOL,"ready_to_press":BOOL,"done":BOOL}}. If no event has
completed, return {{"event":"no_completed_event"}}. If only a local transition is visible without
enough history to recover the cumulative count, return {{"event":"insufficient_history"}}."""


def prompt_for_task(target_color: str, required_count: int) -> str:
    if target_color not in {"red", "green", "blue"}:
        raise ValueError(f"Unsupported target color: {target_color!r}")
    if required_count not in {1, 2, 3, 4, 5}:
        raise ValueError(f"Unsupported required count: {required_count}")
    return _USER_PROMPT.format(target_color=target_color, required_count=required_count)


def compact_response(
    sample_type: str,
    *,
    event: str | None = None,
    completed_count: int | None = None,
    required_count: int | None = None,
) -> str:
    if sample_type == "no_event":
        value: dict[str, Any] = {"event": "no_completed_event"}
    elif sample_type == "local_only":
        value = {"event": "insufficient_history"}
    elif sample_type == "causal_prefix":
        if event not in EVENTS:
            raise ValueError(f"Invalid event: {event!r}")
        if not isinstance(completed_count, int) or not isinstance(required_count, int):
            raise ValueError("completed_count and required_count must be integers")
        if not 0 <= completed_count <= required_count <= 5:
            raise ValueError(f"Invalid progress state: {completed_count=}, {required_count=}")
        if event == "pick_complete" and completed_count >= required_count:
            raise ValueError("A pick cannot follow the final completed placement")
        if event == "press_complete" and completed_count != required_count:
            raise ValueError("The stop press is valid only after all placements")
        value = {
            "event": event,
            "completed_count": completed_count,
            "holding": event == "pick_complete",
            "ready_to_press": event == "place_complete" and completed_count == required_count,
            "done": event == "press_complete",
        }
    else:
        raise ValueError(f"Unsupported sample type: {sample_type!r}")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_compact_response(text: str, *, required_count: int | None = None) -> dict[str, Any]:
    value = json.loads(text.strip())
    if value in ({"event": "no_completed_event"}, {"event": "insufficient_history"}):
        return value
    if not isinstance(value, dict) or set(value) != {
        "event",
        "completed_count",
        "holding",
        "ready_to_press",
        "done",
    }:
        raise ValueError(f"Unsupported compact response: {value!r}")
    if required_count is None:
        raise ValueError("required_count is needed to validate a progress response")
    expected = json.loads(
        compact_response(
            "causal_prefix",
            event=str(value["event"]),
            completed_count=value["completed_count"],
            required_count=required_count,
        )
    )
    if value != expected:
        raise ValueError(f"Inconsistent progress response: {value!r}")
    return value

