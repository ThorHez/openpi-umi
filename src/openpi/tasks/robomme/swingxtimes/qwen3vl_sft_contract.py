"""Compact Qwen3-VL supervision contract for SwingXtimes progress memory."""

from __future__ import annotations

import json
from typing import Any

EVENTS = ("right_arrival", "left_arrival")
REJECTION_EVENTS = ("no_completed_arrival", "insufficient_history")
SAMPLE_TYPES = ("causal_prefix", "no_event", "local_only")

SYSTEM_PROMPT = """You are a visual progress-memory encoder for a robot. Inspect only the
supplied chronological execution frames and answer with exactly one compact JSON object. A
right-to-left cycle is complete only after one right-target arrival followed by one left-target
arrival. Count distinct arrivals, not repeated frames at the same target. Do not explain."""

_USER_PROMPT = """The instruction requires {target_round_trips} right-to-left round trip(s) with
the {target_color} cube. Infer the completed visual progress in this causal frame prefix. For a
grounded prefix return exactly {{"event":"SIDE_arrival","right_count":R,"left_count":L,
"completed_round_trips":C,"ready_to_stop":BOOL}}. If no target arrival has completed, return
{{"event":"no_completed_arrival"}}. If the clip is only a local arrival without enough history to
recover cumulative counts, return {{"event":"insufficient_history"}}."""


def prompt_for_task(target_color: str, target_round_trips: int) -> str:
    if target_color not in {"red", "green", "blue"}:
        raise ValueError(f"Unsupported target color: {target_color!r}")
    if target_round_trips not in {1, 2, 3}:
        raise ValueError(f"Unsupported target round trips: {target_round_trips}")
    return _USER_PROMPT.format(target_color=target_color, target_round_trips=target_round_trips)


def compact_response(
    sample_type: str,
    *,
    event: str | None = None,
    right_count: int | None = None,
    left_count: int | None = None,
    target_round_trips: int | None = None,
) -> str:
    if sample_type == "no_event":
        value: dict[str, Any] = {"event": "no_completed_arrival"}
    elif sample_type == "local_only":
        value = {"event": "insufficient_history"}
    elif sample_type == "causal_prefix":
        if event not in EVENTS:
            raise ValueError(f"Invalid arrival event: {event!r}")
        if not all(isinstance(count, int) for count in (right_count, left_count, target_round_trips)):
            raise ValueError("All progress counts must be integers")
        assert right_count is not None
        assert left_count is not None
        assert target_round_trips is not None
        if not (0 <= left_count <= right_count <= target_round_trips <= 3):
            raise ValueError(
                f"Invalid progress state: {right_count=}, {left_count=}, {target_round_trips=}"
            )
        if event == "right_arrival" and right_count != left_count + 1:
            raise ValueError("A right arrival must lead its paired left arrival by one")
        if event == "left_arrival" and right_count != left_count:
            raise ValueError("A left arrival must complete the current pair")
        value = {
            "event": event,
            "right_count": right_count,
            "left_count": left_count,
            "completed_round_trips": left_count,
            "ready_to_stop": left_count == target_round_trips,
        }
    else:
        raise ValueError(f"Unsupported sample type: {sample_type!r}")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_compact_response(text: str, *, target_round_trips: int | None = None) -> dict[str, Any]:
    value = json.loads(text.strip())
    if value in ({"event": "no_completed_arrival"}, {"event": "insufficient_history"}):
        return value
    if not isinstance(value, dict) or set(value) != {
        "event",
        "right_count",
        "left_count",
        "completed_round_trips",
        "ready_to_stop",
    }:
        raise ValueError(f"Unsupported compact response: {value!r}")
    target = target_round_trips
    if target is None:
        target = int(value["left_count"]) if bool(value["ready_to_stop"]) else 3
    expected = compact_response(
        "causal_prefix",
        event=str(value["event"]),
        right_count=value["right_count"],
        left_count=value["left_count"],
        target_round_trips=target,
    )
    normalized = json.loads(expected)
    if value != normalized:
        raise ValueError(f"Inconsistent progress response: {value!r}")
    return value
