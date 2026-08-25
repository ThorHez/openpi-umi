"""Compact Qwen3-VL supervision contract for ShellGame visual events.

The vision-language model predicts only camera-relative facts.  A deterministic
task adapter converts these compact facts into the generic planner patch and
world-coordinate state update used by the recurrent memory.
"""

from __future__ import annotations

import json
from typing import Any

from openpi.tasks.shellgame.qwenvl_event_adapter import SCREEN_CUPS

SAMPLE_TYPES = ("reveal", "swap", "no_event", "incomplete_event")

SYSTEM_PROMPT = """You are a visual event classifier for robot memory. Inspect the supplied
chronological video frames and answer with exactly one compact JSON object. Use camera-relative
cup labels screen_left_cup, screen_middle_cup, and screen_right_cup. Never use simulator/world
coordinates. Do not explain the answer and do not infer from frames outside the supplied clip."""

_EXCHANGE_PROMPT = """Determine which two screen-relative cups complete an exchange of horizontal
positions in this clip. Track the two cup trajectories from the first frame to the last. Return
exactly {\"screen_pair\":[\"LEFT_LABEL\",\"RIGHT_LABEL\"]}; order the two labels left-to-right
on screen. Return {\"event\":\"no_event\"} if the cups remain stationary, or
{\"event\":\"incomplete_event\"} if motion is visible but the clip lacks enough before-and-after
evidence to establish a completed exchange."""

_PROMPTS = {
    "reveal": """The clip shows the reveal phase. Identify the screen-relative cup lifted above
the visible yellow ball. Return exactly {\"screen_cup\":\"LABEL\"}, replacing LABEL with one
allowed screen cup label.""",
    # These three classes deliberately share one prompt.  The target therefore
    # cannot leak through a phase-specific instruction during SFT.
    "swap": _EXCHANGE_PROMPT,
    "no_event": _EXCHANGE_PROMPT,
    "incomplete_event": _EXCHANGE_PROMPT,
}


def prompt_for_sample_type(sample_type: str) -> str:
    try:
        return _PROMPTS[sample_type]
    except KeyError as exc:
        raise ValueError(f"Unknown Qwen SFT sample type: {sample_type!r}") from exc


def compact_response(sample_type: str, label: Any = None) -> str:
    if sample_type == "reveal":
        if label not in SCREEN_CUPS:
            raise ValueError(f"Invalid reveal screen cup: {label!r}")
        value = {"screen_cup": label}
    elif sample_type == "swap":
        pair = tuple(label) if isinstance(label, list | tuple) else ()
        if len(pair) != 2 or pair[0] not in SCREEN_CUPS or pair[1] not in SCREEN_CUPS:
            raise ValueError(f"Invalid swap screen pair: {label!r}")
        if pair[0] == pair[1] or SCREEN_CUPS.index(pair[0]) >= SCREEN_CUPS.index(pair[1]):
            raise ValueError(f"Swap pair must be distinct and screen-left-to-right: {label!r}")
        value = {"screen_pair": list(pair)}
    elif sample_type in {"no_event", "incomplete_event"}:
        value = {"event": sample_type}
    else:
        raise ValueError(f"Unknown Qwen SFT sample type: {sample_type!r}")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_compact_response(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("Compact response must be a one-key JSON object")
    if "screen_cup" in value:
        compact_response("reveal", value["screen_cup"])
    elif "screen_pair" in value:
        compact_response("swap", value["screen_pair"])
    elif value.get("event") in {"no_event", "incomplete_event"}:
        compact_response(str(value["event"]))
    else:
        raise ValueError(f"Unsupported compact response: {value!r}")
    return value
