"""Qwen3-VL supervision contract for labeled real-world cup swaps."""

from __future__ import annotations

import json
from typing import Any

SCREEN_CUPS = ("screen_left_cup", "screen_middle_cup", "screen_right_cup")
SAMPLE_TYPES = ("local_swap", "sequence", "full_initial", "full_swap", "full_final")

SYSTEM_PROMPT = """You are a visual memory event reader. Inspect the chronological real-world
video frames and answer with exactly one compact JSON object. Use only the camera-relative labels
screen_left_cup, screen_middle_cup, and screen_right_cup. Track cups across occlusion and human-hand
motion. Do not explain the answer and do not use information outside the supplied clip."""

_LOCAL_PROMPT = """This clip spans exactly one completed exchange. Determine which two initial
screen positions exchange cups. Return exactly {"screen_pair":["LEFT_LABEL","RIGHT_LABEL"]},
ordering the two labels from screen-left to screen-right."""

_SEQUENCE_PROMPT = """This clip spans the complete observation demonstration: the ball is first
shown under one screen-position cup, followed by exactly three completed cup exchanges. Track the
ball through all three exchanges. Return exactly one JSON object with keys initial_cup, moves, and
final_cup. moves must contain the three exchanged screen-position pairs in chronological order.
Each pair must be ordered from screen-left to screen-right."""

_FULL_INITIAL_PROMPT = """This uncut clip spans the complete observation demonstration. Identify
the screen-position cup that initially covers the shown ball, before any exchange. Return exactly
{"initial_cup":"LABEL"}."""

_FULL_SWAP_PROMPT = """This uncut clip spans the complete observation demonstration with exactly
three cup exchanges. Identify only exchange number {ordinal} in chronological order. Return exactly
{{"screen_pair":["LEFT_LABEL","RIGHT_LABEL"]}}, ordering the pair screen-left to screen-right."""

_FULL_FINAL_PROMPT = """This uncut clip spans the complete observation demonstration. Track the
initially shown ball through all three exchanges and report its screen-position cup at the decision
point. Return exactly {"final_cup":"LABEL"}."""


def prompt_for_sample_type(sample_type: str, event_index: int | None = None) -> str:
    if sample_type == "local_swap":
        return _LOCAL_PROMPT
    if sample_type == "sequence":
        return _SEQUENCE_PROMPT
    if sample_type == "full_initial":
        return _FULL_INITIAL_PROMPT
    if sample_type == "full_swap":
        if event_index not in {0, 1, 2}:
            raise ValueError(f"full_swap requires event_index in [0,2], got {event_index!r}")
        return _FULL_SWAP_PROMPT.format(ordinal=("first", "second", "third")[event_index])
    if sample_type == "full_final":
        return _FULL_FINAL_PROMPT
    raise ValueError(f"Unknown real-cup sample type: {sample_type!r}")


def _validate_cup(value: Any) -> str:
    if value not in SCREEN_CUPS:
        raise ValueError(f"Invalid screen cup: {value!r}")
    return str(value)


def _validate_pair(value: Any) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Expected a two-cup list, got {value!r}")
    left, right = (_validate_cup(item) for item in value)
    if SCREEN_CUPS.index(left) >= SCREEN_CUPS.index(right):
        raise ValueError(f"Pair must be distinct and ordered screen-left to screen-right: {value!r}")
    return left, right


def compact_local_target(pair: list[str] | tuple[str, str]) -> str:
    left, right = _validate_pair(list(pair))
    return json.dumps({"screen_pair": [left, right]}, separators=(",", ":"))


def compact_cup_target(kind: str, cup: str) -> str:
    if kind not in {"initial_cup", "final_cup"}:
        raise ValueError(f"Unknown cup target kind: {kind!r}")
    return json.dumps({kind: _validate_cup(cup)}, separators=(",", ":"))


def compact_sequence_target(initial_cup: str, moves: list[list[str]], final_cup: str) -> str:
    value = {
        "initial_cup": _validate_cup(initial_cup),
        "moves": [list(_validate_pair(pair)) for pair in moves],
        "final_cup": _validate_cup(final_cup),
    }
    if len(value["moves"]) != 3:
        raise ValueError(f"Expected exactly three moves, got {moves!r}")
    tracked = value["initial_cup"]
    for left, right in value["moves"]:
        if tracked == left:
            tracked = right
        elif tracked == right:
            tracked = left
    if tracked != value["final_cup"]:
        raise ValueError(f"Sequence rollout ends at {tracked}, not {value['final_cup']}")
    return json.dumps(value, separators=(",", ":"))


def validate_response(text: str, sample_type: str, *, require_consistent: bool = False) -> dict[str, Any]:
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("Response must be a JSON object")
    if sample_type in {"local_swap", "full_swap"}:
        if set(value) != {"screen_pair"}:
            raise ValueError(f"Invalid local-swap keys: {sorted(value)}")
        _validate_pair(value["screen_pair"])
    elif sample_type == "sequence":
        if set(value) != {"initial_cup", "moves", "final_cup"}:
            raise ValueError(f"Invalid sequence keys: {sorted(value)}")
        _validate_cup(value["initial_cup"])
        if not isinstance(value["moves"], list) or len(value["moves"]) != 3:
            raise ValueError(f"Expected exactly three moves, got {value['moves']!r}")
        for pair in value["moves"]:
            _validate_pair(pair)
        _validate_cup(value["final_cup"])
        if require_consistent:
            compact_sequence_target(value["initial_cup"], value["moves"], value["final_cup"])
    elif sample_type in {"full_initial", "full_final"}:
        key = "initial_cup" if sample_type == "full_initial" else "final_cup"
        if set(value) != {key}:
            raise ValueError(f"Invalid {sample_type} keys: {sorted(value)}")
        _validate_cup(value[key])
    else:
        raise ValueError(f"Unknown real-cup sample type: {sample_type!r}")
    return value
