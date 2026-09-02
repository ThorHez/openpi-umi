"""Compact Qwen3-VL contract for VideoUnmask demonstration memory."""

from __future__ import annotations

import json
import re
from typing import Any

COLORS = ("red", "green", "blue")
SAMPLE_TYPES = ("visible_grounding", "paired_memory", "masked_only")
SYSTEM_PROMPT = """You are a visual memory encoder for a robot. Inspect only the supplied
chronological demonstration frames and answer with exactly one compact JSON object. Coordinates
are camera-relative 8x8 image cells, with rows increasing top-to-bottom and columns increasing
left-to-right. Do not use robot execution, simulator state, or unstated evidence. Do not explain."""

_USER_PROMPT = """The target color is {color}. Determine what the supplied demonstration frames
prove about this target. If the target cube itself is visible, report event target_visible and its
cell. If the sequence first shows the target and later shows it covered, report event
target_covered and the remembered cell. If the frames show only opaque containers and never show
the target, return exactly {{"event":"insufficient_evidence"}}. A grounded answer must be exactly
{{"event":"EVENT","target_color":"{color}","target_cell":"rR_cC"}}."""

_CELL_RE = re.compile(r"r([0-7])_c([0-7])\Z")


def prompt_for_target(color: str) -> str:
    if color not in COLORS:
        raise ValueError(f"Unsupported target color: {color!r}")
    return _USER_PROMPT.format(color=color)


def cell_from_yx(y: float, x: float) -> str:
    row = min(7, max(0, int(y) // 32))
    column = min(7, max(0, int(x) // 32))
    return f"r{row}_c{column}"


def compact_response(sample_type: str, color: str | None = None, cell: str | None = None) -> str:
    if sample_type == "masked_only":
        value: dict[str, Any] = {"event": "insufficient_evidence"}
    else:
        if sample_type not in {"visible_grounding", "paired_memory"}:
            raise ValueError(f"Unsupported sample type: {sample_type!r}")
        if color not in COLORS or not isinstance(cell, str) or _CELL_RE.fullmatch(cell) is None:
            raise ValueError(f"Invalid grounded target: {color=}, {cell=}")
        event = "target_visible" if sample_type == "visible_grounding" else "target_covered"
        value = {"event": event, "target_color": color, "target_cell": cell}
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_compact_response(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    if value == {"event": "insufficient_evidence"}:
        return value
    if not isinstance(value, dict) or set(value) != {"event", "target_color", "target_cell"}:
        raise ValueError(f"Unsupported compact response: {value!r}")
    event = value["event"]
    sample_type = {"target_visible": "visible_grounding", "target_covered": "paired_memory"}.get(event)
    if sample_type is None:
        raise ValueError(f"Unsupported memory event: {event!r}")
    compact_response(sample_type, str(value["target_color"]), str(value["target_cell"]))
    return value

