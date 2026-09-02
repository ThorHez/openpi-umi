"""Local place/swap event contract for VideoPlaceOrder."""

from __future__ import annotations

import json
import re
from typing import Any


COLORS = ("red", "green", "blue")
NEGATIVE_EVENTS = ("no_completed_event", "incomplete_event")
_CELL_RE = re.compile(r"r([0-7])_c([0-7])\Z")

SYSTEM_PROMPT = """You are a local visual event detector for a robot demonstration. Inspect only
the supplied chronological short clip. Report a completed target placement or a completed swap of
two targets. Do not infer the final answer for the full task. Return exactly one compact JSON
object and do not explain."""

_USER_PROMPT = """Detect only the latest local event in the demonstration. For a completed
placement of the demonstration cube, report place_complete and the target cell. For a completed
target swap, report swap_complete and the two target cells. Do not infer the placement ordinal or
the final task answer. Return no_completed_event or incomplete_event when appropriate."""


def prompt_for_local_event() -> str:
    return _USER_PROMPT


def _check_cell(cell: str) -> None:
    if _CELL_RE.fullmatch(cell) is None:
        raise ValueError(f"Invalid 8x8 cell: {cell!r}")


def compact_place_response(target_cell: str) -> str:
    _check_cell(target_cell)
    return json.dumps(
        {"event": "place_complete", "target_cell": target_cell},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def compact_swap_response(cell_a: str, cell_b: str) -> str:
    _check_cell(cell_a)
    _check_cell(cell_b)
    if cell_a == cell_b:
        raise ValueError("A swap requires two distinct cells")
    return json.dumps(
        {"event": "swap_complete", "target_cells": sorted((cell_a, cell_b))},
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
    if event == "place_complete" and set(value) == {"event", "target_cell"}:
        compact_place_response(value["target_cell"])
        return value
    if event == "swap_complete" and set(value) == {"event", "target_cells"}:
        cells = value["target_cells"]
        if not isinstance(cells, list) or len(cells) != 2:
            raise ValueError(f"Invalid swap cells: {cells!r}")
        compact_swap_response(str(cells[0]), str(cells[1]))
        return value
    raise ValueError(f"Unsupported response: {value!r}")
