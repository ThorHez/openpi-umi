"""Compact Qwen3-VL contract for VideoPlaceOrder demonstration memory."""

from __future__ import annotations

import json
import re
from typing import Any

COLORS = ("red", "green", "blue")
ORDINALS = (1, 2, 3, 4)
SYSTEM_PROMPT = """You are a visual demonstration-memory encoder for a robot. Inspect only the
supplied chronological demonstration frames and answer with exactly one compact JSON object.
Coordinates are camera-relative 8x8 image cells, with rows increasing top-to-bottom and columns
increasing left-to-right. Track the identity of a demonstrated target if targets move or swap.
Do not use later robot execution frames and do not explain."""

_USER_PROMPT = """The robot must place the {target_color} cube on the target used in demonstrated
placement number {ordinal}. Recover that target's location at the end of the supplied
demonstration. A grounded answer must be exactly {{"event":"ordinal_target_grounded",
"target_color":"{target_color}","ordinal":{ordinal},"target_cell":"rR_cC"}}. If the supplied
frames do not establish both the requested order and final target location, return exactly
{{"event":"insufficient_evidence"}}."""

_CELL_RE = re.compile(r"r([0-7])_c([0-7])\Z")


def prompt_for_task(target_color: str, ordinal: int) -> str:
    if target_color not in COLORS:
        raise ValueError(f"Unsupported target color: {target_color!r}")
    if ordinal not in ORDINALS:
        raise ValueError(f"Unsupported ordinal: {ordinal}")
    return _USER_PROMPT.format(target_color=target_color, ordinal=ordinal)


def cell_from_xy(x: float, y: float) -> str:
    row = min(7, max(0, int(y) // 32))
    column = min(7, max(0, int(x) // 32))
    return f"r{row}_c{column}"


def compact_response(
    sample_type: str,
    *,
    target_color: str | None = None,
    ordinal: int | None = None,
    target_cell: str | None = None,
) -> str:
    if sample_type in {"truncated_demo", "local_only"}:
        value: dict[str, Any] = {"event": "insufficient_evidence"}
    elif sample_type == "full_demo":
        if (
            target_color not in COLORS
            or ordinal not in ORDINALS
            or not isinstance(target_cell, str)
            or _CELL_RE.fullmatch(target_cell) is None
        ):
            raise ValueError(
                f"Invalid grounded target: {target_color=}, {ordinal=}, {target_cell=}"
            )
        value = {
            "event": "ordinal_target_grounded",
            "target_color": target_color,
            "ordinal": ordinal,
            "target_cell": target_cell,
        }
    else:
        raise ValueError(f"Unsupported sample type: {sample_type!r}")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_compact_response(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    if value == {"event": "insufficient_evidence"}:
        return value
    if not isinstance(value, dict) or set(value) != {
        "event",
        "target_color",
        "ordinal",
        "target_cell",
    }:
        raise ValueError(f"Unsupported compact response: {value!r}")
    if value["event"] != "ordinal_target_grounded":
        raise ValueError(f"Unsupported event: {value['event']!r}")
    compact_response(
        "full_demo",
        target_color=value["target_color"],
        ordinal=value["ordinal"],
        target_cell=value["target_cell"],
    )
    return value

