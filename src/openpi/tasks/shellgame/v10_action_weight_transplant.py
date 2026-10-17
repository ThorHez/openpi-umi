"""Strictly transplant the Pi0 action branch from a V10 checkpoint.

The Qwen-distilled ShellGame policy and the old tracker policy do not share
their memory modules.  They do share Pi0.5's action expert and the four
action/time projections.  This helper replaces only those common leaves so a
closed-loop run can isolate action quality without changing the semantic
memory conditioner or waypoint decoder.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import re
from typing import Any

import flax.traverse_util

_ACTION_EXPERT = re.compile(r".*PaliGemma/llm/.*_1.*")
_ACTION_PROJECTIONS = re.compile(r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*")


def is_v10_action_path(path: tuple[Any, ...] | str) -> bool:
    """Return whether ``path`` belongs to the trainable V10 action branch."""
    name = path if isinstance(path, str) else "/".join(str(part) for part in path)
    return bool(_ACTION_EXPERT.fullmatch(name) or _ACTION_PROJECTIONS.fullmatch(name))


@dataclasses.dataclass(frozen=True)
class TransplantReport:
    selected_leaves: int
    selected_elements: int
    action_expert_leaves: int
    projection_leaves: int
    selected_paths: tuple[str, ...]


def transplant_v10_action_params(
    current_params: Mapping[str, Any],
    v10_params: Mapping[str, Any],
) -> tuple[dict[str, Any], TransplantReport]:
    """Replace only shape-compatible V10 action leaves in ``current_params``.

    Raises instead of silently falling back when a selected leaf is absent or
    shape-incompatible.  Extra old-tracker leaves are deliberately ignored.
    """
    flat_current = flax.traverse_util.flatten_dict(dict(current_params))
    flat_v10 = flax.traverse_util.flatten_dict(dict(v10_params))
    selected = [path for path in flat_current if is_v10_action_path(path)]
    if not selected:
        raise ValueError("V10 action selector matched no leaves in the current model")

    missing = [path for path in selected if path not in flat_v10]
    if missing:
        names = ["/".join(str(part) for part in path) for path in missing[:8]]
        raise ValueError(f"V10 checkpoint is missing selected action leaves: {names}")

    mismatched = []
    for path in selected:
        current_shape = tuple(getattr(flat_current[path], "shape", ()))
        v10_shape = tuple(getattr(flat_v10[path], "shape", ()))
        if current_shape != v10_shape:
            mismatched.append(("/".join(str(part) for part in path), current_shape, v10_shape))
    if mismatched:
        raise ValueError(f"V10 action leaves have incompatible shapes: {mismatched[:8]}")

    merged = dict(flat_current)
    for path in selected:
        merged[path] = flat_v10[path]

    names = tuple(sorted("/".join(str(part) for part in path) for path in selected))
    report = TransplantReport(
        selected_leaves=len(selected),
        selected_elements=sum(int(getattr(flat_current[path], "size", 0)) for path in selected),
        action_expert_leaves=sum(bool(_ACTION_EXPERT.fullmatch(name)) for name in names),
        projection_leaves=sum(bool(_ACTION_PROJECTIONS.fullmatch(name)) for name in names),
        selected_paths=names,
    )
    return flax.traverse_util.unflatten_dict(merged), report
