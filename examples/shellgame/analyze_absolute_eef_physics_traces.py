"""Summarize absolute-EEF ShellGame physics traces for grasp diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _target_contact(row: dict, target: str) -> bool:
    target_body = f"{target}_cup_root"
    for contact in row.get("contacts", []):
        body1 = str(contact.get("body1", ""))
        body2 = str(contact.get("body2", ""))
        if target_body not in (body1, body2):
            continue
        other = body2 if body1 == target_body else body1
        if other.startswith(("robot0", "gripper0")):
            return True
    return False


def _snapshot_metrics(row: dict, target: str) -> dict:
    cup = np.asarray(row["cup_pos"][target], dtype=np.float64)
    actual = np.asarray(row["eef_pos"], dtype=np.float64)
    command = np.asarray(row["env_action"][:3], dtype=np.float64)
    qvel = np.asarray(row["cup_qvel"][target], dtype=np.float64)
    return {
        "step": int(row["step"]),
        "actual_xy_error_mm": float(np.linalg.norm(actual[:2] - cup[:2]) * 1_000.0),
        "command_xy_error_mm": float(np.linalg.norm(command[:2] - cup[:2]) * 1_000.0),
        "actual_z_above_cup_mm": float((actual[2] - cup[2]) * 1_000.0),
        "command_z_above_cup_mm": float((command[2] - cup[2]) * 1_000.0),
        "gripper_action": float(row["gripper_action"]),
        "cup_angular_speed_rad_s": float(np.linalg.norm(qvel[3:])),
    }


def _first(rows: list[dict], predicate) -> dict | None:
    return next((row for row in rows if predicate(row)), None)


def analyze_trace(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["trace"]
    target = payload["target_cup"]
    if not rows:
        raise ValueError(f"Empty trace: {path}")

    first_step = int(rows[0]["step"])
    initial_command_z = float(rows[0]["env_action"][2])
    first_descent = _first(
        rows,
        lambda row: float(row["env_action"][2]) < initial_command_z - 0.005,
    )
    first_close = _first(rows, lambda row: float(row["gripper_action"]) > 0.0)
    first_contact = _first(rows, lambda row: _target_contact(row, target))
    before_descent = rows if first_descent is None else [
        row for row in rows if int(row["step"]) < int(first_descent["step"])
    ]
    first_three = [row for row in rows if int(row["step"]) < first_step + 3]
    first_three_command_z = np.asarray(
        [float(row["env_action"][2]) for row in first_three], dtype=np.float64
    )
    guarded_steps = [
        row
        for row in rows
        if row.get("target10") is not None
        and abs(float(row["env_action"][2]) - float(row["target10"][2])) > 1e-6
    ]

    actual_xy = [
        _snapshot_metrics(row, target)["actual_xy_error_mm"] for row in rows
    ]
    command_xy = [
        _snapshot_metrics(row, target)["command_xy_error_mm"] for row in rows
    ]
    before_descent_xy = [
        _snapshot_metrics(row, target)["actual_xy_error_mm"] for row in before_descent
    ]
    target_initial_z = float(rows[0]["cup_pos"][target][2])
    target_max_lift_mm = max(
        (float(row["cup_pos"][target][2]) - target_initial_z) * 1_000.0 for row in rows
    )
    return {
        "trial": int(payload["trial"]),
        "episode_seed": int(payload["episode_seed"]),
        "target": target,
        "success": bool(payload["success"]),
        "selection_correct": bool(payload["cup_selection_correct"]),
        "trace_first_step": first_step,
        "trace_length": len(rows),
        "guarded_steps": len(guarded_steps),
        "guarded_step_rate": len(guarded_steps) / len(rows),
        "first_three_command_z_drift_mm": (
            float(np.ptp(first_three_command_z) * 1_000.0)
            if len(first_three_command_z)
            else None
        ),
        "initial": _snapshot_metrics(rows[0], target),
        "first_descent": None if first_descent is None else _snapshot_metrics(first_descent, target),
        "first_close": None if first_close is None else _snapshot_metrics(first_close, target),
        "first_contact": None if first_contact is None else _snapshot_metrics(first_contact, target),
        "min_actual_xy_before_descent_mm": None if not before_descent_xy else float(min(before_descent_xy)),
        "min_actual_xy_mm": float(min(actual_xy)),
        "min_command_xy_mm": float(min(command_xy)),
        "target_max_lift_mm": float(target_max_lift_mm),
        "path": str(path),
    }


def _values(rows: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value: object = row
        for part in key.split("."):
            if value is None or not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if value is not None:
            values.append(float(value))
    return values


def _stats(rows: list[dict], key: str) -> dict | None:
    values = _values(rows, key)
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def summarize(rows: list[dict]) -> dict:
    metrics = (
        "first_three_command_z_drift_mm",
        "first_descent.actual_xy_error_mm",
        "first_descent.command_xy_error_mm",
        "first_close.actual_xy_error_mm",
        "first_close.actual_z_above_cup_mm",
        "first_close.cup_angular_speed_rad_s",
        "first_contact.actual_xy_error_mm",
        "first_contact.actual_z_above_cup_mm",
        "first_contact.cup_angular_speed_rad_s",
        "min_actual_xy_before_descent_mm",
        "min_actual_xy_mm",
        "target_max_lift_mm",
        "guarded_steps",
        "guarded_step_rate",
    )
    success = [row for row in rows if row["success"]]
    failure = [row for row in rows if not row["success"]]
    return {
        "episodes": len(rows),
        "successes": len(success),
        "success_rate": len(success) / max(len(rows), 1),
        "selection_accuracy": sum(row["selection_correct"] for row in rows) / max(len(rows), 1),
        "all": {key: _stats(rows, key) for key in metrics},
        "success": {key: _stats(success, key) for key in metrics},
        "failure": {key: _stats(failure, key) for key in metrics},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = {}
    for trace_dir in args.trace_dirs:
        paths = sorted(trace_dir.glob("trial_*.json"))
        rows = [analyze_trace(path) for path in paths]
        reports[str(trace_dir)] = {"summary": summarize(rows), "episodes": rows}
    text = json.dumps(reports, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
