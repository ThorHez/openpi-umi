"""Audit converted replan-compatible absolute-EEF correction data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


EXPECTED_FRAMES = 155
FIRST_CONTEXT_FRAME = 60
EXPECTED_MASKED_ROWS = 94


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--min-entry-descent-mm", type=float, default=3.0)
    parser.add_argument("--max-recenter-z-drift-mm", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    errors: list[str] = []
    entry_descent_mm: list[float] = []
    recenter_xy_mm: list[float] = []
    max_rotation_vector_norm = 0.0

    if len(files) != args.expected_episodes:
        errors.append(
            f"episode count: got {len(files)}, expected {args.expected_episodes}"
        )

    for path in files:
        data = pq.read_table(path).to_pydict()
        frame = np.asarray(data["frame_index"], dtype=np.int64)
        actions = np.asarray(data["actions"], dtype=np.float64)
        state_pos = np.asarray(
            data["observation.robot0_eef_pos"], dtype=np.float64
        )[:, 0]
        action_mask = np.asarray(data["action_mask"], dtype=bool)
        prefix = path.name

        if len(frame) != EXPECTED_FRAMES or not np.array_equal(
            frame, np.arange(EXPECTED_FRAMES)
        ):
            errors.append(f"{prefix}: frames are not exactly 0..154")
            continue
        if actions.shape != (EXPECTED_FRAMES, 16, 7):
            errors.append(f"{prefix}: unexpected action shape {actions.shape}")
            continue
        if not np.isfinite(actions).all() or not np.isfinite(state_pos).all():
            errors.append(f"{prefix}: non-finite action or state")
        if int(action_mask.sum()) != EXPECTED_MASKED_ROWS:
            errors.append(
                f"{prefix}: action_mask true count {action_mask.sum()} != "
                f"{EXPECTED_MASKED_ROWS}"
            )
        if not action_mask[60:154].all() or action_mask[:60].any() or action_mask[154]:
            errors.append(f"{prefix}: action_mask must be true only on frames 60..153")

        first_three = actions[FIRST_CONTEXT_FRAME, :3]
        dz_mm = (first_three[:, 2] - state_pos[FIRST_CONTEXT_FRAME, 2]) * 1000.0
        xy_mm = np.linalg.norm(
            first_three[:2, :2] - state_pos[FIRST_CONTEXT_FRAME, :2], axis=1
        ) * 1000.0
        if np.max(np.abs(dz_mm[:2])) > args.max_recenter_z_drift_mm:
            errors.append(f"{prefix}: first two commands change Z by {dz_mm[:2].tolist()} mm")
        if dz_mm[2] > -args.min_entry_descent_mm:
            errors.append(f"{prefix}: third command descends only {-dz_mm[2]:.3f} mm")
        entry_descent_mm.append(float(-dz_mm[2]))
        recenter_xy_mm.extend(xy_mm.tolist())
        max_rotation_vector_norm = max(
            max_rotation_vector_norm,
            float(np.linalg.norm(actions[:, :, 3:6], axis=2).max()),
        )

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    if info.get("total_episodes") != args.expected_episodes:
        errors.append(
            f"meta total_episodes={info.get('total_episodes')} != {args.expected_episodes}"
        )
    expected_total_frames = args.expected_episodes * EXPECTED_FRAMES
    if info.get("total_frames") != expected_total_frames:
        errors.append(
            f"meta total_frames={info.get('total_frames')} != {expected_total_frames}"
        )

    def stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(array.min()),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "max": float(array.max()),
        }

    result = {
        "ok": not errors,
        "dataset": str(root),
        "episodes": len(files),
        "frames_per_episode": EXPECTED_FRAMES,
        "action_mask_true_frames": "60..153",
        "closed_loop_contract": {
            "replan_steps": 3,
            "first_two_commands": "pure_xy_recenter",
            "third_command": "strict_descent",
            "entry_descent_mm": stats(entry_descent_mm),
            "recenter_xy_command_mm": stats(recenter_xy_mm),
        },
        "max_action_rotation_vector_norm": max_rotation_vector_norm,
        "errors": errors,
    }
    (root / "v2_data_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
