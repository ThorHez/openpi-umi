"""Strict audit for raw V8 sustained low-height EEF recovery episodes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import generate_onpolicy_eef_sustained_recovery_dataset_v8 as v8
import numpy as np


def _stats(values: list[float]) -> dict:
    x = np.asarray(values, dtype=np.float64)
    return {
        "min": float(x.min()),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "max": float(x.max()),
    }


def audit(root: Path, *, expected_episodes: int | None = None) -> dict:
    episode_dirs = sorted(root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not episode_dirs:
        raise FileNotFoundError(f"No V8 episode directories below {root}")
    if expected_episodes is not None and len(episode_dirs) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, found {len(episode_dirs)}")

    slots, stages, bins, sectors = Counter(), Counter(), Counter(), Counter()
    offsets, heights, open_steps_all, preclose_xy = [], [], [], []
    hard_rows = 0
    hard_episodes = 0
    all_episode_seeds = set()

    for episode_dir in episode_dirs:
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("dataset_kind") != v8.DATASET_KIND or metadata.get("success") is not True:
            raise ValueError(f"{episode_dir}: invalid V8 kind/success")
        if metadata.get("v8_distribution", {}).get("row_replication_allowed") is not False:
            raise ValueError(f"{episode_dir}: missing V8 no-replication contract")
        all_episode_seeds.add(int(metadata["seed"]))
        for section in ("model_prefix", "anchor_positioning", "perturbation"):
            item = metadata[section]
            if item.get("actions_saved_to_training_file") is not False:
                raise ValueError(f"{episode_dir}: hidden {section} action was serialized")
            if item.get("intermediate_frames_saved_to_training_file") is not False:
                raise ValueError(f"{episode_dir}: hidden {section} frame was serialized")
        contract = metadata["supervision_contract"]
        if contract.get("action_mask_true_source") != "oracle_only":
            raise ValueError(f"{episode_dir}: action mask is not Oracle-only")
        if contract.get("first_training_pair") != "observation[60] -> oracle_action[61]":
            raise ValueError(f"{episode_dir}: first aligned pair is wrong")

        with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
            actions = np.asarray(source["actions"], dtype=np.float32)
            action_mask = np.asarray(source["action_mask"], dtype=bool)
            supervision = np.asarray(source["supervision_source"], dtype=np.uint8)
            phases = np.asarray(source["phase_ids"], dtype=np.int64)
        if actions.shape != (v8.EXPECTED_EPISODE_FRAMES, 7):
            raise ValueError(f"{episode_dir}: action shape {actions.shape}")
        if any(x.shape[0] != v8.EXPECTED_EPISODE_FRAMES for x in (action_mask, supervision, phases)):
            raise ValueError(f"{episode_dir}: inconsistent row count")
        if np.any(action_mask != (supervision == v8.legacy.SUPERVISION_ORACLE)):
            raise ValueError(f"{episode_dir}: action mask/source mismatch")
        if np.any(action_mask[:61]) or not np.all(action_mask[61:]):
            raise ValueError(f"{episode_dir}: Oracle mask boundary is not frame 61")

        oracle = metadata["oracle"]
        open_steps = int(oracle["open_steps"])
        grasp_steps = int(oracle["grasp_steps"])
        lift_steps = int(oracle["lift_steps"])
        if open_steps < v8.MIN_UNIQUE_RECOVERY_ROWS:
            raise ValueError(f"{episode_dir}: only {open_steps} recovery rows")
        if open_steps + grasp_steps + lift_steps != v8.CORRECTION_COMMANDS:
            raise ValueError(f"{episode_dir}: suffix length mismatch")
        grasp_start = 61 + open_steps
        lift_start = grasp_start + grasp_steps
        if np.max(actions[61:grasp_start, -1]) > -0.99:
            raise ValueError(f"{episode_dir}: gripper closed before alignment")
        if np.min(actions[grasp_start:, -1]) < 0.99:
            raise ValueError(f"{episode_dir}: gripper reopened after close")
        if not np.all(phases[grasp_start:lift_start] == v8.legacy.PHASE_GRASP):
            raise ValueError(f"{episode_dir}: grasp phase mismatch")
        if not np.all(phases[lift_start:] == v8.legacy.PHASE_LIFT):
            raise ValueError(f"{episode_dir}: lift phase mismatch")
        trace = oracle["gating_trace"]
        if len(trace) != open_steps:
            raise ValueError(f"{episode_dir}: gating trace length mismatch")
        episode_hard = sum(float(row["pre_command_xy_error_m"]) > 0.005 for row in trace)
        hard_rows += episode_hard
        hard_episodes += episode_hard > 0
        if float(oracle["preclose_xy_error_m"]) > float(oracle["close_xy_threshold_m"]) + 1e-9:
            raise ValueError(f"{episode_dir}: closes while laterally misaligned")

        stage = metadata["anchor_stage"]
        offset_bin = metadata["perturbation"]["offset_bin"]
        offset_m = float(metadata["perturbation"]["measured_offset_m"])
        height_m = float(metadata["anchor_positioning"]["measured_height_above_grasp_m"])
        low, high = v8.OFFSET_BINS_MM[offset_bin]
        if not low - 2.0 <= offset_m * 1_000.0 <= high + 2.0 or offset_m < 0.005:
            raise ValueError(f"{episode_dir}: measured offset violates V8 bin")
        low, high = v8.ANCHOR_BANDS_MM[stage]
        if not low - 10.0 <= height_m * 1_000.0 <= high + 10.0:
            raise ValueError(f"{episode_dir}: measured height violates V8 band")
        slots[metadata["final_spatial_slot"]] += 1
        stages[stage] += 1
        bins[offset_bin] += 1
        sectors[str(metadata["offset_sector"])] += 1
        offsets.append(offset_m)
        heights.append(height_m)
        open_steps_all.append(open_steps)
        preclose_xy.append(float(oracle["preclose_xy_error_m"]))

    if len(all_episode_seeds) != len(episode_dirs):
        raise ValueError("V8 contains duplicate environment seeds")
    if len(episode_dirs) == 1200:
        expected = {
            "slots": {"left": 400, "middle": 400, "right": 400},
            "stages": {"high": 180, "mid": 420, "late": 600},
            "bins": {"small": 300, "medium": 540, "large": 360},
            "sectors": {str(i): 75 for i in range(v8.NUM_OFFSET_SECTORS)},
        }
        actual = {
            "slots": dict(slots),
            "stages": dict(stages),
            "bins": dict(bins),
            "sectors": dict(sectors),
        }
        if actual != expected:
            raise ValueError(f"V8 balance failure: actual={actual}, expected={expected}")

    return {
        "ok": True,
        "episodes": len(episode_dirs),
        "unique_episode_seeds": len(all_episode_seeds),
        "final_spatial_slots": dict(sorted(slots.items())),
        "anchor_stages": dict(sorted(stages.items())),
        "offset_bins": dict(sorted(bins.items())),
        "offset_sectors": dict(sorted(sectors.items())),
        "measured_offset_m": _stats(offsets),
        "measured_anchor_height_m": _stats(heights),
        "open_recovery_steps": _stats(open_steps_all),
        "preclose_xy_error_m": _stats(preclose_xy),
        "episodes_with_gt5mm_recovery": hard_episodes,
        "unique_gt5mm_recovery_rows": hard_rows,
        "hidden_actions_supervised": False,
        "row_replication_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.root.expanduser().resolve(), expected_episodes=args.expected_episodes),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
