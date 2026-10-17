"""Strictly audit raw V6 low-stage gated absolute-EEF episodes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import generate_onpolicy_eef_low_stage_gated_dataset_v6 as v6
import numpy as np


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def audit(root: Path, *, expected_episodes: int | None = None) -> dict:
    episode_dirs = sorted(root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {root}")
    if expected_episodes is not None and len(episode_dirs) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, found {len(episode_dirs)}")

    stages = Counter()
    spatial_slots = Counter()
    offset_bins = Counter()
    sectors = Counter()
    prefixes = Counter()
    offsets = []
    anchor_heights = []
    open_steps_all = []
    preclose_xy = []
    preclose_z = []
    gating_modes = Counter()

    for episode_dir in episode_dirs:
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("dataset_kind") != v6.DATASET_KIND:
            raise ValueError(f"{episode_dir}: unexpected dataset kind")
        if metadata.get("success") is not True:
            raise ValueError(f"{episode_dir}: unsuccessful Oracle suffix")
        contract = metadata["supervision_contract"]
        if contract.get("model_anchor_or_perturb_actions_supervised") is not False:
            raise ValueError(f"{episode_dir}: unsaved setup entered supervision")
        if contract.get("action_mask_true_source") != "oracle_only":
            raise ValueError(f"{episode_dir}: non-Oracle action supervision")
        if contract.get("first_training_pair") != "observation[60] -> oracle_action[61]":
            raise ValueError(f"{episode_dir}: first-pair contract mismatch")
        for section in ("model_prefix", "anchor_positioning", "perturbation"):
            payload = metadata[section]
            if payload.get("actions_saved_to_training_file") is not False:
                raise ValueError(f"{episode_dir}: {section} actions were stored")
            if payload.get("intermediate_frames_saved_to_training_file") is not False:
                raise ValueError(f"{episode_dir}: {section} frames were stored")

        with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
            actions = np.asarray(source["actions"], dtype=np.float32)
            action_mask = np.asarray(source["action_mask"], dtype=bool)
            supervision = np.asarray(source["supervision_source"], dtype=np.uint8)
            phases = np.asarray(source["phase_ids"], dtype=np.int64)
            eef_pos = np.asarray(source["eef_pos"], dtype=np.float32)
        if actions.shape != (v6.EXPECTED_EPISODE_FRAMES, 7):
            raise ValueError(f"{episode_dir}: action shape {actions.shape}")
        if any(array.shape[0] != v6.EXPECTED_EPISODE_FRAMES for array in (action_mask, supervision, phases, eef_pos)):
            raise ValueError(f"{episode_dir}: inconsistent row counts")
        if np.any(action_mask != (supervision == v6.legacy.SUPERVISION_ORACLE)):
            raise ValueError(f"{episode_dir}: mask/source mismatch")
        if np.any(action_mask[:61]) or not np.all(action_mask[61:]):
            raise ValueError(f"{episode_dir}: Oracle-only mask boundary is not 61")

        stage = metadata["anchor_stage"]
        if stage not in v6.ANCHOR_BANDS_MM:
            raise ValueError(f"{episode_dir}: unknown anchor stage {stage}")
        anchor_height = float(metadata["anchor_positioning"]["measured_height_above_grasp_m"])
        low, high = v6.ANCHOR_BANDS_MM[stage]
        tolerance = float(metadata["command_args"].get("anchor_height_tolerance_mm", 10.0))
        if not low - tolerance <= anchor_height * 1_000.0 <= high + tolerance:
            raise ValueError(f"{episode_dir}: anchor height outside stage")

        perturbation = metadata["perturbation"]
        offset_bin = perturbation["offset_bin"]
        offset = float(perturbation["measured_offset_m"])
        low, high = v6.OFFSET_BINS_MM[offset_bin]
        if not low - 3.0 <= offset * 1_000.0 <= high + 3.0:
            raise ValueError(f"{episode_dir}: measured offset outside bin")

        oracle = metadata["oracle"]
        open_steps = int(oracle["open_steps"])
        grasp_steps = int(oracle["grasp_steps"])
        lift_steps = int(oracle["lift_steps"])
        if open_steps + grasp_steps + lift_steps != v6.CORRECTION_COMMANDS:
            raise ValueError(f"{episode_dir}: Oracle suffix length mismatch")
        open_slice = slice(61, 61 + open_steps)
        grasp_slice = slice(61 + open_steps, 61 + open_steps + grasp_steps)
        lift_slice = slice(61 + open_steps + grasp_steps, v6.EXPECTED_EPISODE_FRAMES)
        if np.max(actions[open_slice, -1]) > -0.99:
            raise ValueError(f"{episode_dir}: gripper closes before gate")
        if np.min(actions[grasp_slice, -1]) < 0.99 or np.min(actions[lift_slice, -1]) < 0.99:
            raise ValueError(f"{episode_dir}: gripper opens after gated close")
        if not np.all(phases[grasp_slice] == v6.legacy.PHASE_GRASP):
            raise ValueError(f"{episode_dir}: grasp phase mismatch")
        if not np.all(phases[lift_slice] == v6.legacy.PHASE_LIFT):
            raise ValueError(f"{episode_dir}: lift phase mismatch")

        trace = oracle["gating_trace"]
        if len(trace) != open_steps:
            raise ValueError(f"{episode_dir}: gating trace length mismatch")
        hold_threshold_mm = float(oracle["hold_z_above_xy_error_m"]) * 1_000.0
        aligned_threshold_mm = float(oracle["aligned_xy_threshold_m"]) * 1_000.0
        for index, metric in enumerate(trace):
            mode = metric["stage"]
            pre_error_mm = float(metric["pre_command_xy_error_m"]) * 1_000.0
            command_index = 61 + index
            pre_z = float(eef_pos[command_index - 1, 2])
            target_z = float(actions[command_index, 2])
            if pre_error_mm > hold_threshold_mm:
                if mode != "hold_z_recenter" or abs(target_z - pre_z) > 1e-4:
                    raise ValueError(f"{episode_dir}: large-error Z hold violated")
            elif pre_error_mm > aligned_threshold_mm:
                if mode != "slow_descent_recenter":
                    raise ValueError(f"{episode_dir}: moderate-error slow mode violated")
            elif mode != "aligned_descent":
                raise ValueError(f"{episode_dir}: aligned descent mode violated")
            gating_modes[mode] += 1

        if float(oracle["preclose_xy_error_m"]) > float(oracle["close_xy_threshold_m"]) + 1e-9:
            raise ValueError(f"{episode_dir}: closes while XY misaligned")
        if abs(float(oracle["preclose_z_error_m"])) > float(oracle["close_z_threshold_m"]) + 1e-9:
            raise ValueError(f"{episode_dir}: closes at wrong Z")

        stages[stage] += 1
        spatial_slots[metadata["final_spatial_slot"]] += 1
        offset_bins[offset_bin] += 1
        sectors[str(metadata["offset_sector"])] += 1
        prefixes[str(metadata["switch"]["prefix_steps"])] += 1
        offsets.append(offset)
        anchor_heights.append(anchor_height)
        open_steps_all.append(open_steps)
        preclose_xy.append(float(oracle["preclose_xy_error_m"]))
        preclose_z.append(abs(float(oracle["preclose_z_error_m"])))

    if len(episode_dirs) == 1200:
        expected_stages = {"high": 120, "mid": 240, "late": 840}
        expected_spatial = {"left": 400, "middle": 400, "right": 400}
        expected_offsets = {"small": 360, "medium": 540, "large": 300}
        expected_sectors = {str(index): 150 for index in range(8)}
        if dict(stages) != expected_stages:
            raise ValueError(f"Stage balance mismatch: {dict(stages)}")
        if dict(spatial_slots) != expected_spatial:
            raise ValueError(f"Spatial balance mismatch: {dict(spatial_slots)}")
        if dict(offset_bins) != expected_offsets:
            raise ValueError(f"Offset balance mismatch: {dict(offset_bins)}")
        if dict(sectors) != expected_sectors:
            raise ValueError(f"Direction balance mismatch: {dict(sectors)}")

    return {
        "ok": True,
        "episodes": len(episode_dirs),
        "anchor_stages": dict(sorted(stages.items())),
        "final_spatial_slots": dict(sorted(spatial_slots.items())),
        "offset_bins": dict(sorted(offset_bins.items())),
        "offset_sectors": dict(sorted(sectors.items())),
        "prefix_steps": dict(sorted(prefixes.items())),
        "gating_modes": dict(sorted(gating_modes.items())),
        "measured_offset_m": _stats(offsets),
        "measured_anchor_height_m": _stats(anchor_heights),
        "open_steps": _stats(open_steps_all),
        "preclose_xy_error_m": _stats(preclose_xy),
        "preclose_abs_z_error_m": _stats(preclose_z),
        "oracle_only_mask_boundary": 61,
        "premature_close_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()
    result = audit(args.root.expanduser().resolve(), expected_episodes=args.expected_episodes)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
