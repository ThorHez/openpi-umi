"""Strict raw-data audit for V9 safe balanced EEF recovery episodes."""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import generate_onpolicy_eef_safe_balanced_recovery_dataset_v9 as v9
import numpy as np


def _stats(values: list[float]) -> dict:
    x = np.asarray(values, dtype=np.float64)
    return {
        "min": float(x.min()),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "max": float(x.max()),
    }


def _nearest_sector(offset_xy: np.ndarray) -> int:
    angle = math.atan2(float(offset_xy[1]), float(offset_xy[0])) % (2.0 * math.pi)
    return int(np.rint(angle * v9.NUM_OFFSET_SECTORS / (2.0 * math.pi))) % v9.NUM_OFFSET_SECTORS


def audit(
    root: Path,
    *,
    expected_episodes: int | None = None,
    require_complete_quota: bool = False,
) -> dict:
    root = root.expanduser().resolve()
    episode_dirs = sorted(root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not episode_dirs:
        raise FileNotFoundError(f"No V9 episode directories below {root}")
    if expected_episodes is not None and len(episode_dirs) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, found {len(episode_dirs)}")
    if require_complete_quota and len(episode_dirs) != 1200:
        raise ValueError(f"Complete V9 quota requires 1200 episodes, found {len(episode_dirs)}")

    design_args = SimpleNamespace(prefix_steps="30,36,42")
    slots, stages, bins, sectors = Counter(), Counter(), Counter(), Counter()
    slot_stage = defaultdict(Counter)
    slot_bin = defaultdict(Counter)
    slot_sector = defaultdict(Counter)
    seeds = set()
    offsets, heights, initial_errors, open_steps_all = [], [], [], []
    hard_rows = 0

    for episode_dir in episode_dirs:
        slot_index = int(episode_dir.name.split("_")[-1])
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("accepted_index") != slot_index:
            raise ValueError(f"{episode_dir}: accepted_index does not match directory")
        expected = v9._balanced_design(design_args, slot_index)  # noqa: SLF001
        if metadata.get("dataset_kind") != v9.DATASET_KIND or metadata.get("success") is not True:
            raise ValueError(f"{episode_dir}: invalid V9 kind/success")
        if metadata["final_spatial_slot"] != expected["final_spatial_slot"]:
            raise ValueError(f"{episode_dir}: final spatial-slot quota mismatch")
        if metadata["anchor_stage"] != expected["anchor_stage"]:
            raise ValueError(f"{episode_dir}: anchor-stage design mismatch")
        if metadata["perturbation"]["offset_bin"] != expected["offset_bin"]:
            raise ValueError(f"{episode_dir}: offset-bin design mismatch")
        if int(metadata["offset_sector"]) != int(expected["offset_sector"]):
            raise ValueError(f"{episode_dir}: requested-sector design mismatch")
        if int(metadata["switch"]["prefix_steps"]) != int(expected["prefix_steps"]):
            raise ValueError(f"{episode_dir}: prefix-step design mismatch")

        contract = metadata.get("v9_distribution", {})
        safe = metadata.get("v9_safe_switch_state", {})
        if contract.get("fixed_design_slot_retry_until_filled") is not True:
            raise ValueError(f"{episode_dir}: missing fixed-slot retry contract")
        if int(contract.get("minimum_open_recovery_steps", -1)) != 0:
            raise ValueError(f"{episode_dir}: artificial minimum-open delay is enabled")
        if contract.get("switch_state_is_precontact") is not True:
            raise ValueError(f"{episode_dir}: switch state is not marked pre-contact")
        if contract.get("low_descent_rows_come_from_oracle_suffix") is not True:
            raise ValueError(f"{episode_dir}: low-stage supervision contract is missing")
        if safe.get("path") != "raise_then_lateral_at_clearance_then_vertical_descent":
            raise ValueError(f"{episode_dir}: unsafe hidden switch-state path")
        if safe.get("hidden_actions_supervised") is not False or safe.get("hidden_frames_supervised") is not False:
            raise ValueError(f"{episode_dir}: hidden safe-path data leaked into supervision")
        cup_move_xy = float(safe["hidden_cup_displacement_xy_m"])
        cup_move_z = abs(float(safe["hidden_cup_displacement_z_m"]))
        if max(cup_move_xy, cup_move_z) > v9.MAX_HIDDEN_CUP_DISPLACEMENT_M:
            raise ValueError(f"{episode_dir}: hidden path moved the target cup")

        for section in ("model_prefix", "anchor_positioning", "perturbation"):
            item = metadata[section]
            if item.get("actions_saved_to_training_file") is not False:
                raise ValueError(f"{episode_dir}: hidden {section} action was serialized")
            if item.get("intermediate_frames_saved_to_training_file") is not False:
                raise ValueError(f"{episode_dir}: hidden {section} frame was serialized")
        supervision_contract = metadata["supervision_contract"]
        if supervision_contract.get("action_mask_true_source") != "oracle_only":
            raise ValueError(f"{episode_dir}: non-Oracle action supervision")
        if supervision_contract.get("first_training_pair") != "observation[60] -> oracle_action[61]":
            raise ValueError(f"{episode_dir}: incorrect observation/action alignment")

        with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
            actions = np.asarray(source["controller_actions"], dtype=np.float32)
            action_mask = np.asarray(source["action_mask"], dtype=bool)
            supervision = np.asarray(source["supervision_source"], dtype=np.uint8)
            phases = np.asarray(source["phase_ids"], dtype=np.int64)
        if actions.shape != (v9.EXPECTED_EPISODE_FRAMES, 7):
            raise ValueError(f"{episode_dir}: action shape {actions.shape}")
        if np.any(action_mask != (supervision == v9.legacy.SUPERVISION_ORACLE)):
            raise ValueError(f"{episode_dir}: action mask/source mismatch")
        if np.any(action_mask[:61]) or not np.all(action_mask[61:]):
            raise ValueError(f"{episode_dir}: Oracle mask boundary is not frame 61")

        oracle = metadata["oracle"]
        open_steps = int(oracle["open_steps"])
        grasp_steps = int(oracle["grasp_steps"])
        lift_steps = int(oracle["lift_steps"])
        if int(oracle.get("minimum_open_recovery_steps", -1)) != 0:
            raise ValueError(f"{episode_dir}: Oracle contains forced-open delay")
        if open_steps + grasp_steps + lift_steps != v9.CORRECTION_COMMANDS:
            raise ValueError(f"{episode_dir}: suffix length mismatch")
        trace = oracle["gating_trace"]
        if len(trace) != open_steps:
            raise ValueError(f"{episode_dir}: gating trace length mismatch")
        initial_error = float(trace[0]["pre_command_xy_error_m"])
        if initial_error <= v9.MIN_INITIAL_XY_ERROR_M:
            raise ValueError(f"{episode_dir}: initial XY error is not >5 mm")
        hard_rows += sum(float(row["pre_command_xy_error_m"]) > 0.005 for row in trace)
        if float(oracle["preclose_xy_error_m"]) > float(oracle["close_xy_threshold_m"]) + 1e-9:
            raise ValueError(f"{episode_dir}: closes before XY alignment")
        grasp_start = 61 + open_steps
        lift_start = grasp_start + grasp_steps
        if np.max(actions[61:grasp_start, -1]) > -0.99:
            raise ValueError(f"{episode_dir}: gripper closed during recovery")
        if np.min(actions[grasp_start:, -1]) < 0.99:
            raise ValueError(f"{episode_dir}: gripper reopened after close")
        if not np.all(phases[grasp_start:lift_start] == v9.legacy.PHASE_GRASP):
            raise ValueError(f"{episode_dir}: grasp phase mismatch")
        if not np.all(phases[lift_start:] == v9.legacy.PHASE_LIFT):
            raise ValueError(f"{episode_dir}: lift phase mismatch")

        measured_xy = np.asarray(metadata["perturbation"]["measured_offset_xy_m"], dtype=np.float64)
        measured_sector = _nearest_sector(measured_xy)
        if measured_sector != int(expected["offset_sector"]):
            raise ValueError(f"{episode_dir}: measured offset crossed sector")
        if int(contract["measured_offset_sector"]) != measured_sector:
            raise ValueError(f"{episode_dir}: stored measured sector mismatch")
        offset_m = float(np.linalg.norm(measured_xy))
        low, high = v9.OFFSET_BINS_MM[expected["offset_bin"]]
        if not low - 2.0 <= offset_m * 1_000.0 <= high + 2.0:
            raise ValueError(f"{episode_dir}: measured offset violates bin")
        height_m = float(metadata["anchor_positioning"]["measured_height_above_grasp_m"])
        low, high = v9.ANCHOR_BANDS_MM[expected["anchor_stage"]]
        if not low - 10.0 <= height_m * 1_000.0 <= high + 10.0:
            raise ValueError(f"{episode_dir}: measured height violates band")

        seed = int(metadata["seed"])
        if seed in seeds:
            raise ValueError(f"{episode_dir}: duplicate environment seed {seed}")
        seeds.add(seed)
        spatial_slot = expected["final_spatial_slot"]
        slots[spatial_slot] += 1
        stages[expected["anchor_stage"]] += 1
        bins[expected["offset_bin"]] += 1
        sectors[str(expected["offset_sector"])] += 1
        slot_stage[spatial_slot][expected["anchor_stage"]] += 1
        slot_bin[spatial_slot][expected["offset_bin"]] += 1
        slot_sector[spatial_slot][str(expected["offset_sector"])] += 1
        offsets.append(offset_m)
        heights.append(height_m)
        initial_errors.append(initial_error)
        open_steps_all.append(open_steps)

    manifest_path = root / "generation_manifest.jsonl"
    manifest_rows = []
    if manifest_path.exists():
        manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    physics = [row for row in manifest_rows if row.get("physics_started", True)]
    physics_reasons = Counter(row.get("reason", "unknown") for row in physics)
    contact_rate = physics_reasons["contact_before_oracle"] / max(len(physics), 1)

    if len(episode_dirs) == 1200:
        expected_marginals = {
            "slots": {"left": 400, "middle": 400, "right": 400},
            "stages": {"high": 180, "mid": 420, "late": 600},
            "bins": {"small": 300, "medium": 540, "large": 360},
            "sectors": {str(index): 75 for index in range(v9.NUM_OFFSET_SECTORS)},
        }
        actual_marginals = {
            "slots": dict(slots),
            "stages": dict(stages),
            "bins": dict(bins),
            "sectors": dict(sectors),
        }
        if actual_marginals != expected_marginals:
            raise ValueError(
                f"V9 marginal quota failure: actual={actual_marginals}, expected={expected_marginals}"
            )
        expected_per_slot = {
            "stage": {"high": 60, "mid": 140, "late": 200},
            "bin": {"small": 100, "medium": 180, "large": 120},
            "sector": {str(index): 25 for index in range(v9.NUM_OFFSET_SECTORS)},
        }
        for spatial_slot in v9.FINAL_SPATIAL_SLOTS:
            if dict(slot_stage[spatial_slot]) != expected_per_slot["stage"]:
                raise ValueError(f"V9 {spatial_slot} stage quota failure")
            if dict(slot_bin[spatial_slot]) != expected_per_slot["bin"]:
                raise ValueError(f"V9 {spatial_slot} radius quota failure")
            if dict(slot_sector[spatial_slot]) != expected_per_slot["sector"]:
                raise ValueError(f"V9 {spatial_slot} sector quota failure")
        if contact_rate > 0.20:
            raise ValueError(
                f"V9 safe path still rejects {contact_rate:.1%} of physics attempts for contact"
            )

    return {
        "ok": True,
        "quota_complete": len(episode_dirs) == 1200,
        "episodes": len(episode_dirs),
        "unique_episode_seeds": len(seeds),
        "final_spatial_slots": dict(sorted(slots.items())),
        "anchor_stages": dict(sorted(stages.items())),
        "offset_bins": dict(sorted(bins.items())),
        "requested_and_measured_offset_sectors": dict(sorted(sectors.items())),
        "per_slot_stage": {key: dict(sorted(value.items())) for key, value in slot_stage.items()},
        "per_slot_bin": {key: dict(sorted(value.items())) for key, value in slot_bin.items()},
        "per_slot_sector": {key: dict(sorted(value.items())) for key, value in slot_sector.items()},
        "measured_offset_m": _stats(offsets),
        "measured_anchor_height_m": _stats(heights),
        "initial_xy_error_m": _stats(initial_errors),
        "unique_gt5mm_recovery_rows": hard_rows,
        "open_recovery_steps": _stats(open_steps_all),
        "physics_attempts": len(physics),
        "physics_reasons": dict(physics_reasons),
        "contact_rejection_rate": contact_rate,
        "hidden_actions_supervised": False,
        "forced_open_delay": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--require-complete-quota", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.root,
                expected_episodes=args.expected_episodes,
                require_complete_quota=args.require_complete_quota,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
