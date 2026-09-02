"""Generate quota-complete, collision-safe absolute-EEF recovery data.

V9 keeps the validated V6 Oracle suffix and fixed history, but fixes the V8
selection bias.  Every design slot is retried until it succeeds, and the
hidden switch-state motion follows a safe three-stage path:

1. raise at the current XY;
2. translate to the requested offset at clearance height; and
3. descend vertically at that offset.

The learned-policy prefix and all hidden positioning commands remain excluded
from supervision.  Unlike V8, V9 never delays grasp merely to manufacture a
minimum number of recovery rows.  Accepted samples must preserve the requested
measured offset sector, start with >5 mm real XY error, move the cup by at most
2 mm during hidden positioning, and pass the unchanged Oracle success test.
"""

# Private V6/V8 helpers are reused to preserve the audited serialization and
# action-alignment contracts.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import math

import generate_onpolicy_eef_low_stage_gated_dataset_v6 as _v6
import generate_onpolicy_eef_sustained_recovery_dataset_v8 as _v8
import numpy as np

DATASET_KIND = "onpolicy_eef_safe_balanced_recovery_v9"
EXPECTED_EPISODE_FRAMES = _v6.EXPECTED_EPISODE_FRAMES
CORRECTION_COMMANDS = _v6.CORRECTION_COMMANDS
DEFAULT_PREFIX_STEPS = _v6.DEFAULT_PREFIX_STEPS
FINAL_SPATIAL_SLOTS = _v6.FINAL_SPATIAL_SLOTS
# The original V8 Cartesian product included physically impossible states:
# e.g. even a 5--12 mm side-cup error only 15--35 mm above grasp height can put
# a finger through the cup.  The perturbation switch must therefore be before
# contact.  The Oracle suffix still records the complete later descent, grasp,
# and lift, so training rows continue to cover the genuinely low stages without
# manufacturing an intersecting initial state.
ANCHOR_BANDS_MM = {
    "high": (80.0, 110.0),
    "mid": (60.0, 80.0),
    "late": (45.0, 60.0),
}
ANCHOR_STAGE_CYCLE = ("high",) * 15 + ("mid",) * 35 + ("late",) * 50
OFFSET_BINS_MM = _v8.OFFSET_BINS_MM
OFFSET_BIN_CYCLE = _v8.OFFSET_BIN_CYCLE
NUM_OFFSET_SECTORS = _v8.NUM_OFFSET_SECTORS

MIN_INITIAL_XY_ERROR_M = 0.005
MAX_HIDDEN_CUP_DISPLACEMENT_M = 0.002
SAFE_CLEARANCE_MM = 180.0
SAFE_RAISE_STEPS = 10
SAFE_LATERAL_STEPS = 10
SAFE_FEEDBACK_STEPS = 6

legacy = _v6.legacy
VisualCorruptionError = _v6.VisualCorruptionError
_VisualGuard = _v6._VisualGuard
_ORIGINAL_ATTEMPT = _v6._attempt
_ORIGINAL_VALIDATE_ARGS = _v6._validate_args


def parse_args() -> argparse.Namespace:
    args = _v6.parse_args()
    if args.dataset_seed == 260819:
        args.dataset_seed = 260825
    if args.offset_bin_tolerance_mm == 3.0:
        args.offset_bin_tolerance_mm = 2.0
    args.safe_offset_approach = True
    args.safe_clearance_mm = SAFE_CLEARANCE_MM
    args.safe_raise_steps = SAFE_RAISE_STEPS
    args.safe_lateral_steps = SAFE_LATERAL_STEPS
    args.safe_feedback_steps = SAFE_FEEDBACK_STEPS
    args.combined_anchor_perturb = False
    args.feedback_perturb = False
    args.minimum_open_recovery_steps = 0
    return args


def _prefix_steps(args: argparse.Namespace) -> tuple[int, ...]:
    return _v6._prefix_steps(args)


def _balanced_design(args: argparse.Namespace, accepted_index: int) -> dict:
    slot_index = accepted_index % len(FINAL_SPATIAL_SLOTS)
    within_slot = accepted_index // len(FINAL_SPATIAL_SLOTS)
    prefixes = _prefix_steps(args)
    # Coprime permutations decorrelate stage, radius, direction, prefix length,
    # and spatial slot while preserving exact per-slot marginals over 400 rows.
    stage_index = (within_slot * 37 + slot_index * 23) % 100
    offset_index = (within_slot * 29 + slot_index * 17) % 100
    return {
        "final_spatial_slot": FINAL_SPATIAL_SLOTS[slot_index],
        "within_spatial_slot_index": within_slot,
        "prefix_steps": prefixes[(within_slot + slot_index) % len(prefixes)],
        "anchor_stage": ANCHOR_STAGE_CYCLE[stage_index],
        "offset_bin": OFFSET_BIN_CYCLE[offset_index],
        "offset_sector": (within_slot * 5 + 3 * slot_index) % NUM_OFFSET_SECTORS,
    }


def _episode_design(
    args: argparse.Namespace,
    *,
    attempt_index: int,
    accepted_index: int,
):
    design = _balanced_design(args, accepted_index)
    rng = np.random.default_rng(
        np.random.SeedSequence([args.dataset_seed, attempt_index, accepted_index, 829])
    )
    anchor_low, anchor_high = ANCHOR_BANDS_MM[design["anchor_stage"]]
    anchor_height_m = float(rng.uniform(anchor_low, anchor_high) / 1_000.0)

    offset_low, offset_high = OFFSET_BINS_MM[design["offset_bin"]]
    angle = 2.0 * math.pi * design["offset_sector"] / NUM_OFFSET_SECTORS
    angle += float(rng.uniform(-math.pi / 32.0, math.pi / 32.0))
    radius_m = float(rng.uniform(offset_low, offset_high) / 1_000.0)
    perturb_xy = radius_m * np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)

    jitter_radius = float(rng.uniform(0.0, args.descent_jitter_mm / 1_000.0))
    jitter_angle = float(rng.uniform(-math.pi, math.pi))
    descent_jitter_xy = jitter_radius * np.asarray(
        [math.cos(jitter_angle), math.sin(jitter_angle)], dtype=np.float64
    )
    return design, anchor_height_m, perturb_xy, descent_jitter_xy


def _predicted_final_spatial_slot(shell, episode_seed: int, initial: str) -> tuple[str, list]:
    swaps = shell.sample_swaps(np.random.default_rng(episode_seed), 3)
    cup_slots = {name: name for name in shell.CUP_NAMES}
    for slot_a, slot_b in swaps:
        cup_a = shell.cup_in_slot(cup_slots, slot_a)
        cup_b = shell.cup_in_slot(cup_slots, slot_b)
        cup_slots[cup_a], cup_slots[cup_b] = slot_b, slot_a
    return cup_slots[initial], swaps


def _configure_v6_backend() -> None:
    _v6.DATASET_KIND = DATASET_KIND
    _v6.ANCHOR_BANDS_MM = ANCHOR_BANDS_MM
    _v6.ANCHOR_STAGE_CYCLE = ANCHOR_STAGE_CYCLE
    _v6.OFFSET_BINS_MM = OFFSET_BINS_MM
    _v6.OFFSET_BIN_CYCLE = OFFSET_BIN_CYCLE
    _v6._balanced_design = _balanced_design
    _v6._episode_design = _episode_design


def _nearest_sector(offset_xy: np.ndarray) -> int:
    angle = math.atan2(float(offset_xy[1]), float(offset_xy[0])) % (2.0 * math.pi)
    return int(np.rint(angle * NUM_OFFSET_SECTORS / (2.0 * math.pi))) % NUM_OFFSET_SECTORS


def _attempt(
    shell,
    client,
    args: argparse.Namespace,
    policy_args,
    visual_guard,
    *,
    attempt_index: int,
    accepted_index: int,
):
    _configure_v6_backend()
    episode_seed, initial = legacy._episode_randomness(
        args.dataset_seed,
        attempt_index,
        shell.CUP_NAMES,
    )
    design = _balanced_design(args, accepted_index)
    predicted_slot, swaps = _predicted_final_spatial_slot(shell, episode_seed, initial)
    if predicted_slot != design["final_spatial_slot"]:
        return {
            "attempt_index": attempt_index,
            "episode_seed": episode_seed,
            "initial_ball_cup": initial,
            "required_final_spatial_slot": design["final_spatial_slot"],
            "predicted_final_spatial_slot": predicted_slot,
            "predicted_swaps": swaps,
            "reason": "prefiltered_final_spatial_slot_mismatch",
            "physics_started": False,
        }, None

    args.safe_offset_approach = True
    args.safe_clearance_mm = SAFE_CLEARANCE_MM
    args.safe_raise_steps = SAFE_RAISE_STEPS
    args.safe_lateral_steps = SAFE_LATERAL_STEPS
    args.safe_feedback_steps = SAFE_FEEDBACK_STEPS
    args.combined_anchor_perturb = False
    args.feedback_perturb = False
    args.minimum_open_recovery_steps = 0
    audit, payload = _ORIGINAL_ATTEMPT(
        shell,
        client,
        args,
        policy_args,
        visual_guard,
        attempt_index=attempt_index,
        accepted_index=accepted_index,
    )
    if payload is None:
        return audit, None

    metadata = payload[5]
    measured_xy = np.asarray(
        metadata["perturbation"].get("measured_offset_xy_m"),
        dtype=np.float64,
    )
    if measured_xy.shape != (2,) or not np.all(np.isfinite(measured_xy)):
        audit = dict(audit, reason="invalid_measured_offset_vector")
        return audit, None
    measured_sector = _nearest_sector(measured_xy)
    if measured_sector != int(design["offset_sector"]):
        audit = dict(
            audit,
            reason="measured_offset_sector_mismatch",
            measured_offset_sector=measured_sector,
            requested_offset_sector=int(design["offset_sector"]),
        )
        return audit, None

    trace = metadata["oracle"]["gating_trace"]
    initial_xy_error_m = float(trace[0]["pre_command_xy_error_m"])
    if initial_xy_error_m <= MIN_INITIAL_XY_ERROR_M:
        audit = dict(
            audit,
            reason="initial_xy_error_not_hard",
            initial_xy_error_m=initial_xy_error_m,
        )
        return audit, None

    cup_move_xy = float(metadata["anchor_positioning"]["hidden_cup_displacement_xy_m"])
    cup_move_z = abs(float(metadata["anchor_positioning"]["hidden_cup_displacement_z_m"]))
    if max(cup_move_xy, cup_move_z) > MAX_HIDDEN_CUP_DISPLACEMENT_M:
        audit = dict(
            audit,
            reason="hidden_safe_approach_moved_cup",
            hidden_cup_displacement_xy_m=cup_move_xy,
            hidden_cup_displacement_z_m=cup_move_z,
        )
        return audit, None

    metadata["dataset_kind"] = DATASET_KIND
    metadata["v9_distribution"] = {
        "fixed_design_slot_retry_until_filled": True,
        "num_offset_sectors": NUM_OFFSET_SECTORS,
        "measured_offset_sector": measured_sector,
        "minimum_initial_xy_error_m": MIN_INITIAL_XY_ERROR_M,
        "minimum_open_recovery_steps": 0,
        "row_replication_required": False,
        "switch_state_is_precontact": True,
        "low_descent_rows_come_from_oracle_suffix": True,
    }
    metadata["v9_safe_switch_state"] = {
        "path": "raise_then_lateral_at_clearance_then_vertical_descent",
        "clearance_m": SAFE_CLEARANCE_MM / 1_000.0,
        "raise_steps": SAFE_RAISE_STEPS,
        "lateral_steps": SAFE_LATERAL_STEPS,
        "feedback_steps": SAFE_FEEDBACK_STEPS,
        "hidden_actions_supervised": False,
        "hidden_frames_supervised": False,
        "hidden_cup_displacement_xy_m": cup_move_xy,
        "hidden_cup_displacement_z_m": cup_move_z,
    }
    audit = dict(
        audit,
        measured_offset_sector=measured_sector,
        initial_xy_error_m=initial_xy_error_m,
    )
    return audit, payload


def _validate_args(args: argparse.Namespace) -> None:
    _configure_v6_backend()
    args.safe_offset_approach = True
    args.minimum_open_recovery_steps = 0
    _ORIGINAL_VALIDATE_ARGS(args)
    if args.num_episodes == 1200 and args.num_episodes % NUM_OFFSET_SECTORS:
        raise ValueError("A full V9 dataset must balance all 16 offset sectors")
    if args.safe_clearance_mm < max(high for _, high in ANCHOR_BANDS_MM.values()) + 50.0:
        raise ValueError("V9 safe clearance must remain at least 50 mm above every anchor band")


def main() -> None:
    _configure_v6_backend()
    _v6.parse_args = parse_args
    _v6._validate_args = _validate_args
    _v6._attempt = _attempt
    _v6.main()


if __name__ == "__main__":
    main()
