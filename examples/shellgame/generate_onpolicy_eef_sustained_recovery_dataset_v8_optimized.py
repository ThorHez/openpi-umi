"""Optimized V8 sustained-recovery generation without wasted physics retries.

This keeps V8's exact slot/stage/radius/direction design, while making three
generation-only changes:

* reject incompatible final-slot seeds before constructing a MuJoCo env;
* approach the low off-centre anchor directly, avoiding a low centre crossing;
* use feedback termination for the hidden perturbation and guarantee at least
  12 consecutive open-gripper Oracle recovery commands.

None of the learned-policy, anchor, or perturbation commands are serialized.
"""

# Private V8 helpers are reused to preserve its audited data contract.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse

import generate_onpolicy_eef_sustained_recovery_dataset_v8 as _v8
import numpy as np

DATASET_KIND = "onpolicy_eef_sustained_recovery_v8_optimized"
EXPECTED_EPISODE_FRAMES = _v8.EXPECTED_EPISODE_FRAMES
CORRECTION_COMMANDS = _v8.CORRECTION_COMMANDS
DEFAULT_PREFIX_STEPS = _v8.DEFAULT_PREFIX_STEPS
FINAL_SPATIAL_SLOTS = _v8.FINAL_SPATIAL_SLOTS
ANCHOR_BANDS_MM = _v8.ANCHOR_BANDS_MM
ANCHOR_STAGE_CYCLE = _v8.ANCHOR_STAGE_CYCLE
OFFSET_BINS_MM = _v8.OFFSET_BINS_MM
OFFSET_BIN_CYCLE = _v8.OFFSET_BIN_CYCLE
NUM_OFFSET_SECTORS = _v8.NUM_OFFSET_SECTORS
MIN_UNIQUE_RECOVERY_ROWS = _v8.MIN_UNIQUE_RECOVERY_ROWS

legacy = _v8.legacy
VisualCorruptionError = _v8.VisualCorruptionError
_VisualGuard = _v8._VisualGuard


def parse_args() -> argparse.Namespace:
    args = _v8.parse_args()
    if args.dataset_seed == 260821:
        args.dataset_seed = 260823
    args.combined_anchor_perturb = True
    args.feedback_perturb = True
    args.minimum_open_recovery_steps = MIN_UNIQUE_RECOVERY_ROWS
    # This is only an upper bound under feedback termination.  Extra capacity
    # avoids discarding the occasional slowly converging controller state.
    if args.perturb_steps == 6:
        args.perturb_steps = 12
    return args


def _prefix_steps(args: argparse.Namespace) -> tuple[int, ...]:
    return _v8._prefix_steps(args)


def _balanced_design(args: argparse.Namespace, accepted_index: int) -> dict:
    return _v8._balanced_design(args, accepted_index)


def _episode_design(
    args: argparse.Namespace,
    *,
    attempt_index: int,
    accepted_index: int,
):
    return _v8._episode_design(
        args,
        attempt_index=attempt_index,
        accepted_index=accepted_index,
    )


def _predicted_final_spatial_slot(shell, episode_seed: int, initial: str) -> tuple[str, list]:
    """Replay the deterministic symbolic swaps without rendering or physics."""
    swaps = shell.sample_swaps(np.random.default_rng(episode_seed), 3)
    cup_slots = {name: name for name in shell.CUP_NAMES}
    for slot_a, slot_b in swaps:
        cup_a = shell.cup_in_slot(cup_slots, slot_a)
        cup_b = shell.cup_in_slot(cup_slots, slot_b)
        cup_slots[cup_a], cup_slots[cup_b] = slot_b, slot_a
    return cup_slots[initial], swaps


def _configure_v6_backend() -> None:
    _v8._configure_v6_backend()


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

    args.combined_anchor_perturb = True
    args.feedback_perturb = True
    args.minimum_open_recovery_steps = MIN_UNIQUE_RECOVERY_ROWS
    audit, payload = _v8._attempt(
        shell,
        client,
        args,
        policy_args,
        visual_guard,
        attempt_index=attempt_index,
        accepted_index=accepted_index,
    )
    if payload is not None:
        metadata = payload[5]
        metadata["dataset_kind"] = DATASET_KIND
        metadata["v8_optimized_generation"] = {
            "final_slot_seed_prefilter": True,
            "predicted_final_slot": predicted_slot,
            "combined_anchor_and_perturb_xy": True,
            "feedback_perturbation": True,
            "minimum_open_recovery_steps": MIN_UNIQUE_RECOVERY_ROWS,
            "hidden_motion_supervised": False,
        }
    return audit, payload


def _validate_args(args: argparse.Namespace) -> None:
    args.combined_anchor_perturb = True
    args.feedback_perturb = True
    args.minimum_open_recovery_steps = MIN_UNIQUE_RECOVERY_ROWS
    _v8._validate_args(args)


def main() -> None:
    _v8._v6.parse_args = parse_args
    _v8._v6._validate_args = _validate_args
    _v8._v6._attempt = _attempt
    _v8._v6.main()


if __name__ == "__main__":
    main()
