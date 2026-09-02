"""Generate balanced corrections from states visited by V10 itself.

This is deliberately a small, independent dataset variant.  The policy runs
to the diagnosed low-stage boundary (80 control steps)
after the scripted 60-frame observation.  Its commands and intermediate
observations are never serialized.  Stored frame 60 is the resulting real
on-policy state and every trainable row after it is produced by the unchanged
Oracle suffix.
"""

# Private helpers are reused to keep the controller/action alignment identical
# to the already audited on-policy correction writer.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
from pathlib import Path

import generate_onpolicy_eef_correction_dataset as legacy
import generate_onpolicy_eef_low_stage_gated_dataset_v6 as v6
import generate_onpolicy_eef_safe_balanced_recovery_dataset_v9 as v9


DATASET_KIND = "v10_real_onpolicy_oracle_correction"
DEFAULT_POLICY_PREFIX_STEPS = (80,)
FINAL_SPATIAL_SLOTS = ("left", "middle", "right")
VisualCorruptionError = v6.VisualCorruptionError
_VisualGuard = v6._VisualGuard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=9000)
    parser.add_argument("--dataset-seed", type=int, default=260820)
    parser.add_argument("--policy-checkpoint-label", required=True)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument(
        "--prefix-steps",
        default=",".join(str(value) for value in DEFAULT_POLICY_PREFIX_STEPS),
    )
    parser.add_argument(
        "--ports",
        default="",
        help="Optional comma-separated policy ports assigned round-robin to spawned workers.",
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-offset-mm", type=float, default=3.0)
    parser.add_argument("--max-offset-mm", type=float, default=35.0)
    parser.add_argument("--min-safe-height-mm", type=float, default=25.0)
    parser.add_argument("--min-open-width-m", type=float, default=0.04)
    parser.add_argument("--recenter-steps", type=int, default=10)
    parser.add_argument("--descend-steps", type=int, default=30)
    parser.add_argument("--grasp-steps", type=int, default=15)
    parser.add_argument("--lift-steps", type=int, default=40)
    parser.add_argument("--hover-height", type=float, default=0.05)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.width != args.height:
        raise ValueError("Fixed-history policy requires square images")
    if args.num_episodes <= 0 or args.max_attempts < args.num_episodes:
        raise ValueError("Require 0 < num_episodes <= max_attempts")
    if args.num_episodes % len(FINAL_SPATIAL_SLOTS):
        raise ValueError("num_episodes must be divisible by three for exact spatial balance")
    if args.replan_steps != 8:
        raise ValueError("This control experiment requires replan_steps=8")
    prefixes = _prefix_steps(args)
    if any(value % args.replan_steps for value in prefixes):
        raise ValueError(f"prefix steps must lie on replan=8 boundaries: {prefixes}")
    if args.min_offset_mm < 0 or args.max_offset_mm <= args.min_offset_mm:
        raise ValueError("Invalid offset interval")


def _prefix_steps(args: argparse.Namespace) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in args.prefix_steps.split(",") if item.strip())
    if not values or any(value <= 32 for value in values):
        raise ValueError(f"Expected diagnosed low-stage prefix steps >32, got {values}")
    return values


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
    visual_guard.reset()
    episode_seed, initial = legacy._episode_randomness(
        args.dataset_seed, attempt_index, shell.CUP_NAMES
    )
    required_slot = FINAL_SPATIAL_SLOTS[accepted_index % len(FINAL_SPATIAL_SLOTS)]
    within_slot = accepted_index // len(FINAL_SPATIAL_SLOTS)
    prefixes = _prefix_steps(args)
    prefix_steps = prefixes[(within_slot + accepted_index % 3) % len(prefixes)]
    predicted_slot, swaps = v9._predicted_final_spatial_slot(shell, episode_seed, initial)
    if predicted_slot != required_slot:
        return {
            "attempt_index": attempt_index,
            "episode_seed": episode_seed,
            "initial_ball_cup": initial,
            "accepted_index": accepted_index,
            "required_final_spatial_slot": required_slot,
            "predicted_final_spatial_slot": predicted_slot,
            "predicted_swaps": swaps,
            "physics_started": False,
            "reason": "prefiltered_final_spatial_slot_mismatch",
        }, None

    # The legacy writer reads this constant both for execution and metadata.
    # Each worker is a separate process, so this assignment cannot race.
    legacy.MODEL_APPROACH_STEPS = prefix_steps
    audit, payload = legacy._attempt(
        shell,
        client,
        args,
        policy_args,
        attempt_index=attempt_index,
        accepted_index=accepted_index,
    )
    audit.update(
        required_final_spatial_slot=required_slot,
        predicted_final_spatial_slot=predicted_slot,
        physics_started=True,
    )
    if payload is None:
        return audit, None

    metadata = payload[5]
    actual_slot = metadata["final_ball_cup"]
    if actual_slot != required_slot:
        return dict(audit, reason="runtime_final_spatial_slot_mismatch"), None
    metadata["dataset_kind"] = DATASET_KIND
    metadata["final_spatial_slot"] = actual_slot
    metadata["required_final_spatial_slot"] = required_slot
    metadata["model_prefix"].update(
        {
            "policy_checkpoint_label": args.policy_checkpoint_label,
            "executed_steps": prefix_steps,
            "replan_steps": args.replan_steps,
            "state_source": "real_v10_closed_loop_no_hidden_perturbation",
            "actions_saved_to_training_file": False,
            "intermediate_frames_saved_to_training_file": False,
        }
    )
    metadata["supervision_contract"].update(
        {
            "model_generated_actions_supervised": False,
            "model_generated_frames_supervised": False,
            "supervised_action_source": "oracle_only",
            "full_consecutive_horizon_required": 16,
        }
    )
    metadata["v10_onpolicy_distribution"] = {
        "balanced_final_spatial_slot": True,
        "hidden_perturbation_used": False,
        "policy_prefix_steps": prefix_steps,
        "replan_steps": args.replan_steps,
    }
    return audit, payload


if __name__ == "__main__":
    raise SystemExit(
        "Use generate_v10_onpolicy_oracle_correction_dataset_parallel.py"
    )
