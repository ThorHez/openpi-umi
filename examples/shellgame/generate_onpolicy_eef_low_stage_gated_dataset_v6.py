"""Generate low-stage absolute-EEF corrections with gated descent and grasp.

Frames 0..59 are the normal three-swap history.  A learned policy is used only
to identify the target cup; its actions and intermediate observations are not
serialized.  Unsaved Oracle positioning moves the EEF to a balanced high,
mid, or low descent anchor, and an unsaved controller perturbation creates a
measured lateral offset.  Stored frame 60 is the resulting off-centre state.

All stored actions after frame 60 are Oracle-only.  The Oracle continuously
targets the live cup centre, holds Z for large XY error, descends slowly for
moderate error, and descends normally only while aligned.  The gripper remains
open until XY and Z have both satisfied explicit thresholds for consecutive
steps.  This creates the low, off-centre, open-gripper examples missing from
the V4/V5 dataset while preserving exact horizon supervision.
"""

# Private helpers are reused to preserve the validated rendering, environment,
# serialization, and fixed-history policy contracts.
# ruff: noqa: FBT001, FBT003, SLF001

from __future__ import annotations

import argparse
from collections import Counter
from collections import deque
import dataclasses
import json
import logging
import math
from pathlib import Path
import time

import generate_onpolicy_eef_continuous_descent_dataset_v4 as v4
import generate_onpolicy_eef_correction_dataset as legacy
import generate_onpolicy_eef_correction_dataset_v3 as v3
import numpy as np

DATASET_KIND = "onpolicy_eef_low_stage_gated_v6"
EXPECTED_EPISODE_FRAMES = 155
CORRECTION_COMMANDS = EXPECTED_EPISODE_FRAMES - legacy.HISTORY_FRAMES - 1
DEFAULT_PREFIX_STEPS = (30, 36, 42)
FINAL_SPATIAL_SLOTS = ("left", "middle", "right")

ANCHOR_BANDS_MM = {
    "high": (110.0, 160.0),
    "mid": (70.0, 110.0),
    "late": (30.0, 70.0),
}
ANCHOR_STAGE_CYCLE = ("high",) * 10 + ("mid",) * 20 + ("late",) * 70
OFFSET_BINS_MM = {
    "small": (5.0, 10.0),
    "medium": (10.0, 18.0),
    "large": (18.0, 25.0),
}
OFFSET_BIN_CYCLE = ("small",) * 30 + ("medium",) * 45 + ("large",) * 25

VisualCorruptionError = v4.VisualCorruptionError
_VisualGuard = v4._VisualGuard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=43200)
    parser.add_argument("--dataset-seed", type=int, default=260819)
    parser.add_argument("--policy-checkpoint-label", required=True)
    parser.add_argument("--replan-steps", type=int, default=3)
    parser.add_argument(
        "--prefix-steps",
        default=",".join(str(value) for value in DEFAULT_PREFIX_STEPS),
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-open-width-m", type=float, default=0.04)
    parser.add_argument("--anchor-steps", type=int, default=10)
    parser.add_argument(
        "--safe-offset-approach",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a collision-avoiding hidden switch-state path: raise at the "
            "current XY, translate to the requested offset at clearance height, "
            "then descend vertically. The default preserves the V6 path."
        ),
    )
    parser.add_argument("--safe-raise-steps", type=int, default=10)
    parser.add_argument("--safe-lateral-steps", type=int, default=10)
    parser.add_argument(
        "--safe-feedback-steps",
        type=int,
        default=0,
        help=(
            "Additional hidden closed-loop commands after the safe vertical "
            "descent. They improve measured small-offset tracking and remain "
            "excluded from supervision. The default preserves prior datasets."
        ),
    )
    parser.add_argument("--safe-clearance-mm", type=float, default=180.0)
    parser.add_argument("--anchor-height-tolerance-mm", type=float, default=10.0)
    parser.add_argument("--perturb-steps", type=int, default=6)
    parser.add_argument("--offset-bin-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--max-open-steps", type=int, default=60)
    parser.add_argument("--grasp-steps", type=int, default=10)
    parser.add_argument("--min-lift-steps", type=int, default=20)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--descent-jitter-mm", type=float, default=2.0)
    parser.add_argument("--hold-z-above-xy-mm", type=float, default=10.0)
    parser.add_argument("--aligned-xy-mm", type=float, default=6.0)
    parser.add_argument("--close-xy-mm", type=float, default=5.0)
    parser.add_argument("--close-z-mm", type=float, default=3.0)
    parser.add_argument("--close-hold-steps", type=int, default=3)
    parser.add_argument("--slow-descent-mm", type=float, default=2.0)
    parser.add_argument("--normal-descent-mm", type=float, default=8.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def _prefix_steps(args: argparse.Namespace) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in args.prefix_steps.split(",") if item.strip())
    if not values or any(value <= 10 for value in values):
        raise ValueError(f"prefix steps must all be > 10, got {values}")
    return values


def _balanced_design(args: argparse.Namespace, accepted_index: int) -> dict:
    """Assign exact spatial/stage/offset/direction quotas for 1200 slots."""
    slot_index = accepted_index % len(FINAL_SPATIAL_SLOTS)
    within_slot = accepted_index // len(FINAL_SPATIAL_SLOTS)
    prefixes = _prefix_steps(args)
    stage_index = (within_slot * 37 + slot_index * 23) % 100
    offset_index = (within_slot * 29 + slot_index * 17) % 100
    return {
        "final_spatial_slot": FINAL_SPATIAL_SLOTS[slot_index],
        "within_spatial_slot_index": within_slot,
        "prefix_steps": prefixes[(within_slot + slot_index) % len(prefixes)],
        "anchor_stage": ANCHOR_STAGE_CYCLE[stage_index],
        "offset_bin": OFFSET_BIN_CYCLE[offset_index],
        "offset_sector": (within_slot + 3 * slot_index) % 8,
    }


def _episode_design(
    args: argparse.Namespace,
    *,
    attempt_index: int,
    accepted_index: int,
) -> tuple[dict, float, np.ndarray, np.ndarray]:
    design = _balanced_design(args, accepted_index)
    rng = np.random.default_rng(np.random.SeedSequence([args.dataset_seed, attempt_index, accepted_index, 619]))
    anchor_low, anchor_high = ANCHOR_BANDS_MM[design["anchor_stage"]]
    anchor_height_m = float(rng.uniform(anchor_low, anchor_high) / 1_000.0)

    offset_low, offset_high = OFFSET_BINS_MM[design["offset_bin"]]
    angle = 2.0 * math.pi * design["offset_sector"] / 8.0
    angle += float(rng.uniform(-math.pi / 16.0, math.pi / 16.0))
    radius_m = float(rng.uniform(offset_low, offset_high) / 1_000.0)
    perturb_xy = radius_m * np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)

    jitter_radius = float(rng.uniform(0.0, args.descent_jitter_mm / 1_000.0))
    jitter_angle = float(rng.uniform(-math.pi, math.pi))
    descent_jitter_xy = jitter_radius * np.asarray([math.cos(jitter_angle), math.sin(jitter_angle)], dtype=np.float64)
    return design, anchor_height_m, perturb_xy, descent_jitter_xy


def _attempt(
    shell,
    client,
    args: argparse.Namespace,
    policy_args: legacy.base.Args,
    visual_guard: _VisualGuard,
    *,
    attempt_index: int,
    accepted_index: int,
) -> tuple[dict, tuple | None]:
    visual_guard.reset()
    episode_seed, initial = legacy._episode_randomness(args.dataset_seed, attempt_index, shell.CUP_NAMES)
    design, anchor_height_m, perturb_xy, descent_jitter_xy = _episode_design(
        args, attempt_index=attempt_index, accepted_index=accepted_index
    )
    prefix_steps = design["prefix_steps"]
    ep_args = legacy.base._episode_namespace(policy_args, seed=episode_seed, initial_ball_cup=initial, num_swaps=3)
    env = shell.make_env(ep_args)
    history: list[dict] = []
    discarded_video: list[np.ndarray] = []
    reject: dict = {
        "attempt_index": attempt_index,
        "episode_seed": episode_seed,
        "initial_ball_cup": initial,
        "prefix_steps": prefix_steps,
        "anchor_stage": design["anchor_stage"],
        "requested_anchor_height_m": anchor_height_m,
        "offset_bin": design["offset_bin"],
        "required_final_spatial_slot": design["final_spatial_slot"],
        "offset_sector": design["offset_sector"],
    }

    try:
        try:
            meta = legacy.base._run_scripted_observation(
                shell,
                env,
                ep_args,
                policy_args,
                history,
                discarded_video,
                client=client,
            )
            if len(history) != legacy.HISTORY_FRAMES:
                raise RuntimeError(f"Expected {legacy.HISTORY_FRAMES} scripted frames, got {len(history)}")
            reject["final_spatial_slot"] = meta["final_ball_cup"]
            if meta["final_ball_cup"] != design["final_spatial_slot"]:
                reject["reason"] = "final_spatial_slot_quota_mismatch"
                return reject, None

            fixed_history = [legacy._snapshot(item) for item in history]
            canonical_quat = np.asarray(history[0]["eef_quat"], dtype=np.float32)
            start_eef_pos = np.asarray(history[0]["eef_pos"], dtype=np.float32)
            action_plan: deque = deque()
            gripper_action = policy_args.default_gripper_action
            votes = dict.fromkeys(shell.CUP_NAMES, 0)
            distance_sums = dict.fromkeys(shell.CUP_NAMES, 0.0)

            # Learned commands only establish target identity.  They are never
            # stored as observations or targets in the training trajectory.
            for step in range(prefix_steps):
                env_action, gripper_action, _ = legacy.base._policy_env_action(
                    shell,
                    env,
                    history,
                    start_eef_pos,
                    canonical_quat,
                    action_plan,
                    gripper_action,
                    client=client,
                    args=policy_args,
                    prompt=legacy.GRASP_PROMPT,
                )
                env.step(env_action)
                legacy.base._append_observation(
                    shell,
                    env,
                    ep_args,
                    meta["wrist_camera_name"],
                    history,
                    [],
                    resize_size=args.width,
                )
                eef = np.asarray(history[-1]["eef_pos"], dtype=np.float64)
                cups = legacy.base._cup_positions(shell, env)
                distances = {
                    cup: float(np.linalg.norm(eef[:2] - np.asarray(pos[:2], dtype=np.float64)))
                    for cup, pos in cups.items()
                }
                nearest = min(distances, key=distances.get)
                if step >= 10 and distances[nearest] <= policy_args.cup_selection_xy_radius:
                    votes[nearest] += 1
                    distance_sums[nearest] += distances[nearest]

            selected = legacy._select_cup(votes, distance_sums)
            reject.update(
                target_cup=meta["target_cup"],
                selected_cup=selected,
                selection_votes=votes,
            )
            if selected is None:
                reject["reason"] = "no_selection_vote"
                return reject, None
            if selected != meta["target_cup"]:
                reject["reason"] = "selection_incorrect"
                return reject, None

            # Move to a controlled descent anchor without saving the motion.
            selected_pos = np.asarray(legacy.base._cup_positions(shell, env)[selected], dtype=np.float64)
            pre_anchor_cup_pos = selected_pos.copy()
            grasp_z = float(env.cup_handle_grasp_z() + ep_args.robot_grasp_z_offset)
            # Newer dataset variants can approach the requested off-centre XY
            # and descent height in one motion.  The default remains the V6
            # centre-first behavior for reproducibility.
            safe_offset_approach = bool(getattr(args, "safe_offset_approach", False))
            combined_anchor_perturb = bool(getattr(args, "combined_anchor_perturb", False))

            def hidden_move(target: np.ndarray, steps: int) -> None:
                for _ in range(steps):
                    command = shell.make_robot_action(
                        env,
                        target_pos=target,
                        target_quat=canonical_quat,
                        gripper_action=-1.0,
                    )
                    env.step(command)
                    legacy.base._append_observation(
                        shell,
                        env,
                        ep_args,
                        meta["wrist_camera_name"],
                        history,
                        [],
                        resize_size=args.width,
                    )

            if safe_offset_approach:
                # Do not take a diagonal shortcut to a low off-centre pose. It
                # crosses the cup for exactly the late / inward-error states we
                # want to retain. Raise first, move laterally at clearance, and
                # only then descend vertically at the requested offset.
                current_eef = np.asarray(history[-1]["eef_pos"], dtype=np.float64)
                clearance_z = max(
                    float(current_eef[2]),
                    grasp_z + float(args.safe_clearance_mm) / 1_000.0,
                )
                hidden_move(
                    np.asarray([current_eef[0], current_eef[1], clearance_z], dtype=np.float64),
                    int(args.safe_raise_steps),
                )
                selected_pos = np.asarray(
                    legacy.base._cup_positions(shell, env)[selected], dtype=np.float64
                )
                offset_xy = selected_pos[:2] + perturb_xy
                hidden_move(
                    np.asarray([offset_xy[0], offset_xy[1], clearance_z], dtype=np.float64),
                    int(args.safe_lateral_steps),
                )
                selected_pos = np.asarray(
                    legacy.base._cup_positions(shell, env)[selected], dtype=np.float64
                )
                offset_xy = selected_pos[:2] + perturb_xy
                hidden_move(
                    np.asarray(
                        [offset_xy[0], offset_xy[1], grasp_z + anchor_height_m],
                        dtype=np.float64,
                    ),
                    int(args.anchor_steps),
                )
                combined_anchor_perturb = True
            else:
                anchor_xy = selected_pos[:2] + perturb_xy if combined_anchor_perturb else selected_pos[:2]
                hidden_move(
                    np.asarray(
                        [anchor_xy[0], anchor_xy[1], grasp_z + anchor_height_m],
                        dtype=np.float64,
                    ),
                    int(args.anchor_steps),
                )

            # Create one measured off-centre state.  These commands and their
            # intermediate observations are also excluded from supervision.
            feedback_perturb = bool(getattr(args, "feedback_perturb", False))
            perturb_executed_steps = 0
            perturb_step_budget = (
                int(args.safe_feedback_steps) if safe_offset_approach else int(args.perturb_steps)
            )
            for _ in range(perturb_step_budget):
                selected_pos = np.asarray(legacy.base._cup_positions(shell, env)[selected], dtype=np.float64)
                anchored_eef = np.asarray(history[-1]["eef_pos"], dtype=np.float64)
                measured_before_m = float(np.linalg.norm(anchored_eef[:2] - selected_pos[:2]))
                offset_low, offset_high = OFFSET_BINS_MM[design["offset_bin"]]
                offset_tolerance_m = args.offset_bin_tolerance_mm / 1_000.0
                if feedback_perturb and (
                    offset_low / 1_000.0 - offset_tolerance_m
                    <= measured_before_m
                    <= offset_high / 1_000.0 + offset_tolerance_m
                ):
                    break
                perturb_target = np.asarray(
                    [
                        selected_pos[0] + perturb_xy[0],
                        selected_pos[1] + perturb_xy[1],
                        grasp_z + anchor_height_m,
                    ],
                    dtype=np.float64,
                )
                command = shell.make_robot_action(
                    env,
                    target_pos=perturb_target,
                    target_quat=canonical_quat,
                    gripper_action=-1.0,
                )
                env.step(command)
                legacy.base._append_observation(
                    shell,
                    env,
                    ep_args,
                    meta["wrist_camera_name"],
                    history,
                    [],
                    resize_size=args.width,
                )
                perturb_executed_steps += 1

            switch = legacy._snapshot(history[-1])
            selected_pos = np.asarray(legacy.base._cup_positions(shell, env)[selected], dtype=np.float64)
            switch_eef = np.asarray(switch["eef_pos"], dtype=np.float64)
            hidden_cup_displacement_xy_m = float(
                np.linalg.norm(selected_pos[:2] - pre_anchor_cup_pos[:2])
            )
            hidden_cup_displacement_z_m = float(selected_pos[2] - pre_anchor_cup_pos[2])
            measured_offset_xy_m = switch_eef[:2] - selected_pos[:2]
            measured_offset_m = float(np.linalg.norm(switch_eef[:2] - selected_pos[:2]))
            measured_anchor_height_m = float(switch_eef[2] - grasp_z)
            switch_contacts = legacy.contact_utils._finger_contact_count(env, selected)
            offset_low, offset_high = OFFSET_BINS_MM[design["offset_bin"]]
            anchor_low, anchor_high = ANCHOR_BANDS_MM[design["anchor_stage"]]
            offset_tolerance_m = args.offset_bin_tolerance_mm / 1_000.0
            anchor_tolerance_m = args.anchor_height_tolerance_mm / 1_000.0
            reject.update(
                switch_offset_m=measured_offset_m,
                switch_offset_xy_m=measured_offset_xy_m.tolist(),
                switch_safe_height_m=measured_anchor_height_m,
                switch_gripper_width_m=switch["gripper_width"],
                switch_contacts=switch_contacts,
                hidden_cup_displacement_xy_m=hidden_cup_displacement_xy_m,
                hidden_cup_displacement_z_m=hidden_cup_displacement_z_m,
                visual_flip_repairs=visual_guard.flip_repairs,
            )
            if safe_offset_approach and (
                hidden_cup_displacement_xy_m > 0.002
                or abs(hidden_cup_displacement_z_m) > 0.002
            ):
                reject["reason"] = "hidden_safe_approach_moved_cup"
                return reject, None
            if not (
                offset_low / 1_000.0 - offset_tolerance_m
                <= measured_offset_m
                <= offset_high / 1_000.0 + offset_tolerance_m
            ):
                reject["reason"] = "perturb_offset_outside_bin"
                return reject, None
            if not (
                anchor_low / 1_000.0 - anchor_tolerance_m
                <= measured_anchor_height_m
                <= anchor_high / 1_000.0 + anchor_tolerance_m
            ):
                reject["reason"] = "anchor_height_outside_stage"
                return reject, None
            if switch["gripper_width"] < args.min_open_width_m:
                reject["reason"] = "gripper_not_open"
                return reject, None
            if switch_contacts:
                reject["reason"] = "contact_before_oracle"
                return reject, None

            observations = [*fixed_history, switch]
            actions = [legacy._placeholder_absolute_action(shell, item) for item in observations]
            action_mask = [False] * len(observations)
            supervision = [legacy.SUPERVISION_CONTEXT] * len(observations)
            phase_ids = [
                *([0] * 10),
                *([1] * 10),
                *([2] * 30),
                *([3] * 10),
                legacy.PHASE_RECENTER,
            ]
            stage_metrics: list[dict] = []
            mode_counts = Counter()

            def execute(
                stage: str,
                phase_id: int,
                step: int,
                target: np.ndarray,
                *,
                close: bool,
                pre_error_m: float | None = None,
            ) -> dict:
                command = shell.make_robot_action(
                    env,
                    target_pos=np.asarray(target, dtype=np.float64),
                    target_quat=canonical_quat,
                    gripper_action=1.0 if close else -1.0,
                )
                env.step(command)
                legacy.base._append_observation(
                    shell,
                    env,
                    ep_args,
                    meta["wrist_camera_name"],
                    history,
                    [],
                    resize_size=args.width,
                )
                observation = legacy._snapshot(history[-1])
                observations.append(observation)
                actions.append(np.asarray(command, dtype=np.float32))
                action_mask.append(True)
                supervision.append(legacy.SUPERVISION_ORACLE)
                phase_ids.append(phase_id)
                actual = np.asarray(observation["eef_pos"], dtype=np.float64)
                live_cup = np.asarray(legacy.base._cup_positions(shell, env)[selected], dtype=np.float64)
                metric = {
                    "stage": stage,
                    "stage_step": step,
                    "pre_command_xy_error_m": pre_error_m,
                    "eef_to_selected_xy_m": float(np.linalg.norm(actual[:2] - live_cup[:2])),
                    "eef_to_grasp_z_m": float(actual[2] - grasp_z),
                    "selected_finger_contacts": legacy.contact_utils._finger_contact_count(env, selected),
                }
                stage_metrics.append(metric)
                return metric

            aligned_run = 0
            ready_to_close = False
            open_steps = 0
            for step in range(args.max_open_steps):
                current_eef = np.asarray(history[-1]["eef_pos"], dtype=np.float64)
                live_cup = np.asarray(legacy.base._cup_positions(shell, env)[selected], dtype=np.float64)
                oracle_xy = live_cup[:2] + descent_jitter_xy
                xy_error_m = float(np.linalg.norm(current_eef[:2] - live_cup[:2]))
                xy_error_mm = xy_error_m * 1_000.0
                if xy_error_mm > args.hold_z_above_xy_mm:
                    target_z = current_eef[2]
                    mode = "hold_z_recenter"
                    phase_id = legacy.PHASE_RECENTER
                elif xy_error_mm > args.aligned_xy_mm:
                    target_z = max(grasp_z, current_eef[2] - args.slow_descent_mm / 1_000.0)
                    mode = "slow_descent_recenter"
                    phase_id = legacy.PHASE_DESCEND
                else:
                    target_z = max(
                        grasp_z,
                        current_eef[2] - args.normal_descent_mm / 1_000.0,
                    )
                    mode = "aligned_descent"
                    phase_id = legacy.PHASE_DESCEND
                mode_counts[mode] += 1
                metric = execute(
                    mode,
                    phase_id,
                    step,
                    np.asarray([oracle_xy[0], oracle_xy[1], target_z], dtype=np.float64),
                    close=False,
                    pre_error_m=xy_error_m,
                )
                open_steps += 1
                aligned_and_low = (
                    metric["eef_to_selected_xy_m"] <= args.close_xy_mm / 1_000.0
                    and abs(metric["eef_to_grasp_z_m"]) <= args.close_z_mm / 1_000.0
                )
                aligned_run = aligned_run + 1 if aligned_and_low else 0
                minimum_open_steps = int(getattr(args, "minimum_open_recovery_steps", 0))
                if aligned_run >= args.close_hold_steps and open_steps >= minimum_open_steps:
                    ready_to_close = True
                    break

            if not ready_to_close:
                reject["reason"] = "gated_descent_not_ready_to_close"
                reject["open_steps"] = open_steps
                reject["final_open_metric"] = stage_metrics[-1]
                return reject, None

            lift_steps = CORRECTION_COMMANDS - open_steps - args.grasp_steps
            if lift_steps < args.min_lift_steps:
                reject["reason"] = "insufficient_lift_steps_after_gated_descent"
                reject["open_steps"] = open_steps
                reject["lift_steps"] = lift_steps
                return reject, None

            live_cup = np.asarray(legacy.base._cup_positions(shell, env)[selected], dtype=np.float64)
            grasp_pos = np.asarray(
                [
                    live_cup[0] + descent_jitter_xy[0],
                    live_cup[1] + descent_jitter_xy[1],
                    grasp_z,
                ],
                dtype=np.float64,
            )
            preclose_xy_m = float(stage_metrics[-1]["eef_to_selected_xy_m"])
            preclose_z_m = float(stage_metrics[-1]["eef_to_grasp_z_m"])
            for step in range(args.grasp_steps):
                execute(
                    "grasp",
                    legacy.PHASE_GRASP,
                    step,
                    grasp_pos,
                    close=True,
                )

            lift_pos = grasp_pos + np.asarray([0.0, 0.0, args.lift_height])
            for step in range(lift_steps):
                progress = v4.v2._linear_progress(step, lift_steps)
                execute(
                    "lift",
                    legacy.PHASE_LIFT,
                    step,
                    (1.0 - progress) * grasp_pos + progress * lift_pos,
                    close=True,
                )

            success, success_stats = legacy.base._success(
                shell,
                env,
                meta["target_cup"],
                meta["settle_cup_pos"],
                policy_args.lift_success_height,
            )
            if not success:
                reject["reason"] = "oracle_suffix_failed"
                reject["success_stats"] = success_stats
                return reject, None
            if len(observations) != EXPECTED_EPISODE_FRAMES:
                raise RuntimeError(f"V6 episode must have {EXPECTED_EPISODE_FRAMES} frames, got {len(observations)}")

            metadata = {
                "env": "ShellGame",
                "fps": args.fps,
                "width": args.width,
                "height": args.height,
                "seed": episode_seed,
                "attempt_index": attempt_index,
                "accepted_index": accepted_index,
                "final_spatial_slot": meta["final_ball_cup"],
                "required_final_spatial_slot": design["final_spatial_slot"],
                "within_spatial_slot_index": design["within_spatial_slot_index"],
                "anchor_stage": design["anchor_stage"],
                "offset_sector": design["offset_sector"],
                "initial_ball_cup": initial,
                "target_cup_identity": meta["target_cup"],
                "final_ball_cup": meta["final_ball_cup"],
                "swaps": [list(pair) for pair in meta["swaps"]],
                "phase_ranges": v3._phase_ranges(phase_ids),
                "num_frames": len(observations),
                "success": True,
                "failure_mode": "success",
                "osc_input_type": "absolute",
                "action_representation": "controller",
                "dataset_kind": DATASET_KIND,
                "model_prefix": {
                    "policy_checkpoint_label": args.policy_checkpoint_label,
                    "executed_steps": prefix_steps,
                    "actions_saved_to_training_file": False,
                    "intermediate_frames_saved_to_training_file": False,
                    "purpose": "target-cup selection only",
                },
                "anchor_positioning": {
                    "stage": design["anchor_stage"],
                    "requested_height_above_grasp_m": anchor_height_m,
                    "measured_height_above_grasp_m": measured_anchor_height_m,
                    "executed_steps": args.anchor_steps,
                    "combined_with_perturbation_xy": combined_anchor_perturb,
                    "safe_raise_translate_vertical_path": safe_offset_approach,
                    "hidden_cup_displacement_xy_m": hidden_cup_displacement_xy_m,
                    "hidden_cup_displacement_z_m": hidden_cup_displacement_z_m,
                    "actions_saved_to_training_file": False,
                    "intermediate_frames_saved_to_training_file": False,
                },
                "perturbation": {
                    "offset_bin": design["offset_bin"],
                    "requested_offset_xy_m": perturb_xy.tolist(),
                    "requested_offset_m": float(np.linalg.norm(perturb_xy)),
                    "measured_offset_m": measured_offset_m,
                    "measured_offset_xy_m": measured_offset_xy_m.tolist(),
                    "executed_steps": perturb_executed_steps,
                    "feedback_terminated": feedback_perturb,
                    "safe_offset_approach": safe_offset_approach,
                    "actions_saved_to_training_file": False,
                    "intermediate_frames_saved_to_training_file": False,
                },
                "supervision_contract": {
                    "stored_frames_0_59": "scripted_history_context_only",
                    "stored_frame_60": "post_anchor_and_perturbation_context_only",
                    "stored_frames_61_plus": "observations_after_oracle_commands",
                    "action_mask_true_source": "oracle_only",
                    "first_training_pair": "observation[60] -> oracle_action[61]",
                    "model_anchor_or_perturb_actions_supervised": False,
                    "full_consecutive_horizon_required": 16,
                },
                "switch": {
                    "selected_cup": selected,
                    "target_cup": meta["target_cup"],
                    "selection_votes": votes,
                    "prefix_steps": prefix_steps,
                    "anchor_stage": design["anchor_stage"],
                    "offset_m": measured_offset_m,
                    "safe_height_m": measured_anchor_height_m,
                    "gripper_width_m": switch["gripper_width"],
                    "contacts": switch_contacts,
                },
                "oracle": {
                    "xy_mode": "live_cup_center_with_error_gated_z_and_gripper",
                    "descent_jitter_xy_m": descent_jitter_xy.tolist(),
                    "open_steps": open_steps,
                    "mode_counts": dict(mode_counts),
                    "gating_trace": stage_metrics[:open_steps],
                    "grasp_steps": args.grasp_steps,
                    "lift_steps": lift_steps,
                    "preclose_xy_error_m": preclose_xy_m,
                    "preclose_z_error_m": preclose_z_m,
                    "close_hold_steps": args.close_hold_steps,
                    "minimum_open_recovery_steps": int(
                        getattr(args, "minimum_open_recovery_steps", 0)
                    ),
                    "close_xy_threshold_m": args.close_xy_mm / 1_000.0,
                    "close_z_threshold_m": args.close_z_mm / 1_000.0,
                    "hold_z_above_xy_error_m": args.hold_z_above_xy_mm / 1_000.0,
                    "aligned_xy_threshold_m": args.aligned_xy_mm / 1_000.0,
                    "lift_height_m": args.lift_height,
                    "final_success_stats": success_stats,
                    "max_bilateral_contact_run": legacy._max_contact_run(stage_metrics),
                },
                "render_integrity": {"flip_repairs": visual_guard.flip_repairs},
                "command_args": {
                    **dataclasses.asdict(policy_args),
                    "seed": episode_seed,
                    "initial_ball_cup": initial,
                    "num_swaps": 3,
                    "action_representation": "controller",
                    "osc_input_type": "absolute",
                },
            }
            reject["reason"] = "accepted"
            return reject, (
                observations,
                actions,
                action_mask,
                phase_ids,
                supervision,
                metadata,
                initial,
                meta["final_ball_cup"],
            )
        except VisualCorruptionError as exc:
            reject["reason"] = "visual_corruption"
            reject["detail"] = str(exc)
            return reject, None
    finally:
        env.close()


def _validate_args(args: argparse.Namespace) -> None:
    prefixes = _prefix_steps(args)
    if args.width != args.height:
        raise ValueError("Current fixed-history policy expects square images")
    if args.num_episodes <= 0 or args.max_attempts < args.num_episodes:
        raise ValueError("Require 0 < num_episodes <= max_attempts")
    if args.num_episodes % len(FINAL_SPATIAL_SLOTS):
        raise ValueError("num_episodes must be divisible by three spatial slots")
    if args.num_episodes == 1200 and args.num_episodes % 300:
        raise ValueError("Full V6 balance requires a multiple of 300 episodes")
    if args.num_episodes % len(prefixes):
        raise ValueError("num_episodes must be divisible by the number of prefix lengths")
    if args.anchor_steps <= 0 or args.perturb_steps <= 0:
        raise ValueError("anchor_steps and perturb_steps must be positive")
    if args.safe_offset_approach and (
        args.safe_raise_steps <= 0
        or args.safe_lateral_steps <= 0
        or args.safe_clearance_mm <= 0
    ):
        raise ValueError("safe approach steps and clearance must be positive")
    if not 0 < args.close_xy_mm <= args.aligned_xy_mm < args.hold_z_above_xy_mm:
        raise ValueError("Require close_xy <= aligned_xy < hold_z_above_xy")
    if args.close_z_mm <= 0 or args.close_hold_steps <= 0:
        raise ValueError("close Z threshold and hold steps must be positive")
    if args.slow_descent_mm <= 0 or args.normal_descent_mm <= args.slow_descent_mm:
        raise ValueError("Require 0 < slow_descent_mm < normal_descent_mm")
    max_open_allowed = CORRECTION_COMMANDS - args.grasp_steps - args.min_lift_steps
    if not 1 <= args.max_open_steps <= max_open_allowed:
        raise ValueError(f"max_open_steps must be <= {max_open_allowed} to preserve lift supervision")
    if args.descent_jitter_mm > args.close_xy_mm:
        raise ValueError("descent jitter must not exceed close XY threshold")


def _summary(output: Path, args: argparse.Namespace, accepted: int) -> dict:
    rows = []
    manifest = output / "generation_manifest.jsonl"
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    accepted_rows = [row for row in rows if row.get("reason") == "accepted"]
    return {
        "dataset_kind": DATASET_KIND,
        "output": str(output),
        "requested_episodes": args.num_episodes,
        "accepted_episodes": accepted,
        "attempts": len(rows),
        "reasons": dict(Counter(row.get("reason", "unknown") for row in rows)),
        "anchor_stages": dict(Counter(row.get("anchor_stage") for row in accepted_rows)),
        "offset_bins": dict(Counter(row.get("offset_bin") for row in accepted_rows)),
        "final_spatial_slots": dict(Counter(row.get("final_spatial_slot") for row in accepted_rows)),
        "offset_sectors": dict(Counter(str(row.get("offset_sector")) for row in accepted_rows)),
        "prefix_steps": dict(Counter(str(row.get("prefix_steps")) for row in accepted_rows)),
        "training_contract": {
            "model_generated_actions_stored": False,
            "anchor_actions_stored": False,
            "perturbation_actions_stored": False,
            "supervised_action_source": "oracle_only",
            "z_and_gripper_gated_by_live_xy_error": True,
            "first_supervised_observation_frame": 60,
            "episode_frames": EXPECTED_EPISODE_FRAMES,
        },
        "settings": {**vars(args), "output": str(output)},
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    _validate_args(args)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "generation_manifest.jsonl"
    existing = sorted(output.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if existing and not args.resume:
        raise FileExistsError(f"{output} already contains {len(existing)} episodes")
    accepted = len(existing)
    attempted = set()
    if manifest_path.exists():
        attempted = {
            int(json.loads(line)["attempt_index"]) for line in manifest_path.read_text().splitlines() if line.strip()
        }

    original_append = legacy.base._append_observation
    visual_guard = _VisualGuard(original_append)
    legacy.base._append_observation = visual_guard
    legacy.base._policy_input = legacy.fixed_eef._fixed_history_policy_input
    from openpi_client import websocket_client_policy

    shell = legacy.base._import_shellgame_tools(args.robosuite_root)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    policy_args = legacy._policy_args(args)
    start_time = time.time()
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest:
            for attempt_index in range(args.max_attempts):
                if accepted >= args.num_episodes:
                    break
                if attempt_index in attempted:
                    continue
                audit, payload = _attempt(
                    shell,
                    client,
                    args,
                    policy_args,
                    visual_guard,
                    attempt_index=attempt_index,
                    accepted_index=accepted,
                )
                if payload is not None:
                    episode_dir = output / f"episode_{accepted:06d}"
                    legacy._save_episode(
                        episode_dir,
                        observations=payload[0],
                        actions=payload[1],
                        action_mask=payload[2],
                        phase_ids=payload[3],
                        supervision_source=payload[4],
                        metadata=payload[5],
                        initial_ball_cup=payload[6],
                        final_ball_cup=payload[7],
                        fps=args.fps,
                    )
                    audit["accepted_index"] = accepted
                    accepted += 1
                manifest.write(json.dumps(audit, sort_keys=True) + "\n")
                manifest.flush()
                logging.info(
                    "attempt=%d reason=%s accepted=%d/%d stage=%s offset=%s elapsed=%.1fmin",
                    attempt_index,
                    audit["reason"],
                    accepted,
                    args.num_episodes,
                    audit.get("anchor_stage"),
                    None if "switch_offset_m" not in audit else f"{audit['switch_offset_m'] * 1_000:.1f}mm",
                    (time.time() - start_time) / 60.0,
                )
    finally:
        legacy.base._append_observation = original_append

    summary = _summary(output, args, accepted)
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if accepted < args.num_episodes:
        raise RuntimeError(f"Generated only {accepted}/{args.num_episodes} episodes after {args.max_attempts} attempts")
    logging.info("Generation complete: %s", json.dumps(summary, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
