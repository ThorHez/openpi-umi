"""Generate balanced absolute-EEF corrections with XY control through descent.

The learned policy is used only to select the cup and visit an approach state.
An unsaved controller perturbation then creates a known 5--35 mm lateral
offset.  Stored action supervision starts at frame 60 and is Oracle-only:
three fixed-Z recenter commands, fifty continuously XY-centred descent
commands, ten close commands, and thirty-one vertical lift commands.
"""

# Private helpers are reused to keep observations, controller semantics, and
# serialization identical to the previously validated V3 generator.
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

import generate_onpolicy_eef_correction_dataset as legacy
import generate_onpolicy_eef_correction_dataset_v2 as v2
import generate_onpolicy_eef_correction_dataset_v3 as v3
import numpy as np

DATASET_KIND = "onpolicy_eef_continuous_descent_v4"
EXPECTED_EPISODE_FRAMES = 155
CORRECTION_COMMANDS = EXPECTED_EPISODE_FRAMES - legacy.HISTORY_FRAMES - 1
DEFAULT_PREFIX_STEPS = (18, 24, 30, 36, 39, 42)
FINAL_SPATIAL_SLOTS = ("left", "middle", "right")
OFFSET_BINS_MM = {
    "small": (5.0, 12.0),
    "medium": (12.0, 22.0),
    "large": (22.0, 35.0),
}
OFFSET_BIN_CYCLE = ("small",) * 25 + ("medium",) * 45 + ("large",) * 30


class VisualCorruptionError(RuntimeError):
    """Raised when MuJoCo returns a frame that is neither continuous nor a flip."""


class _VisualGuard:
    def __init__(self, original):
        self.original = original
        self.previous: dict[str, np.ndarray] = {}
        self.flip_repairs = 0

    def reset(self) -> None:
        self.previous.clear()
        self.flip_repairs = 0

    def __call__(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        history = args[4]
        item = history[-1]
        for key in ("base", "wrist"):
            image = np.asarray(item[key], dtype=np.uint8)
            previous = self.previous.get(key)
            if previous is not None and previous.shape == image.shape:
                current_f = image.astype(np.float32)
                previous_f = previous.astype(np.float32)
                direct = float(np.mean(np.abs(current_f - previous_f)))
                rotated_image = image[::-1, ::-1]
                rotated = float(np.mean(np.abs(rotated_image.astype(np.float32) - previous_f)))
                if rotated + 5.0 < direct:
                    image = np.ascontiguousarray(rotated_image)
                    item[key] = image
                    self.flip_repairs += 1
                    direct = rotated
                if direct > 45.0:
                    raise VisualCorruptionError(f"{key} discontinuity MAD={direct:.1f} (possible snow frame)")
            self.previous[key] = image.copy()
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=240)
    parser.add_argument("--max-attempts", type=int, default=1920)
    parser.add_argument("--dataset-seed", type=int, default=260818)
    parser.add_argument("--policy-checkpoint-label", required=True)
    parser.add_argument("--replan-steps", type=int, default=3)
    parser.add_argument(
        "--prefix-steps",
        default=",".join(str(value) for value in DEFAULT_PREFIX_STEPS),
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-safe-height-mm", type=float, default=60.0)
    parser.add_argument("--max-safe-height-mm", type=float, default=240.0)
    parser.add_argument("--min-open-width-m", type=float, default=0.04)
    parser.add_argument("--perturb-steps", type=int, default=6)
    parser.add_argument("--offset-bin-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--pre-descent-steps", type=int, default=3)
    parser.add_argument("--descend-steps", type=int, default=50)
    parser.add_argument("--grasp-steps", type=int, default=10)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--descent-jitter-mm", type=float, default=2.5)
    parser.add_argument("--max-preclose-xy-mm", type=float, default=5.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _prefix_steps(args: argparse.Namespace) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in args.prefix_steps.split(",") if item.strip())
    if not values or any(value <= 10 for value in values):
        raise ValueError(f"prefix steps must all be > 10, got {values}")
    return values


def _balanced_design(args: argparse.Namespace, accepted_index: int) -> dict:
    """Return the fixed quota assigned to one accepted episode slot.

    The final spatial cup is the outer balance dimension.  The index within
    that cup independently covers offset magnitude and direction.  Prefix
    offsets 0/2/4 make all six approach depths globally balanced without
    coupling a spatial cup to only two depths.
    """
    slot_index = accepted_index % len(FINAL_SPATIAL_SLOTS)
    within_slot = accepted_index // len(FINAL_SPATIAL_SLOTS)
    prefixes = _prefix_steps(args)
    return {
        "final_spatial_slot": FINAL_SPATIAL_SLOTS[slot_index],
        "within_spatial_slot_index": within_slot,
        "prefix_steps": prefixes[(within_slot + 2 * slot_index) % len(prefixes)],
        "offset_bin": OFFSET_BIN_CYCLE[(within_slot * 37) % 100],
        "offset_sector": within_slot % 8,
    }


def _offset_spec(args: argparse.Namespace, attempt_index: int, accepted_index: int):
    design = _balanced_design(args, accepted_index)
    offset_bin = design["offset_bin"]
    low_mm, high_mm = OFFSET_BINS_MM[offset_bin]
    rng = np.random.default_rng(np.random.SeedSequence([args.dataset_seed, attempt_index, accepted_index, 407]))
    sector = design["offset_sector"]
    angle = 2.0 * math.pi * sector / 8.0 + rng.uniform(-math.pi / 16, math.pi / 16)
    radius_m = float(rng.uniform(low_mm, high_mm) / 1000.0)
    perturb_xy = radius_m * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    jitter_radius = float(rng.uniform(0.0, args.descent_jitter_mm / 1000.0))
    jitter_angle = float(rng.uniform(-math.pi, math.pi))
    descent_jitter_xy = jitter_radius * np.array([math.cos(jitter_angle), math.sin(jitter_angle)], dtype=np.float64)
    return offset_bin, perturb_xy, descent_jitter_xy


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
    design = _balanced_design(args, accepted_index)
    prefix_steps = design["prefix_steps"]
    offset_bin, perturb_xy, descent_jitter_xy = _offset_spec(args, attempt_index, accepted_index)
    ep_args = legacy.base._episode_namespace(policy_args, seed=episode_seed, initial_ball_cup=initial, num_swaps=3)
    env = shell.make_env(ep_args)
    history: list[dict] = []
    discarded_video: list[np.ndarray] = []
    reject: dict = {
        "attempt_index": attempt_index,
        "episode_seed": episode_seed,
        "initial_ball_cup": initial,
        "prefix_steps": prefix_steps,
        "offset_bin": offset_bin,
        "required_final_spatial_slot": design["final_spatial_slot"],
        "offset_sector": design["offset_sector"],
    }

    try:
        try:
            meta = legacy.base._run_scripted_observation(
                shell, env, ep_args, policy_args, history, discarded_video, client=client
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

            # Current-policy commands are rollout context only and are discarded.
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
            reject.update(target_cup=meta["target_cup"], selected_cup=selected, selection_votes=votes)
            if selected is None:
                reject["reason"] = "no_selection_vote"
                return reject, None
            if selected != meta["target_cup"]:
                reject["reason"] = "selection_incorrect"
                return reject, None

            # Deliberately create the desired offset, but never serialize these
            # commands or their intermediate observations as training targets.
            cups = legacy.base._cup_positions(shell, env)
            selected_pos = np.asarray(cups[selected], dtype=np.float64)
            pre_perturb_eef = np.asarray(history[-1]["eef_pos"], dtype=np.float64)
            grasp_z = float(env.cup_handle_grasp_z() + ep_args.robot_grasp_z_offset)
            safe_height_m = float(pre_perturb_eef[2] - grasp_z)
            if not args.min_safe_height_mm / 1000.0 <= safe_height_m <= args.max_safe_height_mm / 1000.0:
                reject["reason"] = "unsafe_switch_height"
                reject["switch_safe_height_m"] = safe_height_m
                return reject, None

            perturb_target = np.array(
                [
                    selected_pos[0] + perturb_xy[0],
                    selected_pos[1] + perturb_xy[1],
                    pre_perturb_eef[2],
                ],
                dtype=np.float64,
            )
            for _ in range(args.perturb_steps):
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

            switch = legacy._snapshot(history[-1])
            selected_pos = np.asarray(legacy.base._cup_positions(shell, env)[selected], dtype=np.float64)
            switch_eef = np.asarray(switch["eef_pos"], dtype=np.float64)
            offset_m = float(np.linalg.norm(switch_eef[:2] - selected_pos[:2]))
            safe_height_m = float(switch_eef[2] - grasp_z)
            switch_contacts = legacy.contact_utils._finger_contact_count(env, selected)
            low_mm, high_mm = OFFSET_BINS_MM[offset_bin]
            tolerance_m = args.offset_bin_tolerance_mm / 1000.0
            reject.update(
                switch_offset_m=offset_m,
                requested_offset_m=float(np.linalg.norm(perturb_xy)),
                switch_safe_height_m=safe_height_m,
                switch_gripper_width_m=switch["gripper_width"],
                switch_contacts=switch_contacts,
                visual_flip_repairs=visual_guard.flip_repairs,
            )
            if not low_mm / 1000.0 - tolerance_m <= offset_m <= high_mm / 1000.0 + tolerance_m:
                reject["reason"] = "perturb_offset_outside_bin"
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
            phase_ids = [0] * 10 + [1] * 10 + [2] * 30 + [3] * 10 + [legacy.PHASE_RECENTER]
            stage_metrics: list[dict] = []

            oracle_xy = selected_pos[:2] + descent_jitter_xy
            recenter_pos = np.array([*oracle_xy, switch_eef[2]], dtype=np.float64)
            grasp_pos = np.array([*oracle_xy, grasp_z], dtype=np.float64)
            lift_pos = np.array([*oracle_xy, grasp_z + args.lift_height], dtype=np.float64)

            def execute(stage: str, phase_id: int, step: int, target: np.ndarray, close: bool):
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
                xy_error = float(np.linalg.norm(actual[:2] - live_cup[:2]))
                stage_metrics.append(
                    {
                        "stage": stage,
                        "stage_step": step,
                        "eef_to_selected_xy_m": xy_error,
                        "selected_finger_contacts": legacy.contact_utils._finger_contact_count(env, selected),
                    }
                )
                return xy_error

            for step in range(args.pre_descent_steps):
                execute("recenter", legacy.PHASE_RECENTER, step, recenter_pos, False)

            # Crucial V4 change: every descent command remains centred in XY;
            # V3 only corrected before descent and then replayed a fixed segment.
            for step in range(args.descend_steps):
                live_cup = np.asarray(legacy.base._cup_positions(shell, env)[selected], dtype=np.float64)
                target = np.array(
                    [
                        live_cup[0] + descent_jitter_xy[0],
                        live_cup[1] + descent_jitter_xy[1],
                        (1.0 - v2._linear_progress(step, args.descend_steps)) * switch_eef[2]
                        + v2._linear_progress(step, args.descend_steps) * grasp_z,
                    ],
                    dtype=np.float64,
                )
                execute("descend", legacy.PHASE_DESCEND, step, target, False)

            preclose_xy_m = stage_metrics[-1]["eef_to_selected_xy_m"]
            reject["preclose_xy_error_m"] = preclose_xy_m
            if preclose_xy_m > args.max_preclose_xy_mm / 1000.0:
                reject["reason"] = "preclose_xy_too_large"
                return reject, None

            for step in range(args.grasp_steps):
                execute("grasp", legacy.PHASE_GRASP, step, grasp_pos, True)
            lift_steps = CORRECTION_COMMANDS - args.pre_descent_steps - args.descend_steps - args.grasp_steps
            for step in range(lift_steps):
                progress = v2._linear_progress(step, lift_steps)
                execute(
                    "lift",
                    legacy.PHASE_LIFT,
                    step,
                    (1.0 - progress) * grasp_pos + progress * lift_pos,
                    True,
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
                raise RuntimeError(f"V4 episode must have {EXPECTED_EPISODE_FRAMES} frames, got {len(observations)}")

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
                    "purpose": "select cup and visit deployment approach distribution",
                },
                "perturbation": {
                    "offset_bin": offset_bin,
                    "requested_offset_xy_m": perturb_xy.tolist(),
                    "requested_offset_m": float(np.linalg.norm(perturb_xy)),
                    "measured_offset_m": offset_m,
                    "executed_steps": args.perturb_steps,
                    "actions_saved_to_training_file": False,
                    "intermediate_frames_saved_to_training_file": False,
                },
                "supervision_contract": {
                    "stored_frames_0_59": "scripted_history_context_only",
                    "stored_frame_60": "post_policy_and_unsaved_perturbation_context_only",
                    "stored_frames_61_plus": "observations_after_oracle_commands",
                    "action_mask_true_source": "oracle_only",
                    "first_training_pair": "observation[60] -> oracle_action[61]",
                    "model_or_perturb_actions_supervised": False,
                },
                "switch": {
                    "selected_cup": selected,
                    "target_cup": meta["target_cup"],
                    "selection_votes": votes,
                    "prefix_steps": prefix_steps,
                    "offset_m": offset_m,
                    "safe_height_m": safe_height_m,
                    "gripper_width_m": switch["gripper_width"],
                    "contacts": switch_contacts,
                },
                "oracle": {
                    "xy_mode": "live_cup_center_plus_fixed_episode_jitter_through_descent",
                    "descent_jitter_xy_m": descent_jitter_xy.tolist(),
                    "pre_descent_steps": args.pre_descent_steps,
                    "descend_steps": args.descend_steps,
                    "grasp_steps": args.grasp_steps,
                    "lift_steps": lift_steps,
                    "preclose_xy_error_m": preclose_xy_m,
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
    if args.num_episodes % 600:
        raise ValueError(
            "num_episodes must be divisible by 600 for exact per-spatial-slot offset-bin and 8-direction balance"
        )
    if args.num_episodes % len(prefixes):
        raise ValueError("num_episodes must be divisible by the number of prefix heights")
    if args.pre_descent_steps < args.replan_steps:
        raise ValueError("pre_descent_steps must cover the first deployed chunk")
    lift_steps = CORRECTION_COMMANDS - args.pre_descent_steps - args.descend_steps - args.grasp_steps
    if lift_steps <= 0:
        raise ValueError("Stage lengths leave no commands for lift")
    if args.descent_jitter_mm > args.max_preclose_xy_mm:
        raise ValueError("descent_jitter_mm must not exceed max_preclose_xy_mm")


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
        "offset_bins": dict(Counter(row.get("offset_bin") for row in accepted_rows)),
        "final_spatial_slots": dict(Counter(row.get("final_spatial_slot") for row in accepted_rows)),
        "offset_sectors": dict(Counter(str(row.get("offset_sector")) for row in accepted_rows)),
        "prefix_steps": dict(Counter(str(row.get("prefix_steps")) for row in accepted_rows)),
        "training_contract": {
            "model_generated_actions_stored": False,
            "perturbation_actions_stored": False,
            "supervised_action_source": "oracle_only",
            "continuous_xy_supervision_through_descent": True,
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
                    "attempt=%d reason=%s accepted=%d/%d bin=%s offset=%s elapsed=%.1fmin",
                    attempt_index,
                    audit["reason"],
                    accepted,
                    args.num_episodes,
                    audit.get("offset_bin"),
                    None if "switch_offset_m" not in audit else f"{audit['switch_offset_m'] * 1000:.1f}mm",
                    (time.time() - start_time) / 60,
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
