"""Generate current-policy, multi-height EEF correction demonstrations.

This dataset fixes the failure mode measured in the V3 closed-loop policy:
the policy knows the correct cup, but starts descending while 10--16 mm off
centre and strikes the rim before closing.  Every accepted episode therefore
obeys a stronger deployment contract than the earlier correction datasets:

* the current V3 policy visits a real on-policy switch state;
* switch states are sampled at several approach depths, not only step 30;
* the complete first deployed chunk (three commands) is XY-only at fixed Z;
* recentering continues until the measured EEF/cup XY error is <= 5 mm;
* descent is forbidden before that measured threshold is reached; and
* model-generated prefix actions remain context-only and are never targets.

The suffix length is kept at 94 commands, so every episode still contains 155
frames and remains compatible with ``last_episode_frame=154``.
"""

# Private helpers are reused intentionally to keep rendering, observations,
# controller semantics, and serialization identical to the validated pipeline.
# ruff: noqa: FBT001, FBT003, SLF001

from __future__ import annotations

import argparse
from collections import Counter
from collections import deque
import dataclasses
import json
import logging
from pathlib import Path
import time

import generate_onpolicy_eef_correction_dataset as legacy
import generate_onpolicy_eef_correction_dataset_v2 as v2
import numpy as np

DATASET_KIND = "onpolicy_eef_correction_multiheight_holdz_v3"
DEPLOYMENT_REPLAN_STEPS = 3
EXPECTED_EPISODE_FRAMES = 155
CORRECTION_COMMANDS = EXPECTED_EPISODE_FRAMES - legacy.HISTORY_FRAMES - 1
DEFAULT_PREFIX_STEPS = (18, 24, 30, 36, 39, 42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=600)
    parser.add_argument("--max-attempts", type=int, default=2400)
    parser.add_argument("--dataset-seed", type=int, default=260816)
    parser.add_argument("--policy-checkpoint-label", required=True)
    parser.add_argument("--replan-steps", type=int, default=DEPLOYMENT_REPLAN_STEPS)
    parser.add_argument(
        "--prefix-steps",
        default=",".join(str(value) for value in DEFAULT_PREFIX_STEPS),
        help="Comma-separated current-policy rollout lengths used for switch states.",
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-offset-mm", type=float, default=6.0)
    parser.add_argument("--max-offset-mm", type=float, default=30.0)
    parser.add_argument("--min-safe-height-mm", type=float, default=60.0)
    parser.add_argument("--max-safe-height-mm", type=float, default=240.0)
    parser.add_argument("--min-open-width-m", type=float, default=0.04)
    parser.add_argument("--xy-threshold-mm", type=float, default=5.0)
    parser.add_argument("--min-recenter-steps", type=int, default=DEPLOYMENT_REPLAN_STEPS)
    parser.add_argument("--max-recenter-steps", type=int, default=12)
    parser.add_argument("--descend-steps", type=int, default=30)
    parser.add_argument("--grasp-steps", type=int, default=15)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--min-entry-descent-mm", type=float, default=3.0)
    parser.add_argument("--max-hold-z-drift-mm", type=float, default=0.1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _prefix_steps(args: argparse.Namespace) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in args.prefix_steps.split(",") if item.strip())
    if not values or any(value <= 10 for value in values):
        raise ValueError(f"prefix steps must all be > 10, got {values}")
    return values


def _phase_ranges(phase_ids: list[int]) -> dict[str, list[int]]:
    result = {}
    names = {
        legacy.PHASE_RECENTER: "recenter",
        legacy.PHASE_DESCEND: "descend",
        legacy.PHASE_GRASP: "grasp",
        legacy.PHASE_LIFT: "lift",
    }
    values = np.asarray(phase_ids, dtype=np.int64)
    for phase_id, name in names.items():
        indices = np.flatnonzero(values == phase_id)
        if len(indices):
            result[name] = [int(indices[0]), int(indices[-1])]
    return result


def _attempt(
    shell,
    client,
    args: argparse.Namespace,
    policy_args: legacy.base.Args,
    *,
    attempt_index: int,
    accepted_index: int,
) -> tuple[dict, tuple | None]:
    episode_seed, initial = legacy._episode_randomness(
        args.dataset_seed, attempt_index, shell.CUP_NAMES
    )
    prefix_steps = _prefix_steps(args)[attempt_index % len(_prefix_steps(args))]
    ep_args = legacy.base._episode_namespace(
        policy_args, seed=episode_seed, initial_ball_cup=initial, num_swaps=3
    )
    env = shell.make_env(ep_args)
    history: list[dict] = []
    discarded_video: list[np.ndarray] = []
    reject: dict = {
        "attempt_index": attempt_index,
        "episode_seed": episode_seed,
        "initial_ball_cup": initial,
        "prefix_steps": prefix_steps,
    }

    try:
        meta = legacy.base._run_scripted_observation(
            shell, env, ep_args, policy_args, history, discarded_video, client=client
        )
        if len(history) != legacy.HISTORY_FRAMES:
            raise RuntimeError(
                f"Expected {legacy.HISTORY_FRAMES} scripted frames, got {len(history)}"
            )
        fixed_history = [legacy._snapshot(item) for item in history]
        canonical_quat = np.asarray(history[0]["eef_quat"], dtype=np.float32)
        start_eef_pos = np.asarray(history[0]["eef_pos"], dtype=np.float32)
        start_eef_quat = canonical_quat.copy()
        action_plan: deque = deque()
        gripper_action = policy_args.default_gripper_action
        votes = dict.fromkeys(shell.CUP_NAMES, 0)
        distance_sums = dict.fromkeys(shell.CUP_NAMES, 0.0)

        # Use the current policy only to visit its deployment distribution.
        # None of these commands or intermediate frames are serialized.
        for step in range(prefix_steps):
            env_action, gripper_action, _ = legacy.base._policy_env_action(
                shell,
                env,
                history,
                start_eef_pos,
                start_eef_quat,
                action_plan,
                gripper_action,
                client=client,
                args=policy_args,
                prompt=legacy.GRASP_PROMPT,
            )
            env.step(env_action)
            sink: list[np.ndarray] = []
            legacy.base._append_observation(
                shell,
                env,
                ep_args,
                meta["wrist_camera_name"],
                history,
                sink,
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

        switch = legacy._snapshot(history[-1])
        cups = legacy.base._cup_positions(shell, env)
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

        selected_pos = np.asarray(cups[selected], dtype=np.float64)
        switch_eef = np.asarray(switch["eef_pos"], dtype=np.float64)
        offset_m = float(np.linalg.norm(switch_eef[:2] - selected_pos[:2]))
        grasp_z = float(env.cup_handle_grasp_z() + ep_args.robot_grasp_z_offset)
        safe_height_m = float(switch_eef[2] - grasp_z)
        switch_contacts = legacy.contact_utils._finger_contact_count(env, selected)
        reject.update(
            switch_offset_m=offset_m,
            switch_safe_height_m=safe_height_m,
            switch_gripper_width_m=switch["gripper_width"],
            switch_contacts=switch_contacts,
        )
        if offset_m < args.min_offset_mm / 1000.0:
            reject["reason"] = "offset_too_small"
            return reject, None
        if offset_m > args.max_offset_mm / 1000.0:
            reject["reason"] = "offset_too_large"
            return reject, None
        if safe_height_m < args.min_safe_height_mm / 1000.0:
            reject["reason"] = "unsafe_switch_height"
            return reject, None
        if safe_height_m > args.max_safe_height_mm / 1000.0:
            reject["reason"] = "switch_height_too_high"
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
        stage_commands: dict[str, list[np.ndarray]] = {
            "recenter": [],
            "descend": [],
            "grasp": [],
            "lift": [],
        }

        recenter_pos = np.array(
            [selected_pos[0], selected_pos[1], switch_eef[2]], dtype=np.float64
        )
        grasp_pos = np.array([selected_pos[0], selected_pos[1], grasp_z], dtype=np.float64)
        lift_pos = np.array(
            [selected_pos[0], selected_pos[1], grasp_z + args.lift_height], dtype=np.float64
        )

        def execute(
            stage: str,
            phase_id: int,
            stage_step: int,
            target_pos: np.ndarray,
            close: bool,
        ) -> float:
            command = shell.make_robot_action(
                env,
                target_pos=np.asarray(target_pos, dtype=np.float64),
                target_quat=canonical_quat,
                gripper_action=1.0 if close else -1.0,
            )
            env.step(command)
            sink: list[np.ndarray] = []
            legacy.base._append_observation(
                shell,
                env,
                ep_args,
                meta["wrist_camera_name"],
                history,
                sink,
                resize_size=args.width,
            )
            observation = legacy._snapshot(history[-1])
            command = np.asarray(command, dtype=np.float32)
            observations.append(observation)
            actions.append(command)
            action_mask.append(True)
            supervision.append(legacy.SUPERVISION_ORACLE)
            phase_ids.append(phase_id)
            stage_commands[stage].append(command)
            actual = np.asarray(observation["eef_pos"], dtype=np.float64)
            live_cup = np.asarray(legacy.base._cup_positions(shell, env)[selected], dtype=np.float64)
            xy_error = float(np.linalg.norm(actual[:2] - live_cup[:2]))
            stage_metrics.append(
                {
                    "stage": stage,
                    "stage_step": stage_step,
                    "eef_to_selected_xy_m": xy_error,
                    "selected_finger_contacts": legacy.contact_utils._finger_contact_count(
                        env, selected
                    ),
                }
            )
            return xy_error

        # Hold the exact switch Z for a full deployed chunk, then continue
        # holding it until the measured controller state is genuinely centred.
        recenter_errors = []
        for step in range(args.max_recenter_steps):
            error = execute(
                "recenter", legacy.PHASE_RECENTER, step, recenter_pos, False
            )
            recenter_errors.append(error)
            if step + 1 >= args.min_recenter_steps and error <= args.xy_threshold_mm / 1000.0:
                break
        recenter_steps = len(recenter_errors)
        post_recenter_xy_error_m = recenter_errors[-1]
        if post_recenter_xy_error_m > args.xy_threshold_mm / 1000.0:
            reject["reason"] = "recenter_did_not_converge"
            reject["recenter_steps"] = recenter_steps
            reject["post_recenter_xy_error_m"] = post_recenter_xy_error_m
            return reject, None

        for step in range(args.descend_steps):
            execute(
                "descend",
                legacy.PHASE_DESCEND,
                step,
                (1.0 - v2._linear_progress(step, args.descend_steps)) * recenter_pos
                + v2._linear_progress(step, args.descend_steps) * grasp_pos,
                False,
            )
        for step in range(args.grasp_steps):
            execute("grasp", legacy.PHASE_GRASP, step, grasp_pos, True)

        lift_steps = CORRECTION_COMMANDS - recenter_steps - args.descend_steps - args.grasp_steps
        if lift_steps <= 0:
            raise RuntimeError(f"No room for lift stage: recenter_steps={recenter_steps}")
        for step in range(lift_steps):
            progress = v2._linear_progress(step, lift_steps)
            execute(
                "lift",
                legacy.PHASE_LIFT,
                step,
                (1.0 - progress) * grasp_pos + progress * lift_pos,
                True,
            )

        recenter_commands = np.asarray(stage_commands["recenter"], dtype=np.float64)
        descend_commands = np.asarray(stage_commands["descend"], dtype=np.float64)
        hold_z_drift_m = float(np.max(np.abs(recenter_commands[:, 2] - switch_eef[2])))
        first_chunk_hold_z_drift_m = float(
            np.max(np.abs(recenter_commands[: args.replan_steps, 2] - switch_eef[2]))
        )
        entry_descent_m = float(switch_eef[2] - descend_commands[0, 2])
        reject.update(
            recenter_steps=recenter_steps,
            post_recenter_xy_error_m=post_recenter_xy_error_m,
            hold_z_drift_m=hold_z_drift_m,
            first_chunk_hold_z_drift_m=first_chunk_hold_z_drift_m,
            entry_descent_m=entry_descent_m,
        )
        if hold_z_drift_m > args.max_hold_z_drift_mm / 1000.0:
            reject["reason"] = "hold_z_command_drift"
            return reject, None
        if recenter_steps < args.replan_steps:
            reject["reason"] = "first_chunk_not_all_recenter"
            return reject, None
        if entry_descent_m < args.min_entry_descent_mm / 1000.0:
            reject["reason"] = "weak_descent_entry"
            return reject, None

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
            raise RuntimeError(
                f"V3 episode must have {EXPECTED_EPISODE_FRAMES} frames, got {len(observations)}"
            )

        metadata = {
            "env": "ShellGame",
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "seed": episode_seed,
            "attempt_index": attempt_index,
            "accepted_index": accepted_index,
            "initial_ball_cup": initial,
            "target_cup_identity": meta["target_cup"],
            "final_ball_cup": meta["final_ball_cup"],
            "swaps": [list(pair) for pair in meta["swaps"]],
            "phase_ranges": _phase_ranges(phase_ids),
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
                "purpose": "visit current-policy multi-height offset states",
            },
            "supervision_contract": {
                "stored_frames_0_59": "scripted_history_context_only",
                "stored_frame_60": "post-current-policy offset observation context_only",
                "stored_frames_61_plus": "observations_after_oracle commands",
                "action_mask_true_source": "oracle_only",
                "first_training_pair": "observation[60] -> oracle_action[61]",
                "descent_allowed_only_after_measured_xy_threshold": True,
            },
            "closed_loop_contract": {
                "deployment_replan_steps": args.replan_steps,
                "episode_frames": EXPECTED_EPISODE_FRAMES,
                "last_episode_frame": EXPECTED_EPISODE_FRAMES - 1,
                "first_chunk_is_xy_only_hold_z": True,
                "recenter_steps": recenter_steps,
                "xy_threshold_m": args.xy_threshold_mm / 1000.0,
                "post_recenter_xy_error_m": post_recenter_xy_error_m,
                "hold_z_drift_m": hold_z_drift_m,
                "first_chunk_hold_z_drift_m": first_chunk_hold_z_drift_m,
                "entry_descent_m": entry_descent_m,
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
                "recenter_steps": recenter_steps,
                "recenter_min_steps": args.min_recenter_steps,
                "recenter_max_steps": args.max_recenter_steps,
                "recenter_exit_xy_error_m": post_recenter_xy_error_m,
                "recenter_z_mode": "strict_hold_switch_z_until_measured_xy_threshold",
                "descend_steps": args.descend_steps,
                "grasp_steps": args.grasp_steps,
                "lift_steps": lift_steps,
                "lift_height_m": args.lift_height,
                "final_success_stats": success_stats,
                "max_bilateral_contact_run": legacy._max_contact_run(stage_metrics),
            },
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
    finally:
        env.close()


def _validate_args(args: argparse.Namespace) -> None:
    prefixes = _prefix_steps(args)
    if args.width != args.height:
        raise ValueError("Current fixed-history policy expects square images")
    if args.num_episodes <= 0 or args.max_attempts < args.num_episodes:
        raise ValueError("Require 0 < num_episodes <= max_attempts")
    if args.replan_steps != DEPLOYMENT_REPLAN_STEPS:
        raise ValueError(f"V3 requires replan_steps={DEPLOYMENT_REPLAN_STEPS}")
    if args.min_recenter_steps < args.replan_steps:
        raise ValueError("min_recenter_steps must cover the complete first deployed chunk")
    if args.max_recenter_steps < args.min_recenter_steps:
        raise ValueError("max_recenter_steps must be >= min_recenter_steps")
    if CORRECTION_COMMANDS - args.max_recenter_steps - args.descend_steps - args.grasp_steps <= 0:
        raise ValueError("Stage lengths leave no commands for lift")
    if not 0 < args.xy_threshold_mm < args.min_offset_mm:
        raise ValueError("Require 0 < xy_threshold_mm < min_offset_mm")
    if args.min_safe_height_mm >= args.max_safe_height_mm:
        raise ValueError("min_safe_height_mm must be below max_safe_height_mm")
    if args.num_episodes % len(prefixes) != 0:
        raise ValueError(
            "num_episodes must be divisible by the number of prefix heights so "
            f"the formal dataset is balanced; got {args.num_episodes} and {len(prefixes)}"
        )
    logging.info("V3 prefix steps: %s", prefixes)


def _summarize(output: Path, args: argparse.Namespace, accepted: int) -> dict:
    base = legacy._summarize(output, args, accepted)
    manifest = output / "generation_manifest.jsonl"
    rows = []
    if manifest.exists():
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    accepted_rows = [row for row in rows if row.get("reason") == "accepted"]

    def stats(key: str) -> dict | None:
        values = [float(row[key]) for row in accepted_rows if key in row]
        if not values:
            return None
        return {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }

    base.update(
        dataset_kind=DATASET_KIND,
        policy_checkpoint_label=args.policy_checkpoint_label,
        prefix_steps_distribution=dict(
            sorted(Counter(int(row["prefix_steps"]) for row in accepted_rows).items())
        ),
        switch_safe_height_m=stats("switch_safe_height_m"),
        recenter_steps=stats("recenter_steps"),
        post_recenter_xy_error_m=stats("post_recenter_xy_error_m"),
        first_chunk_hold_z_drift_m=stats("first_chunk_hold_z_drift_m"),
        entry_descent_m=stats("entry_descent_m"),
        contract={
            "first_chunk_is_xy_only_hold_z": True,
            "measured_xy_threshold_m": args.xy_threshold_mm / 1000.0,
            "descent_before_threshold_allowed": False,
            "episode_frames": EXPECTED_EPISODE_FRAMES,
            "last_episode_frame": EXPECTED_EPISODE_FRAMES - 1,
        },
    )
    return base


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
    prefixes = _prefix_steps(args)
    prefix_quota = {value: args.num_episodes // len(prefixes) for value in prefixes}
    accepted_by_prefix = dict.fromkeys(prefixes, 0)
    for episode_dir in existing:
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        value = int(metadata["switch"]["prefix_steps"])
        if value not in accepted_by_prefix:
            raise ValueError(f"{episode_dir}: unexpected prefix_steps={value}")
        accepted_by_prefix[value] += 1
    attempted_indices: set[int] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                attempted_indices.add(int(json.loads(line)["attempt_index"]))

    legacy.base._policy_input = legacy.fixed_eef._fixed_history_policy_input
    from openpi_client import websocket_client_policy

    shell = legacy.base._import_shellgame_tools(args.robosuite_root)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    policy_args = legacy._policy_args(args)
    start_time = time.time()

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for attempt_index in range(args.max_attempts):
            if accepted >= args.num_episodes:
                break
            if attempt_index in attempted_indices:
                continue
            candidate_prefix = prefixes[attempt_index % len(prefixes)]
            if accepted_by_prefix[candidate_prefix] >= prefix_quota[candidate_prefix]:
                continue
            audit, payload = _attempt(
                shell,
                client,
                args,
                policy_args,
                attempt_index=attempt_index,
                accepted_index=accepted,
            )
            if payload is not None:
                episode_dir = output / f"episode_{accepted:06d}"
                if episode_dir.exists():
                    raise FileExistsError(episode_dir)
                (
                    observations,
                    actions,
                    action_mask,
                    phase_ids,
                    supervision,
                    metadata,
                    initial,
                    final_ball_cup,
                ) = payload
                legacy._save_episode(
                    episode_dir,
                    observations=observations,
                    actions=actions,
                    action_mask=action_mask,
                    phase_ids=phase_ids,
                    supervision_source=supervision,
                    metadata=metadata,
                    initial_ball_cup=initial,
                    final_ball_cup=final_ball_cup,
                    fps=args.fps,
                )
                audit["accepted_index"] = accepted
                accepted += 1
                accepted_by_prefix[candidate_prefix] += 1
            manifest.write(json.dumps(audit, sort_keys=True) + "\n")
            manifest.flush()
            logging.info(
                "attempt=%d prefix=%d reason=%s accepted=%d/%d offset=%s height=%s recenter=%s elapsed=%.1fmin",
                attempt_index,
                audit["prefix_steps"],
                audit["reason"],
                accepted,
                args.num_episodes,
                None
                if "switch_offset_m" not in audit
                else f"{audit['switch_offset_m'] * 1000.0:.1f}mm",
                None
                if "switch_safe_height_m" not in audit
                else f"{audit['switch_safe_height_m'] * 1000.0:.1f}mm",
                audit.get("recenter_steps"),
                (time.time() - start_time) / 60.0,
            )

    summary = _summarize(output, args, accepted)
    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if accepted < args.num_episodes:
        raise RuntimeError(
            f"Generated only {accepted}/{args.num_episodes} accepted episodes "
            f"after {args.max_attempts} attempts"
        )
    logging.info("Generation complete: %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
