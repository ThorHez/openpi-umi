"""Generate replan-compatible on-policy absolute-EEF correction demonstrations.

This is a conservative V2 of ``generate_onpolicy_eef_correction_dataset``.
The old generator and old dataset remain untouched.

The important closed-loop contract is that inference executes only the first
three commands of every 16-command chunk.  Therefore V2 makes the transition
from lateral correction to descent observable inside those first commands:

* recenter is exactly two repeated, full-strength XY commands;
* recenter keeps the switch-state Z instead of moving to a fixed hover Z;
* the third command is already the first strictly downward command;
* descend uses endpoint-free linear progress, so its first command cannot be a
  duplicate hover command; and
* every accepted episode is audited against these properties before saving.

Model-prefix actions are still never stored or supervised.  As in V1, the
learned policy is used only to visit deployment-distribution offset states.
"""

# Private helpers are reused intentionally so rendering, preprocessing,
# controller semantics, serialization, and success checks remain identical to
# the validated V1 pipeline.
# ruff: noqa: FBT001, FBT003, SLF001

from __future__ import annotations

import argparse
from collections import deque
import dataclasses
import json
import logging
from pathlib import Path
import time

import generate_onpolicy_eef_correction_dataset as legacy
import numpy as np


DATASET_KIND = "onpolicy_offset_oracle_correction_replan_v2"
DEFAULT_RECENTER_STEPS = 2
DEPLOYMENT_REPLAN_STEPS = 3
EXPECTED_EPISODE_FRAMES = 155


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=900)
    parser.add_argument("--dataset-seed", type=int, default=260815)
    parser.add_argument("--replan-steps", type=int, default=DEPLOYMENT_REPLAN_STEPS)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-offset-mm", type=float, default=3.0)
    parser.add_argument("--max-offset-mm", type=float, default=35.0)
    parser.add_argument("--min-safe-height-mm", type=float, default=80.0)
    parser.add_argument("--min-open-width-m", type=float, default=0.04)
    parser.add_argument("--recenter-steps", type=int, default=DEFAULT_RECENTER_STEPS)
    parser.add_argument("--descend-steps", type=int, default=30)
    parser.add_argument("--grasp-steps", type=int, default=15)
    # 61 context/switch + 2 recenter + 30 descend + 15 grasp + 47 lift = 155.
    # Keeping the exact nominal episode length preserves the model's shared
    # last_episode_frame=154 temporal-loss contract in mixed-data training.
    parser.add_argument("--lift-steps", type=int, default=47)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--min-entry-descent-mm", type=float, default=3.0)
    parser.add_argument("--max-recenter-z-command-drift-mm", type=float, default=0.1)
    parser.add_argument("--max-post-recenter-xy-error-mm", type=float, default=25.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _linear_progress(index: int, count: int) -> float:
    """Return (0, 1] progress; the first command is never a repeated endpoint."""
    if count <= 0:
        raise ValueError(f"Stage length must be positive, got {count}")
    return float(index + 1) / float(count)


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

        # Visit the old model's real deployment distribution.  Never store its
        # actions or intermediate observations as training targets.
        for step in range(legacy.MODEL_APPROACH_STEPS):
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
            {
                "target_cup": meta["target_cup"],
                "selected_cup": selected,
                "selection_votes": votes,
            }
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
            {
                "switch_offset_m": offset_m,
                "switch_safe_height_m": safe_height_m,
                "switch_gripper_width_m": switch["gripper_width"],
                "switch_contacts": switch_contacts,
            }
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
        phase_ids = (
            [0] * 10
            + [1] * 10
            + [2] * 30
            + [3] * 10
            + [legacy.PHASE_CORRECTION_START]
        )
        stage_metrics: list[dict] = []
        stage_commands: dict[str, list[np.ndarray]] = {
            "recenter": [],
            "descend": [],
            "grasp": [],
            "lift": [],
        }

        # Crucially, recenter is pure XY at the *actual switch Z*.  Repeating
        # the full target for two controller steps is faster and less ambiguous
        # than a ten-step smooth interpolation toward a higher hover pose.
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
        ) -> None:
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
            live_cup = np.asarray(
                legacy.base._cup_positions(shell, env)[selected], dtype=np.float64
            )
            stage_metrics.append(
                {
                    "stage": stage,
                    "stage_step": stage_step,
                    "eef_to_selected_xy_m": float(
                        np.linalg.norm(actual[:2] - live_cup[:2])
                    ),
                    "selected_finger_contacts": legacy.contact_utils._finger_contact_count(
                        env, selected
                    ),
                }
            )

        for step in range(args.recenter_steps):
            execute(
                "recenter",
                legacy.PHASE_RECENTER,
                step,
                recenter_pos,
                False,
            )
        for step in range(args.descend_steps):
            progress = _linear_progress(step, args.descend_steps)
            execute(
                "descend",
                legacy.PHASE_DESCEND,
                step,
                (1.0 - progress) * recenter_pos + progress * grasp_pos,
                False,
            )
        for step in range(args.grasp_steps):
            execute("grasp", legacy.PHASE_GRASP, step, grasp_pos, True)
        for step in range(args.lift_steps):
            progress = _linear_progress(step, args.lift_steps)
            execute(
                "lift",
                legacy.PHASE_LIFT,
                step,
                (1.0 - progress) * grasp_pos + progress * lift_pos,
                True,
            )

        # Hard deployment audits.  With recenter_steps=2 and replan_steps=3,
        # the third command predicted from frame 60 must already descend.
        recenter_commands = np.asarray(stage_commands["recenter"], dtype=np.float64)
        descend_commands = np.asarray(stage_commands["descend"], dtype=np.float64)
        recenter_z_drift_m = float(
            np.max(np.abs(recenter_commands[:, 2] - switch_eef[2]))
        )
        entry_descent_m = float(switch_eef[2] - descend_commands[0, 2])
        first_replan_chunk = np.concatenate(
            [recenter_commands, descend_commands[:1]], axis=0
        )
        first_chunk_min_dz_m = float(
            np.min(first_replan_chunk[: args.replan_steps, 2] - switch_eef[2])
        )
        post_recenter_xy_error_m = float(
            stage_metrics[args.recenter_steps - 1]["eef_to_selected_xy_m"]
        )
        reject.update(
            {
                "recenter_z_command_drift_m": recenter_z_drift_m,
                "entry_descent_m": entry_descent_m,
                "first_replan_chunk_min_dz_m": first_chunk_min_dz_m,
                "post_recenter_xy_error_m": post_recenter_xy_error_m,
            }
        )
        if recenter_z_drift_m > args.max_recenter_z_command_drift_mm / 1000.0:
            reject["reason"] = "recenter_z_command_drift"
            return reject, None
        if entry_descent_m < args.min_entry_descent_mm / 1000.0:
            reject["reason"] = "weak_descent_entry"
            return reject, None
        if first_chunk_min_dz_m > -args.min_entry_descent_mm / 1000.0:
            reject["reason"] = "no_descent_in_first_replan_chunk"
            return reject, None
        if post_recenter_xy_error_m > args.max_post_recenter_xy_error_mm / 1000.0:
            reject["reason"] = "post_recenter_xy_error"
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
                "V2 episode length must match the nominal temporal-loss contract: "
                f"got {len(observations)}, expected {EXPECTED_EPISODE_FRAMES}"
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
            "phase_ranges": legacy._phase_ranges(args),
            "num_frames": len(observations),
            "success": True,
            "failure_mode": "success",
            "osc_input_type": "absolute",
            "action_representation": "controller",
            "dataset_kind": DATASET_KIND,
            "model_prefix": {
                "executed_steps": legacy.MODEL_APPROACH_STEPS,
                "actions_saved_to_training_file": False,
                "intermediate_frames_saved_to_training_file": False,
                "purpose": "visit real learned-policy offset state only",
            },
            "supervision_contract": {
                "stored_frames_0_59": "scripted_history_context_only",
                "stored_frame_60": "post_model_approach_offset_observation_context_only",
                "stored_frames_61_plus": "observations_after_oracle_commands",
                "action_mask_true_source": "oracle_only",
                "first_training_pair": "observation[60] -> oracle_action[61]",
            },
            "closed_loop_contract": {
                "deployment_replan_steps": args.replan_steps,
                "episode_frames": len(observations),
                "last_episode_frame": EXPECTED_EPISODE_FRAMES - 1,
                "pure_xy_recenter": True,
                "recenter_steps": args.recenter_steps,
                "descent_command_index_from_switch": args.recenter_steps,
                "recenter_z_command_drift_m": recenter_z_drift_m,
                "entry_descent_m": entry_descent_m,
                "first_replan_chunk_min_dz_m": first_chunk_min_dz_m,
                "post_recenter_xy_error_m": post_recenter_xy_error_m,
            },
            "switch": {
                "selected_cup": selected,
                "target_cup": meta["target_cup"],
                "selection_votes": votes,
                "offset_m": offset_m,
                "safe_height_m": safe_height_m,
                "gripper_width_m": switch["gripper_width"],
                "contacts": switch_contacts,
            },
            "oracle": {
                "recenter_steps": args.recenter_steps,
                "descend_steps": args.descend_steps,
                "grasp_steps": args.grasp_steps,
                "lift_steps": args.lift_steps,
                "recenter_z_mode": "hold_switch_z",
                "descent_progress": "linear_endpoint_free",
                "lift_height_m": args.lift_height,
                "final_success_stats": success_stats,
                "mean_grasp_lift_xy_error_m": float(
                    np.mean(
                        [
                            item["eef_to_selected_xy_m"]
                            for item in stage_metrics
                            if item["stage"] in {"grasp", "lift"}
                        ]
                    )
                ),
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
    if args.width != args.height:
        raise ValueError("Current fixed-history policy expects square images")
    if args.num_episodes <= 0 or args.max_attempts < args.num_episodes:
        raise ValueError("Require 0 < num_episodes <= max_attempts")
    if args.replan_steps != DEPLOYMENT_REPLAN_STEPS:
        raise ValueError(
            f"V2 was designed for deployment replan_steps={DEPLOYMENT_REPLAN_STEPS}; "
            f"got {args.replan_steps}"
        )
    if args.recenter_steps != DEFAULT_RECENTER_STEPS:
        raise ValueError(
            f"V2 requires recenter_steps={DEFAULT_RECENTER_STEPS} so command index 2 "
            "is already a descent command"
        )
    if min(args.descend_steps, args.grasp_steps, args.lift_steps) <= 0:
        raise ValueError("descend/grasp/lift stage lengths must all be positive")
    expected = (
        legacy.HISTORY_FRAMES
        + 1
        + args.recenter_steps
        + args.descend_steps
        + args.grasp_steps
        + args.lift_steps
    )
    if expected != EXPECTED_EPISODE_FRAMES:
        raise ValueError(
            "V2 stage lengths must produce exactly 155 frames for mixed training; "
            f"got {expected}. Keep recenter=2, descend=30, grasp=15, lift=47."
        )


def _summarize(output: Path, args: argparse.Namespace, accepted: int) -> dict:
    base_summary = legacy._summarize(output, args, accepted)
    manifest_path = output / "generation_manifest.jsonl"
    rows = []
    if manifest_path.exists():
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
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

    base_summary["dataset_kind"] = DATASET_KIND
    base_summary["closed_loop_audit"] = {
        "deployment_replan_steps": args.replan_steps,
        "episode_frames": EXPECTED_EPISODE_FRAMES,
        "last_episode_frame": EXPECTED_EPISODE_FRAMES - 1,
        "recenter_steps": args.recenter_steps,
        "pure_xy_recenter": True,
        "recenter_z_command_drift_m": stats("recenter_z_command_drift_m"),
        "entry_descent_m": stats("entry_descent_m"),
        "first_replan_chunk_min_dz_m": stats("first_replan_chunk_min_dz_m"),
        "post_recenter_xy_error_m": stats("post_recenter_xy_error_m"),
    }
    return base_summary


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "generation_manifest.jsonl"
    existing = sorted(output.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if existing and not args.resume:
        raise FileExistsError(f"{output} already contains {len(existing)} episodes")
    accepted = len(existing)
    attempted_indices: set[int] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                attempted_indices.add(int(json.loads(line)["attempt_index"]))

    legacy.base._policy_input = legacy.fixed_eef._fixed_history_policy_input
    from openpi_client import websocket_client_policy

    logging.basicConfig(level=logging.INFO, force=True)
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
            manifest.write(json.dumps(audit, sort_keys=True) + "\n")
            manifest.flush()
            elapsed = time.time() - start_time
            logging.info(
                "attempt=%d reason=%s accepted=%d/%d offset=%s entry_drop=%s elapsed=%.1fmin",
                attempt_index,
                audit["reason"],
                accepted,
                args.num_episodes,
                None
                if "switch_offset_m" not in audit
                else f"{audit['switch_offset_m'] * 1000.0:.1f}mm",
                None
                if "entry_descent_m" not in audit
                else f"{audit['entry_descent_m'] * 1000.0:.1f}mm",
                elapsed / 60.0,
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
