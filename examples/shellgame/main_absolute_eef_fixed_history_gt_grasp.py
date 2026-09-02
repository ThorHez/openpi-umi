"""Fixed-history absolute-EEF evaluation with a deterministic GT-centered grasp suffix.

The learned policy controls only the 30-step cup-selection / approach prefix.
The cup nearest the end effector during that prefix is treated as the model's
selection.  The suffix then uses the selected cup's simulator position to
recenter at a safe hover, descend vertically, hold the gripper closed, and
lift vertically.  Ground truth supplies geometry only after selection: it
does not replace the model's cup identity decision.

This is a diagnostic upper bound, not a deployable vision-only controller.
"""

# This diagnostic deliberately reuses private evaluator helpers so its camera,
# policy-input, success, and contact semantics remain identical to the baseline.
# ruff: noqa: FBT001, FBT003, SLF001

from __future__ import annotations

from collections import deque
import dataclasses
import json
import logging
import math
from pathlib import Path

import imageio.v2 as imageio
import main as base
import main_absolute_eef_fixed_history as fixed_eef
import numpy as np
import oracle_joint_noise_sensitivity as contact_utils
import tyro


@dataclasses.dataclass
class Args(base.Args):
    model_approach_steps: int = 30
    recenter_steps: int = 10
    deterministic_descend_steps: int = 30
    deterministic_grasp_steps: int = 15
    deterministic_lift_steps: int = 40
    deterministic_hover_height: float = 0.22
    deterministic_lift_height: float = 0.20
    summary_path: str = ""


def _cosine_progress(index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    linear = index / (count - 1)
    return 0.5 - 0.5 * math.cos(math.pi * linear)


def _choose_selected_cup(
    votes: dict[str, int],
    distance_sums: dict[str, float],
    current_eef_xy: np.ndarray,
    cups: dict[str, np.ndarray],
) -> str:
    best = max(votes.values(), default=0)
    if best > 0:
        candidates = [cup for cup, count in votes.items() if count == best]
        return min(candidates, key=lambda cup: distance_sums[cup] / votes[cup])
    return min(cups, key=lambda cup: float(np.linalg.norm(current_eef_xy - cups[cup][:2])))


def _save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def _deterministic_command(
    shell,
    env,
    *,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    gripper_action: float,
) -> np.ndarray:
    return shell.make_robot_action(
        env,
        target_pos=np.asarray(target_pos, dtype=np.float64),
        target_quat=np.asarray(target_quat, dtype=np.float64),
        gripper_action=float(gripper_action),
    )


def _run_trial(shell, client, args: Args, *, trial: int, rng: np.random.Generator) -> tuple[dict, list[np.ndarray]]:
    episode_seed = int(rng.integers(0, 2**31 - 1))
    initial = str(rng.choice(shell.CUP_NAMES)) if args.initial_ball_cup == "random" else args.initial_ball_cup
    num_swaps = int(rng.integers(args.min_swaps, args.max_swaps + 1))
    ep_args = base._episode_namespace(args, seed=episode_seed, initial_ball_cup=initial, num_swaps=num_swaps)
    env = shell.make_env(ep_args)
    history: list[dict] = []
    replay: list[np.ndarray] = []

    try:
        meta = base._run_scripted_observation(
            shell, env, ep_args, args, history, replay, client=client
        )
        if len(history) != fixed_eef.HISTORY_FRAMES:
            raise RuntimeError(
                f"Expected exactly {fixed_eef.HISTORY_FRAMES} scripted frames, got {len(history)}"
            )

        start_eef_pos = np.asarray(history[0]["eef_pos"], dtype=np.float32)
        # The scripted observation pose is the canonical grasp orientation used
        # by demonstration generation.  Lock it during the deterministic suffix.
        canonical_quat = np.asarray(history[0]["eef_quat"], dtype=np.float32)
        start_eef_quat = canonical_quat.copy()
        action_plan: deque = deque()
        gripper_action = args.default_gripper_action
        votes = dict.fromkeys(shell.CUP_NAMES, 0)
        distance_sums = dict.fromkeys(shell.CUP_NAMES, 0.0)
        model_trace: list[dict] = []

        # The learned tracker/action model owns cup identity and the complete
        # selection/approach prefix.  This matches source frames 60..89.
        for step in range(args.model_approach_steps):
            env_action, gripper_action, policy_action = base._policy_env_action(
                shell,
                env,
                history,
                start_eef_pos,
                start_eef_quat,
                action_plan,
                gripper_action,
                client=client,
                args=args,
                prompt=args.grasp_task if args.phase_instructions else None,
            )
            env.step(env_action)
            base._append_observation(
                shell,
                env,
                ep_args,
                meta["wrist_camera_name"],
                history,
                replay,
                resize_size=args.resize_size,
            )
            eef = np.asarray(history[-1]["eef_pos"], dtype=np.float64)
            cups = base._cup_positions(shell, env)
            distances = {
                cup: float(np.linalg.norm(eef[:2] - np.asarray(pos[:2], dtype=np.float64)))
                for cup, pos in cups.items()
            }
            nearest = min(distances, key=distances.get)
            if step >= args.cup_selection_skip_frames and distances[nearest] <= args.cup_selection_xy_radius:
                votes[nearest] += 1
                distance_sums[nearest] += distances[nearest]
            model_trace.append(
                {
                    "step": step,
                    "eef": eef.tolist(),
                    "nearest_cup": nearest,
                    "nearest_distance_m": distances[nearest],
                    "policy_action": None if policy_action is None else np.asarray(policy_action).tolist(),
                }
            )

        switch_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
        switch_cups = base._cup_positions(shell, env)
        selected_cup = _choose_selected_cup(votes, distance_sums, switch_eef[:2], switch_cups)
        selected_center = np.asarray(switch_cups[selected_cup], dtype=np.float64)
        selected_xy = selected_center[:2].copy()
        selection_correct = selected_cup == meta["target_cup"]
        switch_xy_error = float(np.linalg.norm(switch_eef[:2] - selected_xy))

        grasp_z = float(env.cup_handle_grasp_z() + ep_args.robot_grasp_z_offset)
        hover_pos = np.array(
            [selected_xy[0], selected_xy[1], grasp_z + args.deterministic_hover_height],
            dtype=np.float64,
        )
        grasp_pos = np.array([selected_xy[0], selected_xy[1], grasp_z], dtype=np.float64)
        lift_pos = np.array(
            [selected_xy[0], selected_xy[1], grasp_z + args.deterministic_lift_height],
            dtype=np.float64,
        )

        stage_trace: list[dict] = []
        ever_success = False
        first_success_stage: str | None = None
        first_success_step: int | None = None
        bilateral_run = 0
        max_bilateral_run = 0
        first_bilateral_stage: str | None = None
        grasp_xy_errors: list[float] = []

        def execute(stage: str, stage_step: int, target_pos: np.ndarray, close: bool) -> None:
            nonlocal ever_success, first_success_stage, first_success_step
            nonlocal bilateral_run, max_bilateral_run, first_bilateral_stage
            command = _deterministic_command(
                shell,
                env,
                target_pos=target_pos,
                target_quat=canonical_quat,
                gripper_action=1.0 if close else -1.0,
            )
            env.step(command)
            base._append_observation(
                shell,
                env,
                ep_args,
                meta["wrist_camera_name"],
                history,
                replay,
                resize_size=args.resize_size,
            )
            actual_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
            cups = base._cup_positions(shell, env)
            selected_error = float(np.linalg.norm(actual_eef[:2] - cups[selected_cup][:2]))
            target_error = float(np.linalg.norm(actual_eef[:2] - cups[meta["target_cup"]][:2]))
            if stage in {"grasp", "lift"}:
                grasp_xy_errors.append(selected_error)
            contacts = contact_utils._finger_contact_count(env, selected_cup)
            if contacts == 2:
                bilateral_run += 1
                max_bilateral_run = max(max_bilateral_run, bilateral_run)
                if first_bilateral_stage is None:
                    first_bilateral_stage = stage
            else:
                bilateral_run = 0
            success, success_stats = base._success(
                shell, env, meta["target_cup"], meta["settle_cup_pos"], args.lift_success_height
            )
            if success and not ever_success:
                ever_success = True
                first_success_stage = stage
                first_success_step = stage_step
            stage_trace.append(
                {
                    "stage": stage,
                    "stage_step": stage_step,
                    "command": np.asarray(command).tolist(),
                    "actual_eef": actual_eef.tolist(),
                    "selected_cup_xy_error_m": selected_error,
                    "target_cup_xy_error_m": target_error,
                    "selected_cup_finger_contacts": contacts,
                    "success": bool(success),
                    "success_stats": success_stats,
                }
            )

        # First return to a collision-free hover while correcting the model's
        # residual lateral error.  Subsequent descent is strictly vertical.
        recenter_start = switch_eef.copy()
        for step in range(args.recenter_steps):
            t = _cosine_progress(step, args.recenter_steps)
            execute("recenter", step, (1.0 - t) * recenter_start + t * hover_pos, False)
        for step in range(args.deterministic_descend_steps):
            t = _cosine_progress(step, args.deterministic_descend_steps)
            execute("descend", step, (1.0 - t) * hover_pos + t * grasp_pos, False)
        for step in range(args.deterministic_grasp_steps):
            execute("grasp", step, grasp_pos, True)
        for step in range(args.deterministic_lift_steps):
            t = _cosine_progress(step, args.deterministic_lift_steps)
            execute("lift", step, (1.0 - t) * grasp_pos + t * lift_pos, True)

        final_success, final_stats = base._success(
            shell, env, meta["target_cup"], meta["settle_cup_pos"], args.lift_success_height
        )
        return (
            {
                "trial": trial,
                "episode_seed": episode_seed,
                "initial_ball_cup": initial,
                "target_cup": meta["target_cup"],
                "final_ball_cup": meta["final_ball_cup"],
                "selected_cup": selected_cup,
                "selection_correct": selection_correct,
                "selection_votes": votes,
                "switch_eef_to_selected_xy_m": switch_xy_error,
                "success": bool(ever_success),
                "final_success": bool(final_success),
                "first_success_stage": first_success_stage,
                "first_success_step": first_success_step,
                "final_success_stats": final_stats,
                "first_bilateral_stage": first_bilateral_stage,
                "max_bilateral_run": max_bilateral_run,
                "mean_grasp_xy_error_m": None
                if not grasp_xy_errors
                else float(np.mean(grasp_xy_errors)),
                "model_trace": model_trace,
                "deterministic_trace": stage_trace,
            },
            replay,
        )
    finally:
        env.close()


def main() -> None:
    args = tyro.cli(Args)
    if args.policy_input_mode != "history":
        raise ValueError("This diagnostic requires --policy-input-mode history")
    if args.num_frames != fixed_eef.TOTAL_FRAMES or args.frame_stride != 1:
        raise ValueError("This diagnostic requires --num-frames 61 --frame-stride 1")
    if args.action_mode != "raw7" or args.action_dim != 7 or args.osc_input_type != "absolute":
        raise ValueError("This diagnostic requires raw7 absolute-OSC actions")
    if args.control_during_scripted_observation:
        raise ValueError("Use --no-control-during-scripted-observation")

    # Keep online preprocessing identical to training and the validated model-only evaluator.
    base._policy_input = fixed_eef._fixed_history_policy_input
    from openpi_client import websocket_client_policy

    logging.basicConfig(level=logging.INFO, force=True)
    shell = base._import_shellgame_tools(args.robosuite_root)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    rng = np.random.default_rng(args.seed)
    video_dir = Path(args.video_out_path)
    rows: list[dict] = []

    for trial in range(args.num_trials):
        row, frames = _run_trial(shell, client, args, trial=trial, rng=rng)
        rows.append(row)
        if args.save_videos:
            suffix = "success" if row["success"] else "failure"
            _save_video(video_dir / f"trial_{trial:04d}_{suffix}.mp4", frames, args.fps)
        logging.info(
            "trial=%d selected=%s target=%s selection_correct=%s success=%s "
            "switch_xy=%.1fmm grasp_xy=%s bilateral_run=%d",
            trial,
            row["selected_cup"],
            row["target_cup"],
            row["selection_correct"],
            row["success"],
            row["switch_eef_to_selected_xy_m"] * 1000.0,
            None
            if row["mean_grasp_xy_error_m"] is None
            else f"{row['mean_grasp_xy_error_m'] * 1000.0:.1f}mm",
            row["max_bilateral_run"],
        )

    successes = sum(row["success"] for row in rows)
    selection_correct = sum(row["selection_correct"] for row in rows)
    summary = {
        "experiment": "normal memory/model approach then selected-cup GT-centered deterministic grasp",
        "settings": dataclasses.asdict(args),
        "selection_correct": selection_correct,
        "selection_accuracy": selection_correct / max(len(rows), 1),
        "successes": successes,
        "success_rate": successes / max(len(rows), 1),
        "final_successes": sum(row["final_success"] for row in rows),
        "mean_switch_xy_error_m": float(
            np.mean([row["switch_eef_to_selected_xy_m"] for row in rows])
        ),
        "mean_grasp_xy_error_m": float(
            np.mean([row["mean_grasp_xy_error_m"] for row in rows])
        ),
        "bilateral_contact_episodes": sum(row["first_bilateral_stage"] is not None for row in rows),
        "trials": rows,
    }
    summary_path = Path(args.summary_path) if args.summary_path else video_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info(
        "final selection=%d/%d (%.1f%%) success=%d/%d (%.1f%%) summary=%s",
        selection_correct,
        len(rows),
        100.0 * summary["selection_accuracy"],
        successes,
        len(rows),
        100.0 * summary["success_rate"],
        summary_path,
    )


if __name__ == "__main__":
    main()
