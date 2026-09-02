"""Paired XY-noise sweep for the deterministic absolute-EEF grasp suffix.

The input is a completed ``main_absolute_eef_fixed_history_gt_grasp`` summary.
For every trial and noise level, this script recreates the exact ShellGame
state and replays the saved 30-step model approach commands.  Only the fixed
XY offset added to the selected cup's ground-truth center changes.
"""

# Private helpers intentionally keep simulator and metric semantics identical
# to the source rollout and prior Oracle diagnostics.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
from pathlib import Path

import imageio.v2 as imageio
import main as base
import main_absolute_eef_fixed_history_gt_grasp as hybrid
import numpy as np
import oracle_joint_noise_sensitivity as contact_utils
import oracle_joint_replay as oracle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=Path(
            "evaluation/shellgame/eef7_5999_model_approach_gt_grasp_20ep_260813/summary.json"
        ),
    )
    parser.add_argument("--noise-mm", default="0,2,4,6,8,10")
    parser.add_argument("--noise-seed", type=int, default=260813)
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--video-noise-mm", default="10")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_levels(text: str) -> list[float]:
    levels = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not levels or any(level < 0 for level in levels):
        raise ValueError("Noise levels must be a non-empty list of non-negative millimeters")
    if len(set(levels)) != len(levels):
        raise ValueError("Noise levels must be unique")
    return sorted(levels)


def _restore_args(settings: dict, robosuite_root: str) -> hybrid.Args:
    args = hybrid.Args()
    for field in dataclasses.fields(args):
        if field.name in settings:
            setattr(args, field.name, settings[field.name])
    args.robosuite_root = robosuite_root
    args.gpu_id = -1
    args.control_during_scripted_observation = False
    args.save_videos = False
    return args


def _noise_direction(trial: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, trial]))
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)


def _select_from_replay(votes: dict[str, int], sums: dict[str, float]) -> str | None:
    best = max(votes.values(), default=0)
    if best <= 0:
        return None
    candidates = [cup for cup, count in votes.items() if count == best]
    return min(candidates, key=lambda cup: sums[cup] / votes[cup])


def _save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def _run_condition(
    shell,
    policy_args: hybrid.Args,
    source: dict,
    *,
    noise_mm: float,
    noise_seed: int,
    save_video: bool,
) -> tuple[dict, list[np.ndarray]]:
    ep_args = base._episode_namespace(
        policy_args,
        seed=int(source["episode_seed"]),
        initial_ball_cup=str(source["initial_ball_cup"]),
        num_swaps=3,
    )
    env = shell.make_env(ep_args)
    history: list[dict] = []
    frames: list[np.ndarray] = []
    try:
        if save_video:
            scene = base._run_scripted_observation(
                shell, env, ep_args, policy_args, history, frames, client=None
            )
            wrist_camera = scene["wrist_camera_name"]
            canonical_quat = np.asarray(history[0]["eef_quat"], dtype=np.float64)
        else:
            oracle._disable_image_observables(env)
            scene = oracle._prepare_scripted_state(shell, env, ep_args)
            wrist_camera = None
            canonical_quat = np.asarray(shell.get_eef_quat(env), dtype=np.float64)

        if scene["target_cup"] != source["target_cup"]:
            raise RuntimeError(
                f"Trial {source['trial']} target mismatch: {scene['target_cup']} != {source['target_cup']}"
            )

        votes = dict.fromkeys(shell.CUP_NAMES, 0)
        distance_sums = dict.fromkeys(shell.CUP_NAMES, 0.0)
        commands = [np.asarray(item["policy_action"], dtype=np.float32) for item in source["model_trace"]]
        if len(commands) != policy_args.model_approach_steps or any(command.shape != (7,) for command in commands):
            raise RuntimeError(f"Trial {source['trial']} has invalid saved approach commands")
        low, high = (np.asarray(value, dtype=np.float32) for value in env.action_spec)
        for step, command in enumerate(commands):
            env.step(np.clip(command, low, high))
            eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
            cups = base._cup_positions(shell, env)
            distances = {
                cup: float(np.linalg.norm(eef[:2] - cups[cup][:2])) for cup in shell.CUP_NAMES
            }
            nearest = min(distances, key=distances.get)
            if (
                step >= policy_args.cup_selection_skip_frames
                and distances[nearest] <= policy_args.cup_selection_xy_radius
            ):
                votes[nearest] += 1
                distance_sums[nearest] += distances[nearest]
            if save_video:
                base._append_observation(
                    shell,
                    env,
                    ep_args,
                    wrist_camera,
                    history,
                    frames,
                    resize_size=policy_args.resize_size,
                )

        switch_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
        saved_switch_eef = np.asarray(source["model_trace"][-1]["eef"], dtype=np.float64)
        replay_error = float(np.linalg.norm(switch_eef - saved_switch_eef))
        selected_cup = _select_from_replay(votes, distance_sums)
        if selected_cup is None:
            cups = base._cup_positions(shell, env)
            selected_cup = min(
                shell.CUP_NAMES,
                key=lambda cup: float(np.linalg.norm(switch_eef[:2] - cups[cup][:2])),
            )
        if selected_cup != source["selected_cup"]:
            raise RuntimeError(
                f"Trial {source['trial']} selection replay mismatch: {selected_cup} != {source['selected_cup']}"
            )

        true_center = np.asarray(base._cup_positions(shell, env)[selected_cup], dtype=np.float64)
        direction = _noise_direction(int(source["trial"]), noise_seed)
        offset = direction * (noise_mm / 1000.0)
        commanded_xy = true_center[:2] + offset
        grasp_z = float(env.cup_handle_grasp_z() + ep_args.robot_grasp_z_offset)
        hover_pos = np.array(
            [commanded_xy[0], commanded_xy[1], grasp_z + policy_args.deterministic_hover_height]
        )
        grasp_pos = np.array([commanded_xy[0], commanded_xy[1], grasp_z])
        lift_pos = np.array(
            [commanded_xy[0], commanded_xy[1], grasp_z + policy_args.deterministic_lift_height]
        )

        bilateral_run = 0
        max_bilateral_run = 0
        first_bilateral_stage = None
        ever_success = False
        first_success_stage = None
        first_success_step = None
        grasp_xy_errors: list[float] = []
        final_stats = None

        def execute(stage: str, stage_step: int, target_pos: np.ndarray, gripper: float) -> None:
            nonlocal bilateral_run, max_bilateral_run, first_bilateral_stage
            nonlocal ever_success, first_success_stage, first_success_step, final_stats
            command = hybrid._deterministic_command(
                shell,
                env,
                target_pos=target_pos,
                target_quat=canonical_quat,
                gripper_action=gripper,
            )
            env.step(command)
            if save_video:
                base._append_observation(
                    shell,
                    env,
                    ep_args,
                    wrist_camera,
                    history,
                    frames,
                    resize_size=policy_args.resize_size,
                )
            actual_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
            current_center = np.asarray(base._cup_positions(shell, env)[selected_cup], dtype=np.float64)
            if stage in {"grasp", "lift"}:
                grasp_xy_errors.append(float(np.linalg.norm(actual_eef[:2] - current_center[:2])))
            contacts = contact_utils._finger_contact_count(env, selected_cup)
            if contacts == 2:
                bilateral_run += 1
                max_bilateral_run = max(max_bilateral_run, bilateral_run)
                if first_bilateral_stage is None:
                    first_bilateral_stage = stage
            else:
                bilateral_run = 0
            success, final_stats = base._success(
                shell,
                env,
                scene["target_cup"],
                scene["settle_cup_pos"],
                policy_args.lift_success_height,
            )
            if success and not ever_success:
                ever_success = True
                first_success_stage = stage
                first_success_step = stage_step

        recenter_start = switch_eef.copy()
        for step in range(policy_args.recenter_steps):
            t = hybrid._cosine_progress(step, policy_args.recenter_steps)
            execute("recenter", step, (1.0 - t) * recenter_start + t * hover_pos, -1.0)
        for step in range(policy_args.deterministic_descend_steps):
            t = hybrid._cosine_progress(step, policy_args.deterministic_descend_steps)
            execute("descend", step, (1.0 - t) * hover_pos + t * grasp_pos, -1.0)
        for step in range(policy_args.deterministic_grasp_steps):
            execute("grasp", step, grasp_pos, 1.0)
        for step in range(policy_args.deterministic_lift_steps):
            t = hybrid._cosine_progress(step, policy_args.deterministic_lift_steps)
            execute("lift", step, (1.0 - t) * grasp_pos + t * lift_pos, 1.0)

        return (
            {
                "trial": int(source["trial"]),
                "noise_mm": noise_mm,
                "noise_direction_xy": direction.tolist(),
                "commanded_offset_xy_m": offset.tolist(),
                "selected_cup": selected_cup,
                "target_cup": scene["target_cup"],
                "selection_correct": selected_cup == scene["target_cup"],
                "selection_votes": votes,
                "approach_replay_eef_error_m": replay_error,
                "success": bool(ever_success),
                "first_success_stage": first_success_stage,
                "first_success_step": first_success_step,
                "first_bilateral_stage": first_bilateral_stage,
                "max_bilateral_run": max_bilateral_run,
                "mean_grasp_xy_error_m": float(np.mean(grasp_xy_errors)),
                "final_success_stats": final_stats,
            },
            frames,
        )
    finally:
        env.close()


def _aggregate(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "selection_accuracy": float(np.mean([row["selection_correct"] for row in rows])),
        "successes": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "bilateral_contact_episodes": sum(row["first_bilateral_stage"] is not None for row in rows),
        "mean_max_bilateral_run": float(np.mean([row["max_bilateral_run"] for row in rows])),
        "mean_grasp_xy_error_m": float(np.mean([row["mean_grasp_xy_error_m"] for row in rows])),
        "max_approach_replay_eef_error_m": float(
            np.max([row["approach_replay_eef_error_m"] for row in rows])
        ),
    }


def main() -> None:
    args = parse_args()
    levels = _parse_levels(args.noise_mm)
    video_levels = set(_parse_levels(args.video_noise_mm)) if args.video_noise_mm else set()
    if not video_levels.issubset(set(levels)):
        raise ValueError("--video-noise-mm must be a subset of --noise-mm")

    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    sources = source_summary["trials"]
    if args.num_trials is not None:
        if args.num_trials <= 0:
            raise ValueError("--num-trials must be positive")
        sources = sources[: args.num_trials]
    policy_args = _restore_args(source_summary["settings"], args.robosuite_root)
    shell = base._import_shellgame_tools(args.robosuite_root)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, force=True)
    rows: list[dict] = []

    # Trial-major ordering ensures all magnitudes for an episode share exactly
    # the same injected direction and makes paired failures easy to inspect.
    for source in sources:
        for noise_mm in levels:
            row, frames = _run_condition(
                shell,
                policy_args,
                source,
                noise_mm=noise_mm,
                noise_seed=args.noise_seed,
                save_video=noise_mm in video_levels,
            )
            rows.append(row)
            if noise_mm in video_levels:
                suffix = "success" if row["success"] else "failure"
                _save_video(
                    output
                    / "videos"
                    / f"noise_{noise_mm:g}mm"
                    / f"trial_{row['trial']:04d}_{suffix}.mp4",
                    frames,
                    policy_args.fps,
                )
            logging.info(
                "trial=%02d noise=%4.1fmm success=%s grasp_xy=%.1fmm bilateral=%d replay=%.3fmm",
                row["trial"],
                noise_mm,
                row["success"],
                row["mean_grasp_xy_error_m"] * 1000.0,
                row["max_bilateral_run"],
                row["approach_replay_eef_error_m"] * 1000.0,
            )

    by_noise = {
        f"{level:g}": _aggregate([row for row in rows if row["noise_mm"] == level])
        for level in levels
    }
    payload = {
        "experiment": "paired GT cup-center XY noise sweep after identical saved model approach",
        "source_summary": str(args.source_summary.expanduser().resolve()),
        "noise_seed": args.noise_seed,
        "noise_levels_mm": levels,
        "noise_semantics": "fixed random unit direction per trial; exact radial XY magnitude",
        "by_noise_mm": by_noise,
        "paired_success_by_trial": {
            str(source["trial"]): {
                f"{level:g}": next(
                    row["success"]
                    for row in rows
                    if row["trial"] == source["trial"] and row["noise_mm"] == level
                )
                for level in levels
            }
            for source in sources
        },
        "rows": rows,
    }
    result_path = output / "results.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("by_noise_mm=%s", json.dumps(by_noise, sort_keys=True))
    logging.info("wrote %s", result_path)


if __name__ == "__main__":
    main()
