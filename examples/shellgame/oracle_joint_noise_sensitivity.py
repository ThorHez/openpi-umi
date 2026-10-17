"""Measure how absolute-joint bias propagates to EEF and grasp success.

The source trajectory remains the recorded Oracle joint trajectory.  For each
episode, one deterministic seven-joint Gaussian direction is sampled and held
constant over the complete robot trajectory.  Noise conditions scale that same
direction, so their results form a paired sensitivity curve rather than five
unrelated perturbations.

This diagnostic is intentionally independent of policy inference.  It measures
both open-loop FK displacement and closed-loop JOINT_POSITION replay under the
same controller used by the absolute-joint evaluator.
"""

# This diagnostic intentionally reuses evaluator and Robosuite internals so
# controller semantics and FK frames exactly match online absolute-joint tests.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import main as base
import main_v2_absolute_joint as joint_eval
import numpy as np
import oracle_joint_replay as oracle
import oracle_joint_replay_by_slot as balanced

JOINT_DIM = joint_eval.JOINT_DIM
ROBOT_PHASE_MIN = oracle.ROBOT_PHASE_MIN
GRASP_PHASES = (5, 6)
SLOTS = balanced.SLOTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../robosuite/outputs/shellgame_absolute_joint_dataset"),
    )
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--episodes-per-slot", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=260811)
    parser.add_argument("--joint-noise-seed", type=int, default=260812)
    parser.add_argument("--noise-deg", default="0,0.25,0.5,1,2")
    parser.add_argument(
        "--skip-single-joint-fk",
        action="store_true",
        help="Skip the per-joint +1 degree FK ablation (useful for parallel nonzero-noise shards).",
    )
    parser.add_argument("--joint-kp", type=float, default=50.0)
    parser.add_argument("--joint-damping-ratio", type=float, default=1.0)
    parser.add_argument("--cup-selection-skip-frames", type=int, default=10)
    parser.add_argument("--cup-selection-window-frames", type=int, default=30)
    parser.add_argument("--cup-selection-xy-radius", type=float, default=0.06)
    parser.add_argument("--lift-success-height", type=float, default=0.08)
    parser.add_argument("--gripper-deadband", type=float, default=0.004)
    parser.add_argument(
        "--gripper-mode",
        choices=("measured_width", "recorded_command"),
        default="measured_width",
    )
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _noise_levels(text: str) -> list[float]:
    levels = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not levels or any(level < 0 for level in levels):
        raise ValueError("--noise-deg must contain non-negative comma-separated values")
    if len(set(levels)) != len(levels):
        raise ValueError("--noise-deg contains duplicates")
    return sorted(levels)


def _episode_number(episode_dir: Path) -> int:
    return int(episode_dir.name.split("_")[-1])


def _bias_direction(episode_dir: Path, seed: int) -> np.ndarray:
    # Seed each episode independently so changing episode ordering cannot alter
    # its perturbation.  RMS-normalize so --noise-deg is exactly per-joint RMS.
    rng = np.random.default_rng(np.random.SeedSequence([seed, _episode_number(episode_dir)]))
    direction = rng.standard_normal(JOINT_DIM)
    return direction / np.sqrt(np.mean(np.square(direction)))


def _eval_args(cli_args: argparse.Namespace, command_args: dict) -> joint_eval.Args:
    args = joint_eval.Args()
    for field in dataclasses.fields(args):
        if field.name in command_args:
            setattr(args, field.name, command_args[field.name])
    args.robosuite_root = cli_args.robosuite_root
    args.width = cli_args.camera_size
    args.height = cli_args.camera_size
    args.resize_size = cli_args.camera_size
    args.gpu_id = -1
    args.control_during_scripted_observation = False
    args.observe_eef_frames = 0
    args.joint_kp = cli_args.joint_kp
    args.joint_damping_ratio = cli_args.joint_damping_ratio
    args.save_videos = False
    return args


def _finger_contact_count(env, target_cup: str) -> int:
    fingers: set[int] = set()
    model = env.sim.model
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        geom1 = model.geom_id2name(contact.geom1) or ""
        geom2 = model.geom_id2name(contact.geom2) or ""
        pair = geom1 + " " + geom2
        if f"{target_cup}_cup" not in pair:
            continue
        if "finger1" in pair:
            fingers.add(1)
        if "finger2" in pair:
            fingers.add(2)
    return len(fingers)


def _fk_positions(shell, env, joint_targets: np.ndarray) -> np.ndarray:
    q_indices = np.asarray(env.robots[0]._ref_joint_pos_indexes, dtype=np.int64)
    positions = []
    for target in joint_targets:
        env.sim.data.qpos[q_indices] = target
        env.sim.forward()
        positions.append(shell.get_eef_pos(env))
    return np.asarray(positions, dtype=np.float64)


def _displacement_metrics(clean: np.ndarray, perturbed: np.ndarray, phase_ids: np.ndarray) -> dict:
    delta = perturbed - clean
    xy = np.linalg.norm(delta[:, :2], axis=1)
    xyz = np.linalg.norm(delta, axis=1)
    grasp_mask = np.isin(phase_ids, GRASP_PHASES)

    def summarize(values: np.ndarray, prefix: str) -> dict:
        return {
            f"{prefix}_mean_m": float(np.mean(values)),
            f"{prefix}_rmse_m": float(np.sqrt(np.mean(np.square(values)))),
            f"{prefix}_max_m": float(np.max(values)),
        }

    return {
        **summarize(xy, "fk_xy"),
        **summarize(xyz, "fk_xyz"),
        **summarize(xy[grasp_mask], "grasp_fk_xy"),
        **summarize(xyz[grasp_mask], "grasp_fk_xyz"),
    }


def _single_joint_fk_sensitivity(
    shell,
    env,
    clean_joint: np.ndarray,
    phase_ids: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> list[dict]:
    state = env.sim.get_state()
    try:
        clean_eef = _fk_positions(shell, env, clean_joint)
        output = []
        for joint_index in range(JOINT_DIM):
            perturbed = clean_joint.copy()
            perturbed[:, joint_index] += np.deg2rad(1.0)
            perturbed = np.clip(perturbed, q_low, q_high)
            eef = _fk_positions(shell, env, perturbed)
            output.append(
                {
                    "joint_index": joint_index,
                    **_displacement_metrics(clean_eef, eef, phase_ids),
                }
            )
        return output
    finally:
        env.sim.set_state(state)
        env.sim.forward()


def replay_episode(
    shell,
    episode_dir: Path,
    cli_args: argparse.Namespace,
    noise_deg: float,
    direction: np.ndarray,
    *,
    compute_joint_sensitivity: bool,
) -> dict:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    command_args = metadata["command_args"]
    args = _eval_args(cli_args, command_args)
    ep_args = joint_eval._episode_namespace(
        args,
        seed=int(command_args["seed"]),
        initial_ball_cup=str(command_args["initial_ball_cup"]),
        num_swaps=int(command_args["num_swaps"]),
    )
    env = shell.make_env(ep_args)
    oracle._disable_image_observables(env)

    try:
        scene = oracle._prepare_scripted_state(shell, env, ep_args)
        with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
            joint_pos = np.asarray(source["joint_pos"], dtype=np.float32)
            eef_reference = np.asarray(source["eef_pos"], dtype=np.float32)
            gripper_state = np.asarray(source["gripper_state"], dtype=np.float32)
            controller_actions = np.asarray(source["controller_actions"], dtype=np.float32)
            action_mask = np.asarray(source["action_mask"], dtype=bool)
            phase_ids_all = np.asarray(source["phase_ids"], dtype=np.int16)
            target_cup_reference = np.asarray(source["target_cup_pos"], dtype=np.float32)

        robot_indices = np.flatnonzero(action_mask & (phase_ids_all >= ROBOT_PHASE_MIN))
        if robot_indices.size == 0:
            raise RuntimeError(f"{episode_dir}: no robot-control frames")
        first_robot_index = int(robot_indices[0])
        clean_joint = joint_pos[robot_indices].astype(np.float64)
        robot_phase_ids = phase_ids_all[robot_indices]

        action_low, action_high = (np.asarray(value, dtype=np.float64) for value in env.action_spec)
        q_low = action_low[:JOINT_DIM]
        q_high = action_high[:JOINT_DIM]
        bias_rad = np.deg2rad(noise_deg) * direction
        noisy_joint = np.clip(clean_joint + bias_rad[None, :], q_low, q_high)

        state = env.sim.get_state()
        try:
            clean_fk = _fk_positions(shell, env, clean_joint)
            noisy_fk = _fk_positions(shell, env, noisy_joint)
            fk_metrics = _displacement_metrics(clean_fk, noisy_fk, robot_phase_ids)
        finally:
            env.sim.set_state(state)
            env.sim.forward()

        joint_sensitivity = None
        if compute_joint_sensitivity:
            joint_sensitivity = _single_joint_fk_sensitivity(
                shell,
                env,
                clean_joint,
                robot_phase_ids,
                q_low,
                q_high,
            )

        expected_swaps = [[str(a), str(b)] for a, b in metadata.get("swaps", [])]
        if expected_swaps and scene["swaps"] != expected_swaps:
            raise RuntimeError(f"{episode_dir}: reconstructed swaps do not match metadata")
        expected_final = str(metadata.get("final_ball_cup", scene["final_ball_cup"]))
        if scene["final_ball_cup"] != expected_final:
            raise RuntimeError(f"{episode_dir}: reconstructed final slot does not match metadata")

        obs = env._get_observations(force_update=True)
        initial_joint = shell.obs_vector(obs, "robot0_joint_pos", size=JOINT_DIM)
        initial_joint_error = initial_joint - joint_pos[first_robot_index - 1]
        target_pos_now = np.asarray(scene["settle_cup_pos"][scene["target_cup"]], dtype=np.float32)
        initial_target_error = target_pos_now - target_cup_reference[first_robot_index - 1]

        votes = dict.fromkeys(shell.CUP_NAMES, 0)
        distance_sums = dict.fromkeys(shell.CUP_NAMES, 0.0)
        target_xy_distances = []
        replay_eef_errors = []
        tracking_errors = []
        total_joint_errors = []
        gripper_action = float(args.default_gripper_action)
        selection_end = cli_args.cup_selection_skip_frames + cli_args.cup_selection_window_frames
        bilateral_run = 0
        max_bilateral_run = 0
        first_bilateral_step = None
        first_grasp_xy_error = None
        initial_cup_z = float(scene["settle_cup_pos"][scene["target_cup"]][2])
        max_target_lift = 0.0

        for replay_step, (source_index, q_target) in enumerate(zip(robot_indices, noisy_joint, strict=True)):
            obs_before = env._get_observations(force_update=True)
            if cli_args.gripper_mode == "recorded_command":
                gripper_action = float(controller_actions[source_index, -1])
            else:
                current_width = base._gripper_width(shell.obs_vector(obs_before, "robot0_gripper_qpos"))
                target_width = base._gripper_width(gripper_state[source_index])
                if target_width < current_width - cli_args.gripper_deadband:
                    gripper_action = 1.0
                elif target_width > current_width + cli_args.gripper_deadband:
                    gripper_action = -1.0

            env_action = np.concatenate([q_target, np.asarray([gripper_action])])
            env.step(np.clip(env_action, action_low, action_high))
            obs_after = env._get_observations(force_update=True)
            actual_joint = shell.obs_vector(obs_after, "robot0_joint_pos", size=JOINT_DIM)
            actual_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
            target_pos = np.asarray(base._cup_positions(shell, env)[scene["target_cup"]], dtype=np.float64)
            target_xy = float(np.linalg.norm(actual_eef[:2] - target_pos[:2]))
            target_xy_distances.append(target_xy)
            tracking_errors.append(actual_joint - q_target)
            total_joint_errors.append(actual_joint - clean_joint[replay_step])
            replay_eef_errors.append(actual_eef - eef_reference[source_index])

            if robot_phase_ids[replay_step] == 6 and first_grasp_xy_error is None:
                first_grasp_xy_error = target_xy
            contact_count = _finger_contact_count(env, scene["target_cup"])
            if contact_count == 2:
                bilateral_run += 1
                max_bilateral_run = max(max_bilateral_run, bilateral_run)
                if first_bilateral_step is None:
                    first_bilateral_step = replay_step
            else:
                bilateral_run = 0
            max_target_lift = max(max_target_lift, float(target_pos[2] - initial_cup_z))

            if cli_args.cup_selection_skip_frames <= replay_step < selection_end:
                distances = {
                    cup: float(np.linalg.norm(actual_eef[:2] - scene["settle_cup_pos"][cup][:2]))
                    for cup in shell.CUP_NAMES
                }
                nearest = min(distances, key=distances.get)
                if distances[nearest] <= cli_args.cup_selection_xy_radius:
                    votes[nearest] += 1
                    distance_sums[nearest] += distances[nearest]

        selected_cup = oracle._select_cup(votes, distance_sums)
        success, success_stats = base._success(
            shell,
            env,
            scene["target_cup"],
            scene["settle_cup_pos"],
            cli_args.lift_success_height,
        )
        tracking = np.asarray(tracking_errors, dtype=np.float64)
        total_joint = np.asarray(total_joint_errors, dtype=np.float64)
        replay_eef = np.asarray(replay_eef_errors, dtype=np.float64)
        return {
            "episode": episode_dir.name,
            "target_cup": scene["target_cup"],
            "final_ball_cup": scene["final_ball_cup"],
            "noise_deg_rms": noise_deg,
            "bias_direction": direction.tolist(),
            "bias_rad": bias_rad.tolist(),
            "actual_bias_rms_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(noisy_joint - clean_joint))))),
            "clipped_joint_values": int(np.count_nonzero(np.abs(noisy_joint - (clean_joint + bias_rad)) > 1e-8)),
            "num_robot_steps": int(robot_indices.size),
            "initial_joint_max_abs_error_rad": float(np.max(np.abs(initial_joint_error))),
            "initial_target_pos_error_m": float(np.linalg.norm(initial_target_error)),
            **fk_metrics,
            "joint_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(tracking)))),
            "joint_total_error_to_clean_rmse_rad": float(np.sqrt(np.mean(np.square(total_joint)))),
            "eef_replay_xy_rmse_m": float(np.sqrt(np.mean(np.square(replay_eef[:, :2])))),
            "target_min_xy_distance_m": float(np.min(target_xy_distances)),
            "first_grasp_xy_error_m": first_grasp_xy_error,
            "first_bilateral_step": first_bilateral_step,
            "max_bilateral_run": max_bilateral_run,
            "max_target_lift_m": max_target_lift,
            "cup_selection": selected_cup,
            "cup_selection_correct": selected_cup == scene["target_cup"],
            "cup_selection_votes": votes,
            "success": bool(success),
            "success_stats": success_stats,
            "single_joint_fk_sensitivity_1deg": joint_sensitivity,
        }
    finally:
        env.close()


def _mean(results: list[dict], key: str) -> float:
    values = [item[key] for item in results if item[key] is not None]
    return float(np.mean(values))


def _aggregate(results: list[dict]) -> dict:
    return {
        "num_episodes": len(results),
        "successes": sum(item["success"] for item in results),
        "success_rate": float(np.mean([item["success"] for item in results])),
        "cup_selection_correct": sum(item["cup_selection_correct"] for item in results),
        "cup_selection_accuracy": float(np.mean([item["cup_selection_correct"] for item in results])),
        "bilateral_contact_episodes": sum(item["first_bilateral_step"] is not None for item in results),
        "mean_actual_bias_rms_deg": _mean(results, "actual_bias_rms_deg"),
        "mean_fk_xy_m": _mean(results, "fk_xy_mean_m"),
        "mean_fk_xy_rmse_m": _mean(results, "fk_xy_rmse_m"),
        "mean_grasp_fk_xy_m": _mean(results, "grasp_fk_xy_mean_m"),
        "mean_grasp_fk_xyz_m": _mean(results, "grasp_fk_xyz_mean_m"),
        "mean_first_grasp_xy_error_m": _mean(results, "first_grasp_xy_error_m"),
        "mean_target_min_xy_distance_m": _mean(results, "target_min_xy_distance_m"),
        "mean_max_bilateral_run": _mean(results, "max_bilateral_run"),
        "mean_max_target_lift_m": _mean(results, "max_target_lift_m"),
        "mean_joint_tracking_rmse_rad": _mean(results, "joint_tracking_rmse_rad"),
        "mean_joint_total_error_to_clean_rmse_rad": _mean(results, "joint_total_error_to_clean_rmse_rad"),
    }


def _aggregate_joint_sensitivity(results: list[dict]) -> list[dict]:
    baseline = [item for item in results if item["single_joint_fk_sensitivity_1deg"] is not None]
    if not baseline:
        return []
    output = []
    for joint_index in range(JOINT_DIM):
        rows = [item["single_joint_fk_sensitivity_1deg"][joint_index] for item in baseline]
        output.append(
            {
                "joint_index": joint_index,
                **{key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "joint_index"},
            }
        )
    return output


def main() -> None:
    args = parse_args()
    levels = _noise_levels(args.noise_deg)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    episode_dirs, selected_names = balanced._balanced_episode_dirs(args)
    shell = base._import_shellgame_tools(args.robosuite_root)
    directions = {episode_dir.name: _bias_direction(episode_dir, args.joint_noise_seed) for episode_dir in episode_dirs}

    results = []
    total = len(levels) * len(episode_dirs)
    count = 0
    for noise_deg in levels:
        for episode_dir in episode_dirs:
            count += 1
            result = replay_episode(
                shell,
                episode_dir,
                args,
                noise_deg,
                directions[episode_dir.name],
                compute_joint_sensitivity=not args.skip_single_joint_fk and noise_deg == 0.0,
            )
            results.append(result)
            logging.info(
                "[%d/%d] noise=%.2fdeg %s slot=%s success=%s fk_xy=%.1fmm grasp_xy=%.1fmm bi=%d",
                count,
                total,
                noise_deg,
                episode_dir.name,
                result["final_ball_cup"],
                result["success"],
                result["fk_xy_mean_m"] * 1000,
                result["first_grasp_xy_error_m"] * 1000,
                result["max_bilateral_run"],
            )

    by_noise = {}
    for noise_deg in levels:
        current = [item for item in results if item["noise_deg_rms"] == noise_deg]
        by_noise[str(noise_deg)] = {
            "overall": _aggregate(current),
            "by_final_slot": {
                slot: _aggregate([item for item in current if item["final_ball_cup"] == slot]) for slot in SLOTS
            },
        }
    output = {
        "experiment": "Oracle absolute-joint constant-bias sensitivity",
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "noise_mode": "episode-constant seven-joint Gaussian bias, RMS-normalized",
        "noise_levels_deg_rms": levels,
        "sample_seed": args.sample_seed,
        "joint_noise_seed": args.joint_noise_seed,
        "episodes_per_slot": args.episodes_per_slot,
        "selected_episodes": selected_names,
        "by_noise": by_noise,
        "single_joint_fk_sensitivity_1deg": _aggregate_joint_sensitivity(results),
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {"by_noise": by_noise, "single_joint_fk_sensitivity_1deg": output["single_joint_fk_sensitivity_1deg"]},
            indent=2,
        )
    )
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
