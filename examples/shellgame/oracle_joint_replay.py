"""Replay recorded ShellGame joint trajectories with JOINT_POSITION control.

This diagnostic isolates the action / controller mismatch in the absolute-joint
dataset.  Each source episode was generated with OSC_POSE, then its measured
joint positions were stored as actions.  Here we deterministically rebuild the
same scripted shell-game state and feed those measured joint positions to the
absolute JOINT_POSITION controller used by ``main_v2_absolute_joint.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

import main as base
import main_v2_absolute_joint as joint_eval


ROBOT_PHASE_MIN = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../robosuite/outputs/shellgame_absolute_joint_dataset"),
    )
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=0)
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
        help=(
            "measured_width reproduces the evaluator's width-to-open/close conversion; "
            "recorded_command reuses the original OSC demo's binary gripper command."
        ),
    )
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/shellgame/oracle_joint_replay/results.json"),
    )
    return parser.parse_args()


def _prepare_scripted_state(shell, env, ep_args):
    """Reproduce the renderer's reveal/cover/swap/settle state without images."""
    rng = np.random.default_rng(ep_args.seed)
    swaps = shell.sample_swaps(rng, ep_args.num_swaps)

    env.reset()
    shell.move_to_observation_pose(env, ep_args)
    cup_slots = {name: name for name in shell.CUP_NAMES}
    target_cup = ep_args.initial_ball_cup

    center_xy = {name: shell.slot_xy(env, slot) for name, slot in cup_slots.items()}
    lifted_target = {target_cup: ep_args.reveal_cup_lift_height}
    reveal_xy = dict(center_xy)
    reveal_ball_xy = shell.slot_xy(
        env,
        cup_slots[target_cup],
        offset=ep_args.reveal_ball_y_offset,
    )
    for _ in range(ep_args.reveal_frames):
        env.set_shellgame_positions(reveal_xy, target_cup, reveal_ball_xy, lifted_target, forward=True)
        base._zero_shellgame_object_velocities(env)

    cover_ball_return_fraction = 0.4
    for i in range(ep_args.cover_frames):
        t = 1.0 if ep_args.cover_frames <= 1 else i / (ep_args.cover_frames - 1)
        cup_xy = dict(center_xy)
        if t <= cover_ball_return_fraction:
            local_t = t / cover_ball_return_fraction
            ball_xy = (1.0 - local_t) * reveal_ball_xy + local_t * center_xy[target_cup]
            cup_z_offsets = lifted_target
        else:
            local_t = (t - cover_ball_return_fraction) / (1.0 - cover_ball_return_fraction)
            ball_xy = center_xy[target_cup]
            cup_z_offsets = {target_cup: ep_args.reveal_cup_lift_height * (1.0 - local_t)}
        env.set_shellgame_positions(cup_xy, target_cup, ball_xy, cup_z_offsets, forward=True)
        base._zero_shellgame_object_velocities(env)

    for slot_a, slot_b in swaps:
        cup_a = shell.cup_in_slot(cup_slots, slot_a)
        cup_b = shell.cup_in_slot(cup_slots, slot_b)
        start_xy = {name: shell.slot_xy(env, slot) for name, slot in cup_slots.items()}
        end_slots = dict(cup_slots)
        end_slots[cup_a], end_slots[cup_b] = slot_b, slot_a
        end_xy = {name: shell.slot_xy(env, slot) for name, slot in end_slots.items()}
        for i in range(ep_args.swap_frames):
            t = 1.0 if ep_args.swap_frames <= 1 else (i + 1) / ep_args.swap_frames
            cup_xy = dict(start_xy)
            cup_xy[cup_a] = shell.interpolate_xy(
                start_xy[cup_a],
                end_xy[cup_a],
                t,
                side_offset=ep_args.swap_arc_offset,
                layout_axis=ep_args.layout_axis,
            )
            cup_xy[cup_b] = shell.interpolate_xy(
                start_xy[cup_b],
                end_xy[cup_b],
                t,
                side_offset=-ep_args.swap_arc_offset,
                layout_axis=ep_args.layout_axis,
            )
            env.set_shellgame_positions(cup_xy, target_cup, cup_xy[target_cup], forward=True)
            base._zero_shellgame_object_velocities(env)
        cup_slots = end_slots

    settle_xy = {name: shell.slot_xy(env, slot) for name, slot in cup_slots.items()}
    for _ in range(ep_args.settle_frames):
        env.set_shellgame_positions(settle_xy, target_cup, settle_xy[target_cup], forward=True)
        base._zero_shellgame_object_velocities(env)

    settle_cup_pos = base._cup_positions(shell, env)
    return {
        "target_cup": target_cup,
        "final_ball_cup": cup_slots[target_cup],
        "settle_cup_pos": settle_cup_pos,
        "swaps": [[str(a), str(b)] for a, b in swaps],
    }


def _eval_args_from_metadata(cli_args, command_args: dict) -> joint_eval.Args:
    args = joint_eval.Args()
    for field in dataclasses.fields(args):
        if field.name in command_args:
            setattr(args, field.name, command_args[field.name])
    args.robosuite_root = cli_args.robosuite_root
    args.width = cli_args.camera_size
    args.height = cli_args.camera_size
    args.resize_size = cli_args.camera_size
    # Let EGL select the first process-visible GPU.  A physical GPU id copied
    # from generation metadata is invalid after CUDA_VISIBLE_DEVICES remapping.
    args.gpu_id = -1
    args.control_during_scripted_observation = False
    args.observe_eef_frames = 0
    args.joint_kp = cli_args.joint_kp
    args.joint_damping_ratio = cli_args.joint_damping_ratio
    args.save_videos = False
    return args


def _select_cup(votes: dict[str, int], distance_sums: dict[str, float]) -> str | None:
    max_votes = max(votes.values(), default=0)
    if max_votes <= 0:
        return None
    candidates = [cup for cup, count in votes.items() if count == max_votes]
    return min(candidates, key=lambda cup: distance_sums[cup] / votes[cup])


def _disable_image_observables(env) -> None:
    """Skip camera rendering; this replay only consumes proprioception."""
    for name, observable in env._observables.items():
        if observable.modality == "image":
            env.modify_observable(name, "active", False)
            env.modify_observable(name, "enabled", False)


def replay_episode(shell, episode_dir: Path, cli_args) -> dict:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    command_args = metadata["command_args"]
    args = _eval_args_from_metadata(cli_args, command_args)
    ep_args = joint_eval._episode_namespace(
        args,
        seed=int(command_args["seed"]),
        initial_ball_cup=str(command_args["initial_ball_cup"]),
        num_swaps=int(command_args["num_swaps"]),
    )
    env = shell.make_env(ep_args)
    _disable_image_observables(env)

    try:
        scene = _prepare_scripted_state(shell, env, ep_args)
        with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
            joint_pos = np.asarray(source["joint_pos"], dtype=np.float32)
            eef_pos_reference = np.asarray(source["eef_pos"], dtype=np.float32)
            gripper_state = np.asarray(source["gripper_state"], dtype=np.float32)
            controller_actions = np.asarray(source["controller_actions"], dtype=np.float32)
            action_mask = np.asarray(source["action_mask"], dtype=bool)
            phase_ids = np.asarray(source["phase_ids"], dtype=np.int16)
            target_cup_pos_reference = np.asarray(source["target_cup_pos"], dtype=np.float32)

        robot_indices = np.flatnonzero(action_mask & (phase_ids >= ROBOT_PHASE_MIN))
        if robot_indices.size == 0:
            raise RuntimeError(f"{episode_dir}: no robot-control frames")
        first_robot_index = int(robot_indices[0])
        previous_index = first_robot_index - 1

        obs = env._get_observations(force_update=True)
        initial_joint = shell.obs_vector(obs, "robot0_joint_pos", size=joint_eval.JOINT_DIM)
        initial_joint_error = initial_joint - joint_pos[previous_index]
        target_pos_now = np.asarray(scene["settle_cup_pos"][scene["target_cup"]], dtype=np.float32)
        initial_target_pos_error = target_pos_now - target_cup_pos_reference[previous_index]

        expected_swaps = [[str(a), str(b)] for a, b in metadata.get("swaps", [])]
        if expected_swaps and scene["swaps"] != expected_swaps:
            raise RuntimeError(
                f"{episode_dir}: reconstructed swaps {scene['swaps']} != metadata {expected_swaps}"
            )
        expected_final = str(metadata.get("final_ball_cup", scene["final_ball_cup"]))
        if scene["final_ball_cup"] != expected_final:
            raise RuntimeError(
                f"{episode_dir}: reconstructed final slot {scene['final_ball_cup']} != {expected_final}"
            )

        votes = {cup: 0 for cup in shell.CUP_NAMES}
        distance_sums = {cup: 0.0 for cup in shell.CUP_NAMES}
        target_selection_distances = []
        joint_errors = []
        eef_errors = []
        phase_joint_errors: dict[int, list[np.ndarray]] = {}
        gripper_action = float(args.default_gripper_action)
        selection_end = cli_args.cup_selection_skip_frames + cli_args.cup_selection_window_frames

        action_low, action_high = (np.asarray(x, dtype=np.float32) for x in env.action_spec)
        for replay_step, source_index in enumerate(robot_indices):
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

            env_action = np.concatenate(
                [joint_pos[source_index], np.asarray([gripper_action], dtype=np.float32)]
            )
            env.step(np.clip(env_action, action_low, action_high))
            obs_after = env._get_observations(force_update=True)
            actual_joint = shell.obs_vector(obs_after, "robot0_joint_pos", size=joint_eval.JOINT_DIM)
            actual_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float32)
            joint_error = actual_joint - joint_pos[source_index]
            joint_errors.append(joint_error)
            eef_errors.append(actual_eef - eef_pos_reference[source_index])
            phase_joint_errors.setdefault(int(phase_ids[source_index]), []).append(joint_error)

            if cli_args.cup_selection_skip_frames <= replay_step < selection_end:
                distances = {
                    cup: float(np.linalg.norm(actual_eef[:2] - scene["settle_cup_pos"][cup][:2]))
                    for cup in shell.CUP_NAMES
                }
                target_selection_distances.append(distances[scene["target_cup"]])
                nearest = min(distances, key=distances.get)
                if distances[nearest] <= cli_args.cup_selection_xy_radius:
                    votes[nearest] += 1
                    distance_sums[nearest] += distances[nearest]

        selected_cup = _select_cup(votes, distance_sums)
        success, success_stats = base._success(
            shell,
            env,
            scene["target_cup"],
            scene["settle_cup_pos"],
            cli_args.lift_success_height,
        )
        joint_errors_arr = np.asarray(joint_errors, dtype=np.float64)
        eef_errors_arr = np.asarray(eef_errors, dtype=np.float64)
        phase_rmse = {
            str(phase): float(np.sqrt(np.mean(np.square(np.asarray(errors)))))
            for phase, errors in phase_joint_errors.items()
        }
        return {
            "episode": episode_dir.name,
            "seed": int(command_args["seed"]),
            "target_cup": scene["target_cup"],
            "final_ball_cup": scene["final_ball_cup"],
            "first_robot_index": first_robot_index,
            "num_robot_steps": int(robot_indices.size),
            "initial_joint_max_abs_error_rad": float(np.max(np.abs(initial_joint_error))),
            "initial_target_pos_error_m": float(np.linalg.norm(initial_target_pos_error)),
            "joint_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(joint_errors_arr)))),
            "joint_tracking_mae_rad": float(np.mean(np.abs(joint_errors_arr))),
            "joint_tracking_max_abs_rad": float(np.max(np.abs(joint_errors_arr))),
            "phase_joint_tracking_rmse_rad": phase_rmse,
            "eef_replay_rmse_m": float(np.sqrt(np.mean(np.square(eef_errors_arr)))),
            "eef_replay_xy_rmse_m": float(np.sqrt(np.mean(np.square(eef_errors_arr[:, :2])))),
            "target_selection_min_xy_distance_m": (
                float(min(target_selection_distances)) if target_selection_distances else None
            ),
            "cup_selection": selected_cup,
            "cup_selection_correct": selected_cup == scene["target_cup"],
            "cup_selection_votes": votes,
            "success": bool(success),
            "success_stats": success_stats,
        }
    finally:
        env.close()


def _aggregate(results: list[dict], cli_args) -> dict:
    def mean(key):
        return float(np.mean([item[key] for item in results]))

    decisions = sum(item["cup_selection"] is not None for item in results)
    correct = sum(item["cup_selection_correct"] for item in results)
    successes = sum(item["success"] for item in results)
    return {
        "num_episodes": len(results),
        "gripper_mode": cli_args.gripper_mode,
        "joint_kp": cli_args.joint_kp,
        "joint_damping_ratio": cli_args.joint_damping_ratio,
        "cup_selection_decisions": decisions,
        "cup_selection_correct": correct,
        "cup_selection_accuracy_all": correct / len(results),
        "cup_selection_accuracy_decided": correct / decisions if decisions else 0.0,
        "successes": successes,
        "success_rate": successes / len(results),
        "mean_initial_joint_max_abs_error_rad": mean("initial_joint_max_abs_error_rad"),
        "mean_initial_target_pos_error_m": mean("initial_target_pos_error_m"),
        "mean_joint_tracking_rmse_rad": mean("joint_tracking_rmse_rad"),
        "mean_joint_tracking_mae_rad": mean("joint_tracking_mae_rad"),
        "mean_eef_replay_rmse_m": mean("eef_replay_rmse_m"),
        "mean_eef_replay_xy_rmse_m": mean("eef_replay_xy_rmse_m"),
    }


def main() -> None:
    cli_args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dataset_root = cli_args.dataset_root.expanduser().resolve()
    episode_dirs = sorted(path for path in dataset_root.glob("episode_*" ) if path.is_dir())
    if cli_args.num_episodes <= 0 or cli_args.num_episodes > len(episode_dirs):
        raise ValueError(f"--num-episodes must be in [1, {len(episode_dirs)}]")
    rng = np.random.default_rng(cli_args.sample_seed)
    selected_indices = np.sort(rng.choice(len(episode_dirs), size=cli_args.num_episodes, replace=False))
    selected_dirs = [episode_dirs[int(index)] for index in selected_indices]

    shell = base._import_shellgame_tools(cli_args.robosuite_root)
    results = []
    for number, episode_dir in enumerate(selected_dirs, 1):
        result = replay_episode(shell, episode_dir, cli_args)
        results.append(result)
        logging.info(
            "[%d/%d] %s target=%s selected=%s correct=%s success=%s "
            "joint_rmse=%.4f eef_xy_rmse=%.4f",
            number,
            len(selected_dirs),
            episode_dir.name,
            result["target_cup"],
            result["cup_selection"],
            result["cup_selection_correct"],
            result["success"],
            result["joint_tracking_rmse_rad"],
            result["eef_replay_xy_rmse_m"],
        )

    output = {
        "dataset_root": str(dataset_root),
        "sample_seed": cli_args.sample_seed,
        "summary": _aggregate(results, cli_args),
        "episodes": results,
    }
    cli_args.output.parent.mkdir(parents=True, exist_ok=True)
    cli_args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(f"Wrote {cli_args.output.resolve()}")


if __name__ == "__main__":
    main()
