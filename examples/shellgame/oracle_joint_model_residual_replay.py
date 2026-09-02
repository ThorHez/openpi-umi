"""Replay the model's real grasp-chunk joint residual inside an Oracle trajectory.

At raw frame 109 the policy predicts raw frames 110..125 (ten grasp frames and
the first six lift frames).  This diagnostic rebuilds the exact held-out
episode, uses the recorded Oracle trajectory everywhere else, and compares:

* ``gt``: recorded joints and measured gripper width for every frame;
* ``model_arm``: model arm joints for frames 110..125, GT gripper throughout;
* ``model_full``: model arm joints and gripper widths for frames 110..125.

Thus ``model_arm`` is precisely ``GT + (prediction - GT)`` over one chunk and
isolates the physical effect of the model's actual structured joint residual.
"""

# This diagnostic intentionally reuses evaluator, policy, and Robosuite
# internals so preprocessing, action conversion, controller, and FK frames match
# the production absolute-joint evaluation exactly.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import collections
import gc
import json
import logging
from pathlib import Path

from eval_full_joint_grasp_checkpoint_fk import CURRENT_FRAME
from eval_full_joint_grasp_checkpoint_fk import FIRST_ACTION_FRAME
from eval_full_joint_grasp_checkpoint_fk import _load_record
import jax
import joint_fk_selection_eval as fk_eval
import main as base
import main_v2_absolute_joint as joint_eval
import numpy as np
import oracle_joint_noise_sensitivity as noise_eval
import oracle_joint_replay as oracle
from serve_old_tracker_full_joint_grasp import _build_config
import training_cup_eval

from openpi.policies import policy_config

JOINT_DIM = joint_eval.JOINT_DIM
ACTION_DIM = joint_eval.ACTION_DIM
CHUNK_LENGTH = 16
LAST_ACTION_FRAME = FIRST_ACTION_FRAME + CHUNK_LENGTH - 1
CONDITIONS = ("gt", "model_arm", "model_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "checkpoints/pi0_shellgame_old_tracker_full_joint_grasp_260810/"
            "full_joint_grasp_phase_balanced_b12_continue6000_260811/5999"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../robosuite/outputs/shellgame_absolute_joint_dataset"),
    )
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=260811)
    parser.add_argument("--joint-kp", type=float, default=50.0)
    parser.add_argument("--joint-damping-ratio", type=float, default=1.0)
    parser.add_argument("--gripper-deadband", type=float, default=0.004)
    parser.add_argument("--lift-success-height", type=float, default=0.08)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _eval_args(cli_args: argparse.Namespace, command_args: dict) -> joint_eval.Args:
    args = oracle._eval_args_from_metadata(cli_args, command_args)
    args.joint_kp = cli_args.joint_kp
    args.joint_damping_ratio = cli_args.joint_damping_ratio
    return args


def _load_source_arrays(episode_dir: Path) -> dict[str, np.ndarray]:
    with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
        return {
            "joint_pos": np.asarray(source["joint_pos"], dtype=np.float32),
            "eef_pos": np.asarray(source["eef_pos"], dtype=np.float32),
            "gripper_state": np.asarray(source["gripper_state"], dtype=np.float32),
            "controller_actions": np.asarray(source["controller_actions"], dtype=np.float32),
            "action_mask": np.asarray(source["action_mask"], dtype=bool),
            "phase_ids": np.asarray(source["phase_ids"], dtype=np.int16),
        }


def _validate_chunk_alignment(arrays: dict[str, np.ndarray], episode_dir: Path) -> None:
    expected = np.arange(FIRST_ACTION_FRAME, LAST_ACTION_FRAME + 1)
    active = arrays["action_mask"][expected]
    phases = arrays["phase_ids"][expected]
    if not np.all(active):
        raise RuntimeError(f"{episode_dir}: model chunk contains non-control frames")
    if not (np.all(phases[:10] == 6) and np.all(phases[10:] == 7)):
        raise RuntimeError(
            f"{episode_dir}: expected grasp x10 + lift x6 at frames "
            f"{FIRST_ACTION_FRAME}..{LAST_ACTION_FRAME}, got phases={phases.tolist()}"
        )


def _predicted_chunk_metrics(
    shell,
    env,
    predicted: np.ndarray,
    reference_joint: np.ndarray,
    reference_gripper: np.ndarray,
) -> dict:
    action_low, action_high = (np.asarray(value, dtype=np.float64) for value in env.action_spec)
    predicted_joint = np.clip(predicted[:, :JOINT_DIM], action_low[:JOINT_DIM], action_high[:JOINT_DIM])
    residual = predicted_joint - reference_joint
    state = env.sim.get_state()
    try:
        reference_fk = noise_eval._fk_positions(shell, env, reference_joint)
        predicted_fk = noise_eval._fk_positions(shell, env, predicted_joint)
    finally:
        env.sim.set_state(state)
        env.sim.forward()
    fk_delta = predicted_fk - reference_fk
    return {
        "joint_rmse_rad": float(np.sqrt(np.mean(np.square(residual)))),
        "joint_mae_rad": float(np.mean(np.abs(residual))),
        "joint_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(residual))))),
        "per_joint_rmse_rad": np.sqrt(np.mean(np.square(residual), axis=0)).tolist(),
        "per_joint_bias_rad": np.mean(residual, axis=0).tolist(),
        "residual_temporal_delta_rmse_rad": float(np.sqrt(np.mean(np.square(np.diff(residual, axis=0))))),
        "first_step_joint_error_rad": residual[0].tolist(),
        "last_step_joint_error_rad": residual[-1].tolist(),
        "fk_xy_rmse_m": float(np.sqrt(np.mean(np.square(fk_delta[:, :2])))),
        "fk_xy_norm_mean_m": float(np.mean(np.linalg.norm(fk_delta[:, :2], axis=1))),
        "fk_xyz_norm_mean_m": float(np.mean(np.linalg.norm(fk_delta, axis=1))),
        "end_grasp_fk_xy_error_m": float(np.linalg.norm(fk_delta[9, :2])),
        "gripper_width_rmse_m": float(np.sqrt(np.mean(np.square(predicted[:, 7] - reference_gripper)))),
        "predicted_chunk": predicted.tolist(),
    }


def _target_width(
    condition: str,
    source_index: int,
    arrays: dict[str, np.ndarray],
    predicted: np.ndarray,
) -> float:
    if condition == "model_full" and FIRST_ACTION_FRAME <= source_index <= LAST_ACTION_FRAME:
        return float(predicted[source_index - FIRST_ACTION_FRAME, 7])
    return float(base._gripper_width(arrays["gripper_state"][source_index]))


def _joint_target(
    condition: str,
    source_index: int,
    arrays: dict[str, np.ndarray],
    predicted: np.ndarray,
) -> np.ndarray:
    if condition != "gt" and FIRST_ACTION_FRAME <= source_index <= LAST_ACTION_FRAME:
        return np.asarray(predicted[source_index - FIRST_ACTION_FRAME, :JOINT_DIM], dtype=np.float64)
    return np.asarray(arrays["joint_pos"][source_index], dtype=np.float64)


def _replay_condition(
    shell,
    env,
    ep_args,
    cli_args: argparse.Namespace,
    episode_dir: Path,
    metadata: dict,
    arrays: dict[str, np.ndarray],
    predicted: np.ndarray,
    condition: str,
    default_gripper_action: float,
) -> dict:
    scene = oracle._prepare_scripted_state(shell, env, ep_args)
    expected_swaps = [[str(a), str(b)] for a, b in metadata.get("swaps", [])]
    if expected_swaps and scene["swaps"] != expected_swaps:
        raise RuntimeError(f"{episode_dir}: reconstructed swaps do not match metadata")
    expected_final = str(metadata.get("final_ball_cup", scene["final_ball_cup"]))
    if scene["final_ball_cup"] != expected_final:
        raise RuntimeError(f"{episode_dir}: reconstructed final slot does not match metadata")

    robot_indices = np.flatnonzero(arrays["action_mask"] & (arrays["phase_ids"] >= oracle.ROBOT_PHASE_MIN))
    action_low, action_high = (np.asarray(value, dtype=np.float64) for value in env.action_spec)
    gripper_action = float(default_gripper_action)
    target_xy_distances = []
    replay_eef_errors = []
    total_joint_errors = []
    bilateral_run = 0
    max_bilateral_run = 0
    first_bilateral_step = None
    first_grasp_xy_error = None
    end_grasp_xy_error = None
    end_injected_xy_error = None
    injected_xy_errors = []
    injected_bilateral_steps = 0
    initial_cup_z = float(scene["settle_cup_pos"][scene["target_cup"]][2])
    max_target_lift = 0.0

    for replay_step, source_index in enumerate(robot_indices):
        obs_before = env._get_observations(force_update=True)
        current_width = base._gripper_width(shell.obs_vector(obs_before, "robot0_gripper_qpos"))
        target_width = _target_width(condition, int(source_index), arrays, predicted)
        if target_width < current_width - cli_args.gripper_deadband:
            gripper_action = 1.0
        elif target_width > current_width + cli_args.gripper_deadband:
            gripper_action = -1.0

        q_target = _joint_target(condition, int(source_index), arrays, predicted)
        env_action = np.concatenate([q_target, np.asarray([gripper_action])])
        env.step(np.clip(env_action, action_low, action_high))
        obs_after = env._get_observations(force_update=True)
        actual_joint = shell.obs_vector(obs_after, "robot0_joint_pos", size=JOINT_DIM)
        actual_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
        cup_positions = base._cup_positions(shell, env)
        target_pos = np.asarray(cup_positions[scene["target_cup"]], dtype=np.float64)
        target_xy = float(np.linalg.norm(actual_eef[:2] - target_pos[:2]))
        target_xy_distances.append(target_xy)
        total_joint_errors.append(actual_joint - arrays["joint_pos"][source_index])
        replay_eef_errors.append(actual_eef - arrays["eef_pos"][source_index])

        if arrays["phase_ids"][source_index] == 6 and first_grasp_xy_error is None:
            first_grasp_xy_error = target_xy
        contact_count = noise_eval._finger_contact_count(env, scene["target_cup"])
        if FIRST_ACTION_FRAME <= source_index <= LAST_ACTION_FRAME:
            injected_xy_errors.append(target_xy)
            injected_bilateral_steps += int(contact_count == 2)
        if source_index == FIRST_ACTION_FRAME + 9:
            end_grasp_xy_error = target_xy
        if source_index == LAST_ACTION_FRAME:
            end_injected_xy_error = target_xy
        if contact_count == 2:
            bilateral_run += 1
            max_bilateral_run = max(max_bilateral_run, bilateral_run)
            if first_bilateral_step is None:
                first_bilateral_step = replay_step
        else:
            bilateral_run = 0
        max_target_lift = max(max_target_lift, float(target_pos[2] - initial_cup_z))

    success, success_stats = base._success(
        shell,
        env,
        scene["target_cup"],
        scene["settle_cup_pos"],
        cli_args.lift_success_height,
    )
    total_joint = np.asarray(total_joint_errors, dtype=np.float64)
    replay_eef = np.asarray(replay_eef_errors, dtype=np.float64)
    return {
        "condition": condition,
        "success": bool(success),
        "success_stats": success_stats,
        "target_min_xy_distance_m": float(np.min(target_xy_distances)),
        "first_grasp_xy_error_m": first_grasp_xy_error,
        "end_grasp_xy_error_m": end_grasp_xy_error,
        "end_injected_xy_error_m": end_injected_xy_error,
        "injected_xy_mean_m": float(np.mean(injected_xy_errors)),
        "injected_xy_max_m": float(np.max(injected_xy_errors)),
        "injected_bilateral_steps": injected_bilateral_steps,
        "first_bilateral_step": first_bilateral_step,
        "max_bilateral_run": max_bilateral_run,
        "max_target_lift_m": max_target_lift,
        "joint_total_error_to_gt_rmse_rad": float(np.sqrt(np.mean(np.square(total_joint)))),
        "eef_replay_xy_rmse_m": float(np.sqrt(np.mean(np.square(replay_eef[:, :2])))),
    }


def _aggregate(rows: list[dict]) -> dict:
    return {
        "num_episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "bilateral_contact_episodes": sum(row["first_bilateral_step"] is not None for row in rows),
        "mean_first_grasp_xy_error_m": float(np.mean([row["first_grasp_xy_error_m"] for row in rows])),
        "mean_target_min_xy_distance_m": float(np.mean([row["target_min_xy_distance_m"] for row in rows])),
        "mean_end_grasp_xy_error_m": float(np.mean([row["end_grasp_xy_error_m"] for row in rows])),
        "mean_end_injected_xy_error_m": float(np.mean([row["end_injected_xy_error_m"] for row in rows])),
        "mean_injected_xy_m": float(np.mean([row["injected_xy_mean_m"] for row in rows])),
        "mean_injected_bilateral_steps": float(np.mean([row["injected_bilateral_steps"] for row in rows])),
        "mean_max_bilateral_run": float(np.mean([row["max_bilateral_run"] for row in rows])),
        "mean_max_target_lift_m": float(np.mean([row["max_target_lift_m"] for row in rows])),
        "mean_joint_total_error_to_gt_rmse_rad": float(
            np.mean([row["joint_total_error_to_gt_rmse_rad"] for row in rows])
        ),
        "mean_eef_replay_xy_rmse_m": float(np.mean([row["eef_replay_xy_rmse_m"] for row in rows])),
    }


def _aggregate_prediction_metrics(rows: list[dict]) -> dict:
    keys = (
        "joint_rmse_rad",
        "joint_mae_rad",
        "joint_rmse_deg",
        "residual_temporal_delta_rmse_rad",
        "fk_xy_rmse_m",
        "fk_xy_norm_mean_m",
        "fk_xyz_norm_mean_m",
        "end_grasp_fk_xy_error_m",
        "gripper_width_rmse_m",
    )
    output = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    output["per_joint_rmse_rad"] = np.mean([row["per_joint_rmse_rad"] for row in rows], axis=0).tolist()
    output["per_joint_bias_rad"] = np.mean([row["per_joint_bias_rad"] for row in rows], axis=0).tolist()
    return output


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    episode_dirs = sorted(path for path in args.dataset_root.expanduser().resolve().glob("episode_*") if path.is_dir())
    selected_ids = training_cup_eval._balanced_validation_ids(
        episode_dirs,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        sample_seed=args.sample_seed,
        num_episodes=args.num_episodes,
    )
    records = [_load_record(episode_dirs[int(index)]) for index in selected_ids]
    logging.info("Loaded %d balanced held-out episodes", len(records))

    config = _build_config(sampling_steps=args.num_sampling_steps)
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=records[0]["observation"]["prompt"],
        sample_kwargs={"num_steps": args.num_sampling_steps},
    )
    predicted_chunks = fk_eval._batched_infer(
        policy,
        [record["observation"] for record in records],
        args.batch_size,
        args.sample_seed,
    )
    del policy
    gc.collect()
    jax.clear_caches()

    shell = base._import_shellgame_tools(args.robosuite_root)
    episodes = []
    prediction_metrics = []
    for number, (episode_id, record, predicted) in enumerate(
        zip(selected_ids, records, predicted_chunks, strict=True), 1
    ):
        episode_dir = episode_dirs[int(episode_id)]
        metadata = record["metadata"]
        arrays = _load_source_arrays(episode_dir)
        _validate_chunk_alignment(arrays, episode_dir)
        command_args = metadata["command_args"]
        eval_args = _eval_args(args, command_args)
        ep_args = joint_eval._episode_namespace(
            eval_args,
            seed=int(command_args["seed"]),
            initial_ball_cup=str(command_args["initial_ball_cup"]),
            num_swaps=int(command_args["num_swaps"]),
        )
        env = shell.make_env(ep_args)
        oracle._disable_image_observables(env)
        try:
            env.reset()
            current_metrics = _predicted_chunk_metrics(
                shell,
                env,
                predicted,
                record["reference_joint"],
                record["reference_gripper"],
            )
            prediction_metrics.append(current_metrics)
            condition_rows = [
                _replay_condition(
                    shell,
                    env,
                    ep_args,
                    args,
                    episode_dir,
                    metadata,
                    arrays,
                    predicted,
                    condition,
                    eval_args.default_gripper_action,
                )
                for condition in CONDITIONS
            ]
        finally:
            env.close()
        episodes.append(
            {
                "episode": episode_dir.name,
                "episode_id": int(episode_id),
                "target_cup": record["target_cup"],
                "target_slot": record["target_slot"],
                "prediction": current_metrics,
                "conditions": {row["condition"]: row for row in condition_rows},
            }
        )
        logging.info(
            "[%d/%d] %s slot=%s joint=%.3fdeg fk_xy=%.1fmm success=%s",
            number,
            len(records),
            episode_dir.name,
            record["target_slot"],
            current_metrics["joint_rmse_deg"],
            current_metrics["end_grasp_fk_xy_error_m"] * 1000,
            {row["condition"]: row["success"] for row in condition_rows},
        )

    by_condition = {}
    for condition in CONDITIONS:
        rows = [episode["conditions"][condition] for episode in episodes]
        by_condition[condition] = {
            "overall": _aggregate(rows),
            "by_target_slot": {
                slot: _aggregate(
                    [episode["conditions"][condition] for episode in episodes if episode["target_slot"] == slot]
                )
                for slot in ("left", "middle", "right")
            },
        }

    output = {
        "experiment": "Oracle trajectory with real model residual injection",
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "current_frame": CURRENT_FRAME,
        "injected_frames": [FIRST_ACTION_FRAME, LAST_ACTION_FRAME],
        "injected_phases": {"robot_grasp": 10, "robot_lift": 6},
        "conditions": list(CONDITIONS),
        "settings": {
            "num_episodes": args.num_episodes,
            "batch_size": args.batch_size,
            "num_sampling_steps": args.num_sampling_steps,
            "split_seed": args.split_seed,
            "val_ratio": args.val_ratio,
            "sample_seed": args.sample_seed,
            "selected_episode_ids": selected_ids.tolist(),
        },
        "target_slot_distribution": dict(collections.Counter(record["target_slot"] for record in records)),
        "prediction_metrics": _aggregate_prediction_metrics(prediction_metrics),
        "by_condition": by_condition,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "prediction_metrics": output["prediction_metrics"],
                "by_condition": by_condition,
            },
            indent=2,
        )
    )
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
