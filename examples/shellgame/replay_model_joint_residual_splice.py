"""Splice a model's real 16-step joint residual into an Oracle replay.

The policy observes raw frame 109 with the fixed tracker prefix 0..59 and
predicts raw frames 110..125 (ten grasp frames plus six early-lift frames).
Only the seven arm joints from that prediction replace the Oracle trajectory;
the gripper and every other source frame remain ground truth.  A full-GT replay
of the same held-out episodes is the paired control.
"""

# This diagnostic intentionally reuses private policy, evaluator, and
# Robosuite interfaces so preprocessing and controller semantics stay exact.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import collections
import gc
import json
import logging
from pathlib import Path

import eval_full_joint_grasp_checkpoint_fk as model_fk
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

DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi0_shellgame_old_tracker_full_joint_grasp_260810/"
    "full_joint_grasp_phase_balanced_b12_continue6000_260811/5999"
)
FIRST_ACTION_FRAME = model_fk.FIRST_ACTION_FRAME
ACTION_HORIZON = 16
LAST_ACTION_FRAME = FIRST_ACTION_FRAME + ACTION_HORIZON - 1
GRASP_END_FRAME = FIRST_ACTION_FRAME + model_fk.GRASP_STEPS - 1
CONDITIONS = ("gt", "model_joint_residual_16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
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
    parser.add_argument("--lift-success-height", type=float, default=0.08)
    parser.add_argument("--gripper-deadband", type=float, default=0.004)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--selection-radius", type=float, default=0.06)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_record(episode_dir: Path) -> dict:
    record = model_fk._load_record(episode_dir)
    with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
        record.update(
            episode_dir=episode_dir,
            joint_pos=np.asarray(source["joint_pos"], dtype=np.float32),
            eef_pos=np.asarray(source["eef_pos"], dtype=np.float32),
            gripper_state=np.asarray(source["gripper_state"], dtype=np.float32),
            controller_actions=np.asarray(source["controller_actions"], dtype=np.float32),
            action_mask=np.asarray(source["action_mask"], dtype=bool),
            phase_ids=np.asarray(source["phase_ids"], dtype=np.int16),
            target_cup_pos=np.asarray(source["target_cup_pos"], dtype=np.float32),
        )
    return record


def _nearest_cup(eef_xy: np.ndarray, cup_positions: dict[str, np.ndarray]) -> tuple[str, float]:
    distances = {
        cup: float(np.linalg.norm(eef_xy - np.asarray(position)[:2])) for cup, position in cup_positions.items()
    }
    cup = min(distances, key=distances.get)
    return cup, distances[cup]


def _select_cup(votes: dict[str, int], distance_sums: dict[str, float]) -> str | None:
    max_votes = max(votes.values(), default=0)
    if max_votes == 0:
        return None
    candidates = [cup for cup, count in votes.items() if count == max_votes]
    return min(candidates, key=lambda cup: distance_sums[cup] / votes[cup])


def _replay(record: dict, predicted: np.ndarray, condition: str, shell, args) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    command_args = record["metadata"]["command_args"]
    eval_args = noise_eval._eval_args(args, command_args)
    ep_args = joint_eval._episode_namespace(
        eval_args,
        seed=int(command_args["seed"]),
        initial_ball_cup=str(command_args["initial_ball_cup"]),
        num_swaps=int(command_args["num_swaps"]),
    )
    env = shell.make_env(ep_args)
    oracle._disable_image_observables(env)
    try:
        scene = oracle._prepare_scripted_state(shell, env, ep_args)
        robot_indices = np.flatnonzero(record["action_mask"] & (record["phase_ids"] >= oracle.ROBOT_PHASE_MIN))
        clean_joint = np.asarray(record["joint_pos"][robot_indices], dtype=np.float64)
        replay_joint = clean_joint.copy()
        splice_rows = np.flatnonzero((robot_indices >= FIRST_ACTION_FRAME) & (robot_indices <= LAST_ACTION_FRAME))
        if splice_rows.size != ACTION_HORIZON:
            raise RuntimeError(f"{record['episode']}: expected {ACTION_HORIZON} splice rows, got {splice_rows.size}")
        if predicted.shape[0] != ACTION_HORIZON or predicted.shape[1] < joint_eval.JOINT_DIM:
            raise ValueError(f"Unexpected predicted action shape {predicted.shape}")
        if condition == "model_joint_residual_16":
            replay_joint[splice_rows] = predicted[:, : joint_eval.JOINT_DIM]

        action_low, action_high = (np.asarray(value, dtype=np.float64) for value in env.action_spec)
        q_low = action_low[: joint_eval.JOINT_DIM]
        q_high = action_high[: joint_eval.JOINT_DIM]
        unclipped_joint = replay_joint.copy()
        replay_joint = np.clip(replay_joint, q_low, q_high)

        state = env.sim.get_state()
        try:
            clean_fk = noise_eval._fk_positions(shell, env, clean_joint[splice_rows])
            replay_fk = noise_eval._fk_positions(shell, env, replay_joint[splice_rows])
        finally:
            env.sim.set_state(state)
            env.sim.forward()
        fk_delta = replay_fk - clean_fk
        fk_xy = np.linalg.norm(fk_delta[:, :2], axis=1)
        fk_xyz = np.linalg.norm(fk_delta, axis=1)
        residual = replay_joint[splice_rows] - clean_joint[splice_rows]
        cup_positions = scene["settle_cup_pos"]
        predicted_end_cup, predicted_end_distance = _nearest_cup(replay_fk[model_fk.GRASP_STEPS - 1, :2], cup_positions)
        target_xy = np.asarray(cup_positions[scene["target_cup"]])[:2]
        predicted_end_target_xy = float(np.linalg.norm(replay_fk[model_fk.GRASP_STEPS - 1, :2] - target_xy))

        next_rows = np.flatnonzero(robot_indices == LAST_ACTION_FRAME + 1)
        boundary = None
        if next_rows.size == 1:
            next_row = int(next_rows[0])
            boundary = {
                "joint_jump_rad": float(np.linalg.norm(clean_joint[next_row] - replay_joint[splice_rows[-1]])),
                "gt_joint_jump_rad": float(np.linalg.norm(clean_joint[next_row] - clean_joint[splice_rows[-1]])),
            }

        obs = env._get_observations(force_update=True)
        initial_joint = shell.obs_vector(obs, "robot0_joint_pos", size=joint_eval.JOINT_DIM)
        initial_joint_error = initial_joint - record["joint_pos"][int(robot_indices[0]) - 1]
        target_now = np.asarray(cup_positions[scene["target_cup"]], dtype=np.float64)
        initial_target_error = target_now - record["target_cup_pos"][int(robot_indices[0]) - 1]

        votes = dict.fromkeys(shell.CUP_NAMES, 0)
        distance_sums = dict.fromkeys(shell.CUP_NAMES, 0.0)
        gripper_action = float(eval_args.default_gripper_action)
        tracking_errors = []
        eef_reference_errors = []
        target_xy_errors = []
        grasp_xy_errors = []
        bilateral_run = 0
        max_bilateral_run = 0
        first_bilateral_source_frame = None
        initial_cup_z = float(target_now[2])
        max_target_lift = 0.0

        for _replay_step, (source_index, q_target) in enumerate(zip(robot_indices, replay_joint, strict=True)):
            obs_before = env._get_observations(force_update=True)
            current_width = base._gripper_width(shell.obs_vector(obs_before, "robot0_gripper_qpos"))
            target_width = base._gripper_width(record["gripper_state"][source_index])
            if target_width < current_width - args.gripper_deadband:
                gripper_action = 1.0
            elif target_width > current_width + args.gripper_deadband:
                gripper_action = -1.0

            env_action = np.concatenate([q_target, np.asarray([gripper_action])])
            env.step(np.clip(env_action, action_low, action_high))
            obs_after = env._get_observations(force_update=True)
            actual_joint = shell.obs_vector(obs_after, "robot0_joint_pos", size=joint_eval.JOINT_DIM)
            actual_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
            current_cups = base._cup_positions(shell, env)
            current_target = np.asarray(current_cups[scene["target_cup"]], dtype=np.float64)
            target_error = float(np.linalg.norm(actual_eef[:2] - current_target[:2]))
            target_xy_errors.append(target_error)
            tracking_errors.append(actual_joint - q_target)
            eef_reference_errors.append(actual_eef - record["eef_pos"][source_index])

            if FIRST_ACTION_FRAME <= source_index <= GRASP_END_FRAME:
                grasp_xy_errors.append(target_error)
                nearest, distance = _nearest_cup(actual_eef[:2], current_cups)
                if distance <= args.selection_radius:
                    votes[nearest] += 1
                    distance_sums[nearest] += distance

            contact_count = noise_eval._finger_contact_count(env, scene["target_cup"])
            if contact_count == 2:
                bilateral_run += 1
                max_bilateral_run = max(max_bilateral_run, bilateral_run)
                if first_bilateral_source_frame is None:
                    first_bilateral_source_frame = int(source_index)
            else:
                bilateral_run = 0
            max_target_lift = max(max_target_lift, float(current_target[2] - initial_cup_z))

        selected_cup = _select_cup(votes, distance_sums)
        success, success_stats = base._success(
            shell,
            env,
            scene["target_cup"],
            cup_positions,
            args.lift_success_height,
        )
        tracking = np.asarray(tracking_errors)
        eef_reference_error = np.asarray(eef_reference_errors)
        return {
            "episode": record["episode"],
            "condition": condition,
            "target_cup": scene["target_cup"],
            "target_slot": record["target_slot"],
            "predicted_end_cup": predicted_end_cup,
            "predicted_end_cup_correct": predicted_end_cup == scene["target_cup"],
            "predicted_end_nearest_distance_m": predicted_end_distance,
            "predicted_end_target_xy_error_m": predicted_end_target_xy,
            "actual_grasp_selected_cup": selected_cup,
            "actual_grasp_selection_correct": selected_cup == scene["target_cup"],
            "actual_grasp_votes": votes,
            "success": bool(success),
            "success_stats": success_stats,
            "joint_residual_rmse_rad": float(np.sqrt(np.mean(np.square(residual)))),
            "joint_residual_mae_rad": float(np.mean(np.abs(residual))),
            "fk_xy_mean_m": float(np.mean(fk_xy)),
            "fk_xy_max_m": float(np.max(fk_xy)),
            "fk_xyz_mean_m": float(np.mean(fk_xyz)),
            "grasp_fk_xy_mean_m": float(np.mean(fk_xy[: model_fk.GRASP_STEPS])),
            "grasp_fk_xyz_mean_m": float(np.mean(fk_xyz[: model_fk.GRASP_STEPS])),
            "first_grasp_xy_error_m": float(grasp_xy_errors[0]),
            "mean_grasp_xy_error_m": float(np.mean(grasp_xy_errors)),
            "end_grasp_xy_error_m": float(grasp_xy_errors[-1]),
            "target_min_xy_distance_m": float(np.min(target_xy_errors)),
            "first_bilateral_source_frame": first_bilateral_source_frame,
            "max_bilateral_run": max_bilateral_run,
            "max_target_lift_m": max_target_lift,
            "joint_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(tracking)))),
            "eef_reference_xy_rmse_m": float(np.sqrt(np.mean(np.square(eef_reference_error[:, :2])))),
            "initial_joint_max_abs_error_rad": float(np.max(np.abs(initial_joint_error))),
            "initial_target_pos_error_m": float(np.linalg.norm(initial_target_error)),
            "clipped_joint_values": int(np.count_nonzero(np.abs(replay_joint - unclipped_joint) > 1e-8)),
            "post_splice_boundary": boundary,
        }
    finally:
        env.close()


def _aggregate(rows: list[dict]) -> dict:
    return {
        "num_episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "predicted_end_cup_correct": sum(row["predicted_end_cup_correct"] for row in rows),
        "actual_grasp_selection_correct": sum(row["actual_grasp_selection_correct"] for row in rows),
        "bilateral_contact_episodes": sum(row["first_bilateral_source_frame"] is not None for row in rows),
        **{
            f"mean_{key}": float(np.mean([row[key] for row in rows]))
            for key in (
                "joint_residual_rmse_rad",
                "fk_xy_mean_m",
                "grasp_fk_xy_mean_m",
                "grasp_fk_xyz_mean_m",
                "predicted_end_target_xy_error_m",
                "first_grasp_xy_error_m",
                "mean_grasp_xy_error_m",
                "end_grasp_xy_error_m",
                "max_bilateral_run",
                "max_target_lift_m",
                "joint_tracking_rmse_rad",
            )
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
        default_prompt=model_fk.PROMPT,
        sample_kwargs={"num_steps": args.num_sampling_steps},
    )
    predictions = fk_eval._batched_infer(
        policy,
        [record["observation"] for record in records],
        args.batch_size,
        args.sample_seed,
    )
    for record in records:
        del record["observation"]
    del policy
    jax.clear_caches()
    gc.collect()

    shell = base._import_shellgame_tools(args.robosuite_root)
    results = []
    total = len(records) * len(CONDITIONS)
    count = 0
    for record, predicted in zip(records, predictions, strict=True):
        for condition in CONDITIONS:
            count += 1
            row = _replay(record, predicted, condition, shell, args)
            results.append(row)
            logging.info(
                "[%d/%d] %s %s success=%s residual=%.3fdeg fk_xy=%.1fmm grasp_xy=%.1fmm bi=%d",
                count,
                total,
                record["episode"],
                condition,
                row["success"],
                np.rad2deg(row["joint_residual_rmse_rad"]),
                row["grasp_fk_xy_mean_m"] * 1000,
                row["mean_grasp_xy_error_m"] * 1000,
                row["max_bilateral_run"],
            )

    by_condition = {}
    for condition in CONDITIONS:
        current = [row for row in results if row["condition"] == condition]
        by_condition[condition] = {
            "overall": _aggregate(current),
            "by_target_slot": {
                slot: _aggregate([row for row in current if row["target_slot"] == slot])
                for slot in ("left", "middle", "right")
            },
        }
    paired = {}
    for record, predicted in zip(records, predictions, strict=True):
        current = [row for row in results if row["episode"] == record["episode"]]
        paired[record["episode"]] = {
            "target_cup": record["target_cup"],
            "target_slot": record["target_slot"],
            "prediction": np.asarray(predicted).tolist(),
            "reference_joint": np.asarray(record["reference_joint"]).tolist(),
            "conditions": {row["condition"]: row for row in current},
        }
    output = {
        "experiment": "step5999 real model joint residual splice into Oracle replay",
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "current_frame": model_fk.CURRENT_FRAME,
        "spliced_raw_frames": [FIRST_ACTION_FRAME, LAST_ACTION_FRAME],
        "arm_joint_only": True,
        "gripper_source": "ground truth measured width",
        "all_other_joint_frames": "ground truth",
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
        "by_condition": by_condition,
        "paired": paired,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(by_condition, indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
