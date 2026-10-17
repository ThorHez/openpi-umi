"""Local-policy GT-prefix cutoff diagnostic for absolute-joint drift.

For each held-out episode, Oracle JOINT_POSITION replay controls the robot up
to a chosen action-frame cutoff.  The step5999 policy then controls arm joints
in closed loop through frame 125 with 16-step replanning.  Ground-truth gripper
commands are retained, and arm control returns to GT at frame 126.  Cutoffs
110, 94, 78, and 62 correspond to 0, 16, 32, and 48 model-controlled steps
before the ten-frame grasp phase begins at frame 110.
"""

# This diagnostic intentionally reuses private policy, evaluator, and
# Robosuite interfaces to preserve exact preprocessing and controller semantics.
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
import main_v2_absolute_joint_fixed_history as fixed
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
MODEL_END_FRAME = 125
GRASP_START_FRAME = 110
GRASP_END_FRAME = 119
CUTOFFS = (110, 94, 78, 62)
CONDITIONS = ("gt", *(f"cutoff_{cutoff}" for cutoff in CUTOFFS))


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
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=260811)
    parser.add_argument("--joint-kp", type=float, default=50.0)
    parser.add_argument("--joint-damping-ratio", type=float, default=1.0)
    parser.add_argument("--lift-success-height", type=float, default=0.08)
    parser.add_argument("--gripper-deadband", type=float, default=0.004)
    parser.add_argument("--camera-size", type=int, default=224)
    parser.add_argument("--selection-radius", type=float, default=0.06)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_record(episode_dir: Path) -> dict:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
        return {
            "episode": episode_dir.name,
            "episode_index": int(episode_dir.name.split("_")[-1]),
            "metadata": metadata,
            "target_cup": str(metadata["target_cup_identity"]),
            "target_slot": str(metadata["final_ball_cup"]),
            "fixed_wrist": np.asarray(source["wrist_images"][: fixed.HISTORY_FRAMES]),
            "fixed_base": np.asarray(source["third_person_images"][: fixed.HISTORY_FRAMES]),
            "joint_pos": np.asarray(source["joint_pos"], dtype=np.float32),
            "eef_pos": np.asarray(source["eef_pos"], dtype=np.float32),
            "gripper_state": np.asarray(source["gripper_state"], dtype=np.float32),
            "action_mask": np.asarray(source["action_mask"], dtype=bool),
            "phase_ids": np.asarray(source["phase_ids"], dtype=np.int16),
        }


def _policy_args(args, command_args: dict) -> joint_eval.Args:
    policy_args = noise_eval._eval_args(args, command_args)
    policy_args.num_frames = fixed.TOTAL_FRAMES
    policy_args.frame_stride = 1
    policy_args.policy_input_mode = "history"
    policy_args.action_horizon = 16
    policy_args.action_dim = joint_eval.ACTION_DIM
    policy_args.action_mode = "joint8"
    policy_args.task = model_fk.PROMPT
    return policy_args


def _fixed_history(record: dict) -> list[dict]:
    return [
        {
            "wrist": record["fixed_wrist"][index],
            "base": record["fixed_base"][index],
        }
        for index in range(fixed.HISTORY_FRAMES)
    ]


def _query_seed(args, record: dict, current_frame: int) -> int:
    return int(args.sample_seed + record["episode_index"] * 1000 + current_frame)


def _infer_chunk(policy, element: dict, seed: int) -> np.ndarray:
    chunk = fk_eval._batched_infer(policy, [element], batch_size=1, seed=seed)[0]
    chunk = np.asarray(chunk, dtype=np.float32)
    expected = (16, joint_eval.ACTION_DIM)
    if chunk.shape != expected:
        raise RuntimeError(f"Policy returned {chunk.shape}, expected {expected}")
    return chunk


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


def _state_metrics(shell, env, record: dict, source_frame: int, target_cup: str) -> dict:
    obs = env._get_observations(force_update=True)
    actual_joint = shell.obs_vector(obs, "robot0_joint_pos", size=joint_eval.JOINT_DIM)
    actual_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
    cup = np.asarray(base._cup_positions(shell, env)[target_cup], dtype=np.float64)
    return {
        "source_frame": source_frame,
        "joint_rmse_to_saved_rad": float(np.sqrt(np.mean(np.square(actual_joint - record["joint_pos"][source_frame])))),
        "eef_xy_to_saved_m": float(np.linalg.norm(actual_eef[:2] - record["eef_pos"][source_frame, :2])),
        "eef_xyz_to_saved_m": float(np.linalg.norm(actual_eef - record["eef_pos"][source_frame])),
        "eef_target_xy_m": float(np.linalg.norm(actual_eef[:2] - cup[:2])),
        "eef_target_dz_m": float(actual_eef[2] - cup[2]),
    }


def _run_condition(record: dict, condition: str, shell, env, policy, args) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    cutoff = None if condition == "gt" else int(condition.split("_")[-1])
    command_args = record["metadata"]["command_args"]
    policy_args = _policy_args(args, command_args)
    ep_args = joint_eval._episode_namespace(
        policy_args,
        seed=int(command_args["seed"]),
        initial_ball_cup=str(command_args["initial_ball_cup"]),
        num_swaps=int(command_args["num_swaps"]),
    )
    scene = oracle._prepare_scripted_state(shell, env, ep_args)
    wrist_camera = shell.resolve_wrist_camera_name(env, ep_args.wrist_camera)
    history = _fixed_history(record)
    replay_images: list[np.ndarray] = []
    robot_indices = np.flatnonzero(record["action_mask"] & (record["phase_ids"] >= oracle.ROBOT_PHASE_MIN))
    action_low, action_high = (np.asarray(value, dtype=np.float64) for value in env.action_spec)
    gripper_action = float(policy_args.default_gripper_action)
    action_plan: collections.deque[np.ndarray] = collections.deque()
    query_diagnostics = []
    votes = dict.fromkeys(shell.CUP_NAMES, 0)
    distance_sums = dict.fromkeys(shell.CUP_NAMES, 0.0)
    target_xy_errors = []
    grasp_xy_errors = []
    tracking_errors = []
    bilateral_run = 0
    max_bilateral_run = 0
    first_bilateral_source_frame = None
    initial_cup_z = float(scene["settle_cup_pos"][scene["target_cup"]][2])
    max_target_lift = 0.0
    handoff_state = None
    pregrasp_state = None

    for raw_source_index in robot_indices:
        source_index = int(raw_source_index)
        model_controls = cutoff is not None and cutoff <= source_index <= MODEL_END_FRAME
        if model_controls:
            if source_index == cutoff:
                handoff_state = _state_metrics(shell, env, record, source_index - 1, scene["target_cup"])
            if not action_plan:
                joint_eval._append_observation(
                    shell,
                    env,
                    ep_args,
                    wrist_camera,
                    history,
                    replay_images,
                    resize_size=args.camera_size,
                )
                element = fixed._fixed_history_policy_input(
                    history,
                    np.asarray(shell.get_eef_pos(env), dtype=np.float32),
                    args=policy_args,
                    prompt=model_fk.PROMPT,
                )
                current_frame = source_index - 1
                seed = _query_seed(args, record, current_frame)
                chunk = _infer_chunk(policy, element, seed)
                action_plan.extend(chunk[:, : joint_eval.JOINT_DIM])
                query_diagnostics.append(
                    {
                        "current_frame": current_frame,
                        "first_action_frame": source_index,
                        "seed": seed,
                        "observed_joint": np.asarray(history[-1]["joint_pos"]).tolist(),
                        "predicted_arm_chunk": chunk[:, : joint_eval.JOINT_DIM].tolist(),
                    }
                )
            q_target = np.asarray(action_plan.popleft(), dtype=np.float64)
        else:
            q_target = np.asarray(record["joint_pos"][source_index], dtype=np.float64)

        if source_index == GRASP_START_FRAME:
            pregrasp_state = _state_metrics(shell, env, record, source_index - 1, scene["target_cup"])

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

        if GRASP_START_FRAME <= source_index <= GRASP_END_FRAME:
            grasp_xy_errors.append(target_error)
            nearest, distance = _nearest_cup(actual_eef[:2], current_cups)
            if distance <= args.selection_radius:
                votes[nearest] += 1
                distance_sums[nearest] += distance

        contacts = noise_eval._finger_contact_count(env, scene["target_cup"])
        if contacts == 2:
            bilateral_run += 1
            max_bilateral_run = max(max_bilateral_run, bilateral_run)
            if first_bilateral_source_frame is None:
                first_bilateral_source_frame = source_index
        else:
            bilateral_run = 0
        max_target_lift = max(max_target_lift, float(current_target[2] - initial_cup_z))

    selected_cup = _select_cup(votes, distance_sums)
    success, success_stats = base._success(
        shell,
        env,
        scene["target_cup"],
        scene["settle_cup_pos"],
        args.lift_success_height,
    )
    tracking = np.asarray(tracking_errors)
    return {
        "episode": record["episode"],
        "condition": condition,
        "cutoff_action_frame": cutoff,
        "model_steps_before_grasp": 0 if cutoff is None else GRASP_START_FRAME - cutoff,
        "num_policy_queries": len(query_diagnostics),
        "target_cup": scene["target_cup"],
        "target_slot": record["target_slot"],
        "actual_grasp_selected_cup": selected_cup,
        "actual_grasp_selection_correct": selected_cup == scene["target_cup"],
        "actual_grasp_votes": votes,
        "success": bool(success),
        "success_stats": success_stats,
        "handoff_state": handoff_state,
        "pregrasp_state_frame109": pregrasp_state,
        "first_grasp_xy_error_m": float(grasp_xy_errors[0]),
        "mean_grasp_xy_error_m": float(np.mean(grasp_xy_errors)),
        "end_grasp_xy_error_m": float(grasp_xy_errors[-1]),
        "target_min_xy_distance_m": float(np.min(target_xy_errors)),
        "first_bilateral_source_frame": first_bilateral_source_frame,
        "max_bilateral_run": max_bilateral_run,
        "max_target_lift_m": max_target_lift,
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(tracking)))),
        "query_diagnostics": query_diagnostics,
    }


def _aggregate(rows: list[dict]) -> dict:
    pregrasp = [row["pregrasp_state_frame109"] for row in rows]
    return {
        "num_episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "actual_grasp_selection_correct": sum(row["actual_grasp_selection_correct"] for row in rows),
        "bilateral_contact_episodes": sum(row["first_bilateral_source_frame"] is not None for row in rows),
        "mean_num_policy_queries": float(np.mean([row["num_policy_queries"] for row in rows])),
        "mean_pregrasp_joint_rmse_to_saved_rad": float(np.mean([item["joint_rmse_to_saved_rad"] for item in pregrasp])),
        "mean_pregrasp_eef_xy_to_saved_m": float(np.mean([item["eef_xy_to_saved_m"] for item in pregrasp])),
        "mean_pregrasp_eef_target_xy_m": float(np.mean([item["eef_target_xy_m"] for item in pregrasp])),
        **{
            f"mean_{key}": float(np.mean([row[key] for row in rows]))
            for key in (
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
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
    shell = base._import_shellgame_tools(args.robosuite_root)
    results = []
    total = len(records) * len(CONDITIONS)
    count = 0
    for record in records:
        command_args = record["metadata"]["command_args"]
        policy_args = _policy_args(args, command_args)
        ep_args = joint_eval._episode_namespace(
            policy_args,
            seed=int(command_args["seed"]),
            initial_ball_cup=str(command_args["initial_ball_cup"]),
            num_swaps=int(command_args["num_swaps"]),
        )
        env = shell.make_env(ep_args)
        try:
            for condition in CONDITIONS:
                count += 1
                row = _run_condition(record, condition, shell, env, policy, args)
                results.append(row)
                logging.info(
                    "[%d/%d] %s %s success=%s pregrasp_xy=%.1fmm grasp_xy=%.1fmm bi=%d",
                    count,
                    total,
                    record["episode"],
                    condition,
                    row["success"],
                    row["pregrasp_state_frame109"]["eef_target_xy_m"] * 1000,
                    row["mean_grasp_xy_error_m"] * 1000,
                    row["max_bilateral_run"],
                )
        finally:
            env.close()

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
    output = {
        "experiment": "GT-prefix cutoff local closed-loop absolute-joint diagnostic",
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "cutoff_action_frames": list(CUTOFFS),
        "grasp_action_frames": [GRASP_START_FRAME, GRASP_END_FRAME],
        "model_control_end_frame": MODEL_END_FRAME,
        "arm_joint_only": True,
        "gripper_source": "ground truth measured width",
        "post_125_arm_source": "ground truth",
        "deterministic_query_seed": "sample_seed + episode_index*1000 + current_frame",
        "settings": {
            "num_episodes": args.num_episodes,
            "num_sampling_steps": args.num_sampling_steps,
            "split_seed": args.split_seed,
            "val_ratio": args.val_ratio,
            "sample_seed": args.sample_seed,
            "selected_episode_ids": selected_ids.tolist(),
        },
        "target_slot_distribution": dict(collections.Counter(record["target_slot"] for record in records)),
        "by_condition": by_condition,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(by_condition, indent=2))
    print(f"Wrote {args.output.resolve()}")
    del policy
    jax.clear_caches()
    gc.collect()


if __name__ == "__main__":
    main()
