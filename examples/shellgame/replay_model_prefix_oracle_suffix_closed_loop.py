"""Normal model-prefix plus Oracle-suffix diagnostic for ShellGame joint control.

The policy always starts from the normal post-observation state (frame 59), so
the first action chunk must use the fixed 60-frame memory to choose a cup.  One
full model rollout is sampled per episode with the deployed five-step temporal
arm ensemble.  Its exact commands are then replayed up to phase boundaries,
after which the saved same-episode joint and gripper trajectory takes over.

This is the causal complement of ``replay_gt_prefix_cutoff_closed_loop.py``:
it asks how much error a normal model rollout has accumulated by the end of
selection, approach, descent, and grasp, and whether an Oracle suffix can still
recover the episode.
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
import main as base
import main_v2_absolute_joint as joint_eval
import main_v2_absolute_joint_fixed_history as fixed
import main_v2_absolute_joint_fixed_history_temporal_ensemble as temporal
import numpy as np
import oracle_joint_noise_sensitivity as noise_eval
import oracle_joint_replay as oracle
import replay_gt_prefix_cutoff_closed_loop as gt_prefix
from serve_old_tracker_full_joint_grasp import _build_config
import training_cup_eval

from openpi.policies import policy_config

DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi0_shellgame_old_tracker_full_joint_grasp_260810/"
    "full_joint_grasp_phase_balanced_b12_continue6000_260811/5999"
)
MODEL_START_FRAME = 60
SELECTION_END_FRAME = 75
APPROACH_END_FRAME = 89
DESCENT_END_FRAME = 109
GRASP_END_FRAME = 119
PREFIX_ENDS = (SELECTION_END_FRAME, APPROACH_END_FRAME, DESCENT_END_FRAME, GRASP_END_FRAME)
CONDITIONS = (
    "gt_all",
    *(f"model_to_{frame}_gt_after" for frame in PREFIX_ENDS),
    "model_all",
)


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
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--ensemble-decay", type=float, default=0.25)
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


def _policy_args(args, command_args: dict) -> joint_eval.Args:
    policy_args = noise_eval._eval_args(args, command_args)
    policy_args.num_frames = fixed.TOTAL_FRAMES
    policy_args.frame_stride = 1
    policy_args.policy_input_mode = "history"
    policy_args.action_horizon = 16
    policy_args.action_dim = joint_eval.ACTION_DIM
    policy_args.action_mode = "joint8"
    policy_args.replan_steps = args.replan_steps
    policy_args.task = model_fk.PROMPT
    return policy_args


def _query_seed(args, record: dict, current_frame: int) -> int:
    return int(args.sample_seed + record["episode_index"] * 1000 + current_frame)


def _nearest_cup(eef_xy: np.ndarray, cup_positions: dict[str, np.ndarray]) -> tuple[str, float]:
    distances = {
        cup: float(np.linalg.norm(eef_xy - np.asarray(position)[:2])) for cup, position in cup_positions.items()
    }
    cup = min(distances, key=distances.get)
    return cup, distances[cup]


def _new_accumulator(shell, scene: dict) -> dict:
    return {
        "target_xy_errors": [],
        "grasp_xy_errors": [],
        "tracking_errors": [],
        "grasp_votes": dict.fromkeys(shell.CUP_NAMES, 0),
        "grasp_distance_sums": dict.fromkeys(shell.CUP_NAMES, 0.0),
        "approach_votes": dict.fromkeys(shell.CUP_NAMES, 0),
        "approach_distance_sums": dict.fromkeys(shell.CUP_NAMES, 0.0),
        "bilateral_run": 0,
        "max_bilateral_run": 0,
        "first_bilateral_source_frame": None,
        "initial_cup_z": float(scene["settle_cup_pos"][scene["target_cup"]][2]),
        "max_target_lift_m": 0.0,
        "first_success_source_frame": None,
        "last_success_stats": None,
        "pregrasp_state_frame109": None,
    }


def _record_step(
    accumulator: dict,
    *,
    shell,
    env,
    scene: dict,
    record: dict,
    source_index: int,
    q_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    obs_after = env._get_observations(force_update=True)
    actual_joint = shell.obs_vector(obs_after, "robot0_joint_pos", size=joint_eval.JOINT_DIM)
    actual_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
    current_cups = base._cup_positions(shell, env)
    current_target = np.asarray(current_cups[scene["target_cup"]], dtype=np.float64)
    target_error = float(np.linalg.norm(actual_eef[:2] - current_target[:2]))
    accumulator["target_xy_errors"].append(target_error)
    accumulator["tracking_errors"].append(actual_joint - q_target)

    control_step = source_index - MODEL_START_FRAME
    if 8 <= control_step < 16:
        nearest, distance = _nearest_cup(actual_eef[:2], current_cups)
        if distance <= 0.08:
            accumulator["approach_votes"][nearest] += 1
            accumulator["approach_distance_sums"][nearest] += distance

    if 110 <= source_index <= GRASP_END_FRAME:
        accumulator["grasp_xy_errors"].append(target_error)
        nearest, distance = _nearest_cup(actual_eef[:2], current_cups)
        if distance <= 0.06:
            accumulator["grasp_votes"][nearest] += 1
            accumulator["grasp_distance_sums"][nearest] += distance

    contacts = noise_eval._finger_contact_count(env, scene["target_cup"])
    if contacts == 2:
        accumulator["bilateral_run"] += 1
        accumulator["max_bilateral_run"] = max(accumulator["max_bilateral_run"], accumulator["bilateral_run"])
        if accumulator["first_bilateral_source_frame"] is None:
            accumulator["first_bilateral_source_frame"] = source_index
    else:
        accumulator["bilateral_run"] = 0

    accumulator["max_target_lift_m"] = max(
        accumulator["max_target_lift_m"],
        float(current_target[2] - accumulator["initial_cup_z"]),
    )
    success, stats = base._success(
        shell,
        env,
        scene["target_cup"],
        scene["settle_cup_pos"],
        0.08,
    )
    accumulator["last_success_stats"] = stats
    if success and accumulator["first_success_source_frame"] is None:
        accumulator["first_success_source_frame"] = source_index
    if source_index == DESCENT_END_FRAME:
        accumulator["pregrasp_state_frame109"] = gt_prefix._state_metrics(
            shell, env, record, source_index, scene["target_cup"]
        )
    return actual_joint.copy(), actual_eef.copy(), current_cups


def _finish_result(
    accumulator: dict,
    *,
    record: dict,
    scene: dict,
    condition: str,
    switch_state: dict | None,
    switch_replay_error: dict | None,
    num_policy_queries: int,
) -> dict:
    approach_cup = gt_prefix._select_cup(accumulator["approach_votes"], accumulator["approach_distance_sums"])
    grasp_cup = gt_prefix._select_cup(accumulator["grasp_votes"], accumulator["grasp_distance_sums"])
    grasp_errors = accumulator["grasp_xy_errors"]
    tracking = np.asarray(accumulator["tracking_errors"])
    success = accumulator["first_success_source_frame"] is not None
    return {
        "episode": record["episode"],
        "condition": condition,
        "target_cup": scene["target_cup"],
        "target_slot": record["target_slot"],
        "success": success,
        "success_stats": accumulator["last_success_stats"],
        "first_success_source_frame": accumulator["first_success_source_frame"],
        "num_policy_queries": num_policy_queries,
        "approach_selected_cup": approach_cup,
        "approach_selection_correct": approach_cup == scene["target_cup"],
        "approach_votes": accumulator["approach_votes"],
        "grasp_selected_cup": grasp_cup,
        "grasp_selection_correct": grasp_cup == scene["target_cup"],
        "grasp_votes": accumulator["grasp_votes"],
        "switch_state": switch_state,
        "switch_replay_error": switch_replay_error,
        "pregrasp_state_frame109": accumulator["pregrasp_state_frame109"],
        "first_grasp_xy_error_m": None if not grasp_errors else float(grasp_errors[0]),
        "mean_grasp_xy_error_m": None if not grasp_errors else float(np.mean(grasp_errors)),
        "end_grasp_xy_error_m": None if not grasp_errors else float(grasp_errors[-1]),
        "target_min_xy_distance_m": float(np.min(accumulator["target_xy_errors"])),
        "first_bilateral_source_frame": accumulator["first_bilateral_source_frame"],
        "max_bilateral_run": accumulator["max_bilateral_run"],
        "max_target_lift_m": accumulator["max_target_lift_m"],
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(tracking)))),
    }


def _run_model_rollout(record: dict, shell, env, policy, args) -> tuple[dict, list[dict]]:
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
    history = gt_prefix._fixed_history(record)
    replay_images: list[np.ndarray] = []
    robot_indices = np.flatnonzero(record["action_mask"] & (record["phase_ids"] >= oracle.ROBOT_PHASE_MIN))
    action_low, action_high = (np.asarray(value, dtype=np.float64) for value in env.action_spec)
    gripper_action = float(policy_args.default_gripper_action)
    chunks: list[temporal._PredictedChunk] = []
    trace = []
    query_diagnostics = []
    accumulator = _new_accumulator(shell, scene)

    for control_step, raw_source_index in enumerate(robot_indices):
        source_index = int(raw_source_index)
        if control_step % args.replan_steps == 0:
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
            chunk = gt_prefix._infer_chunk(policy, element, seed)
            chunks.append(temporal._PredictedChunk(start_step=control_step, actions=chunk.copy()))
            chunks = [item for item in chunks if item.start_step + len(item.actions) > control_step]
            query_diagnostics.append(
                {
                    "current_frame": current_frame,
                    "source_frame": source_index,
                    "control_step": control_step,
                    "seed": seed,
                    "observed_joint": np.asarray(history[-1]["joint_pos"]).tolist(),
                }
            )

        active, predictions = temporal._predictions_for_step(
            chunks,
            control_step=control_step,
            max_chunks=0,
        )
        policy_action, weights = temporal._ensemble_predictions(predictions, decay=args.ensemble_decay)
        policy_action[-1] = predictions[-1, -1]
        q_target = np.asarray(policy_action[: joint_eval.JOINT_DIM], dtype=np.float64)
        obs_before = env._get_observations(force_update=True)
        current_width = base._gripper_width(shell.obs_vector(obs_before, "robot0_gripper_qpos"))
        target_width = float(policy_action[-1])
        if target_width < current_width - args.gripper_deadband:
            gripper_action = 1.0
        elif target_width > current_width + args.gripper_deadband:
            gripper_action = -1.0
        env_action = np.concatenate([q_target, np.asarray([gripper_action])])
        env_action = np.clip(env_action, action_low, action_high)
        env.step(env_action)
        actual_joint, actual_eef, _ = _record_step(
            accumulator,
            shell=shell,
            env=env,
            scene=scene,
            record=record,
            source_index=source_index,
            q_target=q_target,
        )
        trace.append(
            {
                "source_frame": source_index,
                "control_step": control_step,
                "arm_target": q_target.tolist(),
                "gripper_action": float(gripper_action),
                "gripper_target_width": target_width,
                "actual_joint": actual_joint.tolist(),
                "actual_eef": actual_eef.tolist(),
                "ensemble_chunk_start_steps": [item.start_step for item in active],
                "ensemble_weights": np.asarray(weights).tolist(),
            }
        )
        if accumulator["first_success_source_frame"] is not None and source_index >= GRASP_END_FRAME:
            break

    result = _finish_result(
        accumulator,
        record=record,
        scene=scene,
        condition="model_all",
        switch_state=None,
        switch_replay_error=None,
        num_policy_queries=len(query_diagnostics),
    )
    result["query_diagnostics"] = query_diagnostics
    return result, trace


def _run_replay_condition(
    record: dict,
    condition: str,
    model_trace: list[dict],
    shell,
    env,
    args,
) -> dict:
    if condition == "model_all":
        raise ValueError("model_all is produced by _run_model_rollout")
    prefix_end = None if condition == "gt_all" else int(condition.split("_")[2])
    trace_by_frame = {int(item["source_frame"]): item for item in model_trace}
    command_args = record["metadata"]["command_args"]
    policy_args = _policy_args(args, command_args)
    ep_args = joint_eval._episode_namespace(
        policy_args,
        seed=int(command_args["seed"]),
        initial_ball_cup=str(command_args["initial_ball_cup"]),
        num_swaps=int(command_args["num_swaps"]),
    )
    scene = oracle._prepare_scripted_state(shell, env, ep_args)
    robot_indices = np.flatnonzero(record["action_mask"] & (record["phase_ids"] >= oracle.ROBOT_PHASE_MIN))
    action_low, action_high = (np.asarray(value, dtype=np.float64) for value in env.action_spec)
    gripper_action = float(policy_args.default_gripper_action)
    accumulator = _new_accumulator(shell, scene)
    switch_state = None
    switch_replay_error = None

    for raw_source_index in robot_indices:
        source_index = int(raw_source_index)
        model_controls = prefix_end is not None and source_index <= prefix_end
        if model_controls:
            saved = trace_by_frame[source_index]
            q_target = np.asarray(saved["arm_target"], dtype=np.float64)
            gripper_action = float(saved["gripper_action"])
        else:
            q_target = np.asarray(record["joint_pos"][source_index], dtype=np.float64)
            obs_before = env._get_observations(force_update=True)
            current_width = base._gripper_width(shell.obs_vector(obs_before, "robot0_gripper_qpos"))
            target_width = base._gripper_width(record["gripper_state"][source_index])
            if target_width < current_width - args.gripper_deadband:
                gripper_action = 1.0
            elif target_width > current_width + args.gripper_deadband:
                gripper_action = -1.0

        env_action = np.concatenate([q_target, np.asarray([gripper_action])])
        env.step(np.clip(env_action, action_low, action_high))
        actual_joint, actual_eef, _ = _record_step(
            accumulator,
            shell=shell,
            env=env,
            scene=scene,
            record=record,
            source_index=source_index,
            q_target=q_target,
        )

        if prefix_end is not None and source_index == prefix_end:
            switch_state = gt_prefix._state_metrics(shell, env, record, source_index, scene["target_cup"])
            reference = trace_by_frame[source_index]
            switch_replay_error = {
                "joint_rmse_to_original_model_rollout_rad": float(
                    np.sqrt(np.mean(np.square(actual_joint - np.asarray(reference["actual_joint"]))))
                ),
                "eef_xyz_to_original_model_rollout_m": float(
                    np.linalg.norm(actual_eef - np.asarray(reference["actual_eef"]))
                ),
            }

        if accumulator["first_success_source_frame"] is not None and source_index >= GRASP_END_FRAME:
            break

    return _finish_result(
        accumulator,
        record=record,
        scene=scene,
        condition=condition,
        switch_state=switch_state,
        switch_replay_error=switch_replay_error,
        num_policy_queries=0,
    )


def _mean_optional(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row[key] is not None]
    return None if not values else float(np.mean(values))


def _aggregate(rows: list[dict]) -> dict:
    switch_states = [row["switch_state"] for row in rows if row["switch_state"] is not None]
    replay_errors = [row["switch_replay_error"] for row in rows if row["switch_replay_error"] is not None]
    return {
        "num_episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "approach_selection_correct": sum(row["approach_selection_correct"] for row in rows),
        "grasp_selection_correct": sum(row["grasp_selection_correct"] for row in rows),
        "bilateral_contact_episodes": sum(row["first_bilateral_source_frame"] is not None for row in rows),
        "mean_first_success_source_frame": _mean_optional(rows, "first_success_source_frame"),
        "mean_first_grasp_xy_error_m": _mean_optional(rows, "first_grasp_xy_error_m"),
        "mean_mean_grasp_xy_error_m": _mean_optional(rows, "mean_grasp_xy_error_m"),
        "mean_end_grasp_xy_error_m": _mean_optional(rows, "end_grasp_xy_error_m"),
        "mean_max_bilateral_run": float(np.mean([row["max_bilateral_run"] for row in rows])),
        "mean_max_target_lift_m": float(np.mean([row["max_target_lift_m"] for row in rows])),
        "mean_joint_tracking_rmse_rad": float(np.mean([row["joint_tracking_rmse_rad"] for row in rows])),
        "mean_switch_joint_rmse_to_saved_rad": (
            None if not switch_states else float(np.mean([item["joint_rmse_to_saved_rad"] for item in switch_states]))
        ),
        "mean_switch_eef_xyz_to_saved_m": (
            None if not switch_states else float(np.mean([item["eef_xyz_to_saved_m"] for item in switch_states]))
        ),
        "mean_switch_eef_target_xy_m": (
            None if not switch_states else float(np.mean([item["eef_target_xy_m"] for item in switch_states]))
        ),
        "mean_prefix_replay_joint_rmse_rad": (
            None
            if not replay_errors
            else float(np.mean([item["joint_rmse_to_original_model_rollout_rad"] for item in replay_errors]))
        ),
        "mean_prefix_replay_eef_xyz_m": (
            None
            if not replay_errors
            else float(np.mean([item["eef_xyz_to_original_model_rollout_m"] for item in replay_errors]))
        ),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.replan_steps <= 16:
        raise ValueError("--replan-steps must be in [1, 16]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    episode_dirs = sorted(path for path in args.dataset_root.expanduser().resolve().glob("episode_*") if path.is_dir())
    selected_ids = training_cup_eval._balanced_validation_ids(
        episode_dirs,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        sample_seed=args.sample_seed,
        num_episodes=args.num_episodes,
    )
    records = [gt_prefix._load_record(episode_dirs[int(index)]) for index in selected_ids]
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
    completed = 0
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
            model_row, model_trace = _run_model_rollout(record, shell, env, policy, args)
            for condition in CONDITIONS:
                if condition == "model_all":
                    row = model_row
                else:
                    row = _run_replay_condition(record, condition, model_trace, shell, env, args)
                results.append(row)
                completed += 1
                switch = row["switch_state"]
                logging.info(
                    "[%d/%d] %s %s success=%s approach=%s switch_xyz=%s grasp_xy=%s bi=%d",
                    completed,
                    total,
                    record["episode"],
                    condition,
                    row["success"],
                    row["approach_selection_correct"],
                    None if switch is None else f"{switch['eef_xyz_to_saved_m'] * 1000:.1f}mm",
                    None if row["mean_grasp_xy_error_m"] is None else f"{row['mean_grasp_xy_error_m'] * 1000:.1f}mm",
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
        "experiment": "normal model-prefix plus same-episode Oracle-suffix closed-loop diagnostic",
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "conditions": list(CONDITIONS),
        "model_start_frame": MODEL_START_FRAME,
        "prefix_end_frames": list(PREFIX_ENDS),
        "phase_frames": {
            "selection_chunk": [60, 75],
            "approach": [60, 89],
            "descent": [90, 109],
            "grasp": [110, 119],
            "lift": [120, 154],
        },
        "model_rollout": {
            "replan_steps": args.replan_steps,
            "action_horizon": 16,
            "arm_ensemble": "oldest-to-newest exponential temporal ensemble",
            "ensemble_decay": args.ensemble_decay,
            "gripper": "newest chunk target width",
            "fixed_memory_frames": [0, 59],
            "dynamic_current_frame": True,
        },
        "hybrid_replay": "exact recorded model commands through prefix end, then saved same-episode joint/gripper targets",
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
