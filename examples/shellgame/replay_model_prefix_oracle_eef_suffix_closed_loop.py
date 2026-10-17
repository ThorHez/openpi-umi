"""Normal absolute-EEF model prefix followed by the same-episode Oracle suffix.

The first 60 visual frames and every policy query are unchanged from deployed
fixed-history evaluation.  A model rollout is sampled once, then its exact
native OSC commands are replayed through selected phase boundaries.  After the
boundary, the original demonstration's absolute EEF controller commands take
over.  In particular, ``model_to_89_oracle_after`` keeps normal memory-based
cup selection and approach, while replacing descend, grasp, and lift.
"""

# The diagnostic intentionally shares private evaluator helpers so that its
# preprocessing and simulator semantics stay identical to online evaluation.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import collections
import dataclasses
import gc
import json
import logging
from pathlib import Path

import jax
import numpy as np

import joint_fk_selection_eval as infer_utils
import main as base
import main_absolute_eef_fixed_history as fixed_eef
import oracle_joint_noise_sensitivity as contact_utils
import oracle_joint_replay as oracle
from serve_old_tracker_full_absolute_eef import DEFAULT_CHECKPOINT, _build_config
import training_cup_eval

from examples.shellgame.eval_old_tracker_query_action_closed_loop_gate import PROMPT
from openpi.policies import policy_config


MODEL_START_FRAME = 60
SELECTION_CHUNK_END_FRAME = 75
APPROACH_END_FRAME = 89
DESCENT_END_FRAME = 109
GRASP_END_FRAME = 119
PREFIX_ENDS = (
    SELECTION_CHUNK_END_FRAME,
    APPROACH_END_FRAME,
    DESCENT_END_FRAME,
    GRASP_END_FRAME,
)
CONDITIONS = (
    "oracle_all",
    *(f"model_to_{frame}_oracle_after" for frame in PREFIX_ENDS),
    "model_all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../robosuite/outputs/shellgame_absolute_eef_phase_instruction_dataset"),
    )
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--remote-host", default=None)
    parser.add_argument("--remote-port", type=int, default=8000)
    parser.add_argument("--replan-steps", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=260813)
    parser.add_argument("--lift-success-height", type=float, default=0.08)
    parser.add_argument("--selection-radius", type=float, default=0.06)
    parser.add_argument("--camera-size", type=int, default=224)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--video-conditions",
        nargs="*",
        default=("oracle_all", "model_to_89_oracle_after", "model_all"),
    )
    parser.add_argument("--conditions", nargs="*", default=CONDITIONS)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _policy_args(cli_args, command_args: dict) -> base.Args:
    args = base.Args()
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
    args.save_videos = False
    args.num_frames = fixed_eef.TOTAL_FRAMES
    args.frame_stride = 1
    args.policy_input_mode = "history"
    args.action_horizon = 16
    args.action_dim = 7
    args.action_mode = "raw7"
    args.observation_position_frame = "absolute"
    args.osc_input_type = "absolute"
    args.replan_steps = cli_args.replan_steps
    args.task = PROMPT
    args.phase_instructions = True
    args.grasp_task = PROMPT
    return args


def _load_record(episode_dir: Path) -> dict:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
        return {
            "episode": episode_dir.name,
            "episode_index": int(episode_dir.name.split("_")[-1]),
            "metadata": metadata,
            "target_cup": str(metadata["target_cup_identity"]),
            "target_slot": str(metadata["final_ball_cup"]),
            "fixed_wrist": np.asarray(source["wrist_images"][: fixed_eef.HISTORY_FRAMES]),
            "fixed_base": np.asarray(source["third_person_images"][: fixed_eef.HISTORY_FRAMES]),
            "eef_pos": np.asarray(source["eef_pos"], dtype=np.float32),
            "controller_actions": np.asarray(source["controller_actions"], dtype=np.float32),
            "action_mask": np.asarray(source["action_mask"], dtype=bool),
            "phase_ids": np.asarray(source["phase_ids"], dtype=np.int16),
        }


def _fixed_history(record: dict) -> list[dict]:
    return [
        {"wrist": record["fixed_wrist"][index], "base": record["fixed_base"][index]}
        for index in range(fixed_eef.HISTORY_FRAMES)
    ]


def _episode_args(record: dict, cli_args) -> tuple[base.Args, object]:
    command = record["metadata"]["command_args"]
    args = _policy_args(cli_args, command)
    ep_args = base._episode_namespace(
        args,
        seed=int(command["seed"]),
        initial_ball_cup=str(command["initial_ball_cup"]),
        num_swaps=int(command["num_swaps"]),
    )
    return args, ep_args


def _infer_chunk(policy, element: dict, seed: int) -> np.ndarray:
    if hasattr(policy, "infer"):
        # The deployed two-process path. The server owns JAX/CUDA while this
        # evaluator owns EGL, avoiding corrupted camera frames after inference.
        chunk = np.asarray(policy.infer(element)["actions"], dtype=np.float32)
    else:
        chunk = np.asarray(
            infer_utils._batched_infer(policy, [element], batch_size=1, seed=seed)[0],
            dtype=np.float32,
        )
    if chunk.shape != (16, 7):
        raise RuntimeError(f"Policy returned {chunk.shape}, expected (16, 7)")
    if not np.all(np.isfinite(chunk)):
        raise RuntimeError("Policy returned a non-finite absolute EEF command")
    return chunk


def _query_seed(args, record: dict, current_frame: int) -> int:
    return int(args.sample_seed + record["episode_index"] * 1000 + current_frame)


def _nearest_cup(eef_xy: np.ndarray, cups: dict[str, np.ndarray]) -> tuple[str, float]:
    distances = {name: float(np.linalg.norm(eef_xy - pos[:2])) for name, pos in cups.items()}
    name = min(distances, key=distances.get)
    return name, distances[name]


def _select_cup(votes: dict[str, int], distance_sums: dict[str, float]) -> str | None:
    best = max(votes.values(), default=0)
    if best <= 0:
        return None
    candidates = [name for name, count in votes.items() if count == best]
    return min(candidates, key=lambda name: distance_sums[name] / votes[name])


def _new_accumulator(shell, scene: dict) -> dict:
    return {
        "approach_votes": dict.fromkeys(shell.CUP_NAMES, 0),
        "approach_distance_sums": dict.fromkeys(shell.CUP_NAMES, 0.0),
        "grasp_votes": dict.fromkeys(shell.CUP_NAMES, 0),
        "grasp_distance_sums": dict.fromkeys(shell.CUP_NAMES, 0.0),
        "target_xy_errors": [],
        "grasp_xy_errors": [],
        "command_position_errors": [],
        "bilateral_run": 0,
        "max_bilateral_run": 0,
        "first_bilateral_source_frame": None,
        "initial_cup_z": float(scene["settle_cup_pos"][scene["target_cup"]][2]),
        "max_target_lift_m": 0.0,
        "first_success_source_frame": None,
        "last_success_stats": None,
    }


def _record_step(acc: dict, *, shell, env, scene: dict, source_index: int, command: np.ndarray) -> dict:
    actual_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
    cups = base._cup_positions(shell, env)
    target = np.asarray(cups[scene["target_cup"]], dtype=np.float64)
    target_xy_error = float(np.linalg.norm(actual_eef[:2] - target[:2]))
    acc["target_xy_errors"].append(target_xy_error)
    acc["command_position_errors"].append(actual_eef - np.asarray(command[:3], dtype=np.float64))

    # Restrict the diagnostic vote to the approach phase.  In the primary
    # model-to-89 condition every one of these frames is model-controlled, so
    # the Oracle suffix cannot inflate the measured memory-based selection.
    if 70 <= source_index <= APPROACH_END_FRAME:
        nearest, distance = _nearest_cup(actual_eef[:2], cups)
        if distance <= 0.06:
            acc["approach_votes"][nearest] += 1
            acc["approach_distance_sums"][nearest] += distance
    if 110 <= source_index <= GRASP_END_FRAME:
        acc["grasp_xy_errors"].append(target_xy_error)
        nearest, distance = _nearest_cup(actual_eef[:2], cups)
        if distance <= 0.06:
            acc["grasp_votes"][nearest] += 1
            acc["grasp_distance_sums"][nearest] += distance

    contacts = contact_utils._finger_contact_count(env, scene["target_cup"])
    if contacts == 2:
        acc["bilateral_run"] += 1
        acc["max_bilateral_run"] = max(acc["max_bilateral_run"], acc["bilateral_run"])
        if acc["first_bilateral_source_frame"] is None:
            acc["first_bilateral_source_frame"] = source_index
    else:
        acc["bilateral_run"] = 0

    acc["max_target_lift_m"] = max(
        acc["max_target_lift_m"], float(target[2] - acc["initial_cup_z"])
    )
    success, stats = base._success(shell, env, scene["target_cup"], scene["settle_cup_pos"], 0.08)
    acc["last_success_stats"] = stats
    if success and acc["first_success_source_frame"] is None:
        acc["first_success_source_frame"] = source_index
    return {"actual_eef": actual_eef.tolist(), "target_xy_error_m": target_xy_error}


def _state_metrics(shell, env, record: dict, source_frame: int, target_cup: str) -> dict:
    actual_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
    target = np.asarray(base._cup_positions(shell, env)[target_cup], dtype=np.float64)
    return {
        "source_frame": source_frame,
        "eef_xy_to_saved_m": float(np.linalg.norm(actual_eef[:2] - record["eef_pos"][source_frame, :2])),
        "eef_xyz_to_saved_m": float(np.linalg.norm(actual_eef - record["eef_pos"][source_frame])),
        "eef_target_xy_m": float(np.linalg.norm(actual_eef[:2] - target[:2])),
        "eef_target_dz_m": float(actual_eef[2] - target[2]),
    }


def _finish(acc: dict, *, record: dict, scene: dict, condition: str, switch_state, replay_error, queries: int) -> dict:
    command_errors = np.asarray(acc["command_position_errors"], dtype=np.float64)
    grasp_errors = acc["grasp_xy_errors"]
    approach_cup = _select_cup(acc["approach_votes"], acc["approach_distance_sums"])
    grasp_cup = _select_cup(acc["grasp_votes"], acc["grasp_distance_sums"])
    return {
        "episode": record["episode"],
        "condition": condition,
        "target_cup": scene["target_cup"],
        "target_slot": record["target_slot"],
        "success": acc["first_success_source_frame"] is not None,
        "success_stats": acc["last_success_stats"],
        "first_success_source_frame": acc["first_success_source_frame"],
        "num_policy_queries": queries,
        "selection_cup": approach_cup,
        "selection_correct": approach_cup == scene["target_cup"],
        "selection_votes": acc["approach_votes"],
        "grasp_cup": grasp_cup,
        "grasp_selection_correct": grasp_cup == scene["target_cup"],
        "grasp_votes": acc["grasp_votes"],
        "switch_state": switch_state,
        "switch_replay_error": replay_error,
        "first_grasp_xy_error_m": None if not grasp_errors else float(grasp_errors[0]),
        "mean_grasp_xy_error_m": None if not grasp_errors else float(np.mean(grasp_errors)),
        "end_grasp_xy_error_m": None if not grasp_errors else float(grasp_errors[-1]),
        "target_min_xy_distance_m": float(np.min(acc["target_xy_errors"])),
        "first_bilateral_source_frame": acc["first_bilateral_source_frame"],
        "max_bilateral_run": acc["max_bilateral_run"],
        "max_target_lift_m": acc["max_target_lift_m"],
        "eef_command_position_rmse_m": float(np.sqrt(np.mean(np.square(command_errors)))),
    }


def _save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def _append_current(shell, env, ep_args, wrist_camera, history, frames, size: int) -> None:
    sink: list[np.ndarray] = []
    base._append_observation(
        shell, env, ep_args, wrist_camera, history, sink, resize_size=size
    )
    if frames is not None:
        frames.extend(sink)


def _run_model(record: dict, shell, policy, args) -> tuple[dict, list[dict], list[np.ndarray]]:
    policy_args, ep_args = _episode_args(record, args)
    env = shell.make_env(ep_args)
    try:
        scene = oracle._prepare_scripted_state(shell, env, ep_args)
        if scene["final_ball_cup"] != record["target_slot"]:
            raise RuntimeError(f"Scene mismatch for {record['episode']}: {scene['final_ball_cup']} != {record['target_slot']}")
        wrist_camera = shell.resolve_wrist_camera_name(env, ep_args.wrist_camera)
        history = _fixed_history(record)
        video = [np.asarray(frame, dtype=np.uint8) for frame in record["fixed_base"]]
        robot_indices = np.flatnonzero(record["action_mask"] & (record["phase_ids"] >= oracle.ROBOT_PHASE_MIN))
        low, high = (np.asarray(value, dtype=np.float32) for value in env.action_spec)
        action_plan: collections.deque[np.ndarray] = collections.deque()
        trace: list[dict] = []
        acc = _new_accumulator(shell, scene)
        queries = 0

        for control_step, raw_index in enumerate(robot_indices):
            source_index = int(raw_index)
            if not action_plan:
                _append_current(shell, env, ep_args, wrist_camera, history, None, args.camera_size)
                element = fixed_eef._fixed_history_policy_input(
                    history,
                    np.asarray(shell.get_eef_pos(env), dtype=np.float32),
                    args=policy_args,
                    prompt=PROMPT,
                )
                current_frame = source_index - 1
                chunk = _infer_chunk(policy, element, _query_seed(args, record, current_frame))
                action_plan.extend(chunk[: args.replan_steps])
                queries += 1
            command = np.clip(np.asarray(action_plan.popleft(), dtype=np.float32), low, high)
            env.step(command)
            step_metrics = _record_step(
                acc, shell=shell, env=env, scene=scene, source_index=source_index, command=command
            )
            _append_current(shell, env, ep_args, wrist_camera, history, video, args.camera_size)
            trace.append(
                {
                    "source_frame": source_index,
                    "control_step": control_step,
                    "command": command.tolist(),
                    **step_metrics,
                }
            )

        result = _finish(
            acc,
            record=record,
            scene=scene,
            condition="model_all",
            switch_state=None,
            replay_error=None,
            queries=queries,
        )
        return result, trace, video
    finally:
        env.close()


def _run_replay(record: dict, condition: str, model_trace: list[dict], shell, args) -> tuple[dict, list[np.ndarray]]:
    prefix_end = None if condition == "oracle_all" else int(condition.split("_")[2])
    trace_by_frame = {int(item["source_frame"]): item for item in model_trace}
    _, ep_args = _episode_args(record, args)
    env = shell.make_env(ep_args)
    try:
        scene = oracle._prepare_scripted_state(shell, env, ep_args)
        wrist_camera = shell.resolve_wrist_camera_name(env, ep_args.wrist_camera)
        history = _fixed_history(record)
        save_video = args.save_videos and condition in args.video_conditions
        video = [np.asarray(frame, dtype=np.uint8) for frame in record["fixed_base"]] if save_video else []
        robot_indices = np.flatnonzero(record["action_mask"] & (record["phase_ids"] >= oracle.ROBOT_PHASE_MIN))
        low, high = (np.asarray(value, dtype=np.float32) for value in env.action_spec)
        acc = _new_accumulator(shell, scene)
        switch_state = None
        replay_error = None

        for raw_index in robot_indices:
            source_index = int(raw_index)
            model_controls = prefix_end is not None and source_index <= prefix_end
            if model_controls:
                command = np.asarray(trace_by_frame[source_index]["command"], dtype=np.float32)
            else:
                command = np.asarray(record["controller_actions"][source_index], dtype=np.float32)
            command = np.clip(command, low, high)
            env.step(command)
            metrics = _record_step(
                acc, shell=shell, env=env, scene=scene, source_index=source_index, command=command
            )
            if save_video:
                _append_current(shell, env, ep_args, wrist_camera, history, video, args.camera_size)
            if prefix_end is not None and source_index == prefix_end:
                switch_state = _state_metrics(shell, env, record, source_index, scene["target_cup"])
                reference = trace_by_frame[source_index]
                replay_error = {
                    "eef_xyz_to_original_model_rollout_m": float(
                        np.linalg.norm(
                            np.asarray(metrics["actual_eef"]) - np.asarray(reference["actual_eef"])
                        )
                    )
                }

        result = _finish(
            acc,
            record=record,
            scene=scene,
            condition=condition,
            switch_state=switch_state,
            replay_error=replay_error,
            queries=0,
        )
        return result, video
    finally:
        env.close()


def _mean_optional(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row[key] is not None]
    return None if not values else float(np.mean(values))


def _aggregate(rows: list[dict]) -> dict:
    switch = [row["switch_state"] for row in rows if row["switch_state"] is not None]
    replay = [row["switch_replay_error"] for row in rows if row["switch_replay_error"] is not None]
    return {
        "num_episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "selection_correct": sum(row["selection_correct"] for row in rows),
        "selection_accuracy": float(np.mean([row["selection_correct"] for row in rows])),
        "bilateral_contact_episodes": sum(row["first_bilateral_source_frame"] is not None for row in rows),
        "mean_first_grasp_xy_error_m": _mean_optional(rows, "first_grasp_xy_error_m"),
        "mean_grasp_xy_error_m": _mean_optional(rows, "mean_grasp_xy_error_m"),
        "mean_max_bilateral_run": float(np.mean([row["max_bilateral_run"] for row in rows])),
        "mean_max_target_lift_m": float(np.mean([row["max_target_lift_m"] for row in rows])),
        "mean_eef_command_position_rmse_m": float(np.mean([row["eef_command_position_rmse_m"] for row in rows])),
        "mean_switch_eef_xyz_to_saved_m": (
            None if not switch else float(np.mean([item["eef_xyz_to_saved_m"] for item in switch]))
        ),
        "mean_switch_eef_target_xy_m": (
            None if not switch else float(np.mean([item["eef_target_xy_m"] for item in switch]))
        ),
        "mean_prefix_replay_eef_xyz_m": (
            None
            if not replay
            else float(np.mean([item["eef_xyz_to_original_model_rollout_m"] for item in replay]))
        ),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.replan_steps <= 16:
        raise ValueError("--replan-steps must be in [1, 16]")
    unknown_conditions = set(args.conditions) - set(CONDITIONS)
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {sorted(unknown_conditions)}")
    if "model_all" not in args.conditions:
        raise ValueError("--conditions must contain model_all because it generates the causal prefix trace")
    unknown_video = set(args.video_conditions) - set(args.conditions)
    if unknown_video:
        raise ValueError(f"Unknown video conditions: {sorted(unknown_video)}")
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
    logging.info("Loaded %d balanced held-out absolute-EEF episodes", len(records))

    if args.remote_host is not None:
        from openpi_client import websocket_client_policy

        policy = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
        logging.info("Using remote policy at ws://%s:%d", args.remote_host, args.remote_port)
    else:
        config = _build_config(sampling_steps=args.num_sampling_steps)
        policy = policy_config.create_trained_policy(
            config,
            args.checkpoint_dir,
            default_prompt=PROMPT,
            sample_kwargs={"num_steps": args.num_sampling_steps},
        )
    shell = base._import_shellgame_tools(args.robosuite_root)
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    try:
        for episode_number, record in enumerate(records, start=1):
            model_row, trace, model_video = _run_model(record, shell, policy, args)
            for condition in args.conditions:
                if condition == "model_all":
                    row, video = model_row, model_video
                else:
                    row, video = _run_replay(record, condition, trace, shell, args)
                rows.append(row)
                if args.save_videos and condition in args.video_conditions:
                    suffix = "success" if row["success"] else "failure"
                    _save_video(
                        output_dir / "videos" / condition / f"{record['episode']}_{suffix}.mp4",
                        video,
                        fps=10,
                    )
                switch = row["switch_state"]
                logging.info(
                    "[%d/%d] %s %-29s selection=%s success=%s switch_saved=%s grasp_xy=%s",
                    episode_number,
                    len(records),
                    record["episode"],
                    condition,
                    row["selection_correct"],
                    row["success"],
                    None if switch is None else f"{switch['eef_xyz_to_saved_m'] * 1000:.1f}mm",
                    None if row["mean_grasp_xy_error_m"] is None else f"{row['mean_grasp_xy_error_m'] * 1000:.1f}mm",
                )
    finally:
        del policy
        if args.remote_host is None:
            jax.clear_caches()
            gc.collect()

    by_condition = {}
    for condition in args.conditions:
        current = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "overall": _aggregate(current),
            "by_target_slot": {
                slot: _aggregate([row for row in current if row["target_slot"] == slot])
                for slot in ("left", "middle", "right")
            },
        }
    payload = {
        "experiment": "normal fixed-history absolute-EEF model prefix plus same-episode Oracle EEF suffix",
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "conditions": list(args.conditions),
        "phase_frames": {
            "approach": [60, 89],
            "descent": [90, 109],
            "grasp": [110, 119],
            "lift": [120, 154],
        },
        "primary_condition": "model_to_89_oracle_after",
        "hybrid_semantics": "replay exact model raw7 commands through prefix end, then original same-episode absolute OSC controller_actions",
        "settings": {
            "num_episodes": args.num_episodes,
            "num_sampling_steps": args.num_sampling_steps,
            "replan_steps": args.replan_steps,
            "split_seed": args.split_seed,
            "val_ratio": args.val_ratio,
            "sample_seed": args.sample_seed,
            "selected_episode_ids": selected_ids.tolist(),
        },
        "target_slot_distribution": dict(collections.Counter(record["target_slot"] for record in records)),
        "by_condition": by_condition,
        "results": rows,
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(by_condition, indent=2))
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
