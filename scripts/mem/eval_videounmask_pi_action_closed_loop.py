#!/usr/bin/env python3
"""Closed-loop VideoUnmask evaluation for the point-conditioned Pi action expert.

The target point can come from the deterministic oracle control or from the
single-task fixed-chunk recurrent MEM.  In the latter case the shared readout
decodes the remembered target region and that region selects one of the three
row-major candidate points in the first demonstration frame.
"""

# ruff: noqa: E402 -- bootstrap the adjacent RoboMME uv environment before imports.

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

# RoboMME and OpenPI intentionally keep separate uv environments.  Append the
# simulator-only packages after OpenPI's own site-packages so shared packages
# (notably huggingface-hub) keep the versions required by OpenPI.
_WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_WORKSPACE / "robomme" / "src"))
sys.path.append(str(_WORKSPACE / "robomme" / ".venv" / "lib" / "python3.11" / "site-packages"))

import cv2
import jax
import numpy as np
from openpi_client import image_tools

from openpi.policies import policy_config
from openpi.training import config as training_config
from openpi.training.mem.recipes import robomme_videounmask_pi_action as action_recipe

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.mem import eval_videounmask_memory_multichoice as memory_eval
from scripts.mem import robomme_fixed_chunk_inference as fixed_memory

DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi0_robomme_videounmask_point_action_260823/"
    "oracle_point_action_phasebalanced_b12_500steps_6gpu_260823/499"
)
DEFAULT_OUTPUT = Path(
    "evaluation/robomme/videounmask_pi_action_closed_loop/"
    "oracle_point_action_step499.json"
)
DEFAULT_MEMORY_TRAINING = Path(
    "checkpoints/robomme_single_task_unmask_equal_exposure_seed260827_260827"
)
IMAGE_SIZE = 256
POLICY_IMAGE_SIZE = 224
PROMPT = "Pick up and lift the container identified by the target point."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--target-source", choices=("oracle", "fixed_chunk_memory"), default="oracle"
    )
    parser.add_argument("--memory-training-dir", type=Path, default=DEFAULT_MEMORY_TRAINING)
    parser.add_argument("--dataset", choices=("train", "val", "test"), required=True)
    parser.add_argument("--episodes", help="Comma-separated episode indices")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-rollout-steps", type=int, default=160)
    parser.add_argument("--replan-steps", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--max-position-step", type=float, default=0.025)
    parser.add_argument("--max-rotation-step", type=float, default=0.03)
    parser.add_argument(
        "--gripper-control",
        choices=("policy", "oracle_gate"),
        default="policy",
        help="Privileged oracle_gate is a diagnostic ablation, not a deployable controller.",
    )
    parser.add_argument("--oracle-close-xy-threshold", type=float, default=0.025)
    parser.add_argument("--oracle-close-z-min", type=float, default=-0.005)
    parser.add_argument("--oracle-close-z-max", type=float, default=0.04)
    parser.add_argument("--grasp-hold-steps", type=int, default=12)
    parser.add_argument("--reach-xy-threshold", type=float, default=0.04)
    parser.add_argument("--grasp-lift-threshold", type=float, default=0.01)
    parser.add_argument("--lift-threshold", type=float, default=0.05)
    parser.add_argument("--trace-stride", type=int, default=5)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--phase-conditioned", action="store_true")
    parser.add_argument(
        "--action-target-mode",
        choices=("absolute", "phase_waypoint_delta"),
        default="absolute",
    )
    parser.add_argument(
        "--target-conditioning",
        choices=("image_yx", "world_xy"),
        default="image_yx",
    )
    parser.add_argument("--goal-relative-conditioner", action="store_true")
    parser.add_argument("--phase-goal-conditioner", action="store_true")
    parser.add_argument("--phase-control", choices=("policy", "oracle"), default="policy")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _take_last(observation: dict[str, Any], key: str) -> np.ndarray:
    value = observation[key]
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty observation field {key}")
        value = value[-1]
    return memory_eval._as_numpy(value)  # noqa: SLF001


def _actor_position(actor) -> np.ndarray:
    return memory_eval._as_numpy(actor.pose.p).reshape(-1, 3)[0].astype(np.float64)  # noqa: SLF001


def _oracle_target(observation: dict[str, Any], info: dict[str, Any]) -> tuple[list[float], str]:
    frames, _ = memory_eval._reset_frames(observation)  # noqa: SLF001
    prompt = memory_eval._prompt_from_info(info)  # noqa: SLF001
    color = memory_eval._target_color(prompt)  # noqa: SLF001
    point = memory_eval._cube_centers(frames[0])[color]  # noqa: SLF001
    return point, prompt


def _raw_demo_frames(observation: dict[str, Any]) -> np.ndarray:
    values = observation.get("front_rgb_list")
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("reset observation did not contain a demonstration plus current frame")
    return np.stack([memory_eval._as_numpy(value).astype(np.uint8) for value in values[:-1]])  # noqa: SLF001


def _memory_target(observation, info, predictor, backbone) -> dict[str, Any]:
    frames = _raw_demo_frames(observation)
    prompt = memory_eval._prompt_from_info(info)  # noqa: SLF001
    color = memory_eval._target_color(prompt)  # noqa: SLF001
    centers = memory_eval._cube_centers(frames[0])  # noqa: SLF001
    # RoboMME's local regions are assigned in row-major screen order.  The
    # three candidates occupy distinct image rows, with x as a stable tie-break.
    ordered = sorted(centers.items(), key=lambda item: (item[1][0], item[1][1]))
    oracle_region = next(index for index, (name, _) in enumerate(ordered) if name == color)
    tokens = fixed_memory.encode_frames(backbone, frames, batch_size=32)
    chunks = fixed_memory.tokens_to_chunks(tokens)
    output = predictor.predict_encoded(
        chunks,
        task_id=0,
        goal_color_ids=(fixed_memory.COLOR_IDS[color], 0),
        required_count=0,
        queried_ordinal=0,
        num_regions=len(ordered),
    )
    field = fixed_memory.field_index(f"{color}_cell")
    predicted_class = int(output["all_predictions"][-1, field])
    predicted_region = predicted_class - 1
    valid = 0 <= predicted_region < len(ordered)
    if valid:
        selected_color, point = ordered[predicted_region]
    else:
        selected_color, point = "invalid", [(IMAGE_SIZE - 1) / 2.0, (IMAGE_SIZE - 1) / 2.0]
    return {
        "target_point": [float(value) for value in point],
        "task_prompt": prompt,
        "memory_valid": valid,
        "memory_exact": valid and predicted_region == oracle_region,
        "predicted_region": predicted_region,
        "oracle_region": oracle_region,
        "predicted_region_name": (
            f"region_{predicted_region}" if valid else "invalid"
        ),
        "oracle_region_name": f"region_{oracle_region}",
        "selected_candidate_color": selected_color,
        "goal_color": color,
        "demo_frames": len(frames),
        "memory_chunks": len(chunks),
        "final_write_gate": (
            float(output["write_gates"][-1]) if len(output["write_gates"]) else None
        ),
    }


def _resize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size == 0 or float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return image_tools.resize_with_pad(image, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)


def _policy_input(
    observation: dict[str, Any],
    target_point_yx: list[float],
    target_world_xy: np.ndarray,
    *,
    episode: int,
    rollout_step: int,
    episode_t: int,
    phase: int = 0,
    phase_conditioned: bool = False,
    target_conditioning: str = "image_yx",
) -> dict[str, Any]:
    eef_state = _take_last(observation, "eef_state_list").astype(np.float32).reshape(6)
    gripper_width = np.asarray(
        [_take_last(observation, "gripper_state_list").astype(np.float32).sum()],
        dtype=np.float32,
    )
    if target_conditioning == "image_yx":
        target_condition = np.asarray(target_point_yx, dtype=np.float32) / (IMAGE_SIZE - 1)
    elif target_conditioning == "world_xy":
        target_condition = np.asarray(target_world_xy, dtype=np.float32).reshape(2)
    else:
        raise ValueError(f"Unknown target conditioning: {target_conditioning}")
    state = np.concatenate((eef_state, gripper_width, target_condition))
    if phase_conditioned:
        state = np.concatenate((state, np.eye(4, dtype=np.float32)[phase]))
    expected_state = 13 if phase_conditioned else 9
    if state.shape != (expected_state,) or not np.all(np.isfinite(state)):
        raise ValueError(f"Invalid policy state: shape={state.shape}, value={state}")
    return {
        "state_raw": state,
        "front_rgb_0": _resize(_take_last(observation, "front_rgb_list")),
        "wrist_rgb_0": _resize(_take_last(observation, "wrist_rgb_list")),
        "video_frame_valid_mask": {
            "front_rgb": np.ones(1, dtype=np.bool_),
            "wrist_rgb": np.ones(1, dtype=np.bool_),
        },
        "prompt": PROMPT,
        "episode_index": np.int64(episode),
        "frame_index": np.int64(rollout_step),
        "episode_T": np.float32(episode_t),
    }


def _safe_action(
    predicted: np.ndarray,
    observation: dict[str, Any],
    max_position_step: float,
    *,
    target_mode: str = "absolute",
    max_rotation_step: float = 0.03,
) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(predicted)):
        raise ValueError(f"Non-finite policy action: {predicted}")
    current = _take_last(observation, "eef_state_list").astype(np.float64).reshape(6)
    action = predicted.copy()
    if target_mode == "phase_waypoint_delta":
        action[:6] = current + action[:6]
        action[3:6] = (action[3:6] + np.pi) % (2.0 * np.pi) - np.pi
    position_delta = action[:3] - current[:3]
    position_norm = float(np.linalg.norm(position_delta))
    if position_norm > max_position_step:
        position_delta *= max_position_step / max(position_norm, 1e-8)
    action[:3] = current[:3] + position_delta
    action[:3] = np.clip(action[:3], [-0.45, -0.45, 0.045], [0.45, 0.45, 0.65])
    rotation_delta = (action[3:6] - current[3:6] + np.pi) % (2.0 * np.pi) - np.pi
    action[3:6] = current[3:6] + np.clip(
        rotation_delta, -max_rotation_step, max_rotation_step
    )
    action[6] = -1.0 if action[6] < 0.0 else 1.0
    return action


def _video_frame(
    observation: dict[str, Any],
    target_point_yx: list[float],
    *,
    step: int,
    command: np.ndarray,
    target_lift: float,
    xy_distance: float,
) -> np.ndarray:
    front = np.asarray(_take_last(observation, "front_rgb_list"), dtype=np.uint8)
    wrist = np.asarray(_take_last(observation, "wrist_rgb_list"), dtype=np.uint8)
    height, width = front.shape[:2]
    wrist = cv2.resize(wrist, (width, height), interpolation=cv2.INTER_AREA)
    bar_height = 44
    canvas = np.zeros((height + bar_height, 2 * width, 3), dtype=np.uint8)
    canvas[bar_height:, :width] = front
    canvas[bar_height:, width:] = wrist
    y, x = np.rint(target_point_yx).astype(int)
    cv2.circle(canvas, (x, y + bar_height), 10, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "FRONT", (8, bar_height + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, "WRIST", (width + 8, bar_height + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    gripper = "CLOSE" if command[6] < 0 else "OPEN"
    text = (
        f"step {step:03d} | gripper {gripper} | XY {xy_distance * 1000:5.1f} mm "
        f"| lift {target_lift * 1000:5.1f} mm"
    )
    cv2.putText(canvas, text, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if not record.get("evaluator_error", False)]
    count = len(valid)

    def rate(key: str) -> float | None:
        return float(np.mean([bool(record[key]) for record in valid])) if valid else None

    summary = {
        "attempted": len(records),
        "valid": count,
        "evaluator_errors": len(records) - count,
        "environment_errors": sum(record.get("status") == "error" for record in valid),
        "reached": sum(bool(record["reached"]) for record in valid),
        "reach_rate": rate("reached"),
        "grasped": sum(bool(record["grasped"]) for record in valid),
        "grasp_rate": rate("grasped"),
        "lifted": sum(bool(record["lifted"]) for record in valid),
        "lift_rate": rate("lifted"),
        "successes": sum(bool(record["success"]) for record in valid),
        "success_rate": rate("success"),
        "mean_min_target_xy_distance_m": (
            float(np.mean([record["min_target_xy_distance_m"] for record in valid]))
            if valid
            else None
        ),
        "mean_max_target_lift_m": (
            float(np.mean([record["max_target_lift_m"] for record in valid]))
            if valid
            else None
        ),
    }
    memory_records = [record for record in valid if "memory_exact" in record]
    if memory_records:
        summary.update(
            memory_valid=sum(bool(record["memory_valid"]) for record in memory_records),
            memory_valid_rate=float(np.mean([bool(record["memory_valid"]) for record in memory_records])),
            memory_exact=sum(bool(record["memory_exact"]) for record in memory_records),
            memory_exact_rate=float(np.mean([bool(record["memory_exact"]) for record in memory_records])),
            success_given_memory_exact=(
                float(
                    np.mean(
                        [
                            bool(record["success"])
                            for record in memory_records
                            if record["memory_exact"]
                        ]
                    )
                )
                if any(record["memory_exact"] for record in memory_records)
                else None
            ),
        )
    return summary


def _write(path: Path, payload: dict[str, Any]) -> None:
    payload["summary"] = _summarize(payload["episodes"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_policy(
    checkpoint: Path,
    num_sampling_steps: int,
    *,
    phase_conditioned: bool = False,
    goal_relative_conditioner: bool = False,
    phase_goal_conditioner: bool = False,
):
    config = action_recipe.make_train_config(
        config_module=training_config,
        exp_name="closed_loop_eval_only",
        steps=500,
        batch_size=1,
        fsdp_devices=1,
        num_workers=0,
        phase_conditioned=phase_conditioned,
        target_point_relative_to_eef=goal_relative_conditioner,
        phase_goal_conditioner=phase_goal_conditioner,
    )
    config = dataclasses.replace(config, fsdp_devices=1)
    return policy_config.create_trained_policy(
        config,
        checkpoint,
        sample_kwargs={"num_steps": num_sampling_steps},
        default_prompt=PROMPT,
    )


def _run_episode(
    env,
    policy,
    episode: int,
    args: argparse.Namespace,
    target_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observation, info = env.reset()
    if target_record is None:
        target_point, task_prompt = _oracle_target(observation, info)
        target_record = {}
    else:
        target_point = target_record["target_point"]
        task_prompt = target_record["task_prompt"]
    target_actor = env.unwrapped.bin_0
    target_initial = _actor_position(target_actor)
    min_xy = float("inf")
    min_xyz = float("inf")
    max_lift = 0.0
    reached = False
    grasped = False
    lifted = False
    first_close_step = None
    oracle_gate_trigger_step = None
    oracle_gate_hold_remaining = 0
    oracle_gate_hold_pose = None
    status = "ongoing"
    trace: list[dict[str, Any]] = []
    replans = 0
    inference_ms: list[float] = []
    final_info: dict[str, Any] = {}
    video_frames: list[np.ndarray] = []
    rollout_step = 0
    phase = 0

    while rollout_step < args.max_rollout_steps and status == "ongoing":
        inputs = _policy_input(
            observation,
            target_point,
            target_initial[:2],
            episode=episode,
            rollout_step=rollout_step,
            episode_t=args.max_rollout_steps,
            phase=phase,
            phase_conditioned=args.phase_conditioned,
            target_conditioning=args.target_conditioning,
        )
        prediction = policy.infer(inputs)
        actions = np.asarray(prediction["actions"], dtype=np.float64)
        if actions.shape != (16, 7) or not np.all(np.isfinite(actions)):
            raise ValueError(f"Invalid sampled action chunk: shape={actions.shape}")
        inference_ms.append(float(prediction["policy_timing"]["infer_ms"]))
        replans += 1

        for chunk_index, predicted in enumerate(actions[: args.replan_steps]):
            command = _safe_action(
                predicted,
                observation,
                args.max_position_step,
                target_mode=args.action_target_mode,
                max_rotation_step=args.max_rotation_step,
            )
            eef_before = _take_last(observation, "eef_state_list").astype(np.float64).reshape(6)
            target_before = _actor_position(target_actor)
            xy_distance = float(np.linalg.norm(eef_before[:2] - target_before[:2]))
            xyz_distance = float(np.linalg.norm(eef_before[:3] - target_before[:3]))
            target_lift = float(target_before[2] - target_initial[2])

            if args.phase_control == "oracle":
                if phase == 0 and xy_distance <= 0.04:
                    phase = 1
                if phase == 1 and (
                    xy_distance <= args.oracle_close_xy_threshold
                    and args.oracle_close_z_min <= float(eef_before[2] - target_before[2]) <= args.oracle_close_z_max
                ):
                    phase = 2
                if phase == 2 and oracle_gate_trigger_step is not None and oracle_gate_hold_remaining <= 0:
                    phase = 3

            if args.gripper_control == "oracle_gate":
                z_delta = float(eef_before[2] - target_before[2])
                if oracle_gate_trigger_step is None:
                    ready_to_close = (
                        xy_distance <= args.oracle_close_xy_threshold
                        and args.oracle_close_z_min <= z_delta <= args.oracle_close_z_max
                    )
                    if ready_to_close:
                        oracle_gate_trigger_step = rollout_step
                        oracle_gate_hold_remaining = args.grasp_hold_steps
                        oracle_gate_hold_pose = command[:6].copy()
                        oracle_gate_hold_pose[:3] = eef_before[:3]
                        command[6] = -1.0
                    else:
                        command[6] = 1.0
                else:
                    command[6] = -1.0
                    if oracle_gate_hold_remaining > 0:
                        command[:6] = oracle_gate_hold_pose
                        oracle_gate_hold_remaining -= 1
            if command[6] < 0.0 and first_close_step is None:
                first_close_step = rollout_step
            min_xy = min(min_xy, xy_distance)
            min_xyz = min(min_xyz, xyz_distance)
            max_lift = max(max_lift, target_lift)
            reached = reached or xy_distance <= args.reach_xy_threshold
            grasped = grasped or target_lift >= args.grasp_lift_threshold
            lifted = lifted or target_lift >= args.lift_threshold

            if rollout_step % args.trace_stride == 0:
                trace.append(
                    {
                        "step": rollout_step,
                        "replan": replans - 1,
                        "chunk_index": chunk_index,
                        "eef_state": eef_before.tolist(),
                        "target_position": target_before.tolist(),
                        "target_xy_distance_m": xy_distance,
                        "target_lift_m": target_lift,
                        "oracle_gate_triggered": oracle_gate_trigger_step is not None,
                        "oracle_gate_hold_remaining": oracle_gate_hold_remaining,
                        "phase": phase,
                        "predicted_action": predicted.tolist(),
                        "executed_action": command.tolist(),
                    }
                )

            if args.video_dir is not None:
                video_frames.append(
                    _video_frame(
                        observation,
                        target_point,
                        step=rollout_step,
                        command=command,
                        target_lift=target_lift,
                        xy_distance=xy_distance,
                    )
                )

            next_observation, _, terminated, truncated, step_info = env.step(command)
            final_info = step_info
            status = str(step_info.get("status", "ongoing"))
            rollout_step += 1
            # RoboMME may return a terminal summary dict without sensor fields.
            # Keep the last complete observation for final pose diagnostics.
            if next_observation is not None and "eef_state_list" in next_observation:
                observation = next_observation
            if (
                status != "ongoing"
                or memory_eval._as_bool(terminated)  # noqa: SLF001
                or memory_eval._as_bool(truncated)  # noqa: SLF001
                or rollout_step >= args.max_rollout_steps
            ):
                break

    final_target = _actor_position(target_actor)
    final_eef = _take_last(observation, "eef_state_list").astype(np.float64).reshape(6)
    final_lift = float(final_target[2] - target_initial[2])
    max_lift = max(max_lift, final_lift)
    grasped = grasped or max_lift >= args.grasp_lift_threshold
    lifted = lifted or max_lift >= args.lift_threshold
    record = {
        "episode": episode,
        "difficulty": getattr(env.unwrapped, "difficulty", None),
        "task_prompt": task_prompt,
        "target_point_yx": target_point,
        "target_point_normalized_yx": (np.asarray(target_point) / (IMAGE_SIZE - 1)).tolist(),
        "target_initial_position": target_initial.tolist(),
        "target_final_position": final_target.tolist(),
        "final_eef_state": final_eef.tolist(),
        "status": status,
        "success": status == "success",
        "reached": reached,
        "grasped": grasped,
        "lifted": lifted,
        "min_target_xy_distance_m": min_xy,
        "min_target_xyz_distance_m": min_xyz,
        "max_target_lift_m": max_lift,
        "first_close_step": first_close_step,
        "oracle_gate_trigger_step": oracle_gate_trigger_step,
        "rollout_steps": rollout_step,
        "replans": replans,
        "mean_inference_ms": float(np.mean(inference_ms)),
        "trace": trace,
        **{key: value for key, value in target_record.items() if key not in {"target_point", "task_prompt"}},
    }
    if args.video_dir is not None:
        video_path = args.video_dir.expanduser().resolve() / f"{args.dataset}_episode_{episode:03d}.mp4"
        _write_video(video_path, video_frames, args.video_fps)
        record["video"] = str(video_path)
    if final_info.get("error_message"):
        record["environment_error"] = str(final_info["error_message"])
    return record


def main() -> None:
    args = parse_args()
    faulthandler.dump_traceback_later(120, repeat=True)
    positive_ints = (
        args.max_rollout_steps,
        args.replan_steps,
        args.num_sampling_steps,
        args.trace_stride,
        args.grasp_hold_steps + 1,
    )
    if min((*positive_ints, args.video_fps)) < 1 or args.replan_steps > 16:
        raise ValueError("Rollout, replan, sampling, and trace values must be positive; replan <= 16")
    if min(
        args.max_position_step,
        args.reach_xy_threshold,
        args.grasp_lift_threshold,
        args.lift_threshold,
        args.oracle_close_xy_threshold,
        args.oracle_close_z_max,
    ) <= 0:
        raise ValueError("Step and metric thresholds must be positive")
    if args.oracle_close_z_min >= args.oracle_close_z_max:
        raise ValueError("oracle close z min must be smaller than z max")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    os.environ.setdefault("OPENPI_DATA_HOME", "/data2/hzl_workspace_for_pi_mem/.cache/openpi")
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()

    from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(
        env_id="VideoUnmask",
        dataset=args.dataset,
        action_space="ee_pose",
        gui_render=False,
        max_steps=args.max_rollout_steps,
    )
    episodes = memory_eval._episode_indices(builder, args.episodes, args.max_episodes)  # noqa: SLF001
    memory_targets: dict[int, dict[str, Any]] = {}
    if args.target_source == "fixed_chunk_memory":
        logging.info("Precomputing target regions from %s", args.memory_training_dir)
        predictor = fixed_memory.FixedChunkMemoryPredictor(args.memory_training_dir)
        backbone = fixed_memory.load_backbone()
        try:
            for episode in episodes:
                env = builder.make_env_for_episode(episode)
                try:
                    observation, info = env.reset()
                    memory_targets[episode] = _memory_target(
                        observation, info, predictor, backbone
                    )
                    logging.info(
                        "MEM episode=%d predicted=%s oracle=%s exact=%s",
                        episode,
                        memory_targets[episode]["predicted_region_name"],
                        memory_targets[episode]["oracle_region_name"],
                        memory_targets[episode]["memory_exact"],
                    )
                finally:
                    memory_eval._close(env)  # noqa: SLF001
        finally:
            del predictor, backbone
            jax.clear_caches()
            gc.collect()
    logging.info("Loading Pi action policy from %s", checkpoint)
    policy = _load_policy(
        checkpoint,
        args.num_sampling_steps,
        phase_conditioned=args.phase_conditioned,
        goal_relative_conditioner=args.goal_relative_conditioner,
        phase_goal_conditioner=args.phase_goal_conditioner,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": f"{args.target_source} point Pi action closed loop",
        "dataset": args.dataset,
        "episode_indices": episodes,
        "checkpoint": str(checkpoint),
        "target_source": args.target_source,
        "memory_training_dir": (
            str(args.memory_training_dir.resolve())
            if args.target_source == "fixed_chunk_memory"
            else None
        ),
        "control": {
            "action_space": "absolute_eef7",
            "action_horizon": 16,
            "replan_steps": args.replan_steps,
            "num_sampling_steps": args.num_sampling_steps,
            "max_position_step": args.max_position_step,
            "max_rotation_step": args.max_rotation_step,
            "gripper": "zero-thresholded to +/-1",
            "gripper_control": args.gripper_control,
            "phase_conditioned": args.phase_conditioned,
            "phase_control": args.phase_control,
            "action_target_mode": args.action_target_mode,
            "target_conditioning": args.target_conditioning,
            "goal_relative_conditioner": args.goal_relative_conditioner,
            "phase_goal_conditioner": args.phase_goal_conditioner,
            "oracle_close_xy_threshold": args.oracle_close_xy_threshold,
            "oracle_close_z_range": [args.oracle_close_z_min, args.oracle_close_z_max],
            "grasp_hold_steps": args.grasp_hold_steps,
        },
        "thresholds": {
            "reach_xy_m": args.reach_xy_threshold,
            "grasp_lift_m": args.grasp_lift_threshold,
            "lift_m": args.lift_threshold,
        },
        "episodes": [],
    }
    started = time.monotonic()
    try:
        for ordinal, episode in enumerate(episodes, start=1):
            env = None
            try:
                env = builder.make_env_for_episode(episode)
                record = _run_episode(
                    env,
                    policy,
                    episode,
                    args,
                    memory_targets.get(episode),
                )
            except Exception as exc:
                logging.exception("Episode %d failed", episode)
                record = {
                    "episode": episode,
                    "difficulty": builder.metadata_index[("VideoUnmask", episode)].get("difficulty"),
                    "status": "error",
                    "success": False,
                    "evaluator_error": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                memory_eval._close(env)  # noqa: SLF001
            payload["episodes"].append(record)
            _write(output, payload)
            print(
                f"[{ordinal}/{len(episodes)}] episode={episode} status={record['status']} "
                f"reach={record.get('reached')} grasp={record.get('grasped')} "
                f"lift={record.get('lifted')} elapsed={(time.monotonic() - started) / 60:.1f}m",
                flush=True,
            )
    finally:
        del policy
        jax.clear_caches()
        gc.collect()

    _write(output, payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
