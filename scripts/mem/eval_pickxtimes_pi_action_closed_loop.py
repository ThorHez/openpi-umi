#!/usr/bin/env python3
"""Closed-loop PickXtimes smoke test for the Pi0.5 frozen-MEM action expert."""

# ruff: noqa: E402 -- bootstrap the adjacent RoboMME environment before imports.

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import gc
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENPI_DATA_HOME", "/data2/hzl_workspace_for_pi_mem/.cache/openpi")
os.environ.setdefault("MPLCONFIGDIR", "/data2/hzl_workspace_for_pi_mem/.cache/matplotlib")

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / "robomme" / "src"))
sys.path.append(str(WORKSPACE / "robomme" / ".venv" / "lib" / "python3.11" / "site-packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
import cv2
from openpi_client import image_tools

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies import policy as policy_lib
from openpi.tasks.robomme.pickxtimes import semantic_memory_event
from openpi.training import checkpoints
from openpi.training.mem.recipes import robomme_pickxtimes_pi_action as action_recipe

from scripts.mem import cache_pickxtimes_siglip_features as feature_cache
from scripts.mem import eval_pickxtimes_memory_action_chunk_closed_loop as legacy_rollout


DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi0_robomme_pickxtimes_predicted_memory_action_260824/"
    "predicted_mem_gate01_drop25_phasebalanced_b12_300steps_6gpu_260824/2999"
)
DEFAULT_ACTION_ONLY_CHECKPOINT = Path(
    "checkpoints/pi0_robomme_pickxtimes_action_only_memory_action_260824/"
    "action_only_nullmem_phasebalanced_b12_300steps_6gpu_260824/299"
)
DEFAULT_SPLIT = Path(
    "data/robomme_extracted/pickxtimes_split_seed260827_train70_dev15_test15.json"
)
DEFAULT_OUTPUT = Path(
    "evaluation/robomme/pickxtimes_pi_action_round16/"
    "predicted_mem_gate01_drop25_step2999_smoke10.json"
)
COLOR_TO_ID = {"red": 0, "green": 1, "blue": 2}
POLICY_IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--control", choices=("predicted", "action_only"), default="predicted")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--episodes", help="Comma-separated train-domain episode indices")
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--max-rollout-steps", type=int, default=500)
    parser.add_argument("--replan-steps", type=int, default=2)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--max-position-step", type=float, default=0.02)
    parser.add_argument(
        "--min-eef-z",
        type=float,
        default=0.01,
        help="Minimum commanded EEF z in metres. PickXtimes demonstrations grasp below 4.5 cm.",
    )
    parser.add_argument(
        "--gripper-control",
        choices=("policy", "oracle_pick"),
        default="policy",
        help="Keep policy pose commands but optionally gate the first close with simulator geometry.",
    )
    parser.add_argument("--oracle-close-xy-threshold", type=float, default=0.015)
    parser.add_argument("--oracle-close-z-min", type=float, default=-0.01)
    parser.add_argument("--oracle-close-z-max", type=float, default=0.01)
    parser.add_argument("--trace-stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=260824)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _as_bool(value) -> bool:
    return bool(_as_numpy(value).reshape(-1)[0])


def _latest(observation: dict[str, Any], key: str) -> np.ndarray:
    value = observation[key]
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty observation field {key}")
        value = value[-1]
    return _as_numpy(value)


def _goal(prompt: str) -> tuple[str, int]:
    color_match = re.search(r"\b(red|green|blue)\b", prompt.lower())
    count_match = re.search(r"\b(one|two|three|four|five|[1-5])\s+times?\b", prompt.lower())
    if color_match is None:
        raise ValueError(f"Could not parse PickXtimes goal: {prompt!r}")
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    # RoboMME omits the repetition clause for the one-cycle form: "place it
    # on the target, then press the button".  That prompt unambiguously means
    # one pick/place cycle.
    if count_match is None:
        return color_match.group(1), 1
    raw = count_match.group(1)
    return color_match.group(1), words.get(raw, int(raw) if raw.isdigit() else 0)


def _resize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size == 0 or float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return image_tools.resize_with_pad(image, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)


@dataclasses.dataclass(frozen=True)
class OnlinePickXtimesInputs(transforms.DataTransformFn):
    """Build the exact training-time model fields using a live MEM bank."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(data["state_raw"], dtype=np.float32).reshape(-1)
        memory = np.asarray(data["semantic_memory_raw"], dtype=np.float32)
        if state.shape != (11,):
            raise ValueError(f"Expected state11, got {state.shape}")
        if memory.shape != (128, 64):
            raise ValueError(f"Expected semantic memory [128,64], got {memory.shape}")
        return {
            "state": state,
            "semantic_memory": memory,
            "image": {
                "base_rgb": _resize(data["front_rgb"])[None],
                "wrist_rgb": _resize(data["wrist_rgb"])[None],
            },
            "image_mask": {"base_rgb": np.True_, "wrist_rgb": np.True_},
            "prompt": str(data["prompt"]),
            "episode_index": np.int32(data["episode_index"]),
            "frame_index": np.int32(data["frame_index"]),
            "episode_T": np.float32(data["episode_T"]),
        }


def _load_policy(checkpoint: Path, num_sampling_steps: int, control: str) -> policy_lib.Policy:
    model_config = action_recipe.make_model_config(
        use_learned_null_memory=control == "action_only",
        semantic_residual_gate_init=0.1 if control == "predicted" else 1.0,
        semantic_residual_dropout_rate=0.25 if control == "predicted" else 0.0,
    )
    model = model_config.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    model.eval()
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", ".")
    normalize_masks = {"actions": transforms.make_bool_mask(7), "state": transforms.make_bool_mask(11)}
    return policy_lib.Policy(
        model,
        transforms=[
            OnlinePickXtimesInputs(),
            transforms.Normalize(norm_stats, key_masks=normalize_masks),
            transforms.TokenizePrompt(
                tokenizer_lib.PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=True,
                robot_type="ARM=1 G=0 H=0",
            ),
            transforms.PadStatesAndActions(model_config.action_dim),
            transforms.KeepModelKeys(),
        ],
        output_transforms=[
            transforms.ChunkActions(target_dim=7),
            transforms.DropKeys(keys=("state",)),
            transforms.Unnormalize(norm_stats, key_masks=normalize_masks),
        ],
        sample_kwargs={"num_steps": num_sampling_steps},
    )


def _state(observation: dict[str, Any], color: str, count: int) -> np.ndarray:
    eef = _latest(observation, "eef_state_list").astype(np.float32).reshape(6)
    width = np.asarray([_latest(observation, "gripper_state_list").astype(np.float32).sum()])
    color_onehot = np.eye(3, dtype=np.float32)[COLOR_TO_ID[color]]
    return np.concatenate((eef, width, color_onehot, np.asarray([count / 5], dtype=np.float32)))


def _safe_action(
    predicted: np.ndarray,
    observation: dict[str, Any],
    max_position_step: float,
    min_eef_z: float,
) -> np.ndarray:
    action = np.asarray(predicted, dtype=np.float64).reshape(7).copy()
    if not np.all(np.isfinite(action)):
        raise ValueError(f"Non-finite policy action: {action}")
    current = _latest(observation, "eef_state_list").astype(np.float64).reshape(6)
    delta = action[:3] - current[:3]
    norm = float(np.linalg.norm(delta))
    if norm > max_position_step:
        delta *= max_position_step / norm
    action[:3] = current[:3] + delta
    action[:3] = np.clip(action[:3], [-0.45, -0.45, min_eef_z], [0.45, 0.45, 0.65])
    rotation_delta = (action[3:6] - current[3:6] + np.pi) % (2 * np.pi) - np.pi
    action[3:6] = current[3:6] + rotation_delta
    action[6] = -1.0 if action[6] < 0 else 1.0
    return action


def _actor_position(actor: Any) -> np.ndarray:
    """Read a SAPIEN actor position without depending on a specific SAPIEN version."""
    pose = getattr(actor, "pose", None)
    if pose is None and hasattr(actor, "get_pose"):
        pose = actor.get_pose()
    position = getattr(pose, "p", None)
    if position is None:
        raise ValueError("Target actor does not expose pose.p")
    return _as_numpy(position).astype(np.float64).reshape(3)


def _video_frame(
    observation: dict[str, Any],
    *,
    step: int,
    command: np.ndarray,
    oracle_subgoal: str,
    accepted_types: list[int],
) -> np.ndarray:
    front = np.asarray(_latest(observation, "front_rgb_list"), dtype=np.uint8)
    wrist = np.asarray(_latest(observation, "wrist_rgb_list"), dtype=np.uint8)
    height, width = front.shape[:2]
    wrist = cv2.resize(wrist, (width, height), interpolation=cv2.INTER_AREA)
    bar_height = 66
    canvas = np.zeros((height + bar_height, 2 * width, 3), dtype=np.uint8)
    canvas[bar_height:, :width] = front
    canvas[bar_height:, width:] = wrist
    cv2.putText(
        canvas,
        f"step {step:03d} | gripper {'CLOSE' if command[6] < 0 else 'OPEN'} | MEM {accepted_types}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        oracle_subgoal[:90],
        (8, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 220, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "FRONT", (8, bar_height + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(
        canvas,
        "WRIST",
        (width + 8, bar_height + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    return canvas


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _episode_indices(split: Path, episodes: str | None, max_episodes: int) -> list[int]:
    if episodes:
        selected = [int(value.strip()) for value in episodes.split(",") if value.strip()]
    else:
        selected = [int(value) for value in json.loads(split.read_text())["val_episode_indices"]]
    if max_episodes > 0:
        selected = selected[:max_episodes]
    if not selected:
        raise ValueError("No smoke-test episodes selected")
    return selected


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if not record.get("evaluator_error", False)]
    total = len(valid)

    def rate(key: str) -> float | None:
        return float(np.mean([bool(record[key]) for record in valid])) if valid else None

    return {
        "attempted": len(records),
        "valid": total,
        "evaluator_errors": len(records) - total,
        "successes": sum(bool(record["success"]) for record in valid),
        "success_rate": rate("success"),
        "first_pick_completed": sum(bool(record["first_pick_completed"]) for record in valid),
        "first_pick_rate": rate("first_pick_completed"),
        "one_cycle_completed": sum(bool(record["one_cycle_completed"]) for record in valid),
        "one_cycle_rate": rate("one_cycle_completed"),
        "mean_cycles_completed": (
            float(np.mean([record["cycles_completed"] for record in valid])) if valid else None
        ),
        "mean_rollout_steps": (
            float(np.mean([record["rollout_steps"] for record in valid])) if valid else None
        ),
        "environment_errors": sum(record.get("status") == "error" for record in valid),
        "oracle_close_triggers": sum(
            bool(record.get("oracle_close_triggered", False)) for record in valid
        ),
    }


def _write(output: Path, payload: dict[str, Any]) -> None:
    payload["summary"] = _summary(payload["episodes"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_episode(
    env,
    policy: policy_lib.Policy,
    memory_components,
    episode: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    observation, info = env.reset()
    prompt = str(info["task_goal"][0])
    target_color, required_count = _goal(prompt)
    patch_history: list[np.ndarray] = []
    gripper_history: list[bool] = []
    event_logits: list[float] = []
    type_logits: list[np.ndarray] = []
    press_logits: list[float] = []
    accepted_starts: list[int] = []
    accepted_types: list[int] = []
    if memory_components is None:
        classify = apply_tracker = transition_config = backbone = prompt_tokenizer = None
        prompt_mask = prompt_tokens = None
        memory = np.zeros((128, 64), dtype=np.float32)
        observe = None
    else:
        classify, apply_tracker, transition_config, backbone, prompt_tokenizer = memory_components
        token_ids, prompt_mask = prompt_tokenizer.tokenize(prompt)
        prompt_tokens = np.asarray(
            feature_cache.embed_prompts(backbone, jnp.asarray(token_ids, dtype=jnp.int32)[None])
        )[0]

        def observe(current_observation) -> None:
            image = _latest(current_observation, "front_rgb_list").astype(np.uint8)
            patch = np.asarray(feature_cache.encode_images(backbone, jnp.asarray(image)[None]))[0]
            patch_history.append(patch.astype(np.float16))
            gripper_sum = float(np.sum(_latest(current_observation, "gripper_state_list")))
            gripper_history.append(gripper_sum < 0.058)
            if len(patch_history) >= semantic_memory_event.WINDOW_SIZE:
                window = np.asarray(patch_history[-semantic_memory_event.WINDOW_SIZE :])[None]
                gripper_window = np.asarray(
                    gripper_history[-semantic_memory_event.WINDOW_SIZE :], dtype=np.bool_
                )[None]
                event, event_type, press = jax.device_get(
                    classify(jnp.asarray(window), jnp.asarray(gripper_window))
                )
                event_logits.append(float(event[0]))
                type_logits.append(np.asarray(event_type[0]))
                press_logits.append(float(press[0]))

        observe(observation)
        memory = legacy_rollout._memory_from_events(  # noqa: SLF001
            apply_tracker,
            patch_history,
            gripper_history,
            prompt_tokens,
            np.asarray(prompt_mask),
            accepted_starts,
            accepted_types,
        )
    rollout_step = 0
    status = "ongoing"
    max_task_index = max(int(getattr(env.unwrapped, "current_task_index", 0)), 0)
    trace: list[dict[str, Any]] = []
    event_updates: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    final_info = info
    video_frames: list[np.ndarray] = []
    target_actor = getattr(env.unwrapped, "target_cube", None)
    if target_actor is None:
        raise ValueError("PickXtimes environment does not expose target_cube")
    target_initial = _actor_position(target_actor)
    oracle_close_latched = False
    oracle_trigger_step: int | None = None
    oracle_trigger_xy_m: float | None = None
    oracle_trigger_z_delta_m: float | None = None
    min_target_xy_m = float("inf")
    min_target_xyz_m = float("inf")
    min_abs_target_z_delta_m = float("inf")
    max_target_lift_m = 0.0

    while rollout_step < args.max_rollout_steps and status == "ongoing":
        if memory_components is not None and event_logits:
            starts, types = semantic_memory_event.transition_grammar_events(
                np.asarray(gripper_history, dtype=np.bool_),
                np.asarray(event_logits, dtype=np.float32),
                np.asarray(type_logits, dtype=np.float32),
                np.asarray(press_logits, dtype=np.float32),
                required_count=required_count,
                **transition_config,
            )
            if starts != accepted_starts or types != accepted_types:
                accepted_starts, accepted_types = starts, types
                memory = legacy_rollout._memory_from_events(  # noqa: SLF001
                    apply_tracker,
                    patch_history,
                    gripper_history,
                    prompt_tokens,
                    np.asarray(prompt_mask),
                    accepted_starts,
                    accepted_types,
                )
                event_updates.append(
                    {"rollout_step": rollout_step, "starts": starts, "types": types}
                )

        inputs = {
            "state_raw": _state(observation, target_color, required_count),
            "semantic_memory_raw": memory,
            "front_rgb": _latest(observation, "front_rgb_list"),
            "wrist_rgb": _latest(observation, "wrist_rgb_list"),
            "prompt": prompt,
            "episode_index": episode,
            "frame_index": rollout_step,
            "episode_T": args.max_rollout_steps,
        }
        prediction = policy.infer(inputs)
        actions = np.asarray(prediction["actions"], dtype=np.float64)
        if actions.shape != (16, 7) or not np.all(np.isfinite(actions)):
            raise ValueError(f"Invalid action chunk: {actions.shape}")
        inference_ms.append(float(prediction["policy_timing"]["infer_ms"]))

        for chunk_index, predicted in enumerate(actions[: args.replan_steps]):
            eef_before = _latest(observation, "eef_state_list").astype(np.float64).reshape(6)
            target_before = _actor_position(target_actor)
            target_delta_before = eef_before[:3] - target_before
            xy_before = float(np.linalg.norm(target_delta_before[:2]))
            xyz_before = float(np.linalg.norm(target_delta_before))
            z_delta_before = float(target_delta_before[2])
            min_target_xy_m = min(min_target_xy_m, xy_before)
            min_target_xyz_m = min(min_target_xyz_m, xyz_before)
            min_abs_target_z_delta_m = min(min_abs_target_z_delta_m, abs(z_delta_before))
            command = _safe_action(
                predicted,
                observation,
                args.max_position_step,
                args.min_eef_z,
            )
            if args.gripper_control == "oracle_pick":
                oracle_ready = (
                    xy_before <= args.oracle_close_xy_threshold
                    and args.oracle_close_z_min <= z_delta_before <= args.oracle_close_z_max
                )
                if oracle_ready and not oracle_close_latched:
                    oracle_close_latched = True
                    oracle_trigger_step = rollout_step
                    oracle_trigger_xy_m = xy_before
                    oracle_trigger_z_delta_m = z_delta_before
                # The oracle changes only the gripper. The complete 6-DoF pose
                # trajectory remains the Pi policy's safely rate-limited output.
                command[6] = -1.0 if oracle_close_latched else 1.0
            next_observation, _, terminated, truncated, step_info = env.step(command)
            rollout_step += 1
            final_info = step_info
            status = str(step_info.get("status", "ongoing"))
            if next_observation is not None and "front_rgb_list" in next_observation:
                observation = next_observation
                if observe is not None:
                    observe(observation)
            task_index = max(int(getattr(env.unwrapped, "current_task_index", 0)), 0)
            max_task_index = max(max_task_index, task_index)
            eef_after = _latest(observation, "eef_state_list").astype(np.float64).reshape(6)
            target_after = _actor_position(target_actor)
            target_delta_after = eef_after[:3] - target_after
            min_target_xy_m = min(min_target_xy_m, float(np.linalg.norm(target_delta_after[:2])))
            min_target_xyz_m = min(min_target_xyz_m, float(np.linalg.norm(target_delta_after)))
            min_abs_target_z_delta_m = min(
                min_abs_target_z_delta_m, abs(float(target_delta_after[2]))
            )
            max_target_lift_m = max(max_target_lift_m, float(target_after[2] - target_initial[2]))
            terminated_flag, truncated_flag = _as_bool(terminated), _as_bool(truncated)
            if status == "ongoing" and truncated_flag:
                status = "timeout"
            elif status == "ongoing" and terminated_flag:
                status = "terminated"
            if rollout_step % args.trace_stride == 0 or status != "ongoing":
                trace.append(
                    {
                        "step": rollout_step,
                        "status": status,
                        "chunk_index": chunk_index,
                        "eef": _latest(observation, "eef_state_list").astype(float).tolist(),
                        "command": command.tolist(),
                        "task_index": task_index,
                        "target_position": target_after.tolist(),
                        "target_xy_m": float(np.linalg.norm(target_delta_after[:2])),
                        "target_z_delta_m": float(target_delta_after[2]),
                        "oracle_close_latched": oracle_close_latched,
                        "oracle_subgoal": str(step_info.get("simple_subgoal_online", "")),
                        "accepted_event_types": list(accepted_types),
                    }
                )
            if args.video_dir is not None and rollout_step % args.video_stride == 0:
                video_frames.append(
                    _video_frame(
                        observation,
                        step=rollout_step,
                        command=command,
                        oracle_subgoal=str(step_info.get("simple_subgoal_online", "")),
                        accepted_types=accepted_types,
                    )
                )
            if args.gripper_control == "oracle_pick" and max_task_index >= 1:
                status = "diagnostic_first_pick"
            if status != "ongoing" or terminated_flag or truncated_flag:
                break

    cycles_completed = min(max_task_index // 2, required_count)
    record = {
        "episode": episode,
        "difficulty": getattr(env.unwrapped, "difficulty", None),
        "prompt": prompt,
        "target_color": target_color,
        "required_count": required_count,
        "status": status,
        "success": status == "success",
        "rollout_steps": rollout_step,
        "gripper_control": args.gripper_control,
        "oracle_close_triggered": oracle_close_latched,
        "oracle_trigger_step": oracle_trigger_step,
        "oracle_trigger_xy_m": oracle_trigger_xy_m,
        "oracle_trigger_z_delta_m": oracle_trigger_z_delta_m,
        "min_target_xy_m": min_target_xy_m,
        "min_target_xyz_m": min_target_xyz_m,
        "min_abs_target_z_delta_m": min_abs_target_z_delta_m,
        "max_target_lift_m": max_target_lift_m,
        "max_task_index": max_task_index,
        "first_pick_completed": max_task_index >= 1,
        "one_cycle_completed": max_task_index >= 2,
        "cycles_completed": cycles_completed,
        "accepted_event_starts": accepted_starts,
        "accepted_event_types": accepted_types,
        "event_updates": event_updates,
        "final_oracle_subgoal": str(final_info.get("simple_subgoal_online", "")),
        "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None,
        "trace": trace,
    }
    if final_info.get("error_message"):
        record["environment_error"] = str(final_info["error_message"])
    if args.video_dir is not None:
        video_path = args.video_dir.expanduser().resolve() / f"episode_{episode:03d}.mp4"
        _write_video(video_path, video_frames, args.video_fps)
        record["video"] = str(video_path)
    return record


def main() -> None:
    args = parse_args()
    faulthandler.dump_traceback_later(120, repeat=True)
    if min(
        args.max_episodes,
        args.max_rollout_steps,
        args.replan_steps,
        args.num_sampling_steps,
        args.trace_stride,
        args.video_fps,
        args.video_stride,
    ) < 1:
        raise ValueError("Episode, rollout, replanning, sampling, and trace counts must be positive")
    if args.replan_steps > 16 or args.max_position_step <= 0:
        raise ValueError("replan_steps must be <=16 and max_position_step must be positive")
    if not (-0.05 <= args.min_eef_z < 0.65):
        raise ValueError("min_eef_z must be in [-0.05, 0.65)")
    if args.oracle_close_xy_threshold <= 0:
        raise ValueError("oracle_close_xy_threshold must be positive")
    if args.oracle_close_z_min > args.oracle_close_z_max:
        raise ValueError("oracle_close_z_min must not exceed oracle_close_z_max")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    default_checkpoint = DEFAULT_CHECKPOINT if args.control == "predicted" else DEFAULT_ACTION_ONLY_CHECKPOINT
    checkpoint = (args.checkpoint or default_checkpoint).expanduser().resolve()
    split = args.split.expanduser().resolve()
    output = args.output.expanduser().resolve()
    episodes = _episode_indices(split, args.episodes, args.max_episodes)

    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(
        env_id="PickXtimes",
        dataset="train",
        action_space="ee_pose",
        gui_render=False,
        max_steps=args.max_rollout_steps,
    )
    logging.info("Loading Pi0.5 policy from %s", checkpoint)
    policy = _load_policy(checkpoint, args.num_sampling_steps, args.control)
    backbone = None
    if args.control == "predicted":
        logging.info("Loading frozen online event memory")
        classify, apply_tracker, transition_config = legacy_rollout._load_online_memory(  # noqa: SLF001
            legacy_rollout.DEFAULT_MEMORY_RUN,
            legacy_rollout.DEFAULT_MEMORY_CHECKPOINT,
            legacy_rollout.DEFAULT_MEMORY_EVAL,
        )
        backbone = feature_cache.load_backbone(legacy_rollout.DEFAULT_BACKBONE)
        prompt_tokenizer = tokenizer_lib.PaligemmaTokenizer(256)
        memory_components = classify, apply_tracker, transition_config, backbone, prompt_tokenizer
    else:
        memory_components = None

    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": f"PickXtimes Pi0.5 {args.control} closed-loop smoke test",
        "dataset": "train",
        "episode_source": "fixed dev15 episode indices",
        "episode_indices": episodes,
        "frozen_test_accessed": False,
        "checkpoint": str(checkpoint),
        "memory_checkpoint": (
            str(legacy_rollout.DEFAULT_MEMORY_CHECKPOINT.resolve())
            if args.control == "predicted"
            else None
        ),
        "control": {
            "action_space": "absolute_eef7",
            "action_horizon": 16,
            "replan_steps": args.replan_steps,
            "num_sampling_steps": args.num_sampling_steps,
            "max_position_step": args.max_position_step,
            "min_eef_z": args.min_eef_z,
            "gripper_control": args.gripper_control,
            "oracle_close_gate": (
                {
                    "xy_threshold_m": args.oracle_close_xy_threshold,
                    "z_delta_min_m": args.oracle_close_z_min,
                    "z_delta_max_m": args.oracle_close_z_max,
                    "target": "simulator target_cube",
                    "pose_source": "Pi policy (oracle overrides gripper only)",
                    "stop_condition": "environment first-pick subgoal completed",
                }
                if args.gripper_control == "oracle_pick"
                else None
            ),
            "memory": (
                "online sliding-window event detector and recurrent tracker"
                if args.control == "predicted"
                else "shared learned null bank; no temporal information"
            ),
        },
        "episodes": [],
    }
    if args.resume and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("checkpoint") != str(checkpoint):
            raise ValueError("Cannot resume: checkpoint differs from existing output")
        payload["episodes"] = [
            record
            for record in previous.get("episodes", [])
            if int(record.get("episode", -1)) in episodes
            and not record.get("evaluator_error", False)
        ]
    completed = {int(record["episode"]) for record in payload["episodes"]}
    pending = [episode for episode in episodes if episode not in completed]
    logging.info("Smoke-test progress: completed=%d pending=%d", len(completed), len(pending))
    started = time.monotonic()
    try:
        for pending_ordinal, episode in enumerate(pending, start=1):
            env = None
            try:
                env = builder.make_env_for_episode(episode)
                record = _run_episode(env, policy, memory_components, episode, args)
            except Exception as exc:
                logging.exception("Episode %d failed", episode)
                record = {
                    "episode": episode,
                    "status": "error",
                    "success": False,
                    "first_pick_completed": False,
                    "one_cycle_completed": False,
                    "cycles_completed": 0,
                    "rollout_steps": 0,
                    "evaluator_error": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                if env is not None:
                    env.close()
            payload["episodes"].append(record)
            payload["episodes"].sort(key=lambda item: episodes.index(int(item["episode"])))
            _write(output, payload)
            ordinal = len(completed) + pending_ordinal
            print(
                f"[{ordinal}/{len(episodes)}] episode={episode} status={record['status']} "
                f"pick={record.get('first_pick_completed')} cycles={record.get('cycles_completed')} "
                f"events={len(record.get('accepted_event_types', []))} "
                f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                flush=True,
            )
    finally:
        del policy
        if backbone is not None:
            del backbone
        jax.clear_caches()
        gc.collect()

    _write(output, payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
