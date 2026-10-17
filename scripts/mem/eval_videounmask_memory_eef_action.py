#!/usr/bin/env python3
"""Paired VideoUnmask closed loop with a learned EEF action adapter."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

# RoboMME keeps its simulator dependencies in a separate environment.  Append
# them after OpenPI's own site-packages so shared ML dependencies keep the
# versions expected by OpenPI.
_WORKSPACE = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_WORKSPACE / "robomme" / "src"))
sys.path.append(str(_WORKSPACE / "robomme" / ".venv" / "lib" / "python3.11" / "site-packages"))

import flax
import jax
import jax.numpy as jnp
import numpy as np

from openpi.tasks.robomme.videounmask import eef_action_adapter
from openpi.training.mem import robomme_videounmask_action_dataset as action_data

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.mem import eval_videounmask_memory_multichoice as memory_eval

DEFAULT_ACTION_RUN_DIR = pathlib.Path(
    "evaluation/robomme/videounmask_memory_action_adapter_v1_260823/action_target_state_fixed_crop_split"
)
CONTROLS = ("oracle_target", "predicted_target")
VALID_CONTROLS = ("oracle_geometry", *CONTROLS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-run-dir", type=pathlib.Path, default=DEFAULT_ACTION_RUN_DIR)
    parser.add_argument("--action-checkpoint", type=pathlib.Path)
    parser.add_argument("--memory-run-dir", type=pathlib.Path, default=memory_eval.DEFAULT_RUN_DIR)
    parser.add_argument("--memory-checkpoint", type=pathlib.Path)
    parser.add_argument(
        "--backbone-checkpoint",
        type=pathlib.Path,
        default=memory_eval.DEFAULT_BACKBONE_CHECKPOINT,
    )
    parser.add_argument("--dataset", choices=("train", "val", "test"), required=True)
    parser.add_argument("--controls", default=",".join(CONTROLS))
    parser.add_argument("--episodes")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-rollout-steps", type=int, default=140)
    parser.add_argument("--max-position-step", type=float, default=0.02)
    parser.add_argument("--max-rotation-step", type=float, default=0.03)
    parser.add_argument("--grasp-hold-steps", type=int, default=12)
    parser.add_argument(
        "--phase-control",
        choices=("policy", "oracle"),
        default="policy",
        help="oracle is a diagnostic monotonic phase scheduler using simulator geometry.",
    )
    parser.add_argument("--trace-stride", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _parse_controls(value: str) -> list[str]:
    controls = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    invalid = sorted(set(controls) - set(VALID_CONTROLS))
    if invalid or not controls:
        raise ValueError(f"Expected a non-empty subset of {VALID_CONTROLS}; invalid={invalid}")
    return controls


def _load_action_adapter(run_dir: pathlib.Path, checkpoint: pathlib.Path | None):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = checkpoint or run_dir / "checkpoints" / f"step_{int(config['steps'])}.msgpack"
    stats = action_data.ActionNormalization.from_json(config["normalization"])
    model = eef_action_adapter.VideoUnmaskEEFActionAdapter(
        hidden_width=int(config["hidden_width"]),
        depth=int(config["depth"]),
        feature_dim=int(config.get("feature_dim", eef_action_adapter.ACTION_FEATURE_DIM)),
    )
    features = jnp.zeros((1, int(config.get("feature_dim", eef_action_adapter.ACTION_FEATURE_DIM))), dtype=jnp.float32)
    crops = jnp.zeros((1, 32, 32, 3), dtype=jnp.uint8)
    variables = model.init(jax.random.key(0), features, crops, train=False)
    params = flax.serialization.from_bytes(variables["params"], checkpoint.read_bytes())

    @jax.jit
    def predict(normalized_features, target_crops):
        return model.apply({"params": params}, normalized_features, target_crops, train=False)

    return config, checkpoint, stats, predict


def _take_last(observation: dict[str, Any], key: str) -> np.ndarray:
    value = observation[key]
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty observation field {key}")
        value = value[-1]
    return memory_eval._as_numpy(value)  # noqa: SLF001


def _oracle_geometry_target(observation: dict[str, Any], info: dict[str, Any]) -> tuple[list[float], str]:
    frames, _ = memory_eval._reset_frames(observation)  # noqa: SLF001
    prompt = memory_eval._prompt_from_info(info)  # noqa: SLF001
    color = memory_eval._target_color(prompt)  # noqa: SLF001
    point = memory_eval._cube_centers(frames[0])[color]  # noqa: SLF001
    return [float(value) for value in point], prompt


def _action(
    predict,
    stats: action_data.ActionNormalization,
    target_point_yx: list[float],
    observation: dict[str, Any],
    target_crop: np.ndarray,
    rollout_step: int,
    *,
    progress_steps: int,
    max_position_step: float,
    max_rotation_step: float,
    target_mode: str,
    phase: int,
) -> tuple[np.ndarray, dict[str, float]]:
    eef_state = _take_last(observation, "eef_state_list").astype(np.float32)
    feature = action_data.build_action_feature(
        np.asarray(target_point_yx),
        eef_state,
        _take_last(observation, "gripper_state_list"),
        _take_last(observation, "joint_state_list"),
        rollout_step,
        progress_steps=progress_steps,
    )
    if len(stats.feature_mean) == eef_action_adapter.ACTION_FEATURE_DIM + len(action_data.PHASES):
        feature = np.concatenate(
            (feature, np.eye(len(action_data.PHASES), dtype=np.float32)[phase])
        )
    normalized = (feature - stats.feature_mean) / stats.feature_std
    outputs = predict(jnp.asarray(normalized[None]), jnp.asarray(target_crop[None]))
    pose = np.asarray(outputs["normalized_pose"])[0] * stats.pose_std + stats.pose_mean
    if target_mode in {"delta", "phase_waypoint_delta"}:
        pose = pose.copy()
        pose[:3] = eef_state[:3] + pose[:3]
        pose[3:] = eef_state[3:] + pose[3:]
        pose[3:] = (pose[3:] + np.pi) % (2.0 * np.pi) - np.pi
    unclamped_position = pose[:3].copy()
    pose[:3] = eef_state[:3] + np.clip(
        pose[:3] - eef_state[:3],
        -max_position_step,
        max_position_step,
    )
    pose[:3] = np.clip(pose[:3], [-0.45, -0.45, 0.045], [0.45, 0.45, 0.65])
    rotation_delta = (pose[3:] - eef_state[3:] + np.pi) % (2.0 * np.pi) - np.pi
    pose[3:] = eef_state[3:] + np.clip(
        rotation_delta, -max_rotation_step, max_rotation_step
    )
    close_probability = float(jax.nn.sigmoid(outputs["close_logit"])[0])
    action = np.concatenate((pose, np.asarray([-1.0 if close_probability >= 0.5 else 1.0])))
    diagnostics = {
        "close_probability": close_probability,
        "unclamped_position_step_m": float(np.linalg.norm(unclamped_position - eef_state[:3])),
        "executed_position_step_m": float(np.linalg.norm(pose[:3] - eef_state[:3])),
    }
    return action.astype(np.float64), diagnostics


def _write(path: pathlib.Path, result: dict[str, Any]) -> None:
    for control_result in result["controls"].values():
        records = control_result["episodes"]
        success = sum(record.get("success", False) for record in records)
        control_result["summary"] = {
            "attempted": len(records),
            "successes": success,
            "success_rate": success / len(records) if records else None,
            "errors": sum(record.get("status") == "error" for record in records),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if (
        min(args.max_rollout_steps, args.trace_stride) < 1
        or min(args.max_position_step, args.max_rotation_step) <= 0
        or args.grasp_hold_steps < 0
    ):
        raise ValueError("rollout, trace stride, and max position step must be positive")
    os.environ.setdefault("OPENPI_DATA_HOME", "/data2/hzl_workspace_for_pi_mem/.cache/openpi")
    controls = _parse_controls(args.controls)

    from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(
        env_id="VideoUnmask",
        dataset=args.dataset,
        action_space="ee_pose",
        gui_render=False,
        max_steps=args.max_rollout_steps,
    )
    episodes = memory_eval._episode_indices(  # noqa: SLF001
        builder, args.episodes, args.max_episodes
    )
    action_config, action_checkpoint, stats, action_predict = _load_action_adapter(
        args.action_run_dir,
        args.action_checkpoint,
    )
    needs_memory_target = any(
        control in {"oracle_target", "predicted_target"} for control in controls
    )
    memory_checkpoint = None
    memory_predict = None
    backbone = None
    tokenizer = None
    if needs_memory_target:
        memory_checkpoint, memory_predict = memory_eval._load_memory(  # noqa: SLF001
            args.memory_run_dir,
            args.memory_checkpoint,
        )
        print(f"Restoring frozen backbone from {args.backbone_checkpoint.resolve()}", flush=True)
        backbone, tokenizer = memory_eval._load_backbone(args.backbone_checkpoint)  # noqa: SLF001

    result: dict[str, Any] = {
        "schema_version": 1,
        "dataset": args.dataset,
        "episode_indices": episodes,
        "action_checkpoint": str(action_checkpoint.resolve()),
        "memory_checkpoint": str(memory_checkpoint.resolve()) if memory_checkpoint else None,
        "max_rollout_steps": args.max_rollout_steps,
        "max_position_step": args.max_position_step,
        "max_rotation_step": args.max_rotation_step,
        "grasp_hold_steps": args.grasp_hold_steps,
        "action_target_mode": str(action_config.get("target_mode", "absolute")),
        "phase_conditioned": bool(action_config.get("phase_conditioned", False)),
        "phase_control": args.phase_control,
        "controls": {control: {"episodes": []} for control in controls},
    }
    started = time.monotonic()
    for ordinal, episode in enumerate(episodes, start=1):
        targets = None
        prompt = None
        for control in controls:
            env = None
            trace = []
            try:
                env = builder.make_env_for_episode(episode)
                observation, info = env.reset()
                bin_positions = [
                    memory_eval._as_numpy(actor.pose.p).reshape(-1, 3)[0].astype(float).tolist()  # noqa: SLF001
                    for actor in env.unwrapped.spawned_bins
                ]
                if targets is None:
                    geometry_target, prompt = _oracle_geometry_target(observation, info)
                    targets = {"oracle_geometry": geometry_target}
                    if needs_memory_target:
                        conditioning = memory_eval._conditioning(  # noqa: SLF001
                            backbone, tokenizer, observation, info
                        )
                        prompt = conditioning["prompt"]
                        targets.update(
                            oracle_target=memory_eval._predict_point(  # noqa: SLF001
                                memory_predict, conditioning, "oracle"
                            ),
                            predicted_target=memory_eval._predict_point(  # noqa: SLF001
                                memory_predict, conditioning, "full"
                            ),
                        )
                target = targets[control]
                target_crop = action_data.crop_target_image(
                    _take_last(observation, "front_rgb_list"),
                    np.asarray(target),
                )
                status = "timeout"
                final_info: dict[str, Any] = {}
                policy_step = 0
                grasp_pose = None
                grasp_hold_remaining = 0
                phase = 0
                target_actor = env.unwrapped.bin_0
                target_initial = memory_eval._as_numpy(target_actor.pose.p).reshape(-1, 3)[0].astype(np.float64)  # noqa: SLF001
                min_target_xy = float("inf")
                max_target_lift = 0.0
                reached = False
                grasped = False
                lifted = False
                for rollout_step in range(args.max_rollout_steps):
                    eef_before = _take_last(observation, "eef_state_list").astype(np.float64)
                    target_before = memory_eval._as_numpy(target_actor.pose.p).reshape(-1, 3)[0].astype(np.float64)  # noqa: SLF001
                    xy_distance = float(np.linalg.norm(eef_before[:2] - target_before[:2]))
                    z_delta = float(eef_before[2] - target_before[2])
                    target_lift = float(target_before[2] - target_initial[2])
                    min_target_xy = min(min_target_xy, xy_distance)
                    max_target_lift = max(max_target_lift, target_lift)
                    reached = reached or xy_distance <= 0.04
                    grasped = grasped or target_lift >= 0.01
                    lifted = lifted or target_lift >= 0.05
                    if args.phase_control == "oracle":
                        if phase == 0 and xy_distance <= 0.04:
                            phase = 1
                        if phase == 1 and xy_distance <= 0.025 and -0.005 <= z_delta <= 0.04:
                            phase = 2
                            grasp_pose = eef_before[:6].copy()
                            grasp_hold_remaining = args.grasp_hold_steps
                        if phase == 2 and grasp_hold_remaining <= 0:
                            phase = 3
                    command, diagnostics = _action(
                        action_predict,
                        stats,
                        target,
                        observation,
                        target_crop,
                        policy_step,
                        progress_steps=int(action_config["progress_steps"]),
                        max_position_step=args.max_position_step,
                        max_rotation_step=args.max_rotation_step,
                        target_mode=str(action_config.get("target_mode", "absolute")),
                        phase=phase,
                    )
                    if args.phase_control == "oracle":
                        command[6] = 1.0 if phase < 2 else -1.0
                        if phase == 2:
                            command[:6] = grasp_pose
                            grasp_hold_remaining -= 1
                        else:
                            policy_step += 1
                    elif grasp_pose is None and diagnostics["close_probability"] >= 0.5:
                        grasp_pose = _take_last(observation, "eef_state_list")[:3].astype(np.float64)
                        grasp_hold_remaining = args.grasp_hold_steps
                    if args.phase_control == "policy" and grasp_pose is not None:
                        # Keep the fingers centered while they physically close;
                        # after that, retain only XY and let the learned policy
                        # execute its vertical lift.
                        command[:2] = grasp_pose[:2]
                        if grasp_hold_remaining > 0:
                            command[:3] = grasp_pose
                            grasp_hold_remaining -= 1
                        else:
                            policy_step += 1
                    elif args.phase_control == "policy":
                        policy_step += 1
                    diagnostics["policy_step"] = policy_step
                    diagnostics["grasp_hold_remaining"] = grasp_hold_remaining
                    diagnostics["phase"] = phase
                    diagnostics["phase_name"] = action_data.PHASES[phase]
                    diagnostics["target_xy_distance_m"] = xy_distance
                    diagnostics["target_z_delta_m"] = z_delta
                    if rollout_step % args.trace_stride == 0:
                        trace.append(
                            {
                                "step": rollout_step,
                                "eef_state": _take_last(observation, "eef_state_list").astype(float).tolist(),
                                "gripper_state": _take_last(observation, "gripper_state_list").astype(float).tolist(),
                                "action": command.tolist(),
                                **diagnostics,
                            }
                        )
                    next_observation, _, terminated, truncated, step_info = env.step(command)
                    final_info = step_info
                    status = str(step_info.get("status", "ongoing"))
                    if next_observation is not None:
                        observation = next_observation
                    if (
                        status != "ongoing"
                        or memory_eval._as_bool(terminated)  # noqa: SLF001
                        or memory_eval._as_bool(truncated)  # noqa: SLF001
                    ):
                        break
                record = {
                    "episode": episode,
                    "difficulty": builder.metadata_index[("VideoUnmask", episode)].get("difficulty"),
                    "prompt": prompt,
                    "target_point_yx": target,
                    "bin_positions_xyz": bin_positions,
                    "status": status,
                    "success": status == "success",
                    "rollout_steps": rollout_step + 1,
                    "reached": reached,
                    "grasped": grasped,
                    "lifted": lifted,
                    "min_target_xy_distance_m": min_target_xy,
                    "max_target_lift_m": max_target_lift,
                    "final_phase": phase,
                    "trace": trace,
                }
                if final_info.get("error_message"):
                    record["error"] = str(final_info["error_message"])
            except Exception as exc:
                record = {
                    "episode": episode,
                    "difficulty": builder.metadata_index[("VideoUnmask", episode)].get("difficulty"),
                    "prompt": prompt,
                    "target_point_yx": targets.get(control) if targets else None,
                    "status": "error",
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": trace,
                }
            finally:
                memory_eval._close(env)  # noqa: SLF001
            result["controls"][control]["episodes"].append(record)
        _write(args.output, result)
        statuses = ", ".join(
            f"{control}={result['controls'][control]['episodes'][-1]['status']}" for control in controls
        )
        print(
            f"[{ordinal}/{len(episodes)}] episode {episode}: {statuses} ({(time.monotonic() - started) / 60:.1f} min)",
            flush=True,
        )

    _write(args.output, result)
    for control in controls:
        summary = result["controls"][control]["summary"]
        print(
            f"{control}: {summary['successes']}/{summary['attempted']} "
            f"rate={summary['success_rate']:.3f} errors={summary['errors']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
