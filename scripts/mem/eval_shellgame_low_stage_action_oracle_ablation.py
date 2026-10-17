#!/usr/bin/env python3
"""Low-stage action-component Oracle ablation with normal memory selection.

The policy and semantic memory control the complete observation phase, target
selection, and high approach.  An intervention can activate only after the
policy commands a descent toward the correct target cup.  Rotation always
remains the model prediction.  This isolates whether late closed-loop failures
come from XY centering, Z/grasp depth, or gripper/lift continuity.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "shellgame"))

import numpy as np  # noqa: E402

from scripts.mem import eval_shellgame_frozen_mem_action_paired_closed_loop as paired  # noqa: E402
from scripts.mem import eval_shellgame_qwen_event_pi_action_closed_loop as base  # noqa: E402

CONDITIONS = (
    "model",
    "oracle_xy",
    "oracle_z",
    "oracle_gripper_lift",
    "oracle_xy_z",
    "oracle_xy_z_gripper",
)
DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi0_shellgame_qwen_distilled_memory_waypoint_grasp_v6_eef7_260826/"
    "direct_visual_waypoint_grasp_v6_60_30_5_3_2_3k_6gpu_260826/2000"
)
DEFAULT_OUTPUT = Path(
    "evaluation/shellgame/low_stage_action_oracle_ablation5_step2000_replan8_260827/result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--raw-root", type=Path, default=base.DEFAULT_RAW_ROOT)
    parser.add_argument("--direct-memory", type=Path, default=paired.DEFAULT_DIRECT_MEMORY)
    parser.add_argument("--episodes", default="47,80,195,31,16")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--prompt", default=base.PROMPT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8064)
    parser.add_argument("--noise-salt", type=int, default=260826)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--max-policy-steps", type=int, default=120)
    parser.add_argument("--selection-skip", type=int, default=10)
    parser.add_argument("--selection-window", type=int, default=30)
    parser.add_argument("--selection-radius", type=float, default=0.06)
    parser.add_argument("--precision-radius", type=float, default=0.03)
    parser.add_argument("--lift-success-height", type=float, default=0.08)
    parser.add_argument("--activation-height-m", type=float, default=0.12)
    parser.add_argument("--activation-descent-epsilon-m", type=float, default=0.0005)
    parser.add_argument("--hold-z-above-xy-mm", type=float, default=10.0)
    parser.add_argument("--aligned-xy-mm", type=float, default=6.0)
    parser.add_argument("--close-xy-mm", type=float, default=5.0)
    parser.add_argument("--close-z-mm", type=float, default=3.0)
    parser.add_argument("--close-hold-steps", type=int, default=3)
    parser.add_argument("--grasp-hold-steps", type=int, default=10)
    parser.add_argument("--slow-descent-mm", type=float, default=2.0)
    parser.add_argument("--normal-descent-mm", type=float, default=8.0)
    parser.add_argument("--lift-step-mm", type=float, default=10.0)
    parser.add_argument("--lift-height-m", type=float, default=0.20)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_OUTPUT.parent / "videos")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _intervention_flags(condition: str) -> tuple[bool, bool, bool]:
    return (
        condition in {"oracle_xy", "oracle_xy_z", "oracle_xy_z_gripper"},
        condition in {"oracle_z", "oracle_xy_z", "oracle_xy_z_gripper"},
        condition in {"oracle_gripper_lift", "oracle_xy_z_gripper"},
    )


def _new_intervention_state() -> dict[str, Any]:
    return {
        "activated": False,
        "activation_step": None,
        "phase": "model_prefix",
        "aligned_low_run": 0,
        "model_close_run": 0,
        "grasp_steps": 0,
        "lift_steps": 0,
        "applied_xy_steps": 0,
        "applied_z_steps": 0,
        "applied_gripper_steps": 0,
    }


def _intervene(
    command: np.ndarray,
    *,
    condition: str,
    rollout_step: int,
    measured_eef: np.ndarray,
    target_cup_pos: np.ndarray,
    grasp_z: float,
    state: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(command, dtype=np.float32)
    corrected = raw.copy()
    use_xy, use_z, use_gripper = _intervention_flags(condition)
    raw_target_xy = float(np.linalg.norm(raw[:2] - target_cup_pos[:2]))
    measured_target_xy = float(np.linalg.norm(measured_eef[:2] - target_cup_pos[:2]))
    selection_expressed = raw_target_xy <= args.selection_radius
    descending = float(raw[2]) < float(measured_eef[2]) - args.activation_descent_epsilon_m
    low_enough = float(measured_eef[2]) <= grasp_z + args.activation_height_m

    if (
        condition != "model"
        and not state["activated"]
        and selection_expressed
        and descending
        and low_enough
    ):
        state["activated"] = True
        state["activation_step"] = rollout_step
        state["phase"] = "descent"

    if not state["activated"]:
        return corrected, {
            "activated": False,
            "phase": state["phase"],
            "raw_target_xy_mm": raw_target_xy * 1_000.0,
            "measured_target_xy_mm": measured_target_xy * 1_000.0,
            "measured_dz_mm": (float(measured_eef[2]) - grasp_z) * 1_000.0,
            "raw_command": raw.astype(float).tolist(),
            "corrected_command": corrected.astype(float).tolist(),
        }

    if use_xy:
        corrected[:2] = target_cup_pos[:2]
        state["applied_xy_steps"] += 1

    measured_dz = float(measured_eef[2]) - grasp_z
    aligned_and_low = (
        measured_target_xy <= args.close_xy_mm / 1_000.0
        and abs(measured_dz) <= args.close_z_mm / 1_000.0
    )
    state["aligned_low_run"] = state["aligned_low_run"] + 1 if aligned_and_low else 0
    model_close = float(raw[6]) > 0.0
    state["model_close_run"] = state["model_close_run"] + 1 if model_close else 0

    if state["phase"] == "descent":
        if use_z:
            if measured_target_xy > args.hold_z_above_xy_mm / 1_000.0:
                corrected[2] = measured_eef[2]
            elif measured_target_xy > args.aligned_xy_mm / 1_000.0:
                corrected[2] = max(grasp_z, measured_eef[2] - args.slow_descent_mm / 1_000.0)
            else:
                corrected[2] = max(grasp_z, measured_eef[2] - args.normal_descent_mm / 1_000.0)
            state["applied_z_steps"] += 1
        if use_gripper:
            corrected[6] = -1.0
            state["applied_gripper_steps"] += 1
        ready = state["aligned_low_run"] >= args.close_hold_steps
        if ready and (use_gripper or (use_z and state["model_close_run"] > 0)):
            state["phase"] = "grasp"

    if state["phase"] == "grasp":
        corrected[2] = grasp_z if use_z or use_gripper else corrected[2]
        if use_z or use_gripper:
            state["applied_z_steps"] += 1
        if use_gripper:
            corrected[6] = 1.0
            state["applied_gripper_steps"] += 1
        state["grasp_steps"] += 1
        if state["grasp_steps"] >= args.grasp_hold_steps:
            state["phase"] = "lift"

    if state["phase"] == "lift":
        lift_z = min(
            grasp_z + args.lift_height_m,
            max(float(measured_eef[2]), grasp_z) + args.lift_step_mm / 1_000.0,
        )
        corrected[2] = lift_z
        state["applied_z_steps"] += 1
        if use_gripper:
            corrected[6] = 1.0
            state["applied_gripper_steps"] += 1
        state["lift_steps"] += 1

    return corrected, {
        "activated": True,
        "activation_step": state["activation_step"],
        "phase": state["phase"],
        "raw_target_xy_mm": raw_target_xy * 1_000.0,
        "measured_target_xy_mm": measured_target_xy * 1_000.0,
        "measured_dz_mm": measured_dz * 1_000.0,
        "aligned_low_run": state["aligned_low_run"],
        "model_close": model_close,
        "raw_command": raw.astype(float).tolist(),
        "corrected_command": corrected.astype(float).tolist(),
    }


def _run_episode(
    episode: int,
    condition: str,
    policy: paired.FixedNoiseRemotePolicy,
    memory: np.ndarray,
    shell,
    args: argparse.Namespace,
) -> dict[str, Any]:
    episode_dir = args.raw_root / f"episode_{episode:06d}"
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    command_args = metadata["command_args"]
    policy_args = base._episode_args(command_args, args.robosuite_root)  # noqa: SLF001
    ep_args = base.shell_main._episode_namespace(  # noqa: SLF001
        policy_args,
        seed=int(command_args["seed"]),
        initial_ball_cup=str(command_args["initial_ball_cup"]),
        num_swaps=int(command_args["num_swaps"]),
    )
    env = shell.make_env(ep_args)
    replay: list[np.ndarray] = []
    inference_ms: list[float] = []
    intervention = _new_intervention_state()
    try:
        scene = base.oracle_replay._prepare_scripted_state(shell, env, ep_args)  # noqa: SLF001
        if scene["swaps"] != metadata["swaps"] or scene["final_ball_cup"] != metadata["final_ball_cup"]:
            raise RuntimeError("Reconstructed simulator episode does not match dataset metadata")
        wrist_camera = shell.resolve_wrist_camera_name(env, ep_args.wrist_camera)
        with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
            recorded_prefix = np.asarray(source["third_person_images"][:60], dtype=np.uint8)
        replay.extend(np.array(frame, order="C", copy=True) for frame in recorded_prefix)
        live_base, live_wrist, initial_retries = base._current_images(  # noqa: SLF001
            shell, env, ep_args, wrist_camera
        )
        render_retries = initial_retries
        render_mae = float(
            np.mean(np.abs(live_base.astype(np.float32) - recorded_prefix[-1].astype(np.float32)))
        )
        target_cup = scene["target_cup"]
        grasp_z = float(env.cup_handle_grasp_z() + ep_args.robot_grasp_z_offset)
        target_initial_z = float(scene["settle_cup_pos"][target_cup][2])
        target_initial_xy = np.asarray(scene["settle_cup_pos"][target_cup][:2], dtype=np.float32)
        votes = dict.fromkeys(shell.CUP_NAMES, 0)
        distance_sums = dict.fromkeys(shell.CUP_NAMES, 0.0)
        min_target_xy = float("inf")
        max_target_lift = 0.0
        max_close_cup_xy_displacement = 0.0
        close_xy_values: list[float] = []
        first_close_step = None
        first_close_xy = None
        first_close_eef_z = None
        rollout_steps = 0
        success = False
        final_stats = None
        trace: list[dict[str, Any]] = []
        while rollout_steps < args.max_policy_steps and not success:
            state_raw = base._policy_state(shell, env)  # noqa: SLF001
            prediction = policy.infer(
                {
                    "state_raw": state_raw,
                    "semantic_memory_raw": memory,
                    "base_rgb": live_base,
                    "wrist_rgb": live_wrist,
                    "prompt": args.prompt,
                    "episode_index": episode,
                    "frame_index": 59 + rollout_steps,
                    "episode_T": 155,
                }
            )
            actions = np.asarray(prediction["actions"], dtype=np.float32)
            if actions.shape != (16, 7) or not np.all(np.isfinite(actions)):
                raise ValueError(f"Invalid action chunk: {actions.shape}")
            inference_ms.append(float(prediction["policy_timing"]["infer_ms"]))
            for chunk_index, raw_command in enumerate(actions[: args.replan_steps]):
                measured = np.asarray(shell.get_eef_pos(env), dtype=np.float32)
                cups_before = base.shell_main._cup_positions(shell, env)  # noqa: SLF001
                target_before = np.asarray(cups_before[target_cup], dtype=np.float32)
                command, event = _intervene(
                    raw_command,
                    condition=condition,
                    rollout_step=rollout_steps + 1,
                    measured_eef=measured,
                    target_cup_pos=target_before,
                    grasp_z=grasp_z,
                    state=intervention,
                    args=args,
                )
                action_low, action_high = env.action_spec
                command = np.clip(command, action_low, action_high)
                env.step(command)
                rollout_steps += 1
                live_base, live_wrist, retries = base._current_images(  # noqa: SLF001
                    shell, env, ep_args, wrist_camera
                )
                render_retries += retries
                replay.append(np.array(live_base, order="C", copy=True))
                cups = base.shell_main._cup_positions(shell, env)  # noqa: SLF001
                eef = np.asarray(shell.get_eef_pos(env), dtype=np.float32)
                distances = {
                    cup: float(np.linalg.norm(eef[:2] - position[:2]))
                    for cup, position in cups.items()
                }
                nearest = min(distances, key=distances.get)
                min_target_xy = min(min_target_xy, distances[target_cup])
                target_lift = float(cups[target_cup][2] - target_initial_z)
                max_target_lift = max(max_target_lift, target_lift)
                if float(command[6]) > 0.0:
                    close_xy = distances[target_cup]
                    close_xy_values.append(close_xy)
                    max_close_cup_xy_displacement = max(
                        max_close_cup_xy_displacement,
                        float(np.linalg.norm(cups[target_cup][:2] - target_initial_xy)),
                    )
                    if first_close_step is None:
                        first_close_step = rollout_steps
                        first_close_xy = close_xy
                        first_close_eef_z = float(eef[2])
                selection_end = args.selection_skip + args.selection_window
                if (
                    args.selection_skip <= rollout_steps - 1 < selection_end
                    and distances[nearest] <= args.selection_radius
                ):
                    votes[nearest] += 1
                    distance_sums[nearest] += distances[nearest]
                success, final_stats = base.shell_main._success(  # noqa: SLF001
                    shell,
                    env,
                    target_cup,
                    scene["settle_cup_pos"],
                    args.lift_success_height,
                )
                if rollout_steps % 5 == 0 or event.get("activation_step") == rollout_steps or success:
                    trace.append(
                        {
                            "step": rollout_steps,
                            "chunk_index": chunk_index,
                            "eef": eef.astype(float).tolist(),
                            "command": command.astype(float).tolist(),
                            "target_xy_m": distances[target_cup],
                            "target_lift_m": target_lift,
                            "intervention": event,
                        }
                    )
                if success or rollout_steps >= args.max_policy_steps:
                    break
        selected = base._selected_cup(votes, distance_sums)  # noqa: SLF001
        video_path = None
        if not args.no_videos:
            video_path = args.video_dir / condition / (
                f"episode_{episode:06d}_{'success' if success else 'failure'}.mp4"
            )
            base._write_video(video_path, replay, args.video_fps)  # noqa: SLF001
        return {
            "episode": episode,
            "condition": condition,
            "gt_final_slot_scoring_only": str(metadata["final_ball_cup"]),
            "target_cup_identity_scoring_only": target_cup,
            "selected_cup_identity": selected,
            "cup_selection_correct": selected == target_cup,
            "success": bool(success),
            "rollout_steps": rollout_steps,
            "selection_votes": votes,
            "min_target_xy_m": min_target_xy,
            "max_target_lift_m": max_target_lift,
            "first_close_step": first_close_step,
            "first_close_target_xy_m": first_close_xy,
            "first_close_eef_z_m": first_close_eef_z,
            "mean_close_target_xy_m": float(np.mean(close_xy_values)) if close_xy_values else None,
            "max_close_target_xy_m": max(close_xy_values) if close_xy_values else None,
            "max_close_cup_xy_displacement_m": max_close_cup_xy_displacement,
            "grasp_z_m": grasp_z,
            "intervention_summary": intervention,
            "final_stats": final_stats,
            "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None,
            "render_frame59_mae_uint8": render_mae,
            "egl_corrupt_read_retries": render_retries,
            "video": str(video.resolve()) if (video := video_path) is not None else None,
            "trace": trace,
        }
    finally:
        env.close()


def _summary(records: list[dict[str, Any]], precision_radius: float) -> dict[str, Any]:
    values = np.asarray([row["min_target_xy_m"] for row in records], dtype=np.float64)
    return {
        "episodes": len(records),
        "selection_correct": sum(row["cup_selection_correct"] for row in records),
        "lift_successes": sum(row["success"] for row in records),
        "precision_count": int(np.sum(values <= precision_radius)),
        "median_min_target_xy_m": float(np.median(values)) if len(values) else None,
        "mean_max_target_lift_m": (
            float(np.mean([row["max_target_lift_m"] for row in records])) if records else None
        ),
        "activated": sum(row["intervention_summary"]["activated"] for row in records),
    }


def main() -> None:
    args = parse_args()
    for field in ("checkpoint", "raw_root", "direct_memory", "output", "video_dir"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    episodes = [int(value.strip()) for value in args.episodes.split(",") if value.strip()]
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("Episodes must be non-empty and unique")
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Conditions must be non-empty and unique")
    if any(condition not in CONDITIONS for condition in conditions):
        raise ValueError(f"Conditions must be a subset of {CONDITIONS}")
    if not 1 <= args.replan_steps <= 16 or args.max_policy_steps < 1:
        raise ValueError("Invalid rollout settings")
    if min(
        args.activation_height_m,
        args.close_hold_steps,
        args.grasp_hold_steps,
        args.lift_step_mm,
    ) <= 0:
        raise ValueError("Intervention thresholds must be positive")

    direct = paired._load_direct(args.direct_memory)  # noqa: SLF001
    memories = direct["memory"]
    labels = direct["label"]
    predictions = direct["prediction"]
    assert isinstance(memories, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(predictions, np.ndarray)
    if max(episodes) >= len(memories):
        raise ValueError("Requested episode is absent from the direct-memory bank")
    if any(int(predictions[episode]) != int(labels[episode]) for episode in episodes):
        raise ValueError("This diagnostic requires semantically correct direct memories")

    policy = paired.FixedNoiseRemotePolicy(args.host, args.port, salt=args.noise_salt)
    shell = base.shell_main._import_shellgame_tools(args.robosuite_root)  # noqa: SLF001
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "normal semantic-memory selection plus low-stage action-component Oracle ablation",
        "checkpoint": str(args.checkpoint),
        "direct_memory": str(args.direct_memory),
        "noise_seed_contract": "SeedSequence([noise_salt, episode, policy_query_index])",
        "noise_salt": args.noise_salt,
        "episodes": episodes,
        "conditions": conditions,
        "rotation_source": "model in every condition",
        "activation": "correct-cup command within radius + descending + below grasp_z plus activation height",
        "control": {
            "replan_steps": args.replan_steps,
            "max_policy_steps": args.max_policy_steps,
            "precision_radius_m": args.precision_radius,
            "activation_height_m": args.activation_height_m,
            "close_xy_mm": args.close_xy_mm,
            "close_z_mm": args.close_z_mm,
            "close_hold_steps": args.close_hold_steps,
            "grasp_hold_steps": args.grasp_hold_steps,
        },
        "records": [],
    }
    started = time.monotonic()
    try:
        for episode_ordinal, episode in enumerate(episodes):
            offset = episode_ordinal % len(conditions)
            ordered_conditions = conditions[offset:] + conditions[:offset]
            for condition in ordered_conditions:
                policy.start_episode(episode)
                run_args = copy.copy(args)
                record = _run_episode(
                    episode,
                    condition,
                    policy,
                    np.asarray(memories[episode], dtype=np.float32),
                    shell,
                    run_args,
                )
                payload["records"].append(record)
                payload["summary"] = {
                    current: _summary(
                        [row for row in payload["records"] if row["condition"] == current],
                        args.precision_radius,
                    )
                    for current in conditions
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                print(
                    f"ep={episode} condition={condition} select={record['cup_selection_correct']} "
                    f"min_xy={record['min_target_xy_m'] * 1000:.1f}mm "
                    f"lift={record['success']} phase={record['intervention_summary']['phase']} "
                    f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                    flush=True,
                )
    finally:
        policy.close()
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
