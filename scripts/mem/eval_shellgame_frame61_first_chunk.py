#!/usr/bin/env python3
"""Diagnose the very first action chunk produced after ShellGame frame 59.

The policy sees the fixed direct-visual memory from frames 0..59 and the live
post-swap observation.  It is queried exactly once with ``frame_index=59``.
The resulting action[0] is therefore the target for dataset frame 60 (the 61st
step under one-based counting).  We report both command-space intent and the
actual EEF motion after executing the deployment prefix actions[0:8].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples" / "shellgame") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples" / "shellgame"))

import numpy as np

from scripts.mem import eval_shellgame_frozen_mem_action_paired_closed_loop as paired
from scripts.mem import eval_shellgame_qwen_event_pi_action_closed_loop as base


DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi0_shellgame_qwen_distilled_memory_action_v10_eef7_260826/"
    "direct_visual_mem_step999_v10_mix500_6gpu_260826/250"
)
DEFAULT_OUTPUT = Path(
    "evaluation/shellgame/qwen_distilled_mem_v10_step250_frame61_first_chunk6_260826/result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--direct-memory", type=Path, default=paired.DEFAULT_DIRECT_MEMORY)
    parser.add_argument("--raw-root", type=Path, default=base.DEFAULT_RAW_ROOT)
    parser.add_argument("--episodes", default=paired.DEFAULT_EPISODES)
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--prompt", default=base.PROMPT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8033)
    parser.add_argument("--noise-salt", type=int, default=260826)
    parser.add_argument("--execute-steps", type=int, default=8)
    parser.add_argument("--overhead-radius", type=float, default=0.06)
    parser.add_argument("--precision-radius", type=float, default=0.03)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _nearest_cup(xy: np.ndarray, cups: dict[str, np.ndarray]) -> tuple[str, float]:
    distances = {name: float(np.linalg.norm(xy - np.asarray(pos)[:2])) for name, pos in cups.items()}
    nearest = min(distances, key=distances.get)
    return nearest, distances[nearest]


def _command_metric(
    command: np.ndarray,
    *,
    target: str,
    cups: dict[str, np.ndarray],
    overhead_radius: float,
    precision_radius: float,
) -> dict[str, Any]:
    target_position = np.asarray(cups[target], dtype=np.float32)
    xy_error = float(np.linalg.norm(command[:2] - target_position[:2]))
    nearest, nearest_error = _nearest_cup(command[:2], cups)
    return {
        "command": command.astype(float).tolist(),
        "target_xy_error_m": xy_error,
        "target_z_delta_m": float(command[2] - target_position[2]),
        "nearest_cup": nearest,
        "nearest_cup_xy_error_m": nearest_error,
        "nearest_is_target": nearest == target,
        "within_60mm": xy_error <= overhead_radius,
        "within_30mm": xy_error <= precision_radius,
    }


def _run_episode(
    episode: int,
    policy: paired.FixedNoiseRemotePolicy,
    memory: np.ndarray,
    shell,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = paired._metadata(args.raw_root, episode)  # noqa: SLF001
    command_args = metadata["command_args"]
    policy_args = base._episode_args(command_args, args.robosuite_root)  # noqa: SLF001
    ep_args = base.shell_main._episode_namespace(  # noqa: SLF001
        policy_args,
        seed=int(command_args["seed"]),
        initial_ball_cup=str(command_args["initial_ball_cup"]),
        num_swaps=int(command_args["num_swaps"]),
    )
    env = shell.make_env(ep_args)
    try:
        scene = base.oracle_replay._prepare_scripted_state(shell, env, ep_args)  # noqa: SLF001
        if scene["swaps"] != metadata["swaps"] or scene["final_ball_cup"] != metadata["final_ball_cup"]:
            raise RuntimeError(f"Episode reconstruction mismatch for {episode}")
        wrist_camera = shell.resolve_wrist_camera_name(env, ep_args.wrist_camera)
        live_base, live_wrist, retries = base._current_images(shell, env, ep_args, wrist_camera)  # noqa: SLF001
        state = base._policy_state(shell, env)  # noqa: SLF001
        cups = {
            name: np.asarray(position, dtype=np.float32)
            for name, position in base.shell_main._cup_positions(shell, env).items()  # noqa: SLF001
        }
        target = str(scene["target_cup"])
        initial_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float32)

        policy.start_episode(episode)
        prediction = policy.infer(
            {
                "state_raw": state,
                "semantic_memory_raw": memory,
                "base_rgb": live_base,
                "wrist_rgb": live_wrist,
                "prompt": args.prompt,
                "episode_index": episode,
                "frame_index": 59,
                "episode_T": 155,
            }
        )
        actions = np.asarray(prediction["actions"], dtype=np.float32)
        if actions.shape != (16, 7) or not np.all(np.isfinite(actions)):
            raise ValueError(f"Invalid action chunk for episode {episode}: {actions.shape}")

        inspected = {
            str(index): _command_metric(
                actions[index],
                target=target,
                cups=cups,
                overhead_radius=args.overhead_radius,
                precision_radius=args.precision_radius,
            )
            for index in (0, 7, 15)
        }
        execution = []
        min_actual_xy = float("inf")
        for index, raw_command in enumerate(actions[: args.execute_steps]):
            action_low, action_high = env.action_spec
            command = np.clip(raw_command, action_low, action_high)
            env.step(command)
            eef = np.asarray(shell.get_eef_pos(env), dtype=np.float32)
            target_xy = float(np.linalg.norm(eef[:2] - cups[target][:2]))
            min_actual_xy = min(min_actual_xy, target_xy)
            nearest, nearest_error = _nearest_cup(eef[:2], cups)
            execution.append(
                {
                    "action_index": index,
                    "eef": eef.astype(float).tolist(),
                    "target_xy_error_m": target_xy,
                    "target_z_delta_m": float(eef[2] - cups[target][2]),
                    "nearest_cup": nearest,
                    "nearest_cup_xy_error_m": nearest_error,
                }
            )
        final_execution = execution[-1]
        return {
            "episode": episode,
            "target_cup": target,
            "gt_final_slot": str(metadata["final_ball_cup"]),
            "policy_query_frame_index": 59,
            "first_action_dataset_frame": 60,
            "first_action_one_based_step": 61,
            "initial_eef": initial_eef.astype(float).tolist(),
            "target_cup_position": cups[target].astype(float).tolist(),
            "render_retries": int(retries),
            "policy_inference_ms": float(prediction["policy_timing"]["infer_ms"]),
            "command_metrics": inspected,
            "execute_steps": args.execute_steps,
            "executed_trace": execution,
            "executed_min_target_xy_m": min_actual_xy,
            "executed_final_target_xy_m": float(final_execution["target_xy_error_m"]),
            "executed_final_target_z_delta_m": float(final_execution["target_z_delta_m"]),
            "executed_final_nearest_cup": str(final_execution["nearest_cup"]),
            "executed_final_nearest_is_target": final_execution["nearest_cup"] == target,
            "executed_final_within_60mm": float(final_execution["target_xy_error_m"]) <= args.overhead_radius,
            "executed_final_within_30mm": float(final_execution["target_xy_error_m"]) <= args.precision_radius,
        }
    finally:
        env.close()


def _summary(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def count_command(index: int, key: str) -> int:
        return sum(bool(row["command_metrics"][str(index)][key]) for row in records)

    return {
        "episodes": len(records),
        "action0_nearest_target": count_command(0, "nearest_is_target"),
        "action0_within_60mm": count_command(0, "within_60mm"),
        "action0_within_30mm": count_command(0, "within_30mm"),
        "action7_nearest_target": count_command(7, "nearest_is_target"),
        "action7_within_60mm": count_command(7, "within_60mm"),
        "action7_within_30mm": count_command(7, "within_30mm"),
        "action15_nearest_target": count_command(15, "nearest_is_target"),
        "action15_within_60mm": count_command(15, "within_60mm"),
        "action15_within_30mm": count_command(15, "within_30mm"),
        "executed_steps": args.execute_steps,
        "executed_final_nearest_target": sum(bool(row["executed_final_nearest_is_target"]) for row in records),
        "executed_final_within_60mm": sum(bool(row["executed_final_within_60mm"]) for row in records),
        "executed_final_within_30mm": sum(bool(row["executed_final_within_30mm"]) for row in records),
        "mean_action0_target_xy_m": float(np.mean([row["command_metrics"]["0"]["target_xy_error_m"] for row in records])),
        "mean_action7_target_xy_m": float(np.mean([row["command_metrics"]["7"]["target_xy_error_m"] for row in records])),
        "mean_action15_target_xy_m": float(np.mean([row["command_metrics"]["15"]["target_xy_error_m"] for row in records])),
        "mean_executed_final_target_xy_m": float(np.mean([row["executed_final_target_xy_m"] for row in records])),
    }


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.direct_memory = args.direct_memory.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    if not 1 <= args.execute_steps <= 16:
        raise ValueError("--execute-steps must be in [1,16]")
    episodes = [int(value.strip()) for value in args.episodes.split(",") if value.strip()]
    direct = paired._load_direct(args.direct_memory)  # noqa: SLF001
    memory = np.asarray(direct["memory"], dtype=np.float32)
    labels = np.asarray(direct["label"], dtype=np.int32)
    predictions = np.asarray(direct["prediction"], dtype=np.int32)
    if any(predictions[episode] != labels[episode] for episode in episodes):
        raise ValueError("Frame61 probe requires semantically correct direct memories")

    policy = paired.FixedNoiseRemotePolicy(args.host, args.port, salt=args.noise_salt)
    shell = base.shell_main._import_shellgame_tools(args.robosuite_root)  # noqa: SLF001
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "frame59 observation -> first Pi action chunk intent and execution",
        "checkpoint": str(args.checkpoint),
        "direct_memory": str(args.direct_memory),
        "episodes": episodes,
        "noise_salt": args.noise_salt,
        "alignment": "policy frame_index=59; action[0] targets dataset frame60 / one-based step61",
        "thresholds": {
            "overhead_radius_m": args.overhead_radius,
            "precision_radius_m": args.precision_radius,
        },
        "records": [],
    }
    try:
        for episode in episodes:
            row = _run_episode(episode, policy, memory[episode], shell, args)
            payload["records"].append(row)
            payload["summary"] = _summary(payload["records"], args)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(
                f"ep={episode} target={row['target_cup']} "
                f"a0={row['command_metrics']['0']['target_xy_error_m'] * 1000:.1f}mm "
                f"a7={row['command_metrics']['7']['target_xy_error_m'] * 1000:.1f}mm "
                f"a15={row['command_metrics']['15']['target_xy_error_m'] * 1000:.1f}mm "
                f"exec8={row['executed_final_target_xy_m'] * 1000:.1f}mm",
                flush=True,
            )
    finally:
        policy.close()
    payload["summary"] = _summary(payload["records"], args)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
