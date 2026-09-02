"""Generate coherent low-stage recovery data from current-action states.

The current semantic-memory policy executes a real closed-loop prefix on an
existing ShellGame episode. Prefix actions and intermediate frames are never
stored. Stored rows contain the original 60-frame observation context, the
post-prefix switch observation, and a coherent Oracle suffix with sustained
live-cup XY, gated Z descent, closure hold, and continuous lift.
"""

# ruff: noqa: E402, SLF001

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import json
import logging
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "shellgame"))

import generate_onpolicy_eef_correction_dataset as legacy
import numpy as np

from scripts.mem import eval_shellgame_frozen_mem_action_paired_closed_loop as paired
from scripts.mem import eval_shellgame_qwen_event_pi_action_closed_loop as action_eval

DATASET_KIND = "current_action_low_stage_coherent_recovery_v11"
SLOTS = ("left", "middle", "right")
PREFIX_STEPS = (40, 48, 56)
SUFFIX_COMMANDS = 95
GRASP_STEPS = 10
MIN_LIFT_STEPS = 20
DEFAULT_RAW_ROOT = action_eval.DEFAULT_RAW_ROOT
DEFAULT_MEMORY = paired.DEFAULT_DIRECT_MEMORY
DEFAULT_EXCLUDED_EPISODES = (
    1459,
    3906,
    1183,
    828,
    2068,
    473,
    1525,
    402,
    4006,
    1316,
    2005,
    1233,
    1556,
    2021,
    2514,
    800,
    3050,
    545,
    513,
    1859,
    4431,
    31,
    16,
    8,
    47,
    80,
    195,
)

_WORKER: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--direct-memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--ports", default="8060,8061")
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--max-candidates", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dataset-seed", type=int, default=260827)
    parser.add_argument("--noise-salt", type=int, default=26082711)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--prefix-steps", default=",".join(map(str, PREFIX_STEPS)))
    parser.add_argument(
        "--exclude-episodes",
        default=",".join(map(str, DEFAULT_EXCLUDED_EPISODES)),
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-offset-mm", type=float, default=45.0)
    parser.add_argument("--min-safe-height-mm", type=float, default=-5.0)
    parser.add_argument("--max-switch-lift-mm", type=float, default=15.0)
    parser.add_argument("--max-open-steps", type=int, default=60)
    parser.add_argument("--hold-z-above-xy-mm", type=float, default=10.0)
    parser.add_argument("--aligned-xy-mm", type=float, default=6.0)
    parser.add_argument("--close-xy-mm", type=float, default=5.0)
    parser.add_argument("--close-z-mm", type=float, default=3.0)
    parser.add_argument("--close-hold-steps", type=int, default=3)
    parser.add_argument("--slow-descent-mm", type=float, default=2.0)
    parser.add_argument("--normal-descent-mm", type=float, default=8.0)
    parser.add_argument("--descent-jitter-mm", type=float, default=2.0)
    parser.add_argument("--lift-height-m", type=float, default=0.20)
    return parser.parse_args()


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _validate_args(args: argparse.Namespace) -> None:
    prefixes = _parse_ints(args.prefix_steps)
    ports = _parse_ints(args.ports)
    if args.width != args.height or args.width != 224:
        raise ValueError("This recipe requires 224x224 images")
    if args.num_episodes <= 0 or args.num_episodes % (len(SLOTS) * len(prefixes)):
        raise ValueError("num-episodes must be divisible by slots times prefix stages")
    if not ports or args.workers <= 0 or args.max_candidates < args.num_episodes:
        raise ValueError("Invalid workers/ports/candidate budget")
    if args.replan_steps != 8 or any(step % args.replan_steps for step in prefixes):
        raise ValueError("Prefixes must align to replan=8")
    if min(prefixes) < 32 or max(prefixes) > 64:
        raise ValueError("Use low-stage prefixes in [32, 64]")
    if SUFFIX_COMMANDS - args.max_open_steps - GRASP_STEPS < MIN_LIFT_STEPS:
        raise ValueError("max-open-steps leaves too few lift steps")


def _worker_init(args_dict: dict[str, Any]) -> None:
    args = argparse.Namespace(**args_dict)
    identity = mp.current_process()._identity
    worker_index = identity[0] - 1 if identity else 0
    ports = _parse_ints(args.ports)
    port = ports[worker_index % len(ports)]
    shell = action_eval.shell_main._import_shellgame_tools(args.robosuite_root)
    direct = paired._load_direct(Path(args.direct_memory))
    policy = paired.FixedNoiseRemotePolicy("127.0.0.1", port, salt=args.noise_salt)
    _WORKER.update(args=args, shell=shell, direct=direct, policy=policy)


def _recorded_context(path: Path) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        third = np.asarray(source["third_person_images"][:60], dtype=np.uint8)
        wrist = np.asarray(source["wrist_images"][:60], dtype=np.uint8)
        position = np.asarray(source["eef_pos"][:60], dtype=np.float32)
        quaternion = np.asarray(source["eef_quat"][:60], dtype=np.float32)
        gripper = np.asarray(source["gripper_state"][:60], dtype=np.float32)
    if third.shape != (60, 224, 224, 3) or wrist.shape != third.shape:
        raise ValueError(f"Invalid recorded context shape: {path}")
    return [
        {
            "base": third[index],
            "wrist": wrist[index],
            "eef_pos": position[index],
            "eef_quat": quaternion[index],
            "gripper_width": action_eval.shell_main._gripper_width(gripper[index]),
        }
        for index in range(60)
    ]


def _live_snapshot(shell, env, base_image: np.ndarray, wrist_image: np.ndarray) -> dict[str, Any]:
    state = action_eval._policy_state(shell, env)
    return {
        "base": np.asarray(base_image, dtype=np.uint8),
        "wrist": np.asarray(wrist_image, dtype=np.uint8),
        "eef_pos": np.asarray(shell.get_eef_pos(env), dtype=np.float32),
        "eef_quat": np.asarray(shell.get_eef_quat(env), dtype=np.float32),
        "gripper_width": float(state[-1]),
    }


def _phase_ranges(open_steps: int, lift_steps: int) -> dict[str, list[int]]:
    open_end = 60 + open_steps
    grasp_end = open_end + GRASP_STEPS
    return {
        "reveal": [0, 9],
        "cover": [10, 19],
        "swap_0": [20, 29],
        "swap_1": [30, 39],
        "swap_2": [40, 49],
        "settle": [50, 59],
        "correction_start": [60, 60],
        "oracle_gated_descent": [61, open_end],
        "oracle_grasp": [open_end + 1, grasp_end],
        "oracle_lift": [grasp_end + 1, grasp_end + lift_steps],
    }


def _attempt(task: tuple[int, int]) -> dict[str, Any]:
    episode, prefix_steps = task
    args = _WORKER["args"]
    shell = _WORKER["shell"]
    direct = _WORKER["direct"]
    policy: paired.FixedNoiseRemotePolicy = _WORKER["policy"]
    raw_root = Path(args.raw_root)
    episode_dir = raw_root / f"episode_{episode:06d}"
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    final_slot = str(metadata["final_ball_cup"])
    target_identity = str(metadata["target_cup_identity"])
    result: dict[str, Any] = {
        "source_episode": episode,
        "prefix_steps": prefix_steps,
        "final_spatial_slot": final_slot,
        "target_cup_identity": target_identity,
    }
    prediction = np.asarray(direct["prediction"])
    label = np.asarray(direct["label"])
    if int(prediction[episode]) != int(label[episode]):
        return result | {"reason": "memory_incorrect"}

    command_args = metadata["command_args"]
    policy_args = action_eval._episode_args(command_args, args.robosuite_root)
    ep_args = action_eval.shell_main._episode_namespace(
        policy_args,
        seed=int(command_args["seed"]),
        initial_ball_cup=str(command_args["initial_ball_cup"]),
        num_swaps=int(command_args["num_swaps"]),
    )
    env = shell.make_env(ep_args)
    temp_dir = Path(args.output) / f".candidate_{episode:06d}_{prefix_steps}"
    try:
        scene = action_eval.oracle_replay._prepare_scripted_state(shell, env, ep_args)
        if scene["swaps"] != metadata["swaps"] or scene["target_cup"] != target_identity:
            raise RuntimeError(f"episode {episode}: simulator reconstruction mismatch")
        wrist_camera = shell.resolve_wrist_camera_name(env, ep_args.wrist_camera)
        live_base, live_wrist, render_retries = action_eval._current_images(
            shell, env, ep_args, wrist_camera
        )
        memory = np.asarray(direct["memory"][episode], dtype=np.float32)
        policy.start_episode(episode)
        votes = dict.fromkeys(shell.CUP_NAMES, 0)
        distance_sums = dict.fromkeys(shell.CUP_NAMES, 0.0)
        executed = 0
        while executed < prefix_steps:
            state = action_eval._policy_state(shell, env)
            output = policy.infer(
                {
                    "state_raw": state,
                    "semantic_memory_raw": memory,
                    "base_rgb": live_base,
                    "wrist_rgb": live_wrist,
                    "prompt": legacy.GRASP_PROMPT,
                    "episode_index": episode,
                    "frame_index": 59 + executed,
                    "episode_T": 155,
                }
            )
            actions = np.asarray(output["actions"], dtype=np.float32)
            if actions.shape != (16, 7) or not np.all(np.isfinite(actions)):
                raise ValueError(f"Invalid model action chunk: {actions.shape}")
            for command in actions[: args.replan_steps]:
                low, high = env.action_spec
                env.step(np.clip(command, low, high))
                executed += 1
                live_base, live_wrist, retries = action_eval._current_images(
                    shell, env, ep_args, wrist_camera
                )
                render_retries += retries
                cups = action_eval.shell_main._cup_positions(shell, env)
                eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
                distances = {
                    cup: float(np.linalg.norm(eef[:2] - np.asarray(pos[:2], dtype=np.float64)))
                    for cup, pos in cups.items()
                }
                nearest = min(distances, key=distances.get)
                if executed > 10 and distances[nearest] <= args.max_offset_mm / 1_000.0:
                    votes[nearest] += 1
                    distance_sums[nearest] += distances[nearest]
                if executed >= prefix_steps:
                    break

        selected = legacy._select_cup(votes, distance_sums)
        if selected != target_identity:
            return result | {"reason": "selection_incorrect", "selected_cup": selected}
        cups = action_eval.shell_main._cup_positions(shell, env)
        target = np.asarray(cups[target_identity], dtype=np.float64)
        switch_eef = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
        grasp_z = float(env.cup_handle_grasp_z() + ep_args.robot_grasp_z_offset)
        offset_m = float(np.linalg.norm(switch_eef[:2] - target[:2]))
        safe_height_m = float(switch_eef[2] - grasp_z)
        target_lift_m = float(target[2] - scene["settle_cup_pos"][target_identity][2])
        result.update(
            selected_cup=selected,
            switch_offset_m=offset_m,
            switch_safe_height_m=safe_height_m,
            switch_target_lift_m=target_lift_m,
        )
        if offset_m > args.max_offset_mm / 1_000.0:
            return result | {"reason": "offset_too_large"}
        if safe_height_m < args.min_safe_height_mm / 1_000.0:
            return result | {"reason": "below_safe_height"}
        if target_lift_m > args.max_switch_lift_mm / 1_000.0:
            return result | {"reason": "target_already_lifted"}

        observations = _recorded_context(episode_dir / "vla_trajectory.npz")
        observations.append(_live_snapshot(shell, env, live_base, live_wrist))
        actions = [legacy._placeholder_absolute_action(shell, item) for item in observations]
        action_mask = [False] * 61
        supervision = [legacy.SUPERVISION_CONTEXT] * 61
        phase_ids = [0] * 10 + [1] * 10 + [2] * 30 + [3] * 10 + [legacy.PHASE_RECENTER]
        canonical_quat = np.asarray(observations[0]["eef_quat"], dtype=np.float64)
        rng = np.random.default_rng(np.random.SeedSequence([args.dataset_seed, episode, prefix_steps]))
        jitter = rng.uniform(
            -args.descent_jitter_mm / 1_000.0,
            args.descent_jitter_mm / 1_000.0,
            size=2,
        )
        stage_metrics: list[dict[str, Any]] = []

        def execute(
            stage: str,
            phase_id: int,
            target_pos: np.ndarray,
            *,
            close: bool,
        ) -> dict[str, Any]:
            command = shell.make_robot_action(
                env,
                target_pos=np.asarray(target_pos, dtype=np.float64),
                target_quat=canonical_quat,
                gripper_action=1.0 if close else -1.0,
            )
            env.step(command)
            nonlocal live_base, live_wrist, render_retries
            live_base, live_wrist, retries = action_eval._current_images(
                shell, env, ep_args, wrist_camera
            )
            render_retries += retries
            observation = _live_snapshot(shell, env, live_base, live_wrist)
            observations.append(observation)
            actions.append(np.asarray(command, dtype=np.float32))
            action_mask.append(True)
            supervision.append(legacy.SUPERVISION_ORACLE)
            phase_ids.append(phase_id)
            actual = np.asarray(observation["eef_pos"], dtype=np.float64)
            live_cup = np.asarray(
                action_eval.shell_main._cup_positions(shell, env)[target_identity],
                dtype=np.float64,
            )
            metric = {
                "stage": stage,
                "eef_to_target_xy_m": float(np.linalg.norm(actual[:2] - live_cup[:2])),
                "eef_to_grasp_z_m": float(actual[2] - grasp_z),
                "target_lift_m": float(
                    live_cup[2] - scene["settle_cup_pos"][target_identity][2]
                ),
                "finger_contacts": legacy.contact_utils._finger_contact_count(env, target_identity),
            }
            stage_metrics.append(metric)
            return metric

        aligned_run = 0
        open_steps = 0
        ready = False
        for _ in range(args.max_open_steps):
            measured = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
            live_cup = np.asarray(
                action_eval.shell_main._cup_positions(shell, env)[target_identity],
                dtype=np.float64,
            )
            oracle_xy = live_cup[:2] + jitter
            xy_error = float(np.linalg.norm(measured[:2] - live_cup[:2]))
            if xy_error > args.hold_z_above_xy_mm / 1_000.0:
                target_z = measured[2]
                stage = "hold_z_recenter"
                phase = legacy.PHASE_RECENTER
            elif xy_error > args.aligned_xy_mm / 1_000.0:
                target_z = max(grasp_z, measured[2] - args.slow_descent_mm / 1_000.0)
                stage = "slow_descent_recenter"
                phase = legacy.PHASE_DESCEND
            else:
                target_z = max(grasp_z, measured[2] - args.normal_descent_mm / 1_000.0)
                stage = "aligned_descent"
                phase = legacy.PHASE_DESCEND
            metric = execute(stage, phase, np.asarray([*oracle_xy, target_z]), close=False)
            open_steps += 1
            aligned_low = (
                metric["eef_to_target_xy_m"] <= args.close_xy_mm / 1_000.0
                and abs(metric["eef_to_grasp_z_m"]) <= args.close_z_mm / 1_000.0
            )
            aligned_run = aligned_run + 1 if aligned_low else 0
            if aligned_run >= args.close_hold_steps:
                ready = True
                break
        if not ready:
            return result | {"reason": "oracle_not_ready", "open_steps": open_steps}

        lift_steps = SUFFIX_COMMANDS - open_steps - GRASP_STEPS
        if lift_steps < MIN_LIFT_STEPS:
            return result | {"reason": "insufficient_lift_steps", "open_steps": open_steps}
        live_cup = np.asarray(
            action_eval.shell_main._cup_positions(shell, env)[target_identity], dtype=np.float64
        )
        grasp_pos = np.asarray([*(live_cup[:2] + jitter), grasp_z], dtype=np.float64)
        for _ in range(GRASP_STEPS):
            live_cup = np.asarray(
                action_eval.shell_main._cup_positions(shell, env)[target_identity],
                dtype=np.float64,
            )
            grasp_pos[:2] = live_cup[:2] + jitter
            execute("grasp", legacy.PHASE_GRASP, grasp_pos, close=True)
        for step in range(lift_steps):
            progress = legacy._cosine_progress(step, lift_steps)
            live_cup = np.asarray(
                action_eval.shell_main._cup_positions(shell, env)[target_identity],
                dtype=np.float64,
            )
            lift_pos = np.asarray(
                [*(live_cup[:2] + jitter), grasp_z + progress * args.lift_height_m],
                dtype=np.float64,
            )
            execute("lift", legacy.PHASE_LIFT, lift_pos, close=True)

        success, success_stats = action_eval.shell_main._success(
            shell,
            env,
            target_identity,
            scene["settle_cup_pos"],
            policy_args.lift_success_height,
        )
        if not success:
            return result | {"reason": "oracle_suffix_failed", "success_stats": success_stats}
        if len(observations) != 156:
            raise RuntimeError(f"Expected 156 stored rows, got {len(observations)}")

        raw_metadata = {
            "env": "ShellGame",
            "dataset_kind": DATASET_KIND,
            "source_episode": episode,
            "seed": int(command_args["seed"]),
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "initial_ball_cup": str(metadata["initial_ball_cup"]),
            "target_cup_identity": target_identity,
            "final_ball_cup": final_slot,
            "swaps": metadata["swaps"],
            "phase_ranges": _phase_ranges(open_steps, lift_steps),
            "num_frames": 156,
            "success": True,
            "osc_input_type": "absolute",
            "action_representation": "controller",
            "model_prefix": {
                "checkpoint_label": "current_action_waypoint_grasp_v6_step2000",
                "executed_steps": prefix_steps,
                "replan_steps": args.replan_steps,
                "semantic_memory_source": "direct_visual_step999",
                "actions_saved": False,
                "intermediate_frames_saved": False,
            },
            "switch": {
                "selected_cup": selected,
                "target_cup": target_identity,
                "offset_m": offset_m,
                "safe_height_m": safe_height_m,
                "target_lift_m": target_lift_m,
            },
            "oracle": {
                "open_steps": open_steps,
                "grasp_steps": GRASP_STEPS,
                "lift_steps": lift_steps,
                "sustained_live_cup_xy": True,
                "gated_z_descent": True,
                "closure_requires_measured_xy_z": True,
                "continuous_lift": True,
                "final_success_stats": success_stats,
                "max_grasp_lift_xy_error_m": float(
                    max(
                        item["eef_to_target_xy_m"]
                        for item in stage_metrics
                        if item["stage"] in {"grasp", "lift"}
                    )
                ),
            },
            "supervision_contract": {
                "stored_frames_0_59": "recorded_scripted_history_context_only",
                "stored_frame_60": "post_current_policy_prefix_context_only",
                "stored_frames_61_plus": "coherent_oracle_suffix",
                "first_training_pair": "observation[60] -> oracle_action[61]",
                "model_generated_actions_supervised": False,
                "model_generated_frames_supervised": False,
                "full_consecutive_horizon_required": 16,
            },
            "render_retries": render_retries,
            "command_args": {
                **command_args,
                "action_representation": "controller",
                "osc_input_type": "absolute",
            },
        }
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        legacy._save_episode(
            temp_dir,
            observations=observations,
            actions=actions,
            action_mask=action_mask,
            phase_ids=phase_ids,
            supervision_source=supervision,
            metadata=raw_metadata,
            initial_ball_cup=str(metadata["initial_ball_cup"]),
            final_ball_cup=final_slot,
            fps=args.fps,
        )
        return result | {
            "reason": "accepted",
            "open_steps": open_steps,
            "lift_steps": lift_steps,
            "candidate_dir": str(temp_dir),
            "max_grasp_lift_xy_error_m": raw_metadata["oracle"]["max_grasp_lift_xy_error_m"],
        }
    except Exception as error:  # preserve candidate-level progress
        logging.exception("candidate episode=%d prefix=%d failed", episode, prefix_steps)
        return result | {"reason": "exception", "error": repr(error)}
    finally:
        env.close()


def _candidate_tasks(args: argparse.Namespace) -> list[tuple[int, int]]:
    excluded = set(_parse_ints(args.exclude_episodes))
    prefixes = _parse_ints(args.prefix_steps)
    direct = paired._load_direct(Path(args.direct_memory))
    labels = np.asarray(direct["label"])
    predictions = np.asarray(direct["prediction"])
    candidates: dict[str, list[int]] = {slot: [] for slot in SLOTS}
    for episode_dir in sorted(Path(args.raw_root).glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")):
        episode = int(episode_dir.name.split("_")[-1])
        if episode in excluded or episode >= len(predictions):
            continue
        if int(predictions[episode]) != int(labels[episode]):
            continue
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        candidates[str(metadata["final_ball_cup"])].append(episode)
    rng = np.random.default_rng(args.dataset_seed)
    for values in candidates.values():
        rng.shuffle(values)
    tasks = []
    slot_cursor = dict.fromkeys(SLOTS, 0)
    for round_index in range(args.max_candidates // len(SLOTS)):
        prefix = prefixes[round_index % len(prefixes)]
        for slot in SLOTS:
            cursor = slot_cursor[slot]
            if cursor < len(candidates[slot]):
                tasks.append((candidates[slot][cursor], prefix))
                slot_cursor[slot] += 1
            if len(tasks) >= args.max_candidates:
                return tasks
    return tasks


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.direct_memory = args.direct_memory.expanduser().resolve()
    _validate_args(args)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = _candidate_tasks(args)
    if len(tasks) < args.num_episodes:
        raise RuntimeError(f"Only {len(tasks)} candidate tasks available")
    prefixes = _parse_ints(args.prefix_steps)
    quota_each = args.num_episodes // (len(SLOTS) * len(prefixes))
    accepted = Counter()
    records: list[dict[str, Any]] = []
    next_index = 0
    started = time.monotonic()
    context = mp.get_context("spawn")
    args_dict = {**vars(args), "output": str(args.output), "raw_root": str(args.raw_root), "direct_memory": str(args.direct_memory)}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(args_dict,),
    ) as executor:
        task_iter = iter(tasks)
        pending: dict[concurrent.futures.Future, tuple[int, int]] = {}
        for _ in range(args.workers * 2):
            try:
                task = next(task_iter)
            except StopIteration:
                break
            pending[executor.submit(_attempt, task)] = task
        while pending and next_index < args.num_episodes:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                task = pending.pop(future)
                record = future.result()
                key = (record["final_spatial_slot"], str(record["prefix_steps"]))
                accepted_now = record.get("reason") == "accepted" and accepted[key] < quota_each
                candidate_dir = Path(record["candidate_dir"]) if record.get("candidate_dir") else None
                if accepted_now:
                    destination = args.output / f"episode_{next_index:06d}"
                    candidate_dir.rename(destination)
                    record["accepted_index"] = next_index
                    record["output_episode"] = str(destination)
                    accepted[key] += 1
                    next_index += 1
                elif candidate_dir is not None and candidate_dir.exists():
                    shutil.rmtree(candidate_dir)
                    if record.get("reason") == "accepted":
                        record["reason"] = "quota_full"
                records.append(record)
                with (args.output / "generation_manifest.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                logging.info(
                    "task=%s reason=%s accepted=%d/%d bins=%s elapsed=%.1fmin",
                    task,
                    record.get("reason"),
                    next_index,
                    args.num_episodes,
                    dict(accepted),
                    (time.monotonic() - started) / 60.0,
                )
                if next_index < args.num_episodes:
                    try:
                        new_task = next(task_iter)
                    except StopIteration:
                        continue
                    pending[executor.submit(_attempt, new_task)] = new_task
        for future in pending:
            future.cancel()

    # Futures that were already running cannot be cancelled by the executor.
    # They may finish after the requested quota and leave private candidate
    # directories whose trajectories must never enter conversion.
    stale_candidates = list(args.output.glob(".candidate_*"))
    for candidate_dir in stale_candidates:
        if candidate_dir.is_dir():
            shutil.rmtree(candidate_dir)
    if stale_candidates:
        logging.info("Removed %d post-quota candidate directories", len(stale_candidates))

    summary = {
        "dataset_kind": DATASET_KIND,
        "requested_episodes": args.num_episodes,
        "accepted_episodes": next_index,
        "balanced_bins": {f"{slot}:{prefix}": accepted[(slot, str(prefix))] for slot in SLOTS for prefix in prefixes},
        "reasons": dict(Counter(record.get("reason", "unknown") for record in records)),
        "excluded_evaluation_episodes": sorted(set(_parse_ints(args.exclude_episodes))),
        "training_contract": {
            "current_policy_prefix": True,
            "direct_semantic_memory": True,
            "model_generated_actions_stored": False,
            "model_generated_actions_supervised": False,
            "coherent_oracle_suffix": True,
            "first_training_pair": "observation[60] -> oracle_action[61]",
            "full_consecutive_horizon": 16,
        },
        "elapsed_s": time.monotonic() - started,
        "settings": {**vars(args), "output": str(args.output), "raw_root": str(args.raw_root), "direct_memory": str(args.direct_memory)},
    }
    (args.output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if next_index != args.num_episodes:
        raise RuntimeError(f"Generated only {next_index}/{args.num_episodes}: {summary}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
