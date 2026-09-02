"""Generate on-policy absolute-EEF correction demonstrations.

Each accepted raw episode contains only:

* source frames 0..59: the normal three-swap visual history;
* stored frame 60: the real observation after a 30-step learned-policy approach;
* stored frames 61+: GT-centered recenter / descend / grasp / vertical-lift actions.

The learned approach commands and their intermediate frames are deliberately
NOT written to ``vla_trajectory.npz``.  Consequently, observation 60 aligns to
Oracle action 61, and every row marked for action supervision has an Oracle
target.  The learned policy is used only to visit its real deployment-state
distribution.
"""

# Private helpers are reused intentionally so rendering, policy preprocessing,
# controller semantics, and success checks exactly match the validated eval.
# ruff: noqa: FBT001, FBT003, SLF001

from __future__ import annotations

import argparse
from collections import deque
import dataclasses
import json
import logging
import math
from pathlib import Path
import shutil
import time

import main as base
import main_absolute_eef_fixed_history as fixed_eef
import numpy as np
import oracle_joint_noise_sensitivity as contact_utils

OBSERVE_PROMPT = "Observe the ball moving under a cup and remember which cup contains it."
GRASP_PROMPT = "The shell game has ended. Grasp and lift the cup containing the ball."

HISTORY_FRAMES = 60
MODEL_APPROACH_STEPS = 30
PHASE_CORRECTION_START = 8
PHASE_RECENTER = 8
PHASE_DESCEND = 9
PHASE_GRASP = 10
PHASE_LIFT = 11
SUPERVISION_CONTEXT = 0
SUPERVISION_ORACLE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=750)
    parser.add_argument("--dataset-seed", type=int, default=260814)
    parser.add_argument("--replan-steps", type=int, default=3)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-offset-mm", type=float, default=3.0)
    parser.add_argument("--max-offset-mm", type=float, default=35.0)
    parser.add_argument("--min-safe-height-mm", type=float, default=80.0)
    parser.add_argument("--min-open-width-m", type=float, default=0.04)
    parser.add_argument("--recenter-steps", type=int, default=10)
    parser.add_argument("--descend-steps", type=int, default=30)
    parser.add_argument("--grasp-steps", type=int, default=15)
    parser.add_argument("--lift-steps", type=int, default=40)
    parser.add_argument("--hover-height", type=float, default=0.22)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _policy_args(args: argparse.Namespace) -> base.Args:
    return base.Args(
        host=args.host,
        port=args.port,
        robosuite_root=args.robosuite_root,
        replan_steps=args.replan_steps,
        width=args.width,
        height=args.height,
        resize_size=args.width,
        fps=args.fps,
        initial_ball_cup="random",
        min_swaps=3,
        max_swaps=3,
        num_frames=fixed_eef.TOTAL_FRAMES,
        frame_stride=1,
        action_horizon=16,
        action_dim=7,
        policy_input_mode="history",
        action_mode="raw7",
        phase_instructions=True,
        observe_task=OBSERVE_PROMPT,
        grasp_task=GRASP_PROMPT,
        task=GRASP_PROMPT,
        observation_position_frame="absolute",
        osc_input_type="absolute",
        control_during_scripted_observation=False,
        observe_eef_frames=0,
        save_videos=False,
    )


def _episode_randomness(dataset_seed: int, attempt_index: int, cups) -> tuple[int, str]:
    rng = np.random.default_rng(np.random.SeedSequence([dataset_seed, attempt_index]))
    return int(rng.integers(0, 2**31 - 1)), str(rng.choice(cups))


def _cosine_progress(index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    linear = index / (count - 1)
    return 0.5 - 0.5 * math.cos(math.pi * linear)


def _select_cup(votes: dict[str, int], distances: dict[str, float]) -> str | None:
    best = max(votes.values(), default=0)
    if best <= 0:
        return None
    candidates = [cup for cup, count in votes.items() if count == best]
    return min(candidates, key=lambda cup: distances[cup] / votes[cup])


def _snapshot(history_item: dict) -> dict:
    return {
        "base": np.asarray(history_item["base"], dtype=np.uint8),
        "wrist": np.asarray(history_item["wrist"], dtype=np.uint8),
        "eef_pos": np.asarray(history_item["eef_pos"], dtype=np.float32),
        "eef_quat": np.asarray(history_item["eef_quat"], dtype=np.float32),
        "gripper_width": float(history_item["gripper_width"]),
    }


def _placeholder_absolute_action(shell, observation: dict) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2axisangle

    return np.concatenate(
        (
            np.asarray(observation["eef_pos"], dtype=np.float32),
            quat2axisangle(np.asarray(observation["eef_quat"], dtype=np.float64)).astype(np.float32),
            np.array([-1.0], dtype=np.float32),
        )
    )


def _phase_ranges(args: argparse.Namespace) -> dict[str, list[int]]:
    cursor = HISTORY_FRAMES
    ranges = {
        "reveal": [0, 9],
        "cover": [10, 19],
        "swap_0": [20, 29],
        "swap_1": [30, 39],
        "swap_2": [40, 49],
        "settle": [50, 59],
        "correction_start": [cursor, cursor],
    }
    for name, count in (
        ("oracle_recenter", args.recenter_steps),
        ("oracle_descend", args.descend_steps),
        ("oracle_grasp", args.grasp_steps),
        ("oracle_lift", args.lift_steps),
    ):
        start = cursor + 1
        cursor += count
        ranges[name] = [start, cursor]
    return ranges


def _save_episode(
    episode_dir: Path,
    *,
    observations: list[dict],
    actions: list[np.ndarray],
    action_mask: list[bool],
    phase_ids: list[int],
    supervision_source: list[int],
    metadata: dict,
    initial_ball_cup: str,
    final_ball_cup: str,
    fps: int,
) -> None:
    t = len(observations)
    lengths = {len(actions), len(action_mask), len(phase_ids), len(supervision_source), t}
    if len(lengths) != 1:
        raise RuntimeError(f"Raw episode length mismatch: {lengths}")
    supervision = np.asarray(supervision_source, dtype=np.uint8)
    mask = np.asarray(action_mask, dtype=bool)
    if np.any(mask != (supervision == SUPERVISION_ORACLE)):
        raise RuntimeError("action_mask must exactly equal Oracle supervision_source")
    if np.any(mask[: HISTORY_FRAMES + 1]) or not np.all(mask[HISTORY_FRAMES + 1 :]):
        raise RuntimeError("Expected context rows 0..60 and Oracle-only actions from row 61")

    third = np.stack([item["base"] for item in observations]).astype(np.uint8)
    wrist = np.stack([item["wrist"] for item in observations]).astype(np.uint8)
    eef_pos = np.stack([item["eef_pos"] for item in observations]).astype(np.float32)
    eef_quat = np.stack([item["eef_quat"] for item in observations]).astype(np.float32)
    gripper = np.asarray([[item["gripper_width"]] for item in observations], dtype=np.float32)
    controller_actions = np.stack(actions).astype(np.float32)
    phases = np.asarray(phase_ids, dtype=np.int16)
    instructions = np.where(phases >= PHASE_CORRECTION_START, GRASP_PROMPT, OBSERVE_PROMPT)

    tmp_dir = episode_dir.with_name(f".{episode_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    np.savez_compressed(
        tmp_dir / "vla_trajectory.npz",
        third_person_images=third,
        wrist_images=wrist,
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        gripper_state=gripper,
        actions=controller_actions,
        controller_actions=controller_actions,
        action_mask=mask,
        supervision_source=supervision,
        timestamps=np.arange(t, dtype=np.float32) / float(fps),
        phase_ids=phases,
        instruction=np.asarray(GRASP_PROMPT),
        instructions=instructions,
        final_ball_cup=np.asarray(final_ball_cup),
        initial_ball_cup=np.asarray(initial_ball_cup),
    )
    np.savez_compressed(
        tmp_dir / "trajectory.npz",
        actions=controller_actions,
        controller_actions=controller_actions,
        timestamps=np.arange(t, dtype=np.float32) / float(fps),
    )
    (tmp_dir / "label.txt").write_text(final_ball_cup + "\n", encoding="utf-8")
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tmp_dir.rename(episode_dir)


def _attempt(
    shell,
    client,
    args: argparse.Namespace,
    policy_args: base.Args,
    *,
    attempt_index: int,
    accepted_index: int,
) -> tuple[dict, tuple | None]:
    episode_seed, initial = _episode_randomness(args.dataset_seed, attempt_index, shell.CUP_NAMES)
    ep_args = base._episode_namespace(policy_args, seed=episode_seed, initial_ball_cup=initial, num_swaps=3)
    env = shell.make_env(ep_args)
    history: list[dict] = []
    discarded_video: list[np.ndarray] = []
    reject: dict = {"attempt_index": attempt_index, "episode_seed": episode_seed, "initial_ball_cup": initial}

    try:
        meta = base._run_scripted_observation(
            shell, env, ep_args, policy_args, history, discarded_video, client=client
        )
        if len(history) != HISTORY_FRAMES:
            raise RuntimeError(f"Expected {HISTORY_FRAMES} scripted frames, got {len(history)}")
        fixed_history = [_snapshot(item) for item in history]
        canonical_quat = np.asarray(history[0]["eef_quat"], dtype=np.float32)
        start_eef_pos = np.asarray(history[0]["eef_pos"], dtype=np.float32)
        start_eef_quat = canonical_quat.copy()
        action_plan: deque = deque()
        gripper_action = policy_args.default_gripper_action
        votes = dict.fromkeys(shell.CUP_NAMES, 0)
        distance_sums = dict.fromkeys(shell.CUP_NAMES, 0.0)

        # Execute the learned prefix but never retain its actions or intermediate
        # observations in the raw training episode.
        for step in range(MODEL_APPROACH_STEPS):
            env_action, gripper_action, _ = base._policy_env_action(
                shell,
                env,
                history,
                start_eef_pos,
                start_eef_quat,
                action_plan,
                gripper_action,
                client=client,
                args=policy_args,
                prompt=GRASP_PROMPT,
            )
            env.step(env_action)
            sink: list[np.ndarray] = []
            base._append_observation(
                shell,
                env,
                ep_args,
                meta["wrist_camera_name"],
                history,
                sink,
                resize_size=args.width,
            )
            eef = np.asarray(history[-1]["eef_pos"], dtype=np.float64)
            cups = base._cup_positions(shell, env)
            distances = {
                cup: float(np.linalg.norm(eef[:2] - np.asarray(pos[:2], dtype=np.float64)))
                for cup, pos in cups.items()
            }
            nearest = min(distances, key=distances.get)
            if step >= 10 and distances[nearest] <= policy_args.cup_selection_xy_radius:
                votes[nearest] += 1
                distance_sums[nearest] += distances[nearest]

        switch = _snapshot(history[-1])
        cups = base._cup_positions(shell, env)
        selected = _select_cup(votes, distance_sums)
        reject.update({"target_cup": meta["target_cup"], "selected_cup": selected, "selection_votes": votes})
        if selected is None:
            reject["reason"] = "no_selection_vote"
            return reject, None
        if selected != meta["target_cup"]:
            reject["reason"] = "selection_incorrect"
            return reject, None

        selected_pos = np.asarray(cups[selected], dtype=np.float64)
        switch_eef = np.asarray(switch["eef_pos"], dtype=np.float64)
        offset_m = float(np.linalg.norm(switch_eef[:2] - selected_pos[:2]))
        grasp_z = float(env.cup_handle_grasp_z() + ep_args.robot_grasp_z_offset)
        safe_height_m = float(switch_eef[2] - grasp_z)
        switch_contacts = contact_utils._finger_contact_count(env, selected)
        reject.update(
            {
                "switch_offset_m": offset_m,
                "switch_safe_height_m": safe_height_m,
                "switch_gripper_width_m": switch["gripper_width"],
                "switch_contacts": switch_contacts,
            }
        )
        if offset_m < args.min_offset_mm / 1000.0:
            reject["reason"] = "offset_too_small"
            return reject, None
        if offset_m > args.max_offset_mm / 1000.0:
            reject["reason"] = "offset_too_large"
            return reject, None
        if safe_height_m < args.min_safe_height_mm / 1000.0:
            reject["reason"] = "unsafe_switch_height"
            return reject, None
        if switch["gripper_width"] < args.min_open_width_m:
            reject["reason"] = "gripper_not_open"
            return reject, None
        if switch_contacts:
            reject["reason"] = "contact_before_oracle"
            return reject, None

        observations = [*fixed_history, switch]
        actions = [_placeholder_absolute_action(shell, item) for item in observations]
        action_mask = [False] * len(observations)
        supervision = [SUPERVISION_CONTEXT] * len(observations)
        phase_ids = [0] * 10 + [1] * 10 + [2] * 30 + [3] * 10 + [PHASE_CORRECTION_START]
        stage_metrics: list[dict] = []

        hover_pos = np.array(
            [selected_pos[0], selected_pos[1], grasp_z + args.hover_height], dtype=np.float64
        )
        grasp_pos = np.array([selected_pos[0], selected_pos[1], grasp_z], dtype=np.float64)
        lift_pos = np.array(
            [selected_pos[0], selected_pos[1], grasp_z + args.lift_height], dtype=np.float64
        )

        def execute(stage: str, phase_id: int, stage_step: int, target_pos: np.ndarray, close: bool) -> None:
            command = shell.make_robot_action(
                env,
                target_pos=np.asarray(target_pos, dtype=np.float64),
                target_quat=canonical_quat,
                gripper_action=1.0 if close else -1.0,
            )
            env.step(command)
            sink: list[np.ndarray] = []
            base._append_observation(
                shell,
                env,
                ep_args,
                meta["wrist_camera_name"],
                history,
                sink,
                resize_size=args.width,
            )
            observation = _snapshot(history[-1])
            observations.append(observation)
            actions.append(np.asarray(command, dtype=np.float32))
            action_mask.append(True)
            supervision.append(SUPERVISION_ORACLE)
            phase_ids.append(phase_id)
            actual = np.asarray(observation["eef_pos"], dtype=np.float64)
            live_cup = np.asarray(base._cup_positions(shell, env)[selected], dtype=np.float64)
            stage_metrics.append(
                {
                    "stage": stage,
                    "stage_step": stage_step,
                    "eef_to_selected_xy_m": float(np.linalg.norm(actual[:2] - live_cup[:2])),
                    "selected_finger_contacts": contact_utils._finger_contact_count(env, selected),
                }
            )

        recenter_start = switch_eef.copy()
        for step in range(args.recenter_steps):
            progress = _cosine_progress(step, args.recenter_steps)
            execute(
                "recenter",
                PHASE_RECENTER,
                step,
                (1.0 - progress) * recenter_start + progress * hover_pos,
                False,
            )
        for step in range(args.descend_steps):
            progress = _cosine_progress(step, args.descend_steps)
            execute(
                "descend",
                PHASE_DESCEND,
                step,
                (1.0 - progress) * hover_pos + progress * grasp_pos,
                False,
            )
        for step in range(args.grasp_steps):
            execute("grasp", PHASE_GRASP, step, grasp_pos, True)
        for step in range(args.lift_steps):
            progress = _cosine_progress(step, args.lift_steps)
            execute(
                "lift",
                PHASE_LIFT,
                step,
                (1.0 - progress) * grasp_pos + progress * lift_pos,
                True,
            )

        success, success_stats = base._success(
            shell, env, meta["target_cup"], meta["settle_cup_pos"], policy_args.lift_success_height
        )
        if not success:
            reject["reason"] = "oracle_suffix_failed"
            reject["success_stats"] = success_stats
            return reject, None

        metadata = {
            "env": "ShellGame",
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "seed": episode_seed,
            "attempt_index": attempt_index,
            "accepted_index": accepted_index,
            "initial_ball_cup": initial,
            "target_cup_identity": meta["target_cup"],
            "final_ball_cup": meta["final_ball_cup"],
            "swaps": [list(pair) for pair in meta["swaps"]],
            "phase_ranges": _phase_ranges(args),
            "num_frames": len(observations),
            "success": True,
            "failure_mode": "success",
            "osc_input_type": "absolute",
            "action_representation": "controller",
            "dataset_kind": "onpolicy_offset_oracle_correction",
            "model_prefix": {
                "executed_steps": MODEL_APPROACH_STEPS,
                "actions_saved_to_training_file": False,
                "intermediate_frames_saved_to_training_file": False,
                "purpose": "visit real learned-policy offset state only",
            },
            "supervision_contract": {
                "stored_frames_0_59": "scripted_history_context_only",
                "stored_frame_60": "post_model_approach_offset_observation_context_only",
                "stored_frames_61_plus": "observations_after_oracle_commands",
                "action_mask_true_source": "oracle_only",
                "first_training_pair": "observation[60] -> oracle_action[61]",
            },
            "switch": {
                "selected_cup": selected,
                "target_cup": meta["target_cup"],
                "selection_votes": votes,
                "offset_m": offset_m,
                "safe_height_m": safe_height_m,
                "gripper_width_m": switch["gripper_width"],
                "contacts": switch_contacts,
            },
            "oracle": {
                "recenter_steps": args.recenter_steps,
                "descend_steps": args.descend_steps,
                "grasp_steps": args.grasp_steps,
                "lift_steps": args.lift_steps,
                "hover_height_m": args.hover_height,
                "lift_height_m": args.lift_height,
                "final_success_stats": success_stats,
                "mean_grasp_lift_xy_error_m": float(
                    np.mean(
                        [
                            item["eef_to_selected_xy_m"]
                            for item in stage_metrics
                            if item["stage"] in {"grasp", "lift"}
                        ]
                    )
                ),
                "max_bilateral_contact_run": _max_contact_run(stage_metrics),
            },
            "command_args": {
                **dataclasses.asdict(policy_args),
                "seed": episode_seed,
                "initial_ball_cup": initial,
                "num_swaps": 3,
                "action_representation": "controller",
                "osc_input_type": "absolute",
            },
        }
        reject["reason"] = "accepted"
        return reject, (
            observations,
            actions,
            action_mask,
            phase_ids,
            supervision,
            metadata,
            initial,
            meta["final_ball_cup"],
        )
    finally:
        env.close()


def _max_contact_run(metrics: list[dict]) -> int:
    best = 0
    current = 0
    for item in metrics:
        if item["selected_finger_contacts"] == 2:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def main() -> None:
    args = parse_args()
    if args.width != args.height:
        raise ValueError("Current fixed-history policy expects square images")
    if args.num_episodes <= 0 or args.max_attempts < args.num_episodes:
        raise ValueError("Require 0 < num_episodes <= max_attempts")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "generation_manifest.jsonl"
    existing = sorted(output.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if existing and not args.resume:
        raise FileExistsError(f"{output} already contains {len(existing)} episodes")
    accepted = len(existing)
    attempted_indices: set[int] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                attempted_indices.add(int(json.loads(line)["attempt_index"]))

    base._policy_input = fixed_eef._fixed_history_policy_input
    from openpi_client import websocket_client_policy

    logging.basicConfig(level=logging.INFO, force=True)
    shell = base._import_shellgame_tools(args.robosuite_root)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    policy_args = _policy_args(args)
    start_time = time.time()

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for attempt_index in range(args.max_attempts):
            if accepted >= args.num_episodes:
                break
            if attempt_index in attempted_indices:
                continue
            audit, payload = _attempt(
                shell,
                client,
                args,
                policy_args,
                attempt_index=attempt_index,
                accepted_index=accepted,
            )
            if payload is not None:
                episode_dir = output / f"episode_{accepted:06d}"
                if episode_dir.exists():
                    raise FileExistsError(episode_dir)
                (
                    observations,
                    actions,
                    action_mask,
                    phase_ids,
                    supervision,
                    metadata,
                    initial,
                    final_ball_cup,
                ) = payload
                _save_episode(
                    episode_dir,
                    observations=observations,
                    actions=actions,
                    action_mask=action_mask,
                    phase_ids=phase_ids,
                    supervision_source=supervision,
                    metadata=metadata,
                    initial_ball_cup=initial,
                    final_ball_cup=final_ball_cup,
                    fps=args.fps,
                )
                audit["accepted_index"] = accepted
                accepted += 1
            manifest.write(json.dumps(audit, sort_keys=True) + "\n")
            manifest.flush()
            elapsed = time.time() - start_time
            logging.info(
                "attempt=%d reason=%s accepted=%d/%d offset=%s elapsed=%.1fmin",
                attempt_index,
                audit["reason"],
                accepted,
                args.num_episodes,
                None
                if "switch_offset_m" not in audit
                else f"{audit['switch_offset_m'] * 1000.0:.1f}mm",
                elapsed / 60.0,
            )

    summary = _summarize(output, args, accepted)
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if accepted < args.num_episodes:
        raise RuntimeError(
            f"Generated only {accepted}/{args.num_episodes} accepted episodes after {args.max_attempts} attempts"
        )
    logging.info("Generation complete: %s", json.dumps(summary, sort_keys=True))


def _summarize(output: Path, args: argparse.Namespace, accepted: int) -> dict:
    rows = []
    manifest = output / "generation_manifest.jsonl"
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    reasons: dict[str, int] = {}
    for row in rows:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    offsets = [row["switch_offset_m"] for row in rows if row["reason"] == "accepted"]
    return {
        "dataset_kind": "onpolicy_offset_oracle_correction",
        "output": str(output),
        "requested_episodes": args.num_episodes,
        "accepted_episodes": accepted,
        "attempts": len(rows),
        "reasons": reasons,
        "accepted_offset_m": None
        if not offsets
        else {
            "min": float(np.min(offsets)),
            "mean": float(np.mean(offsets)),
            "median": float(np.median(offsets)),
            "max": float(np.max(offsets)),
        },
        "training_contract": {
            "model_generated_actions_stored": False,
            "model_generated_actions_supervised": False,
            "first_supervised_observation_frame": 60,
            "first_supervised_target_action_source_frame": 61,
            "supervised_action_source": "oracle_only",
        },
        "settings": vars(args) | {"output": str(output)},
    }


if __name__ == "__main__":
    main()
