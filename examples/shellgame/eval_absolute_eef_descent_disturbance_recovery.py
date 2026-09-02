"""Measure closed-loop visual-servo recovery from lateral descent disturbances.

The normal fixed-history policy selects and approaches the target cup.  Once
the measured EEF reaches a configured height above that cup, the evaluator
temporarily replaces policy control with an unrecorded absolute-EEF command
that creates a known lateral offset.  Cached policy actions are discarded.
The policy then receives the post-disturbance image/state and controls the next
20 steps normally.

This diagnostic measures feedback recovery, not pre-descent waypoint accuracy:
XY error is reported immediately after the disturbance and after 5/10/20
policy-control steps, together with command feedback alignment and Z progress.
Client-specified diffusion noise makes V4/V5 comparisons reproducible.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import multiprocessing as mp
import pathlib
from typing import Any

import eval_absolute_eef_xy_residual_stage_ablation as stable_eval
import main as base
import main_absolute_eef_fixed_history as fixed_eef
import numpy as np
import tyro

NOISE_SEED_KEY = "__openpi_deterministic_noise_seed__"
VALID_DIRECTIONS = {
    "pos_x": np.asarray((1.0, 0.0), dtype=np.float32),
    "neg_x": np.asarray((-1.0, 0.0), dtype=np.float32),
    "pos_y": np.asarray((0.0, 1.0), dtype=np.float32),
    "neg_y": np.asarray((0.0, -1.0), dtype=np.float32),
}
CUP_NAMES = ("left", "middle", "right")


@dataclasses.dataclass(frozen=True)
class Condition:
    stage: str
    trigger_z_above_cup_m: float
    direction: str
    magnitude_mm: int

    @property
    def name(self) -> str:
        return f"{self.stage}_{self.direction}_{self.magnitude_mm:02d}mm"


@dataclasses.dataclass
class Args(base.Args):
    num_trials: int = 20
    seed: int = 260813
    episode_index: int = 14
    video_out_path: str = "evaluation/shellgame/absolute_eef_descent_disturbance_recovery"
    physics_debug: bool = True
    physics_debug_window: int = 180
    websocket_reconnect_interval: int = 4

    stages: str = "high:0.16,mid:0.10,late:0.06"
    directions: str = "pos_x,neg_x,pos_y,neg_y"
    magnitudes_mm: str = "10,20,30"
    recovery_checkpoints: str = "5,10,20"
    recovery_success_threshold_mm: float = 8.0
    perturb_steps: int = 6
    force_gripper_open_during_recovery: bool = True
    deterministic_sample_salt: int = 260816


def _parse_stages(text: str) -> list[tuple[str, float]]:
    stages = []
    for item in text.split(","):
        name, value = item.strip().split(":", maxsplit=1)
        stages.append((name.strip(), float(value)))
    if not stages or any(not name or value <= 0 for name, value in stages):
        raise ValueError("--stages must contain positive name:value entries")
    return stages


def _parse_ints(text: str, *, name: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values or values[0] <= 0:
        raise ValueError(f"--{name} must contain positive integers")
    return values


def _conditions(args: Args) -> list[Condition]:
    directions = [item.strip() for item in args.directions.split(",") if item.strip()]
    if not directions or any(item not in VALID_DIRECTIONS for item in directions):
        raise ValueError(f"--directions must contain only {sorted(VALID_DIRECTIONS)}")
    magnitudes = _parse_ints(args.magnitudes_mm, name="magnitudes-mm")
    return [
        Condition(stage, trigger_z, direction, magnitude)
        for stage, trigger_z in _parse_stages(args.stages)
        for magnitude in magnitudes
        for direction in directions
    ]


def _episode_spec(args: Args) -> dict[str, Any]:
    if args.episode_index < 0 or args.episode_index >= args.num_trials:
        raise ValueError("--episode-index must be inside [0, num-trials)")
    rng = np.random.default_rng(args.seed)
    spec = None
    for episode_index in range(args.num_trials):
        candidate = {
            "episode_index": episode_index,
            "episode_seed": int(rng.integers(0, 2**31 - 1)),
            "initial_ball_cup": (
                str(rng.choice(CUP_NAMES)) if args.initial_ball_cup == "random" else args.initial_ball_cup
            ),
            "num_swaps": int(rng.integers(args.min_swaps, args.max_swaps + 1)),
        }
        if episode_index == args.episode_index:
            spec = candidate
            break
    if spec is None:
        raise RuntimeError("Failed to construct episode spec")
    return spec


def _clip_action(env, action: np.ndarray) -> np.ndarray:
    low, high = env.action_spec
    return np.clip(np.asarray(action, dtype=np.float32), low, high)


def _hold_action(shell, env, *, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    return _clip_action(
        env,
        shell.make_robot_action(
            env,
            target_pos=target_pos,
            target_quat=target_quat,
            gripper_action=-1.0,
        ),
    )


def _error_snapshot(shell, env, target_cup: str) -> dict[str, Any]:
    eef = np.asarray(shell.get_eef_pos(env), dtype=np.float32)
    cup = np.asarray(base._cup_positions(shell, env)[target_cup], dtype=np.float32)  # noqa: SLF001
    error_xy = eef[:2] - cup[:2]
    return {
        "eef_pos": eef.tolist(),
        "cup_pos": cup.tolist(),
        "error_xy_mm": (error_xy * 1_000.0).tolist(),
        "error_norm_mm": float(np.linalg.norm(error_xy) * 1_000.0),
        "z_above_cup_mm": float((eef[2] - cup[2]) * 1_000.0),
    }


def _finalize(state: dict[str, Any], args: Args) -> dict[str, Any]:
    checkpoints = state["checkpoint_snapshots"]
    initial = float(checkpoints["0"]["error_norm_mm"])
    reductions = {key: initial - float(value["error_norm_mm"]) for key, value in checkpoints.items()}
    step_errors = np.asarray(state["step_error_norm_mm"], dtype=np.float64)
    alignments = np.asarray(state["feedback_alignment_cosine"], dtype=np.float64)
    final_key = str(max(state["recovery_checkpoints"]))
    final_error = float(checkpoints[final_key]["error_norm_mm"])
    initial_z = float(checkpoints["0"]["z_above_cup_mm"])
    final_z = float(checkpoints[final_key]["z_above_cup_mm"])
    initial_cup_xy = np.asarray(checkpoints["0"]["cup_pos"][:2], dtype=np.float64)
    final_cup_xy = np.asarray(checkpoints[final_key]["cup_pos"][:2], dtype=np.float64)
    return {
        "condition": dataclasses.asdict(state["condition"]),
        "condition_name": state["condition"].name,
        "target_cup": state["target_cup"],
        "trigger_policy_step": state["trigger_policy_step"],
        "perturb_steps": args.perturb_steps,
        "requested_target_xy_mm": (state["requested_target_xy"] * 1_000.0).tolist(),
        "checkpoint_snapshots": checkpoints,
        "error_reduction_mm": reductions,
        "recovery_success_threshold_mm": args.recovery_success_threshold_mm,
        "recovered_within_threshold": final_error <= args.recovery_success_threshold_mm,
        "z_descent_during_recovery_mm": initial_z - final_z,
        "cup_xy_displacement_during_recovery_mm": float(np.linalg.norm(final_cup_xy - initial_cup_xy) * 1_000.0),
        "force_gripper_open_during_recovery": args.force_gripper_open_during_recovery,
        "per_step_error_decrease_fraction": (
            float(np.mean(np.diff(step_errors) < 0.0)) if len(step_errors) > 1 else None
        ),
        "feedback_alignment_positive_fraction": (float(np.mean(alignments > 0.0)) if len(alignments) else None),
        "feedback_alignment_cosine_mean": (float(np.mean(alignments)) if len(alignments) else None),
        "recovery_policy_steps": state["recovery_elapsed"],
        "completed": True,
    }


def _run_condition(args: Args, *, condition: Condition, spec: dict[str, Any]) -> None:
    from openpi_client import websocket_client_policy

    output_dir = pathlib.Path(args.video_out_path) / condition.name
    child_args = dataclasses.replace(
        args,
        num_trials=1,
        seed=0,
        initial_ball_cup=spec["initial_ball_cup"],
        min_swaps=spec["num_swaps"],
        max_swaps=spec["num_swaps"],
        video_out_path=str(output_dir),
    )
    recovery_checkpoints = _parse_ints(args.recovery_checkpoints, name="recovery-checkpoints")
    max_recovery_steps = max(recovery_checkpoints)
    state_holder: dict[str, Any] = {}
    result_holder: dict[str, Any] = {}
    noise_state = {"prefix_index": 0, "recovery_index": None, "total_calls": 0}

    original_episode_namespace = base._episode_namespace  # noqa: SLF001
    original_run_scripted = base._run_scripted_observation  # noqa: SLF001
    original_policy_action = base._policy_env_action  # noqa: SLF001
    original_success = base._success  # noqa: SLF001

    def episode_namespace(current_args, *, seed, initial_ball_cup, num_swaps):
        del seed
        return original_episode_namespace(
            current_args,
            seed=spec["episode_seed"],
            initial_ball_cup=initial_ball_cup,
            num_swaps=num_swaps,
        )

    def run_scripted_with_state(*positional, **kwargs):
        meta = original_run_scripted(*positional, **kwargs)
        env = positional[1]
        state = {
            "condition": condition,
            "mode": "seeking_trigger",
            "target_cup": meta["target_cup"],
            "policy_step": 0,
            "trigger_policy_step": None,
            "perturb_executed": 0,
            "perturb_target_pos": None,
            "perturb_target_quat": None,
            "requested_target_xy": None,
            "recovery_elapsed": 0,
            "recovery_checkpoints": recovery_checkpoints,
            "checkpoint_snapshots": {},
            "step_error_norm_mm": [],
            "feedback_alignment_cosine": [],
            "last_event": None,
        }
        env._descent_disturbance_state = state  # noqa: SLF001
        state_holder["state"] = state
        return meta

    def policy_action_with_disturbance(
        shell,
        env,
        history,
        start_eef_pos,
        start_eef_quat,
        action_plan,
        gripper_action,
        *,
        client,
        args,
        prompt=None,
    ):
        state = env._descent_disturbance_state  # noqa: SLF001
        current = _error_snapshot(shell, env, state["target_cup"])
        mode = state["mode"]

        if mode == "seeking_trigger":
            close_enough_to_target = current["error_norm_mm"] <= args.cup_selection_xy_radius * 1_000.0
            below_trigger = current["z_above_cup_mm"] <= condition.trigger_z_above_cup_m * 1_000.0
            if close_enough_to_target and below_trigger:
                cup = np.asarray(current["cup_pos"], dtype=np.float32)
                direction = VALID_DIRECTIONS[condition.direction]
                requested_xy = cup[:2] + direction * (condition.magnitude_mm / 1_000.0)
                state.update(
                    mode="perturbing",
                    trigger_policy_step=state["policy_step"],
                    perturb_target_pos=np.asarray(
                        [requested_xy[0], requested_xy[1], current["eef_pos"][2]],
                        dtype=np.float32,
                    ),
                    perturb_target_quat=np.asarray(shell.get_eef_quat(env), dtype=np.float32),
                    requested_target_xy=requested_xy,
                )
                action_plan.clear()
                mode = "perturbing"
                logging.info(
                    "disturbance trigger condition=%s step=%d z=%.1fmm pre_error=%.1fmm",
                    condition.name,
                    state["policy_step"],
                    current["z_above_cup_mm"],
                    current["error_norm_mm"],
                )

        if mode == "perturbing":
            if state["perturb_executed"] >= args.perturb_steps:
                state["mode"] = "recovering"
                state["recovery_elapsed"] = 0
                noise_state["recovery_index"] = 0
                action_plan.clear()
                mode = "recovering"
                current = _error_snapshot(shell, env, state["target_cup"])
                logging.info(
                    "disturbance complete condition=%s achieved_error=%.1fmm z=%.1fmm",
                    condition.name,
                    current["error_norm_mm"],
                    current["z_above_cup_mm"],
                )
            else:
                action_plan.clear()
                state["perturb_executed"] += 1
                state["last_event"] = {
                    "mode": "perturbing",
                    "executed": state["perturb_executed"],
                    "snapshot": current,
                }
                state["policy_step"] += 1
                return (
                    _hold_action(
                        shell,
                        env,
                        target_pos=state["perturb_target_pos"],
                        target_quat=state["perturb_target_quat"],
                    ),
                    -1.0,
                    None,
                )

        if mode == "recovering":
            elapsed = int(state["recovery_elapsed"])
            current = _error_snapshot(shell, env, state["target_cup"])
            state["step_error_norm_mm"].append(current["error_norm_mm"])
            if elapsed == 0 or elapsed in recovery_checkpoints:
                state["checkpoint_snapshots"][str(elapsed)] = current
            if elapsed >= max_recovery_steps:
                state["mode"] = "done"
                result_holder["result"] = _finalize(state, args)
                state["last_event"] = {
                    "mode": "done",
                    "recovery_elapsed": elapsed,
                    "snapshot": current,
                }
                action_plan.clear()
                state["policy_step"] += 1
                return (
                    _hold_action(
                        shell,
                        env,
                        target_pos=np.asarray(current["eef_pos"], dtype=np.float32),
                        target_quat=np.asarray(shell.get_eef_quat(env), dtype=np.float32),
                    ),
                    -1.0,
                    None,
                )

            env_action, next_gripper, policy_action = original_policy_action(
                shell,
                env,
                history,
                start_eef_pos,
                start_eef_quat,
                action_plan,
                gripper_action,
                client=client,
                args=args,
                prompt=prompt,
            )
            if args.force_gripper_open_during_recovery:
                env_action = np.asarray(env_action, dtype=np.float32).copy()
                env_action[-1] = -1.0
                next_gripper = -1.0
            if policy_action is not None:
                actual_xy = np.asarray(current["eef_pos"][:2], dtype=np.float32)
                cup_xy = np.asarray(current["cup_pos"][:2], dtype=np.float32)
                error = actual_xy - cup_xy
                command_delta = np.asarray(env_action[:2], dtype=np.float32) - actual_xy
                denom = float(np.linalg.norm(error) * np.linalg.norm(command_delta))
                cosine = float(np.dot(-error, command_delta) / denom) if denom > 1e-9 else 0.0
                state["feedback_alignment_cosine"].append(cosine)
            state["recovery_elapsed"] += 1
            state["last_event"] = {
                "mode": "recovering",
                "recovery_elapsed_before_action": elapsed,
                "snapshot": current,
                "policy_action": None if policy_action is None else np.asarray(policy_action).tolist(),
            }
            state["policy_step"] += 1
            return env_action, next_gripper, policy_action

        env_action, next_gripper, policy_action = original_policy_action(
            shell,
            env,
            history,
            start_eef_pos,
            start_eef_quat,
            action_plan,
            gripper_action,
            client=client,
            args=args,
            prompt=prompt,
        )
        state["last_event"] = {"mode": "seeking_trigger", "snapshot": current}
        state["policy_step"] += 1
        return env_action, next_gripper, policy_action

    def success_when_complete(shell, env, target_cup, settle_cup_pos, lift_height):
        state = getattr(env, "_descent_disturbance_state", None)
        if state is not None and state["mode"] == "done":
            return True, {
                "lifts": dict.fromkeys(("left", "middle", "right"), 0.0),
                "target_lift": 0.0,
                "max_other_lift": 0.0,
                "diagnostic_stop": True,
            }
        ok, stats = original_success(shell, env, target_cup, settle_cup_pos, lift_height)
        return False, stats | {"underlying_task_success": bool(ok)}

    original_client = websocket_client_policy.WebsocketClientPolicy

    class DeterministicClient:
        def __init__(self, host="0.0.0.0", port=None, api_key=None):
            self._host = host
            self._port = port
            self._api_key = api_key
            self._client = original_client(host, port, api_key)

        def _reconnect(self):
            self._client._ws.close()  # noqa: SLF001
            self._client = original_client(self._host, self._port, self._api_key)

        def infer(self, observation):
            total_calls = int(noise_state["total_calls"])
            if total_calls and total_calls % args.websocket_reconnect_interval == 0:
                self._reconnect()
            recovery_index = noise_state["recovery_index"]
            if recovery_index is None:
                seed_parts = [
                    args.deterministic_sample_salt,
                    int(spec["episode_seed"]),
                    0,
                    int(noise_state["prefix_index"]),
                ]
                noise_state["prefix_index"] += 1
            else:
                direction_index = tuple(VALID_DIRECTIONS).index(condition.direction)
                stage_index = [name for name, _ in _parse_stages(args.stages)].index(condition.stage)
                seed_parts = [
                    args.deterministic_sample_salt,
                    int(spec["episode_seed"]),
                    1,
                    stage_index,
                    direction_index,
                    condition.magnitude_mm,
                    int(recovery_index),
                ]
                noise_state["recovery_index"] += 1
            sample_seed = int(np.random.SeedSequence(seed_parts).generate_state(1, dtype=np.uint32)[0])
            noise_state["total_calls"] += 1
            return self._client.infer({**observation, NOISE_SEED_KEY: sample_seed})

        def get_server_metadata(self):
            return self._client.get_server_metadata()

        def reset(self):
            return self._client.reset()

    base._episode_namespace = episode_namespace  # noqa: SLF001
    base._run_scripted_observation = run_scripted_with_state  # noqa: SLF001
    base._policy_input = fixed_eef._fixed_history_policy_input  # noqa: SLF001
    base._policy_env_action = policy_action_with_disturbance  # noqa: SLF001
    base._append_observation = stable_eval._append_observation_orientation_stable  # noqa: SLF001
    base._success = success_when_complete  # noqa: SLF001
    websocket_client_policy.WebsocketClientPolicy = DeterministicClient

    logging.basicConfig(level=logging.INFO, force=True)
    logging.info(
        "descent disturbance condition=%s episode=%d seed=%d",
        condition.name,
        spec["episode_index"],
        spec["episode_seed"],
    )
    base.eval_shellgame(child_args)

    state = state_holder.get("state")
    result = result_holder.get("result")
    if result is None:
        result = {
            "condition": dataclasses.asdict(condition),
            "condition_name": condition.name,
            "completed": False,
            "failure_mode": "trigger_not_reached"
            if state is None or state["trigger_policy_step"] is None
            else "recovery_incomplete",
            "state": None if state is None else copy.deepcopy(state["last_event"]),
        }
    result.update(
        episode_index=int(spec["episode_index"]),
        episode_seed=int(spec["episode_seed"]),
        initial_ball_cup=spec["initial_ball_cup"],
    )
    (output_dir / "recovery_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def _aggregate(results: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    completed = [row for row in results if row.get("completed")]

    def group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"conditions": 0}
        checkpoint_steps = [0, *_parse_ints(args.recovery_checkpoints, name="recovery-checkpoints")]
        final_step = str(max(checkpoint_steps))
        return {
            "conditions": len(rows),
            "mean_initial_error_mm": float(
                np.mean([row["checkpoint_snapshots"]["0"]["error_norm_mm"] for row in rows])
            ),
            "mean_final_error_mm": float(
                np.mean([row["checkpoint_snapshots"][final_step]["error_norm_mm"] for row in rows])
            ),
            "mean_final_reduction_mm": float(np.mean([row["error_reduction_mm"][final_step] for row in rows])),
            "recovered_within_threshold": int(sum(row["recovered_within_threshold"] for row in rows)),
            "recovery_rate": float(np.mean([row["recovered_within_threshold"] for row in rows])),
            "mean_z_descent_mm": float(np.mean([row["z_descent_during_recovery_mm"] for row in rows])),
            "mean_cup_xy_displacement_mm": float(
                np.mean([row["cup_xy_displacement_during_recovery_mm"] for row in rows])
            ),
            "mean_feedback_alignment_positive_fraction": float(
                np.mean([row["feedback_alignment_positive_fraction"] for row in rows])
            ),
            "mean_error_mm_by_step": {
                str(step): float(np.mean([row["checkpoint_snapshots"][str(step)]["error_norm_mm"] for row in rows]))
                for step in checkpoint_steps
            },
            "mean_reduction_mm_by_step": {
                str(step): float(np.mean([row["error_reduction_mm"][str(step)] for row in rows]))
                for step in checkpoint_steps
            },
        }

    by_stage = {
        stage: group([row for row in completed if row["condition"]["stage"] == stage])
        for stage, _ in _parse_stages(args.stages)
    }
    by_magnitude = {
        str(magnitude): group([row for row in completed if row["condition"]["magnitude_mm"] == magnitude])
        for magnitude in _parse_ints(args.magnitudes_mm, name="magnitudes-mm")
    }
    by_direction = {
        direction: group([row for row in completed if row["condition"]["direction"] == direction])
        for direction in [item.strip() for item in args.directions.split(",") if item.strip()]
    }
    return {
        "experiment": "closed-loop absolute-EEF visual-servo recovery after lateral descent disturbance",
        "episode_index": args.episode_index,
        "deterministic_sample_salt": args.deterministic_sample_salt,
        "recovery_checkpoints": _parse_ints(args.recovery_checkpoints, name="recovery-checkpoints"),
        "recovery_success_threshold_mm": args.recovery_success_threshold_mm,
        "conditions_requested": len(results),
        "conditions_completed": len(completed),
        "overall": group(completed),
        "by_stage": by_stage,
        "by_magnitude_mm": by_magnitude,
        "by_direction": by_direction,
        "results": results,
    }


def main(args: Args) -> None:
    if args.num_frames != fixed_eef.TOTAL_FRAMES or args.frame_stride != 1:
        raise ValueError("This diagnostic requires --num-frames 61 --frame-stride 1")
    if args.action_mode != "raw7" or args.action_dim != 7 or args.osc_input_type != "absolute":
        raise ValueError("This diagnostic requires absolute raw7 actions")
    if args.control_during_scripted_observation:
        raise ValueError("Use --no-control-during-scripted-observation")
    if args.perturb_steps <= 0 or args.websocket_reconnect_interval <= 0:
        raise ValueError("Perturb and reconnect steps must be positive")
    if args.physics_debug_window < args.max_policy_steps:
        raise ValueError("Use --physics-debug-window >= --max-policy-steps")

    root = pathlib.Path(args.video_out_path)
    root.mkdir(parents=True, exist_ok=True)
    conditions = _conditions(args)
    spec = _episode_spec(args)
    context = mp.get_context("spawn")
    results = []
    for index, condition in enumerate(conditions):
        output_dir = root / condition.name
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir}")
        process = context.Process(
            target=_run_condition,
            kwargs={"args": args, "condition": condition, "spec": spec},
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"condition={condition.name} exited {process.exitcode}")
        result = json.loads((output_dir / "recovery_result.json").read_text(encoding="utf-8"))
        results.append(result)
        logging.info(
            "disturbance aggregate completed=%d/%d condition=%s valid=%s recovered=%s",
            index + 1,
            len(conditions),
            condition.name,
            result.get("completed"),
            result.get("recovered_within_threshold"),
        )

    summary = _aggregate(results, args)
    result_path = root / "result.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("descent disturbance result=%s", result_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
