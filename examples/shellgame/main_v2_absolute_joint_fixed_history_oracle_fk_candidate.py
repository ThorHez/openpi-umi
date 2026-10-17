"""Select one action chunk with oracle target-cup FK at a chosen replan.

This diagnostic samples multiple chunks only at ``oracle_replan_step``.  It
temporarily applies each candidate's joint targets to MuJoCo, scores the EEF
trajectory against the simulator's true target-cup grasp position, restores
the complete simulator state, and executes the lowest-error candidate.

The oracle cup pose is never added to the policy observation.  This is an
upper-bound diagnostic, not a deployable policy.
"""

from __future__ import annotations

from collections import deque
import dataclasses
import logging
from typing import Any

import main as base
import main_v2_absolute_joint as joint
import main_v2_absolute_joint_fixed_history as fixed
import numpy as np
import tyro


@dataclasses.dataclass
class Args(joint.Args):
    candidate_count: int = 16
    oracle_replan_step: int = 48
    candidate_score_start: int = 4
    candidate_score_end: int = 13
    target_grasp_z_offset: float = 0.04


@dataclasses.dataclass
class _OracleState:
    control_step: int = 0
    replan_index: int = 0


_STATES: dict[int, _OracleState] = {}
_DIAGNOSTICS: list[dict[str, Any]] = []


def reset_oracle_candidate_state() -> None:
    _STATES.clear()
    _DIAGNOSTICS.clear()


def oracle_candidate_diagnostics() -> list[dict[str, Any]]:
    return list(_DIAGNOSTICS)


def _validate_args(args: Args) -> None:
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")
    if args.replan_steps != args.action_horizon:
        raise ValueError("Oracle FK diagnostic requires --replan-steps == --action-horizon")
    if args.oracle_replan_step < 0 or args.oracle_replan_step % args.replan_steps:
        raise ValueError("--oracle-replan-step must be a non-negative replan boundary")
    if not 0 <= args.candidate_score_start < args.candidate_score_end <= args.action_horizon:
        raise ValueError("Candidate score range must lie inside the action horizon")


def _infer_candidates(client, element: dict, *, count: int, args: Args) -> list[np.ndarray]:
    candidates = []
    for _ in range(count):
        actions = np.asarray(client.infer(element)["actions"], dtype=np.float32)
        expected = (args.action_horizon, joint.ACTION_DIM)
        if actions.shape != expected:
            raise RuntimeError(f"Policy returned shape {actions.shape}; expected {expected}")
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("Policy returned non-finite actions")
        candidates.append(actions)
    return candidates


def _score_candidates(shell, env, candidates: list[np.ndarray], *, args: Args) -> list[dict[str, Any]]:
    target_cup = args.initial_ball_cup
    if target_cup not in shell.CUP_NAMES:
        raise ValueError(f"Oracle FK scoring requires a concrete target cup, got {target_cup!r}")
    cup_position = base._cup_positions(shell, env)[target_cup]  # noqa: SLF001
    desired = np.asarray(cup_position, dtype=np.float32).copy()
    desired[2] += args.target_grasp_z_offset

    q_indices = np.asarray(env.robots[0]._ref_joint_pos_indexes, dtype=np.int64)  # noqa: SLF001
    action_low, action_high = (np.asarray(value, dtype=np.float32) for value in env.action_spec)
    q_low = action_low[: joint.JOINT_DIM]
    q_high = action_high[: joint.JOINT_DIM]
    simulator_state = env.sim.get_state()
    rows = []
    try:
        for candidate in candidates:
            clipped = np.clip(candidate[:, : joint.JOINT_DIM], q_low, q_high)
            eef_positions = []
            for q_target in clipped:
                env.sim.data.qpos[q_indices] = q_target
                env.sim.forward()
                eef_positions.append(np.asarray(shell.get_eef_pos(env), dtype=np.float32))
            eef_positions = np.asarray(eef_positions)
            scored = eef_positions[args.candidate_score_start : args.candidate_score_end]
            errors = np.linalg.norm(scored - desired[None, :], axis=1)
            xy_errors = np.linalg.norm(scored[:, :2] - desired[None, :2], axis=1)
            z_errors = np.abs(scored[:, 2] - desired[2])
            best_local = int(np.argmin(errors))
            best_offset = args.candidate_score_start + best_local
            rows.append(
                {
                    "score_m": float(errors[best_local]),
                    "best_offset": best_offset,
                    "best_xy_error_m": float(xy_errors[best_local]),
                    "best_z_error_m": float(z_errors[best_local]),
                    "best_eef_pos": eef_positions[best_offset].tolist(),
                    "best_gripper_width": float(candidate[best_offset, -1]),
                    "clipped_joint_values": int(
                        np.count_nonzero(np.abs(clipped - candidate[:, : joint.JOINT_DIM]) > 1e-6)
                    ),
                }
            )
    finally:
        env.sim.set_state(simulator_state)
        env.sim.forward()
    return rows


def _oracle_fk_candidate_policy_env_action(
    shell,
    env,
    history: list[dict],
    start_eef_pos: np.ndarray,
    start_eef_quat: np.ndarray,
    action_plan: deque,
    gripper_action: float,
    *,
    client,
    args: Args,
    prompt: str | None = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    if args.zero_action_policy:
        return base._zero_env_action(env, args.default_gripper_action), gripper_action, None  # noqa: SLF001

    _validate_args(args)
    state = _STATES.setdefault(id(action_plan), _OracleState())
    if not action_plan:
        if client is None:
            raise RuntimeError("Policy client is required unless zero_action_policy is enabled")
        element = base._policy_input(history, start_eef_pos, args=args, prompt=prompt)  # noqa: SLF001
        count = args.candidate_count if state.control_step == args.oracle_replan_step else 1
        candidates = _infer_candidates(client, element, count=count, args=args)
        selected_index = 0
        score_rows = None
        if count > 1:
            score_rows = _score_candidates(shell, env, candidates, args=args)
            selected_index = int(np.argmin([row["score_m"] for row in score_rows]))
            logging.info(
                "oracle FK step=%d selected=%d baseline_score=%.4f oracle_score=%.4f scores=%s",
                state.control_step,
                selected_index,
                score_rows[0]["score_m"],
                score_rows[selected_index]["score_m"],
                [round(row["score_m"], 4) for row in score_rows],
            )
        selected = candidates[selected_index]
        for action in selected[: args.replan_steps]:
            action_plan.append((action, None, None))
        _DIAGNOSTICS.append(
            {
                "replan_index": state.replan_index,
                "control_step": state.control_step,
                "candidate_count": count,
                "selected_index": selected_index,
                "target_cup": args.initial_ball_cup,
                "score_start": args.candidate_score_start,
                "score_end": args.candidate_score_end,
                "target_grasp_z_offset": args.target_grasp_z_offset,
                "candidates": score_rows,
            }
        )
        state.replan_index += 1

    policy_action, plan_base_pos, plan_base_quat = action_plan.popleft()
    state.control_step += 1
    env_action, gripper_action = base._target_action_to_env_action(  # noqa: SLF001
        shell,
        env,
        policy_action,
        start_eef_pos=start_eef_pos,
        start_eef_quat=start_eef_quat,
        plan_base_pos=plan_base_pos,
        plan_base_quat=plan_base_quat,
        last_gripper_action=gripper_action,
        deadband=args.gripper_deadband,
        args=args,
    )
    return env_action, gripper_action, policy_action


def main() -> None:
    base._episode_namespace = joint._episode_namespace  # noqa: SLF001
    base._append_observation = joint._append_observation  # noqa: SLF001
    base._policy_action_dim = joint._policy_action_dim  # noqa: SLF001
    base._policy_input = fixed._fixed_history_policy_input  # noqa: SLF001
    base._zero_env_action = joint._zero_env_action  # noqa: SLF001
    base._target_action_to_env_action = joint._absolute_joint_action_to_env_action  # noqa: SLF001
    base._policy_env_action = _oracle_fk_candidate_policy_env_action  # noqa: SLF001
    reset_oracle_candidate_state()

    logging.basicConfig(level=logging.INFO, force=True)
    base.eval_shellgame(tyro.cli(Args))


if __name__ == "__main__":
    main()
