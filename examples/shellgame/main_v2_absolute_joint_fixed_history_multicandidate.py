"""Fixed-history absolute-joint evaluation with continuity-ranked action chunks.

At every replan boundary, the evaluator samples multiple independent 16-step
chunks from the same policy observation.  It selects one chunk using only the
seven arm joints:

* first-step RMS distance to the currently measured joints; and
* RMS distance to the unexecuted suffix of the previously selected chunk.

The gripper dimension is deliberately excluded from selection so its measured
width scale cannot dominate arm-trajectory continuity.  The selected chunk is
executed through the unchanged absolute JOINT_POSITION and gripper converters.
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
    candidate_count: int = 4
    continuity_current_weight: float = 1.0
    continuity_overlap_weight: float = 1.0


@dataclasses.dataclass
class _ContinuityState:
    previous_chunk: np.ndarray | None = None
    replan_index: int = 0


_STATES: dict[int, _ContinuityState] = {}
_DIAGNOSTICS: list[dict[str, Any]] = []


def reset_multicandidate_state() -> None:
    _STATES.clear()
    _DIAGNOSTICS.clear()


def multicandidate_diagnostics() -> list[dict[str, Any]]:
    return list(_DIAGNOSTICS)


def _validate_args(args: Args) -> None:
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")
    if not 1 <= args.replan_steps <= args.action_horizon:
        raise ValueError(
            "Multicandidate evaluation requires 1 <= replan_steps <= action_horizon; "
            f"got {args.replan_steps} and {args.action_horizon}"
        )
    if args.continuity_current_weight < 0 or args.continuity_overlap_weight < 0:
        raise ValueError("Continuity weights must be non-negative")
    if args.continuity_current_weight + args.continuity_overlap_weight <= 0:
        raise ValueError("At least one continuity weight must be positive")


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(left) - np.asarray(right)))))


def _score_candidate(
    candidate: np.ndarray,
    *,
    current_joint: np.ndarray,
    previous_chunk: np.ndarray | None,
    args: Args,
) -> tuple[float, float, float | None]:
    current_rms = _rms(candidate[0, : joint.JOINT_DIM], current_joint)
    overlap_rms = None
    if previous_chunk is not None:
        previous_remaining = previous_chunk[args.replan_steps :, : joint.JOINT_DIM]
        overlap = min(len(previous_remaining), len(candidate))
        if overlap > 0:
            overlap_rms = _rms(
                candidate[:overlap, : joint.JOINT_DIM],
                previous_remaining[:overlap],
            )

    score = args.continuity_current_weight * current_rms
    if overlap_rms is not None:
        score += args.continuity_overlap_weight * overlap_rms
    return score, current_rms, overlap_rms


def _multicandidate_policy_env_action(
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
    state = _STATES.setdefault(id(action_plan), _ContinuityState())

    if not action_plan:
        element = base._policy_input(history, start_eef_pos, args=args, prompt=prompt)  # noqa: SLF001
        if client is None:
            raise RuntimeError("Policy client is required unless zero_action_policy is enabled")
        action_dim = base._policy_action_dim(args)  # noqa: SLF001
        first_result = client.infer(element)
        batched = first_result.get("actions_candidates")
        if batched is not None:
            batched = np.asarray(batched, dtype=np.float32)
            expected = (args.candidate_count, args.action_horizon, action_dim)
            if batched.shape != expected:
                raise RuntimeError(f"Candidate server returned shape {batched.shape}; expected {expected}")
            raw_candidates = list(batched)
        else:
            raw_candidates = [np.asarray(first_result["actions"], dtype=np.float32)]
            for _ in range(1, args.candidate_count):
                raw_candidates.append(np.asarray(client.infer(element)["actions"], dtype=np.float32))

        candidates = []
        for actions in raw_candidates:
            if actions.ndim != 2 or actions.shape != (args.action_horizon, action_dim):
                raise RuntimeError(
                    "Multicandidate policy returned unexpected actions shape "
                    f"{actions.shape}; expected ({args.action_horizon}, {action_dim})"
                )
            if not np.all(np.isfinite(actions)):
                raise RuntimeError("Multicandidate policy returned non-finite actions")
            candidates.append(actions)

        current_joint = np.asarray(
            env.robots[0]._joint_positions,  # noqa: SLF001
            dtype=np.float32,
        ).reshape(-1)
        if current_joint.shape != (joint.JOINT_DIM,):
            raise RuntimeError(f"Expected measured joints ({joint.JOINT_DIM},), got {current_joint.shape}")

        score_rows = [
            _score_candidate(
                candidate,
                current_joint=current_joint,
                previous_chunk=state.previous_chunk,
                args=args,
            )
            for candidate in candidates
        ]
        selected_index = int(np.argmin([row[0] for row in score_rows]))
        selected = candidates[selected_index]
        previous_last_executed = (
            None if state.previous_chunk is None else state.previous_chunk[args.replan_steps - 1, : joint.JOINT_DIM]
        )
        selected_boundary_rms = (
            None if previous_last_executed is None else _rms(selected[0, : joint.JOINT_DIM], previous_last_executed)
        )
        diagnostic = {
            "replan_index": state.replan_index,
            "candidate_count": args.candidate_count,
            "selected_index": selected_index,
            "scores": [float(row[0]) for row in score_rows],
            "current_joint_rms": [float(row[1]) for row in score_rows],
            "overlap_joint_rms": [None if row[2] is None else float(row[2]) for row in score_rows],
            "selected_boundary_rms": selected_boundary_rms,
            "selected_first_gripper_width": float(selected[0, joint.JOINT_DIM]),
        }
        _DIAGNOSTICS.append(diagnostic)
        logging.info(
            "multicandidate replan=%d selected=%d scores=%s current_rms=%s overlap_rms=%s boundary_rms=%s",
            state.replan_index,
            selected_index,
            [round(value, 5) for value in diagnostic["scores"]],
            [round(value, 5) for value in diagnostic["current_joint_rms"]],
            [None if value is None else round(value, 5) for value in diagnostic["overlap_joint_rms"]],
            None if selected_boundary_rms is None else round(selected_boundary_rms, 5),
        )

        for action in selected[: args.replan_steps, :action_dim]:
            action_plan.append((action, None, None))
        state.previous_chunk = selected.copy()
        state.replan_index += 1

    policy_action, plan_base_pos, plan_base_quat = action_plan.popleft()
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
    base._policy_env_action = _multicandidate_policy_env_action  # noqa: SLF001
    reset_multicandidate_state()

    logging.basicConfig(level=logging.INFO, force=True)
    base.eval_shellgame(tyro.cli(Args))


if __name__ == "__main__":
    main()
