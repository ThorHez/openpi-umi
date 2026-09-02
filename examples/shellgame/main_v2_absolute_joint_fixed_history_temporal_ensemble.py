"""Fixed-history absolute-joint evaluation with temporal action ensembling.

Each replan produces one 16-step action chunk.  Instead of discarding the
unexecuted suffixes of older chunks, this evaluator aligns every prediction on
the global control-step timeline and averages all chunks that cover the action
being executed.  Exponential weights are assigned from oldest to newest, so an
existing plan anchors the motion while fresh predictions can still correct it.

The ensemble is performed in the model's absolute joint + gripper-width action
space before the unchanged JOINT_POSITION and gripper converters are applied.
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
    ensemble_decay: float = 0.25
    ensemble_max_chunks: int = 0


@dataclasses.dataclass
class _PredictedChunk:
    start_step: int
    actions: np.ndarray


@dataclasses.dataclass
class _EnsembleState:
    chunks: list[_PredictedChunk] = dataclasses.field(default_factory=list)
    control_step: int = 0
    replan_index: int = 0
    last_planned_action: np.ndarray | None = None


_STATES: dict[int, _EnsembleState] = {}
_DIAGNOSTICS: list[dict[str, Any]] = []


def reset_temporal_ensemble_state() -> None:
    _STATES.clear()
    _DIAGNOSTICS.clear()


def temporal_ensemble_diagnostics() -> list[dict[str, Any]]:
    return list(_DIAGNOSTICS)


def _validate_args(args: Args) -> None:
    if not 1 <= args.replan_steps <= args.action_horizon:
        raise ValueError(
            "Temporal ensemble requires 1 <= replan_steps <= action_horizon; "
            f"got {args.replan_steps} and {args.action_horizon}"
        )
    if args.ensemble_decay < 0:
        raise ValueError("--ensemble-decay must be non-negative")
    if args.ensemble_max_chunks < 0:
        raise ValueError("--ensemble-max-chunks must be non-negative")


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(left) - np.asarray(right)))))


def _predictions_for_step(
    chunks: list[_PredictedChunk],
    *,
    control_step: int,
    max_chunks: int,
) -> tuple[list[_PredictedChunk], np.ndarray]:
    active = [chunk for chunk in chunks if chunk.start_step <= control_step < chunk.start_step + len(chunk.actions)]
    if max_chunks > 0:
        active = active[-max_chunks:]
    if not active:
        raise RuntimeError(f"No action chunk covers control step {control_step}")
    predictions = np.stack(
        [chunk.actions[control_step - chunk.start_step] for chunk in active],
        axis=0,
    )
    return active, predictions


def _ensemble_predictions(
    predictions: np.ndarray,
    *,
    decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    if predictions.ndim != 2 or len(predictions) == 0:
        raise ValueError(f"Expected non-empty [chunks, action_dim], got {predictions.shape}")
    # Rows are chronological.  This matches the original temporal-ensemble
    # formulation: older predictions receive the largest weight and anchor the
    # plan; decay=0 becomes an unweighted mean.
    weights = np.exp(-decay * np.arange(len(predictions), dtype=np.float32))
    weights /= np.sum(weights)
    action = np.sum(predictions * weights[:, None], axis=0)
    return action.astype(np.float32), weights


def _temporal_ensemble_policy_env_action(
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
    state = _STATES.setdefault(id(action_plan), _EnsembleState())

    if not action_plan:
        element = base._policy_input(history, start_eef_pos, args=args, prompt=prompt)  # noqa: SLF001
        if client is None:
            raise RuntimeError("Policy client is required unless zero_action_policy is enabled")
        action_dim = base._policy_action_dim(args)  # noqa: SLF001
        actions = np.asarray(client.infer(element)["actions"], dtype=np.float32)
        expected = (args.action_horizon, action_dim)
        if actions.shape != expected:
            raise RuntimeError(f"Policy returned shape {actions.shape}; expected {expected}")
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("Policy returned non-finite actions")

        state.chunks.append(_PredictedChunk(start_step=state.control_step, actions=actions.copy()))
        state.chunks = [chunk for chunk in state.chunks if chunk.start_step + len(chunk.actions) > state.control_step]

        step_rows = []
        for offset in range(args.replan_steps):
            step = state.control_step + offset
            active, predictions = _predictions_for_step(
                state.chunks,
                control_step=step,
                max_chunks=args.ensemble_max_chunks,
            )
            ensembled, weights = _ensemble_predictions(predictions, decay=args.ensemble_decay)
            arm_predictions = predictions[:, : joint.JOINT_DIM]
            arm_dispersion = _rms(arm_predictions, ensembled[None, : joint.JOINT_DIM])
            newest_delta = _rms(
                ensembled[: joint.JOINT_DIM],
                predictions[-1, : joint.JOINT_DIM],
            )
            boundary_rms = None
            if offset == 0 and state.last_planned_action is not None:
                boundary_rms = _rms(
                    ensembled[: joint.JOINT_DIM],
                    state.last_planned_action[: joint.JOINT_DIM],
                )
            step_rows.append(
                {
                    "control_step": step,
                    "chunk_start_steps": [chunk.start_step for chunk in active],
                    "weights": [float(value) for value in weights],
                    "arm_dispersion_rms": arm_dispersion,
                    "newest_arm_delta_rms": newest_delta,
                    "boundary_rms": boundary_rms,
                    "gripper_width_predictions": [float(value) for value in predictions[:, -1]],
                    "ensembled_gripper_width": float(ensembled[-1]),
                }
            )
            action_plan.append((ensembled, None, None))
            state.last_planned_action = ensembled.copy()

        diagnostic = {
            "replan_index": state.replan_index,
            "chunk_start_step": state.control_step,
            "num_retained_chunks": len(state.chunks),
            "steps": step_rows,
        }
        _DIAGNOSTICS.append(diagnostic)
        first = step_rows[0]
        logging.info(
            "temporal ensemble replan=%d step=%d contributors=%d weights=%s "
            "dispersion=%.5f newest_delta=%.5f boundary_rms=%s",
            state.replan_index,
            state.control_step,
            len(first["weights"]),
            [round(value, 4) for value in first["weights"]],
            first["arm_dispersion_rms"],
            first["newest_arm_delta_rms"],
            None if first["boundary_rms"] is None else round(first["boundary_rms"], 5),
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
    base._policy_env_action = _temporal_ensemble_policy_env_action  # noqa: SLF001
    reset_temporal_ensemble_state()

    logging.basicConfig(level=logging.INFO, force=True)
    base.eval_shellgame(tyro.cli(Args))


if __name__ == "__main__":
    main()
