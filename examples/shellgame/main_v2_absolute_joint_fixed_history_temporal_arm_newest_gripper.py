"""Temporal arm-joint ensemble with gripper width from the newest chunk.

This is a strict control-variable variant of
``main_v2_absolute_joint_fixed_history_temporal_ensemble.py``.  The seven arm
joints retain the same aligned exponential ensemble, while the gripper target
is copied from the newest prediction covering the current control step.
"""

from __future__ import annotations

from collections import deque
import logging

import main as base
import main_v2_absolute_joint as joint
import main_v2_absolute_joint_fixed_history as fixed
import main_v2_absolute_joint_fixed_history_temporal_ensemble as temporal
import numpy as np
import tyro

Args = temporal.Args
_BASE_ENSEMBLE_PREDICTIONS = temporal._ensemble_predictions  # noqa: SLF001
_BASE_POLICY_ENV_ACTION = temporal._temporal_ensemble_policy_env_action  # noqa: SLF001


def _ensemble_arm_newest_gripper(
    predictions: np.ndarray,
    *,
    decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    action, weights = _BASE_ENSEMBLE_PREDICTIONS(predictions, decay=decay)
    action[-1] = predictions[-1, -1]
    return action, weights


def _temporal_arm_newest_gripper_policy_env_action(
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
    original = temporal._ensemble_predictions  # noqa: SLF001
    temporal._ensemble_predictions = _ensemble_arm_newest_gripper  # noqa: SLF001
    try:
        return _BASE_POLICY_ENV_ACTION(
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
    finally:
        temporal._ensemble_predictions = original  # noqa: SLF001


def main() -> None:
    base._episode_namespace = joint._episode_namespace  # noqa: SLF001
    base._append_observation = joint._append_observation  # noqa: SLF001
    base._policy_action_dim = joint._policy_action_dim  # noqa: SLF001
    base._policy_input = fixed._fixed_history_policy_input  # noqa: SLF001
    base._zero_env_action = joint._zero_env_action  # noqa: SLF001
    base._target_action_to_env_action = joint._absolute_joint_action_to_env_action  # noqa: SLF001
    base._policy_env_action = _temporal_arm_newest_gripper_policy_env_action  # noqa: SLF001
    temporal.reset_temporal_ensemble_state()

    logging.basicConfig(level=logging.INFO, force=True)
    base.eval_shellgame(tyro.cli(Args))


if __name__ == "__main__":
    main()
