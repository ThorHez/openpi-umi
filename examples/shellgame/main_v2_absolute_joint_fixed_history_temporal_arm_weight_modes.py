"""Compare temporal arm weighting modes while keeping the newest gripper target.

This control-variable evaluator leaves the existing temporal-ensemble entry
points unchanged.  Only the seven arm-joint weights differ:

* ``oldest_heavy`` reproduces the existing implementation.
* ``newest_heavy`` reverses those exponential weights.
* ``newest_only`` executes the newest chunk without arm ensembling.

The gripper width always comes from the newest chunk.
"""

from __future__ import annotations

from collections import deque
import dataclasses
import logging
from typing import Literal

import main as base
import main_v2_absolute_joint as joint
import main_v2_absolute_joint_fixed_history as fixed
import main_v2_absolute_joint_fixed_history_temporal_ensemble as temporal
import numpy as np
import tyro

ArmEnsembleMode = Literal["oldest_heavy", "newest_heavy", "newest_only"]
_BASE_POLICY_ENV_ACTION = temporal._temporal_ensemble_policy_env_action  # noqa: SLF001


@dataclasses.dataclass
class Args(temporal.Args):
    arm_ensemble_mode: ArmEnsembleMode = "oldest_heavy"


def _ensemble_arm_newest_gripper(
    predictions: np.ndarray,
    *,
    decay: float,
    mode: ArmEnsembleMode,
) -> tuple[np.ndarray, np.ndarray]:
    if predictions.ndim != 2 or len(predictions) == 0:
        raise ValueError(f"Expected non-empty [chunks, action_dim], got {predictions.shape}")

    if mode == "oldest_heavy":
        exponents = np.arange(len(predictions), dtype=np.float32)
        weights = np.exp(-decay * exponents)
    elif mode == "newest_heavy":
        exponents = np.arange(len(predictions) - 1, -1, -1, dtype=np.float32)
        weights = np.exp(-decay * exponents)
    elif mode == "newest_only":
        weights = np.zeros(len(predictions), dtype=np.float32)
        weights[-1] = 1.0
    else:
        raise ValueError(f"Unknown arm ensemble mode: {mode!r}")

    weights /= np.sum(weights)
    action = np.sum(predictions * weights[:, None], axis=0).astype(np.float32)
    action[-1] = predictions[-1, -1]
    return action, weights


def _policy_env_action(
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

    def ensemble(predictions: np.ndarray, *, decay: float) -> tuple[np.ndarray, np.ndarray]:
        return _ensemble_arm_newest_gripper(
            predictions,
            decay=decay,
            mode=args.arm_ensemble_mode,
        )

    temporal._ensemble_predictions = ensemble  # noqa: SLF001
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
    base._policy_env_action = _policy_env_action  # noqa: SLF001
    temporal.reset_temporal_ensemble_state()

    logging.basicConfig(level=logging.INFO, force=True)
    base.eval_shellgame(tyro.cli(Args))


if __name__ == "__main__":
    main()
