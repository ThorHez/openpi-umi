"""Fixed-history absolute-joint evaluation with one gripper-latch intervention.

The model, joint actions, diffusion sampling, replanning, and environment are
unchanged.  Starting at policy step 50 (the demonstrated grasp phase), the
first close command latches the environment gripper command to +1 until the
episode ends.  This isolates repeated gripper reopenings observed in the
unmodified closed-loop evaluation.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import tyro

import main as base
import main_v2_absolute_joint as joint
from main_v2_absolute_joint_fixed_history import HISTORY_FRAMES
from main_v2_absolute_joint_fixed_history import _fixed_history_policy_input


@dataclasses.dataclass
class Args(joint.Args):
    gripper_latch_after_step: int = 50


_ORIGINAL_POLICY_ENV_ACTION = base._policy_env_action  # noqa: SLF001
_gripper_latched = False


def _latched_policy_env_action(
    shell,
    env,
    history,
    start_eef_pos,
    start_eef_quat,
    action_plan,
    gripper_action,
    *,
    client,
    args: Args,
    prompt=None,
):
    global _gripper_latched  # noqa: PLW0603
    policy_step = len(history) - HISTORY_FRAMES
    if policy_step == 0:
        _gripper_latched = False

    env_action, proposed_gripper_action, policy_action = _ORIGINAL_POLICY_ENV_ACTION(
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
    if policy_step >= args.gripper_latch_after_step and proposed_gripper_action > 0.0:
        if not _gripper_latched:
            logging.info("gripper latched closed at policy_step=%d", policy_step)
        _gripper_latched = True
    if _gripper_latched:
        env_action = np.asarray(env_action, dtype=np.float32).copy()
        env_action[-1] = 1.0
        proposed_gripper_action = 1.0
    return env_action, proposed_gripper_action, policy_action


def main() -> None:
    base._episode_namespace = joint._episode_namespace  # noqa: SLF001
    base._append_observation = joint._append_observation  # noqa: SLF001
    base._policy_action_dim = joint._policy_action_dim  # noqa: SLF001
    base._policy_input = _fixed_history_policy_input  # noqa: SLF001
    base._zero_env_action = joint._zero_env_action  # noqa: SLF001
    base._target_action_to_env_action = joint._absolute_joint_action_to_env_action  # noqa: SLF001
    base._policy_env_action = _latched_policy_env_action  # noqa: SLF001

    logging.basicConfig(level=logging.INFO, force=True)
    base.eval_shellgame(tyro.cli(Args))


if __name__ == "__main__":
    main()
