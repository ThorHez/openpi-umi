"""ShellGame evaluation with absolute joint-position state and actions.

Policy state:
    robot0_joint_pos      [1, 7] absolute arm joint angles in radians
    robot0_gripper_width  [1, 1] measured gripper opening in meters

Policy action:
    [q1, ..., q7, gripper_width] absolute targets

The arm runs Robosuite's absolute JOINT_POSITION controller. The measured
gripper-width target is converted to the gripper controller's open / close
command with the same deadband behavior used by the pose-based evaluator.
"""

# This entry point intentionally patches selected helpers in the sibling
# evaluator so its rollout, metrics, phase prompts, and video logic are reused.
# pylint: disable=protected-access

from __future__ import annotations

import dataclasses
import logging
from argparse import Namespace

import numpy as np
import tyro

import main as base


JOINT_DIM = 7
ACTION_DIM = 8


@dataclasses.dataclass
class Args(base.Args):
    action_dim: int = ACTION_DIM
    action_mode: str = "joint8"
    resize_size: int = 224
    fps: int = 10
    arm_controller_type: str = "JOINT_POSITION"
    joint_kp: float = 50.0
    joint_damping_ratio: float = 1.0


_ORIGINAL_EPISODE_NAMESPACE = base._episode_namespace
_ORIGINAL_APPEND_OBSERVATION = base._append_observation


def _episode_namespace(args: Args, *, seed: int, initial_ball_cup: str, num_swaps: int) -> Namespace:
    if args.observe_eef_frames != 0:
        raise ValueError("Absolute-joint evaluation requires --observe-eef-frames 0")
    namespace = _ORIGINAL_EPISODE_NAMESPACE(
        args,
        seed=seed,
        initial_ball_cup=initial_ball_cup,
        num_swaps=num_swaps,
    )
    namespace.arm_controller_type = args.arm_controller_type
    namespace.joint_kp = args.joint_kp
    namespace.joint_damping_ratio = args.joint_damping_ratio
    return namespace


def _append_observation(
    shell,
    env,
    ep_args: Namespace,
    wrist_camera_name: str | None,
    history: list[dict],
    replay: list[np.ndarray],
    *,
    resize_size: int,
):
    _ORIGINAL_APPEND_OBSERVATION(
        shell,
        env,
        ep_args,
        wrist_camera_name,
        history,
        replay,
        resize_size=resize_size,
    )
    obs = env._get_observations(force_update=True)
    joint_pos = shell.obs_vector(obs, "robot0_joint_pos", size=JOINT_DIM)
    if joint_pos.shape != (JOINT_DIM,):
        raise RuntimeError(f"Expected {JOINT_DIM} Panda arm joints, got shape={joint_pos.shape}")
    history[-1]["joint_pos"] = joint_pos.astype(np.float32)


def _policy_action_dim(args: Args) -> int:
    if args.action_mode != "joint8" or args.action_dim != ACTION_DIM:
        raise ValueError(
            "Absolute-joint evaluator requires action_mode='joint8' and action_dim=8; "
            f"got action_mode={args.action_mode!r}, action_dim={args.action_dim}"
        )
    return ACTION_DIM


def _policy_input(
    history: list[dict],
    start_eef_pos: np.ndarray,
    *,
    args: Args,
    prompt: str | None = None,
) -> dict:
    del start_eef_pos
    current = history[-1]
    joint_pos = np.asarray(current["joint_pos"], dtype=np.float32).reshape(1, JOINT_DIM)
    gripper_width = np.asarray([[current["gripper_width"]]], dtype=np.float32)
    element = {
        "robot0_joint_pos": joint_pos,
        "robot0_gripper_width": gripper_width,
        "actions": np.zeros((args.action_horizon, ACTION_DIM), dtype=np.float32),
        "prompt": args.task if prompt is None else prompt,
    }
    if args.policy_input_mode == "history":
        frames = base._window(history, num_frames=args.num_frames, frame_stride=args.frame_stride)
        for index, frame in enumerate(frames):
            element[f"left_wrist_0_rgb_0_{index}"] = frame["wrist"]
            element[f"left_wrist_0_rgb_1_{index}"] = frame["base"]
    elif args.policy_input_mode == "single_frame":
        element["left_wrist_0_rgb_0"] = current["wrist"]
        element["left_wrist_0_rgb_1"] = current["base"]
    else:
        raise ValueError(
            f"Unknown policy_input_mode={args.policy_input_mode!r}; "
            "expected 'history' or 'single_frame'"
        )
    return element


def _zero_env_action(env, gripper_action: float) -> np.ndarray:
    """Hold the achieved arm joints while preserving the gripper command."""
    action_low, action_high = env.action_spec
    action = np.zeros_like(action_low, dtype=np.float32)
    joint_pos = np.asarray(env.robots[0]._joint_positions, dtype=np.float32).reshape(-1)
    if joint_pos.shape != (JOINT_DIM,) or action.shape != (ACTION_DIM,):
        raise RuntimeError(
            f"Absolute-joint environment mismatch: joints={joint_pos.shape}, action_spec={action.shape}"
        )
    action[:JOINT_DIM] = joint_pos
    action[-1] = gripper_action
    return np.clip(action, action_low, action_high)


def _absolute_joint_action_to_env_action(
    shell,
    env,
    target8: np.ndarray,
    *,
    start_eef_pos: np.ndarray,
    start_eef_quat: np.ndarray,
    plan_base_pos: np.ndarray | None,
    plan_base_quat: np.ndarray | None,
    last_gripper_action: float,
    deadband: float,
    args: Args,
) -> tuple[np.ndarray, float]:
    del start_eef_pos, start_eef_quat, plan_base_pos, plan_base_quat, args
    target8 = np.asarray(target8, dtype=np.float32).reshape(-1)
    action_low, action_high = env.action_spec
    action_low = np.asarray(action_low, dtype=np.float32)
    action_high = np.asarray(action_high, dtype=np.float32)
    if target8.shape != (ACTION_DIM,) or action_low.shape != (ACTION_DIM,):
        raise RuntimeError(
            f"Expected policy and environment actions shaped ({ACTION_DIM},), "
            f"got policy={target8.shape}, env={action_low.shape}"
        )
    if not np.all(np.isfinite(target8)):
        raise RuntimeError(f"Policy returned non-finite absolute-joint action: {target8}")

    obs = env._get_observations(force_update=True)
    current_width = base._gripper_width(shell.obs_vector(obs, "robot0_gripper_qpos"))
    target_width = float(target8[-1])
    gripper_action = last_gripper_action
    if target_width < current_width - deadband:
        gripper_action = 1.0
    elif target_width > current_width + deadband:
        gripper_action = -1.0

    env_action = np.empty(ACTION_DIM, dtype=np.float32)
    env_action[:JOINT_DIM] = target8[:JOINT_DIM]
    env_action[-1] = gripper_action
    return np.clip(env_action, action_low, action_high), gripper_action


def main() -> None:
    base._episode_namespace = _episode_namespace
    base._append_observation = _append_observation
    base._policy_action_dim = _policy_action_dim
    base._policy_input = _policy_input
    base._zero_env_action = _zero_env_action
    base._target_action_to_env_action = _absolute_joint_action_to_env_action

    logging.basicConfig(level=logging.INFO, force=True)
    base.eval_shellgame(tyro.cli(Args))


if __name__ == "__main__":
    main()
