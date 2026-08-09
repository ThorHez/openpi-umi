"""ShellGame inference with episode-relative state and current-relative actions.

This entry point keeps ``main.py`` unchanged and reuses its rollout machinery.
V2 changes only the pose conventions and the OSC command mode:

    observation state: T_episode_start^-1 @ T_current
    policy action:      T_plan_anchor^-1 @ T_future
    environment action: absolute world-frame EEF pose
"""

# This V2 entry point intentionally reuses selected private helpers from the
# sibling script so main.py remains untouched.
# pylint: disable=protected-access

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from scipy.spatial.transform import Rotation
import tyro

import main as base


@dataclasses.dataclass
class Args(base.Args):
    # Frame used by each predicted 10D action waypoint:
    #   "current": T_current^-1 @ T_future (V2 training-label convention).
    #   "episode_start": T_start^-1 @ T_future (legacy convention).
    action_pose_frame: str = "current"

    # Rot6D matrix serialization:
    #   "openpi": flatten the first two rotation-matrix rows.
    #   "shellgame_legacy": flatten the first two columns.
    # This must exactly match the dataset converter and trained checkpoint.
    rot6d_convention: str = "openpi"

    # Robosuite OSC_POSE command representation:
    #   "absolute": env action is world-frame [xyz, rotation-vector, gripper].
    #   "delta": env action is normalized current-pose delta plus gripper.
    # V2 restores relative policy predictions to world targets, then uses
    # absolute OSC to execute those targets.
    osc_input_type: str = "absolute"


def _matrix_to_rot6d(mat: np.ndarray, convention: str) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    if convention == "openpi":
        return mat[:2, :].reshape(6).astype(np.float32)
    if convention == "shellgame_legacy":
        return mat[:, :2].reshape(6).astype(np.float32)
    raise ValueError(f"Unknown rot6d convention: {convention}")


def _quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = max(float(np.linalg.norm([x, y, z, w])), 1e-8)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _episode_namespace(args: Args, *, seed: int, initial_ball_cup: str, num_swaps: int):
    namespace = _ORIGINAL_EPISODE_NAMESPACE(
        args,
        seed=seed,
        initial_ball_cup=initial_ball_cup,
        num_swaps=num_swaps,
    )
    namespace.osc_input_type = args.osc_input_type
    return namespace


def _policy_input(
    history: list[dict],
    start_eef_pos: np.ndarray,
    *,
    args: Args,
    prompt: str | None = None,
) -> dict:
    frames = base._window(history, num_frames=args.num_frames, frame_stride=args.frame_stride)
    start = history[0]
    current = history[-1]

    start_world = np.eye(4, dtype=np.float64)
    start_world[:3, :3] = _quat_to_matrix(start["eef_quat"])
    start_world[:3, 3] = np.asarray(start_eef_pos, dtype=np.float64)
    current_world = np.eye(4, dtype=np.float64)
    current_world[:3, :3] = _quat_to_matrix(current["eef_quat"])
    current_world[:3, 3] = np.asarray(current["eef_pos"], dtype=np.float64)
    current_wrt_start = np.linalg.inv(start_world) @ current_world

    eef_rel = current_wrt_start[:3, 3].astype(np.float32)
    rot6d = _matrix_to_rot6d(current_wrt_start[:3, :3], args.rot6d_convention)
    identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    width = np.array([current["gripper_width"]], dtype=np.float32)

    element = {
        "robot0_eef_pos": np.stack([eef_rel, np.zeros(3, dtype=np.float32)], axis=0),
        "robot0_eef_rot_axis_angle": np.stack([rot6d, identity_rot6d], axis=0),
        "robot0_gripper_width": np.stack([width, width], axis=0),
        "actions": np.zeros((args.action_horizon, args.action_dim), dtype=np.float32),
        "prompt": args.task if prompt is None else prompt,
    }
    for i, frame in enumerate(frames):
        element[f"left_wrist_0_rgb_0_{i}"] = frame["wrist"]
        element[f"left_wrist_0_rgb_1_{i}"] = frame["base"]
    return element


def _zero_env_action(env, gripper_action: float) -> np.ndarray:
    """Hold the achieved pose instead of commanding world-frame zero."""
    action_low, action_high = env.action_spec
    action = np.zeros_like(action_low)
    robot = env.robots[0]
    arm = next(iter(robot.eef_site_id))
    controller = robot.part_controllers[arm]
    if controller.input_type == "absolute":
        site_id = robot.eef_site_id[arm]
        action[:3] = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64)
        site_mat = np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        action[3:6] = Rotation.from_matrix(site_mat).as_rotvec()
    action[-1] = gripper_action
    return np.clip(action, action_low, action_high)


_ORIGINAL_EPISODE_NAMESPACE = base._episode_namespace


def main() -> None:
    # PolicyReplanner resolves these helpers through main.py's module globals,
    # so patch only this V2 process; importing or running main.py is unchanged.
    base._episode_namespace = _episode_namespace
    base._policy_input = _policy_input
    base._zero_env_action = _zero_env_action

    logging.basicConfig(level=logging.INFO, force=True)
    base.eval_shellgame(tyro.cli(Args))


if __name__ == "__main__":
    main()
