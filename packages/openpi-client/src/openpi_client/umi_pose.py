"""Single-arm UMI action coordinates without importing the model runtime.

The pose convention matches umi-arx-deploy's ``utils/pose_repr_util.py`` and
``utils/pose_util.py``: model actions are position, the first two *rows* of a
rotation matrix, and gripper width. World commands use position, rotvec, grip.
Install ``openpi-client[umi]`` to use these optional geometry helpers.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp


def _finite_array(value, shape_tail: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != shape_tail or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite array with shape (N, {shape_tail})")
    return result


def _reference(current_tcp_pose, pose_repr: str):
    if pose_repr not in ("abs", "rel", "relative"):
        raise ValueError("pose_repr must be abs, rel, or relative; delta actions are unsupported by elec_meta RTC")
    pose = np.asarray(current_tcp_pose, dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("current_tcp_pose must be a finite (6,) position/rotvec pose")
    return pose[:3], Rotation.from_rotvec(pose[3:]).as_matrix()


def _matrix_from_rows(rows: np.ndarray) -> np.ndarray:
    first, second = rows[..., :3], rows[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("Model rotation rows must be nonzero and independent")
    first = first / first_norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("Model rotation rows must be nonzero and independent")
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def world_actions_to_model_actions(world_actions, current_tcp_pose, *, pose_repr: str) -> np.ndarray:
    """Rebase absolute ``(N,7)`` commands into unnormalized ``(N,10)`` actions.

    ``relative`` uses the current TCP's full SE(3) frame. ``rel`` preserves the
    deployed legacy convention: translation in world axes and right-relative
    rotation. The caller must use the same convention as its training data.
    """
    world = _finite_array(world_actions, 7, "world_actions")
    reference_pos, reference_rot = _reference(current_tcp_pose, pose_repr)
    position = world[:, :3].copy()
    rotation = Rotation.from_rotvec(world[:, 3:6]).as_matrix()
    if pose_repr == "relative":
        position = (position - reference_pos) @ reference_rot
        rotation = reference_rot.T @ rotation
    elif pose_repr == "rel":
        position = position - reference_pos
        rotation = rotation @ reference_rot.T
    return np.concatenate((position, rotation[:, :2, :].reshape(-1, 6), world[:, 6:7]), axis=-1)


def model_actions_to_world_actions(actions, current_tcp_pose, *, pose_repr: str) -> np.ndarray:
    """Decode unnormalized ``(N,10)`` model actions to absolute ``(N,7)`` poses.

    Normalize the predicted rotation rows with the same Gram-Schmidt rule as
    the existing UMI deploy client. This function does not apply IK, offsets,
    clamping, or gripper thresholding.
    """
    actions = _finite_array(actions, 10, "actions")
    reference_pos, reference_rot = _reference(current_tcp_pose, pose_repr)
    position = actions[:, :3].copy()
    rotation = _matrix_from_rows(actions[:, 3:9])
    if pose_repr == "relative":
        position = position @ reference_rot.T + reference_pos
        rotation = reference_rot @ rotation
    elif pose_repr == "rel":
        position = position + reference_pos
        rotation = rotation @ reference_rot
    rotvec = Rotation.from_matrix(rotation).as_rotvec()
    return np.concatenate((position, rotvec, actions[:, 9:10]), axis=-1)


def interpolate_world_actions(actions, timestamps, target_times) -> np.ndarray:
    """Interpolate an absolute plan onto a new observation's controller grid.

    Position and gripper width are linear; orientation uses shortest-path
    quaternion SLERP. Both time arrays must be strictly increasing, and every
    target must lie inside the original plan. No extrapolation is performed.
    """
    actions = _finite_array(actions, 7, "actions")
    timestamps = np.asarray(timestamps, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    for name, times in (("timestamps", timestamps), ("target_times", target_times)):
        if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError(f"{name} must be a nonempty finite strictly increasing time array")
    if len(timestamps) != len(actions):
        raise ValueError("timestamps must contain one time per action")
    if target_times[0] < timestamps[0] or target_times[-1] > timestamps[-1]:
        raise ValueError("target_times must stay within the original plan; extrapolation is forbidden")
    if len(timestamps) == 1:
        return np.repeat(actions, len(target_times), axis=0)
    result = np.empty((len(target_times), 7), dtype=np.float64)
    for dimension in (0, 1, 2, 6):
        result[:, dimension] = np.interp(target_times, timestamps, actions[:, dimension])
    origin = timestamps[0]
    rotations = Slerp(timestamps - origin, Rotation.from_rotvec(actions[:, 3:6]))(target_times - origin)
    result[:, 3:6] = rotations.as_rotvec()
    return result
