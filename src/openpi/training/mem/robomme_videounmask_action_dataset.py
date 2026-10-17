"""Small in-memory behavior-cloning dataset for VideoUnmask EEF7 control."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import h5py
import numpy as np

from openpi.tasks.robomme.videounmask import eef_action_adapter

DEFAULT_PROGRESS_STEPS = 128


@dataclasses.dataclass(frozen=True)
class ActionNormalization:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    pose_mean: np.ndarray
    pose_std: np.ndarray

    def to_json(self) -> dict[str, list[float]]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "pose_mean": self.pose_mean.tolist(),
            "pose_std": self.pose_std.tolist(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, list[float]]) -> ActionNormalization:
        return cls(**{key: np.asarray(value, dtype=np.float32) for key, value in payload.items()})


@dataclasses.dataclass(frozen=True)
class ActionArrays:
    features: np.ndarray
    target_crops: np.ndarray
    poses: np.ndarray
    close_targets: np.ndarray
    episode_indices: np.ndarray
    timesteps: np.ndarray
    phases: np.ndarray


PHASES = ("pregrasp", "descend", "close_hold", "lift")


def _episode_phases(eef_z: np.ndarray, close_targets: np.ndarray) -> np.ndarray:
    """Derive a monotonic four-stage manipulation contract from one demonstration."""
    count = len(eef_z)
    phases = np.zeros(count, dtype=np.int32)
    close_rows = np.flatnonzero(close_targets > 0.5)
    if len(close_rows) == 0:
        return phases
    first_close = int(close_rows[0])
    close_z = float(np.min(eef_z[max(first_close - 3, 0) : min(first_close + 8, count)]))
    descend_rows = np.flatnonzero(eef_z[: first_close + 1] <= close_z + 0.12)
    descend_start = int(descend_rows[0]) if len(descend_rows) else max(first_close - 20, 0)
    phases[descend_start:first_close] = 1
    # The close/settle stage ends once the demonstrated EEF has risen 2 cm
    # above the local contact height.  This is robust to gripper chatter.
    lift_rows = np.flatnonzero(eef_z[first_close:] >= close_z + 0.02)
    lift_start = first_close + int(lift_rows[0]) if len(lift_rows) else count
    lift_start = max(lift_start, min(first_close + 8, count))
    phases[first_close:lift_start] = 2
    phases[lift_start:] = 3
    return phases


def crop_target_image(
    image: np.ndarray,
    target_point_yx: np.ndarray,
    *,
    crop_extent: int = 64,
) -> np.ndarray:
    """Extract a fixed local RGB view centered on the memory target point."""
    if crop_extent < eef_action_adapter.TARGET_CROP_SIZE:
        raise ValueError("crop_extent must be at least TARGET_CROP_SIZE")
    image = np.asarray(image, dtype=np.uint8)
    half = crop_extent // 2
    y, x = np.rint(np.asarray(target_point_yx)).astype(int)
    padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode="edge")
    patch = padded[y : y + crop_extent, x : x + crop_extent]
    indices = np.rint(np.linspace(0, crop_extent - 1, eef_action_adapter.TARGET_CROP_SIZE)).astype(int)
    return patch[np.ix_(indices, indices)].astype(np.uint8, copy=False)


def build_action_feature(
    target_point_yx: np.ndarray,
    eef_state: np.ndarray,
    gripper_state: np.ndarray,
    joint_state: np.ndarray,
    rollout_step: int,
    *,
    progress_steps: int = DEFAULT_PROGRESS_STEPS,
) -> np.ndarray:
    """Build the raw target/state feature used identically in train and eval."""
    feature = np.concatenate(
        (
            np.asarray(target_point_yx, dtype=np.float32).reshape(2) / 255.0,
            np.asarray(eef_state, dtype=np.float32).reshape(6),
            np.asarray(gripper_state, dtype=np.float32).reshape(2),
            np.asarray(joint_state, dtype=np.float32).reshape(7),
            np.asarray([min(max(rollout_step, 0) / progress_steps, 1.0)], dtype=np.float32),
        )
    )
    if feature.shape != (eef_action_adapter.ACTION_FEATURE_DIM,):
        raise AssertionError(feature.shape)
    return feature


def load_single_target_action_arrays(
    h5_path: str | pathlib.Path,
    labels_path: str | pathlib.Path,
    *,
    episode_indices: list[int],
    progress_steps: int = DEFAULT_PROGRESS_STEPS,
    target_mode: str = "absolute",
    phase_conditioned: bool = False,
) -> ActionArrays:
    if target_mode not in {"absolute", "delta", "phase_waypoint_delta"}:
        raise ValueError(
            "Expected target_mode absolute, delta, or phase_waypoint_delta; "
            f"got {target_mode!r}"
        )
    payload = json.loads(pathlib.Path(labels_path).read_text(encoding="utf-8"))
    metadata = {int(item["episode_index"]): item for item in payload["episodes"]}
    features = []
    target_crops = []
    poses = []
    close_targets = []
    source_episodes = []
    source_timesteps = []
    source_phases = []
    with h5py.File(pathlib.Path(h5_path).expanduser().resolve(), "r") as source:
        for episode_index in episode_indices:
            item = metadata[int(episode_index)]
            if int(item["num_targets"]) != 1:
                continue
            episode = source[item["episode_name"]]
            start = int(item["execution_start"])
            end = int(item["num_steps"])
            target_point = np.asarray(item["target_point_yx"], dtype=np.float32)
            initial_crop = crop_target_image(
                episode[f"timestep_{start}/obs/front_rgb"][()],
                target_point,
            )
            row_timesteps = list(range(start, end - 1))
            row_eef_z = np.asarray(
                [episode[f"timestep_{timestep}/obs/eef_state"][()][2] for timestep in row_timesteps],
                dtype=np.float32,
            )
            row_close = np.asarray(
                [episode[f"timestep_{timestep + 1}/action/eef_action"][()][6] < 0.0 for timestep in row_timesteps],
                dtype=np.float32,
            )
            row_phases = _episode_phases(row_eef_z, row_close)
            row_eef_pose = np.stack(
                [episode[f"timestep_{timestep}/obs/eef_state"][()] for timestep in row_timesteps]
            ).astype(np.float32)
            first_descend = np.flatnonzero(row_phases == 1)
            first_close_phase = np.flatnonzero(row_phases == 2)
            descend_endpoint = int(first_descend[0]) if len(first_descend) else len(row_phases) - 1
            contact_endpoint = (
                int(first_close_phase[0]) if len(first_close_phase) else len(row_phases) - 1
            )
            phase_endpoints = np.stack(
                (
                    row_eef_pose[descend_endpoint],
                    row_eef_pose[contact_endpoint],
                    row_eef_pose[contact_endpoint],
                    row_eef_pose[-1],
                )
            )
            # Observation t supervises the command recorded at t+1.  HDF5
            # observations are post-step, so same-row actions would teach an
            # identity/stalling controller.
            for row_index, timestep in enumerate(row_timesteps):
                observation = episode[f"timestep_{timestep}/obs"]
                next_action = episode[f"timestep_{timestep + 1}/action/eef_action"][()]
                feature = build_action_feature(
                        target_point,
                        observation["eef_state"][()],
                        observation["gripper_state"][()],
                        observation["joint_state"][()],
                        timestep - start,
                        progress_steps=progress_steps,
                    )
                phase = int(row_phases[row_index])
                if phase_conditioned:
                    feature = np.concatenate((feature, np.eye(len(PHASES), dtype=np.float32)[phase]))
                features.append(feature)
                # Keep scene conditioning fixed at execution start.  A
                # teacher-forced current crop becomes robot-occluded during
                # rollout and creates a severe closed-loop distribution shift.
                target_crops.append(initial_crop)
                next_pose = np.asarray(next_action[:6], dtype=np.float32)
                if target_mode == "phase_waypoint_delta":
                    next_pose = phase_endpoints[phase].copy()
                if target_mode in {"delta", "phase_waypoint_delta"}:
                    current_pose = np.asarray(observation["eef_state"][()], dtype=np.float32)
                    next_pose = next_pose - current_pose
                    # RPY has an equivalent +/- pi representation.  The
                    # shortest wrapped difference avoids a spurious 2*pi
                    # target at that boundary.
                    next_pose[3:] = (next_pose[3:] + np.pi) % (2.0 * np.pi) - np.pi
                poses.append(next_pose)
                close_targets.append(float(next_action[6] < 0.0))
                source_episodes.append(episode_index)
                source_timesteps.append(timestep)
                source_phases.append(phase)
    return ActionArrays(
        features=np.stack(features),
        target_crops=np.stack(target_crops),
        poses=np.stack(poses),
        close_targets=np.asarray(close_targets, dtype=np.float32),
        episode_indices=np.asarray(source_episodes, dtype=np.int32),
        timesteps=np.asarray(source_timesteps, dtype=np.int32),
        phases=np.asarray(source_phases, dtype=np.int32),
    )


def compute_normalization(arrays: ActionArrays) -> ActionNormalization:
    return ActionNormalization(
        feature_mean=arrays.features.mean(axis=0).astype(np.float32),
        feature_std=np.maximum(arrays.features.std(axis=0), 1e-4).astype(np.float32),
        pose_mean=arrays.poses.mean(axis=0).astype(np.float32),
        pose_std=np.maximum(arrays.poses.std(axis=0), 1e-4).astype(np.float32),
    )


def normalize_arrays(arrays: ActionArrays, stats: ActionNormalization) -> ActionArrays:
    return dataclasses.replace(
        arrays,
        features=((arrays.features - stats.feature_mean) / stats.feature_std).astype(np.float32),
        poses=((arrays.poses - stats.pose_mean) / stats.pose_std).astype(np.float32),
    )
