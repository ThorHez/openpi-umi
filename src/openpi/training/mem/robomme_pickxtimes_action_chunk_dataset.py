"""Action-chunk targets aligned with the PickXtimes memory-action cache."""

from __future__ import annotations

import dataclasses
import pathlib

import h5py
import numpy as np

from openpi.training.mem import robomme_pickxtimes_action_dataset as action_data

SPATIAL_GRID_SIZE = 4
SPATIAL_VISUAL_TOKENS = SPATIAL_GRID_SIZE**2


def pool_spatial_patch_tokens(patch_tokens: np.ndarray) -> np.ndarray:
    """Pool a 16x16 SigLIP grid to 4x4 while preserving token positions."""
    patch_tokens = np.asarray(patch_tokens)
    if patch_tokens.ndim != 3 or patch_tokens.shape[1] != 256:
        raise ValueError(f"Expected [T,256,D] patch tokens, got {patch_tokens.shape}")
    time, _, width = patch_tokens.shape
    grid = patch_tokens.reshape(time, 16, 16, width)
    return grid.reshape(time, 4, 4, 4, 4, width).mean(axis=(2, 4)).reshape(time, 16, width)


def write_spatial_visual_cache(
    feature_path: str | pathlib.Path,
    base_cache_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> None:
    with h5py.File(pathlib.Path(base_cache_path).resolve(), "r") as base:
        episode_indices = np.asarray(base["episode_indices"], dtype=np.int32)
        timesteps = np.asarray(base["timesteps"], dtype=np.int32)
        frozen_test_accessed = bool(base.attrs["frozen_test_accessed"])
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(pathlib.Path(feature_path).resolve(), "r") as features, h5py.File(output_path, "w") as output:
        output.attrs.update(
            schema_version=1,
            source_features=str(pathlib.Path(feature_path).resolve()),
            base_cache=str(pathlib.Path(base_cache_path).resolve()),
            spatial_grid_size=SPATIAL_GRID_SIZE,
            spatial_visual_tokens=SPATIAL_VISUAL_TOKENS,
            frozen_test_accessed=frozen_test_accessed,
        )
        visual = output.create_dataset(
            "visual_features",
            shape=(len(timesteps), SPATIAL_VISUAL_TOKENS, 1152),
            dtype=np.float16,
            chunks=(1, SPATIAL_VISUAL_TOKENS, 1152),
        )
        output.create_dataset("episode_indices", data=episode_indices)
        output.create_dataset("timesteps", data=timesteps)
        unique_episodes = np.unique(episode_indices)
        for ordinal, episode_index in enumerate(unique_episodes, start=1):
            rows = np.flatnonzero(episode_indices == episode_index)
            episode_timesteps = timesteps[rows]
            patch_tokens = np.asarray(features[f"episode_{int(episode_index)}/patch_tokens"][episode_timesteps])
            visual[rows] = pool_spatial_patch_tokens(patch_tokens).astype(np.float16)
            print(
                f"[{ordinal}/{len(unique_episodes)}] spatial features episode={episode_index} rows={len(rows)}",
                flush=True,
            )


def load_spatial_visual_features(
    spatial_cache_path: str | pathlib.Path,
    *,
    episode_indices: list[int],
    expected_episodes: np.ndarray,
    expected_timesteps: np.ndarray,
) -> np.ndarray:
    with h5py.File(pathlib.Path(spatial_cache_path).resolve(), "r") as source:
        source_episodes = np.asarray(source["episode_indices"], dtype=np.int32)
        mask = np.isin(source_episodes, np.asarray(episode_indices, dtype=np.int32))
        selected_timesteps = np.asarray(source["timesteps"], dtype=np.int32)[mask]
        if not np.array_equal(source_episodes[mask], expected_episodes):
            raise ValueError("Spatial/base episode row alignment mismatch")
        if not np.array_equal(selected_timesteps, expected_timesteps):
            raise ValueError("Spatial/base timestep row alignment mismatch")
        return np.asarray(source["visual_features"], dtype=np.float16)[mask]


@dataclasses.dataclass(frozen=True)
class ActionChunkArrays:
    visual_features: np.ndarray
    robot_goal: np.ndarray
    poses: np.ndarray
    close_targets: np.ndarray
    action_mask: np.ndarray
    phase_targets: np.ndarray
    memory_bank: np.ndarray
    memory_indices: np.ndarray
    episode_indices: np.ndarray
    timesteps: np.ndarray

    def __len__(self) -> int:
        return len(self.visual_features)


def write_action_chunk_targets(
    h5_path: str | pathlib.Path,
    base_cache_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    *,
    action_horizon: int = 8,
) -> None:
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    with h5py.File(pathlib.Path(base_cache_path).resolve(), "r") as base:
        episode_indices = np.asarray(base["episode_indices"], dtype=np.int32)
        timesteps = np.asarray(base["timesteps"], dtype=np.int32)
        frozen_test_accessed = bool(base.attrs["frozen_test_accessed"])
    poses = np.empty((len(timesteps), action_horizon, 6), dtype=np.float32)
    close = np.empty((len(timesteps), action_horizon), dtype=np.bool_)
    valid = np.empty((len(timesteps), action_horizon), dtype=np.bool_)
    with h5py.File(pathlib.Path(h5_path).resolve(), "r") as source:
        for ordinal, episode_index in enumerate(np.unique(episode_indices), start=1):
            row_indices = np.flatnonzero(episode_indices == episode_index)
            episode = source[f"episode_{int(episode_index)}"]
            step_keys = [key for key in episode if key.startswith("timestep_")]
            num_steps = len(step_keys)
            raw_actions = np.stack(
                [
                    np.asarray(episode[f"timestep_{step}/action/eef_action"][()], dtype=np.float32)
                    for step in range(num_steps)
                ]
            )
            source_steps = timesteps[row_indices, None] + 1 + np.arange(action_horizon)[None]
            row_valid = source_steps < num_steps
            source_steps = np.minimum(source_steps, num_steps - 1)
            chunks = raw_actions[source_steps]
            poses[row_indices] = chunks[..., :6]
            close[row_indices] = chunks[..., 6] < 0.0
            valid[row_indices] = row_valid
            print(
                f"[{ordinal}/{len(np.unique(episode_indices))}] chunk targets episode={episode_index} "
                f"rows={len(row_indices)}",
                flush=True,
            )
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as output:
        output.attrs.update(
            schema_version=1,
            source_h5=str(pathlib.Path(h5_path).resolve()),
            base_cache=str(pathlib.Path(base_cache_path).resolve()),
            action_horizon=action_horizon,
            frozen_test_accessed=frozen_test_accessed,
            action_alignment=f"observation[t] -> eef_action[t+1:t+{action_horizon + 1}]",
        )
        for key, value in {
            "episode_indices": episode_indices,
            "timesteps": timesteps,
            "poses": poses,
            "close_targets": close,
            "action_mask": valid,
        }.items():
            output.create_dataset(key, data=value, compression="gzip", compression_opts=1)


def load_action_chunk_arrays(
    base_cache_path: str | pathlib.Path,
    chunk_cache_path: str | pathlib.Path,
    *,
    episode_indices: list[int],
    memory_mode: str,
) -> ActionChunkArrays:
    base = action_data.load_action_arrays(
        base_cache_path,
        episode_indices=episode_indices,
        memory_mode=memory_mode,
    )
    with h5py.File(pathlib.Path(chunk_cache_path).resolve(), "r") as source:
        source_episodes = np.asarray(source["episode_indices"], dtype=np.int32)
        mask = np.isin(source_episodes, np.asarray(episode_indices, dtype=np.int32))
        selected_timesteps = np.asarray(source["timesteps"], dtype=np.int32)[mask]
        if not np.array_equal(source_episodes[mask], base.episode_indices):
            raise ValueError("Base/chunk episode row alignment mismatch")
        if not np.array_equal(selected_timesteps, base.timesteps):
            raise ValueError("Base/chunk timestep row alignment mismatch")
        return ActionChunkArrays(
            visual_features=base.visual_features,
            robot_goal=base.robot_goal,
            poses=np.asarray(source["poses"], dtype=np.float32)[mask],
            close_targets=np.asarray(source["close_targets"], dtype=np.float32)[mask],
            action_mask=np.asarray(source["action_mask"], dtype=np.float32)[mask],
            phase_targets=base.phase_targets,
            memory_bank=base.memory_bank,
            memory_indices=base.memory_indices,
            episode_indices=base.episode_indices,
            timesteps=base.timesteps,
        )


def compute_normalization(arrays: ActionChunkArrays) -> action_data.ActionNormalization:
    valid = arrays.action_mask.astype(np.bool_)
    valid_poses = arrays.poses[valid]
    return action_data.ActionNormalization(
        robot_goal_mean=arrays.robot_goal.mean(axis=0).astype(np.float32),
        robot_goal_std=np.maximum(arrays.robot_goal.std(axis=0), 1e-4).astype(np.float32),
        pose_mean=valid_poses.mean(axis=0).astype(np.float32),
        pose_std=np.maximum(valid_poses.std(axis=0), 1e-4).astype(np.float32),
    )


def normalize_arrays(
    arrays: ActionChunkArrays,
    stats: action_data.ActionNormalization,
) -> ActionChunkArrays:
    return dataclasses.replace(
        arrays,
        robot_goal=((arrays.robot_goal - stats.robot_goal_mean) / stats.robot_goal_std).astype(np.float32),
        poses=((arrays.poses - stats.pose_mean) / stats.pose_std).astype(np.float32),
    )
