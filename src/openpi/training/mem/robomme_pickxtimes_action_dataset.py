"""Cached behavior-cloning arrays for PickXtimes memory-action probes."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import h5py
import numpy as np

from openpi.tasks.robomme.pickxtimes import eef_action_adapter

COLOR_TO_ID = {"red": 0, "green": 1, "blue": 2}
MEMORY_MODES = ("action_only", "initial_memory", "oracle", "predicted")


@dataclasses.dataclass(frozen=True)
class ActionNormalization:
    robot_goal_mean: np.ndarray
    robot_goal_std: np.ndarray
    pose_mean: np.ndarray
    pose_std: np.ndarray

    def to_json(self) -> dict[str, list[float]]:
        return {field.name: getattr(self, field.name).tolist() for field in dataclasses.fields(self)}

    @classmethod
    def from_json(cls, payload: dict[str, list[float]]) -> ActionNormalization:
        return cls(**{key: np.asarray(value, dtype=np.float32) for key, value in payload.items()})


@dataclasses.dataclass(frozen=True)
class ActionArrays:
    visual_features: np.ndarray
    robot_goal: np.ndarray
    poses: np.ndarray
    close_targets: np.ndarray
    phase_targets: np.ndarray
    memory_bank: np.ndarray
    memory_indices: np.ndarray
    episode_indices: np.ndarray
    timesteps: np.ndarray

    def __len__(self) -> int:
        return len(self.visual_features)


def build_robot_goal(
    eef_state: np.ndarray,
    gripper_state: np.ndarray,
    joint_state: np.ndarray,
    *,
    target_color: str,
    required_count: int,
) -> np.ndarray:
    color = np.eye(3, dtype=np.float32)[COLOR_TO_ID[target_color]]
    result = np.concatenate(
        (
            np.asarray(eef_state, dtype=np.float32).reshape(6),
            np.asarray(gripper_state, dtype=np.float32).reshape(2),
            np.asarray(joint_state, dtype=np.float32).reshape(7),
            color,
            np.asarray([required_count / 5.0], dtype=np.float32),
        )
    )
    if result.shape != (eef_action_adapter.ROBOT_GOAL_DIM,):
        raise AssertionError(result.shape)
    return result


def phase_at_timestep(events: list[dict], timestep: int) -> int:
    for event in events:
        if int(event["start"]) <= timestep < int(event["end"]):
            return int(event["event_type_id"])
    return int(events[-1]["event_type_id"])


def latest_memory_offsets(visible_timesteps: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    """Return 0 for initial memory and k+1 after stage k becomes visible."""
    return np.searchsorted(
        np.asarray(visible_timesteps, dtype=np.int32),
        np.asarray(timesteps, dtype=np.int32),
        side="right",
    ).astype(np.int32)


def write_action_cache(
    h5_path: str | pathlib.Path,
    feature_path: str | pathlib.Path,
    label_path: str | pathlib.Path,
    split_path: str | pathlib.Path,
    memory_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    *,
    frame_stride: int = 2,
) -> None:
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    labels_payload = json.loads(pathlib.Path(label_path).read_text(encoding="utf-8"))
    metadata = {int(item["episode_index"]): item for item in labels_payload["episodes"]}
    split = json.loads(pathlib.Path(split_path).read_text(encoding="utf-8"))
    train_indices = [int(value) for value in split["train_episode_indices"]]
    dev_indices = [int(value) for value in split["val_episode_indices"]]
    selected = train_indices + dev_indices
    if set(selected) & {int(value) for value in split.get("test_episode_indices", [])}:
        raise ValueError("Frozen-test leakage in PickXtimes action cache")

    visual_rows = []
    robot_rows = []
    pose_rows = []
    close_rows = []
    phase_rows = []
    episode_rows = []
    timestep_rows = []
    memory_indices: dict[str, list[np.ndarray]] = {"oracle": [], "predicted": []}
    memory_banks: dict[str, list[np.ndarray]] = {"oracle": [], "predicted": []}
    memory_counts = {"oracle": 0, "predicted": 0}

    with (
        h5py.File(pathlib.Path(h5_path).resolve(), "r") as source,
        h5py.File(pathlib.Path(feature_path).resolve(), "r") as features,
        h5py.File(pathlib.Path(memory_path).resolve(), "r") as memories,
    ):
        for ordinal, episode_index in enumerate(selected, start=1):
            item = metadata[episode_index]
            episode = source[item["episode_name"]]
            feature_episode = features[item["episode_name"]]
            memory_episode = memories[item["episode_name"]]
            timesteps = np.arange(0, int(item["num_steps"]) - 1, frame_stride, dtype=np.int32)
            patch_tokens = np.asarray(feature_episode["patch_tokens"][::frame_stride][: len(timesteps)])
            visual_rows.append(patch_tokens.astype(np.float32).mean(axis=1).astype(np.float16))

            per_episode_robot = []
            per_episode_pose = []
            for timestep in timesteps:
                observation = episode[f"timestep_{int(timestep)}/obs"]
                next_action = np.asarray(
                    episode[f"timestep_{int(timestep) + 1}/action/eef_action"][()], dtype=np.float32
                )
                per_episode_robot.append(
                    build_robot_goal(
                        observation["eef_state"][()],
                        observation["gripper_state"][()],
                        observation["joint_state"][()],
                        target_color=item["target_color"],
                        required_count=int(item["required_count"]),
                    )
                )
                per_episode_pose.append(next_action[:6])
                close_rows.append(next_action[6] < 0.0)
                phase_rows.append(phase_at_timestep(item["events"], int(timestep)))
            robot_rows.append(np.stack(per_episode_robot))
            pose_rows.append(np.stack(per_episode_pose))
            episode_rows.append(np.full(len(timesteps), episode_index, dtype=np.int32))
            timestep_rows.append(timesteps)

            initial = np.asarray(memory_episode["initial_memory"], dtype=np.float16)[None]
            for mode in ("oracle", "predicted"):
                mode_group = memory_episode[mode]
                stages = np.asarray(mode_group["stage_memories"], dtype=np.float16)
                bank = np.concatenate((initial, stages), axis=0)
                base = memory_counts[mode]
                offsets = latest_memory_offsets(np.asarray(mode_group["visible_timesteps"]), timesteps)
                memory_indices[mode].append(offsets + base)
                memory_banks[mode].append(bank)
                memory_counts[mode] += len(bank)
            print(f"[{ordinal}/{len(selected)}] action rows episode={episode_index}: {len(timesteps)}", flush=True)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as output:
        output.attrs.update(
            schema_version=1,
            frame_stride=frame_stride,
            train_episode_indices=np.asarray(train_indices, dtype=np.int32),
            dev_episode_indices=np.asarray(dev_indices, dtype=np.int32),
            frozen_test_accessed=False,
            action_alignment="observation[t] -> eef_action[t+1]",
        )
        datasets = {
            "visual_features": np.concatenate(visual_rows),
            "robot_goal": np.concatenate(robot_rows),
            "poses": np.concatenate(pose_rows),
            "close_targets": np.asarray(close_rows, dtype=np.bool_),
            "phase_targets": np.asarray(phase_rows, dtype=np.int8),
            "episode_indices": np.concatenate(episode_rows),
            "timesteps": np.concatenate(timestep_rows),
        }
        for key, value in datasets.items():
            output.create_dataset(key, data=value, compression="gzip", compression_opts=1)
        for mode in ("oracle", "predicted"):
            group = output.create_group(mode)
            group.create_dataset(
                "memory_bank", data=np.concatenate(memory_banks[mode]), compression="gzip", compression_opts=1
            )
            group.create_dataset("memory_indices", data=np.concatenate(memory_indices[mode]))


def load_action_arrays(
    cache_path: str | pathlib.Path,
    *,
    episode_indices: list[int],
    memory_mode: str,
) -> ActionArrays:
    if memory_mode not in MEMORY_MODES:
        raise ValueError(f"Unknown memory mode {memory_mode!r}; expected {MEMORY_MODES}")
    with h5py.File(pathlib.Path(cache_path).resolve(), "r") as source:
        source_episodes = np.asarray(source["episode_indices"], dtype=np.int32)
        mask = np.isin(source_episodes, np.asarray(episode_indices, dtype=np.int32))
        if not np.any(mask):
            raise ValueError("No action rows selected")
        if memory_mode == "action_only":
            memory_bank = np.zeros(
                (1, eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH), dtype=np.float16
            )
            selected_memory_indices = np.zeros(int(mask.sum()), dtype=np.int32)
        elif memory_mode == "initial_memory":
            memory_bank = np.asarray(source["predicted/memory_bank"], dtype=np.float16)
            all_memory_indices = np.asarray(source["predicted/memory_indices"], dtype=np.int32)
            selected_memory_indices = np.empty(int(mask.sum()), dtype=np.int32)
            selected_episodes = source_episodes[mask]
            for episode_index in np.unique(selected_episodes):
                # Banks are appended episode-by-episode with initial memory
                # first, so the minimum referenced index is the initial state.
                initial_index = int(all_memory_indices[source_episodes == episode_index].min())
                selected_memory_indices[selected_episodes == episode_index] = initial_index
        else:
            memory_bank = np.asarray(source[f"{memory_mode}/memory_bank"], dtype=np.float16)
            selected_memory_indices = np.asarray(source[f"{memory_mode}/memory_indices"], dtype=np.int32)[mask]
        return ActionArrays(
            visual_features=np.asarray(source["visual_features"], dtype=np.float16)[mask],
            robot_goal=np.asarray(source["robot_goal"], dtype=np.float32)[mask],
            poses=np.asarray(source["poses"], dtype=np.float32)[mask],
            close_targets=np.asarray(source["close_targets"], dtype=np.float32)[mask],
            phase_targets=np.asarray(source["phase_targets"], dtype=np.int32)[mask],
            memory_bank=memory_bank,
            memory_indices=selected_memory_indices,
            episode_indices=source_episodes[mask],
            timesteps=np.asarray(source["timesteps"], dtype=np.int32)[mask],
        )


def compute_normalization(arrays: ActionArrays) -> ActionNormalization:
    return ActionNormalization(
        robot_goal_mean=arrays.robot_goal.mean(axis=0).astype(np.float32),
        robot_goal_std=np.maximum(arrays.robot_goal.std(axis=0), 1e-4).astype(np.float32),
        pose_mean=arrays.poses.mean(axis=0).astype(np.float32),
        pose_std=np.maximum(arrays.poses.std(axis=0), 1e-4).astype(np.float32),
    )


def normalize_arrays(arrays: ActionArrays, stats: ActionNormalization) -> ActionArrays:
    return dataclasses.replace(
        arrays,
        robot_goal=((arrays.robot_goal - stats.robot_goal_mean) / stats.robot_goal_std).astype(np.float32),
        poses=((arrays.poses - stats.pose_mean) / stats.pose_std).astype(np.float32),
    )
