"""论文 Ours/UOT 的跨场景联合数据加载器。

逻辑：BC 分支复用标准多数据集 loader 的逐样本隐式加权；UOT 分支读取离线 DTW 配对，在每个目标帧上
按轨迹距离加权选择辅助轨迹，再在 DTW 对齐点附近采样辅助帧。
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import jax
import numpy as np
import torch
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.multi_data_loader import MultiDataLoaderImpl, create_multi_data_loader


def _canonical(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _current_frame(column: np.ndarray, *, expected_dim: int, key: str) -> np.ndarray:
    column = np.asarray(column, dtype=np.float32)
    if column.ndim >= 3:
        column = column[:, -1]
    column = column.reshape(len(column), -1)
    if column.shape[-1] != expected_dim:
        raise ValueError(f"{key} must end in {expected_dim} values, got {column.shape}")
    return column


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), eps)


def _rot6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    first = _normalize(rotation_6d[..., :3])
    second = rotation_6d[..., 3:]
    second = _normalize(second - np.sum(first * second, axis=-1, keepdims=True) * first)
    third = np.cross(first, second, axis=-1)
    return np.stack((first, second, third), axis=-2)


def _build_state_labels(raw_dataset, state_keys: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Build current-frame SE(3)+gripper labels, rebased to each episode's first frame."""
    episode_indices = np.asarray(raw_dataset.hf_dataset["episode_index"]).reshape(-1)
    frame_indices = np.asarray(raw_dataset.hf_dataset["frame_index"]).reshape(-1)
    positions = [
        _current_frame(np.asarray(raw_dataset.hf_dataset[key]), expected_dim=3, key=key)
        for key in state_keys["translation"]
    ]
    rotations = [
        _current_frame(np.asarray(raw_dataset.hf_dataset[key]), expected_dim=6, key=key)
        for key in state_keys["rotation"]
    ]
    grippers = [
        _current_frame(np.asarray(raw_dataset.hf_dataset[key]), expected_dim=1, key=key)
        for key in state_keys["gripper"]
    ]
    labels = {
        "translation": np.empty((len(episode_indices), 3 * len(positions)), dtype=np.float32),
        "rotation": np.empty((len(episode_indices), 9 * len(rotations)), dtype=np.float32),
        "gripper": np.empty((len(episode_indices), len(grippers)), dtype=np.float32),
    }
    for episode_id in np.unique(episode_indices):
        indices = np.flatnonzero(episode_indices == episode_id)
        indices = indices[np.argsort(frame_indices[indices])]
        translation_parts = []
        rotation_parts = []
        gripper_parts = []
        for position, rotation, gripper in zip(positions, rotations, grippers, strict=True):
            episode_position = position[indices]
            episode_rotation = _rot6d_to_matrix(rotation[indices])
            initial_rotation_inverse = episode_rotation[0].T
            relative_rotation = np.einsum("ij,tjk->tik", initial_rotation_inverse, episode_rotation)
            relative_translation = np.einsum(
                "ij,tj->ti", initial_rotation_inverse, episode_position - episode_position[0]
            )
            translation_parts.append(relative_translation)
            rotation_parts.append(relative_rotation.reshape(len(indices), 9))
            gripper_parts.append(gripper[indices])
        labels["translation"][indices] = np.concatenate(translation_parts, axis=-1)
        labels["rotation"][indices] = np.concatenate(rotation_parts, axis=-1)
        labels["gripper"][indices] = np.concatenate(gripper_parts, axis=-1)
    return labels


class DTWPairedDataset(torch.utils.data.Dataset):
    """把 DTW 元数据与已完成 OpenPI transforms 的各场景数据集拼成成对样本。"""

    def __init__(
        self,
        datasets: list[torch.utils.data.Dataset],
        data_configs: list[_config.DataConfig],
        pair_info_path: str,
        *,
        target_dataset_index: int,
        distance_cutoff: float,
        distance_sharpness: float,
        temporal_window: int,
    ):
        pair_path = Path(pair_info_path)
        pair_file = (
            gzip.open(pair_path, "rt", encoding="utf-8")
            if pair_path.suffix == ".gz"
            else pair_path.open("r", encoding="utf-8")
        )
        with pair_file as file:
            self._pair_info = json.load(file)
        if self._pair_info.get("schema_version") != 2:
            raise ValueError("Unsupported DTW pair file schema; regenerate it with precompute_dtw_pairs.py")

        self._datasets = datasets
        self._target_dataset_index = target_dataset_index
        self._distance_cutoff = distance_cutoff
        self._distance_sharpness = distance_sharpness
        self._temporal_window = temporal_window

        # 标签与离线 DTW 使用完全相同的当前帧 SE(3)+绝对夹爪定义，并放进共享内存。
        state_keys = self._pair_info["state_keys"]
        self._labels = []
        for data_config in data_configs:
            raw_dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
            labels = _build_state_labels(raw_dataset, state_keys)
            self._labels.append(
                {name: torch.from_numpy(values).share_memory_() for name, values in labels.items()}
            )

        config_by_repo = {_canonical(dc.repo_id): index for index, dc in enumerate(data_configs)}
        self._source_dataset_indices = []
        for source in self._pair_info["sources"]:
            repo = _canonical(source["repo_id"])
            if repo not in config_by_repo:
                raise ValueError(f"DTW source dataset is absent from config: {source['repo_id']}")
            self._source_dataset_indices.append(config_by_repo[repo])
        expected_target = _canonical(self._pair_info["target"]["repo_id"])
        actual_target = _canonical(data_configs[target_dataset_index].repo_id)
        if expected_target != actual_target:
            raise ValueError(f"DTW target mismatch: file={expected_target}, config={actual_target}")

        self._target_episodes = self._pair_info["target"]["episodes"]
        self._pairs_by_target_episode = {
            item["target_episode_list_index"]: item["candidates"] for item in self._pair_info["pairs"]
        }
        self._flat_target_frames = [
            (episode_index, local_t)
            for episode_index, episode in enumerate(self._target_episodes)
            for local_t in range(len(episode["indices"]))
        ]
        if not self._flat_target_frames:
            raise ValueError("DTW pair file contains no target frames")

    def __len__(self) -> int:
        return len(self._flat_target_frames)

    def __getitem__(self, index: int):
        target_episode_index, target_t = self._flat_target_frames[index]
        target_episode = self._target_episodes[target_episode_index]
        candidates = self._pairs_by_target_episode[target_episode_index]
        distances = torch.tensor([candidate["normalized_distance"] for candidate in candidates], dtype=torch.float64)
        weights = torch.sigmoid(self._distance_sharpness * (self._distance_cutoff - distances))
        if float(weights.sum()) <= 0:
            weights = torch.ones_like(weights)
        candidate = candidates[int(torch.multinomial(weights, 1).item())]

        source_repo_index = candidate["source_repo_index"]
        source_episode_index = candidate["source_episode_list_index"]
        source_episode = self._pair_info["sources"][source_repo_index]["episodes"][source_episode_index]
        aligned_source_times = candidate["target_to_source"][target_t]
        aligned_source_t = int(aligned_source_times[int(torch.randint(len(aligned_source_times), (1,)).item())])
        if self._temporal_window > 0:
            offset = int(torch.randint(-self._temporal_window, self._temporal_window + 1, (1,)).item())
            aligned_source_t = min(max(aligned_source_t + offset, 0), len(source_episode["indices"]) - 1)

        source_dataset_index = self._source_dataset_indices[source_repo_index]
        source_sample = self._datasets[source_dataset_index][source_episode["indices"][aligned_source_t]]
        target_sample = self._datasets[self._target_dataset_index][target_episode["indices"][target_t]]
        return {
            "source": source_sample,
            "target": target_sample,
            "source_label": {
                name: values[source_episode["indices"][aligned_source_t]]
                for name, values in self._labels[source_dataset_index].items()
            },
            "target_label": {
                name: values[target_episode["indices"][target_t]]
                for name, values in self._labels[self._target_dataset_index].items()
            },
        }


class OTDataLoader:
    """将成对字典批次转换为训练步所需的 Observation 与状态标签。"""

    def __init__(self, torch_loader: _data_loader.TorchDataLoader):
        self._torch_loader = torch_loader

    def __iter__(self):
        for batch in self._torch_loader:
            yield (
                _model.Observation.from_dict(batch["source"]),
                _model.Observation.from_dict(batch["target"]),
                batch["source_label"],
                batch["target_label"],
            )


def create_uot_loaders(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding,
) -> tuple[MultiDataLoaderImpl, OTDataLoader]:
    """一次构建共享 transforms/norm-stats 的 BC 与 UOT 两条数据流。"""
    if config.uot is None or not config.uot.enabled:
        raise ValueError("The selected config does not enable UOT")
    if not isinstance(config.data, _config.MultiDataConfigFactory):
        raise TypeError("UOT co-training requires MultiDataConfigFactory")
    if config.uot.ot_batch_size % jax.device_count() != 0:
        raise ValueError("uot.ot_batch_size must be divisible by the number of JAX devices")

    data_configs = config.data.create_all(config.assets_dirs, config.model)
    weights = config.data.weights
    data_configs_and_weights = [
        (data_config, weights[index] if weights and index < len(weights) else 1.0)
        for index, data_config in enumerate(data_configs)
    ]
    bc_loader = create_multi_data_loader(
        config,
        data_configs_and_weights,
        sharding=sharding,
        shuffle=True,
    )
    datasets = []
    for data_config in data_configs:
        dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        datasets.append(_data_loader.transform_dataset(dataset, data_config))
    paired_dataset = DTWPairedDataset(
        datasets,
        data_configs,
        config.uot.pair_info_path,
        target_dataset_index=config.uot.target_dataset_index,
        distance_cutoff=config.uot.dtw_distance_cutoff,
        distance_sharpness=config.uot.dtw_distance_sharpness,
        temporal_window=config.uot.temporal_window,
    )
    torch_loader = _data_loader.TorchDataLoader(
        paired_dataset,
        local_batch_size=config.uot.ot_batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed + 1,
    )
    return bc_loader, OTDataLoader(torch_loader)
