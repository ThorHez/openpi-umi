"""论文 Ours/UOT 的跨场景联合数据加载器。

逻辑：BC 分支按显式数据集概率采样；UOT 分支读取离线 DTW 配对，在每个目标帧上
按轨迹距离加权选择辅助轨迹，再在 DTW 对齐点附近采样辅助帧。
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import jax
import numpy as np
import torch
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.multi_data_loader import MultiDataLoaderImpl, WeightedConcatDataset


def _canonical(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def create_probability_bc_loader(
    config: _config.TrainConfig,
    data_configs: list[_config.DataConfig],
    probabilities: tuple[float, ...],
    *,
    sharding: jax.sharding.Sharding,
) -> MultiDataLoaderImpl:
    """创建严格按数据集概率采样的 BC loader，而不是按数据集帧数隐式加权。"""
    if len(probabilities) != len(data_configs) or any(p <= 0 for p in probabilities):
        raise ValueError("bc_dataset_probabilities must contain one positive value per dataset")
    probability_sum = float(sum(probabilities))
    probabilities = tuple(float(p) / probability_sum for p in probabilities)

    datasets = []
    for data_config in data_configs:
        dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        datasets.append(_data_loader.transform_dataset(dataset, data_config))

    # WeightedRandomSampler 的概率与单样本权重×数据集长度成正比，因此用 p_i / N_i。
    per_item_weights = [probability / len(dataset) for probability, dataset in zip(probabilities, datasets, strict=True)]
    concat = WeightedConcatDataset(datasets, weights=per_item_weights)
    index_weights = torch.tensor(concat.get_dataset_weights_for_sampler(), dtype=torch.double)
    generator = torch.Generator().manual_seed(config.seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        index_weights,
        num_samples=len(concat),
        replacement=True,
        generator=generator,
    )
    torch_loader = _data_loader.TorchDataLoader(
        concat,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=False,
        sampler=sampler,
        num_workers=config.num_workers,
        seed=config.seed,
        image_keys=config.model.image_keys,
        depth_keys=config.model.depth_keys,
    )
    logging.info("UOT BC dataset probabilities: %s", probabilities)
    return MultiDataLoaderImpl(data_configs, torch_loader)


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
        if self._pair_info.get("schema_version") != 1:
            raise ValueError("Unsupported DTW pair file schema; regenerate it with precompute_dtw_pairs.py")

        self._datasets = datasets
        self._target_dataset_index = target_dataset_index
        self._distance_cutoff = distance_cutoff
        self._distance_sharpness = distance_sharpness
        self._temporal_window = temporal_window

        # 标签从 LeRobot 表的数值列读取并放进共享内存，避免把 185 万帧浮点数重复写入 JSON 和 worker。
        position_keys = self._pair_info.get("position_keys")
        if position_keys is None:
            position_keys = [self._pair_info["position_key"]]
        self._labels = []
        for data_config in data_configs:
            raw_dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
            label_parts = [
                np.asarray(raw_dataset.hf_dataset[position_key], dtype=np.float32)
                for position_key in position_keys
            ]
            labels = np.concatenate(
                [part.reshape(len(part), -1) for part in label_parts],
                axis=-1,
            )
            label_tensor = torch.from_numpy(labels).share_memory_()
            self._labels.append(label_tensor)

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
            "source_label": self._labels[source_dataset_index][source_episode["indices"][aligned_source_t]],
            "target_label": self._labels[self._target_dataset_index][target_episode["indices"][target_t]],
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
    bc_loader = create_probability_bc_loader(
        config,
        data_configs,
        config.uot.bc_dataset_probabilities,
        sharding=sharding,
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
        image_keys=config.model.image_keys,
        depth_keys=config.model.depth_keys,
    )
    return bc_loader, OTDataLoader(torch_loader)
