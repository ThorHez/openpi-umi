"""
Multi-dataset data loader for Pi0 Hybrid training.

Supports concatenating multiple LeRobot datasets with per-dataset transforms and
normalization stats. Each dataset specifies action_loss_mask in its DataConfig;
transform_dataset injects that mask into every sample, so each batch can mix
samples from different datasets and each sample has the correct per-sample mask.
"""

from __future__ import annotations

import logging
import math
import numbers
import time
from typing import Literal

import jax
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


class WeightedConcatDataset(torch.utils.data.Dataset):
    """Concatenates multiple datasets; supports weighted sampling via WeightedRandomSampler."""

    def __init__(self, datasets: list[torch.utils.data.Dataset], weights: list[float] | None = None):
        self._datasets = list(datasets)
        self._weights = None
        self._sampler_weights = None
        if weights is not None:
            if len(weights) != len(self._datasets):
                raise ValueError("weights length must match the number of datasets")
            validated = []
            for weight in weights:
                if isinstance(weight, bool) or not isinstance(weight, numbers.Real):
                    raise ValueError("Each dataset weight must be a positive finite number")
                value = float(weight)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Each dataset weight must be a positive finite number")
                validated.append(value)
            scale = max(validated, default=1.0)
            scaled = [weight / scale for weight in validated]
            if any(weight == 0 for weight in scaled):
                raise ValueError("Dataset weight dynamic range exceeds finite float64 precision")
            self._weights = validated  # Original per-row weights, not source mixture probabilities.
            self._sampler_weights = scaled
        self._cumulative = [0]
        for d in self._datasets:
            self._cumulative.append(self._cumulative[-1] + len(d))

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        for i in range(len(self._datasets)):
            if idx < self._cumulative[i + 1]:
                local_idx = idx - self._cumulative[i]
                return self._datasets[i][local_idx]
        raise IndexError(idx)

    def get_dataset_weights_for_sampler(self) -> list[float]:
        """Return scaled per-row weights with source mass proportional to N_i*w_i."""
        if self._sampler_weights is None:
            return [1.0] * len(self)
        out = []
        for i, d in enumerate(self._datasets):
            w = self._sampler_weights[i]
            out.extend([w] * len(d))
        return out


class MultiDataLoaderImpl(_data_loader.DataLoader):
    """DataLoader that exposes the first data_config for backward compat and all configs for checkpoint saving."""

    def __init__(
        self,
        data_configs: list[_config.DataConfig],
        data_loader: _data_loader.TorchDataLoader,
    ):
        self._data_configs = data_configs
        self._data_loader = data_loader
        self._timing = {
            "obs_from_dict_s": 0.0,
            "obs_from_dict_max_s": 0.0,
        }

    def data_config(self) -> _config.DataConfig:
        return self._data_configs[0]

    def data_configs(self) -> list[_config.DataConfig]:
        return self._data_configs

    def pop_timing(self) -> dict[str, float]:
        timing = dict(self._timing)
        self._timing = dict.fromkeys(self._timing, 0.0)
        timing.update(self._data_loader.pop_timing())
        return timing

    def __iter__(self):
        for batch in self._data_loader:
            obs_start = time.perf_counter()
            observation = _model.Observation.from_dict(batch)
            obs_from_dict_s = time.perf_counter() - obs_start
            self._timing["obs_from_dict_s"] += obs_from_dict_s
            self._timing["obs_from_dict_max_s"] = max(self._timing["obs_from_dict_max_s"], obs_from_dict_s)
            yield observation, batch["actions"]


def create_multi_data_loader(
    config: _config.TrainConfig,
    data_configs_and_weights: list[tuple[_config.DataConfig, float]],
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> _data_loader.DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader over multiple datasets with optional per-dataset sampling weights.

    Args:
        data_configs_and_weights: list of (DataConfig, weight) pairs. DataConfigs should be
            pre-built via MultiDataConfigFactory.create_all so that factory-level transforms
            (e.g. state_pad_dim) are already applied.

    Returns a MultiDataLoaderImpl that implements data_config() (first config) and
    data_configs() (all configs for checkpoint saving).
    """
    model_config = config.model
    action_horizon = config.model.action_horizon
    batch_size = config.batch_size
    num_workers = getattr(config, "num_workers", 0)
    seed = config.seed

    datasets = []
    data_configs = []
    weights = []

    for dc, w in data_configs_and_weights:
        data_configs.append(dc)
        logging.info(f"Multi-dataset entry: repo_id={dc.repo_id}, asset_id={dc.asset_id}")
        ds = _data_loader.create_torch_dataset(dc, action_horizon, model_config)
        ds = _data_loader.transform_dataset(ds, dc, skip_norm_stats=skip_norm_stats)
        datasets.append(ds)
        weights.append(w)

    # Validate every supplied weight, including equal invalid values which
    # must not silently become an unweighted sampler.
    concat = WeightedConcatDataset(datasets, weights=weights)

    for i, (dc, ds) in enumerate(zip(data_configs, datasets)):
        logging.info(f"  Dataset {i}: repo_id={dc.repo_id}, len={len(ds)}, weight={weights[i]}")

    # Sampler for weighted sampling when weights differ.
    # Perf: one multinomial(N)+tolist() per epoch; N=200k ~38ms, N=500k ~110ms -> <0.01% of epoch.
    sampler = None
    use_weights = concat._weights is not None and len(set(concat._weights)) > 1
    if use_weights:
        index_weights = torch.tensor(concat.get_dataset_weights_for_sampler(), dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            index_weights,
            num_samples=len(concat),
            replacement=True,
        )

    if framework == "pytorch" and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            concat,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle if sampler is None else False,
            drop_last=True,
        )
        local_batch_size = batch_size // torch.distributed.get_world_size()
    else:
        local_batch_size = batch_size // jax.process_count() if framework == "jax" else batch_size

    if len(concat) < local_batch_size:
        raise ValueError(
            f"Concatenated dataset size ({len(concat)}) is smaller than local_batch_size ({local_batch_size})."
        )

    torch_loader = _data_loader.TorchDataLoader(
        concat,
        image_keys=model_config.image_keys,
        depth_keys=model_config.depth_keys,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return MultiDataLoaderImpl(data_configs, torch_loader)
