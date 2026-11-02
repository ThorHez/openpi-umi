"""Compute statistics before normalization and model padding.

Shared statistics scan every transformed row twice with per-row source weights.
Existing per-dataset quantiles are never averaged. Dataset processors may still
decode images before the numeric fields are selected.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from openpi.shared.weighted_stats import compute_weighted_statistics


class RemoveStrings:
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class _NumericDataset:
    """Keep measured fields after the actual dataset transforms."""

    def __init__(self, dataset, limit=None):
        self.dataset = dataset
        self.limit = min(len(dataset), limit) if limit is not None else len(dataset)

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        sample = self.dataset[index]
        return {key: np.asarray(sample[key]) for key in ("state", "actions")}


def _numeric_collate(samples):
    return {key: np.stack([sample[key] for sample in samples]) for key in ("state", "actions")}


def create_torch_dataloader(data_config, action_horizon, batch_size, model_config, num_workers, max_frames=None):
    """A finite, sequential loader including small datasets and tail rows."""
    import torch

    import openpi.training.data_loader as data_loader

    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be nonnegative")
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer")
    dataset = data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = data_loader.TransformedDataset(
        dataset, [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
    )
    dataset = _NumericDataset(dataset, max_frames)
    if not len(dataset):
        raise ValueError(f"Dataset {data_config.repo_id!r} is empty")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=_numeric_collate,
        **({"multiprocessing_context": "spawn"} if num_workers else {}),
    )
    return loader, math.ceil(len(dataset) / batch_size)


def create_rlds_dataloader(data_config, action_horizon, batch_size, max_frames=None):
    """Retain the single-source RLDS entry point and its dataset batching."""
    import openpi.training.data_loader as data_loader

    dataset = data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = data_loader.IterableTransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, RemoveStrings()],
        is_batched=True,
    )
    count = min(len(dataset), max_frames) if max_frames is not None else len(dataset)
    num_batches = count // batch_size
    return data_loader.RLDSDataLoader(dataset, num_batches=num_batches), num_batches


def _local_path(value):
    if not str(value).strip():
        raise ValueError("Statistics output path must not be empty")
    if "://" in str(value):
        raise ValueError("Statistics output must be local; provide --output-dir")
    return Path(value).expanduser().resolve()


def _assets_directory(factory, config):
    asset_id = factory.assets.asset_id or factory.repo_id
    if not asset_id or not isinstance(asset_id, str):
        raise ValueError("A statistics asset_id or repo_id is required")
    return _local_path(factory.assets.assets_dir or config.assets_dirs) / asset_id


def _check_output(output, factories, config, *, multi):
    """Protect datasets and, in shared mode, every old statistics asset."""
    for factory in factories:
        repo_id = str(factory.repo_id)
        if "://" not in repo_id and (Path(repo_id).is_absolute() or Path(repo_id).exists()):
            dataset = _local_path(repo_id)
            if output.is_relative_to(dataset):
                raise ValueError(f"Statistics output must be outside input dataset {dataset}")
        if multi and "://" not in str(factory.assets.assets_dir or config.assets_dirs):
            old_assets = _assets_directory(factory, config).resolve()
            if output.is_relative_to(old_assets):
                raise ValueError(f"Shared output must be outside input statistics asset {old_assets}")


def _publish(output: Path, document: dict, *, overwrite: bool):
    output.mkdir(parents=True, exist_ok=True)
    target = output / "norm_stats.json"
    descriptor, temporary = tempfile.mkstemp(prefix=".norm-stats-", suffix=".tmp", dir=output)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(document, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            # An exclusive link publishes the complete file and refuses a
            # destination concurrently created by another statistics process.
            os.link(temporary, target)
        directory = os.open(output, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_file_config(config_file, output_dir):
    """Load an explicitly selected local builder without invoking training."""
    if "://" in str(config_file):
        raise ValueError("--config-file must name an existing local Python file; URLs are not supported")
    path = Path(config_file).expanduser().resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ValueError("--config-file must name an existing local Python .py file")
    name = "_openpi_norm_stats_config_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the local --config-file Python module")
    module = importlib.util.module_from_spec(spec)
    # Retain the module so locally defined configuration classes remain
    # inspectable by the processor provenance checks during both scans.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    builder = getattr(module, "build_config", None)
    if not callable(builder) or inspect.iscoroutinefunction(builder):
        raise ValueError("--config-file must declare a synchronous callable build_config(output_dir)")
    placeholder = str(_local_path(output_dir) / ".config-only-output")
    try:
        inspect.signature(builder).bind(placeholder)
    except (TypeError, ValueError) as exc:
        raise ValueError("build_config must accept one output_dir argument") from exc
    config = builder(placeholder)
    if not all(hasattr(config, attribute) for attribute in ("data", "assets_dirs", "model", "batch_size", "num_workers")):
        raise ValueError("build_config(output_dir) did not return a training configuration")
    return config


def main(
    config_name: str | None = None,
    max_frames: int | None = None,
    output_dir: str | None = None,
    overwrite: bool = False,
    config_file: str | None = None,
):
    if (config_name is None) == (config_file is None):
        raise ValueError("Specify exactly one of --config-name or --config-file")
    if config_name is not None and not config_name.strip():
        raise ValueError("--config-name must not be empty")
    if config_file is not None and output_dir is None:
        raise ValueError("--config-file requires an independent --output-dir")
    from openpi.shared.runtime_paths import configure_training_cache

    configure_training_cache()
    # Set cache environment before importing the training dependencies.
    import openpi.shared.normalize as normalize
    import openpi.training.config as config_module

    config = _load_file_config(config_file, output_dir) if config_file is not None else config_module.get_config(config_name)
    factory = config.data
    multi = isinstance(factory, config_module.MultiDataConfigFactory)
    if multi:
        from openpi.training.shared_norm import source_descriptors, validate_shared_metadata

        if max_frames is not None:
            raise ValueError("Shared multi-dataset statistics require full coverage; omit --max-frames")
        shared_path = getattr(factory, "shared_norm_stats_path", None)
        if output_dir is None and shared_path is None:
            raise ValueError("Shared multi-dataset statistics require an independent --output-dir")
        output = _local_path(output_dir if output_dir is not None else shared_path)
        if output_dir is None and output.suffix == ".json":
            if output.name != "norm_stats.json":
                raise ValueError("shared_norm_stats_path must name norm_stats.json or its directory")
            output = output.parent
        factories = list(factory.datasets)
        weights = list(factory.weights) if factory.weights is not None else [1.0] * len(factories)
        if not factories or len(weights) != len(factories):
            raise ValueError("Shared statistics require nonempty datasets and one weight per dataset")
        _check_output(output, factories, config, multi=True)
        if (output / "norm_stats.json").exists() and not overwrite:
            raise FileExistsError("Statistics already exist; choose another output directory or use --overwrite")
        before = source_descriptors(factories, weights)
        if len({source["compatibility_sha256"] for source in before}) != 1:
            raise ValueError("Shared statistics require compatible state/action processing across every dataset")
        unbound = dataclasses.replace(factory, use_merged_norm_stats=False, shared_norm_stats_path=None)
        data_configs = unbound.create_all(config.assets_dirs, config.model)
        if any(dc.rlds_data_dir is not None for dc in data_configs):
            raise ValueError("Full shared statistics currently require finite LeRobot datasets")
        loaders = [
            create_torch_dataloader(
                dc, config.model.action_horizon, config.batch_size, config.model, config.num_workers
            )[0]
            for dc in data_configs
        ]
        numeric_stats, coverage = compute_weighted_statistics(
            [lambda loader=loader: iter(loader) for loader in loaders], weights
        )
        after = source_descriptors(factories, weights)
        if before != after:
            raise ValueError(
                "Dataset files or processing changed during statistics computation; no output was published"
            )
        norm_stats = {
            key: normalize.NormStats(**values, quantile_method="q01_q99") for key, values in numeric_stats.items()
        }
        document = json.loads(normalize.serialize_json(norm_stats))
        document["shared_normalization"] = {
            "schema_version": 1,
            "method": "weighted_raw_samples",
            "weight_semantics": "per_sample_repeat",
            "coverage": "full",
            "sources": [dict(source, **counts) for source, counts in zip(before, coverage, strict=True)],
            "quantile_method": "weighted_histogram",
            "quantile_bins": 5000,
        }
        validate_shared_metadata(document, factories, weights)
    else:
        import tqdm

        data_config = factory.create(config.assets_dirs, config.model)
        output = _local_path(output_dir) if output_dir is not None else _assets_directory(factory, config).resolve()
        _check_output(output, [factory], config, multi=False)
        if (output / "norm_stats.json").exists() and not overwrite:
            raise FileExistsError("Statistics already exist; choose another output directory or use --overwrite")
        if data_config.rlds_data_dir is not None:
            loader, batches = create_rlds_dataloader(
                data_config, config.model.action_horizon, config.batch_size, max_frames
            )
        else:
            loader, batches = create_torch_dataloader(
                data_config,
                config.model.action_horizon,
                config.batch_size,
                config.model,
                config.num_workers,
                max_frames,
            )
        stats = {key: normalize.RunningStats() for key in ("state", "actions")}
        for batch in tqdm.tqdm(loader, total=batches, desc="Computing stats"):
            for key, accumulator in stats.items():
                accumulator.update(np.asarray(batch[key]))
        document = json.loads(normalize.serialize_json({key: value.get_statistics() for key, value in stats.items()}))
    _publish(output, document, overwrite=overwrite)
    print(f"Wrote normalization statistics to: {output / 'norm_stats.json'}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
