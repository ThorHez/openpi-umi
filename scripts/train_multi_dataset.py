"""
Multi-dataset training script for Pi0.

Same as train.py but creates a multi-dataset data loader when config.data
is MultiDataConfigFactory (multiple LeRobot datasets with optional per-dataset weights
and unified state_pad_dim). Checkpoint saving stores norm_stats for each dataset
under its asset_id.

Usage:
    python scripts/train_multi_dataset.py --config pi05_umi_multi_dataset --exp_name multi_task_v1

For single-dataset configs this script behaves identically to train.py.
"""

from __future__ import annotations

import os

from openpi.shared.runtime_paths import configure_training_cache

configure_training_cache()

import dataclasses
import functools
import logging
import platform
import sys
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

# Reuse init_logging, init_wandb, init_train_state, train_step from train.py
import importlib.util
from pathlib import Path

_train_path = Path(__file__).resolve().parent / "train.py"
_spec = importlib.util.spec_from_file_location("train", _train_path)
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)

init_logging = _train_mod.init_logging
init_wandb = _train_mod.init_wandb
init_train_state = _train_mod.init_train_state
train_step = _train_mod.train_step


def _create_data_loader(config: _config.TrainConfig, data_sharding):
    """Create data loader: multi-dataset if config.data is MultiDataConfigFactory, else single."""
    if isinstance(config.data, _config.MultiDataConfigFactory):
        from openpi.training.multi_data_loader import create_multi_data_loader

        multi_factory = config.data
        all_configs = multi_factory.create_all(config.assets_dirs, config.model)
        weights_list = multi_factory.weights
        dc_and_weights = [
            (dc, weights_list[i] if (weights_list and i < len(weights_list)) else 1.0)
            for i, dc in enumerate(all_configs)
        ]
        return create_multi_data_loader(
            config,
            data_configs_and_weights=dc_and_weights,
            sharding=data_sharding,
            shuffle=True,
        )
    return _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )


def main(config: _config.TrainConfig, *, data_loader=None):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info("Dataset cache: %s; temporary files: %s", os.environ["HF_DATASETS_CACHE"], os.environ["TMPDIR"])

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    if data_loader is None:
        data_loader = _create_data_loader(config, data_sharding)

    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        # Orbax may restore the sharding recorded in the checkpoint. Explicitly
        # move the restored state onto the current mesh so runs can resume with
        # a different FSDP device count (for example, 2 GPUs -> 4 GPUs).
        train_state = jax.device_put(train_state, train_state_sharding, donate=True)
        jax.block_until_ready(train_state)
        logging.info(f"Resharded restored train state onto {config.fsdp_devices} FSDP devices")

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []

    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    config = _config.cli()

    print("\n" + "=" * 60)
    print("MULTI-DATASET TRAINING")
    print("=" * 60)
    if isinstance(config.data, _config.MultiDataConfigFactory):
        print(f"  Using multi-dataset loader: {len(config.data.datasets)} dataset(s)")
        if config.data.state_pad_dim:
            print(f"  State pad dim: {config.data.state_pad_dim}")
    else:
        print("  Using single-dataset loader")
    print("=" * 60 + "\n")

    main(config)
