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
from pathlib import Path

# Keep Hugging Face / JAX / OpenPI caches and temp files off the small root disk.
# Must be set before importing OpenPI / LeRobot / datasets / JAX.
_CACHE_HOME = Path("/data2/hzl_workspace_for_pi/.cache")
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_HOME))
os.environ.setdefault("OPENPI_DATA_HOME", str(_CACHE_HOME / "openpi"))
os.environ.setdefault("HF_HOME", str(_CACHE_HOME / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_HOME / "huggingface" / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE_HOME / "huggingface" / "transformers"))
if "TMPDIR" not in os.environ:
    _tmp = _CACHE_HOME / "tmp"
    _tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_tmp)

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
import numpy as np
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

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update(
        "jax_compilation_cache_dir",
        str(_CACHE_HOME / "jax"),
    )

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
    loss_history = []
    anomaly_dir = config.checkpoint_dir / "anomalies"
    anomaly_dir.mkdir(exist_ok=True)

    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        current_loss = float(info["loss"])
        loss_history.append(current_loss)
        if len(loss_history) > 100:
            loss_history.pop(0)

        is_anomaly = False
        anomaly_reason = ""
        loss_history_size = config.log_interval * 2

        if jnp.isnan(current_loss) or jnp.isinf(current_loss):
            is_anomaly = True
            anomaly_reason = "NaN or Inf loss"
        elif len(loss_history) > loss_history_size:
            recent_mean = np.mean(loss_history[-loss_history_size:])
            recent_std = np.std(loss_history[-loss_history_size:])
            if current_loss > recent_mean + 3 * recent_std and recent_std > 1e-6:
                is_anomaly = True
                anomaly_reason = f"Spike: {current_loss:.4f} vs recent {recent_mean:.4f}±{recent_std:.4f}"

        if is_anomaly:
            import pickle
            # anomaly_file = anomaly_dir / f"step_{step:06d}_{current_loss:.4f}.pkl"
            pbar.write(f"⚠️  ANOMALY DETECTED at step {step}: {anomaly_reason}")
            # pbar.write(f"   Saving data to {anomaly_file}")

            # batch_cpu = jax.device_get(batch)
            # anomaly_data = {
            #     "step": step,
            #     "loss": current_loss,
            #     "loss_history": loss_history[-20:],
            #     "reason": anomaly_reason,
            #     "observation": batch_cpu[0],
            #     "actions": batch_cpu[1],
            #     "info": jax.device_get(info),
            # }

            # with open(anomaly_file, "wb") as f:
            #     pickle.dump(anomaly_data, f)
            # pbar.write(f"   ✓ Anomaly data saved!")

        stats_interval = 10000
        if step > 0 and step % stats_interval == 0 and len(loss_history) > 10:
            import pickle
            stats_dir = config.checkpoint_dir / "periodic_stats"
            stats_dir.mkdir(exist_ok=True)

            batch_cpu = jax.device_get(batch)
            stats_file = stats_dir / f"step_{step:06d}_{current_loss:.4f}.pkl"

            pbar.write(f"\n📊 Saving periodic statistics at step {step} to {stats_file}")

            stats_data = {
                "step": step,
                "loss": current_loss,
                "loss_history": loss_history.copy(),
                "reason": f"Periodic stats at step {step}",
                "observation": batch_cpu[0],
                "actions": batch_cpu[1],
                "info": jax.device_get(info),
            }

            with open(stats_file, "wb") as f:
                pickle.dump(stats_data, f)
            pbar.write(f"   ✓ Statistics data saved!")

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
