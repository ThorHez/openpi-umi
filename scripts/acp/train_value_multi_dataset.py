"""Value-only training (Pi0Value) with multi-dataset LeRobot loading.

Same training loop as ``train_value.py``, but when ``config.data`` is a
``MultiDataConfigFactory`` the data loader is built via
``openpi.training.multi_data_loader.create_multi_data_loader`` (weighted
concatenation + optional merged norm stats), matching ``train_multi_dataset.py``.

For single-dataset configs this script behaves like ``train_value.py``.

Usage:
    python scripts/train_value_multidataset.py pi0_value_umi_bimanual_headview_depth_multi_dataset \\
        --exp_name=my_value_multi

    # Or any TrainConfig whose ``data`` is MultiDataConfigFactory.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import torch
import torch.utils.data as torch_data
import tqdm_loggable.auto as tqdm
import etils.epath as epath
from flax.training import common_utils

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.checkpoints as _checkpoints

_tv_path = Path(__file__).resolve().parent / "train_value.py"
_spec = importlib.util.spec_from_file_location("openpi_train_value", _tv_path)
_tv_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tv_mod)

init_logging = _tv_mod.init_logging
init_wandb = _tv_mod.init_wandb
load_episode_metadata = _tv_mod.load_episode_metadata
create_train_val_data_loaders = _tv_mod.create_train_val_data_loaders
value_batch_iterator = _tv_mod.value_batch_iterator
run_evaluation = _tv_mod.run_evaluation
train_step = _tv_mod.train_step
eval_step = _tv_mod.eval_step
init_train_state = _tv_mod.init_train_state

import wandb


def _create_train_data_loader(config: _config.TrainConfig, data_sharding):
    """Training loader: multi-dataset if ``config.data`` is MultiDataConfigFactory."""
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


def create_train_val_multi_data_loaders(
    config: _config.TrainConfig,
    data_sharding: jax.sharding.Sharding,
):
    """Train/val split over the concatenated multi-dataset (same idea as ``train_value.py``).

    Note: when ``val_ratio > 0``, batches are drawn with uniform shuffle over the train
    subset only; per-dataset ``weights`` (weighted sampling) are not applied in this mode.
    Use ``val_ratio=0`` for weighted multi-dataset training.
    """
    from openpi.training.multi_data_loader import MultiDataLoaderImpl, WeightedConcatDataset

    if not isinstance(config.data, _config.MultiDataConfigFactory):
        raise TypeError("create_train_val_multi_data_loaders requires MultiDataConfigFactory")

    multi_factory = config.data
    all_configs = multi_factory.create_all(config.assets_dirs, config.model)
    weights_list = multi_factory.weights or []
    model_config = config.model
    action_horizon = config.model.action_horizon

    for dc in all_configs:
        if dc.rlds_data_dir is not None:
            raise ValueError("Dataset splitting is not supported for RLDS datasets. Set val_ratio=0.")

    datasets: list[torch.utils.data.Dataset] = []
    for dc in all_configs:
        ds = _data_loader.create_torch_dataset(dc, action_horizon, model_config)
        ds = _data_loader.transform_dataset(ds, dc)
        datasets.append(ds)

    wts = [weights_list[i] if i < len(weights_list) else 1.0 for i in range(len(datasets))]
    concat = WeightedConcatDataset(datasets, weights=wts if len(set(wts)) > 1 else None)

    n = len(concat)
    n_val = max(1, int(n * config.val_ratio))
    n_train = n - n_val

    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = torch_data.random_split(concat, [n_train, n_val], generator=generator)

    local_batch_size = config.batch_size // jax.process_count()

    train_torch = _data_loader.TorchDataLoader(
        train_subset,
        local_batch_size=local_batch_size,
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_batch_size = min(local_batch_size, len(val_subset))
    val_torch = _data_loader.TorchDataLoader(
        val_subset,
        local_batch_size=val_batch_size,
        sharding=data_sharding,
        shuffle=False,
        num_workers=0,
        seed=config.seed,
    )

    train_dl = MultiDataLoaderImpl(all_configs, train_torch)
    val_dl = MultiDataLoaderImpl(all_configs, val_torch)

    logging.info(
        "Multi-dataset split: %d train, %d val (val_ratio=%s). "
        "Per-dataset sampling weights are disabled in this val mode.",
        n_train,
        n_val,
        config.val_ratio,
    )
    return train_dl, val_dl


def main(config: _config.TrainConfig):
    init_logging()
    logging.info("Running on: %s (train_value_multidataset)", platform.node())

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

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

    episode_info, task_max_lengths = load_episode_metadata(config)
    if not episode_info or not task_max_lengths:
        logging.warning(
            "episode_info or task_max_lengths is empty. Set episode_metadata_path (JSON) or task_max_lengths "
            "for value target computation. Using empty dicts may cause KeyError when batch has episode_index."
        )

    use_val = config.val_ratio > 0
    val_data_loader = None

    if use_val:
        if isinstance(config.data, _config.MultiDataConfigFactory):
            data_loader, val_data_loader = create_train_val_multi_data_loaders(config, data_sharding)
        else:
            data_loader, val_data_loader = create_train_val_data_loaders(config, data_sharding)
    else:
        data_loader = _create_train_data_loader(config, data_sharding)

    data_iter = iter(data_loader)
    value_iter = value_batch_iterator(data_iter, episode_info, task_max_lengths, config)
    batch_obs, value_targets_batch = next(value_iter)
    logging.info(
        "Initialized data loader (first value batch): value_targets shape %s", value_targets_batch.shape
    )

    if use_val:
        val_value_iter = value_batch_iterator(iter(val_data_loader), episode_info, task_max_lengths, config)
        logging.info(
            "Initialized val data loader (eval_interval=%s, eval_batches=%s)",
            config.eval_interval,
            config.eval_batches,
        )

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(
        "Initialized train state (value head only):\n%s", training_utils.array_tree_to_info(train_state.params)
    )

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    if use_val:
        peval_step = jax.jit(
            functools.partial(eval_step, config),
            in_shardings=(train_state_sharding, data_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        best_val_loss = float("inf")

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []

    for step in pbar:
        step_rng, train_rng = jax.random.split(train_rng)
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(step_rng, train_state, batch_obs, value_targets_batch)
        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if use_val and step % config.eval_interval == 0 and step > start_step:
            val_metrics, val_value_iter = run_evaluation(
                peval_step,
                val_value_iter,
                val_data_loader,
                episode_info,
                task_max_lengths,
                config,
                mesh,
                train_state,
            )
            if val_metrics:
                val_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                pbar.write(f"Step {step} [eval]: {val_str}")
                wandb.log(val_metrics, step=step)
                cur_val_loss = val_metrics.get("val/loss", float("inf"))
                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    pbar.write(f"Step {step}: new best val/loss = {best_val_loss:.4f}")

        try:
            batch_obs, value_targets_batch = next(value_iter)
        except StopIteration:
            value_iter = value_batch_iterator(iter(data_loader), episode_info, task_max_lengths, config)
            batch_obs, value_targets_batch = next(value_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    if use_val:
        logging.info("Running final evaluation on validation set...")
        val_metrics, _ = run_evaluation(
            peval_step,
            val_value_iter,
            val_data_loader,
            episode_info,
            task_max_lengths,
            config,
            mesh,
            train_state,
        )
        if val_metrics:
            val_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            logging.info("Final eval: %s", val_str)
            wandb.log(val_metrics, step=config.num_train_steps)
            logging.info("Best val/loss during training: %s", best_val_loss)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    config = _config.cli()
    print("\n" + "=" * 60)
    print("VALUE TRAINING (MULTI-DATASET)")
    print("=" * 60)
    if isinstance(config.data, _config.MultiDataConfigFactory):
        print(f"  Multi-dataset loader: {len(config.data.datasets)} dataset(s)")
        if config.data.state_pad_dim:
            print(f"  State pad dim: {config.data.state_pad_dim}")
        print(f"  Merged norm stats: {config.data.use_merged_norm_stats}")
    else:
        print("  Single-dataset loader (same as train_value.py)")
    print("=" * 60 + "\n")
    main(config)
