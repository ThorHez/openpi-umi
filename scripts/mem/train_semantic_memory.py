"""Standalone supervised semantic-memory trainer.

This entry point intentionally does not call ``train_mem.py`` and never
constructs an action loss.  Task supervision and recipe choices live in
``shellgame_semantic_memory_pretrain``; this file only owns the reusable JAX
training, validation, sharding, and checkpoint loop.

Example::

    uv run python scripts/mem/train_semantic_memory.py \
        shellgame_semantic_memory_pretrain --exp-name=memory_v1
"""

# Environment variables must be set before importing JAX and datasets.
# ruff: noqa: E402

from __future__ import annotations

import os
from pathlib import Path

_CACHE_HOME = Path("/data2/hzl_workspace_for_pi_mem/.cache")
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_HOME))
os.environ.setdefault("OPENPI_DATA_HOME", str(_CACHE_HOME / "openpi"))
os.environ.setdefault("HF_HOME", str(_CACHE_HOME / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_HOME / "huggingface" / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE_HOME / "huggingface" / "transformers"))
if "TMPDIR" not in os.environ:
    _tmp = _CACHE_HOME / "tmp"
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
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import tqdm_loggable.auto as tqdm
import wandb

import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.config_pi0_mem as _config_pi0_mem
import openpi.training.data_loader as _data_loader
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as _recipe
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from train import _load_weights_and_validate
from train import init_logging
from train import init_wandb


def _count_params(state) -> int:
    total = 0
    for leaf in jax.tree.leaves(state):
        value = leaf.value if hasattr(leaf, "value") else leaf
        if hasattr(value, "shape"):
            total += int(np.prod(value.shape))
    return total


def init_train_state(
    config: _recipe.ShellGameSemanticMemoryPretrainConfig,
    init_rng,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    """Initialize only the optimizer state for memory-pretrain parameters."""
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init_params(rng, partial_params=None):
        _, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda parameter: parameter.replace(parameter.value.astype(jnp.bfloat16)),
        )
        return params, nnx.graphdef(model)

    params_shape, _ = jax.eval_shape(init_params, init_rng)
    trainable_params_shape = params_shape.filter(config.trainable_filter)

    def init(rng, partial_params=None):
        params, model_def = init_params(rng, partial_params)
        trainable_params = params.filter(config.trainable_filter)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=model_def,
            tx=tx,
            opt_state=tx.init(trainable_params),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    total = _count_params(params_shape)
    trainable = _count_params(trainable_params_shape)
    logging.info(
        "Semantic-memory parameters: trainable=%d (%.2fM), frozen=%d (%.2fM), total=%d (%.2fM)",
        trainable,
        trainable / 1e6,
        total - trainable,
        (total - trainable) / 1e6,
        total,
        total / 1e6,
    )
    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


def train_step(config, label_table, rng, state, batch):
    """Update the memory modules from classification and recurrent-memory losses."""
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, _actions = batch
    train_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(memory_model):
        return _recipe.compute_objective(
            config,
            memory_model,
            train_rng,
            observation,
            label_table,
            train=True,
        )

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
    trainable_params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, trainable_params)
    updated_params = optax.apply_updates(trainable_params, updates)
    nnx.update(model, updated_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=training_utils.ema_merge_trees(state.ema_decay, state.ema_params, new_params),
        )
    return new_state, {
        **info,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(updated_params),
        "learning_rate": config.lr_schedule.create()(state.step),
        "objective_loss": loss,
    }


def eval_step(config, label_table, rng, state, batch):
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, _actions = batch
    _, info = _recipe.compute_objective(
        config,
        model,
        rng,
        observation,
        label_table,
        train=False,
    )
    return {f"val/{key}": value for key, value in info.items()}


def _episode_split_indices(dataset, val_ratio: float, seed: int):
    """Split by episode so the same fixed video prefix cannot leak."""
    current = dataset
    hf_dataset = None
    sample_indices = None
    while current is not None:
        if sample_indices is None:
            sample_indices = getattr(current, "sample_indices", None)
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            break
        current = getattr(current, "_dataset", None)
    if hf_dataset is None or "episode_index" not in getattr(hf_dataset, "column_names", ()):
        raise ValueError("Episode-held-out validation requires an episode_index column.")
    episode_indices = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    if sample_indices is not None:
        episode_indices = episode_indices[np.asarray(sample_indices, dtype=np.int64)]
    if episode_indices.shape != (len(dataset),):
        raise ValueError(f"episode_index shape {episode_indices.shape} does not match dataset length {len(dataset)}")
    episodes = np.unique(episode_indices)
    if episodes.size < 2:
        raise ValueError("Validation requires at least two episodes.")
    shuffled = np.random.default_rng(seed).permutation(episodes)
    num_val = min(max(1, round(episodes.size * val_ratio)), episodes.size - 1)
    val_mask = np.isin(episode_indices, shuffled[:num_val])
    return np.flatnonzero(~val_mask).tolist(), np.flatnonzero(val_mask).tolist()


def _make_loader(config, data_config, dataset, data_sharding, *, shuffle: bool):
    local_batch_size = config.batch_size // jax.process_count()
    if len(dataset) < local_batch_size:
        raise ValueError(f"Dataset split has {len(dataset)} rows, smaller than local batch {local_batch_size}.")
    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=data_sharding,
        shuffle=shuffle,
        num_workers=config.num_workers if shuffle else 0,
        seed=config.seed,
        prefetch_factor=2 if shuffle and config.num_workers > 0 else None,
    )
    return _data_loader.DataLoaderImpl(data_config, torch_loader)


def create_train_val_data_loaders(config, data_sharding):
    if not 0.0 < config.val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {config.val_ratio}")
    if not isinstance(config.data, _config.MultiDataConfigFactory):
        raise ValueError("The semantic-memory recipe currently requires MultiDataConfigFactory.")
    if len(config.data.datasets) != 1:
        raise ValueError("The semantic-memory recipe currently requires exactly one dataset.")
    data_config = config.data.create_all(config.assets_dirs, config.model)[0]
    child = config.data.datasets[0]
    dataset = _config_pi0_mem._build_pi0_mem_dataset(  # noqa: SLF001
        data_config,
        child.video_frame_config(),
        action_horizon=config.model.action_horizon,
        skip_norm_stats=False,
    )
    split_seed = getattr(config, "split_seed", config.seed)
    train_indices, val_indices = _episode_split_indices(dataset, config.val_ratio, split_seed)
    logging.info(
        "Memory dataset episode split: train=%d, val=%d, split_seed=%d (one fixed prefix per row)",
        len(train_indices),
        len(val_indices),
        split_seed,
    )
    return (
        _make_loader(
            config,
            data_config,
            torch.utils.data.Subset(dataset, train_indices),
            data_sharding,
            shuffle=True,
        ),
        _make_loader(
            config,
            data_config,
            torch.utils.data.Subset(dataset, val_indices),
            data_sharding,
            shuffle=False,
        ),
    )


def run_evaluation(peval_step, rng, state, val_iter, num_batches: int):
    infos = [peval_step(jax.random.fold_in(rng, index), state, next(val_iter)) for index in range(num_batches)]
    reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
    return {key: float(value) for key, value in reduced.items()}


def main(config: _recipe.ShellGameSemanticMemoryPretrainConfig):
    init_logging()
    logging.info("Running semantic-memory pretraining on %s", platform.node())
    if not isinstance(config, _recipe.ShellGameSemanticMemoryPretrainConfig):
        raise TypeError("train_semantic_memory.py only accepts a semantic-memory pretraining recipe.")
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {config.batch_size} must be divisible by {jax.device_count()} devices.")
    if config.eval_interval <= 0 or config.eval_batches <= 0:
        raise ValueError("eval_interval and eval_batches must both be positive.")
    if (
        min(
            config.initial_loss_weight,
            config.relation_loss_weight,
            config.stage_memory_loss_weight,
        )
        < 0
    ):
        raise ValueError("Memory objective weights must be nonnegative.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    label_table = _recipe.load_episode_label_table(config)
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    train_loader, val_loader = create_train_val_data_loaders(config, data_sharding)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    batch = next(train_iter)
    logging.info(
        "Initialized semantic-memory batch:\n%s",
        training_utils.array_tree_to_info(batch),
    )

    state, state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(state)
    if resuming:
        state = _checkpoints.restore_state(checkpoint_manager, state, train_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config, label_table),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        functools.partial(eval_step, config, label_table),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=replicated,
    )

    start_step = int(state.step)
    best_val_loss = float("inf")
    infos = []
    progress = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
    for step in progress:
        with sharding.set_mesh(mesh):
            state, info = ptrain_step(train_rng, state, batch)
        infos.append(info)
        batch = next(train_iter)

        if step % config.log_interval == 0:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            metrics = {key: round(float(value), 6) for key, value in reduced.items()}
            progress.write(f"Step {step}: " + ", ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
            wandb.log(metrics, step=step)
            infos = []

        if step > start_step and step % config.eval_interval == 0:
            metrics = run_evaluation(
                peval_step,
                jax.random.fold_in(train_rng, step),
                state,
                val_iter,
                config.eval_batches,
            )
            progress.write(f"Step {step} [eval]: " + ", ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
            wandb.log(metrics, step=step)
            best_val_loss = min(best_val_loss, metrics["val/loss"])

        if (step > start_step and step % config.save_interval == 0) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, state, train_loader, step)

    final_metrics = run_evaluation(
        peval_step,
        jax.random.fold_in(train_rng, config.num_train_steps),
        state,
        val_iter,
        config.eval_batches,
    )
    wandb.log(final_metrics, step=config.num_train_steps)
    logging.info(
        "Final semantic-memory validation: %s; best val/loss=%s",
        final_metrics,
        min(best_val_loss, final_metrics["val/loss"]),
    )
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_recipe.cli())
