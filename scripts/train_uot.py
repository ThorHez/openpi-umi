"""论文 Sim-and-Real Co-Training 的 Ours 联合训练入口（适配跨场景 UMI 数据）。

每一步只做一次参数更新：行为克隆损失使用按 90% source / 10% target 混合的 batch，
UOT 损失使用独立的 DTW 配对 batch；总损失为 BC + lambda * UOT。
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import logging
import os
import platform
from pathlib import Path

from openpi.shared.runtime_paths import configure_training_cache

configure_training_cache()

import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from openpi.training.uot import compute_uot_alignment_loss
from openpi.training.uot_data_loader import create_uot_loaders


# === [论文 Ours/UOT] 复用仓库既有初始化与日志逻辑，避免产生第二套 checkpoint 语义。 ===
_train_path = Path(__file__).resolve().parent / "train.py"
_spec = importlib.util.spec_from_file_location("openpi_base_train", _train_path)
_train_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train_module)
init_logging = _train_module.init_logging
init_wandb = _train_module.init_wandb
init_train_state = _train_module.init_train_state


def train_step(
    config: _config.TrainConfig,
    rng: jax.Array,
    state: training_utils.TrainState,
    bc_batch: tuple[_model.Observation, _model.Actions],
    ot_batch: tuple[
        _model.Observation,
        _model.Observation,
        dict[str, jax.Array],
        dict[str, jax.Array],
    ],
) -> tuple[training_utils.TrainState, dict[str, jax.Array]]:
    """在同一梯度步骤中联合优化 π0.5 行为克隆和论文 UOT 对齐目标。"""
    if config.uot is None:
        raise ValueError("Missing UOT config")
    model = nnx.merge(state.model_def, state.params)
    model.train()
    train_rng = jax.random.fold_in(rng, state.step)
    bc_rng, source_rng, target_rng = jax.random.split(train_rng, 3)
    bc_observation, actions = bc_batch
    source_observation, target_observation, source_labels, target_labels = ot_batch

    def loss_fn(model):
        bc_loss = jnp.mean(model.compute_loss(bc_rng, bc_observation, actions, train=True))
        if not hasattr(model, "encode_visual_features"):
            raise TypeError("Selected model does not expose encode_visual_features required by UOT")
        source_features = model.encode_visual_features(source_rng, source_observation, train=True)
        target_features = model.encode_visual_features(target_rng, target_observation, train=True)
        uot_loss, uot_metrics = compute_uot_alignment_loss(
            source_features,
            target_features,
            source_labels,
            target_labels,
            embedding_cost_scale=config.uot.embedding_cost_scale,
            label_cost_scale=config.uot.label_cost_scale,
            entropy_reg=config.uot.entropy_reg,
            marginal_relaxation=config.uot.marginal_relaxation,
            sinkhorn_iterations=config.uot.sinkhorn_iterations,
            # === [论文 Ours/UOT] 官方 heuristic=true 使用对角 KL 参考测度。 ===
            diagonal_reference=config.uot.diagonal_reference,
        )
        total_loss = bc_loss + config.uot.alignment_weight * uot_loss
        return total_loss, {"bc_loss": bc_loss, **uot_metrics}

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_trainable_params = optax.apply_updates(params, updates)
    nnx.update(model, new_trainable_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=training_utils.ema_merge_trees(state.ema_decay, state.ema_params, new_params),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, value: value.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "bc_loss": aux["bc_loss"],
        "uot_loss": aux["uot_loss"],
        "weighted_uot_loss": config.uot.alignment_weight * aux["uot_loss"],
        "uot_plan_mass": aux["uot_plan_mass"],
        "uot_embedding_cost": aux["uot_embedding_cost"],
        "uot_label_cost": aux["uot_label_cost"],
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig) -> None:
    init_logging()
    logging.info("Running UOT co-training on: %s", platform.node())
    if config.uot is None or not config.uot.enabled:
        raise ValueError("Use a TrainConfig whose uot.enabled is true")
    if config.batch_size % jax.device_count() != 0:
        raise ValueError("batch_size must be divisible by the number of JAX devices")
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
    bc_loader, ot_loader = create_uot_loaders(config, sharding=data_sharding)
    bc_iter, ot_iter = iter(bc_loader), iter(ot_loader)
    bc_batch, ot_batch = next(bc_iter), next(ot_iter)
    logging.info("BC batch:\n%s", training_utils.array_tree_to_info(bc_batch))
    logging.info("OT batch:\n%s", training_utils.array_tree_to_info(ot_batch))

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, bc_loader)
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    progress = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
    infos = []
    for step in progress:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, bc_batch, ot_batch)
        infos.append(info)
        if step % config.log_interval == 0:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            metric_text = ", ".join(f"{key}={value:.4f}" for key, value in reduced.items())
            progress.write(f"Step {step}: {metric_text}")
            wandb.log(reduced, step=step)
            infos = []
        bc_batch, ot_batch = next(bc_iter), next(ot_iter)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, bc_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
