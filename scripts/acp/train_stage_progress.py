"""Train a stage-progress estimator and publish an ACP-labeled LeRobot copy."""

import dataclasses
import functools
import importlib.util
import logging
from pathlib import Path
import random

from flax import nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

from openpi.models import pi0_stage_progress
from openpi.training import checkpoints
from openpi.training import config as train_config
from openpi.training import sharding
from openpi.training import stage_progress
from openpi.training import stage_progress_labels
from openpi.training import utils as training_utils


_value_entry_spec = importlib.util.spec_from_file_location(
    "openpi_stage_progress_value_entry", Path(__file__).with_name("train_value.py")
)
_value_entry = importlib.util.module_from_spec(_value_entry_spec)
_value_entry_spec.loader.exec_module(_value_entry)
init_wandb = _value_entry.init_wandb


def train_step(config, state, batch):
    estimator = nnx.merge(state.model_def, state.params)
    estimator.train()
    reference, current, targets = batch

    def loss_fn(model):
        prediction = model.predict_progress(reference, current)
        return jnp.mean(jnp.square(prediction - targets)), jnp.mean(prediction)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, prediction_mean), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(estimator)
    params = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(estimator, optax.apply_updates(params, updates))
    params = nnx.state(estimator)
    ema_params = state.ema_params
    if state.ema_decay is not None:
        ema_params = training_utils.ema_merge_trees(state.ema_decay, ema_params, params)
    return dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt_state, ema_params=ema_params), {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "progress_pred_mean": prediction_mean,
        "progress_target_mean": jnp.mean(targets),
    }


def make_predictor(source, config, state, state_sharding, data_sharding, mesh):
    graph = state.model_def
    params = state.ema_params if state.ema_params is not None else state.params
    params_sharding = state_sharding.ema_params if state.ema_params is not None else state_sharding.params

    def score(model_params, reference, current):
        estimator = nnx.merge(graph, model_params)
        estimator.eval()
        return estimator.predict_progress(reference, current)

    compiled_score = jax.jit(score, in_shardings=(params_sharding, data_sharding, data_sharding),
                             out_shardings=data_sharding)
    batch_size = config.batch_size

    def predict(reference_rows, current_rows):
        count = len(current_rows)
        if count != len(reference_rows) or not 0 < count <= batch_size:
            raise ValueError("阶段进度标注请求必须包含 1 到 batch_size 个成对样本")
        reference_rows = np.asarray(reference_rows)
        current_rows = np.asarray(current_rows)
        # Decode a shared episode anchor once per batch and pad the final short
        # batch with real rows, so sharding never wraps into a different episode.
        observations = {int(row): source.observation(row) for row in np.union1d(reference_rows, current_rows)}
        reference_rows = np.pad(reference_rows, (0, batch_size - count), mode="edge")
        current_rows = np.pad(current_rows, (0, batch_size - count), mode="edge")
        reference = source.observation_batch(reference_rows, observations)
        current = source.observation_batch(current_rows, observations)

        def to_device(value):
            return jax.make_array_from_process_local_data(data_sharding, value)

        with sharding.set_mesh(mesh):
            scores = compiled_score(params, jax.tree.map(to_device, reference), jax.tree.map(to_device, current))
        return np.asarray(jax.device_get(scores))[:count]

    return predict


def main(config, *, output_dir=None):
    if type(config.model) is not pi0_stage_progress.Pi0StageProgressConfig:
        raise ValueError("阶段进度训练需要 Pi0StageProgressConfig")
    if jax.process_count() != 1:
        raise ValueError("阶段进度训练及标注需要单机训练环境")
    if config.batch_size < 1 or config.batch_size % jax.device_count():
        raise ValueError("阶段进度训练 batch_size 必须能被本机设备数整除")
    if config.num_train_steps < 1:
        raise ValueError("阶段进度训练至少执行一步")

    _value_entry.init_logging()
    jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))
    random.seed(config.seed)
    source = stage_progress.StageProgressSource(config)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = stage_progress.StageProgressDataLoader(source, config, data_sharding)
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    state, state_sharding = _value_entry.init_train_state(config, jax.random.key(config.seed), mesh, resume=resuming)
    if resuming:
        state = checkpoints.restore_state(manager, state, loader, sharding=state_sharding)

    update = jax.jit(functools.partial(train_step, config), in_shardings=(state_sharding, data_sharding),
                     out_shardings=(state_sharding, replicated), donate_argnums=(0,))
    batches = iter(loader)
    start = int(state.step)
    progress = tqdm.tqdm(range(start, config.num_train_steps), initial=start, total=config.num_train_steps,
                         dynamic_ncols=True)
    pending = []
    for step in progress:
        batch = next(batches)
        with sharding.set_mesh(mesh):
            state, info = update(state, batch)
        pending.append(info)
        final = step == config.num_train_steps - 1
        if step % config.log_interval == 0 or final:
            means = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(pending)))
            progress.write(f"Step {step}: " + ", ".join(f"{key}={value:.4f}" for key, value in means.items()))
            wandb.log(means, step=step)
            pending = []
        if final or (step > start and step % config.save_interval == 0):
            checkpoints.save_state(manager, state, loader, step)
    manager.wait_until_finished()
    manager.close()
    del batches, loader

    predict = make_predictor(source, config, state, state_sharding, data_sharding, mesh)
    del state
    output_dir = Path(config.checkpoint_base_dir).parent if output_dir is None else output_dir
    output = stage_progress_labels.annotate_dataset(config, source, predict, output_dir)
    logging.info("阶段进度优势标注数据已生成：%s", output)


if __name__ == "__main__":
    config = train_config.cli()
    main(config, output_dir=config.checkpoint_dir)
