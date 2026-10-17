"""Value-only training: fit Pi0 value head to normalized value targets.

Uses Pi0Value model (based on pi0.py only). Targets are taken from the batch
``value_target`` field when present; otherwise they are computed via
compute_normalized_value_targets(episode_indices, frame_indices, episode_info,
task_max_lengths, c_fail_coef) when the batch provides episode_index and
frame_index (Option B). Mirror of train_advantage.py with value loss only.
"""

import json
import os
from pathlib import Path

if "TMPDIR" not in os.environ:
    _tmp = Path(os.environ.get("HOME", "/root")) / "tmp"
    _tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_tmp)

import dataclasses
import functools
import logging
import platform
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
import openpi.training.value_targets as value_targets
import openpi.training.weight_loaders as _weight_loaders
import torch
import torch.utils.data as torch_data


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _discover_lerobot_dataset_roots(data: _config.DataConfigFactory) -> list[Path]:
    """Local dataset directories that contain ``meta/episodes.jsonl`` (for value-target auto metadata)."""
    if isinstance(data, _config.MultiDataConfigFactory):
        merged: list[Path] = []
        for sub in data.datasets:
            merged.extend(_discover_lerobot_dataset_roots(sub))
        seen_multi: set[str] = set()
        deduped: list[Path] = []
        for p in merged:
            key = str(p.resolve())
            if key not in seen_multi:
                seen_multi.add(key)
                deduped.append(p)
        return deduped

    candidates: list[Path] = []
    assets = getattr(data, "assets", None)
    if assets is not None and assets.assets_dir:
        candidates.append(Path(assets.assets_dir).expanduser())
    repo_id = getattr(data, "repo_id", None)
    if isinstance(repo_id, str) and repo_id not in ("fake", "multi"):
        candidates.append(Path(repo_id).expanduser())

    out: list[Path] = []
    seen: set[str] = set()
    for raw in candidates:
        try:
            p = raw.resolve()
        except OSError:
            continue
        if not p.is_dir():
            continue
        if not (p / "meta" / "episodes.jsonl").is_file():
            continue
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def load_episode_metadata(
    config: _config.TrainConfig,
) -> tuple[dict[int, value_targets.EpisodeTargetInfo], dict[int, int]]:
    """Load episode_info and task_max_lengths from JSON, CLI overrides, and LeRobot ``meta/`` when present."""
    episode_info: dict[int, value_targets.EpisodeTargetInfo] = {}
    task_max_lengths: dict[int, int] = {}

    if config.episode_metadata_path:
        path = Path(config.episode_metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"episode_metadata_path not found: {path}")
        with open(path) as f:
            data = json.load(f)
        for ep in data.get("episodes", []):
            idx = int(ep["episode_index"])
            episode_info[idx] = value_targets.EpisodeTargetInfo(
                task_index=int(ep["task_index"]),
                length=int(ep["length"]),
                success=bool(ep["success"]),
            )
        raw_tml = data.get("task_max_lengths", {})
        for k, v in raw_tml.items():
            task_max_lengths[int(k)] = int(v)

    if config.task_max_lengths:
        task_max_lengths.update(config.task_max_lengths)

    for root in _discover_lerobot_dataset_roots(config.data):
        try:
            extra_info, extra_tml = value_targets.load_episode_target_info_from_lerobot_meta(root)
        except FileNotFoundError:
            continue
        for k, v in extra_info.items():
            if k not in episode_info:
                episode_info[k] = v
        for t, m in extra_tml.items():
            task_max_lengths[t] = max(task_max_lengths.get(t, 0), m)

    return episode_info, task_max_lengths


def create_train_val_data_loaders(
    config: _config.TrainConfig,
    data_sharding: jax.sharding.Sharding,
):
    """Split dataset into train/val subsets and return separate data loaders."""
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        raise ValueError("Dataset splitting is not supported for RLDS datasets. Set val_ratio=0.")

    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)

    n = len(dataset)
    n_val = max(1, int(n * config.val_ratio))
    n_train = n - n_val

    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = torch_data.random_split(dataset, [n_train, n_val], generator=generator)

    local_batch_size = config.batch_size // jax.process_count()

    train_loader = _data_loader.TorchDataLoader(
        train_subset,
        local_batch_size=local_batch_size,
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    val_batch_size = min(local_batch_size, len(val_subset))
    val_loader = _data_loader.TorchDataLoader(
        val_subset,
        local_batch_size=val_batch_size,
        sharding=data_sharding,
        shuffle=False,
        num_workers=0,
        seed=config.seed,
    )

    train_dl = _data_loader.DataLoaderImpl(data_config, train_loader)
    val_dl = _data_loader.DataLoaderImpl(data_config, val_loader)

    logging.info(f"Dataset split: {n_train} train, {n_val} val (ratio={config.val_ratio})")
    return train_dl, val_dl


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch_obs: _model.Observation,
    value_targets_batch: at.Float[at.Array, " b"],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, observation, targets):
        loss_per_sample = model.compute_value_loss_from_targets(
            observation, targets, train=True, rng=rng
        )
        return jnp.mean(loss_per_sample)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, batch_obs, value_targets_batch)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=training_utils.ema_merge_trees(state.ema_decay, state.ema_params, new_params),
        )

    with jax.named_scope("value_stats"):
        logits = model.forward_value_logits(batch_obs)
        value_pred = model.expected_value_from_logits(logits)

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "value_pred_mean": jnp.mean(value_pred),
        "value_target_mean": jnp.mean(value_targets_batch),
    }
    return new_state, info


@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch_obs: _model.Observation,
    value_targets_batch: at.Float[at.Array, " b"],
) -> dict[str, at.Array]:
    """Single evaluation step: forward pass without gradients."""
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()

    loss_per_sample = model.compute_value_loss_from_targets(
        batch_obs, value_targets_batch, train=False
    )
    loss = jnp.mean(loss_per_sample)

    logits = model.forward_value_logits(batch_obs)
    value_pred = model.expected_value_from_logits(logits)

    return {
        "val/loss": loss,
        "val/value_pred_mean": jnp.mean(value_pred),
        "val/value_target_mean": jnp.mean(value_targets_batch),
    }


def run_evaluation(
    peval_step,
    val_value_iter,
    val_data_loader,
    episode_info: dict,
    task_max_lengths: dict,
    config: _config.TrainConfig,
    mesh: jax.sharding.Mesh,
    state: training_utils.TrainState,
) -> tuple[dict[str, float], any]:
    """Run evaluation over multiple val batches and return averaged metrics."""
    eval_infos = []
    for _ in range(config.eval_batches):
        try:
            val_obs, val_targets = next(val_value_iter)
        except StopIteration:
            val_value_iter = value_batch_iterator(
                iter(val_data_loader), episode_info, task_max_lengths, config
            )
            try:
                val_obs, val_targets = next(val_value_iter)
            except StopIteration:
                break
        with sharding.set_mesh(mesh):
            eval_info = peval_step(state, val_obs, val_targets)
        eval_infos.append(eval_info)

    if not eval_infos:
        return {}, val_value_iter

    stacked = common_utils.stack_forest(eval_infos)
    reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
    return reduced, val_value_iter


def value_batch_iterator(
    data_iter,
    episode_info: dict[int, value_targets.EpisodeTargetInfo],
    task_max_lengths: dict[int, int],
    config: _config.TrainConfig,
):
    """Wrap data loader: (obs, actions) -> (obs, value_targets) using episode_index/frame_index and episode_info."""
    c_fail = config.c_fail_coef
    clip_min = config.value_clip_min
    clip_max = config.value_clip_max

    for batch in data_iter:
        obs, actions = batch
        del actions

        batch_size = obs.state.shape[0]

        vt_obs = getattr(obs, "value_target", None)
        if vt_obs is not None:
            value_targets_jax = jnp.asarray(vt_obs, dtype=jnp.float32).reshape(-1)
            if value_targets_jax.shape[0] != batch_size:
                raise ValueError(
                    f"value_target length {value_targets_jax.shape[0]} does not match batch size {batch_size}."
                )
        else:
            ep_idx = getattr(obs, "episode_index", None)
            fr_idx = getattr(obs, "frame_index", None)

            if ep_idx is None or fr_idx is None:
                if not episode_info and not task_max_lengths:
                    ep_idx_np = np.zeros(batch_size, dtype=np.int64)
                    fr_idx_np = np.arange(batch_size, dtype=np.int64)
                    dummy_ep_info = {0: value_targets.EpisodeTargetInfo(task_index=0, length=batch_size, success=True)}
                    dummy_tml = {0: max(batch_size, 100)}
                    value_targets_np = value_targets.compute_normalized_value_targets(
                        ep_idx_np, fr_idx_np, dummy_ep_info, dummy_tml, c_fail, clip_min=clip_min, clip_max=clip_max
                    )
                else:
                    raise ValueError(
                        "Value training requires observation to have episode_index and frame_index. "
                        "Add them to the data repack/transform when using episode_metadata_path or task_max_lengths."
                    )
            else:
                ep_idx_np = np.asarray(ep_idx).reshape(-1)
                fr_idx_np = np.asarray(fr_idx).reshape(-1)
                value_targets_np = value_targets.compute_normalized_value_targets(
                    ep_idx_np,
                    fr_idx_np,
                    episode_info,
                    task_max_lengths,
                    c_fail,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

            value_targets_jax = jnp.asarray(value_targets_np, dtype=jnp.float32)
        yield obs, value_targets_jax


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

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
        data_loader, val_data_loader = create_train_val_data_loaders(config, data_sharding)
    else:
        data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
        )

    data_iter = iter(data_loader)
    value_iter = value_batch_iterator(data_iter, episode_info, task_max_lengths, config)
    batch_obs, value_targets_batch = next(value_iter)
    logging.info(f"Initialized data loader (first value batch): obs keys, value_targets shape {value_targets_batch.shape}")

    if use_val:
        val_value_iter = value_batch_iterator(
            iter(val_data_loader), episode_info, task_max_lengths, config
        )
        logging.info(f"Initialized val data loader (eval_interval={config.eval_interval}, eval_batches={config.eval_batches})")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state (value head only):\n{training_utils.array_tree_to_info(train_state.params)}")

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
                peval_step, val_value_iter, val_data_loader,
                episode_info, task_max_lengths, config, mesh, train_state,
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
            value_iter = value_batch_iterator(
                iter(data_loader), episode_info, task_max_lengths, config
            )
            batch_obs, value_targets_batch = next(value_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    if use_val:
        logging.info("Running final evaluation on validation set...")
        val_metrics, _ = run_evaluation(
            peval_step, val_value_iter, val_data_loader,
            episode_info, task_max_lengths, config, mesh, train_state,
        )
        if val_metrics:
            val_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            logging.info(f"Final eval: {val_str}")
            wandb.log(val_metrics, step=config.num_train_steps)
            logging.info(f"Best val/loss during training: {best_val_loss:.4f}")

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
