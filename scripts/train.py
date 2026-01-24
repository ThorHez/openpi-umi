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
import jax.experimental
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


def init_logging():
    """Custom logging format for better readability."""
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


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
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

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
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

    # === Debug: Log frozen vs trainable params using abstract shapes ===
    def count_params_from_shape(params_dict):
        """Count params from shape info (works with jax.ShapeDtypeStruct)."""
        total = 0
        for leaf in jax.tree.leaves(params_dict):
            if hasattr(leaf, 'shape'):
                total += int(np.prod(leaf.shape))
            elif hasattr(leaf, 'value') and hasattr(leaf.value, 'shape'):
                total += int(np.prod(leaf.value.shape))
        return total
    
    all_params_shape = train_state_shape.params
    frozen_params_shape = all_params_shape.filter(config.freeze_filter)
    trainable_params_shape = all_params_shape.filter(config.trainable_filter)
    
    total_count = count_params_from_shape(all_params_shape)
    frozen_count = count_params_from_shape(frozen_params_shape)
    trainable_count = count_params_from_shape(trainable_params_shape)
    
    logging.info("=" * 60)
    logging.info("FREEZE FILTER ANALYSIS:")
    logging.info(f"  Total params:     {total_count:,} ({total_count/1e6:.2f}M)")
    logging.info(f"  Frozen params:    {frozen_count:,} ({frozen_count/1e6:.2f}M) ({frozen_count/total_count*100:.1f}%)")
    logging.info(f"  Trainable params: {trainable_count:,} ({trainable_count/1e6:.2f}M) ({trainable_count/total_count*100:.1f}%)")
    logging.info("=" * 60)
    
    # Helper to get param size
    def get_param_size(leaf):
        if hasattr(leaf, 'shape'):
            return int(np.prod(leaf.shape))
        elif hasattr(leaf, 'value') and hasattr(leaf.value, 'shape'):
            return int(np.prod(leaf.value.shape))
        return 0
    
    # Log ALL frozen param names grouped by top-level module
    frozen_flat = traverse_util.flatten_dict(frozen_params_shape.to_pure_dict())
    if frozen_flat:
        logging.info(f"ALL FROZEN param paths ({len(frozen_flat)} total):")
        # Group by top-level module (first 2 path components)
        from collections import defaultdict
        frozen_by_module = defaultdict(list)
        for key, value in frozen_flat.items():
            module = '/'.join(key[:2]) if len(key) >= 2 else key[0]
            size = get_param_size(value)
            frozen_by_module[module].append((key, size))
        
        for module in sorted(frozen_by_module.keys()):
            params = frozen_by_module[module]
            module_size = sum(p[1] for p in params)
            logging.info(f"  [{module}] ({len(params)} params, {module_size/1e6:.2f}M)")
            for key, size in params:
                logging.info(f"    - {'/'.join(key)} [{size:,}]")
    
    logging.info("-" * 60)
    
    # Log ALL trainable param names grouped by top-level module
    trainable_flat = traverse_util.flatten_dict(trainable_params_shape.to_pure_dict())
    if trainable_flat:
        logging.info(f"ALL TRAINABLE param paths ({len(trainable_flat)} total):")
        from collections import defaultdict
        trainable_by_module = defaultdict(list)
        for key, value in trainable_flat.items():
            module = '/'.join(key[:2]) if len(key) >= 2 else key[0]
            size = get_param_size(value)
            trainable_by_module[module].append((key, size))
        
        for module in sorted(trainable_by_module.keys()):
            params = trainable_by_module[module]
            module_size = sum(p[1] for p in params)
            logging.info(f"  [{module}] ({len(params)} params, {module_size/1e6:.2f}M)")
            for key, size in params:
                logging.info(f"    - {'/'.join(key)} [{size:,}]")
    
    logging.info("=" * 60)
    # === End Debug ===

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


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

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    # images_to_log = [
    #     wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
    #     for i in range(min(5, len(next(iter(batch[0].images.values())))))
    # ]
    # wandb.log({"camera_views": images_to_log}, step=0)

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
    loss_history = []  # Track recent loss values for anomaly detection
    anomaly_dir = config.checkpoint_dir / "anomalies"
    anomaly_dir.mkdir(exist_ok=True)
    
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        
        # Anomaly detection
        current_loss = float(info["loss"])
        loss_history.append(current_loss)
        if len(loss_history) > 100:  # Keep last 100 steps
            loss_history.pop(0)
        
        # Detect anomalies: NaN, Inf, or sudden spike
        is_anomaly = False
        anomaly_reason = ""
        loss_history_size = config.log_interval * 2

        if jnp.isnan(current_loss) or jnp.isinf(current_loss):
            is_anomaly = True
            anomaly_reason = "NaN or Inf loss"
        elif len(loss_history) > loss_history_size:
            recent_mean = np.mean(loss_history[-loss_history_size:])
            recent_std = np.std(loss_history[-loss_history_size:])
            # If loss is more than 5 std deviations above recent mean
            if current_loss > recent_mean + 3 * recent_std and recent_std > 1e-6:
                is_anomaly = True
                anomaly_reason = f"Spike: {current_loss:.4f} vs recent {recent_mean:.4f}±{recent_std:.4f}"
        
        if is_anomaly:
            import pickle
            anomaly_file = anomaly_dir / f"step_{step:06d}_{current_loss:.4f}.pkl"
            pbar.write(f"⚠️  ANOMALY DETECTED at step {step}: {anomaly_reason}")
            pbar.write(f"   Saving data to {anomaly_file}")
            
            # Get data to CPU and save
            batch_cpu = jax.device_get(batch)
            anomaly_data = {
                "step": step,
                "loss": current_loss,
                "loss_history": loss_history[-20:],  # Save last 20 losses
                "reason": anomaly_reason,
                "observation": batch_cpu[0],
                "actions": batch_cpu[1],
                "info": jax.device_get(info),
            }
            
            with open(anomaly_file, "wb") as f:
                pickle.dump(anomaly_data, f)
            pbar.write(f"   ✓ Anomaly data saved!")
        
        # Periodic statistics logging (every 1000 steps)
        stats_interval = 1000
        if step > 0 and step % stats_interval == 0 and len(loss_history) > 10:
            import pickle
            stats_dir = config.checkpoint_dir / "periodic_stats"
            stats_dir.mkdir(exist_ok=True)
            
            batch_cpu = jax.device_get(batch)
            stats_file = stats_dir / f"step_{step:06d}_{current_loss:.4f}.pkl"
            
            pbar.write(f"\n📊 Saving periodic statistics at step {step} to {stats_file}")
            
            # Save data in the same format as anomaly detection
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
    main(_config.cli())
