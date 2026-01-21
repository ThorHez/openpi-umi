"""
Hybrid Training Script for Pi0.5 + FAST

This script implements joint training of Flow Matching (Pi0.5) and FAST (autoregressive token prediction)
as described in the Pi0.5 paper. The total loss is:

    L_total = L_flow_matching + lambda_fast * L_fast_ce

where:
    - L_flow_matching: MSE loss for flow matching (predicting velocity field)
    - L_fast_ce: Cross-entropy loss for FAST token prediction

Usage:
    python scripts/train_hybrid.py --config pi05_umi_hybrid --exp_name my_experiment

Author: OpenPI Team (Extended for hybrid training)
"""

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
from openpi.models import tokenizer as _tokenizer


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

    # === Debug: Log frozen vs trainable params ===
    def count_params_from_shape(params_dict):
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


# ============================================================================
# HYBRID LOSS COMPUTATION
# ============================================================================

@at.typecheck
def compute_flow_matching_loss(
    model: _model.BaseModel,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    *,
    train: bool = False,
) -> at.Float[at.Array, "*b"]:
    """Compute Flow Matching loss (same as Pi0.5 original loss)."""
    # This directly calls the model's compute_loss which implements flow matching
    chunked_loss = model.compute_loss(rng, observation, actions, train=train)
    return jnp.mean(chunked_loss)  # Average over action horizon


@at.typecheck  
def compute_fast_ce_loss(
    model: _model.BaseModel,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    fast_tokenizer: _tokenizer.FASTTokenizer,
    *,
    train: bool = False,
) -> at.Float[at.Array, ""]:
    """
    Compute FAST Cross-Entropy loss for action token prediction.
    
    This tokenizes the actions using FAST tokenizer and computes CE loss
    on the predicted action tokens.
    
    Note: This is a simplified implementation. In practice, you'd want to:
    1. Pre-tokenize actions in the data pipeline for efficiency
    2. Use the model's language head for token prediction
    """
    # For now, return a placeholder - full implementation requires model architecture changes
    # The FAST loss would require adding an autoregressive head to the Pi0.5 model
    # which shares the VLM backbone but has a separate output projection
    
    # Placeholder: return 0 loss (FAST loss requires model changes to fully implement)
    batch_size = actions.shape[0]
    return jnp.zeros(())


@at.typecheck
def train_step_hybrid(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    lambda_fast: float = 0.1,  # Weight for FAST loss
    use_fast_loss: bool = False,  # Enable/disable FAST loss
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """
    Hybrid training step that computes both Flow Matching and (optionally) FAST losses.
    
    The total loss is: L_total = L_flow + lambda_fast * L_fast
    
    Args:
        config: Training configuration
        rng: Random key
        state: Current training state
        batch: Tuple of (observation, actions)
        lambda_fast: Weight for FAST loss in total loss
        use_fast_loss: Whether to compute and use FAST loss (requires Pi0Hybrid model)
        
    Returns:
        Updated training state and info dict with loss values
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, 
        rng: at.KeyArrayLike, 
        observation: _model.Observation, 
        actions: _model.Actions
    ):
        # Check if model supports hybrid loss
        if hasattr(model, 'compute_hybrid_loss') and use_fast_loss:
            # Use hybrid loss computation (Flow Matching + FAST)
            total_loss, loss_info = model.compute_hybrid_loss(rng, observation, actions, train=True)
            return total_loss, loss_info
        else:
            # Standard Flow Matching loss only
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            flow_loss = jnp.mean(chunked_loss)
            return flow_loss, {"flow_loss": flow_loss, "fast_loss": jnp.zeros(()), "total_loss": flow_loss}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, loss_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

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
        "flow_loss": loss_info.get("flow_loss", loss),
        "fast_loss": loss_info.get("fast_loss", jnp.zeros(())),
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main(config: _config.TrainConfig, *, lambda_fast: float = 0.1, use_fast_loss: bool = False):
    """
    Main training function for hybrid Pi0.5 + FAST training.
    
    Args:
        config: Training configuration
        lambda_fast: Weight for FAST loss (default: 0.1)
        use_fast_loss: Whether to use FAST loss (default: False, flow matching only)
    """
    init_logging()
    logging.info(f"Running HYBRID training on: {platform.node()}")
    logging.info(f"  lambda_fast: {lambda_fast}")
    logging.info(f"  use_fast_loss: {use_fast_loss}")

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

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # Create partial train step with hybrid loss
    ptrain_step = jax.jit(
        functools.partial(
            train_step_hybrid, 
            config,
            lambda_fast=lambda_fast,
            use_fast_loss=use_fast_loss,
        ),
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
    
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        
        # Track loss for monitoring
        current_loss = float(info["loss"])
        loss_history.append(current_loss)
        if len(loss_history) > 100:
            loss_history.pop(0)
        
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
    import argparse
    
    # Parse additional hybrid training arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lambda_fast", type=float, default=0.1, 
                        help="Weight for FAST loss in hybrid training")
    parser.add_argument("--use_fast_loss", action="store_true",
                        help="Enable FAST loss (requires model support)")
    
    # Parse known args first, let tyro handle the rest
    args, remaining = parser.parse_known_args()
    
    # Restore sys.argv for tyro
    import sys
    sys.argv = [sys.argv[0]] + remaining
    
    # Get config from tyro CLI
    config = _config.cli()
    
    main(config, lambda_fast=args.lambda_fast, use_fast_loss=args.use_fast_loss)
