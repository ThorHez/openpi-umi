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


_filter_vlm_grads_logged = False  # Global flag to only log once

def _filter_vlm_grads(grads, debug=False):
    """Filter gradients to only keep VLM parameters (exclude action_expert, discrete_head, etc.)
    
    VLM params: .*llm.* but NOT .*llm.*_1.* (action expert) and NOT discrete_head/action_proj
    """
    global _filter_vlm_grads_logged
    import re
    vlm_pattern = re.compile(r".*llm.*")
    action_expert_pattern = re.compile(r".*llm.*_1.*")
    exclude_pattern = re.compile(r".*(discrete_head|action_in_proj|action_out_proj|action_time_mlp|state_proj|time_mlp).*")
    
    flat_grads = traverse_util.flatten_dict(grads, sep="/")
    vlm_grads = {}
    
    # For debug info
    kept_keys = []
    filtered_keys = []
    filter_reasons = {"not_vlm": [], "action_expert": [], "excluded_head": []}
    
    for k, v in flat_grads.items():
        # Include if it's a VLM param (llm.*) but not action expert (llm.*_1.*)
        # and not one of the excluded heads
        is_vlm = vlm_pattern.match(k) is not None
        is_action_expert = action_expert_pattern.match(k) is not None
        is_excluded = exclude_pattern.match(k) is not None
        
        if is_vlm and not is_action_expert and not is_excluded:
            vlm_grads[k] = v
            kept_keys.append(k)
        else:
            filtered_keys.append(k)
            if not is_vlm:
                filter_reasons["not_vlm"].append(k)
            elif is_action_expert:
                filter_reasons["action_expert"].append(k)
            elif is_excluded:
                filter_reasons["excluded_head"].append(k)
    
    # Print debug info only once
    if not _filter_vlm_grads_logged:
        _filter_vlm_grads_logged = True
        logging.info("=" * 60)
        logging.info("VLM GRADIENT FILTER DEBUG INFO")
        logging.info("=" * 60)
        logging.info(f"  Total layers: {len(flat_grads)}")
        logging.info(f"  Kept (VLM params): {len(kept_keys)}")
        logging.info(f"  Filtered out: {len(filtered_keys)}")
        logging.info("-" * 60)
        logging.info(f"  Filter breakdown:")
        logging.info(f"    - Not VLM (no 'llm' in path): {len(filter_reasons['not_vlm'])}")
        logging.info(f"    - Action expert (llm.*_1.*): {len(filter_reasons['action_expert'])}")
        logging.info(f"    - Excluded heads: {len(filter_reasons['excluded_head'])}")
        
        # Detailed analysis of not_vlm layers
        if filter_reasons['not_vlm']:
            logging.info("-" * 60)
            logging.info("  Detailed breakdown of 'Not VLM' layers:")
            not_vlm_categories = {}
            for k in filter_reasons['not_vlm']:
                # Extract the top-level component name
                parts = k.split('/')
                if len(parts) > 0:
                    # Get first meaningful component (skip 'params' if present)
                    top_level = parts[0]
                    if top_level == 'params' and len(parts) > 1:
                        top_level = parts[1]
                    # Further categorize by second level if available
                    if len(parts) > 2:
                        category = f"{top_level}/{parts[2] if parts[1] == 'params' else parts[1]}"
                    else:
                        category = top_level
                else:
                    category = "unknown"
                
                if category not in not_vlm_categories:
                    not_vlm_categories[category] = []
                not_vlm_categories[category].append(k)
            
            # Sort by count (descending)
            sorted_categories = sorted(not_vlm_categories.items(), key=lambda x: -len(x[1]))
            for category, keys in sorted_categories:
                logging.info(f"      [{category}]: {len(keys)} layers")
            
            # Show sample keys for each category
            logging.info("-" * 60)
            logging.info("  Sample keys per category (first 2 each):")
            for category, keys in sorted_categories[:5]:  # Top 5 categories
                logging.info(f"    [{category}]:")
                for k in keys[:2]:
                    logging.info(f"      - {k}")
        
        logging.info("-" * 60)
        if debug and kept_keys:
            logging.info("  Sample kept keys (first 5):")
            for k in kept_keys[:5]:
                logging.info(f"    + {k}")
        if debug and filtered_keys:
            logging.info("  Sample filtered keys (first 5):")
            for k in filtered_keys[:5]:
                logging.info(f"    - {k}")
        logging.info("=" * 60)
    
    return traverse_util.unflatten_dict(vlm_grads, sep="/")


# ============================================================================
# FAST WARM-UP TRAINING STEP
# ============================================================================

_warmup_filter_logged = False  # Global flag to only log once

def train_step_fast_warmup(
    config: _config.TrainConfig,
    rng,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    warmup_opt_state,  # Optimizer state for discrete_head only
    *,
    warmup_tx,  # Simple optimizer for discrete_head (keyword-only, passed via partial)
):
    """
    FAST warm-up training step: only train discrete_head with discrete loss.
    
    During warm-up phase:
    - Only discrete (FAST) loss is computed  
    - Only discrete_head params are updated (other params completely frozen)
    - Uses a separate simple optimizer for discrete_head only
    
    This helps initialize the discrete head before joint training.
    """
    global _warmup_filter_logged
    
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def fast_only_loss_fn(
        model: _model.BaseModel, 
        rng: at.KeyArrayLike, 
        observation: _model.Observation, 
    ):
        # Only compute discrete loss
        if hasattr(model, 'compute_discrete_loss'):
            fast_loss = model.compute_discrete_loss(rng, observation, train=True)
            return jnp.mean(fast_loss)
        return jnp.zeros(())

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter to only compute gradients for discrete_head
    discrete_head_filter = nnx.All(nnx.Param, nnx.PathContains("discrete_head"))
    diff_state = nnx.DiffState(0, discrete_head_filter)
    
    loss, grads = nnx.value_and_grad(fast_only_loss_fn, argnums=diff_state)(
        model, train_rng, observation
    )
    
    # Get discrete_head params for update
    discrete_head_params = state.params.filter(discrete_head_filter)
    
    # Debug: Log which layers are being trained during warmup (only once)
    if not _warmup_filter_logged:
        _warmup_filter_logged = True
        
        # Get all params for comparison
        all_params = state.params.filter(config.trainable_filter)
        all_params_flat = traverse_util.flatten_dict(all_params.to_pure_dict(), sep="/")
        discrete_params_flat = traverse_util.flatten_dict(discrete_head_params.to_pure_dict(), sep="/")
        
        trainable_keys = list(discrete_params_flat.keys())
        frozen_keys = [k for k in all_params_flat.keys() if k not in discrete_params_flat]
        
        # Count parameters
        trainable_param_count = sum(v.size for v in discrete_params_flat.values())
        frozen_param_count = sum(v.size for k, v in all_params_flat.items() if k not in discrete_params_flat)
        
        logging.info("=" * 60)
        logging.info("FAST WARMUP FILTER DEBUG INFO")
        logging.info("=" * 60)
        logging.info(f"  Total trainable layers (normal training): {len(all_params_flat)}")
        logging.info(f"  Trainable layers (warmup, discrete_head): {len(trainable_keys)}")
        logging.info(f"  Frozen layers (warmup): {len(frozen_keys)}")
        logging.info("-" * 60)
        logging.info(f"  Trainable params (warmup): {trainable_param_count:,}")
        logging.info(f"  Frozen params (warmup): {frozen_param_count:,}")
        logging.info("-" * 60)
        logging.info("  [NOTE] Using separate optimizer for discrete_head ONLY")
        logging.info("  [NOTE] Other params completely frozen (no optimizer state)")
        logging.info("-" * 60)
        logging.info("  Trainable layers during warmup (discrete_head):")
        for k in trainable_keys[:10]:  # Show first 10
            logging.info(f"    + {k}")
        if len(trainable_keys) > 10:
            logging.info(f"    ... and {len(trainable_keys) - 10} more")
        logging.info("-" * 60)
        logging.info("  Frozen layers during warmup (sample, first 10):")
        for k in frozen_keys[:10]:
            logging.info(f"    - {k}")
        if len(frozen_keys) > 10:
            logging.info(f"    ... and {len(frozen_keys) - 10} more")
        logging.info("=" * 60)
    
    # Update only discrete_head params with simple optimizer
    updates, new_warmup_opt_state = warmup_tx.update(grads, warmup_opt_state, discrete_head_params)
    new_discrete_head_params = optax.apply_updates(discrete_head_params, updates)

    # Merge updated discrete_head params back into full model
    nnx.update(model, new_discrete_head_params)
    new_full_params = nnx.state(model)

    # Keep main opt_state unchanged during warmup (will be used fresh after warmup)
    new_state = dataclasses.replace(
        state, 
        step=state.step + 1, 
        params=new_full_params,
        # Note: opt_state is NOT updated during warmup - it stays fresh for hybrid training
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_full_params
            ),
        )

    # Filter out params that aren't kernels for param_norm
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    
    # Compute grad norm for discrete_head
    discrete_head_grad_norm = optax.global_norm(grads)
    
    info = {
        "loss": loss,
        "flow_loss": jnp.zeros(()),  # No flow loss during warmup
        "fast_loss": loss,
        "grad_norm": discrete_head_grad_norm,
        "flow_grad_norm": jnp.zeros(()),
        "fast_grad_norm": discrete_head_grad_norm,
        "flow_vlm_grad_norm": jnp.zeros(()),
        "fast_vlm_grad_norm": jnp.zeros(()),
        "param_norm": optax.global_norm(kernel_params),
        "lambda_fast": jnp.ones(()),  # During warmup, effectively lambda=1 (only fast loss)
        "warmup": jnp.ones(()),  # Flag to indicate warmup phase
    }
    return new_state, info, new_warmup_opt_state


# ============================================================================
# HYBRID TRAINING STEP
# ============================================================================

@at.typecheck
def train_step_hybrid(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    lambda_fast: at.Float[at.Array, ""],  # Weight for FAST loss (passed as JAX array for JIT)
    *,
    use_fast_loss: bool = True,  # Enable/disable FAST loss
    compute_separate_grad_norms: bool = True,  # Compute grad norms for flow and fast separately
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
        compute_separate_grad_norms: Whether to compute separate gradient norms for flow and fast losses
        
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
        actions: _model.Actions,
        lf: at.Float[at.Array, ""],  # lambda_fast value
    ):
        # Check if model supports hybrid loss
        if hasattr(model, 'compute_hybrid_loss') and use_fast_loss:
            # Use hybrid loss computation (Flow Matching + FAST)
            # Pass lambda_fast_override for adaptive lambda support
            total_loss, loss_info = model.compute_hybrid_loss(
                rng, observation, actions, train=True, lambda_fast_override=lf
            )
            # Must reduce to scalar for gradient computation!
            # loss_info already contains mean values for logging
            return jnp.mean(total_loss), loss_info
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
        model, train_rng, observation, actions, lambda_fast
    )

    # Compute separate gradient norms for flow and fast losses (VLM params only)
    flow_grad_norm = jnp.zeros(())
    fast_grad_norm = jnp.zeros(())
    flow_vlm_grad_norm = jnp.zeros(())
    fast_vlm_grad_norm = jnp.zeros(())
    
    if compute_separate_grad_norms and use_fast_loss and hasattr(model, 'compute_hybrid_loss'):
        # Need to recompute model for separate gradient computations
        model_for_grad = nnx.merge(state.model_def, state.params)
        model_for_grad.train()
        
        @at.typecheck
        def flow_loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, obs: _model.Observation, act: _model.Actions):
            # Only compute flow loss
            chunked_loss = model.compute_loss(rng, obs, act, train=True)
            return jnp.mean(chunked_loss)
        
        @at.typecheck 
        def fast_loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, obs: _model.Observation, act: _model.Actions):
            # Only compute fast loss
            if hasattr(model, 'compute_discrete_loss'):
                fast_loss = model.compute_discrete_loss(rng, obs, train=True)
                return jnp.mean(fast_loss)
            return jnp.zeros(())
        
        # Compute flow gradients
        flow_grads = nnx.grad(flow_loss_fn, argnums=diff_state)(model_for_grad, train_rng, observation, actions)
        flow_grad_norm = optax.global_norm(flow_grads)
        
        # Compute VLM-only flow gradient norm (for adaptive lambda)
        flow_vlm_grads = _filter_vlm_grads(flow_grads.to_pure_dict())
        flow_vlm_grad_norm = optax.global_norm(flow_vlm_grads) if flow_vlm_grads else jnp.zeros(())
        
        # Compute fast gradients (only if fast loss is available)
        if observation.fast_tokenized_prompt is not None:
            # Need fresh model state for fast grad computation
            model_for_fast = nnx.merge(state.model_def, state.params)
            model_for_fast.train()
            fast_grads = nnx.grad(fast_loss_fn, argnums=diff_state)(model_for_fast, train_rng, observation, actions)
            fast_grad_norm = optax.global_norm(fast_grads)
            
            # Compute VLM-only fast gradient norm (for adaptive lambda)
            fast_vlm_grads = _filter_vlm_grads(fast_grads.to_pure_dict())
            fast_vlm_grad_norm = optax.global_norm(fast_vlm_grads) if fast_vlm_grads else jnp.zeros(())

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
        "flow_grad_norm": flow_grad_norm,
        "fast_grad_norm": fast_grad_norm,
        "flow_vlm_grad_norm": flow_vlm_grad_norm,
        "fast_vlm_grad_norm": fast_vlm_grad_norm,
        "param_norm": optax.global_norm(kernel_params),
        "lambda_fast": lambda_fast,
    }
    return new_state, info


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main(
    config: _config.TrainConfig, 
    *, 
    lambda_fast: float = 0.1, 
    use_fast_loss: bool = True, 
    compute_separate_grad_norms: bool = True,
    grad_norm_compute_interval: int = 100,  # Interval for computing separate grad norms
    # FAST warm-up parameters
    fast_warmup_steps: int = 0,  # Number of warm-up steps for FAST head only
    # Adaptive lambda parameters
    adaptive_lambda: bool = True,
    adaptive_r: float = 0.2,  # Target ratio: lambda = r * (flow_grad / ce_grad)
    lambda_ema_decay: float = 0.99,  # EMA decay for lambda smoothing
    lambda_min: float = 1e-4,  # Minimum lambda value
    lambda_max: float = 0.3,  # Maximum lambda value
):
    """
    Main training function for hybrid Pi0.5 + FAST training.
    
    Args:
        config: Training configuration
        lambda_fast: Initial weight for FAST loss (default: 0.1)
        use_fast_loss: Whether to use FAST loss (default: False, flow matching only)
        compute_separate_grad_norms: Whether to compute separate gradient norms for flow and fast losses.
            WARNING: This adds ~2x compute overhead (extra backward passes). Use for debugging only.
        grad_norm_compute_interval: Interval for computing separate gradient norms (default: 100).
            Computing grad norms every step is expensive, so we only compute every N steps.
        fast_warmup_steps: Number of warm-up steps for FAST head only (default: 0).
            During warm-up: only discrete_head params are updated with discrete loss.
            After warm-up: normal hybrid training resumes.
        adaptive_lambda: Whether to adaptively adjust lambda_fast based on VLM gradient norms.
            Formula: lambda_fast = r * ||∇_vlm L_flow|| / (||∇_vlm L_ce|| + ε)
        adaptive_r: Target ratio for adaptive lambda (default: 0.2, range 0.1~0.3)
        lambda_ema_decay: EMA decay for smoothing lambda updates (default: 0.99)
        lambda_min: Minimum lambda value for clipping (default: 1e-4)
        lambda_max: Maximum lambda value for clipping (default: 0.3)
    """
    init_logging()
    logging.info(f"Running HYBRID training on: {platform.node()}")
    logging.info(f"  lambda_fast (initial): {lambda_fast}")
    logging.info(f"  use_fast_loss: {use_fast_loss}")
    logging.info(f"  fast_warmup_steps: {fast_warmup_steps}")
    if fast_warmup_steps > 0:
        logging.info(f"    [WARMUP] First {fast_warmup_steps} steps: only discrete_head trained with discrete loss")
    logging.info(f"  compute_separate_grad_norms: {compute_separate_grad_norms}")
    if compute_separate_grad_norms:
        logging.info(f"    grad_norm_compute_interval: {grad_norm_compute_interval} (computed every {grad_norm_compute_interval} steps to reduce overhead)")
    logging.info(f"  adaptive_lambda: {adaptive_lambda}")
    if adaptive_lambda:
        logging.info(f"    adaptive_r: {adaptive_r}")
        logging.info(f"    lambda_ema_decay: {lambda_ema_decay}")
        logging.info(f"    lambda_range: [{lambda_min}, {lambda_max}]")
        # Force compute_separate_grad_norms if adaptive_lambda is enabled
        if not compute_separate_grad_norms:
            logging.info("  [INFO] Enabling compute_separate_grad_norms for adaptive_lambda")
            compute_separate_grad_norms = True

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
    # Note: lambda_fast is passed dynamically for adaptive lambda support
    # Create two versions: one with grad norm computation, one without (for performance)
    ptrain_step_with_grad_norms = jax.jit(
        functools.partial(
            train_step_hybrid, 
            config,
            use_fast_loss=use_fast_loss,
            compute_separate_grad_norms=True,
        ),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    ptrain_step_no_grad_norms = jax.jit(
        functools.partial(
            train_step_hybrid, 
            config,
            use_fast_loss=use_fast_loss,
            compute_separate_grad_norms=False,
        ),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    
    # FAST warmup step: only train discrete_head with discrete loss
    # Uses a separate simple optimizer that ONLY operates on discrete_head params
    
    warmup_tx = None
    warmup_opt_state = None
    ptrain_step_fast_warmup = None
    
    if fast_warmup_steps > 0:
        # Create a simple optimizer for warmup (same LR as main optimizer)
        # This operates ONLY on discrete_head params (filtered separately)
        warmup_tx = optax.adam(learning_rate=config.lr_schedule.peak_lr)
        
        # Filter to get only discrete_head params
        discrete_head_filter = nnx.All(nnx.Param, nnx.PathContains("discrete_head"))
        discrete_head_params = train_state.params.filter(discrete_head_filter)
        
        # Initialize optimizer state for discrete_head only
        warmup_opt_state = warmup_tx.init(discrete_head_params)
        
        # Count params
        flat_params = traverse_util.flatten_dict(discrete_head_params.to_pure_dict(), sep="/")
        param_count = sum(v.size for v in flat_params.values())
        logging.info(f"Created warmup optimizer for discrete_head ({len(flat_params)} layers, {param_count:,} params)")
        
        # JIT the warmup step - warmup_tx is passed via functools.partial
        ptrain_step_fast_warmup = jax.jit(
            functools.partial(train_step_fast_warmup, config, warmup_tx=warmup_tx),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding, None),
            out_shardings=(train_state_sharding, replicated_sharding, None),
            donate_argnums=(1, 3),
        )
    
    # Note: grad_norm_compute_interval controls how often to compute expensive separate grad norms

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    loss_history = []
    
    # Current lambda value (will be updated if adaptive_lambda is enabled)
    current_lambda = lambda_fast
    # EMA of lambda for smoothing (initialized to initial lambda)
    lambda_ema = lambda_fast

    # Note: adaptive lambda update uses the same interval as grad_norm_compute_interval
    # since it depends on fresh gradient norms
    
    # Cache for gradient norms (updated every grad_norm_compute_interval steps)
    cached_grad_norms = {
        "flow_grad_norm": jnp.zeros(()),
        "fast_grad_norm": jnp.zeros(()),
        "flow_vlm_grad_norm": jnp.zeros(()),
        "fast_vlm_grad_norm": jnp.zeros(()),
    }
    
    # Cache for adaptive lambda info (to ensure consistent keys in info dict)
    cached_lambda_info = {
        "lambda_target": jnp.array(lambda_fast),
        "lambda_ema": jnp.array(lambda_fast),
    }
    
    # Track if we've logged the warmup end message
    warmup_ended_logged = False
    
    for step in pbar:
        # Check if we're in FAST warmup phase
        in_warmup_phase = fast_warmup_steps > 0 and step < fast_warmup_steps
        
        # Log warmup phase transition
        if fast_warmup_steps > 0 and step == fast_warmup_steps and not warmup_ended_logged:
            warmup_ended_logged = True
            logging.info(f"[WARMUP END] Step {step}: Transitioning from FAST warmup to hybrid training")
            logging.info(f"  - train_state.step: {train_state.step}")
            logging.info(f"  - discrete_head params have been warmed up")
            logging.info(f"  - Starting fresh optimizer state for full hybrid training")
            logging.info(f"  - (warmup momentum not preserved - this is intentional)")
        
        # Convert lambda to JAX array for JIT
        lambda_array = jnp.array(current_lambda, dtype=jnp.float32)
        
        # Decide whether to compute separate gradient norms this step
        # Only compute every grad_norm_compute_interval steps to save compute
        # Skip during warmup phase (no need for separate grad norms)
        should_compute_grad_norms = (
            compute_separate_grad_norms and 
            step % grad_norm_compute_interval == 0 and
            not in_warmup_phase
        )
        
        with sharding.set_mesh(mesh):
            if in_warmup_phase:
                # FAST warmup: only train discrete_head with discrete loss
                # Uses optax.masked optimizer - ONLY discrete_head is updated
                train_state, info, warmup_opt_state = ptrain_step_fast_warmup(
                    train_rng, train_state, batch, warmup_opt_state
                )
                # Add warmup flag and use cached/zero grad norms
                info["flow_grad_norm"] = cached_grad_norms["flow_grad_norm"]
                info["fast_grad_norm"] = info["grad_norm"]  # The grad_norm during warmup is fast grad norm
                info["flow_vlm_grad_norm"] = cached_grad_norms["flow_vlm_grad_norm"]
                info["fast_vlm_grad_norm"] = cached_grad_norms["fast_vlm_grad_norm"]
            elif should_compute_grad_norms:
                train_state, info = ptrain_step_with_grad_norms(train_rng, train_state, batch, lambda_array)
                # Update cached gradient norms
                cached_grad_norms = {
                    "flow_grad_norm": info["flow_grad_norm"],
                    "fast_grad_norm": info["fast_grad_norm"],
                    "flow_vlm_grad_norm": info["flow_vlm_grad_norm"],
                    "fast_vlm_grad_norm": info["fast_vlm_grad_norm"],
                }
                info["warmup"] = jnp.zeros(())  # Not in warmup
            else:
                train_state, info = ptrain_step_no_grad_norms(train_rng, train_state, batch, lambda_array)
                # Use cached gradient norms for logging
                info["flow_grad_norm"] = cached_grad_norms["flow_grad_norm"]
                info["fast_grad_norm"] = cached_grad_norms["fast_grad_norm"]
                info["flow_vlm_grad_norm"] = cached_grad_norms["flow_vlm_grad_norm"]
                info["fast_vlm_grad_norm"] = cached_grad_norms["fast_vlm_grad_norm"]
                info["warmup"] = jnp.zeros(())  # Not in warmup
        
        # === Adaptive Lambda Update ===
        # Only update when we have fresh gradient norms (same interval as grad_norm_compute_interval)
        # Skip during warmup phase
        if adaptive_lambda and use_fast_loss and should_compute_grad_norms and not in_warmup_phase:
            # Get VLM gradient norms from info
            flow_vlm_grad = float(jax.device_get(info["flow_vlm_grad_norm"]))
            fast_vlm_grad = float(jax.device_get(info["fast_vlm_grad_norm"]))
            
            # Compute target lambda: r * ||∇_vlm L_flow|| / (||∇_vlm L_ce|| + ε)
            epsilon = 1e-8
            if fast_vlm_grad > epsilon:
                target_lambda = adaptive_r * flow_vlm_grad / (fast_vlm_grad + epsilon)
            else:
                target_lambda = current_lambda  # Keep current if no valid gradient
            
            # Apply EMA smoothing
            lambda_ema = lambda_ema_decay * lambda_ema + (1 - lambda_ema_decay) * target_lambda
            
            # Clip to valid range
            current_lambda = max(lambda_min, min(lambda_max, lambda_ema))
            
            # Update cached lambda info
            cached_lambda_info = {
                "lambda_target": jnp.array(target_lambda),
                "lambda_ema": jnp.array(lambda_ema),
            }
        
        # Always add lambda info to ensure consistent keys for stack_forest
        if adaptive_lambda and use_fast_loss:
            info["lambda_target"] = cached_lambda_info["lambda_target"]
            info["lambda_ema"] = cached_lambda_info["lambda_ema"]
        
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
                        help="Initial weight for FAST loss in hybrid training")
    parser.add_argument("--use_fast_loss", action="store_true", default=True,
                        help="Enable FAST loss (enabled by default, requires model support)")
    parser.add_argument("--compute_separate_grad_norms", action="store_true", default=True,
                        help="Compute separate gradient norms for flow and fast losses (enabled by default)")
    parser.add_argument("--grad_norm_compute_interval", type=int, default=100,
                        help="Interval for computing separate gradient norms (default: 100, reduces overhead)")
    
    # FAST warmup arguments
    parser.add_argument("--fast_warmup_steps", type=int, default=3000,
                        help="Number of warmup steps for FAST head only (default: 0, disabled)")
    
    # Adaptive lambda arguments
    parser.add_argument("--adaptive_lambda", action="store_true", default=True,
                        help="Enable adaptive lambda_fast based on VLM gradient norms (enabled by default)")
    parser.add_argument("--adaptive_r", type=float, default=0.2,
                        help="Target ratio for adaptive lambda: lambda = r * (flow_grad / ce_grad). Range: 0.1~0.3")
    parser.add_argument("--lambda_ema_decay", type=float, default=0.99,
                        help="EMA decay for lambda smoothing")
    parser.add_argument("--lambda_min", type=float, default=1e-4,
                        help="Minimum lambda value for clipping")
    parser.add_argument("--lambda_max", type=float, default=0.3,
                        help="Maximum lambda value for clipping")
    
    # Parse known args first, let tyro handle the rest
    args, remaining = parser.parse_known_args()
    
    # Restore sys.argv for tyro
    import sys
    sys.argv = [sys.argv[0]] + remaining
    
    # Get config from tyro CLI
    config = _config.cli()
    # Handle compute_separate_grad_norms flag (default True, can be disabled with --no_compute_separate_grad_norms)
    compute_grad_norms = args.compute_separate_grad_norms
    
    # Print hybrid training configuration table
    print("\n" + "=" * 60)
    print("HYBRID TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  {'Parameter':<35} {'Value':<20}")
    print("-" * 60)
    print(f"  {'lambda_fast':<35} {args.lambda_fast:<20}")
    print(f"  {'use_fast_loss':<35} {args.use_fast_loss:<20}")
    print(f"  {'compute_separate_grad_norms':<35} {compute_grad_norms:<20}")
    print(f"  {'grad_norm_compute_interval':<35} {args.grad_norm_compute_interval:<20}")
    print("-" * 60)
    print("  FAST Warmup Settings:")
    print(f"  {'fast_warmup_steps':<35} {args.fast_warmup_steps:<20}")
    if args.fast_warmup_steps > 0:
        print(f"    [INFO] First {args.fast_warmup_steps} steps: only discrete_head trained")
    print("-" * 60)
    print("  Adaptive Lambda Settings:")
    print(f"  {'adaptive_lambda':<35} {args.adaptive_lambda:<20}")
    print(f"  {'adaptive_r':<35} {args.adaptive_r:<20}")
    print(f"  {'lambda_ema_decay':<35} {args.lambda_ema_decay:<20}")
    print(f"  {'lambda_min':<35} {args.lambda_min:<20}")
    print(f"  {'lambda_max':<35} {args.lambda_max:<20}")
    print("=" * 60 + "\n")
    
    main(
        config, 
        lambda_fast=args.lambda_fast, 
        use_fast_loss=args.use_fast_loss,
        compute_separate_grad_norms=compute_grad_norms,
        grad_norm_compute_interval=args.grad_norm_compute_interval,
        fast_warmup_steps=args.fast_warmup_steps,
        adaptive_lambda=args.adaptive_lambda,
        adaptive_r=args.adaptive_r,
        lambda_ema_decay=args.lambda_ema_decay,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
    )
