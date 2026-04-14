"""
Pi0-FAST Hybrid Model with FAST + Flow Matching Loss

This module implements a hybrid model based on Pi0-FAST that supports joint training of:
1. FAST Cross-Entropy loss (discrete action token prediction) - Primary
2. Flow Matching loss (continuous action prediction) - Auxiliary

This approach adds Flow Matching to Pi0-FAST rather than adding FAST to Pi0.5,
because Pi0-FAST uses a simpler single-Gemma architecture that's better suited
for autoregressive token prediction.

Usage:
    python scripts/train_hybrid.py --config your_config --use_fast_loss --lambda_flow 0.1
"""

import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_fast
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi0FASTHybridConfig(pi0_fast.Pi0FASTConfig):
    """Configuration for Pi0-FAST Hybrid model with FAST + Flow Matching loss."""
    
    # Hybrid loss weights
    lambda_flow: float = 0.1  # Weight for Flow Matching loss
    
    # Whether to use Flow Matching loss during training
    use_flow_loss: bool = True
    
    # Flow Matching specific settings
    flow_num_steps: int = 10  # Number of denoising steps for sampling
    
    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTHybrid":
        return Pi0FASTHybrid(self, rngs=nnx.Rngs(rng))


class Pi0FASTHybrid(pi0_fast.Pi0FAST):
    """
    Pi0-FAST Hybrid Model supporting both FAST and Flow Matching losses.
    
    Extends Pi0-FAST with:
    1. Original FAST CE loss for discrete action token prediction (primary)
    2. Flow Matching loss for continuous action prediction (auxiliary)
    
    The model uses a single Gemma for token prediction (FAST) and adds
    a lightweight action projection head for Flow Matching.
    """
    
    def __init__(self, config: Pi0FASTHybridConfig, rngs: nnx.Rngs):
        # Initialize base Pi0-FAST
        super().__init__(config, rngs=rngs)
        
        self.lambda_flow = config.lambda_flow
        self.use_flow_loss = config.use_flow_loss
        self.flow_num_steps = config.flow_num_steps
        
        # Additional projections for Flow Matching
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        hidden_dim = paligemma_config.width
        
        # Action input projection (for noisy actions)
        self.action_in_proj = nnx.Linear(config.action_dim, hidden_dim, rngs=rngs)
        
        # Time embedding MLP
        self.time_mlp_in = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.time_mlp_out = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        
        # Action output projection (for velocity prediction)
        self.action_out_proj = nnx.Linear(hidden_dim, config.action_dim, rngs=rngs)
        
        logger.info(f"Pi0FASTHybrid initialized with lambda_flow={self.lambda_flow}, use_flow_loss={self.use_flow_loss}")

    @at.typecheck
    def _posemb_sincos(
        self, pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float = 4e-3, max_period: float = 4.0
    ) -> at.Float[at.Array, "b {embedding_dim}"]:
        """Computes sine-cosine positional embedding vectors for scalar positions (timestep)."""
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

        fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
        period = min_period * (max_period / min_period) ** fraction
        sinusoid_input = jnp.einsum(
            "i,j->ij",
            pos,
            1.0 / period * 2 * jnp.pi,
            precision=jax.lax.Precision.HIGHEST,
        )
        return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)

    def compute_flow_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, ""]:
        """
        Compute Flow Matching loss for continuous action prediction.
        
        This adds a lightweight Flow Matching head to the FAST model.
        
        Args:
            rng: Random key
            observation: Model observation
            actions: Ground truth continuous actions [B, T, D]
            train: Whether in training mode
            
        Returns:
            Scalar Flow Matching loss (MSE on velocity prediction)
        """
        noise_rng, time_rng = jax.random.split(rng)
        
        batch_size = actions.shape[0]
        
        # Sample noise and timestep
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
        time_expanded = time[:, None, None]
        
        # Interpolate between noise and actions
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # Target velocity
        
        # Get prefix embeddings (images + prompt)
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        
        # Project noisy actions
        action_tokens = self.action_in_proj(x_t)  # [B, T, D]
        
        # Time embedding with MLP
        time_emb = self._posemb_sincos(time, self.action_in_proj.out_features)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        
        # Add time embedding to action tokens
        action_tokens = action_tokens + time_emb[:, None, :]
        
        # Concatenate prefix and action tokens
        full_embeddings = jnp.concatenate([input_token_embeddings, action_tokens], axis=1)
        
        # Create attention mask (prefix bidirectional, actions causal)
        action_seq_len = actions.shape[1]
        action_mask = jnp.ones((batch_size, action_seq_len), dtype=jnp.bool_)
        action_ar_mask = jnp.ones((batch_size, action_seq_len), dtype=jnp.int32)
        action_ar_mask = action_ar_mask.at[:, 0].set(1)  # First action token starts causal
        
        full_mask = jnp.concatenate([input_mask, action_mask], axis=1)
        full_ar_mask = jnp.concatenate([ar_mask, action_ar_mask], axis=1)
        
        attn_mask = pi0_fast.make_attn_mask(full_mask, full_ar_mask)
        
        # Forward pass through Gemma
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=full_embeddings,
            mask=attn_mask,
            return_prelogits=True,
        )
        
        # Extract action outputs
        action_outputs = pre_logits[:, -action_seq_len:]
        
        # Project to velocity prediction
        v_t = self.action_out_proj(action_outputs)
        
        # MSE loss
        loss = jnp.mean(jnp.square(v_t - u_t))
        
        return loss

    def compute_hybrid_loss(
        self, 
        rng: at.KeyArrayLike, 
        observation: _model.Observation, 
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Float[at.Array, ""]]]:
        """
        Compute hybrid loss combining FAST and Flow Matching losses.
        
        L_total = L_fast + lambda_flow * L_flow
        
        Args:
            rng: Random key
            observation: Model observation (should include FAST tokenized actions)
            actions: Ground truth continuous actions
            train: Whether in training mode
            
        Returns:
            Tuple of (total_loss, loss_info_dict)
        """
        rng, fast_rng, flow_rng = jax.random.split(rng, 3)
        
        # 1. Compute FAST CE loss (primary objective)
        fast_loss_per_batch = self.compute_loss(fast_rng, observation, actions, train=train)
        fast_loss = jnp.mean(fast_loss_per_batch)
        
        # 2. Compute Flow Matching loss (auxiliary objective) - if enabled
        if self.use_flow_loss:
            flow_loss = self.compute_flow_loss(flow_rng, observation, actions, train=train)
        else:
            flow_loss = jnp.zeros(())
        
        # 3. Combine losses: L_total = L_fast + lambda_flow * L_flow
        total_loss = fast_loss + self.lambda_flow * flow_loss
        
        loss_info = {
            "fast_loss": fast_loss,
            "flow_loss": flow_loss,
            "total_loss": total_loss,
        }
        
        return total_loss, loss_info

    def sample_actions_flow(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
    ) -> _model.Actions:
        """
        Sample actions using Flow Matching (iterative denoising).
        
        Alternative to FAST autoregressive sampling.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        
        batch_size = observation.state.shape[0]
        dt = -1.0 / num_steps
        
        # Start from noise
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        
        # Get prefix embeddings
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)

        def step(carry):
            x_t, time = carry
            
            # Project noisy actions
            action_tokens = self.action_in_proj(x_t)
            
            # Time embedding
            time_emb = self._posemb_sincos(jnp.broadcast_to(time, (batch_size,)), self.action_in_proj.out_features)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_tokens = action_tokens + time_emb[:, None, :]
            
            # Forward pass
            full_embeddings = jnp.concatenate([input_token_embeddings, action_tokens], axis=1)
            action_seq_len = x_t.shape[1]
            action_mask = jnp.ones((batch_size, action_seq_len), dtype=jnp.bool_)
            action_ar_mask = jnp.ones((batch_size, action_seq_len), dtype=jnp.int32)
            
            full_mask = jnp.concatenate([input_mask, action_mask], axis=1)
            full_ar_mask = jnp.concatenate([ar_mask, action_ar_mask], axis=1)
            attn_mask = pi0_fast.make_attn_mask(full_mask, full_ar_mask)
            
            pre_logits, _, _ = self.PaliGemma.llm(
                embedded_prefix=full_embeddings,
                mask=attn_mask,
                return_prelogits=True,
            )
            
            action_outputs = pre_logits[:, -action_seq_len:]
            v_t = self.action_out_proj(action_outputs)
            
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
