"""
Pi0.5 Hybrid Model with Flow Matching + FAST Loss

This module implements a hybrid Pi0.5 model that supports joint training of:
1. Flow Matching loss (continuous action prediction via velocity field)
2. FAST Cross-Entropy loss (discrete action token prediction)

The model architecture shares the VLM backbone (PaliGemma) between both objectives,
with separate output projections for each loss type.

Reference: Pi0.5 paper - Joint training of Flow Matching and FAST objectives

Usage:
    1. Configure data pipeline to provide FAST tokenized actions via:
       - observation.token_ar_mask: Autoregressive mask for attention
       - observation.token_loss_mask: Which tokens to compute loss on
    2. Use compute_hybrid_loss() for joint training
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
from openpi.models import pi0_config
from openpi.models import pi0
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask_for_fast(input_mask, mask_ar):
    """
    Create attention mask for FAST autoregressive prediction.
    
    Adapted from pi0_fast.py - supports both causal and prefix-lm attention.
    
    Args:
        input_mask: bool[B, N] true if its part of the input, false if padding.
        mask_ar: int[B, N] mask that's 1 for causal attention, 0 for bidirectional.
    
    Returns:
        Attention mask of shape [B, N, N]
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@dataclasses.dataclass(frozen=True)
class Pi0HybridConfig(pi0_config.Pi0Config):
    """Configuration for Pi0.5 Hybrid model with Flow Matching + FAST loss."""
    
    # FAST tokenizer settings
    fast_tokenizer_path: str = "physical-intelligence/fast"
    fast_max_token_len: int = 256
    
    # Hybrid loss weights
    lambda_fast: float = 0.1  # Weight for FAST loss: L_total = L_flow + lambda_fast * L_fast
    
    # Whether to use FAST loss during training
    use_fast_loss: bool = True
    
    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05
    
    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Hybrid":
        return Pi0Hybrid(self, rngs=nnx.Rngs(rng))


class Pi0Hybrid(_model.BaseModel):
    """
    Pi0.5 Hybrid Model supporting both Flow Matching and FAST losses.
    
    This model extends Pi0.5 with:
    1. Original Flow Matching loss for continuous action prediction
    2. FAST tokenization and CE loss for discrete action prediction
    
    The VLM backbone is shared, but there are separate output heads:
    - action_out_proj: For flow matching velocity prediction
    - (implicit via language model): For FAST token prediction
    """
    
    def __init__(self, config: Pi0HybridConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05
        self.lambda_fast = config.lambda_fast
        self.use_fast_loss = config.use_fast_loss
        
        # Store action loss mask if provided
        self.action_loss_mask = config.action_loss_mask
        if self.action_loss_mask is not None:
            logger.info(f"action_loss_mask: {self.action_loss_mask}")

        # Initialize VLM backbone (same as Pi0.5)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        
        # Flow Matching projections (same as Pi0.5)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        
        # Store vocab size for FAST loss computation
        self.vocab_size = paligemma_config.vocab_size if hasattr(paligemma_config, 'vocab_size') else 256000
        
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed observation prefix (images + language tokens)."""
        input_mask = []
        ar_mask = []
        tokens = []
        
        # Embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        # Add language tokens
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
            
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Embed action suffix with flow matching time conditioning."""
        input_mask = []
        ar_mask = []
        tokens = []
        
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = pi0.posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
            
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """
        Compute Flow Matching loss (original Pi0.5 loss).
        
        For hybrid training, use compute_hybrid_loss() which combines this with FAST loss.
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Forward pass
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # Compute squared error
        squared_error = jnp.square(v_t - u_t)
        
        # Apply action dimension mask if provided
        if self.action_loss_mask is not None:
            mask = jnp.asarray(self.action_loss_mask)
            squared_error_masked = squared_error * mask
            loss_per_timestep = jnp.sum(squared_error_masked, axis=-1) / jnp.sum(mask)
        else:
            loss_per_timestep = jnp.mean(squared_error, axis=-1)
        
        return loss_per_timestep

    def _embed_inputs_for_fast(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        """
        Embed inputs for FAST loss computation (images + tokenized prompt with action tokens).
        
        Similar to pi0_fast.embed_inputs but adapted for Pi0.5 architecture.
        """
        input_mask = []
        ar_mask = []
        token_embeddings = []
        
        # Embed images
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)
            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            # Image tokens attend to each other --> AR mask = 0
            ar_mask.append(jnp.zeros_like(input_mask[-1], dtype=jnp.int32))

        # Add tokenized inputs (prompt + FAST action tokens)
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required for FAST loss"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required for FAST loss"
        assert obs.token_ar_mask is not None, "Token AR mask is required for FAST loss"
        
        # Use VLM's embedding layer (first Gemma in the dual-Gemma architecture)
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    def compute_fast_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, ""]:
        """
        Compute FAST Cross-Entropy loss for action token prediction.
        
        This follows the implementation in pi0_fast.py:
        1. Embed inputs (images + tokenized prompt with FAST action tokens)
        2. Create attention mask (prefix-lm style: bidirectional on prefix, causal on suffix)
        3. Predict next token using VLM
        4. Compute CE loss only on action tokens (using token_loss_mask)
        
        Requirements:
            observation must contain:
            - tokenized_prompt: Token IDs including FAST-encoded actions
            - tokenized_prompt_mask: Valid token mask
            - token_ar_mask: Autoregressive mask (0 for prefix, 1 for causal)
            - token_loss_mask: Which tokens to compute loss on (True for action tokens)
        
        Args:
            rng: Random key
            observation: Model observation with FAST-tokenized actions
            actions: Ground truth actions (not used directly, actions are in tokenized_prompt)
            train: Whether in training mode
            
        Returns:
            Scalar FAST CE loss
        """
        # Check if observation has required FAST fields
        if observation.token_ar_mask is None or observation.token_loss_mask is None:
            # FAST loss requires tokenized actions - return 0 if not provided
            logger.warning("FAST loss requires token_ar_mask and token_loss_mask in observation. Returning 0.")
            return jnp.zeros(())
        
        # Preprocess observation (image augmentation etc.)
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        # Embed inputs: images + tokenized prompt with FAST action tokens
        input_token_embeddings, input_mask, ar_mask = self._embed_inputs_for_fast(observation)
        attn_mask = make_attn_mask_for_fast(input_mask, ar_mask)

        # Compute one-hot targets: predict *next* token, so shift by one
        # Get vocab size from the VLM's embedder
        try:
            vocab_size = self.PaliGemma.llm.module.embedder.vocab_size
        except AttributeError:
            vocab_size = self.vocab_size  # Fallback to stored value
        
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            vocab_size,
        )

        # Forward pass through VLM (first Gemma only, not action expert)
        # Each input predicts *next* token, so don't input the last token
        # Use the VLM Gemma for autoregressive token prediction
        positions = jnp.cumsum(input_mask[:, :-1], axis=1) - 1
        
        (vlm_out, _), _ = self.PaliGemma.llm(
            [input_token_embeddings[:, :-1], None],  # Only VLM path, no action expert
            mask=attn_mask[:, :-1, :-1],
            positions=positions,
            adarms_cond=[None, None],
        )
        
        # Get pre-logits for target tokens only (to save memory)
        pre_logits = vlm_out[:, -targets.shape[1]:]
        
        # Compute logits by projecting through embedding matrix transpose
        # This is the standard LM head: logits = pre_logits @ embedding.T
        # Access the embedder through the module wrapper
        try:
            # Access the internal Flax module's embedder
            embedder = self.PaliGemma.llm.module.embedder
            logits = embedder.decode(pre_logits.astype(jnp.float32))
        except AttributeError:
            # Fallback: manually compute using embedding table
            logger.warning("Could not access embedder.decode, computing logits manually")
            try:
                embed_table = self.PaliGemma.llm.module.embedder.input_embedding_table
                logits = jnp.dot(pre_logits.astype(jnp.float32), embed_table.T)
            except AttributeError:
                logger.error("Could not access embedding table for FAST logits computation")
                return jnp.zeros(())
        
        logp = jax.nn.log_softmax(logits, axis=-1)

        # Compute CE loss on action tokens only (using loss mask)
        loss_mask = observation.token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)  # Log probability of correct token
        
        # Negative log likelihood, normalized by number of loss tokens
        fast_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, axis=-1), 1)
        
        return jnp.mean(fast_loss)  # Average over batch

    def compute_hybrid_loss(
        self, 
        rng: at.KeyArrayLike, 
        observation: _model.Observation, 
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Float[at.Array, ""]]]:
        """
        Compute hybrid loss combining Flow Matching and FAST losses.
        
        L_total = L_flow + lambda_fast * L_fast
        
        The Flow Matching loss trains the action expert to predict continuous actions.
        The FAST loss trains the VLM to predict discrete action tokens autoregressively.
        
        Args:
            rng: Random key
            observation: Model observation (should include FAST tokenized actions for full hybrid training)
            actions: Ground truth actions
            train: Whether in training mode
            
        Returns:
            Tuple of (total_loss, loss_info_dict)
        """
        rng, flow_rng, fast_rng = jax.random.split(rng, 3)
        
        # 1. Compute Flow Matching loss (primary objective)
        flow_loss_per_step = self.compute_loss(flow_rng, observation, actions, train=train)
        flow_loss = jnp.mean(flow_loss_per_step)
        
        # 2. Compute FAST loss (auxiliary objective) - if enabled
        if self.use_fast_loss:
            fast_loss = self.compute_fast_loss(fast_rng, observation, actions, train=train)
        else:
            fast_loss = jnp.zeros(())
        
        # 3. Combine losses: L_total = L_flow + lambda_fast * L_fast
        total_loss = flow_loss + self.lambda_fast * fast_loss
        
        loss_info = {
            "flow_loss": flow_loss,
            "fast_loss": fast_loss,
            "total_loss": total_loss,
        }
        
        return total_loss, loss_info

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions using Flow Matching (same as Pi0.5)."""
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Fill KV cache with prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0


# ============================================================================
# Utility functions for hybrid training
# ============================================================================

def create_fast_tokenizer(config: Pi0HybridConfig) -> _tokenizer.FASTTokenizer:
    """Create FAST tokenizer for action tokenization."""
    return _tokenizer.FASTTokenizer(
        max_len=config.fast_max_token_len,
        fast_tokenizer_path=config.fast_tokenizer_path,
    )


def tokenize_actions_for_fast(
    fast_tokenizer: _tokenizer.FASTTokenizer,
    prompt: str,
    state: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tokenize actions using FAST tokenizer for CE loss computation.
    
    Returns:
        tokens: Token IDs
        token_mask: Valid token mask
        ar_mask: Autoregressive mask
        loss_mask: Which tokens to compute loss on
    """
    return fast_tokenizer.tokenize(prompt, state, actions)
