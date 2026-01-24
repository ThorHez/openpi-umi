"""
Pi0.5 Model with Discrete Token Head for Hybrid Training.

This module extends the Pi0 (Pi0.5) model with an additional discrete token
prediction head, enabling hybrid training with both:
- Flow Matching loss (continuous action head, via Pi0/Pi0.5 compute_loss)
- FAST-style cross-entropy loss (token prediction, Pi0FAST-like single-stream)

Key features:
- Based on Pi0.5 backbone (multi-expert Gemma + adaRMSNorm), inherited from openpi.models.pi0.Pi0
- Adds an extra `discrete_head` (Linear: hidden_dim -> vocab_size) for token prediction
- Hybrid objective:
    L_total = L_flow + lambda_fast * L_fast

Important design note (current implementation):
- The FAST-style CE loss path DOES NOT use the Pi0.5 "Action Expert" stream.
  In `compute_discrete_loss`, the model calls:
      self.PaliGemma.llm(embedded=[token_embeddings, None], ...)
  i.e. the second (action-expert) stream is explicitly set to None.
  Therefore:
    - L_fast updates only the VLM-side parameters that participate in this forward
      (and any shared/common Gemma components, if present in the implementation),
      plus the added `discrete_head`.
    - L_fast does NOT update the action-expert parameters, because the action-expert
      stream is not executed in the CE forward pass.

Consequences:
- The "Knowledge Insulation" described in earlier drafts (stop_gradient on action-expert
  features) is not required for the current CE loss path, since action-expert is not used.
- The CE loss forward structure is closer to `Pi0FAST` (single-stream token prediction)
  than to Pi0.5's multi-expert prefix/suffix forward.

Weight loading:
- `CheckpointWeightLoaderWithDiscreteHead` loads a Pi0/Pi0.5 checkpoint and keeps random
  initialization for parameters missing from the checkpoint.
- The loader treats keys matching `(lora|discrete_head)` as expected-missing (kept random init).
  (If your discrete head is named `discrete_head`, it will be covered.)

Usage:
  1) Use `Pi0DiscreteConfig` instead of `Pi0Config`
  2) Instantiate the model normally; the `discrete_head` will be created
  3) Use `compute_hybrid_loss` to combine Flow Matching and CE losses
"""


import logging
from typing import Any

import einops

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0 as _pi0
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

import dataclasses

logger = logging.getLogger("openpi")

# PaliGemma vocab size (from gemma.py)
PALIGEMMA_VOCAB_SIZE = _gemma.PALIGEMMA_VOCAB_SIZE

@dataclasses.dataclass(frozen=True)
class Pi0DiscreteConfig(pi0_config.Pi0Config):
    """Configuration for Pi0.5 model with discrete action head.
<<<<<<< HEAD

    Extends Pi0Config with options for the discrete action head used in hybrid training.

    Action loss mask (inherited from Pi0Config):
    - action_loss_mask: When set, only the specified action dimensions contribute to the
      Flow Matching loss (e.g. (1.0,) * 10 + (0.0,) * 22 for 32-dim with first 10 real).
    - Per-sample mask from observation.action_loss_mask (e.g. via InjectActionLossMask)
      is supported in compute_hybrid_loss and takes precedence over this config-level mask.
    """

=======
    
    Extends Pi0Config with options for the discrete action head used in hybrid training.
    """
    
>>>>>>> b467a42 (update code)
    # Whether to enable discrete action head (for FAST-style token prediction)
    enable_discrete_head: bool = True
    
    # Vocab size for discrete action tokens (default: use full PaliGemma vocab)
    # Can be reduced if using a custom action tokenizer
    discrete_vocab_size: int = PALIGEMMA_VOCAB_SIZE
    
    # Weight for FAST loss in hybrid training: L_total = L_flow + lambda_fast * L_fast
    lambda_fast: float = 0.1

    # Keyword arguments for the fast model tokenizer.
    fast_model_tokenizer_kwargs: dict[str, Any] | None = None
    
    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Discrete":
        return Pi0Discrete(self, rngs=nnx.Rngs(rng))

def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


import jax
import jax.numpy as jnp

def _debug_check_masks_and_shift(
    obs,
    logits,          # [B, T-1, V]
    targets_id,      # [B, T-1]
    loss_mask_shift, # [B, T-1]
    token_mask_shift,# [B, T-1]
):
    # 1) padding 不应被监督：统计违反数量（而不是 assert）
    bad = jnp.sum((loss_mask_shift & (~token_mask_shift)).astype(jnp.int32))

    # 2) 形状检查：用 debug.print 输出
    jax.debug.print(
        "DBG shapes: logits={ls}, targets={ts}, loss_mask={ms}, token_mask={tms}",
        ls=logits.shape, ts=targets_id.shape, ms=loss_mask_shift.shape, tms=token_mask_shift.shape
    )

    # 3) postfix acc
    pred_id = jnp.argmax(logits, axis=-1)
    correct = (pred_id == targets_id) & loss_mask_shift
    acc = jnp.sum(correct) / jnp.clip(jnp.sum(loss_mask_shift), 1)

    # 4) 打印第0个样本的首个 loss idx / token 对齐
    b0 = 0
    k = jnp.argmax(loss_mask_shift[b0]).astype(jnp.int32)
    p = pred_id[b0, k]
    t = targets_id[b0, k]

    jax.debug.print(
        "CE-align: bad_loss_on_pad={bad}, acc(postfix)={acc:.4f}, first_loss_idx={k}, pred={p}, target={t}",
        bad=bad, acc=acc, k=k, p=p, t=t
    )

    return acc




class Pi0Discrete(_pi0.Pi0):
    """
    Pi0.5 Model with Discrete Action Head for Hybrid Training.
    
    This class extends Pi0 with a discrete action head that can predict action tokens
    for FAST-style training. The discrete head weights can be:
    1. Shared with the embedder (using embedder.decode())
    2. Independent (using a separate linear projection)
    
    When loading from pi05_base checkpoint, the discrete head weights will be
    randomly initialized if they don't exist in the checkpoint.
    """

    # Keyword arguments for the fast model tokenizer.
    fast_model_tokenizer_kwargs: dict[str, Any] | None = None
    
    def __init__(self, config: Pi0DiscreteConfig, rngs: nnx.Rngs):
        # Initialize parent Pi0 class
        super().__init__(config, rngs)
        
        # Store config options
        self.enable_discrete_head = config.enable_discrete_head
        # self.share_discrete_head_weights = config.share_discrete_head_weights
        self.discrete_vocab_size = config.discrete_vocab_size
        self.lambda_fast = config.lambda_fast
        
        # Get action expert config for hidden dim
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
    
        self.discrete_head = nnx.Linear(
                paligemma_config.width,  # Use VLM hidden dim after fast_out_proj
                PALIGEMMA_VOCAB_SIZE, 
                rngs=rngs,
            )
        logger.info(
            "Created discrete_head: %d -> %d", 
            paligemma_config.width, 
            PALIGEMMA_VOCAB_SIZE
        )

    @at.typecheck
    def embed_prefix_discrete(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
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
            # image tokens attend to each other --> AR mask = 0
            ar_mask.append(0 * input_mask[-1])

        # add tokenized inputs
        assert obs.fast_tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.fast_tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.fast_token_ar_mask is not None, "Token auto-regressive mask is required"
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.fast_tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs_embeddings)
        input_mask.append(obs.fast_tokenized_prompt_mask)
        ar_mask.append(obs.fast_token_ar_mask)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        return tokens, input_mask, ar_mask

    
    def get_discrete_logits(
        self, 
        embedded_output: at.Float[at.Array, "b t d"]
    ) -> at.Float[at.Array, "b t vocab"]:
        """
        Project embedded output to discrete vocab space for token prediction.
        
        Args:
            embedded_output: Output embeddings from the transformer [batch, seq_len, hidden_dim]
            
        Returns:
            Logits over vocab [batch, seq_len, vocab_size]
        """
        if not self.enable_discrete_head:
            raise RuntimeError("Discrete head is not enabled. Set enable_discrete_head=True in config.")
        
        # if self.share_discrete_head_weights:
        #     # Use the embedder's decode method (shared weights)
        #     # Access the underlying Linen module's embedder
        #     return self.PaliGemma.llm.module.embedder.decode(embedded_output)
        # else:
        #     # Use separate linear projection
        #     return self.discrete_head(embedded_output)
        return self.discrete_head(embedded_output)
    
    def compute_discrete_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
        use_knowledge_insulation: bool = True
    ) -> at.Float[at.Array, " b"]:
        """
        Compute cross-entropy loss for discrete token prediction (FAST-style).

        Current implementation (matches Pi0FAST-style single-stream CE):
        - Builds a single token sequence = [image tokens] + [fast_tokenized_prompt tokens]
        where `fast_token_ar_mask` defines prefix-LM behavior (0 on prefix, 1 on postfix).
        - Runs a ONE-stream Gemma forward pass via:
            self.PaliGemma.llm(embedded=[..., None], ...)
        The second stream (Pi0.5 Action Expert) is explicitly set to None and is not executed.
        - Projects the resulting pre-logits to vocab using `self.discrete_head`
        (Linear: hidden_dim -> vocab_size) and applies next-token cross-entropy.

        Gradient implications:
        - This CE loss updates:
            (1) VLM-side parameters used by the executed stream in `self.PaliGemma.llm`
            (2) the added `discrete_head`
        It does NOT update Action Expert parameters, because the Action Expert stream is not used.

        About `use_knowledge_insulation`:
        - Kept for API compatibility / future extension.
        - In the current code path, no additional stop_gradient is required to block
        gradients to the Action Expert, since that branch is not part of the forward pass.

        Note on targets/masks:
        - Targets are `fast_tokenized_prompt[:, 1:]` (next-token prediction).
        - Loss mask is `fast_token_loss_mask[:, 1:]` (compute CE only on postfix/action tokens).

        Args:
            rng: Random key.
            observation: Model observation containing:
                - images / image_masks
                - fast_tokenized_prompt, fast_tokenized_prompt_mask
                - fast_token_ar_mask (prefix-LM AR mask)
                - fast_token_loss_mask (which tokens contribute to CE)
            train: Whether in training mode (passed to preprocess_observation).
            use_knowledge_insulation: Unused in current implementation (reserved).

        Returns:
            Per-example cross-entropy loss: shape [batch].
        """
        preprocess_rng = rng
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        
        # Get target tokens and loss mask from observation (like FAST model)
        assert observation.fast_tokenized_prompt is not None, "fast_tokenized_prompt is required for discrete loss"
        assert observation.fast_token_loss_mask is not None, "fast_token_loss_mask is required for discrete loss"
        assert observation.fast_token_ar_mask is not None, "fast_token_ar_mask is required for discrete loss"
        assert observation.fast_tokenized_prompt_mask is not None, "fast_tokenized_prompt_mask is required for discrete loss"
        

        # Compute one-hot targets: we predict *next* token, so shift the input tokens by one.
        targets = jax.nn.one_hot(
            observation.fast_tokenized_prompt[:, 1:],
            PALIGEMMA_VOCAB_SIZE,
        )
        
        # Embed prefix (images + language) - VLM Expert
        input_token_embeddings, input_mask, ar_mask = self.embed_prefix_discrete(observation)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.maximum(jnp.cumsum(input_mask, axis=1) - 1, 0).astype(jnp.int32)
        positions_in = positions[:, :-1]

       # Each input predicts *next* token, so we don't input the last token.
        pre_logits, _ = self.PaliGemma.llm(
            embedded = [input_token_embeddings[:, :-1], None],
            mask=attn_mask[:, :-1, :-1],
            positions=positions_in,
            adarms_cond=[None, None],
        )
        
        # Only decode logits for the target tokens to save memory
        # (decoding matmul is large because it is a seq_len x vocab_size dense layer).
        # logits, _ = self.PaliGemma.llm(
        #     pre_logits=pre_logits[:, -targets.shape[1] :],
        # )
        text_shape = observation.fast_tokenized_prompt.shape[1]
        # logits = self.PaliGemma.llm(pre_logits[0][:, -(text_shape-1):], method="decode")
        logits = self.get_discrete_logits(pre_logits[0][:, -(text_shape-1):])
        logp = jax.nn.log_softmax(logits, axis=-1)

        # Compute CE loss on token targets
        assert observation.fast_token_loss_mask is not None, "Token loss mask is required"
        loss_mask = observation.fast_token_loss_mask[:, 1:]

        # targets_id = observation.fast_tokenized_prompt[:, 1:]          # [B, T-1]
        # token_mask_shift = observation.fast_tokenized_prompt_mask[:, 1:]
        # loss_mask_shift = observation.fast_token_loss_mask[:, 1:]
        # acc = _debug_check_masks_and_shift(observation, logits, targets_id, loss_mask_shift, token_mask_shift)


        token_pplx = jnp.sum(targets * logp, axis=-1)
        return -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

    def compute_hybrid_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        use_knowledge_insulation: bool = True,
        lambda_fast_override: float | None = None,  # Override self.lambda_fast if provided
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, Any]]:
        """
        Compute hybrid loss combining Flow Matching loss (Pi0/Pi0.5) and a FAST-style
        discrete token cross-entropy loss.

            L_total = L_flow + lambda_fast * L_fast

        OPTIMIZED: This version shares image encoding between Flow and FAST branches
        to avoid redundant SigLIP forward passes (~2x speedup vs naive implementation).

        Args:
            rng: Random key (split internally for flow/discrete).
            observation: Model observation.
            actions: Continuous actions for Flow Matching.
            train: Whether in training mode.
            use_knowledge_insulation: Unused in current implementation (reserved).
            lambda_fast_override: If provided, use this value instead of self.lambda_fast.
                Useful for adaptive lambda scheduling.

        Returns:
            (total_loss, loss_info_dict)
        """
        rng, preprocess_rng, flow_rng, discrete_rng = jax.random.split(rng, 4)
        
        # === SHARED PREPROCESSING (only once!) ===
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        
        # === SHARED IMAGE ENCODING (only once!) ===
        # Compute image embeddings that will be reused by both branches
        image_tokens_dict = {}
        image_masks_dict = {}
        for name in observation.images:
            img_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            image_tokens_dict[name] = img_tokens
            image_masks_dict[name] = einops.repeat(
                observation.image_masks[name],
                "b -> b s",
                s=img_tokens.shape[1],
            )
        
        # === 1. FLOW MATCHING LOSS (using cached image embeddings) ===
        flow_loss = self._compute_flow_loss_with_cached_images(
            flow_rng, observation, actions, image_tokens_dict, image_masks_dict
        )
        
        # === 2. FAST CE LOSS (using cached image embeddings) ===
        has_fast_tokens = (
            observation.fast_tokenized_prompt is not None and 
            observation.fast_token_loss_mask is not None and 
            self.enable_discrete_head
        )
        if has_fast_tokens:
            fast_loss = self._compute_discrete_loss_with_cached_images(
                discrete_rng, 
                observation,
                image_tokens_dict,
                image_masks_dict,
            )
            # Expand fast_loss to match flow_loss shape if needed
            if fast_loss.ndim < flow_loss.ndim:
                fast_loss = jnp.expand_dims(fast_loss, axis=-1)
                fast_loss = jnp.broadcast_to(fast_loss, flow_loss.shape)
        else:
            fast_loss = jnp.zeros_like(flow_loss)
        
        # === 3. COMBINE LOSSES ===
        # Use override lambda if provided, otherwise use self.lambda_fast
        effective_lambda = lambda_fast_override if lambda_fast_override is not None else self.lambda_fast
        total_loss = flow_loss + effective_lambda * fast_loss
        
        loss_info = {
            "flow_loss": jnp.mean(flow_loss),
            "fast_loss": jnp.mean(fast_loss),
            "total_loss": jnp.mean(total_loss),
            "lambda_fast": effective_lambda,
            "knowledge_insulation": use_knowledge_insulation,
        }
        
        return total_loss, loss_info

    def _compute_flow_loss_with_cached_images(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        image_tokens_dict: dict[str, at.Array],
        image_masks_dict: dict[str, at.Array],
    ) -> at.Float[at.Array, "*b ah"]:
        """
        Compute Flow Matching loss using pre-computed image embeddings.
        This is equivalent to parent compute_loss but reuses cached image tokens.
        """
        noise_rng, time_rng = jax.random.split(rng, 2)
        
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Build prefix using cached image tokens (no re-encoding!)
        input_mask = []
        ar_mask = []
        tokens = []
        
        for name in observation.images:
            tokens.append(image_tokens_dict[name])
            input_mask.append(image_masks_dict[name])
            ar_mask += [False] * image_tokens_dict[name].shape[1]

        # Add language tokens
        if observation.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(observation.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        
        prefix_tokens = jnp.concatenate(tokens, axis=1)
        prefix_mask = jnp.concatenate(input_mask, axis=1)
        prefix_ar_mask = jnp.array(ar_mask)

        # Build suffix (action expert tokens)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        
        # Combine and forward
        full_input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(full_input_mask, full_ar_mask)
        positions = jnp.cumsum(full_input_mask, axis=1) - 1
        
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

<<<<<<< HEAD
        # Compute squared error per action dimension
        squared_error = jnp.square(v_t - u_t)  # (*batch, action_horizon, action_dim)

        # Apply action dimension mask: per-sample (from data) for multi-dataset, or config-level for single-dataset
        if observation.action_loss_mask is not None:
            # Per-sample mask shape (*batch, ad) -> expand to (*batch, 1, ad) for broadcasting with (*batch, ah, ad)
            mask = observation.action_loss_mask[..., None, :]  # (*batch, 1, ad)
            squared_error_masked = squared_error * mask
            mask_sum = jnp.sum(mask, axis=-1, keepdims=True)
            mask_sum = jnp.maximum(mask_sum, 1e-8)
            loss_per_timestep = jnp.sum(squared_error_masked, axis=-1) / jnp.squeeze(mask_sum, axis=-1)
        elif self.action_loss_mask is not None:
            # Config-level mask (single dataset)
=======
        # Compute loss
        squared_error = jnp.square(v_t - u_t)
        if self.action_loss_mask is not None:
>>>>>>> b467a42 (update code)
            mask = jnp.asarray(self.action_loss_mask)
            squared_error_masked = squared_error * mask
            loss_per_timestep = jnp.sum(squared_error_masked, axis=-1) / jnp.sum(mask)
        else:
            loss_per_timestep = jnp.mean(squared_error, axis=-1)
<<<<<<< HEAD

=======
        
>>>>>>> b467a42 (update code)
        return loss_per_timestep

    def _compute_discrete_loss_with_cached_images(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        image_tokens_dict: dict[str, at.Array],
        image_masks_dict: dict[str, at.Array],
    ) -> at.Float[at.Array, " b"]:
        """
        Compute FAST CE loss using pre-computed image embeddings.
        This is equivalent to compute_discrete_loss but reuses cached image tokens.
        """
        assert observation.fast_tokenized_prompt is not None
        assert observation.fast_token_loss_mask is not None
        assert observation.fast_token_ar_mask is not None
        assert observation.fast_tokenized_prompt_mask is not None
        
        # Compute one-hot targets
        targets = jax.nn.one_hot(
            observation.fast_tokenized_prompt[:, 1:],
            PALIGEMMA_VOCAB_SIZE,
        )
        
        # Build input using cached image tokens (no re-encoding!)
        input_mask = []
        ar_mask = []
        tokens = []
        
        for name in observation.images:
            tokens.append(image_tokens_dict[name])
            input_mask.append(image_masks_dict[name])
            # image tokens attend to each other --> AR mask = 0
            ar_mask.append(0 * image_masks_dict[name])

        # Add tokenized inputs (FAST tokens)
        tokenized_inputs_embeddings = self.PaliGemma.llm(observation.fast_tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs_embeddings)
        input_mask.append(observation.fast_tokenized_prompt_mask)
        ar_mask.append(observation.fast_token_ar_mask)
        
        input_token_embeddings = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.maximum(jnp.cumsum(input_mask, axis=1) - 1, 0).astype(jnp.int32)
        positions_in = positions[:, :-1]

        # Forward pass (single stream, no action expert)
        pre_logits, _ = self.PaliGemma.llm(
            embedded=[input_token_embeddings[:, :-1], None],
            mask=attn_mask[:, :-1, :-1],
            positions=positions_in,
            adarms_cond=[None, None],
        )
        
        # Decode logits
        text_shape = observation.fast_tokenized_prompt.shape[1]
        logits = self.get_discrete_logits(pre_logits[0][:, -(text_shape-1):])
        logp = jax.nn.log_softmax(logits, axis=-1)

        # Compute CE loss
        loss_mask = observation.fast_token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)
        return -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)


# ============================================================================
# Weight Loaders for Pi0Discrete (Training & Inference)
# ============================================================================

from openpi.training import weight_loaders

# Re-export weight loaders for convenience
# Training: Load pi05_base checkpoint into Pi0Discrete model (keeps discrete_head random init)
CheckpointWeightLoaderWithDiscreteHead = weight_loaders.CheckpointWeightLoaderWithDiscreteHead

# Inference: Load hybrid-trained checkpoint into standard Pi0 model (ignores discrete_head)
<<<<<<< HEAD
CheckpointWeightLoaderIgnoreDiscreteHead = weight_loaders.CheckpointWeightLoaderIgnoreDiscreteHead
=======
CheckpointWeightLoaderIgnoreDiscreteHead = weight_loaders.CheckpointWeightLoaderIgnoreDiscreteHead
>>>>>>> b467a42 (update code)
