"""Pi0/Pi0.5 with the compressed-history visual encoder.

Same training contract as :mod:`openpi.models.pi0_mem` (accepts video clips
``[B, T, H, W, C]`` and exposes the same downstream prefix/suffix tokens), but
the visual backbone is :mod:`openpi.models.siglip_mem_compress` instead of
:mod:`openpi.models.siglip_mem`:

- Historical frames are compressed once into ``history_memory_tokens`` (M)
  learned memory tokens via a small resampler.
- Transformer blocks carry only current-frame tokens ``[B, N, D]``; every
  ``memory_every`` layers, current tokens cross-attend to the compressed
  history through a sigmoid-gated branch (zero-init gate).

This keeps the per-block compute close to a single-frame SigLIP forward
regardless of T, while still letting the policy condition on visual history.

The action / state / time-step modeling code is byte-for-byte identical to
``pi0_mem.Pi0Mem`` so existing PaliGemma weight loaders, EMA schedules and
checkpoint formats keep working.
"""

import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip_mem_compress as _siglip_mem_compress
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision (identical to ``pi0_mem.make_attn_mask``)."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Sine-cosine positional embedding for scalar positions (same as pi0_mem)."""
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


@dataclasses.dataclass(frozen=True)
class Pi0MemCompressConfig(pi0_config.Pi0Config):
    """Pi0/Pi0.5 config with the compressed-history MEM visual encoder.

    Mirrors ``Pi0MemConfig`` for the action/state side. The visual-side fields
    map directly onto the keyword arguments of
    ``siglip_mem_compress._Module``.
    """

    # Number of frames in the video clip (T). Default 1 means single image.
    num_frames: int = 1
    # Apply the cross-attention-to-history branch every K Transformer layers.
    # K==0 disables the memory branch (effectively pure single-frame SigLIP).
    memory_every: int = 4
    # Which frame is the policy-relevant "current" frame (default last).
    current_frame_index: int = -1
    # M: number of learned compressed history tokens.
    history_memory_tokens: int = 256
    # Number of cross-attention + MLP refinement layers inside the resampler.
    history_resampler_depth: int = 1
    # If True, condition the M memory queries on a pooled current-frame token.
    history_use_current_condition: bool = True
    # Initial logit of the sigmoid history gate. -6.9 -> sigmoid ~= 1e-3, so the
    # history branch starts tiny but the resampler still receives gradient.
    history_gate_init: float = -6.9
    # Optional fixed history gate probability for controlled experiments.
    # ``None`` keeps the learned sigmoid(logit) gate. Set to 0.5 or 1.0 to
    # force the memory branch to participate regardless of the learned logit.
    history_gate_fixed: float | None = None
    # Coefficient of the memory-token-diversity regularizer applied to the
    # compressed-history tensor returned by HistoryResampler. ``0.0`` disables
    # it entirely (no extra compute, no extra graph nodes). A small positive
    # value, e.g. ``1e-2`` -- ``5e-2``, pushes the M memory tokens apart and
    # avoids the typical "all queries collapse to one direction" failure
    # mode that shows up as ``memory/hist_token_cosine_offdiag_mean -> 1``
    # in wandb while ``grad/memory_queries_l2 -> 0``. See
    # ``scripts.mem.train_pi0_mem_compress.memory_diversity_loss``.
    diversity_weight: float = 0.0
    # Train-only augmentation used by scripts/mem/train_pi0_mem_compress.py:
    # with this per-sample probability, replace the policy-relevant current
    # frame by the neutral image value (0.0 in normalized [-1, 1] space) while
    # leaving historical frames intact. This forces the model to learn whether
    # history can recover action-relevant information.
    current_frame_dropout_prob: float = 0.0
    # Train-only per-pixel masking probability for the current frame. This is a
    # softer version of full-frame dropout and is applied only by the memory
    # training script when the value is > 0.
    current_frame_mask_prob: float = 0.0
    # Optional auxiliary action loss on a second view where only the current
    # frame is corrupted and history remains clean:
    #   action_loss = (loss(clean_current, clean_history)
    #                + weight * loss(corrupted_current, clean_history))
    #                / (1 + weight)
    # This encourages the policy to remain correct when current-frame evidence
    # is incomplete while keeping the history frames clean.
    current_frame_corrupt_loss_weight: float = 0.0
    # Optional optimizer update multiplier for ``history_memory_gate_logit``.
    # Kept at 1.0 by default so existing checkpoint resumes keep the same
    # optimizer-state structure. Set to e.g. 10.0 for a new run if the gate
    # remains stuck near its initialization value.
    history_gate_lr_multiplier: float = 1.0
    # Gradient checkpointing policy for the SigLIP encoder. See
    # ``Pi0MemConfig.siglip_remat_policy`` for the trade-off discussion; the
    # current-frame-only carry already makes the compressed encoder cheaper
    # than the temporal-attn one, so ``nothing_saveable`` is usually fine.
    siglip_remat_policy: str = "nothing_saveable"

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0MemCompress":
        return Pi0MemCompress(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, self.num_frames, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec


class Pi0MemCompress(_model.BaseModel):
    """Pi0 model with the compressed-history SigLIP visual encoder.

    Replaces the temporal-attention encoder of ``Pi0Mem`` with
    ``siglip_mem_compress``. The downstream PaliGemma LLM and action expert
    are unchanged.
    """

    def __init__(self, config: Pi0MemCompressConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.action_loss_mask = config.action_loss_mask
        self.num_frames = config.num_frames

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

        # Compressed-history visual encoder. Shape contract matches
        # ``siglip_mem.Module``: input ``[B, T, H, W, C]``, output
        # ``[B, N, paligemma_config.width]`` when ``pool_type='none'``.
        img = nnx_bridge.ToNNX(
            _siglip_mem_compress.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
                memory_every=config.memory_every,
                current_frame_index=config.current_frame_index,
                history_memory_tokens=config.history_memory_tokens,
                history_resampler_depth=config.history_resampler_depth,
                history_use_current_condition=config.history_use_current_condition,
                history_gate_init=config.history_gate_init,
                history_gate_fixed=config.history_gate_fixed,
                remat_policy=config.siglip_remat_policy,
            )
        )

        fake_obs = config.fake_obs()
        sample_image = next(iter(fake_obs.images.values()))
        img.lazy_init(sample_image, train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.deterministic = True

    def _embed_prefix_with_history_mem(self, obs: _model.Observation):
        """Internal embed-prefix that also returns the compressed history memory.

        Mirrors :meth:`embed_prefix` exactly but additionally collects the
        ``history_mem`` tensor produced by :mod:`siglip_mem_compress` for each
        image stream. Per-stream ``history_mem`` tensors are concatenated
        along the batch axis so the caller sees a single ``[B * S, M, D]``
        tensor (suitable for collapse / diversity metrics).

        Returns:
            ``(tokens, input_mask, ar_mask, history_mem_stacked)`` where
            ``history_mem_stacked`` is the concatenated history memory or
            ``None`` if the observation had no image streams.
        """
        input_mask = []
        ar_mask = []
        tokens = []
        history_mems = []
        for name in obs.images:
            image = obs.images[name]

            # Single image -> add a singleton time dim so the encoder always
            # sees [B, T, H, W, C]. Same handling as Pi0Mem.
            if image.ndim == 4:
                image = image[:, None, :, :, :]

            image_tokens, encoder_aux = self.PaliGemma.img(image, train=False)
            # ``encoder_aux["encoder"]["history_mem"]`` has shape [B, M, D].
            # It is zeros (shape-stable) when T==1 / cur_idx==0, so safe to
            # collect unconditionally.
            history_mems.append(encoder_aux["encoder"]["history_mem"])

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        history_mem_stacked = (
            jnp.concatenate(history_mems, axis=0) if history_mems else None
        )
        return tokens, input_mask, ar_mask, history_mem_stacked

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        tokens, input_mask, ar_mask, _ = self._embed_prefix_with_history_mem(obs)
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
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
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
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        squared_error = jnp.square(v_t - u_t)

        if observation.action_loss_mask is not None:
            mask = observation.action_loss_mask[..., None, :]
            squared_error_masked = squared_error * mask
            mask_sum = jnp.sum(mask, axis=-1, keepdims=True)
            mask_sum = jnp.maximum(mask_sum, 1e-8)
            loss_per_timestep = jnp.sum(squared_error_masked, axis=-1) / jnp.squeeze(mask_sum, axis=-1)
        elif self.action_loss_mask is not None:
            mask = jnp.asarray(self.action_loss_mask)
            squared_error_masked = squared_error * mask
            loss_per_timestep = jnp.sum(squared_error_masked, axis=-1) / jnp.sum(mask)
        else:
            loss_per_timestep = jnp.mean(squared_error, axis=-1)

        return loss_per_timestep

    def compute_loss_with_memory_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        """Same training loss as :meth:`compute_loss`, but also returns
        ``aux = {"history_mem": [B*S, M, D]}`` so trainer-side monitors can
        log memory collapse / diversity without an extra forward pass.

        Mirrors :meth:`compute_loss` exactly except it routes the prefix
        embedding through :meth:`_embed_prefix_with_history_mem` so the
        compressed history tokens computed inside
        :mod:`openpi.models.siglip_mem_compress` are surfaced.

        Returns:
            ``(loss_per_timestep, aux)``. ``aux["history_mem"]`` is the
            per-image-stream history memory concatenated along the batch
            axis (i.e. shape ``[B * num_image_streams, M, D]``). It is
            zero-shaped only if the observation has no image streams.
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask, history_mem = (
            self._embed_prefix_with_history_mem(observation)
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        squared_error = jnp.square(v_t - u_t)

        if observation.action_loss_mask is not None:
            mask = observation.action_loss_mask[..., None, :]
            squared_error_masked = squared_error * mask
            mask_sum = jnp.sum(mask, axis=-1, keepdims=True)
            mask_sum = jnp.maximum(mask_sum, 1e-8)
            loss_per_timestep = jnp.sum(squared_error_masked, axis=-1) / jnp.squeeze(mask_sum, axis=-1)
        elif self.action_loss_mask is not None:
            mask = jnp.asarray(self.action_loss_mask)
            squared_error_masked = squared_error * mask
            loss_per_timestep = jnp.sum(squared_error_masked, axis=-1) / jnp.sum(mask)
        else:
            loss_per_timestep = jnp.mean(squared_error, axis=-1)

        aux = {"history_mem": history_mem}
        return loss_per_timestep, aux

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
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
