"""Pi0/Pi0.5 with the unified Past-Future Temporal Bottleneck visual encoder.

Extends :mod:`openpi.models.pi0_mem_compress` with the future-aware training
machinery from the paper "Past-Future Temporal Bottleneck for Memory-Augmented
Future-Aware Vision-Language-Action Control":

- The visual backbone is :mod:`openpi.models.siglip_pf`: past frames are
  compressed into history memory ``Hmem`` and future frames into a posterior
  latent ``Zpost`` by a *shared* Unified Temporal Resampler (UTR); current
  frame tokens read both through dual gated cross-attention (GTCA) branches
  inside the ViT blocks.
- A model-level **Future Latent Prior Encoder** predicts ``Zprior`` from
  information available at inference time only: current-frame visual tokens,
  ``Hmem``, language-token embeddings and the robot state.
- **Prior/posterior dual-branch training**: the same backbone + action expert
  run twice per training step, once with ``Zprior`` injected (prior branch,
  the deployed path) and once with ``Zpost`` injected (posterior branch, the
  train-only teacher). Both branches share the flow-matching noise / time
  draw, and a latent alignment loss pulls ``Zprior`` toward
  ``stop_gradient(Zpost)``.
- **Inference** keeps only the prior branch: future frames, the UTR future
  path and the posterior branch are never evaluated by ``sample_actions``.

Clip layout convention (must match the data pipeline):

    [oldest_past, ..., current, future_1, ..., future_F]

with ``current`` at index ``num_frames - 1`` and ``F = num_future_frames``
appended after it. At inference clips may simply omit the future frames
(``T == num_frames``); the encoder then emits a shape-stable zero ``Zpost``
which is never used anyway.

The action / state / time-step modeling code is byte-for-byte identical to
``pi0_mem_compress.Pi0MemCompress`` so existing PaliGemma weight loaders, EMA
schedules and checkpoint formats keep working.
"""

import dataclasses
import logging

import einops
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_mem_compress
import openpi.models.gemma as _gemma
import openpi.models.siglip_pf as _siglip_pf
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

make_attn_mask = pi0_mem_compress.make_attn_mask
posemb_sincos = pi0_mem_compress.posemb_sincos


# ---------------------------------------------------------------------------
# Future Latent Prior Encoder (model-level, linen, wrapped with nnx bridge).
# ---------------------------------------------------------------------------


class FutureLatentPriorEncoder(nn.Module):
    """Predict a future-aware latent ``Zprior`` from inference-time inputs.

    A set of ``num_latents`` learnable queries cross-attends over the fused
    conditioning context

        C_prior = concat(Px(x_cur), Ph(Hmem), Pl(E_lang), Ps(state))

    through ``depth`` pre-LN cross-attention + FFN blocks (paper eq. for
    ``PriorEncoder_phi``). Language positions respect the tokenized-prompt
    mask. The output is post-processed with the same
    center -> LayerNorm -> center recipe as the UTR latents so ``Zprior``
    lives on the same manifold as ``Zpost`` and the alignment loss is not
    dominated by scale/bias mismatches.
    """

    num_latents: int = 64
    width: int = 1152
    depth: int = 2
    num_heads: int = 12
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x_cur, hist_mem, lang_emb, lang_mask, state, deterministic=True):  # noqa: FBT002
        """Args:
        x_cur:    [B, N, Dv] patch-embedded current-frame tokens.
        hist_mem: [B, Mp, Dv] compressed history memory from the UTR.
        lang_emb: [B, L, Dl] language-token embeddings (LLM embedder output).
        lang_mask:[B, L] bool validity mask for the language tokens.
        state:    [B, Ds] robot proprioceptive state.

        Returns:
            [B, num_latents, width] prior latent ``Zprior``.
        """
        b = x_cur.shape[0]

        proj_kwargs = {
            "dtype": self.dtype_mm,
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.zeros,
        }
        x_c = nn.Dense(self.width, name="Px", **proj_kwargs)(x_cur.astype(self.dtype_mm))
        h_m = nn.Dense(self.width, name="Ph", **proj_kwargs)(hist_mem.astype(self.dtype_mm))
        l_e = nn.Dense(self.width, name="Pl", **proj_kwargs)(lang_emb.astype(self.dtype_mm))
        s_t = nn.Dense(self.width, name="Ps", **proj_kwargs)(state.astype(self.dtype_mm))[:, None, :]

        ctx = jnp.concatenate([x_c, h_m, l_e, s_t], axis=1)

        # Only language tokens can be padding; visual / memory / state slots
        # are always valid.
        valid = jnp.concatenate(
            [
                jnp.ones((b, x_c.shape[1]), dtype=jnp.bool_),
                jnp.ones((b, h_m.shape[1]), dtype=jnp.bool_),
                jnp.asarray(lang_mask, dtype=jnp.bool_),
                jnp.ones((b, 1), dtype=jnp.bool_),
            ],
            axis=1,
        )
        attn_mask = valid[:, None, None, :]

        queries = self.param(
            "prior_queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_latents, self.width),
            jnp.float32,
        )
        q = jnp.tile(queries.astype(self.dtype_mm), (b, 1, 1))

        for lyr in range(self.depth):
            q_norm = nn.LayerNorm(name=f"query_ln_{lyr}", dtype=self.dtype_mm)(q)
            c_norm = nn.LayerNorm(name=f"context_ln_{lyr}", dtype=self.dtype_mm)(ctx)
            y = nn.MultiHeadDotProductAttention(
                name=f"CrossAttention_{lyr}",
                num_heads=self.num_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                deterministic=deterministic,
                dtype=self.dtype_mm,
            )(q_norm, c_norm, mask=attn_mask)
            y = nn.Dropout(rate=self.dropout)(y, deterministic)
            q = q + y

            y = nn.LayerNorm(name=f"MlpLayerNorm_{lyr}", dtype=self.dtype_mm)(q)
            y = _siglip_pf.MlpBlock(
                name=f"MlpBlock_{lyr}",
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
            )(y, deterministic)
            y = nn.Dropout(rate=self.dropout)(y, deterministic)
            q = q + y

        # Same anti-collapse normalization as the UTR outputs so Zprior and
        # Zpost are directly comparable in the alignment loss.
        q = q - jnp.mean(q, axis=1, keepdims=True)
        q = nn.LayerNorm(name="out_ln", dtype=self.dtype_mm)(q)
        q = q - jnp.mean(q, axis=1, keepdims=True)
        return q


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Pi0MemPFConfig(pi0_mem_compress.Pi0MemCompressConfig):
    """Pi0/Pi0.5 config with the past-future temporal bottleneck encoder.

    Inherits every Pi0MemCompress field (``num_frames`` still means
    "past frames + current frame"; the current frame stays the *last past*
    slot). New fields configure the future side.
    """

    # Stride between consecutive past/current frames in raw dataset rows.
    # Training injects this into the Pi0Mem-aware data factory.
    frame_stride: int = 1
    # F: number of future frames appended after the current frame in the clip.
    # 0 disables the posterior path entirely (the model degenerates to a
    # compress-style history-only model plus a prior latent branch).
    num_future_frames: int = 0
    # Stride between consecutive future frames in raw dataset rows.
    # Training injects this into the Pi0Mem-aware data factory.
    future_frame_stride: int = 1
    # Mz: number of latent tokens for Zpost / Zprior.
    future_latent_tokens: int = 64
    # Condition Zpost on the pooled current frame (CVAE-style posterior
    # q(z | current, future)). The added projection is zero-init, so this is a
    # no-op at step 0 and stays checkpoint-compatible either way.
    future_use_current_condition: bool = True
    # Initial logit / optional fixed probability of the per-block future gate.
    future_gate_init: float = -6.9
    future_gate_fixed: float | None = None
    # Depth (cross-attn + FFN blocks) of the Future Latent Prior Encoder.
    prior_encoder_depth: int = 2
    # Loss weights: L = lp*L_prior + lq*L_post + la*L_align + lr*L_reg.
    lambda_prior: float = 1.0
    lambda_post: float = 1.0
    lambda_align: float = 1.0
    lambda_reg: float = 1e-4
    # Optional projection dim for the alignment loss (P_prior / P_post in the
    # paper). ``None`` aligns Zprior to sg(Zpost) directly with no projection
    # parameters.
    align_proj_dim: int | None = None

    @property
    def total_frames(self) -> int:
        """Clip length T = past+current (num_frames) + future frames."""
        return self.num_frames + self.num_future_frames

    @property
    def resolved_current_frame_index(self) -> int:
        """Current-frame index within the [past..., current, future...] clip.

        A negative ``current_frame_index`` is resolved against ``num_frames``
        (the past+current segment), NOT against the full clip, so the default
        ``-1`` keeps meaning "the last past frame" even when future frames
        are appended.
        """
        if self.current_frame_index >= 0:
            return self.current_frame_index
        return self.num_frames + self.current_frame_index

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0MemPF":
        return Pi0MemPF(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, self.total_frames, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Pi0MemPF(_model.BaseModel):
    """Pi0 model with the past-future temporal bottleneck visual encoder.

    Training runs the prior and posterior branches through the shared
    backbone; inference keeps only the prior branch.
    """

    def __init__(self, config: Pi0MemPFConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.action_loss_mask = config.action_loss_mask
        self.num_frames = config.num_frames
        self.num_future_frames = config.num_future_frames
        self.future_latent_tokens = config.future_latent_tokens

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

        vit_variant = "So400m/14"
        vit_width = _siglip_pf.decode_variant(vit_variant)["width"]
        vit_heads = _siglip_pf.decode_variant(vit_variant)["num_heads"]
        self._vit_width = vit_width
        self._llm_width = paligemma_config.width

        img = nnx_bridge.ToNNX(
            _siglip_pf.Module(
                num_classes=paligemma_config.width,
                variant=vit_variant,
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
                memory_every=config.memory_every,
                current_frame_index=config.resolved_current_frame_index,
                history_memory_tokens=config.history_memory_tokens,
                history_resampler_depth=config.history_resampler_depth,
                history_use_current_condition=config.history_use_current_condition,
                history_gate_init=config.history_gate_init,
                history_gate_fixed=config.history_gate_fixed,
                future_latent_tokens=config.future_latent_tokens,
                future_use_current_condition=config.future_use_current_condition,
                future_gate_init=config.future_gate_init,
                future_gate_fixed=config.future_gate_fixed,
                remat_policy=config.siglip_remat_policy,
            )
        )

        # ``mode="full"`` traces the UTR (both directions when future frames
        # are present) and the dual-branch blocks, creating every encoder
        # parameter in one pass.
        fake_obs = config.fake_obs()
        sample_image = next(iter(fake_obs.images.values()))
        img.lazy_init(sample_image, train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Future Latent Prior Encoder. The dummy sequence lengths used for
        # lazy_init are irrelevant: every parameter shape depends only on
        # feature widths.
        prior = nnx_bridge.ToNNX(
            FutureLatentPriorEncoder(
                num_latents=config.future_latent_tokens,
                width=vit_width,
                depth=config.prior_encoder_depth,
                num_heads=vit_heads,
                dtype_mm=config.dtype,
            )
        )
        prior.lazy_init(
            jnp.zeros((1, 4, vit_width)),
            jnp.zeros((1, max(config.history_memory_tokens, 1), vit_width)),
            jnp.zeros((1, 4, paligemma_config.width)),
            jnp.zeros((1, 4), dtype=jnp.bool_),
            jnp.zeros((1, config.action_dim)),
            rngs=rngs,
        )
        self.FuturePrior = prior

        # Optional alignment projections (P_prior / P_post). ``None`` means
        # the alignment loss compares the latents directly.
        self.align_proj_dim = config.align_proj_dim
        if config.align_proj_dim is not None:
            self.align_proj_prior = nnx.Linear(vit_width, config.align_proj_dim, rngs=rngs)
            self.align_proj_post = nnx.Linear(vit_width, config.align_proj_dim, rngs=rngs)

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

    # ------------------------------------------------------------------
    # Stage A: per-stream UTR memories + prior latent prediction.
    # ------------------------------------------------------------------

    def _encode_memories(self, obs: _model.Observation):
        """Run the UTR stage on every image stream.

        Returns three dicts keyed by image-stream name:
            ``x_curs``:    [B, N, Dv] patch-embedded current-frame tokens.
            ``hist_mems``: [B, Mp, Dv] history memory (Hmem).
            ``z_posts``:   [B, Mz, Dv] posterior latent (Zpost; zeros when the
                clip carries no future frames).
        """
        x_curs, hist_mems, z_posts = {}, {}, {}
        for name in obs.images:
            image = obs.images[name]
            if image.ndim == 4:
                image = image[:, None, :, :, :]
            x_cur, aux = self.PaliGemma.img(image, train=False, mode="memories")
            x_curs[name] = x_cur
            hist_mems[name] = aux["encoder"]["history_mem"]
            z_posts[name] = aux["encoder"]["future_post"]
        return x_curs, hist_mems, z_posts

    def _language_embedding(self, obs: _model.Observation):
        """Language-token embeddings + mask for prior-encoder conditioning."""
        if obs.tokenized_prompt is not None:
            lang_emb = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            lang_mask = obs.tokenized_prompt_mask
        else:
            b = obs.state.shape[0]
            lang_emb = jnp.zeros((b, 1, self._llm_width), dtype=jnp.float32)
            lang_mask = jnp.zeros((b, 1), dtype=jnp.bool_)
        return lang_emb, lang_mask

    def _compute_prior_latents(self, obs: _model.Observation, x_curs, hist_mems):
        """Predict Zprior for every image stream (shared prior-encoder params)."""
        lang_emb, lang_mask = self._language_embedding(obs)
        return {
            name: self.FuturePrior(
                x_curs[name], hist_mems[name], lang_emb, lang_mask, obs.state, deterministic=True
            )
            for name in x_curs
        }

    # ------------------------------------------------------------------
    # Stage B: branch-specific prefix embedding.
    # ------------------------------------------------------------------

    def _embed_prefix_with_latents(self, obs: _model.Observation, hist_mems, future_latents):
        """Embed the prefix with a specific injected future latent per stream.

        ``future_latents[name]`` is either Zprior (prior branch / inference)
        or Zpost (posterior branch). Mirrors
        ``Pi0MemCompress._embed_prefix_with_history_mem`` otherwise.
        """
        input_mask = []
        ar_mask = []
        tokens = []
        encoder_auxes = []
        for name in obs.images:
            image = obs.images[name]
            if image.ndim == 4:
                image = image[:, None, :, :, :]

            image_tokens, aux = self.PaliGemma.img(
                image,
                hist_mems[name],
                future_latents[name],
                train=False,
                mode="current",
            )
            encoder_auxes.append(aux["encoder"])

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
        return tokens, input_mask, ar_mask, tuple(encoder_auxes)

    def embed_prefix(self, obs: _model.Observation):
        """Prior-branch prefix embedding (the deployed path)."""
        x_curs, hist_mems, _ = self._encode_memories(obs)
        z_priors = self._compute_prior_latents(obs, x_curs, hist_mems)
        tokens, input_mask, ar_mask, _ = self._embed_prefix_with_latents(obs, hist_mems, z_priors)
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

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _flow_matching_loss(self, prefix, suffix, u_t, observation):
        """One LLM forward + per-timestep flow-matching loss for one branch."""
        prefix_tokens, prefix_mask, prefix_ar_mask = prefix
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = suffix

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
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

    def _align_and_reg_losses(self, z_prior, z_post):
        """Latent alignment (prior -> sg(posterior)) + scale regularization.

        Args:
            z_prior: [B*, Mz, D] stacked prior latents (across image streams).
            z_post:  [B*, Mz, D] stacked posterior latents.
        """
        z_prior = jnp.asarray(z_prior, dtype=jnp.float32)
        z_post = jnp.asarray(z_post, dtype=jnp.float32)

        if self.align_proj_dim is not None:
            p_prior = self.align_proj_prior(z_prior)
            p_post = self.align_proj_post(z_post)
        else:
            p_prior = z_prior
            p_post = z_post

        align = jnp.mean(jnp.square(p_prior - jax.lax.stop_gradient(p_post)))
        reg = jnp.mean(jnp.square(p_prior)) + jnp.mean(jnp.square(p_post))
        return align, reg

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """Prior-branch flow-matching loss only (deployment-path objective).

        The full PF objective (posterior branch + alignment + regularization)
        lives in :meth:`compute_loss_with_pf_aux`; this method exists for
        drop-in compatibility with generic eval / ablation tooling.
        """
        loss_prior, _ = self.compute_loss_with_pf_aux(rng, observation, actions, train=train, run_posterior=False)
        return loss_prior

    def compute_loss_with_pf_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        run_posterior: bool = True,
    ):
        """Dual-branch PF training losses.

        Runs stage A (UTR memories) once, predicts Zprior, then runs stage B
        (ViT blocks) + the LLM once per branch with the *same* flow-matching
        noise / time draw, so the prior-posterior loss gap directly reflects
        the value of the injected latent rather than sampling noise.

        Returns:
            ``(loss_prior_per_timestep, aux)`` where ``aux`` contains:

            - ``loss_post``: posterior-branch per-timestep loss (zeros when
              ``run_posterior=False``).
            - ``align_loss`` / ``reg_loss``: scalar latent losses.
            - ``history_mem`` / ``future_post`` / ``future_prior``: latents
              stacked along the batch axis ``[B * num_streams, M, D]`` for
              trainer-side collapse / alignment monitors.
            - ``encoder_auxes``: prior-branch per-stream block internals
              (gates, branch residual norms).
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Stage A once: UTR memories + current-frame tokens per stream.
        x_curs, hist_mems, z_posts = self._encode_memories(observation)
        z_priors = self._compute_prior_latents(observation, x_curs, hist_mems)

        # The suffix is branch-independent (state / noisy actions / time).
        suffix = self.embed_suffix(observation, x_t, time)

        # Prior branch (deployment path).
        prior_prefix_tokens, prefix_mask, prefix_ar_mask, prior_encoder_auxes = self._embed_prefix_with_latents(
            observation, hist_mems, z_priors
        )
        loss_prior = self._flow_matching_loss(
            (prior_prefix_tokens, prefix_mask, prefix_ar_mask), suffix, u_t, observation
        )

        # Posterior branch (train-only teacher).
        if run_posterior:
            post_prefix_tokens, post_mask, post_ar_mask, post_encoder_auxes = self._embed_prefix_with_latents(
                observation, hist_mems, z_posts
            )
            loss_post = self._flow_matching_loss(
                (post_prefix_tokens, post_mask, post_ar_mask), suffix, u_t, observation
            )
        else:
            loss_post = jnp.zeros_like(loss_prior)
            post_encoder_auxes = ()

        # Latent alignment / regularization on the stacked latents.
        z_prior_stacked = jnp.concatenate(list(z_priors.values()), axis=0)
        z_post_stacked = jnp.concatenate(list(z_posts.values()), axis=0)
        hist_mem_stacked = jnp.concatenate(list(hist_mems.values()), axis=0)
        align_loss, reg_loss = self._align_and_reg_losses(z_prior_stacked, z_post_stacked)

        aux = {
            "loss_post": loss_post,
            "align_loss": align_loss,
            "reg_loss": reg_loss,
            "history_mem": hist_mem_stacked,
            "future_post": z_post_stacked,
            "future_prior": z_prior_stacked,
            "encoder_auxes": prior_encoder_auxes,
            "post_encoder_auxes": post_encoder_auxes,
        }
        return loss_prior, aux

    # ------------------------------------------------------------------
    # Inference (prior branch only)
    # ------------------------------------------------------------------

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
            _x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
