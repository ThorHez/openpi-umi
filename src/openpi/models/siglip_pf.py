"""SigLIP current-frame encoder with a unified Past-Future Temporal Bottleneck.

This extends :mod:`openpi.models.siglip_mem_compress` (compressed visual
history) with the future-aware machinery from the paper "Past-Future Temporal
Bottleneck for Memory-Augmented Future-Aware Vision-Language-Action Control":

- **Unified Temporal Resampler (UTR)**: past frames and future frames are
  compressed by the *same* ``ResamplerCore`` (shared cross-attention + FFN
  parameters), but each direction has its own learnable queries, role
  embedding and output projection. The past path yields the history memory
  ``Hmem`` (usable at train and inference time); the future path yields the
  posterior latent ``Zpost`` (train-only, since future frames are only
  available during training).
- **Dual gated temporal cross-attention (GTCA) inside the ViT blocks**: on
  scheduled memory layers the current-frame tokens read from ``Hmem`` through
  the existing sigmoid-gated history branch, *and* from a branch-specific
  future latent ``Zb`` (either ``Zprior`` or ``Zpost``) through a second,
  independently gated cross-attention branch. Both extra branches use
  zero-ish-init output projections so the pretrained single-frame behavior is
  preserved at initialization.
- **Two-stage forward API**: the Future Latent Prior Encoder lives at the
  model level (it needs language / robot-state tokens that only exist there),
  so this module exposes:

  1. ``mode="memories"``: patch-embed every frame, run the UTR, and return
     ``(x_cur_tokens, out)`` where ``out["encoder"]["history_mem"]`` is
     ``Hmem`` and ``out["encoder"]["future_post"]`` is ``Zpost``. No ViT
     blocks are run.
  2. ``mode="current"``: given ``hist_mem`` and a branch-specific
     ``future_latent`` (``Zprior`` or ``Zpost``), patch-embed only the
     current frame and run the ViT blocks with the dual gated injection.
     Training calls this twice (prior branch and posterior branch); inference
     calls it once with ``Zprior``.
  3. ``mode="full"`` (default): both stages in one call. When no explicit
     ``future_latent`` is passed, the posterior latent (if future frames are
     present) is injected. This mode is what ``lazy_init`` traces, so it
     creates every parameter.

Input layout convention: clips are ``[B, T, H, W, C]`` with frames ordered
``[oldest_past, ..., current, future_1, ..., future_F]``. The current frame
is selected by ``current_frame_index`` (e.g. ``num_past_frames`` when future
frames are appended; ``-1`` keeps the old "last frame is current" behavior
for clips without future frames). Frames strictly before the current index
feed the past path; frames strictly after it feed the future path.

Weight-compatibility notes (for the checkpoint weight loader):

- The current-frame ViT path keeps the original parameter names
  (``LayerNorm_0``, ``MultiHeadDotProductAttention_0``, ``LayerNorm_1``,
  ``MlpBlock_0``, ``embedding``, ``pos_embedding``, ``encoder_norm``) so
  pretrained SigLIP weights load unchanged.
- The history branch keeps the ``siglip_mem_compress`` names
  (``HistoryLayerNorm_0``, ``HistoryMultiHeadDotProductAttention_0``,
  ``HistoryOutProj``, ``history_memory_gate_logit``), so a trained
  Pi0MemCompress checkpoint maps 1:1 onto the PF history branch.
- Inside the UTR, the past queries / current-condition / core-layer /
  ``out_ln`` parameter names match ``HistoryResampler`` exactly
  (``memory_queries``, ``current_condition_ln``, ``current_condition_proj``,
  ``query_ln_{l}``, ``history_ln_{l}``, ``CrossAttention_{l}``,
  ``MlpLayerNorm_{l}``, ``MlpBlock_{l}``, ``out_ln``); only the enclosing
  scope changes from ``HistoryResampler_0`` to ``UTR_0`` (queries et al.) and
  ``UTR_0/ResamplerCore_0`` (core layers). The new direction-specific pieces
  (``past_role_embedding``, ``past_out_proj``, ``future_*``) are zero /
  identity initialized, so a PF model initialized from a compress checkpoint
  reproduces the compress history path bit-for-bit at step 0.
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def posemb_sincos_1d(length, width, temperature=10_000.0, dtype=jnp.float32):
    """Fixed 1D sine-cosine temporal positional embedding."""
    assert width % 2 == 0, "Width must be even for 1D sincos posemb"
    positions = jnp.arange(length)
    omega = jnp.arange(width // 2) / max(width // 2 - 1, 1)
    omega = 1.0 / (temperature**omega)
    sinusoid = jnp.einsum("m,d->md", positions, omega)
    pe = jnp.concatenate([jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


def _resolve_current_frame_index(current_frame_index: int, t: int) -> int:
    """Resolve a possibly-negative current-frame index against clip length T."""
    cur_idx = current_frame_index if current_frame_index >= 0 else t + current_frame_index
    if cur_idx < 0 or cur_idx >= t:
        raise ValueError(f"current_frame_index={current_frame_index} is out of range for T={t}")
    return cur_idx


def _identity_kernel_init(_key, shape, dtype=jnp.float32):
    """Identity initializer for square Dense kernels (direction out-projections).

    Keeps the UTR output projections a no-op at initialization so the past
    path matches a loaded Pi0MemCompress checkpoint exactly at step 0.
    """
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"identity init requires a square 2D kernel, got shape={shape}")
    return jnp.eye(shape[0], dtype=dtype)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class ResamplerCore(nn.Module):
    """Shared temporal compression core of the UTR.

    A stack of ``depth`` blocks, each: cross-attention (queries read the
    flattened temporal visual tokens) + FFN, with residual connections and
    pre-LayerNorms. Both the past path and the future path call the *same*
    instance of this module, so the parameters are shared across directions
    ("参数共享、表示分离" in the paper).

    The per-layer parameter names intentionally match ``HistoryResampler``
    from :mod:`openpi.models.siglip_mem_compress` (``query_ln_{l}``,
    ``history_ln_{l}``, ``CrossAttention_{l}``, ``MlpLayerNorm_{l}``,
    ``MlpBlock_{l}``) so compress checkpoints remap directly.
    """

    num_heads: int = 12
    depth: int = 1
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, q, x_src, deterministic=True):  # noqa: FBT002
        """Args:
        q:     [B, M, D] direction-specific query tokens.
        x_src: [B, S, D] flattened temporal visual tokens for one direction.

        Returns:
            [B, M, D] compressed latent tokens.
        """
        for lyr in range(self.depth):
            q_norm = nn.LayerNorm(name=f"query_ln_{lyr}", dtype=self.dtype_mm)(q)
            src_norm = nn.LayerNorm(name=f"history_ln_{lyr}", dtype=self.dtype_mm)(x_src)

            y = nn.MultiHeadDotProductAttention(
                name=f"CrossAttention_{lyr}",
                num_heads=self.num_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                deterministic=deterministic,
                dtype=self.dtype_mm,
            )(q_norm, src_norm)
            y = nn.Dropout(rate=self.dropout)(y, deterministic)
            q = q + y

            y = nn.LayerNorm(name=f"MlpLayerNorm_{lyr}", dtype=self.dtype_mm)(q)
            y = MlpBlock(
                name=f"MlpBlock_{lyr}",
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
            )(y, deterministic)
            y = nn.Dropout(rate=self.dropout)(y, deterministic)
            q = q + y
            q = sharding.activation_sharding_constraint(q)

        return q


class UnifiedTemporalResampler(nn.Module):
    """Unified Past-Future Temporal Resampler (UTR).

    Compresses past visual tokens into ``num_past_tokens`` history memory
    tokens ``Hmem`` and future visual tokens into ``num_future_tokens``
    posterior latent tokens ``Zpost``, using a *shared* :class:`ResamplerCore`
    but direction-specific queries, role embeddings and output projections:

        Y0_past = Q_past + E_past          Y0_fut = Q_fut + E_fut
        Hmem  = Proj_past(Core(Y0_past, X_hist))
        Zpost = Proj_fut (Core(Y0_fut,  X_fut))

    Temporal position information is expected to be added to the inputs
    *before* this module (the encoder adds a shared 1D sincos temporal PE over
    the whole clip, so past and future frames carry distinct positions).

    Both outputs go through the same anti-collapse post-processing as the
    original ``HistoryResampler``: subtract the cross-token mean, LayerNorm,
    subtract the mean again.

    Args:
        num_past_tokens: number of compressed history tokens M_p.
        num_future_tokens: number of posterior latent tokens M_z.
        num_heads: attention heads for the shared core.
        depth: number of cross-attention + MLP layers in the shared core.
        mlp_dim: FFN hidden dimension. Defaults to 4 * D.
        dropout: dropout rate.
        dtype_mm: matmul dtype.
        use_current_condition: condition the past queries on the pooled
            current-frame representation (kept from ``HistoryResampler``).
        future_use_current_condition: condition the future queries on the
            pooled current-frame representation as well. This makes ``Zpost``
            a CVAE-style posterior ``q(z | current, future)`` that encodes
            "what changes relative to now" instead of absolute future content,
            which puts it on a manifold the prior encoder (conditioned on
            current/history/language/state) can actually reach. The projection
            is zero-init, so enabling the flag is a no-op at step 0. Disabled
            by default to keep the paper-faithful pure-future posterior.
    """

    num_past_tokens: int = 256
    num_future_tokens: int = 64
    num_heads: int = 12
    depth: int = 1
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"
    use_current_condition: bool = True
    future_use_current_condition: bool = False

    @nn.compact
    def __call__(self, x_hist, x_fut, x_cur, deterministic=True):  # noqa: FBT002
        """Args:
        x_hist: [B, T_h, N, D] past frame tokens. T_h may be 0.
        x_fut:  [B, T_f, N, D] future frame tokens. T_f may be 0.
        x_cur:  [B, N, D] current frame tokens (for past-query conditioning).

        Returns:
            ``(hist_mem, future_post)`` with shapes ``[B, M_p, D]`` and
            ``[B, M_z, D]``. Whenever a direction has no input frames (or its
            token budget is 0) the corresponding output is a shape-stable
            zeros tensor, so downstream code never has to special-case.
        """
        b, n, d = x_cur.shape

        core = ResamplerCore(
            name="ResamplerCore_0",
            num_heads=self.num_heads,
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )

        # --- Past path -> Hmem -------------------------------------------------
        if x_hist.shape[1] > 0 and self.num_past_tokens > 0:
            th = x_hist.shape[1]
            hist = x_hist.reshape(b, th * n, d)
            hist = sharding.activation_sharding_constraint(hist)

            # Same name as HistoryResampler so compress checkpoints remap 1:1.
            past_queries = self.param(
                "memory_queries",
                nn.initializers.normal(stddev=0.02),
                (1, self.num_past_tokens, d),
                x_hist.dtype,
            )
            # Zero-init role embedding: a no-op at initialization (keeps parity
            # with compress checkpoints) that lets the shared core tell the two
            # directions apart as training progresses.
            past_role = self.param(
                "past_role_embedding",
                nn.initializers.zeros,
                (1, 1, d),
                x_hist.dtype,
            )
            q = jnp.tile(past_queries + past_role, (b, 1, 1))

            if self.use_current_condition:
                cur_pool = jnp.mean(x_cur, axis=1)
                cur_pool = nn.LayerNorm(name="current_condition_ln", dtype=self.dtype_mm)(cur_pool)
                cur_bias = nn.Dense(
                    d,
                    name="current_condition_proj",
                    dtype=self.dtype_mm,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                )(cur_pool)
                q = q + cur_bias[:, None, :]

            q = core(q, hist, deterministic)

            # Direction-specific output projection (identity-init -> no-op at
            # step 0, so the past path matches HistoryResampler exactly).
            q = nn.Dense(
                d,
                name="past_out_proj",
                dtype=self.dtype_mm,
                kernel_init=_identity_kernel_init,
                bias_init=nn.initializers.zeros,
            )(q)

            # Anti-collapse post-processing, identical to HistoryResampler.
            q = q - jnp.mean(q, axis=1, keepdims=True)
            q = nn.LayerNorm(name="out_ln", dtype=self.dtype_mm)(q)
            q = q - jnp.mean(q, axis=1, keepdims=True)
            hist_mem = q
        else:
            hist_mem = jnp.zeros((b, self.num_past_tokens, d), dtype=x_cur.dtype)

        # --- Future path -> Zpost (train-only posterior latent) ----------------
        if x_fut.shape[1] > 0 and self.num_future_tokens > 0:
            tf = x_fut.shape[1]
            fut = x_fut.reshape(b, tf * n, d)
            fut = sharding.activation_sharding_constraint(fut)

            future_queries = self.param(
                "future_queries",
                nn.initializers.normal(stddev=0.02),
                (1, self.num_future_tokens, d),
                x_fut.dtype,
            )
            future_role = self.param(
                "future_role_embedding",
                nn.initializers.zeros,
                (1, 1, d),
                x_fut.dtype,
            )
            q = jnp.tile(future_queries + future_role, (b, 1, 1))

            if self.future_use_current_condition:
                # Same zero-init recipe as the past path: at step 0 the bias
                # is exactly zero, so Zpost stays a pure function of the
                # future frames until training moves the projection.
                fut_cur_pool = jnp.mean(x_cur, axis=1)
                fut_cur_pool = nn.LayerNorm(name="future_current_condition_ln", dtype=self.dtype_mm)(fut_cur_pool)
                fut_cur_bias = nn.Dense(
                    d,
                    name="future_current_condition_proj",
                    dtype=self.dtype_mm,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                )(fut_cur_pool)
                q = q + fut_cur_bias[:, None, :]

            q = core(q, fut, deterministic)

            q = nn.Dense(
                d,
                name="future_out_proj",
                dtype=self.dtype_mm,
                kernel_init=_identity_kernel_init,
                bias_init=nn.initializers.zeros,
            )(q)

            q = q - jnp.mean(q, axis=1, keepdims=True)
            q = nn.LayerNorm(name="future_out_ln", dtype=self.dtype_mm)(q)
            q = q - jnp.mean(q, axis=1, keepdims=True)
            future_post = q
        else:
            future_post = jnp.zeros((b, self.num_future_tokens, d), dtype=x_cur.dtype)

        return hist_mem, future_post


class Encoder1DBlockCurrentOnlyMemoryPF(nn.Module):
    """ViT block with dual gated temporal cross-attention (GTCA).

    The normal spatial self-attention and MLP run on ``x_cur: [B, N, D]``.
    On scheduled memory layers, two extra gated cross-attention branches let
    the current tokens read the compressed history ``hist_mem: [B, M_p, D]``
    and the branch-specific future latent ``fut_mem: [B, M_z, D]``:

        y = y_spatial + g_mem * mem_update + g_fut * fut_update

    The history branch is byte-for-byte the compress one (same parameter
    names); the future branch mirrors it with independent gate, LayerNorm,
    cross-attention and a near-zero-init output projection.
    """

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    # Logits. sigmoid(-6.9) ~= 1e-3, so both temporal branches start tiny but
    # their parameters still receive gradients from the first step.
    history_gate_init: float = -6.9
    history_gate_fixed: float | None = None
    future_gate_init: float = -6.9
    future_gate_fixed: float | None = None

    @nn.compact
    def __call__(self, x_cur, hist_mem, fut_mem, use_memory, use_future, deterministic=True):  # noqa: FBT002
        out = {}
        x_cur = sharding.activation_sharding_constraint(x_cur)

        ln1 = nn.LayerNorm(name="LayerNorm_0", dtype=self.dtype_mm)
        attn = nn.MultiHeadDotProductAttention(
            name="MultiHeadDotProductAttention_0",
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )
        history_ln = nn.LayerNorm(name="HistoryLayerNorm_0", dtype=self.dtype_mm)
        history_attn = nn.MultiHeadDotProductAttention(
            name="HistoryMultiHeadDotProductAttention_0",
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )
        future_ln = nn.LayerNorm(name="FutureLayerNorm_0", dtype=self.dtype_mm)
        future_attn = nn.MultiHeadDotProductAttention(
            name="FutureMultiHeadDotProductAttention_0",
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )
        y_cur = ln1(x_cur)

        # Baseline experiment (kept for easy restoration):
        # history_out_proj = nn.Dense(
        #     y_cur.shape[-1],
        #     name="HistoryOutProj",
        #     kernel_init=nn.initializers.normal(stddev=1e-4),
        #     bias_init=nn.initializers.zeros,
        #     dtype=self.dtype_mm,
        # )
        # future_out_proj = nn.Dense(
        #     y_cur.shape[-1],
        #     name="FutureOutProj",
        #     kernel_init=nn.initializers.normal(stddev=1e-4),
        #     bias_init=nn.initializers.zeros,
        #     dtype=self.dtype_mm,
        # )

        y_spatial = out["sa"] = attn(y_cur, y_cur)
        out["y_spatial"] = y_spatial

        hist_gate_logit = self.param(
            "history_memory_gate_logit",
            nn.initializers.constant(self.history_gate_init),
            (),
            y_spatial.dtype,
        )
        if self.history_gate_fixed is None:
            hist_gate = nn.sigmoid(hist_gate_logit)
        else:
            if not 0.0 <= self.history_gate_fixed <= 1.0:
                raise ValueError(f"history_gate_fixed must be in [0, 1], got {self.history_gate_fixed}")
            hist_gate = jnp.asarray(self.history_gate_fixed, dtype=y_spatial.dtype)

        fut_gate_logit = self.param(
            "future_memory_gate_logit",
            nn.initializers.constant(self.future_gate_init),
            (),
            y_spatial.dtype,
        )
        if self.future_gate_fixed is None:
            fut_gate = nn.sigmoid(fut_gate_logit)
        else:
            if not 0.0 <= self.future_gate_fixed <= 1.0:
                raise ValueError(f"future_gate_fixed must be in [0, 1], got {self.future_gate_fixed}")
            fut_gate = jnp.asarray(self.future_gate_fixed, dtype=y_spatial.dtype)

        zero_scalar = jnp.asarray(0.0, dtype=y_spatial.dtype)

        # --- History branch (identical semantics to the compress block) -------
        # Avoid tracing attention with an empty K/V length when M == 0.
        if hist_mem.shape[1] == 0:
            mem_contrib = jnp.zeros_like(y_spatial)
            mem_update = jnp.zeros_like(y_spatial)
            hist_gate_value = zero_scalar
        else:
            # Cross-attend only to compressed history. Keep Flax module calls
            # outside lax.cond so parameter creation cannot leak tracers.
            y_mem = history_ln(hist_mem)
            mem_update = history_attn(y_cur, y_mem)
            # Baseline experiment:
            # mem_update = history_out_proj(mem_update)
            # Fixed-scale identity ablation: remove the trainable near-zero
            # projection while keeping history injection conservative.
            mem_update = jnp.asarray(1.0, dtype=mem_update.dtype) * mem_update

            def memory_branch(_):
                return hist_gate * mem_update, mem_update, hist_gate

            def no_memory_branch(_):
                return jnp.zeros_like(y_spatial), jnp.zeros_like(y_spatial), zero_scalar

            mem_contrib, mem_update, hist_gate_value = jax.lax.cond(
                jnp.asarray(use_memory, dtype=jnp.bool_),
                memory_branch,
                no_memory_branch,
                operand=None,
            )

        # --- Future branch (independent gate, same recipe) --------------------
        if fut_mem.shape[1] == 0:
            fut_contrib = jnp.zeros_like(y_spatial)
            fut_update = jnp.zeros_like(y_spatial)
            fut_gate_value = zero_scalar
        else:
            y_fut = future_ln(fut_mem)
            fut_update = future_attn(y_cur, y_fut)
            # Baseline experiment:
            # fut_update = future_out_proj(fut_update)
            # Match the history ablation with the same fixed identity scale.
            fut_update = jnp.asarray(1.0, dtype=fut_update.dtype) * fut_update

            def future_branch(_):
                return fut_gate * fut_update, fut_update, fut_gate

            def no_future_branch(_):
                return jnp.zeros_like(y_spatial), jnp.zeros_like(y_spatial), zero_scalar

            fut_contrib, fut_update, fut_gate_value = jax.lax.cond(
                jnp.asarray(use_future, dtype=jnp.bool_),
                future_branch,
                no_future_branch,
                operand=None,
            )

        y = y_spatial + mem_contrib + fut_contrib
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x_cur = out["+sa"] = x_cur + y
        out["mem_update"] = mem_update
        out["history_gate"] = hist_gate_value
        out["fut_update"] = fut_update
        out["future_gate"] = fut_gate_value

        y = nn.LayerNorm(name="LayerNorm_1", dtype=self.dtype_mm)(x_cur)
        y = out["mlp"] = MlpBlock(
            name="MlpBlock_0",
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x_cur = out["+mlp"] = x_cur + y
        x_cur = sharding.activation_sharding_constraint(x_cur)
        return x_cur, out


class EncoderCurrentOnlyPF(nn.Module):
    """Current-frame encoder with a unified past-future temporal bottleneck.

    Modes (see the module docstring for the full contract):

    - ``memories``: ``x`` is [B, T, N, D]; slice past / current / future, add
      the shared temporal PE, run the UTR, and return
      ``(x_cur, {"history_mem", "future_post"})`` *without* the ViT blocks.
    - ``current``: ``x`` is [B, N, D]; run the ViT blocks with the provided
      ``hist_mem`` / ``future_latent`` and return ``([B, 1, N, D], out)``.
    - ``full``: both stages. ``future_latent`` (if given) overrides the
      internally computed posterior latent as the injected ``Zb``.
    """

    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    memory_every: int = 4
    current_frame_index: int = -1
    history_memory_tokens: int = 256
    history_resampler_depth: int = 1
    history_use_current_condition: bool = True
    history_gate_init: float = -6.9
    history_gate_fixed: float | None = None
    future_latent_tokens: int = 64
    future_use_current_condition: bool = False
    future_gate_init: float = -6.9
    future_gate_fixed: float | None = None

    @nn.compact
    def __call__(  # noqa: FBT002
        self,
        x,
        hist_mem=None,
        future_latent=None,
        deterministic=True,
        mode="full",
        use_memory=True,
        use_future=True,
    ):
        if mode not in ("memories", "current", "full"):
            raise ValueError(f"Unknown encoder mode: {mode!r}")
        out = {}

        memory_active = bool(use_memory)
        future_active = bool(use_future)

        if mode in ("memories", "full"):
            if x.ndim == 3:
                x = x[:, None, ...]
            if x.ndim != 4:
                raise ValueError(f"Expected x rank 3 or 4, got shape={x.shape}")

            b, t, _, d = x.shape
            cur_idx = _resolve_current_frame_index(self.current_frame_index, t)
            x_cur = x[:, cur_idx, :, :]

            past_available = (cur_idx > 0) and (self.history_memory_tokens > 0)
            future_available = (cur_idx < t - 1) and (self.future_latent_tokens > 0)

            if past_available or future_available:
                # Shared temporal PE over the whole clip (past + current +
                # future) before slicing; otherwise the resampler cannot
                # distinguish old from recent (or near-future from far-future)
                # observations. Direction identity itself comes from the
                # per-direction queries / role embeddings inside the UTR.
                tpe = posemb_sincos_1d(t, d, dtype=x.dtype)[:, :, None, :]
                x_with_tpe = x + tpe
                x_hist_for_mem = x_with_tpe[:, :cur_idx, :, :]
                x_fut_for_mem = x_with_tpe[:, cur_idx + 1 :, :, :]
                x_cur_for_cond = x_with_tpe[:, cur_idx, :, :]

                computed_hist_mem, future_post = UnifiedTemporalResampler(
                    name="UTR_0",
                    num_past_tokens=self.history_memory_tokens,
                    num_future_tokens=self.future_latent_tokens,
                    num_heads=self.num_heads,
                    depth=self.history_resampler_depth,
                    mlp_dim=self.mlp_dim,
                    dropout=self.dropout,
                    dtype_mm=self.dtype_mm,
                    use_current_condition=self.history_use_current_condition,
                    future_use_current_condition=self.future_use_current_condition,
                )(x_hist_for_mem, x_fut_for_mem, x_cur_for_cond, deterministic)
            else:
                computed_hist_mem = jnp.zeros((b, self.history_memory_tokens, d), dtype=x.dtype)
                future_post = jnp.zeros((b, self.future_latent_tokens, d), dtype=x.dtype)

            out["history_mem"] = computed_hist_mem
            out["future_post"] = future_post

            if mode == "memories":
                return x_cur, out

            # mode == "full": resolve the injected memories from this stage.
            hist_mem = computed_hist_mem
            explicit_future_latent = future_latent is not None
            if future_latent is None:
                # Default to the posterior latent (teacher path). Callers that
                # want the prior branch pass future_latent=Zprior explicitly.
                future_latent = future_post if future_available else None
            memory_active = memory_active and past_available
            # An explicitly injected latent (e.g. Zprior) keeps the future
            # branch active even when the clip itself carries no future frames
            # (``future_available`` only gates the internally computed Zpost).
            future_active = future_active and (future_available or explicit_future_latent)
        else:
            # mode == "current": x is the already-embedded current frame.
            if x.ndim != 3:
                raise ValueError(f"Expected x rank 3 in 'current' mode, got shape={x.shape}")
            x_cur = x
            b, _, d = x_cur.shape
            if hist_mem is None:
                hist_mem = jnp.zeros((b, self.history_memory_tokens, d), dtype=x_cur.dtype)
                memory_active = False

        hist_mem = jnp.asarray(hist_mem, dtype=x_cur.dtype)
        if future_latent is None:
            future_latent = jnp.zeros((b, self.future_latent_tokens, d), dtype=x_cur.dtype)
            future_active = False
        fut_mem = jnp.asarray(future_latent, dtype=x_cur.dtype)

        memory_active = memory_active and (self.memory_every > 0) and (hist_mem.shape[1] > 0)
        future_active = future_active and (self.memory_every > 0) and (fut_mem.shape[1] > 0)

        if self.scan:
            memory_flags = (
                (jnp.arange(self.depth) + 1) % self.memory_every == 0
                if memory_active
                else jnp.zeros((self.depth,), dtype=jnp.bool_)
            )
            future_flags = (
                (jnp.arange(self.depth) + 1) % self.memory_every == 0
                if future_active
                else jnp.zeros((self.depth,), dtype=jnp.bool_)
            )

            block = nn.remat(
                Encoder1DBlockCurrentOnlyMemoryPF,
                prevent_cse=False,
                static_argnums=(6,),
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            ScanBlock = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(nn.broadcast, nn.broadcast, 0, 0, nn.broadcast),
                length=self.depth,
            )
            x_cur, scan_out = ScanBlock(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                history_gate_init=self.history_gate_init,
                history_gate_fixed=self.history_gate_fixed,
                future_gate_init=self.future_gate_init,
                future_gate_fixed=self.future_gate_fixed,
            )(x_cur, hist_mem, fut_mem, memory_flags, future_flags, deterministic)

            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            for lyr in range(self.depth):
                block_cur = Encoder1DBlockCurrentOnlyMemoryPF(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    history_gate_init=self.history_gate_init,
                    history_gate_fixed=self.history_gate_fixed,
                    future_gate_init=self.future_gate_init,
                    future_gate_fixed=self.future_gate_fixed,
                )
                on_schedule = (lyr + 1) % self.memory_every == 0 if self.memory_every > 0 else False
                use_memory_l = memory_active and on_schedule
                use_future_l = future_active and on_schedule
                x_cur, out[f"block{lyr:02d}"] = block_cur(
                    x_cur, hist_mem, fut_mem, use_memory_l, use_future_l, deterministic
                )

        out["pre_ln"] = x_cur
        x_cur = nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x_cur)
        return x_cur[:, None, :, :], out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype_mm=self.dtype_mm)(y)
        return x[:, 0]


class _Module(nn.Module):
    """SigLIP/ViT module with the unified past-future temporal bottleneck.

    Call contract (all modes return ``(tokens, out)``):

    - ``mode="full"`` (default, used by ``lazy_init``): full pipeline. Input
      ``image`` is ``[B, T, H, W, C]`` (or a single ``[B, H, W, C]`` image).
      Optional ``future_latent`` overrides the injected ``Zb``; without it the
      internally computed posterior latent is injected when future frames are
      present. Returns the head-projected current-frame tokens like the
      compress module.
    - ``mode="memories"``: stage A. Returns
      ``(x_cur_embedded [B, N, width], out)`` with
      ``out["encoder"]["history_mem"]`` (= Hmem) and
      ``out["encoder"]["future_post"]`` (= Zpost). No ViT blocks are run;
      ``x_cur_embedded`` is the patch-embedded current frame (with 2D posemb)
      for conditioning the model-level Future Latent Prior Encoder.
    - ``mode="current"``: stage B. Re-embeds only the current frame from
      ``image`` (cheap conv) and runs the ViT blocks with the provided
      ``hist_mem`` and branch-specific ``future_latent`` (Zprior or Zpost).
      ``use_memory`` / ``use_future`` let callers statically disable a branch.
    """

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None
    num_heads: int = 12
    posemb: str = "learn"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"  # "gap", "map", "tok", "0", "none"
    head_zeroinit: bool = True
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    memory_every: int = 4
    current_frame_index: int = -1
    history_memory_tokens: int = 256
    history_resampler_depth: int = 1
    history_use_current_condition: bool = True
    history_gate_init: float = -6.9
    history_gate_fixed: float | None = None
    future_latent_tokens: int = 64
    future_use_current_condition: bool = False
    future_gate_init: float = -6.9
    future_gate_fixed: float | None = None

    @nn.compact
    def __call__(  # noqa: FBT002
        self,
        image,
        hist_mem=None,
        future_latent=None,
        *,
        train=False,
        mode="full",
        use_memory=True,
        use_future=True,
    ):
        if mode not in ("memories", "current", "full"):
            raise ValueError(f"Unknown module mode: {mode!r}")
        out = {}

        image = jnp.asarray(image, jnp.float32)
        if image.ndim == 4:
            image = image[:, None, ...]
        elif image.ndim != 5:
            raise ValueError(f"Expected image rank 4 or 5, got shape={image.shape}")

        b, t, h_in, w_in, c_in = image.shape
        if mode == "current":
            # Stage B only needs the current frame; slice before the conv so
            # patch embedding is not recomputed for past/future frames.
            cur_idx = _resolve_current_frame_index(self.current_frame_index, t) if t > 1 else 0
            image = image[:, cur_idx : cur_idx + 1, :, :, :]
            t = 1
        image_bt = image.reshape(b * t, h_in, w_in, c_in)

        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image_bt)

        _, h, w, c = x.shape
        num_patches = h * w
        x = x.reshape(b, t, num_patches, c)

        pe2d = get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)
        x = out["with_posemb"] = x + pe2d[:, None, :, :]

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, 1, c), x.dtype)
            cls = jnp.tile(cls, [b, t, 1, 1])
            x = jnp.concatenate([cls, x], axis=2)

        x = nn.Dropout(rate=self.dropout)(x, not train)
        x = x.astype(self.dtype_mm)

        encoder = EncoderCurrentOnlyPF(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            memory_every=self.memory_every,
            current_frame_index=self.current_frame_index,
            history_memory_tokens=self.history_memory_tokens,
            history_resampler_depth=self.history_resampler_depth,
            history_use_current_condition=self.history_use_current_condition,
            history_gate_init=self.history_gate_init,
            history_gate_fixed=self.history_gate_fixed,
            future_latent_tokens=self.future_latent_tokens,
            future_use_current_condition=self.future_use_current_condition,
            future_gate_init=self.future_gate_init,
            future_gate_fixed=self.future_gate_fixed,
            name="Transformer",
        )

        if mode == "memories":
            x_cur, out["encoder"] = encoder(x, deterministic=not train, mode="memories")
            # Pre-block current-frame tokens (width-dim), used by the
            # model-level Future Latent Prior Encoder for conditioning.
            out["x_cur"] = x_cur
            return x_cur, out

        if mode == "current":
            x, out["encoder"] = encoder(
                x[:, 0, :, :],
                hist_mem=hist_mem,
                future_latent=future_latent,
                deterministic=not train,
                mode="current",
                use_memory=use_memory,
                use_future=use_future,
            )
        else:  # mode == "full"
            x, out["encoder"] = encoder(
                x,
                future_latent=future_latent,
                deterministic=not train,
                mode="full",
                use_memory=use_memory,
                use_future=use_future,
            )

        out["encoded_video"] = x

        # The efficient encoder always returns one current frame [B, 1, N, D].
        current_tokens = x[:, 0, :, :]
        encoded_current = current_tokens
        out["encoded"] = encoded_current

        if self.pool_type == "map":
            x_out = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype_mm=self.dtype_mm,
            )(current_tokens)
        elif self.pool_type == "gap":
            x_out = out["head_input"] = jnp.mean(current_tokens, axis=1)
        elif self.pool_type == "0":
            x_out = out["head_input"] = current_tokens[:, 0]
        elif self.pool_type == "tok":
            x_out = out["head_input"] = current_tokens[:, 0]
            encoded_current = encoded_current[:, 1:]
        elif self.pool_type == "none":
            x_out = current_tokens
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type!r}")

        x_2d = jnp.reshape(encoded_current, [b, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            x_2d = nn.tanh(hid(x_2d))
            x_out = nn.tanh(hid(x_out))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x_out

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x_out = out["logits"] = head(x_out)

        return x_out, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function."""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        **patch,
    }
