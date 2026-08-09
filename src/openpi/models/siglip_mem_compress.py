"""SigLIP current-frame encoder with learned compressed visual history.

This is a cleaned version of the MEM-style video encoder that keeps only the
compute-efficient path:

- Inputs may be a single image [B, H, W, C] or a clip [B, T, H, W, C].
- Patch embedding is still applied to every frame so history can be observed.
- Historical frame tokens are compressed once into M memory tokens.
- The Transformer blocks carry only current-frame tokens [B, N, D].
- Every ``memory_every`` layers, current-frame tokens read compressed history
  through a gated cross-attention branch.
- The output interface remains the current-frame representation, matching the
  original SigLIP/Pi visual-token usage.

Removed from the previous mixed file:
- The old all-frames MEM carry [B, T, N, D].
- The patch-aligned causal temporal-attention branch.
- ``encoder_mode`` / ``temporal_mode`` switches.

The current-frame path keeps the original ViT block parameter names
(``LayerNorm_0``, ``MultiHeadDotProductAttention_0``, ``LayerNorm_1``,
``MlpBlock_0``) so pretrained SigLIP/ViT weights can still be reused for the
main image path.
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


class HistoryResampler(nn.Module):
    """Compress historical visual tokens into fixed-size memory tokens.

    Args:
        num_memory_tokens: Number of compressed history tokens M.
        num_heads: Attention heads for cross-attention.
        depth: Number of cross-attention + MLP refinement layers.
        mlp_dim: FFN hidden dimension. Defaults to 4 * D.
        dropout: Dropout rate.
        dtype_mm: Matmul dtype.
        use_current_condition: Condition learned memory queries on the current
            frame pooled representation.
    """

    num_memory_tokens: int = 256
    num_heads: int = 12
    depth: int = 1
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"
    use_current_condition: bool = True

    @nn.compact
    def __call__(self, x_hist, x_cur, deterministic=True):  # noqa: FBT002
        """Args:
        x_hist: [B, T_h, N, D] historical frame tokens. T_h must be > 0.
        x_cur:  [B, N, D] current frame tokens.

        Returns:
            [B, M, D] compressed history memory.
        """
        b, th, n, d = x_hist.shape
        hist = x_hist.reshape(b, th * n, d)
        hist = sharding.activation_sharding_constraint(hist)

        queries = self.param(
            "memory_queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, d),
            x_hist.dtype,
        )
        q = jnp.tile(queries, (b, 1, 1))

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

        for lyr in range(self.depth):
            q_norm = nn.LayerNorm(name=f"query_ln_{lyr}", dtype=self.dtype_mm)(q)
            h_norm = nn.LayerNorm(name=f"history_ln_{lyr}", dtype=self.dtype_mm)(hist)

            y = nn.MultiHeadDotProductAttention(
                name=f"CrossAttention_{lyr}",
                num_heads=self.num_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                deterministic=deterministic,
                dtype=self.dtype_mm,
            )(q_norm, h_norm)
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

        # Remove the common component shared by all memory slots.
        q = q - jnp.mean(q, axis=1, keepdims=True)

        # Normalize each memory token to keep the scale stable.
        q = nn.LayerNorm(name="out_ln", dtype=self.dtype_mm)(q)

        # Optional but recommended: remove slot-wise common component again,
        # because LayerNorm is applied per token and may reintroduce a small common direction.
        q = q - jnp.mean(q, axis=1, keepdims=True)

        return q


class Encoder1DBlockCurrentOnlyMemory(nn.Module):
    """ViT block that updates current-frame tokens only.

    The normal spatial self-attention and MLP run on ``x_cur: [B, N, D]``.
    On scheduled memory layers, an extra gated cross-attention branch lets the
    current tokens read ``hist_mem: [B, M, D]``.
    """

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    # This is a logit. sigmoid(-6.9) ~= 1e-3, so the history branch starts tiny
    # but the resampler still receives gradients from the first step.
    history_gate_init: float = -6.9
    # If set, use this fixed gate probability instead of sigmoid(history_memory_gate_logit).
    # The parameter is still created so checkpoint structure stays stable, but it
    # is not used by the forward pass in fixed-gate experiments.
    history_gate_fixed: float | None = None

    @nn.compact
    def __call__(self, x_cur, hist_mem, use_memory, deterministic=True):  # noqa: FBT002
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
        y_cur = ln1(x_cur)

        # history_out_proj = nn.Dense(
        #     y_cur.shape[-1],
        #     name="HistoryOutProj",
        #     kernel_init=nn.initializers.normal(stddev=1e-4),
        #     bias_init=nn.initializers.zeros,
        #     dtype=self.dtype_mm,
        # )

        y_spatial = out["sa"] = attn(y_cur, y_cur)
        out["y_spatial"] = y_spatial

        gate_logit = self.param(
            "history_memory_gate_logit",
            nn.initializers.constant(self.history_gate_init),
            (),
            y_spatial.dtype,
        )
        if self.history_gate_fixed is None:
            gate = nn.sigmoid(gate_logit)
        else:
            if not 0.0 <= self.history_gate_fixed <= 1.0:
                raise ValueError(f"history_gate_fixed must be in [0, 1], got {self.history_gate_fixed}")
            gate = jnp.asarray(self.history_gate_fixed, dtype=y_spatial.dtype)

        # Avoid tracing attention with an empty K/V length when M == 0.
        if hist_mem.shape[1] == 0:
            mem_update = jnp.zeros_like(y_spatial)
            y = y_spatial
            gate_value = jnp.asarray(0.0, dtype=y_spatial.dtype)
        else:
            # Cross-attend only to compressed history. Keep Flax module calls
            # outside lax.cond so parameter creation cannot leak tracers.
            y_mem = history_ln(hist_mem)
            mem_update = history_attn(y_cur, y_mem)

            # Zero-init adapter: start as no-op, then learn useful residual.
            # mem_update = history_out_proj(mem_update)

            def memory_branch(_):
                y = y_spatial + gate * mem_update
                return y, mem_update, gate

            def spatial_branch(_):
                return y_spatial, jnp.zeros_like(y_spatial), jnp.asarray(0.0, dtype=y_spatial.dtype)

            y, mem_update, gate_value = jax.lax.cond(
                jnp.asarray(use_memory, dtype=jnp.bool_),
                memory_branch,
                spatial_branch,
                operand=None,
            )

        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x_cur = out["+sa"] = x_cur + y
        out["mem_update"] = mem_update
        out["history_gate"] = gate_value

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


class EncoderCurrentOnlyCompressed(nn.Module):
    """Efficient current-frame encoder with compressed causal history.

    Input ``x`` is [B, T, N, D]. The encoder returns [B, 1, N, D].
    History is compressed once into [B, M, D], then every Transformer block
    carries only current-frame tokens [B, N, D].
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

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        if x.ndim == 3:
            x = x[:, None, ...]
        if x.ndim != 4:
            raise ValueError(f"Expected x rank 3 or 4, got shape={x.shape}")

        b, t, _, d = x.shape
        cur_idx = self.current_frame_index if self.current_frame_index >= 0 else t + self.current_frame_index
        if cur_idx < 0 or cur_idx >= t:
            raise ValueError(f"current_frame_index={self.current_frame_index} is out of range for T={t}")

        x_cur = x[:, cur_idx, :, :]

        history_available = (t > 1) and (cur_idx > 0) and (self.history_memory_tokens > 0)
        if history_available:
            # Add temporal position before flattening; otherwise the resampler
            # cannot distinguish old from recent observations.
            tpe = posemb_sincos_1d(t, d, dtype=x.dtype)[:, :, None, :]
            x_with_tpe = x + tpe
            x_hist_for_mem = x_with_tpe[:, :cur_idx, :, :]
            x_cur_for_cond = x_with_tpe[:, cur_idx, :, :]

            hist_mem = HistoryResampler(
                name="HistoryResampler_0",
                num_memory_tokens=self.history_memory_tokens,
                num_heads=self.num_heads,
                depth=self.history_resampler_depth,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
                use_current_condition=self.history_use_current_condition,
            )(x_hist_for_mem, x_cur_for_cond, deterministic)
        else:
            hist_mem = jnp.zeros((b, self.history_memory_tokens, d), dtype=x.dtype)

        out["history_mem"] = hist_mem

        memory_active = history_available and self.memory_every > 0
        if self.scan:
            memory_flags = (
                (jnp.arange(self.depth) + 1) % self.memory_every == 0
                if memory_active
                else jnp.zeros((self.depth,), dtype=jnp.bool_)
            )

            block = nn.remat(
                Encoder1DBlockCurrentOnlyMemory,
                prevent_cse=False,
                static_argnums=(4,),
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            ScanBlock = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(nn.broadcast, 0, nn.broadcast),
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
            )(x_cur, hist_mem, memory_flags, deterministic)

            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            for lyr in range(self.depth):
                block_cur = Encoder1DBlockCurrentOnlyMemory(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    history_gate_init=self.history_gate_init,
                    history_gate_fixed=self.history_gate_fixed,
                )
                use_memory = memory_active and ((lyr + 1) % self.memory_every == 0)
                x_cur, out[f"block{lyr:02d}"] = block_cur(x_cur, hist_mem, use_memory, deterministic)

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
    """SigLIP/ViT module with current-frame compressed visual history."""

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

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        image = jnp.asarray(image, jnp.float32)
        if image.ndim == 4:
            image = image[:, None, ...]
        elif image.ndim != 5:
            raise ValueError(f"Expected image rank 4 or 5, got shape={image.shape}")

        b, t, h_in, w_in, c_in = image.shape
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

        x, out["encoder"] = EncoderCurrentOnlyCompressed(
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
            name="Transformer",
        )(x, deterministic=not train)

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
