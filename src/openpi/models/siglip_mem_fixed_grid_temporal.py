"""SigLIP current-frame encoder with topology-preserving temporal history.

Historical frames are kept out of the 27-layer current-frame ViT.  Their
patch embeddings follow a separate, inexpensive path:

    [B, T_h, 16*16, 1152]
      -> fixed non-overlapping 2x2 spatial average pooling
      -> [B, T_h, 8*8, 1152]
      -> projection to width 256
      -> two factorized temporal/spatial Transformer blocks
      -> learned final compression to [B, M, 1152]

The fixed grid gives token k a stable spatial identity across time.  Current
frame tokens then read the resulting memory through the same periodic history
cross-attention branches used by :mod:`siglip_mem_compress`.
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip_mem_compress as _compress


class FactorizedSpaceTimeBlock(nn.Module):
    """Temporal attention per fixed grid cell, then spatial attention per frame."""

    width: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        b, t, n, d = x.shape
        if d != self.width:
            raise ValueError(f"Expected temporal width {self.width}, got {d}")

        y = nn.LayerNorm(name="temporal_ln", dtype=self.dtype_mm)(x)
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(b * n, t, d)
        y = nn.MultiHeadDotProductAttention(
            name="temporal_attn",
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = y.reshape(b, n, t, d).transpose(0, 2, 1, 3)
        x = x + y

        y = nn.LayerNorm(name="spatial_ln", dtype=self.dtype_mm)(x)
        y = y.reshape(b * t, n, d)
        y = nn.MultiHeadDotProductAttention(
            name="spatial_attn",
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = y.reshape(b, t, n, d)
        x = x + y

        y = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(x)
        y = nn.Dense(
            self.width * self.mlp_ratio,
            name="mlp_in",
            dtype=self.dtype_mm,
        )(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)
        y = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(y)
        return x + y


class FinalMemoryCompressor(nn.Module):
    """Compress contextualized fixed-grid tracks to Pi0-width memory tokens."""

    width: int = 256
    output_width: int = 1152
    num_memory_tokens: int = 128
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        if x.ndim != 4:
            raise ValueError(f"Expected [B,T,K,D] history, got {x.shape}")
        b = x.shape[0]
        flat = x.reshape(b, -1, self.width)
        flat = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(flat)

        memory_queries = self.param(
            "memory_queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.width),
            flat.dtype,
        )
        queries = jnp.tile(memory_queries, (b, 1, 1))
        q_norm = nn.LayerNorm(name="query_ln", dtype=self.dtype_mm)(queries)
        update = nn.MultiHeadDotProductAttention(
            name="cross_attention",
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(q_norm, flat)
        memory = queries + update

        y = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(memory)
        y = nn.Dense(
            self.width * self.mlp_ratio,
            name="mlp_in",
            dtype=self.dtype_mm,
        )(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)
        y = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(y)
        memory = memory + y
        memory = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(memory)
        memory = nn.Dense(
            self.output_width,
            name="output_projection",
            dtype=self.dtype_mm,
        )(memory)

        memory = memory - jnp.mean(memory, axis=1, keepdims=True)
        memory = nn.LayerNorm(name="pi0_output_ln", dtype=self.dtype_mm)(memory)
        return memory - jnp.mean(memory, axis=1, keepdims=True)


class FixedGridTemporalHistory(nn.Module):
    """Build history memory without sending historical frames through the ViT."""

    input_width: int = 1152
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2
    num_memory_tokens: int = 128
    output_width: int = 1152
    dropout: float = 0.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens, deterministic=True):  # noqa: FBT002
        if patch_tokens.ndim != 4:
            raise ValueError(f"Expected [B,T,N,D] patch tokens, got {patch_tokens.shape}")
        b, t, n, d = patch_tokens.shape
        if d != self.input_width:
            raise ValueError(f"Expected input width {self.input_width}, got {d}")

        input_grid_size = int(np.sqrt(n))
        if input_grid_size**2 != n:
            raise ValueError(f"Patch count {n} is not a square grid")
        if input_grid_size % self.spatial_pool_factor != 0:
            raise ValueError(f"Grid {input_grid_size} is not divisible by pool factor {self.spatial_pool_factor}")
        output_grid_size = input_grid_size // self.spatial_pool_factor

        x = patch_tokens.reshape(
            b,
            t,
            output_grid_size,
            self.spatial_pool_factor,
            output_grid_size,
            self.spatial_pool_factor,
            d,
        )
        x = jnp.mean(x, axis=(3, 5))
        x = x.reshape(b, t, output_grid_size**2, d)

        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(x)
        x = nn.Dense(
            self.temporal_width,
            name="input_projection",
            dtype=self.dtype_mm,
        )(x)
        temporal_pos = self.param(
            "temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, t, 1, self.temporal_width),
            x.dtype,
        )
        x = x + temporal_pos

        for block_index in range(self.temporal_depth):
            x = FactorizedSpaceTimeBlock(
                name=f"temporal_block_{block_index}",
                width=self.temporal_width,
                num_heads=self.temporal_heads,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
            )(x, deterministic=deterministic)

        return FinalMemoryCompressor(
            name="final_memory_compressor",
            width=self.temporal_width,
            output_width=self.output_width,
            num_memory_tokens=self.num_memory_tokens,
            num_heads=self.temporal_heads,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(x, deterministic=deterministic)


class EncoderCurrentOnlyFixedGridTemporal(nn.Module):
    """Run fixed-grid temporal history once, then carry only current tokens."""

    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    memory_every: int = 4
    current_frame_index: int = -1
    history_memory_tokens: int = 128
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2
    history_gate_init: float = -6.9
    history_gate_fixed: float | None = None

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        if x.ndim == 3:
            x = x[:, None, ...]
        if x.ndim != 4:
            raise ValueError(f"Expected x rank 3 or 4, got {x.shape}")

        b, t, _, d = x.shape
        cur_idx = self.current_frame_index if self.current_frame_index >= 0 else t + self.current_frame_index
        if cur_idx < 0 or cur_idx >= t:
            raise ValueError(f"current_frame_index={self.current_frame_index} is out of range for T={t}")
        x_cur = x[:, cur_idx]

        history_available = (t > 1) and (cur_idx > 0) and (self.history_memory_tokens > 0)
        if history_available:
            hist_mem = FixedGridTemporalHistory(
                name="FixedGridTemporalHistory_0",
                input_width=d,
                temporal_width=self.temporal_width,
                temporal_depth=self.temporal_depth,
                temporal_heads=self.temporal_heads,
                spatial_pool_factor=self.spatial_pool_factor,
                num_memory_tokens=self.history_memory_tokens,
                output_width=d,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
            )(x[:, :cur_idx], deterministic)
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
                _compress.Encoder1DBlockCurrentOnlyMemory,
                prevent_cse=False,
                static_argnums=(4,),
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            scan_block = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(nn.broadcast, 0, nn.broadcast),
                length=self.depth,
            )
            x_cur, scan_out = scan_block(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                history_gate_init=self.history_gate_init,
                history_gate_fixed=self.history_gate_fixed,
            )(x_cur, hist_mem, memory_flags, deterministic)
            for layer in range(self.depth):
                out[f"block{layer:02d}"] = jax.tree.map(lambda value, layer=layer: value[layer], scan_out)
        else:
            for layer in range(self.depth):
                block_cur = _compress.Encoder1DBlockCurrentOnlyMemory(
                    name=f"encoderblock_{layer}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    history_gate_init=self.history_gate_init,
                    history_gate_fixed=self.history_gate_fixed,
                )
                use_memory = memory_active and ((layer + 1) % self.memory_every == 0)
                x_cur, out[f"block{layer:02d}"] = block_cur(x_cur, hist_mem, use_memory, deterministic)

        out["pre_ln"] = x_cur
        x_cur = nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x_cur)
        return x_cur[:, None], out


class _Module(nn.Module):
    """SigLIP/ViT module with fixed-grid temporal history memory."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None
    num_heads: int = 12
    posemb: str = "learn"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"
    head_zeroinit: bool = True
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    memory_every: int = 4
    current_frame_index: int = -1
    history_memory_tokens: int = 128
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2
    history_gate_init: float = -6.9
    history_gate_fixed: float | None = None

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}
        image = jnp.asarray(image, jnp.float32)
        if image.ndim == 4:
            image = image[:, None]
        elif image.ndim != 5:
            raise ValueError(f"Expected image rank 4 or 5, got {image.shape}")

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
        x = x.reshape(b, t, h * w, c)
        pe2d = _compress.get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)
        x = out["with_posemb"] = x + pe2d[:, None]

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [b, t, 1, 1]), x], axis=2)

        x = nn.Dropout(rate=self.dropout)(x, not train).astype(self.dtype_mm)
        x, out["encoder"] = EncoderCurrentOnlyFixedGridTemporal(
            name="Transformer",
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
            temporal_width=self.temporal_width,
            temporal_depth=self.temporal_depth,
            temporal_heads=self.temporal_heads,
            spatial_pool_factor=self.spatial_pool_factor,
            history_gate_init=self.history_gate_init,
            history_gate_fixed=self.history_gate_fixed,
        )(x, deterministic=not train)

        out["encoded_video"] = x
        current_tokens = x[:, 0]
        encoded_current = current_tokens
        out["encoded"] = encoded_current

        if self.pool_type == "map":
            x_out = out["head_input"] = _compress.MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype_mm=self.dtype_mm,
            )(current_tokens)
        elif self.pool_type == "gap":
            x_out = out["head_input"] = jnp.mean(current_tokens, axis=1)
        elif self.pool_type in ("0", "tok"):
            x_out = out["head_input"] = current_tokens[:, 0]
            if self.pool_type == "tok":
                encoded_current = encoded_current[:, 1:]
        elif self.pool_type == "none":
            x_out = current_tokens
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type!r}")

        x_2d = jnp.reshape(encoded_current, [b, h, w, -1])
        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hidden = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            x_2d = nn.tanh(hidden(x_2d))
            x_out = nn.tanh(hidden(x_out))
        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x_out

        if self.num_classes:
            kwargs = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kwargs)
            x_2d = out["logits_2d"] = head(x_2d)
            x_out = out["logits"] = head(x_out)
        return x_out, out


def Module(num_classes=None, *, variant=None, **kwargs):  # noqa: N802
    """Factory matching the public SigLIP module API."""
    return _Module(num_classes, **{**_compress.decode_variant(variant), **kwargs})


decode_variant = _compress.decode_variant
