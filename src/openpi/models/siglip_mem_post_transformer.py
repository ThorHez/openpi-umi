"""Experimental SigLIP encoder that transforms compressed history tokens.

This module intentionally lives next to, rather than replacing,
``siglip_mem_compress``.  It is a capacity experiment for the following path:

    historical patches -> HistoryResampler -> memory tokens
    [memory tokens, current-frame patches] -> pretrained SigLIP blocks

The first ``joint_start_layer`` blocks use a block-diagonal attention mask, so
memory and current tokens are semanticized independently.  The remaining
blocks use full self-attention and jointly update both token groups.  Only the
current-frame tokens are returned to PaliGemma; the post-Transformer memory is
exposed in ``out["encoder"]["history_mem"]`` for probes and diagnostics.

The standard Transformer parameter names and scanned layout match the original
SigLIP encoder, allowing its pretrained / fine-tuned spatial weights to load.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import siglip_mem_compress as _base
import openpi.training.sharding as sharding


class HistoryResampler(nn.Module):
    """Compress raw historical patches without forcing slot-wise zero mean."""

    num_memory_tokens: int = 256
    num_heads: int = 12
    depth: int = 1
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"
    use_current_condition: bool = True

    @nn.compact
    def __call__(self, x_hist, x_cur, deterministic=True):  # noqa: FBT002
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
            y = _base.MlpBlock(
                name=f"MlpBlock_{lyr}",
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
            )(y, deterministic)
            y = nn.Dropout(rate=self.dropout)(y, deterministic)
            q = sharding.activation_sharding_constraint(q + y)

        # Unlike siglip_mem_compress.HistoryResampler, preserve the common
        # episode-level component.  Per-token normalization keeps scale stable.
        return nn.LayerNorm(name="out_ln", dtype=self.dtype_mm)(q)


class Encoder1DBlockJoint(nn.Module):
    """Standard SigLIP block with scheduled memory/current interaction."""

    memory_tokens: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, joint_attention, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        seq_len = x.shape[1]
        if self.memory_tokens <= 0 or self.memory_tokens >= seq_len:
            raise ValueError(f"memory_tokens must be in [1, seq_len), got {self.memory_tokens} for {seq_len=}")

        token_ids = jnp.arange(seq_len)
        is_memory = token_ids < self.memory_tokens
        same_group = jnp.equal(is_memory[:, None], is_memory[None, :])
        full_attention = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
        attention_mask = jnp.where(joint_attention, full_attention, same_group)[None, None, :, :]

        y = nn.LayerNorm(name="LayerNorm_0", dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            name="MultiHeadDotProductAttention_0",
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y, mask=attention_mask)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(name="LayerNorm_1", dtype=self.dtype_mm)(x)
        y = out["mlp"] = _base.MlpBlock(
            name="MlpBlock_0",
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        return sharding.activation_sharding_constraint(x), out


class EncoderCompressedPostTransformer(nn.Module):
    """Compress history, then transform memory and current tokens together."""

    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = True
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    current_frame_index: int = -1
    history_memory_tokens: int = 256
    history_resampler_depth: int = 1
    history_use_current_condition: bool = True
    joint_start_layer: int = 18

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        if x.ndim == 3:
            x = x[:, None, ...]
        if x.ndim != 4:
            raise ValueError(f"Expected x rank 3 or 4, got shape={x.shape}")
        if not 0 <= self.joint_start_layer < self.depth:
            raise ValueError(f"joint_start_layer must be in [0, {self.depth}), got {self.joint_start_layer}")

        b, t, _, d = x.shape
        cur_idx = self.current_frame_index if self.current_frame_index >= 0 else t + self.current_frame_index
        if cur_idx <= 0 or cur_idx >= t:
            raise ValueError(f"This experiment requires history before current frame, got {cur_idx=} for {t=}")

        tpe = _base.posemb_sincos_1d(t, d, dtype=x.dtype)[:, :, None, :]
        x_with_tpe = x + tpe
        x_cur = x[:, cur_idx, :, :]
        x_hist = x_with_tpe[:, :cur_idx, :, :]
        x_cur_for_cond = x_with_tpe[:, cur_idx, :, :]

        history_mem_pre = HistoryResampler(
            name="HistoryResampler_0",
            num_memory_tokens=self.history_memory_tokens,
            num_heads=self.num_heads,
            depth=self.history_resampler_depth,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
            use_current_condition=self.history_use_current_condition,
        )(x_hist, x_cur_for_cond, deterministic)

        tokens = jnp.concatenate([history_mem_pre, x_cur], axis=1)
        joint_flags = jnp.arange(self.depth) >= self.joint_start_layer

        if not self.scan:
            raise ValueError("The isolated post-Transformer experiment requires scan=True for checkpoint compatibility")

        block = nn.remat(
            Encoder1DBlockJoint,
            prevent_cse=False,
            static_argnums=(3,),
            policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
        )
        scan_block = nn.scan(
            block,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast),
            length=self.depth,
        )
        tokens, scan_out = scan_block(
            name="encoderblock",
            memory_tokens=self.history_memory_tokens,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(tokens, joint_flags, deterministic)

        out = {f"block{lyr:02d}": jax.tree.map(lambda value, i=lyr: value[i], scan_out) for lyr in range(self.depth)}
        out["pre_ln"] = tokens
        tokens = nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(tokens)
        history_mem = tokens[:, : self.history_memory_tokens]
        current_tokens = tokens[:, self.history_memory_tokens :]
        out["history_mem_pre_transform"] = history_mem_pre
        out["history_mem"] = history_mem
        return current_tokens[:, None, :, :], out


class _Module(_base._Module):  # noqa: SLF001
    """Drop-in visual module for the isolated post-Transformer experiment."""

    joint_start_layer: int = 18

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}
        image = jnp.asarray(image, jnp.float32)
        if image.ndim == 4:
            image = image[:, None, ...]
        elif image.ndim != 5:
            raise ValueError(f"Expected image rank 4 or 5, got shape={image.shape}")

        b, t, h_in, w_in, _ = image.shape
        image_bt = image.reshape(b * t, h_in, w_in, image.shape[-1])
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
        pe2d = _base.get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)
        x = out["with_posemb"] = x + pe2d[:, None, :, :]
        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [b, t, 1, 1]), x], axis=2)
        x = nn.Dropout(rate=self.dropout)(x, not train).astype(self.dtype_mm)

        x, out["encoder"] = EncoderCompressedPostTransformer(
            name="Transformer",
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            current_frame_index=self.current_frame_index,
            history_memory_tokens=self.history_memory_tokens,
            history_resampler_depth=self.history_resampler_depth,
            history_use_current_condition=self.history_use_current_condition,
            joint_start_layer=self.joint_start_layer,
        )(x, deterministic=not train)

        out["encoded_video"] = x
        current_tokens = x[:, 0]
        encoded_current = current_tokens
        out["encoded"] = encoded_current

        if self.pool_type == "map":
            x_out = out["head_input"] = _base.MAPHead(
                num_heads=self.num_heads, mlp_dim=self.mlp_dim, dtype_mm=self.dtype_mm
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
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            x_2d = nn.tanh(hid(x_2d))
            x_out = nn.tanh(hid(x_out))
        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x_out

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            out["logits_2d"] = head(x_2d)
            x_out = out["logits"] = head(x_out)
        return x_out, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    return _Module(num_classes, **{**_base.decode_variant(variant), **kw})


decode_variant = _base.decode_variant
