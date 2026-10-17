"""SigLIP video/memory encoder for MEM-style short-term video context.

This module keeps the original SigLIP/ViT patch embedding, spatial attention, and MLP
structure, and adds an optional patch-aligned causal temporal attention branch inside
each encoder block. The temporal branch reuses the SAME LayerNorm and MHSA weights as
the spatial branch, so no extra learnable parameters are introduced by the temporal
path itself.

Key design choices:
- Input can be either a single image [B, H, W, C] or a short video clip [B, T, H, W, C].
- Spatial attention is applied frame-wise.
- Every `temporal_every` layers, an extra causal temporal attention is applied along
  the time dimension for each aligned patch index independently.
- When T == 1, the temporal branch is skipped, so the model reduces to the original
  single-frame behavior at the block level.
- The output is the CURRENT frame representation (default: last frame), matching how
  MEM passes only the current-step visual tokens to the downstream policy.

Notes:
- This file is intentionally close to openpi.models.siglip.py to make weight reuse
  easier and the code easier to diff.
- The `scan=True` path preserves the original SigLIP parameter-tree style under
  `Transformer/encoderblock/...`, while adding a per-layer temporal schedule as a
  scanned input.
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


class Encoder1DBlockMEM(nn.Module):
    """Single transformer block with optional patch-aligned causal temporal attention.

    Accepts two carry shapes so the same module (and the same parameter tree
    under ``encoderblock/...``) can serve both regimes:

    - 4D ``[B, T, N, D]``: standard video case, supports ``T == 1`` as well
      as ``T > 1``.
    - 3D ``[B, N, D]``: ``T == 1`` fast path used by :class:`Encoder` when
      the time dim has been squeezed before entering ``nn.scan``. This keeps
      the scan carry rank-3 — matching vanilla SigLIP exactly — so XLA does
      not drag a phantom ``size=1`` time dim through ``scan + remat + sharding``.

    Args:
        x: ``[B, T, N, D]`` or ``[B, N, D]``.
        use_temporal: scalar bool / bool[] determining whether this layer
            activates the temporal branch. Ignored in the 3D fast path
            (temporal attention is a no-op when T == 1).
        deterministic: dropout flag.
    """

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, use_temporal, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)

        # Detect carry rank at trace time. Both ranks produce identical
        # parameter shapes for the shared modules below, so checkpoints carry
        # over freely between 3D and 4D call sites.
        is_4d = x.ndim == 4
        if is_4d:
            b, t, n, d = x.shape
        else:
            b, n, d = x.shape
            t = 1

        # Shared LN + attention module reused by both spatial and temporal branches.
        ln1 = nn.LayerNorm(name="LayerNorm_0", dtype=self.dtype_mm)
        attn = nn.MultiHeadDotProductAttention(
            name="MultiHeadDotProductAttention_0",
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )

        # 1) Spatial attention: apply independently on each frame.
        y = ln1(x)
        if is_4d:
            y = y.reshape(b * t, n, d)
            y = out["sa"] = attn(y, y)
            y = y.reshape(b, t, n, d)
        else:
            y = out["sa"] = attn(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        # 2) Optional causal temporal attention per patch index.
        # Static short-circuit: when t == 1 (3D fast path, or T=1 ablation
        # runs with 4D carry), the temporal branch is a no-op by construction.
        # Avoid even tracing ``jax.lax.cond`` so the dead branch never enters
        # the HLO graph — this is what makes T=1 Pi0Mem compile-equivalent to
        # vanilla Pi0 instead of paying a phantom temporal-branch tax inside
        # remat+scan.
        if t > 1:
            def temporal_branch(x_in):
                y_t = ln1(x_in)
                tpe = posemb_sincos_1d(t, d, dtype=y_t.dtype)[:, :, None, :]  # [1, T, 1, D]
                y_t = y_t + tpe

                # [B, T, N, D] -> [B*N, T, D]
                y_t = jnp.transpose(y_t, (0, 2, 1, 3)).reshape(b * n, t, d)
                mask = nn.make_causal_mask(jnp.ones((b * n, t), dtype=jnp.bool_), dtype=jnp.bool_)
                y_t = attn(y_t, y_t, mask=mask)
                y_t = y_t.reshape(b, n, t, d)
                y_t = jnp.transpose(y_t, (0, 2, 1, 3))
                y_t = sharding.activation_sharding_constraint(y_t)
                y_t = nn.Dropout(rate=self.dropout)(y_t, deterministic)
                x_out = x_in + y_t
                return x_out, y_t

            apply_temporal = jnp.asarray(use_temporal, dtype=jnp.bool_)
            x, temporal_y = jax.lax.cond(
                apply_temporal,
                temporal_branch,
                lambda x_in: (x_in, jnp.zeros_like(x_in)),
                x,
            )
        else:
            # t == 1: temporal attention is identity by definition. Skip the
            # branch entirely so the remat/scan compiler doesn't pay for it.
            temporal_y = jnp.zeros_like(x)
        out["ta"] = temporal_y
        out["+ta"] = x

        # 3) MLP branch.
        y = nn.LayerNorm(name="LayerNorm_1", dtype=self.dtype_mm)(x)
        mlp_block = MlpBlock(
            name="MlpBlock_0",
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )
        if is_4d:
            y = out["mlp"] = mlp_block(y.reshape(b * t, n, d), deterministic).reshape(b, t, n, d)
        else:
            y = out["mlp"] = mlp_block(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


class Encoder(nn.Module):
    """Transformer encoder with optional periodic temporal attention."""

    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    temporal_every: int = 4

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Args:
        x: [B, T, N, D]
        """
        out = {}

        # T==1 fast path: drop the time dim so scan / remat / sharding
        # constraints all operate on a 3D carry (B, N, D), matching vanilla
        # SigLIP's compile graph exactly. We restore the time dim before
        # returning so callers can keep relying on the (B, T, N, D) contract.
        # Because no block can ever activate the temporal branch when T==1,
        # we also skip the entire ``temporal_flags`` plumbing in this case.
        drop_time = x.ndim == 4 and x.shape[1] == 1
        if drop_time:
            x = x.squeeze(1)
            temporal_active = False
        else:
            temporal_active = self.temporal_every > 0

        if self.scan:
            if temporal_active:
                temporal_flags = ((jnp.arange(self.depth) + 1) % self.temporal_every == 0)
            else:
                temporal_flags = jnp.zeros((self.depth,), dtype=jnp.bool_)

            block = nn.remat(
                Encoder1DBlockMEM,
                prevent_cse=False,
                static_argnums=(3,),  # 0=self, 1=x(carry), 2=use_temporal(xs), 3=deterministic(xs)
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )

            ScanBlock = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(0, nn.broadcast),
                length=self.depth,
            )

            x, scan_out = ScanBlock(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, temporal_flags, deterministic)

            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            for lyr in range(self.depth):
                block_cur = Encoder1DBlockMEM(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                use_temporal = temporal_active and ((lyr + 1) % self.temporal_every == 0)
                x, out[f"block{lyr:02d}"] = block_cur(x, use_temporal, deterministic)

        out["pre_ln"] = x
        # Final norm: frame-wise when 4D, token-wise when 3D (T==1 fast path).
        if x.ndim == 4:
            b, t, n, d = x.shape
            x = nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(
                x.reshape(b * t, n, d)
            ).reshape(b, t, n, d)
        else:
            x = nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x)

        # Restore the time dim so downstream code keeps the (B, T, N, D) contract.
        if drop_time:
            x = x[:, None, ...]
        return x, out


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
    """MEM-style SigLIP/ViT model.

    Differences from the original single-frame SigLIP:
    - Accepts video clips [B, T, H, W, C] in addition to single images [B, H, W, C].
    - Runs spatial attention per frame and periodic causal temporal attention across frames.
    - Returns the CURRENT frame representation, so downstream policy code can stay close
      to the original `image_tokens, _ = img(...)` usage.
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

    # MEM-specific options.
    temporal_every: int = 4
    current_frame_index: int = -1

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # Accept either [B, H, W, C] or [B, T, H, W, C].
        image = jnp.asarray(image, jnp.float32)
        if image.ndim == 4:
            image = image[:, None, ...]  # [B, 1, H, W, C]
        elif image.ndim != 5:
            raise ValueError(f"Expected image rank 4 or 5, got shape={image.shape}")

        b, t, h_in, w_in, c_in = image.shape
        image_bt = image.reshape(b * t, h_in, w_in, c_in)

        # Patch extraction.
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

        # [B*T, h, w, c] -> [B, T, N, c]
        x = x.reshape(b, t, num_patches, c)

        # Add 2D posemb to every frame before any temporal mixing.
        pe2d = get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)  # [1, N, C]
        x = out["with_posemb"] = x + pe2d[:, None, :, :]

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, 1, c), x.dtype)
            cls = jnp.tile(cls, [b, t, 1, 1])
            x = jnp.concatenate([cls, x], axis=2)

        x = nn.Dropout(rate=self.dropout)(x, not train)
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            temporal_every=self.temporal_every,
            name="Transformer",
        )(x, deterministic=not train)

        out["encoded_video"] = x

        # Select the current frame representation only.
        current_idx = self.current_frame_index
        current_tokens = x[:, current_idx, :, :]
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
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

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
