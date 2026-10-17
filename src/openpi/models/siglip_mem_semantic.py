"""Task-agnostic visual encoding and recurrent-memory building blocks.

This module contains no assumptions about a dataset, event vocabulary, class
count, frame layout, or downstream task.  Task adapters are responsible for
turning observations into temporal segments and, when useful, semantic event
tokens.  The reusable core here only performs:

* topology-preserving pooling of square patch grids;
* factorized temporal/spatial visual encoding;
* recurrent cross-attention updates of persistent memory; and
* task-neutral reads from compact memory.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def pool_fixed_grid(patch_tokens: jax.Array, *, pool_factor: int = 2) -> jax.Array:
    """Average-pool a square patch grid while preserving its 2-D topology."""
    if patch_tokens.ndim != 4:
        raise ValueError(f"Expected [B,T,N,D] patch tokens, got {patch_tokens.shape}")
    if pool_factor <= 0:
        raise ValueError(f"pool_factor must be positive, got {pool_factor}")

    batch, frames, tokens, width = patch_tokens.shape
    input_grid = int(np.sqrt(tokens))
    if input_grid**2 != tokens or input_grid % pool_factor != 0:
        raise ValueError(
            f"Expected a square patch grid divisible by pool_factor={pool_factor}, got N={tokens}"
        )
    output_grid = input_grid // pool_factor
    x = patch_tokens.reshape(
        batch,
        frames,
        output_grid,
        pool_factor,
        output_grid,
        pool_factor,
        width,
    )
    return jnp.mean(x, axis=(3, 5)).reshape(
        batch, frames, output_grid * output_grid, width
    )


class FactorizedSpaceTimeBlock(nn.Module):
    """Apply temporal attention per grid cell, then spatial attention per frame."""

    width: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, x, *, train: bool):
        batch, frames, tokens, width = x.shape
        if width != self.width:
            raise ValueError(f"Expected width {self.width}, got {width}")

        y = nn.LayerNorm(name="temporal_ln", dtype=self.dtype_mm)(x)
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(batch * tokens, frames, width)
        y = nn.MultiHeadDotProductAttention(
            name="temporal_attn",
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
            dtype=self.dtype_mm,
        )(y, y)
        y = y.reshape(batch, tokens, frames, width).transpose(0, 2, 1, 3)
        x = x + y

        y = nn.LayerNorm(name="spatial_ln", dtype=self.dtype_mm)(x)
        y = y.reshape(batch * frames, tokens, width)
        y = nn.MultiHeadDotProductAttention(
            name="spatial_attn",
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
            dtype=self.dtype_mm,
        )(y, y)
        y = y.reshape(batch, frames, tokens, width)
        x = x + y

        y = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(x)
        y = nn.Dense(self.width * self.mlp_ratio, name="mlp_in", dtype=self.dtype_mm)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=not train)
        y = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(y)
        return x + y


class FactorizedSpaceTimeEncoder(nn.Module):
    """Encode an arbitrary fixed-length visual segment into one semantic vector."""

    segment_size: int
    spatial_tokens: int
    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, segment_tokens, *, train: bool = False):
        expected = (self.segment_size, self.spatial_tokens, self.input_width)
        if segment_tokens.ndim != 4 or segment_tokens.shape[1:] != expected:
            raise ValueError(f"Expected [B,{expected}], got {segment_tokens.shape}")

        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(segment_tokens)
        x = nn.Dense(self.width, name="input_projection", dtype=self.dtype_mm)(x)
        temporal_pos = self.param(
            "relative_temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.segment_size, 1, self.width),
            x.dtype,
        )
        x = x + temporal_pos
        for block_index in range(self.depth):
            x = FactorizedSpaceTimeBlock(
                name=f"block_{block_index}",
                width=self.width,
                num_heads=self.num_heads,
                dropout=0.0,
                dtype_mm=self.dtype_mm,
            )(x, train=train)

        flat = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(
            x.reshape(x.shape[0], -1, self.width)
        )
        readout_query = self.param(
            "readout_query",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.width),
            flat.dtype,
        )
        query = jnp.tile(readout_query, (flat.shape[0], 1, 1))
        pooled = nn.MultiHeadDotProductAttention(
            name="readout_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=not train,
            dtype=self.dtype_mm,
        )(query, flat)
        return nn.LayerNorm(name="readout_ln", dtype=self.dtype_mm)(pooled[:, 0])


class MemoryUpdateBlock(nn.Module):
    """Refine persistent memory using an arbitrary set of evidence tokens."""

    width: int = 64
    num_heads: int = 4
    mlp_ratio: int = 4
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, memory, evidence_tokens):
        memory_norm = nn.LayerNorm(name="memory_ln", dtype=self.dtype_mm)(memory)
        # Keep the internal name checkpoint-compatible; the tensor itself has
        # no segment semantics and may come from any modality or task.
        evidence_norm = nn.LayerNorm(name="segment_ln", dtype=self.dtype_mm)(
            evidence_tokens
        )
        update = nn.MultiHeadDotProductAttention(
            name="cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(memory_norm, evidence_norm)
        memory = memory + update

        y = nn.LayerNorm(name="self_ln", dtype=self.dtype_mm)(memory)
        y = nn.MultiHeadDotProductAttention(
            name="self_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(y, y)
        memory = memory + y

        y = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(memory)
        y = nn.Dense(self.width * self.mlp_ratio, name="mlp_in", dtype=self.dtype_mm)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(y)
        return memory + y


def recurrently_update_memory(
    memory: jax.Array,
    evidence_steps: jax.Array,
    *,
    width: int,
    depth: int,
    num_heads: int,
    dtype_mm: str,
    step_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Apply one shared memory transition over an evidence sequence.

    This function must be called inside a Linen ``@compact`` method. It makes
    no assumptions about how evidence was produced, how many update steps are
    present, or how tokens within a step are ordered.

    Args:
        memory: Initial persistent state ``[B, M, D]``.
        evidence_steps: Evidence sequence ``[B, S, N, D]``.
        width: Shared memory/evidence feature width.
        depth: Number of refinement blocks in each recurrent transition.
        num_heads: Attention heads used by each refinement block.
        dtype_mm: Matrix-multiplication dtype.
        step_mask: Optional validity mask ``[B, S]``. Invalid updates preserve
            the previous memory state.

    Returns:
        Final memory and all recurrent states ``[B, S, M, D]``.
    """
    if memory.ndim != 3 or memory.shape[-1] != width:
        raise ValueError(f"Expected memory [B,M,{width}], got {memory.shape}")
    if (
        evidence_steps.ndim != 4
        or evidence_steps.shape[0] != memory.shape[0]
        or evidence_steps.shape[-1] != width
    ):
        raise ValueError(
            f"Expected evidence_steps [B,S,N,{width}] with matching batch, "
            f"got {evidence_steps.shape}"
        )
    if evidence_steps.shape[1] < 1:
        raise ValueError("evidence_steps must contain at least one update")
    if step_mask is not None and step_mask.shape != evidence_steps.shape[:2]:
        raise ValueError(
            f"Expected step_mask {evidence_steps.shape[:2]}, got {step_mask.shape}"
        )

    blocks = tuple(
        MemoryUpdateBlock(
            name=f"update_block_{block_index}",
            width=width,
            num_heads=num_heads,
            dtype_mm=dtype_mm,
        )
        for block_index in range(depth)
    )
    output_norm = nn.LayerNorm(name="state_output_ln", dtype=dtype_mm)
    states = []
    for step_index in range(evidence_steps.shape[1]):
        candidate = memory
        evidence = evidence_steps[:, step_index]
        for block in blocks:
            candidate = block(candidate, evidence)
        candidate = output_norm(candidate)
        if step_mask is None:
            memory = candidate
        else:
            valid = step_mask[:, step_index, None, None]
            memory = jnp.where(valid, candidate, memory)
        states.append(memory)
    return memory, jnp.stack(states, axis=1)


class RecurrentMemoryUpdater(nn.Module):
    """Task-neutral recurrent update over arbitrary evidence-token steps."""

    width: int = 64
    depth: int = 2
    num_heads: int = 4
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, memory, evidence_steps, *, step_mask=None):
        return recurrently_update_memory(
            memory,
            evidence_steps,
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            dtype_mm=self.dtype_mm,
            step_mask=step_mask,
        )


class SingleHistoryReadAdapter(nn.Module):
    """Read compact memory and inject it as a residual into wider tokens."""

    memory_width: int = 64
    current_width: int = 1152
    num_heads: int = 4
    residual_scale: float = 1.0

    @nn.compact
    def __call__(self, current_tokens, memory):
        read_query = self.param(
            "read_query",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.memory_width),
            jnp.float32,
        )
        read_query = jnp.tile(read_query, (memory.shape[0], 1, 1))
        query_norm = nn.LayerNorm(name="query_ln", dtype=jnp.float32)(read_query)
        memory_norm = nn.LayerNorm(name="memory_ln", dtype=jnp.float32)(memory)
        attended = nn.MultiHeadDotProductAttention(
            name="cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=jnp.float32,
        )(query_norm, memory_norm)
        semantic = nn.LayerNorm(name="semantic_ln", dtype=jnp.float32)(read_query + attended)
        residual = nn.Dense(self.current_width, name="residual_projection", dtype=jnp.float32)(semantic)
        residual = nn.LayerNorm(name="residual_ln", dtype=jnp.float32)(residual)
        return current_tokens + self.residual_scale * residual
