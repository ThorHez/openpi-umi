"""Semantic recurrent visual memory used by the validated ShellGame policy.

This module consolidates the visual-memory components that were previously
spread across several ``examples/shellgame`` probes.  It intentionally keeps
their Flax module and parameter names stable so the validated checkpoints can
be restored without remapping:

* pool the 16x16 SigLIP patch grid to a topology-preserving 8x8 grid;
* classify the three ten-frame swap clips into discrete relations;
* write the initial target slot and relations into a recurrent [128, 64]
  semantic memory; and
* retain the diagnostic history readout used by the source checkpoints.

The surrounding Pi0 policy lives in :mod:`openpi.models.pi0_mem_semantic_action`.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

HISTORY_FRAMES = 60
TOTAL_INPUT_FRAMES = 61
SWAP_SLICES = ((20, 30), (30, 40), (40, 50))
SWAP_SEGMENT_SIZE = 10
SPATIAL_TOKENS = 64


def pool_fixed_grid(patch_tokens: jax.Array) -> jax.Array:
    """Pool a 16x16 SigLIP patch grid to a topology-preserving 8x8 grid."""
    if patch_tokens.ndim != 4:
        raise ValueError(f"Expected [B,T,N,D] patch tokens, got {patch_tokens.shape}")
    b, t, n, d = patch_tokens.shape
    input_grid = int(np.sqrt(n))
    pool_factor = 2
    output_grid = input_grid // pool_factor
    if input_grid**2 != n or output_grid**2 != SPATIAL_TOKENS:
        raise ValueError(f"Expected a 16x16 patch grid, got N={n}")
    x = patch_tokens.reshape(b, t, output_grid, pool_factor, output_grid, pool_factor, d)
    return jnp.mean(x, axis=(3, 5)).reshape(b, t, SPATIAL_TOKENS, d)


class FactorizedSpaceTimeBlock(nn.Module):
    """Apply temporal attention per grid cell, then spatial attention per frame."""

    width: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, x, *, train: bool):
        b, t, n, d = x.shape
        if d != self.width:
            raise ValueError(f"Expected width {self.width}, got {d}")

        y = nn.LayerNorm(name="temporal_ln", dtype=self.dtype_mm)(x)
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(b * n, t, d)
        y = nn.MultiHeadDotProductAttention(
            name="temporal_attn",
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
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
            deterministic=not train,
            dtype=self.dtype_mm,
        )(y, y)
        y = y.reshape(b, t, n, d)
        x = x + y

        y = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(x)
        y = nn.Dense(self.width * self.mlp_ratio, name="mlp_in", dtype=self.dtype_mm)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=not train)
        y = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(y)
        return x + y


class PretrainedSwapSemanticEncoder(nn.Module):
    """Encode one ten-frame swap clip with the validated relation encoder."""

    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    segment_size: int = SWAP_SEGMENT_SIZE
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, segment_tokens):
        expected = (self.segment_size, SPATIAL_TOKENS, self.input_width)
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
            )(x, train=False)

        flat = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(x.reshape(x.shape[0], -1, self.width))
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
            deterministic=True,
            dtype=self.dtype_mm,
        )(query, flat)
        return nn.LayerNorm(name="readout_ln", dtype=self.dtype_mm)(pooled[:, 0])


class FrozenSwapRelationClassifier(nn.Module):
    """Run the complete pretrained swap-pair classifier."""

    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    segment_size: int = SWAP_SEGMENT_SIZE
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, segment_tokens):
        semantic = PretrainedSwapSemanticEncoder(
            name="semantic_encoder",
            input_width=self.input_width,
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=self.segment_size,
            dtype_mm=self.dtype_mm,
        )(segment_tokens)
        return nn.Dense(3, name="classifier", dtype=jnp.float32)(semantic.astype(jnp.float32))


class SegmentMemoryUpdateBlock(nn.Module):
    """Cross-attend persistent memory to one relation segment and refine it."""

    width: int = 64
    num_heads: int = 4
    mlp_ratio: int = 4
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, memory, segment_tokens):
        memory_norm = nn.LayerNorm(name="memory_ln", dtype=self.dtype_mm)(memory)
        segment_norm = nn.LayerNorm(name="segment_ln", dtype=self.dtype_mm)(segment_tokens)
        update = nn.MultiHeadDotProductAttention(
            name="cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(memory_norm, segment_norm)
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


class SharedSegmentMemoryUpdater(nn.Module):
    """A recurrent updater whose parameters are shared across all three swaps."""

    width: int = 64
    depth: int = 2
    num_heads: int = 4
    segment_size: int = SWAP_SEGMENT_SIZE
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, memory, segment):
        if segment.ndim != 4 or segment.shape[1] != self.segment_size:
            raise ValueError(f"Expected [B,{self.segment_size},K,D] segment, got {segment.shape}")
        relative_pos = self.param(
            "relative_temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.segment_size, 1, self.width),
            segment.dtype,
        )
        segment_tokens = (segment + relative_pos).reshape(segment.shape[0], -1, self.width)
        for block_index in range(self.depth):
            memory = SegmentMemoryUpdateBlock(
                name=f"update_block_{block_index}",
                width=self.width,
                num_heads=self.num_heads,
                dtype_mm=self.dtype_mm,
            )(memory, segment_tokens)
        return nn.LayerNorm(name="state_output_ln", dtype=self.dtype_mm)(memory)


class SingleHistoryReadAdapter(nn.Module):
    """Preserve the checkpoint-compatible diagnostic read from compact memory."""

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


class SharedMemoryTokenReadout(nn.Module):
    """Read a three-way slot prediction from the wide diagnostic tokens."""

    width: int = 1152

    @nn.compact
    def __call__(self, memory):
        x = nn.LayerNorm(name="input_ln", dtype=jnp.float32)(memory)
        scores = nn.Dense(1, name="attention", dtype=jnp.float32)(x)
        weights = nn.softmax(scores, axis=1)
        pooled = jnp.sum(weights * x, axis=1)
        pooled = nn.LayerNorm(name="pooled_ln", dtype=jnp.float32)(pooled)
        return nn.Dense(3, name="classifier", dtype=jnp.float32)(pooled)


class FrozenFrame0InitialCupClassifier(nn.Module):
    """Checkpoint-compatible spatial classifier for the initial target cup."""

    input_width: int = 1152

    @nn.compact
    def __call__(self, patch_features):
        if patch_features.ndim != 3 or patch_features.shape[1:] != (256, self.input_width):
            raise ValueError(f"Expected frame-0 patches [B,256,{self.input_width}], got {patch_features.shape}")
        x = nn.LayerNorm(name="initial_ln", dtype=jnp.bfloat16)(patch_features)
        return nn.Dense(3, name="initial_head", dtype=jnp.bfloat16)(x.reshape(x.shape[0], -1))


class ThreeSwapVisualRelationMemoryTracker(nn.Module):
    """Decode three visual relations and update compact semantic memory."""

    num_frames: int = HISTORY_FRAMES
    input_width: int = 1152
    encoder_width: int = 256
    encoder_depth: int = 2
    encoder_heads: int = 8
    memory_width: int = 64
    memory_depth: int = 2
    memory_heads: int = 4
    adapter_heads: int = 4
    num_memory_tokens: int = 128
    num_current_tokens: int = 256
    current_width: int = 1152
    residual_scale: float = 1.0
    relation_mode: str = "one_hot"
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens, initial_slots):
        b, t, n, d = patch_tokens.shape
        expected = (self.num_frames, 256, self.input_width)
        if (t, n, d) != expected:
            raise ValueError(f"Expected [B,{expected}], got {patch_tokens.shape}")
        if initial_slots.shape != (b,):
            raise ValueError(f"Expected initial slots [B], got {initial_slots.shape}")

        pooled = pool_fixed_grid(patch_tokens)
        clips = jnp.stack([pooled[:, start:end] for start, end in SWAP_SLICES], axis=1).reshape(
            b * len(SWAP_SLICES), SWAP_SEGMENT_SIZE, SPATIAL_TOKENS, self.input_width
        )
        relation_logits = FrozenSwapRelationClassifier(
            name="swap_relation_classifier",
            input_width=self.input_width,
            width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            segment_size=SWAP_SEGMENT_SIZE,
            dtype_mm=self.dtype_mm,
        )(clips).reshape(b, len(SWAP_SLICES), 3)
        relation_ids = jnp.argmax(relation_logits, axis=-1)
        if self.relation_mode == "one_hot":
            relation_codes = jax.nn.one_hot(relation_ids, 3, dtype=jnp.float32)
        elif self.relation_mode == "probabilities":
            relation_codes = jax.nn.softmax(relation_logits, axis=-1).astype(jnp.float32)
        elif self.relation_mode == "logits":
            relation_codes = relation_logits.astype(jnp.float32)
        else:
            raise ValueError(f"Unknown relation_mode={self.relation_mode!r}")

        segment_tokens = jnp.zeros(
            (b, len(SWAP_SLICES), SWAP_SEGMENT_SIZE, SPATIAL_TOKENS, self.memory_width),
            dtype=jnp.float32,
        )
        segment_tokens = segment_tokens.at[..., :3].add(relation_codes[:, :, None, None, :])

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.memory_width),
            jnp.float32,
        )
        memory = jnp.tile(base_memory, (b, 1, 1))
        initial_code = jax.nn.one_hot(initial_slots, 3, dtype=jnp.float32)
        memory = memory.at[:, 0, :3].add(initial_code)

        updater = SharedSegmentMemoryUpdater(
            name="shared_swap_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            segment_size=SWAP_SEGMENT_SIZE,
            dtype_mm="float32",
        )
        adapter = SingleHistoryReadAdapter(
            name="shared_history_read_adapter",
            memory_width=self.memory_width,
            current_width=self.current_width,
            num_heads=self.adapter_heads,
            residual_scale=self.residual_scale,
        )
        readout = SharedMemoryTokenReadout(name="shared_readout", width=self.current_width)
        base_current_tokens = self.param(
            "base_current_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_current_tokens, self.current_width),
            jnp.float32,
        )
        base_current_tokens = jnp.tile(base_current_tokens, (b, 1, 1))

        stage_logits = []
        stage_memories = []
        for stage_index in range(len(SWAP_SLICES)):
            memory = updater(memory, segment_tokens[:, stage_index])
            stage_memories.append(memory)
            current_tokens = adapter(base_current_tokens, memory)
            stage_logits.append(readout(current_tokens))

        stage_logits = jnp.stack(stage_logits, axis=1)
        stage_memories = jnp.stack(stage_memories, axis=1)
        logits_0, logits_1, logits_2 = stage_logits[:, 0], stage_logits[:, 1], stage_logits[:, 2]
        joint_logits = (
            logits_0[:, :, None, None] + logits_1[:, None, :, None] + logits_2[:, None, None, :]
        ).reshape(b, 27)
        return joint_logits, stage_logits, stage_memories, relation_logits, relation_ids
