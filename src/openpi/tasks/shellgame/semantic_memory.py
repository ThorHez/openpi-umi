"""ShellGame-specific symbolic event encoder and recurrent memory tracker.

The discrete three-cup and three-swap assumptions intentionally live in the
task layer rather than in :mod:`openpi.models`.  Explicit Flax module names are
kept identical to the validated experiment so existing checkpoints remain
loadable without parameter remapping.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import siglip_mem_semantic as memory_core

HISTORY_FRAMES = 60
TOTAL_INPUT_FRAMES = 61
SWAP_SLICES = ((20, 30), (30, 40), (40, 50))
SWAP_SEGMENT_SIZE = 10
DEFAULT_SWAP_FRAME_INDICES = tuple(
    tuple(range(start, end)) for start, end in SWAP_SLICES
)
SPATIAL_TOKENS = 64
NUM_CUPS = 3


class FrozenSwapRelationClassifier(nn.Module):
    """Run the pretrained three-way ShellGame swap-relation classifier."""

    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    segment_size: int = SWAP_SEGMENT_SIZE
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, segment_tokens):
        semantic = memory_core.FactorizedSpaceTimeEncoder(
            name="semantic_encoder",
            input_width=self.input_width,
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=self.segment_size,
            spatial_tokens=SPATIAL_TOKENS,
            dtype_mm=self.dtype_mm,
        )(segment_tokens, train=False)
        return nn.Dense(NUM_CUPS, name="classifier", dtype=jnp.float32)(
            semantic.astype(jnp.float32)
        )


class SharedMemoryTokenReadout(nn.Module):
    """Read a three-way ShellGame slot prediction from diagnostic tokens."""

    width: int = 1152

    @nn.compact
    def __call__(self, memory):
        x = nn.LayerNorm(name="input_ln", dtype=jnp.float32)(memory)
        scores = nn.Dense(1, name="attention", dtype=jnp.float32)(x)
        weights = nn.softmax(scores, axis=1)
        pooled = jnp.sum(weights * x, axis=1)
        pooled = nn.LayerNorm(name="pooled_ln", dtype=jnp.float32)(pooled)
        return nn.Dense(NUM_CUPS, name="classifier", dtype=jnp.float32)(pooled)


class FrozenFrame0InitialCupClassifier(nn.Module):
    """Checkpoint-compatible classifier for the initial target cup."""

    input_width: int = 1152

    @nn.compact
    def __call__(self, patch_features):
        if patch_features.ndim != 3 or patch_features.shape[1:] != (256, self.input_width):
            raise ValueError(f"Expected frame-0 patches [B,256,{self.input_width}], got {patch_features.shape}")
        x = nn.LayerNorm(name="initial_ln", dtype=jnp.bfloat16)(patch_features)
        return nn.Dense(NUM_CUPS, name="initial_head", dtype=jnp.bfloat16)(
            x.reshape(x.shape[0], -1)
        )


class ShellGameSwapRecurrentMemoryUpdater(nn.Module):
    """Convert fixed swap clips to evidence and run the generic recurrence."""

    width: int = 64
    depth: int = 2
    num_heads: int = 4
    segment_size: int = SWAP_SEGMENT_SIZE
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, memory, swap_segments):
        if (
            swap_segments.ndim != 5
            or swap_segments.shape[0] != memory.shape[0]
            or swap_segments.shape[2] != self.segment_size
            or swap_segments.shape[-1] != self.width
        ):
            raise ValueError(
                "Expected swap_segments "
                f"[B,S,{self.segment_size},K,{self.width}], got {swap_segments.shape}"
            )
        relative_pos = self.param(
            "relative_temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.segment_size, 1, self.width),
            swap_segments.dtype,
        )
        evidence_steps = (swap_segments + relative_pos[:, None]).reshape(
            swap_segments.shape[0],
            swap_segments.shape[1],
            -1,
            self.width,
        )
        return memory_core.recurrently_update_memory(
            memory,
            evidence_steps,
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            dtype_mm=self.dtype_mm,
        )


class ThreeSwapVisualRelationMemoryTracker(nn.Module):
    """Decode the three ShellGame swaps and update compact semantic memory."""

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
    swap_frame_indices: tuple[tuple[int, ...], ...] = DEFAULT_SWAP_FRAME_INDICES

    @nn.compact
    def __call__(self, patch_tokens, initial_slots, relation_ids_override=None):
        batch, frames, tokens, width = patch_tokens.shape
        expected = (self.num_frames, 256, self.input_width)
        if (frames, tokens, width) != expected:
            raise ValueError(f"Expected [B,{expected}], got {patch_tokens.shape}")
        if initial_slots.shape != (batch,):
            raise ValueError(f"Expected initial slots [B], got {initial_slots.shape}")
        if len(self.swap_frame_indices) != 3:
            raise ValueError(
                f"ShellGame requires exactly three swap stages, got {len(self.swap_frame_indices)}"
            )
        segment_sizes = {len(indices) for indices in self.swap_frame_indices}
        if len(segment_sizes) != 1 or not segment_sizes or 0 in segment_sizes:
            raise ValueError(
                "swap_frame_indices must contain three non-empty, equal-length stages"
            )
        segment_size = next(iter(segment_sizes))
        flat_indices = tuple(index for stage in self.swap_frame_indices for index in stage)
        if min(flat_indices) < 0 or max(flat_indices) >= frames:
            raise ValueError(
                f"swap_frame_indices must stay inside [0, {frames}); got "
                f"[{min(flat_indices)}, {max(flat_indices)}]"
            )
        if relation_ids_override is not None and relation_ids_override.shape != (
            batch,
            len(self.swap_frame_indices),
        ):
            raise ValueError(
                "Expected teacher-forced relation ids "
                f"[B,{len(self.swap_frame_indices)}], got {relation_ids_override.shape}"
            )

        pooled = memory_core.pool_fixed_grid(patch_tokens, pool_factor=2)
        if pooled.shape[2] != SPATIAL_TOKENS:
            raise ValueError(
                f"ShellGame relation encoder requires {SPATIAL_TOKENS} pooled tokens, "
                f"got {pooled.shape[2]}"
            )
        clips = jnp.stack(
            [pooled[:, jnp.asarray(indices)] for indices in self.swap_frame_indices],
            axis=1,
        ).reshape(
            batch * len(self.swap_frame_indices),
            segment_size,
            SPATIAL_TOKENS,
            self.input_width,
        )
        relation_logits = FrozenSwapRelationClassifier(
            name="swap_relation_classifier",
            input_width=self.input_width,
            width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            segment_size=segment_size,
            dtype_mm=self.dtype_mm,
        )(clips).reshape(batch, len(self.swap_frame_indices), NUM_CUPS)
        relation_ids = jnp.argmax(relation_logits, axis=-1)
        memory_relation_ids = (
            relation_ids
            if relation_ids_override is None
            else relation_ids_override.astype(jnp.int32)
        )
        if self.relation_mode == "one_hot":
            relation_codes = jax.nn.one_hot(
                memory_relation_ids, NUM_CUPS, dtype=jnp.float32
            )
        elif self.relation_mode == "probabilities":
            relation_codes = jax.nn.softmax(relation_logits, axis=-1).astype(jnp.float32)
        elif self.relation_mode == "logits":
            relation_codes = relation_logits.astype(jnp.float32)
        else:
            raise ValueError(f"Unknown relation_mode={self.relation_mode!r}")

        segment_tokens = jnp.zeros(
            (
                batch,
                len(self.swap_frame_indices),
                segment_size,
                SPATIAL_TOKENS,
                self.memory_width,
            ),
            dtype=jnp.float32,
        )
        segment_tokens = segment_tokens.at[..., :NUM_CUPS].add(
            relation_codes[:, :, None, None, :]
        )

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.memory_width),
            jnp.float32,
        )
        memory = jnp.tile(base_memory, (batch, 1, 1))
        initial_code = jax.nn.one_hot(initial_slots, NUM_CUPS, dtype=jnp.float32)
        memory = memory.at[:, 0, :NUM_CUPS].add(initial_code)

        updater = ShellGameSwapRecurrentMemoryUpdater(
            name="shared_swap_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            segment_size=segment_size,
            dtype_mm="float32",
        )
        adapter = memory_core.SingleHistoryReadAdapter(
            name="shared_history_read_adapter",
            memory_width=self.memory_width,
            current_width=self.current_width,
            num_heads=self.adapter_heads,
            residual_scale=self.residual_scale,
        )
        readout = SharedMemoryTokenReadout(
            name="shared_readout", width=self.current_width
        )
        base_current_tokens = self.param(
            "base_current_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_current_tokens, self.current_width),
            jnp.float32,
        )
        base_current_tokens = jnp.tile(base_current_tokens, (batch, 1, 1))

        _, stage_memories = updater(memory, segment_tokens)
        stage_logits = []
        for stage_index in range(len(SWAP_SLICES)):
            current_tokens = adapter(
                base_current_tokens, stage_memories[:, stage_index]
            )
            stage_logits.append(readout(current_tokens))

        stage_logits = jnp.stack(stage_logits, axis=1)
        logits_0, logits_1, logits_2 = (
            stage_logits[:, 0],
            stage_logits[:, 1],
            stage_logits[:, 2],
        )
        joint_logits = (
            logits_0[:, :, None, None]
            + logits_1[:, None, :, None]
            + logits_2[:, None, None, :]
        ).reshape(batch, NUM_CUPS**len(SWAP_SLICES))
        return joint_logits, stage_logits, stage_memories, relation_logits, relation_ids
