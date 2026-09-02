"""ShellGame adapter for the generic causal event-memory components.

Only this task layer assigns three semantic classes to the event-type head and
limits fixed-shape diagnostics to the first three triggers.  The reusable
model code remains agnostic to cups, swaps, and episode timing.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import siglip_mem_semantic as memory_core
from openpi.models import siglip_mem_semantic_event as event_memory
from openpi.tasks.shellgame import semantic_memory

WINDOW_SIZE = 6
HISTORY_FRAMES = 60
NUM_WINDOWS = HISTORY_FRAMES - WINDOW_SIZE + 1
NUM_STAGES = 3


class ShellGameSlidingWindowEventRelationClassifier(event_memory.SlidingWindowEventClassifier):
    """Checkpoint-compatible six-frame event/relation classifier."""

    spatial_tokens: int = semantic_memory.SPATIAL_TOKENS
    num_event_classes: int = semantic_memory.NUM_CUPS
    event_type_head_name: str = "relation_classifier"


class ShellGameEventRecurrentMemoryUpdater(nn.Module):
    """Add within-window position and apply one shared recurrent transition."""

    width: int = 64
    depth: int = 2
    num_heads: int = 4
    segment_size: int = WINDOW_SIZE
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, memory, event_segments):
        if (
            event_segments.ndim != 5
            or event_segments.shape[0] != memory.shape[0]
            or event_segments.shape[2] != self.segment_size
            or event_segments.shape[-1] != self.width
        ):
            raise ValueError(
                f"Expected event_segments [B,S,{self.segment_size},K,{self.width}], got {event_segments.shape}"
            )
        relative_pos = self.param(
            "relative_temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.segment_size, 1, self.width),
            event_segments.dtype,
        )
        evidence_steps = (event_segments + relative_pos[:, None]).reshape(
            event_segments.shape[0], event_segments.shape[1], -1, self.width
        )
        return memory_core.recurrently_update_memory(
            memory,
            evidence_steps,
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            dtype_mm=self.dtype_mm,
        )


class ShellGameSlidingWindowEventMemoryTracker(nn.Module):
    """Train on sampled windows or causally scan an entire ShellGame history."""

    num_frames: int = HISTORY_FRAMES
    window_size: int = WINDOW_SIZE
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
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(
        self,
        patch_tokens,
        initial_slots,
        window_starts,
        *,
        causal_selection: bool,
        train: bool = False,
    ):
        batch, frames, tokens, width = patch_tokens.shape
        expected = (self.num_frames, 256, self.input_width)
        if (frames, tokens, width) != expected:
            raise ValueError(f"Expected [B,{expected}], got {patch_tokens.shape}")
        if initial_slots.shape != (batch,):
            raise ValueError(f"Expected initial_slots [B], got {initial_slots.shape}")
        if window_starts.ndim != 2 or window_starts.shape[0] != batch:
            raise ValueError(f"Expected window_starts [B,W], got {window_starts.shape}")

        pooled = memory_core.pool_fixed_grid(patch_tokens, pool_factor=2)
        windows, window_starts = event_memory.extract_sliding_windows(
            pooled, window_size=self.window_size, starts=window_starts
        )
        num_candidates = windows.shape[1]
        flat_windows = windows.reshape(
            batch * num_candidates,
            self.window_size,
            semantic_memory.SPATIAL_TOKENS,
            self.input_width,
        )
        event_logits, relation_logits = ShellGameSlidingWindowEventRelationClassifier(
            name="window_classifier",
            input_width=self.input_width,
            width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            segment_size=self.window_size,
            dtype_mm=self.dtype_mm,
        )(flat_windows, train=train)
        event_logits = event_logits.reshape(batch, num_candidates)
        relation_logits = relation_logits.reshape(batch, num_candidates, semantic_memory.NUM_CUPS)

        if causal_selection:
            trigger_mask, event_active = event_memory.causal_event_triggers(event_logits)
            selected_positions, trigger_count = event_memory.first_trigger_positions(
                trigger_mask, max_events=NUM_STAGES
            )
        else:
            if num_candidates < NUM_STAGES:
                raise ValueError("Sampled-window training requires three positive candidates first")
            selected_positions = jnp.tile(jnp.arange(NUM_STAGES, dtype=jnp.int32)[None], (batch, 1))
            trigger_count = jnp.full((batch,), NUM_STAGES, dtype=jnp.int32)
            trigger_mask = jnp.zeros_like(event_logits, dtype=jnp.bool_).at[:, :NUM_STAGES].set(True)
            event_active = jnp.zeros((batch,), dtype=jnp.bool_)

        batch_axis = jnp.arange(batch, dtype=jnp.int32)[:, None]
        selected_relation_logits = relation_logits[batch_axis, selected_positions]
        relation_codes = jax.nn.softmax(selected_relation_logits, axis=-1).astype(jnp.float32)
        event_segments = event_memory.broadcast_event_codes(
            relation_codes,
            memory_width=self.memory_width,
            tokens_per_event=self.window_size * semantic_memory.SPATIAL_TOKENS,
        ).reshape(
            batch,
            NUM_STAGES,
            self.window_size,
            semantic_memory.SPATIAL_TOKENS,
            self.memory_width,
        )

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.memory_width),
            jnp.float32,
        )
        memory = jnp.tile(base_memory, (batch, 1, 1))
        initial_code = jax.nn.one_hot(initial_slots, semantic_memory.NUM_CUPS, dtype=jnp.float32)
        memory = memory.at[:, 0, : semantic_memory.NUM_CUPS].add(initial_code)
        _, stage_memories = ShellGameEventRecurrentMemoryUpdater(
            name="shared_swap_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            segment_size=self.window_size,
            dtype_mm="float32",
        )(memory, event_segments)

        adapter = memory_core.SingleHistoryReadAdapter(
            name="shared_history_read_adapter",
            memory_width=self.memory_width,
            current_width=self.current_width,
            num_heads=self.adapter_heads,
            residual_scale=self.residual_scale,
        )
        readout = semantic_memory.SharedMemoryTokenReadout(name="shared_readout", width=self.current_width)
        base_current_tokens = self.param(
            "base_current_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_current_tokens, self.current_width),
            jnp.float32,
        )
        current_tokens = jnp.tile(base_current_tokens, (batch, 1, 1))
        stage_logits = jnp.stack(
            [readout(adapter(current_tokens, stage_memories[:, stage_index])) for stage_index in range(NUM_STAGES)],
            axis=1,
        )
        selected_starts = jnp.take_along_axis(window_starts, selected_positions, axis=1)
        return {
            "event_logits": event_logits,
            "relation_logits": relation_logits,
            "selected_relation_logits": selected_relation_logits,
            "selected_positions": selected_positions,
            "selected_starts": selected_starts,
            "trigger_mask": trigger_mask,
            "trigger_count": trigger_count,
            "selection_valid": trigger_count == NUM_STAGES,
            "event_active": event_active,
            "stage_logits": stage_logits,
            "stage_memories": stage_memories,
        }


def causal_first_three_event_positions(event_logits):
    """ShellGame diagnostic view over the uncapped generic event stream."""
    trigger_mask, _ = event_memory.causal_event_triggers(event_logits)
    return event_memory.first_trigger_positions(trigger_mask, max_events=3)


def relation_probabilities_to_evidence(relation_logits, *, memory_width: int):
    """Build the validated soft relation-code evidence for one event stream."""
    relation_probabilities = jnp.softmax(relation_logits, axis=-1)
    return event_memory.broadcast_event_codes(
        relation_probabilities,
        memory_width=memory_width,
        tokens_per_event=6 * semantic_memory.SPATIAL_TOKENS,
    )
