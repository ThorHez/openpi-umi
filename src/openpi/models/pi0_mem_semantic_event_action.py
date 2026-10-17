"""Task-agnostic action interface for causal event-driven memory.

This is the streaming counterpart of
:mod:`openpi.models.pi0_mem_semantic_action`.  It composes a causal recurrent
write with the existing raw-memory query resampler and action-token
cross-attention.  Task adapters still own visual encoding, event labels,
initial-memory construction, and the action loss.
"""

from __future__ import annotations

import flax.linen as nn

from openpi.models import pi0_mem_semantic_action as action_core
from openpi.models import siglip_mem_semantic_event as event_memory


class EventDrivenMemoryActionInterface(nn.Module):
    """Ingest new event evidence, then condition action tokens on memory."""

    memory_width: int = 64
    memory_tokens: int = 128
    memory_depth: int = 2
    memory_heads: int = 4
    event_high_threshold: float = 0.0
    event_low_threshold: float = 0.0
    query_width: int = 256
    query_tokens: int = 16
    query_depth: int = 2
    query_heads: int = 4
    action_width: int = 1024
    action_heads: int = 8
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(
        self,
        action_tokens,
        memory,
        evidence_steps,
        event_logits,
        *,
        previous_event_active=None,
        event_valid_mask=None,
    ):
        recurrent = event_memory.EventTriggeredRecurrentMemory(
            name="event_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            high_threshold=self.event_high_threshold,
            low_threshold=self.event_low_threshold,
            dtype_mm="float32",
        )(
            memory,
            evidence_steps,
            event_logits,
            previous_active=previous_event_active,
            valid_mask=event_valid_mask,
        )
        conditioned, resampled_memory = action_core.MemoryActionInterface(
            name="memory_action_interface",
            memory_width=self.memory_width,
            memory_tokens=self.memory_tokens,
            query_width=self.query_width,
            query_tokens=self.query_tokens,
            query_depth=self.query_depth,
            query_heads=self.query_heads,
            action_width=self.action_width,
            action_heads=self.action_heads,
            dtype_mm=self.dtype_mm,
        )(action_tokens, recurrent["memory"])
        return {
            **recurrent,
            "conditioned_action_tokens": conditioned,
            "resampled_memory_tokens": resampled_memory,
        }
