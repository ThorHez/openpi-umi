"""Task-agnostic causal event memory for short overlapping visual windows.

The modules in this file extend :mod:`openpi.models.siglip_mem_semantic`
without changing its checkpoint contract.  They deliberately do not assume a
particular task, event count, video length, or event vocabulary:

* a short visual window is encoded into an event-presence logit and a
  configurable event-type distribution;
* overlapping windows are converted to causal triggers with a Schmitt-style
  high/low threshold; and
* only triggered windows update persistent recurrent memory.

Task adapters own event labels and the interpretation of event-type logits.
At streaming inference they should retain ``event_active`` together with the
memory tensor and feed both values into the next call.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import siglip_mem_semantic as memory_core


def extract_sliding_windows(
    sequence: jax.Array,
    *,
    window_size: int,
    stride: int = 1,
    starts: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Gather overlapping windows from a batch-major temporal sequence.

    Args:
        sequence: Tensor ``[B,T,...]``.
        window_size: Number of consecutive frames in each window.
        stride: Start-index stride when ``starts`` is omitted.
        starts: Optional integer indices ``[B,W]``.  This is useful for
            sampled-window training.  Callers are responsible for keeping
            dynamic indices in ``[0, T-window_size]``.

    Returns:
        Windows ``[B,W,window_size,...]`` and their starts ``[B,W]``.
    """
    if sequence.ndim < 3:
        raise ValueError(f"Expected sequence [B,T,...], got {sequence.shape}")
    if window_size < 1 or window_size > sequence.shape[1]:
        raise ValueError(f"window_size must be in [1,{sequence.shape[1]}], got {window_size}")
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")

    batch = sequence.shape[0]
    if starts is None:
        shared_starts = jnp.arange(
            0,
            sequence.shape[1] - window_size + 1,
            stride,
            dtype=jnp.int32,
        )
        starts = jnp.broadcast_to(shared_starts[None], (batch, shared_starts.shape[0]))
    else:
        if starts.ndim != 2 or starts.shape[0] != batch:
            raise ValueError(f"Expected starts [B,W] with B={batch}, got {starts.shape}")
        starts = starts.astype(jnp.int32)

    frame_offsets = jnp.arange(window_size, dtype=jnp.int32)
    frame_indices = starts[..., None] + frame_offsets
    batch_indices = jnp.arange(batch, dtype=jnp.int32)[:, None, None]
    return sequence[batch_indices, frame_indices], starts


def causal_event_triggers(
    event_logits: jax.Array,
    *,
    high_threshold: float = 0.0,
    low_threshold: float | None = None,
    previous_active: jax.Array | None = None,
    valid_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Turn chronological event evidence into causal, de-duplicated writes.

    A trigger is emitted only while the detector is inactive and the current
    logit crosses the high threshold.  It is re-armed after the logit reaches
    the low threshold.  Setting both thresholds to zero exactly recovers a
    binary rising-edge detector.

    Args:
        event_logits: Chronological logits ``[B,S]``.
        high_threshold: Inactive-to-active threshold in logit space.
        low_threshold: Active-to-inactive threshold. Defaults to
            ``high_threshold``; choose a lower value for hysteresis.
        previous_active: Detector state ``[B]`` from a previous stream chunk.
        valid_mask: Optional valid timestep mask ``[B,S]``. Invalid positions
            neither trigger nor change detector state.

    Returns:
        Boolean trigger mask ``[B,S]`` and final detector state ``[B]``.
    """
    if event_logits.ndim != 2:
        raise ValueError(f"Expected event_logits [B,S], got {event_logits.shape}")
    low_threshold = high_threshold if low_threshold is None else low_threshold
    if low_threshold > high_threshold:
        raise ValueError(f"low_threshold ({low_threshold}) cannot exceed high_threshold ({high_threshold})")
    batch, steps = event_logits.shape
    if previous_active is None:
        active = jnp.zeros((batch,), dtype=jnp.bool_)
    else:
        if previous_active.shape != (batch,):
            raise ValueError(f"Expected previous_active {(batch,)}, got {previous_active.shape}")
        active = previous_active.astype(jnp.bool_)
    if valid_mask is None:
        valid_mask = jnp.ones((batch, steps), dtype=jnp.bool_)
    elif valid_mask.shape != (batch, steps):
        raise ValueError(f"Expected valid_mask {(batch, steps)}, got {valid_mask.shape}")
    else:
        valid_mask = valid_mask.astype(jnp.bool_)

    triggers = []
    for step in range(steps):
        valid = valid_mask[:, step]
        score = event_logits[:, step]
        trigger = valid & ~active & (score > high_threshold)
        deactivate = active & (score <= low_threshold)
        next_active = jnp.where(deactivate, jnp.zeros_like(active), active | trigger)
        active = jnp.where(valid, next_active, active)
        triggers.append(trigger)
    return jnp.stack(triggers, axis=1), active


def first_trigger_positions(
    trigger_mask: jax.Array,
    *,
    max_events: int,
) -> tuple[jax.Array, jax.Array]:
    """Store the first ``max_events`` trigger positions and count all events.

    This helper is intended for fixed-shape diagnostics and supervised losses.
    The recurrent updater itself does not cap the number of events.
    """
    if trigger_mask.ndim != 2:
        raise ValueError(f"Expected trigger_mask [B,S], got {trigger_mask.shape}")
    if max_events < 1:
        raise ValueError(f"max_events must be positive, got {max_events}")
    batch, steps = trigger_mask.shape
    selected = jnp.zeros((batch, max_events), dtype=jnp.int32)
    count = jnp.zeros((batch,), dtype=jnp.int32)
    batch_axis = jnp.arange(batch, dtype=jnp.int32)
    for position in range(steps):
        trigger = trigger_mask[:, position]
        slot = jnp.minimum(count, max_events - 1)
        should_store = trigger & (count < max_events)
        old_value = selected[batch_axis, slot]
        selected = selected.at[batch_axis, slot].set(jnp.where(should_store, position, old_value))
        count += trigger.astype(jnp.int32)
    return selected, count


def broadcast_event_codes(
    event_codes: jax.Array,
    *,
    memory_width: int,
    tokens_per_event: int = 1,
) -> jax.Array:
    """Embed task-owned event codes into parameter-free memory evidence.

    The code occupies the leading memory channels and is repeated into a
    configurable number of evidence tokens.  This retains the validated
    soft-probability interface while keeping event semantics out of the core.
    """
    if event_codes.ndim != 3:
        raise ValueError(f"Expected event_codes [B,S,C], got {event_codes.shape}")
    if event_codes.shape[-1] > memory_width:
        raise ValueError(f"Event code width {event_codes.shape[-1]} exceeds memory_width={memory_width}")
    if tokens_per_event < 1:
        raise ValueError(f"tokens_per_event must be positive, got {tokens_per_event}")
    evidence = jnp.zeros(
        (*event_codes.shape[:2], tokens_per_event, memory_width),
        dtype=jnp.float32,
    )
    return evidence.at[..., : event_codes.shape[-1]].set(event_codes.astype(jnp.float32)[:, :, None, :])


class SlidingWindowEventClassifier(nn.Module):
    """Classify event presence and event identity from one short window."""

    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    segment_size: int = 6
    spatial_tokens: int = 64
    num_event_classes: int = 3
    event_type_head_name: str = "event_type_classifier"
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, window_tokens, *, train: bool = False, return_features: bool = False):
        if self.num_event_classes < 1:
            raise ValueError(f"num_event_classes must be positive, got {self.num_event_classes}")
        semantic = memory_core.FactorizedSpaceTimeEncoder(
            name="semantic_encoder",
            input_width=self.input_width,
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=self.segment_size,
            spatial_tokens=self.spatial_tokens,
            dtype_mm=self.dtype_mm,
        )(window_tokens, train=train)
        semantic = semantic.astype(jnp.float32)
        event_logits = nn.Dense(
            1,
            name="event_classifier",
            dtype=jnp.float32,
        )(semantic)[..., 0]
        event_type_logits = nn.Dense(
            self.num_event_classes,
            name=self.event_type_head_name,
            dtype=jnp.float32,
        )(semantic)
        if return_features:
            return event_logits, event_type_logits, semantic
        return event_logits, event_type_logits


class EventTriggeredRecurrentMemory(nn.Module):
    """Update persistent memory only at learned causal event triggers."""

    width: int = 64
    depth: int = 2
    num_heads: int = 4
    high_threshold: float = 0.0
    low_threshold: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(
        self,
        memory,
        evidence_steps,
        event_logits,
        *,
        previous_active=None,
        valid_mask=None,
    ):
        if event_logits.shape != evidence_steps.shape[:2]:
            raise ValueError(f"Expected event_logits {evidence_steps.shape[:2]}, got {event_logits.shape}")
        triggers, event_active = causal_event_triggers(
            event_logits,
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold,
            previous_active=previous_active,
            valid_mask=valid_mask,
        )
        final_memory, memory_states = memory_core.recurrently_update_memory(
            memory,
            evidence_steps,
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            dtype_mm=self.dtype_mm,
            step_mask=triggers,
        )
        return {
            "memory": final_memory,
            "memory_states": memory_states,
            "trigger_mask": triggers,
            "event_active": event_active,
            "trigger_count": jnp.sum(triggers, axis=1),
        }
