"""Goal-conditioned sliding-window event memory for RoboMME PickXtimes.

The generic model modules remain unaware of counting, gripper state, or the
PickXtimes event vocabulary.  This adapter assigns three event-completion
classes and decodes the recurrent state into task diagnostics.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip_mem_semantic as memory_core
from openpi.models import siglip_mem_semantic_event as event_memory
from openpi.models import siglip_mem_semantic_goal as goal_memory

WINDOW_SIZE = 10
SPATIAL_TOKENS = 64
NUM_EVENT_CLASSES = 3
NUM_COLORS = 3
MAX_REQUIRED_COUNT = 5
NUM_COUNT_CLASSES = MAX_REQUIRED_COUNT + 1
MAX_EVENTS = 2 * MAX_REQUIRED_COUNT + 1
NUM_GOAL_TOKENS = 2

PICK_COMPLETE = 0
PLACE_COMPLETE = 1
PRESS_COMPLETE = 2


def transition_grammar_events(
    gripper_closed,
    event_logits,
    event_type_logits,
    press_event_logits=None,
    *,
    required_count: int,
    pick_place_probability: float = 0.5,
    press_probability: float = 0.6,
    press_type_probability: float = 0.5,
    press_min_gap: int = 20,
    require_press_type: bool = True,
) -> tuple[list[int], list[int]]:
    """Decode a causal PickXtimes event sequence from gripper transitions.

    A close/open transition proposes PICK/PLACE candidates.  The strongest
    window whose transition lies at positions 4--8 is accepted only when its
    visual event probability passes the gate.  The finite-state grammar then
    enforces ``(PICK, PLACE) * required_count, PRESS``.  PRESS remains visual
    because it does not have a unique gripper transition.

    Selecting among the five transition-aligned windows adds at most eight
    frames of bounded causal latency; PRESS local-maximum confirmation adds
    one frame.
    """
    if not 1 <= int(required_count) <= MAX_REQUIRED_COUNT:
        raise ValueError(f"required_count must be in [1,{MAX_REQUIRED_COUNT}], got {required_count}")
    for name, probability in (
        ("pick_place_probability", pick_place_probability),
        ("press_probability", press_probability),
        ("press_type_probability", press_type_probability),
    ):
        if not 0.0 < probability < 1.0:
            raise ValueError(f"{name} must be in (0,1), got {probability}")
    if press_min_gap < 0:
        raise ValueError(f"press_min_gap must be nonnegative, got {press_min_gap}")

    gripper = np.asarray(gripper_closed, dtype=np.int8)
    gate_logits = np.asarray(event_logits, dtype=np.float32)
    type_logits = np.asarray(event_type_logits, dtype=np.float32)
    press_gate_logits = gate_logits if press_event_logits is None else np.asarray(press_event_logits, dtype=np.float32)
    if gripper.ndim != 1:
        raise ValueError(f"Expected gripper_closed [T], got {gripper.shape}")
    if gate_logits.ndim != 1:
        raise ValueError(f"Expected event_logits [K], got {gate_logits.shape}")
    if type_logits.shape != (gate_logits.shape[0], NUM_EVENT_CLASSES):
        raise ValueError(
            f"Expected event_type_logits {(gate_logits.shape[0], NUM_EVENT_CLASSES)}, got {type_logits.shape}"
        )
    if press_gate_logits.shape != gate_logits.shape:
        raise ValueError(f"Expected press_event_logits {gate_logits.shape}, got {press_gate_logits.shape}")
    if gripper.shape[0] != gate_logits.shape[0] + WINDOW_SIZE - 1:
        raise ValueError(
            "gripper timeline must be WINDOW_SIZE-1 longer than window logits, got "
            f"{gripper.shape[0]} and {gate_logits.shape[0]}"
        )

    pick_place_logit = float(np.log(pick_place_probability / (1.0 - pick_place_probability)))
    triggers: list[int] = []
    event_types: list[int] = []
    holding = False
    completed_count = 0
    transition_deltas = np.diff(gripper)
    for transition_index in np.flatnonzero(transition_deltas):
        transition_frame = int(transition_index + 1)
        proposed_type = PICK_COMPLETE if transition_deltas[transition_index] > 0 else PLACE_COMPLETE
        expected_type = PLACE_COMPLETE if holding else PICK_COMPLETE
        if completed_count >= required_count or proposed_type != expected_type:
            continue
        candidates = [
            transition_frame - position
            for position in range(4, 9)
            if 0 <= transition_frame - position < gate_logits.shape[0]
        ]
        if not candidates:
            continue
        selected = max(candidates, key=lambda index: float(gate_logits[index]))
        if float(gate_logits[selected]) < pick_place_logit:
            continue
        triggers.append(selected)
        event_types.append(proposed_type)
        if proposed_type == PICK_COMPLETE:
            holding = True
        else:
            holding = False
            completed_count += 1
            if completed_count == required_count:
                break

    if completed_count != required_count:
        return triggers, event_types

    press_logit = float(np.log(press_probability / (1.0 - press_probability)))
    shifted = type_logits - np.max(type_logits, axis=-1, keepdims=True)
    type_probabilities = np.exp(shifted)
    type_probabilities /= np.sum(type_probabilities, axis=-1, keepdims=True)
    first_press_start = max(triggers[-1] + press_min_gap, 1)
    for start in range(first_press_start, press_gate_logits.shape[0] - 1):
        is_local_maximum = (
            press_gate_logits[start] >= press_gate_logits[start - 1]
            and press_gate_logits[start] >= press_gate_logits[start + 1]
        )
        if (
            press_gate_logits[start] >= press_logit
            and (
                not require_press_type
                or (
                    type_probabilities[start, PRESS_COMPLETE] >= press_type_probability
                    and int(np.argmax(type_logits[start])) == PRESS_COMPLETE
                )
            )
            and is_local_maximum
        ):
            triggers.append(start)
            event_types.append(PRESS_COMPLETE)
            break
    return triggers, event_types


class PickXtimesSlidingWindowEventClassifier(event_memory.SlidingWindowEventClassifier):
    """Recognize complete PICK, PLACE, and PRESS transitions."""

    segment_size: int = WINDOW_SIZE
    spatial_tokens: int = SPATIAL_TOKENS
    num_event_classes: int = NUM_EVENT_CLASSES


class PickXtimesGripperTypeFusion(nn.Module):
    """Fuse gripper transitions into type logits without changing the gate."""

    @nn.compact
    def __call__(self, visual_type_logits, gripper_closed):
        if gripper_closed.ndim != 2 or gripper_closed.shape[1] != WINDOW_SIZE:
            raise ValueError(f"Expected gripper_closed [B,{WINDOW_SIZE}], got {gripper_closed.shape}")
        if visual_type_logits.shape != (gripper_closed.shape[0], NUM_EVENT_CLASSES):
            raise ValueError(
                f"Expected visual_type_logits {(gripper_closed.shape[0], NUM_EVENT_CLASSES)}, "
                f"got {visual_type_logits.shape}"
            )
        state = gripper_closed.astype(jnp.float32)
        delta = state[:, 1:] - state[:, :-1]
        transitions = jnp.stack(
            (
                jnp.sum(jax.nn.relu(delta), axis=1),
                jnp.sum(jax.nn.relu(-delta), axis=1),
            ),
            axis=-1,
        )
        # No bias: a window without a gripper transition receives exactly the
        # old visual logits. This avoids turning causal false triggers into
        # artificial PRESS events merely because the gripper stayed stable.
        residual = nn.Dense(
            NUM_EVENT_CLASSES,
            name="type_residual",
            use_bias=False,
            kernel_init=nn.initializers.zeros_init(),
        )(transitions)
        return visual_type_logits.astype(jnp.float32) + residual


class PickXtimesGripperGateFusion(nn.Module):
    """Add position-aware gripper transitions to the visual event gate."""

    @nn.compact
    def __call__(self, visual_event_logits, gripper_closed):
        if gripper_closed.ndim != 2 or gripper_closed.shape[1] != WINDOW_SIZE:
            raise ValueError(f"Expected gripper_closed [B,{WINDOW_SIZE}], got {gripper_closed.shape}")
        if visual_event_logits.shape != (gripper_closed.shape[0],):
            raise ValueError(
                f"Expected visual_event_logits {(gripper_closed.shape[0],)}, got {visual_event_logits.shape}"
            )
        state = gripper_closed.astype(jnp.float32)
        delta = state[:, 1:] - state[:, :-1]
        transition_positions = jnp.concatenate((jax.nn.relu(delta), jax.nn.relu(-delta)), axis=-1)
        # Bias-free zero initialization preserves old checkpoints exactly and
        # leaves transition-free windows, including most PRESS windows, alone.
        residual = nn.Dense(
            1,
            name="gate_residual",
            use_bias=False,
            kernel_init=nn.initializers.zeros_init(),
        )(transition_positions)[..., 0]
        return visual_event_logits.astype(jnp.float32) + residual


class PickXtimesPressGateFusion(nn.Module):
    """Add a visual residual used only for goal-gated PRESS detection."""

    @nn.compact
    def __call__(self, visual_event_logits, semantic_features):
        if visual_event_logits.ndim != 1:
            raise ValueError(f"Expected visual_event_logits [B], got {visual_event_logits.shape}")
        if semantic_features.ndim != 2 or semantic_features.shape[0] != visual_event_logits.shape[0]:
            raise ValueError(
                f"Expected semantic_features [B,D] with B={visual_event_logits.shape[0]}, got {semantic_features.shape}"
            )
        # This adapter is intentionally trained on top of a frozen detector.
        # Stop input gradients here so focused PRESS training does not retain
        # the full space-time encoder backward graph for every hard negative.
        frozen_event_logits = jax.lax.stop_gradient(visual_event_logits.astype(jnp.float32))
        frozen_semantic_features = jax.lax.stop_gradient(semantic_features.astype(jnp.float32))
        residual = nn.Dense(
            1,
            name="press_residual",
            dtype=jnp.float32,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(frozen_semantic_features)[..., 0]
        return frozen_event_logits + residual


class PickXtimesMemoryStateReadout(nn.Module):
    """Decode task diagnostics after every recurrent event update."""

    width: int = 64

    @nn.compact
    def __call__(self, stage_memories):
        if stage_memories.ndim != 4 or stage_memories.shape[-1] != self.width:
            raise ValueError(f"Expected stage_memories [B,S,M,{self.width}], got {stage_memories.shape}")
        x = nn.LayerNorm(name="memory_ln", dtype=jnp.float32)(stage_memories)
        scores = nn.Dense(1, name="attention", dtype=jnp.float32)(x)
        weights = nn.softmax(scores, axis=2)
        pooled = jnp.sum(weights * x, axis=2)
        pooled = nn.LayerNorm(name="pooled_ln", dtype=jnp.float32)(pooled)

        def categorical(name: str, classes: int):
            return nn.Dense(classes, name=name, dtype=jnp.float32)(pooled)

        def binary(name: str):
            return nn.Dense(1, name=name, dtype=jnp.float32)(pooled)[..., 0]

        return {
            "completed_count_logits": categorical("completed_count_head", NUM_COUNT_CLASSES),
            "remaining_count_logits": categorical("remaining_count_head", NUM_COUNT_CLASSES),
            "holding_logits": binary("holding_head"),
            "should_press_logits": binary("should_press_head"),
            "done_logits": binary("done_head"),
            "next_event_logits": categorical("next_event_head", NUM_EVENT_CLASSES),
        }


class PickXtimesGoalReadout(nn.Module):
    """Bind the two learned goal queries to color and repetition semantics."""

    width: int = 64

    @nn.compact
    def __call__(self, goal_tokens):
        expected = (NUM_GOAL_TOKENS, self.width)
        if goal_tokens.ndim != 3 or goal_tokens.shape[1:] != expected:
            raise ValueError(f"Expected goal_tokens [B,{expected}], got {goal_tokens.shape}")
        color_token = nn.LayerNorm(name="color_ln", dtype=jnp.float32)(goal_tokens[:, 0])
        count_token = nn.LayerNorm(name="count_ln", dtype=jnp.float32)(goal_tokens[:, 1])
        return {
            "goal_color_logits": nn.Dense(NUM_COLORS, name="color_head", dtype=jnp.float32)(color_token),
            # Labels are zero-based required counts: 1 -> 0, ..., 5 -> 4.
            "goal_required_count_logits": nn.Dense(
                MAX_REQUIRED_COUNT,
                name="required_count_head",
                dtype=jnp.float32,
            )(count_token),
        }


class PickXtimesSlidingWindowEventMemoryTracker(nn.Module):
    """Train on sampled windows or causally scan chronological candidates.

    Args:
        window_patch_tokens: SigLIP patch tokens ``[B,K,W,256,1152]``.
        window_gripper_closed: Boolean gripper state ``[B,K,W]``.
        prompt_tokens: PaliGemma prompt embeddings ``[B,L,2048]``.
        prompt_mask: Valid language mask ``[B,L]``.
        sequence_positions: Positions ``[B,S]`` of chronological positive
            windows during teacher-window training. Ignored in causal mode.
        sequence_mask: Valid event mask ``[B,S]`` during teacher-window
            training. Ignored in causal mode.
        causal_selection: If true, use predicted hysteresis triggers and keep
            the first ``MAX_EVENTS`` only for fixed-shape diagnostics.
    """

    input_width: int = 1152
    prompt_width: int = 2048
    encoder_width: int = 256
    encoder_depth: int = 2
    encoder_heads: int = 8
    memory_width: int = 64
    memory_depth: int = 2
    memory_heads: int = 4
    num_memory_tokens: int = 128
    evidence_tokens_per_event: int = WINDOW_SIZE * SPATIAL_TOKENS
    high_threshold: float = 0.8472978604
    low_threshold: float = -0.8472978604
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(
        self,
        window_patch_tokens,
        window_gripper_closed,
        prompt_tokens,
        prompt_mask,
        sequence_positions,
        sequence_mask,
        *,
        causal_selection: bool,
        candidate_valid_mask=None,
        previous_active=None,
        sequence_event_types=None,
        train: bool = False,
    ):
        if window_patch_tokens.ndim != 5:
            raise ValueError(f"Expected window_patch_tokens [B,K,W,N,D], got {window_patch_tokens.shape}")
        batch, candidates, frames, patches, width = window_patch_tokens.shape
        expected = (WINDOW_SIZE, 256, self.input_width)
        if (frames, patches, width) != expected:
            raise ValueError(f"Expected window suffix {expected}, got {(frames, patches, width)}")
        if window_gripper_closed.shape != (batch, candidates, WINDOW_SIZE):
            raise ValueError(
                f"Expected window_gripper_closed {(batch, candidates, WINDOW_SIZE)}, got {window_gripper_closed.shape}"
            )

        flat = window_patch_tokens.reshape(batch * candidates, frames, patches, width)
        pooled = memory_core.pool_fixed_grid(flat, pool_factor=2)
        event_logits, event_type_logits, semantic_features = PickXtimesSlidingWindowEventClassifier(
            name="window_classifier",
            input_width=self.input_width,
            width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            segment_size=WINDOW_SIZE,
            dtype_mm=self.dtype_mm,
        )(pooled, train=train, return_features=True)
        event_logits = event_logits.reshape(batch, candidates)
        event_type_logits = event_type_logits.reshape(batch, candidates, NUM_EVENT_CLASSES)
        event_logits = PickXtimesGripperGateFusion(name="gripper_gate_fusion")(
            event_logits.reshape(batch * candidates),
            window_gripper_closed.reshape(batch * candidates, WINDOW_SIZE),
        ).reshape(batch, candidates)
        press_event_logits = PickXtimesPressGateFusion(name="press_gate_fusion")(
            event_logits.reshape(batch * candidates),
            semantic_features,
        ).reshape(batch, candidates)
        event_type_logits = PickXtimesGripperTypeFusion(name="gripper_type_fusion")(
            event_type_logits.reshape(batch * candidates, NUM_EVENT_CLASSES),
            window_gripper_closed.reshape(batch * candidates, WINDOW_SIZE),
        ).reshape(batch, candidates, NUM_EVENT_CLASSES)

        if causal_selection:
            trigger_mask, event_active = event_memory.causal_event_triggers(
                event_logits,
                high_threshold=self.high_threshold,
                low_threshold=self.low_threshold,
                previous_active=previous_active,
                valid_mask=candidate_valid_mask,
            )
            sequence_positions, trigger_count = event_memory.first_trigger_positions(
                trigger_mask,
                max_events=MAX_EVENTS,
            )
            sequence_mask = jnp.arange(MAX_EVENTS)[None] < jnp.minimum(trigger_count[:, None], MAX_EVENTS)
        else:
            if sequence_positions.ndim != 2 or sequence_positions.shape[0] != batch:
                raise ValueError(f"Expected sequence_positions [B,S], got {sequence_positions.shape}")
            if sequence_positions.shape != sequence_mask.shape:
                raise ValueError(
                    f"sequence_positions and sequence_mask must match, got "
                    f"{sequence_positions.shape} and {sequence_mask.shape}"
                )
            trigger_mask = jnp.zeros((batch, candidates), dtype=jnp.bool_)
            trigger_count = jnp.sum(sequence_mask, axis=1)
            event_active = jnp.zeros((batch,), dtype=jnp.bool_)

        batch_axis = jnp.arange(batch, dtype=jnp.int32)[:, None]
        selected_type_logits = event_type_logits[batch_axis, sequence_positions]
        if sequence_event_types is None:
            event_codes = jax.nn.softmax(selected_type_logits, axis=-1).astype(jnp.float32)
        else:
            if sequence_event_types.shape != sequence_positions.shape:
                raise ValueError(
                    "sequence_event_types must match sequence_positions, got "
                    f"{sequence_event_types.shape} and {sequence_positions.shape}"
                )
            event_codes = jax.nn.one_hot(sequence_event_types, NUM_EVENT_CLASSES, dtype=jnp.float32)
        evidence_steps = event_memory.broadcast_event_codes(
            event_codes,
            memory_width=self.memory_width,
            tokens_per_event=self.evidence_tokens_per_event,
        )

        final_memory, stage_memories, goal_tokens, initial_memory = goal_memory.GoalConditionedRecurrentMemory(
            name="goal_conditioned_recurrent_memory",
            prompt_width=self.prompt_width,
            memory_width=self.memory_width,
            num_memory_tokens=self.num_memory_tokens,
            num_goal_tokens=NUM_GOAL_TOKENS,
            goal_heads=4,
            memory_depth=self.memory_depth,
            memory_heads=self.memory_heads,
            dtype_mm="float32",
        )(
            prompt_tokens,
            evidence_steps,
            prompt_mask=prompt_mask,
            step_mask=sequence_mask,
        )
        state_outputs = PickXtimesMemoryStateReadout(
            name="memory_state_readout",
            width=self.memory_width,
        )(stage_memories)
        goal_outputs = PickXtimesGoalReadout(
            name="goal_readout",
            width=self.memory_width,
        )(goal_tokens)
        return {
            "event_logits": event_logits,
            "press_event_logits": press_event_logits,
            "event_type_logits": event_type_logits,
            "selected_event_type_logits": selected_type_logits,
            "sequence_positions": sequence_positions,
            "sequence_mask": sequence_mask,
            "trigger_mask": trigger_mask,
            "trigger_count": trigger_count,
            "event_active": event_active,
            "goal_tokens": goal_tokens,
            "initial_memory": initial_memory,
            "stage_memories": stage_memories,
            "memory": final_memory,
            **goal_outputs,
            **state_outputs,
        }
