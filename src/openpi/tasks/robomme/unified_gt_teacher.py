"""Unified GT-event recurrent teacher for the four RoboMME memory tasks.

The teacher is deliberately independent of Qwen and pixels.  It receives a
goal description encoded as compact categorical fields plus an ordered GT
event sequence.  One shared recurrent updater produces fixed-size latent
memory, and one shared query readout decodes every task state.  Task-specific
semantics are expressed only through loss masks, never through routed heads.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import siglip_mem_semantic as memory_core

TASKS = (
    "videounmask_variable_demo",
    "videounmaskswap_local_event",
    "videoplaceorder_local_event",
    "pickxtimes_local_event",
)

# Zero is reserved for padding/no-op.  The three rejection classes are valid
# teacher inputs but are hard-gated to preserve memory exactly.
EVENTS = (
    "padding",
    "target_visible",
    "target_covered",
    "swap_complete",
    "pick_complete",
    "place_complete",
    "press_complete",
    "no_completed_event",
    "incomplete_event",
    "insufficient_evidence",
)
STATE_CHANGING_EVENT_COUNT = 6

COLORS = ("none", "red", "green", "blue")
REGIONS = ("none", "region_0", "region_1", "region_2", "region_3")

STATE_FIELDS = (
    "task",
    "target_color_0",
    "target_color_1",
    "red_cell",
    "green_cell",
    "blue_cell",
    "ordered_cell_0",
    "ordered_cell_1",
    "ordered_cell_2",
    "ordered_cell_3",
    "covered",
    "completed_swap_count",
    "written_count",
    "required_count",
    "completed_count",
    "holding",
    "ready_to_press",
    "done",
    "queried_ordinal",
)

# A single six-class projection is shared by all fields.  Invalid tail classes
# are masked before loss and argmax.
FIELD_CLASS_COUNTS = (
    4,  # task
    4,  # target_color_0: none/red/green/blue
    4,  # target_color_1
    5,  # red_cell: none + four local regions
    5,  # green_cell
    5,  # blue_cell
    5,  # ordered_cell_0
    5,  # ordered_cell_1
    5,  # ordered_cell_2
    5,  # ordered_cell_3
    2,  # covered
    4,  # completed_swap_count: 0..3
    5,  # written_count: 0..4
    6,  # required_count: 0..5
    6,  # completed_count: 0..5
    2,  # holding
    2,  # ready_to_press
    2,  # done
    5,  # queried_ordinal: 0..4
)
MAX_FIELD_CLASSES = max(FIELD_CLASS_COUNTS)


def compute_teacher_losses(
    outputs: dict[str, jnp.ndarray],
    state_targets: jnp.ndarray,
    state_field_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute episode-balanced state reconstruction loss and strict metrics."""

    logits = outputs["state_logits"].astype(jnp.float32)
    expected = logits.shape[:-1]
    if state_targets.shape != expected or state_field_mask.shape != expected:
        raise ValueError(
            f"Expected targets/mask {expected}, got {state_targets.shape} and "
            f"{state_field_mask.shape}"
        )
    mask = state_field_mask.astype(jnp.float32)
    token_losses = -jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1),
        state_targets[..., None],
        axis=-1,
    )[..., 0]
    fields_per_state = jnp.sum(mask, axis=-1)
    state_valid = fields_per_state > 0
    losses_per_state = jnp.sum(token_losses * mask, axis=-1) / jnp.maximum(fields_per_state, 1.0)
    states_per_episode = jnp.sum(state_valid, axis=-1)
    losses_per_episode = jnp.sum(losses_per_state * state_valid, axis=-1) / jnp.maximum(
        states_per_episode, 1
    )
    rollout_loss = jnp.mean(losses_per_episode)

    predictions = jnp.argmax(logits, axis=-1)
    correct_or_masked = (predictions == state_targets) | ~state_field_mask
    state_exact = jnp.all(correct_or_masked, axis=-1) & state_valid
    state_accuracy = jnp.sum(state_exact) / jnp.maximum(jnp.sum(state_valid), 1)
    sequence_exact = jnp.all(state_exact | ~state_valid, axis=-1)
    final_indices = jnp.maximum(states_per_episode.astype(jnp.int32) - 1, 0)
    final_exact = state_exact[jnp.arange(state_exact.shape[0]), final_indices]
    field_accuracy = jnp.sum((predictions == state_targets) * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    metrics = {
        "loss": rollout_loss,
        "rollout_state_loss": rollout_loss,
        "field_accuracy": field_accuracy,
        "state_exact_accuracy": state_accuracy,
        "sequence_exact_accuracy": jnp.mean(sequence_exact),
        "final_state_exact_accuracy": jnp.mean(final_exact),
        "memory_token_variance": jnp.var(outputs["all_memories"].astype(jnp.float32)),
    }
    for field_index, field_name in enumerate(STATE_FIELDS):
        field_mask = mask[..., field_index]
        metrics[f"field/{field_name}_accuracy"] = jnp.sum(
            (predictions[..., field_index] == state_targets[..., field_index]) * field_mask
        ) / jnp.maximum(jnp.sum(field_mask), 1.0)
    if "gt_state_logits" not in outputs:
        return rollout_loss, metrics

    canonical_outputs = {
        "state_logits": outputs["gt_state_logits"],
        "all_memories": outputs["gt_state_memories"],
    }
    canonical_loss, canonical_metrics = compute_teacher_losses(
        canonical_outputs,
        state_targets,
        state_field_mask,
    )
    rollout_memory = outputs["all_memories"].astype(jnp.float32)
    canonical_memory = jax.lax.stop_gradient(outputs["gt_state_memories"].astype(jnp.float32))
    valid_state = jnp.any(state_field_mask, axis=-1).astype(jnp.float32)
    rollout_normalized = rollout_memory / jnp.maximum(
        jnp.linalg.norm(rollout_memory, axis=-1, keepdims=True), 1e-6
    )
    canonical_normalized = canonical_memory / jnp.maximum(
        jnp.linalg.norm(canonical_memory, axis=-1, keepdims=True), 1e-6
    )
    cosine = 1.0 - jnp.sum(
        rollout_normalized * canonical_normalized,
        axis=-1,
    )
    cosine_loss = jnp.sum(cosine * valid_state[..., None]) / jnp.maximum(
        jnp.sum(valid_state) * rollout_memory.shape[-2], 1.0
    )
    mse_per_token = jnp.mean(jnp.square(rollout_memory - canonical_memory), axis=-1)
    mse_loss = jnp.sum(mse_per_token * valid_state[..., None]) / jnp.maximum(
        jnp.sum(valid_state) * rollout_memory.shape[-2], 1.0
    )
    memory_alignment_loss = cosine_loss + 0.1 * mse_loss
    loss = rollout_loss + canonical_loss + memory_alignment_loss
    metrics.update(
        {
            "loss": loss,
            "canonical_state_loss": canonical_loss,
            "canonical_field_accuracy": canonical_metrics["field_accuracy"],
            "canonical_state_exact_accuracy": canonical_metrics["state_exact_accuracy"],
            "canonical_sequence_exact_accuracy": canonical_metrics["sequence_exact_accuracy"],
            "canonical_final_state_exact_accuracy": canonical_metrics[
                "final_state_exact_accuracy"
            ],
            "memory_alignment_loss": memory_alignment_loss,
            "memory_cosine_loss": cosine_loss,
            "memory_mse_loss": mse_loss,
        }
    )
    return loss, metrics


def mask_invalid_field_classes(logits: jnp.ndarray) -> jnp.ndarray:
    """Mask class columns that do not exist for a state field."""

    if logits.ndim < 2 or logits.shape[-2:] != (len(STATE_FIELDS), MAX_FIELD_CLASSES):
        raise ValueError(
            "Expected logits [...,"
            f"{len(STATE_FIELDS)},{MAX_FIELD_CLASSES}], got {logits.shape}"
        )
    class_ids = jnp.arange(MAX_FIELD_CLASSES)[None, :]
    valid = class_ids < jnp.asarray(FIELD_CLASS_COUNTS)[:, None]
    return jnp.where(valid, logits, jnp.asarray(-1e9, dtype=logits.dtype))


class UnifiedStateReadout(nn.Module):
    """Decode all task states with shared field queries and one classifier."""

    width: int = 64
    num_heads: int = 4
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, memory: jnp.ndarray) -> jnp.ndarray:
        if memory.ndim != 3 or memory.shape[-1] != self.width:
            raise ValueError(f"Expected memory [B,M,{self.width}], got {memory.shape}")
        queries = self.param(
            "field_queries",
            nn.initializers.normal(stddev=0.02),
            (1, len(STATE_FIELDS), self.width),
            jnp.float32,
        )
        queries = jnp.tile(queries, (memory.shape[0], 1, 1))
        query_norm = nn.LayerNorm(name="query_ln", dtype=self.dtype_mm)(queries)
        memory_norm = nn.LayerNorm(name="memory_ln", dtype=self.dtype_mm)(memory)
        attended = nn.MultiHeadDotProductAttention(
            name="memory_cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(query_norm, memory_norm)
        fields = nn.LayerNorm(name="field_output_ln", dtype=self.dtype_mm)(queries + attended)
        logits = nn.Dense(
            MAX_FIELD_CLASSES,
            name="shared_field_classifier",
            dtype=jnp.float32,
        )(fields.astype(jnp.float32))
        return mask_invalid_field_classes(logits)


class GTStateMemoryEncoder(nn.Module):
    """Encode a masked structured GT state into the canonical teacher memory."""

    width: int = 64
    num_memory_tokens: int = 128
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(
        self,
        state_targets: jnp.ndarray,
        state_field_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        if state_targets.shape != state_field_mask.shape or state_targets.ndim != 3:
            raise ValueError(
                "Expected state targets/mask [B,S,F] with matching shapes, got "
                f"{state_targets.shape} and {state_field_mask.shape}"
            )
        if state_targets.shape[-1] != len(STATE_FIELDS):
            raise ValueError(f"Expected {len(STATE_FIELDS)} state fields, got {state_targets.shape}")
        if self.num_memory_tokens < len(STATE_FIELDS):
            raise ValueError(
                f"num_memory_tokens={self.num_memory_tokens} must cover {len(STATE_FIELDS)} fields"
            )
        batch, states, _ = state_targets.shape
        field_identity = self.param(
            "field_identity",
            nn.initializers.normal(stddev=0.02),
            (1, 1, len(STATE_FIELDS), self.width),
            jnp.float32,
        )
        class_embedding = nn.Embed(
            MAX_FIELD_CLASSES,
            self.width,
            name="state_class_embedding",
            dtype=self.dtype_mm,
        )(state_targets)
        inactive = self.param(
            "inactive_field_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, 1, self.width),
            jnp.float32,
        )
        active_tokens = field_identity + class_embedding
        inactive_tokens = field_identity + inactive
        field_tokens = jnp.where(
            state_field_mask[..., None],
            active_tokens,
            inactive_tokens,
        )
        remaining = self.num_memory_tokens - len(STATE_FIELDS)
        if remaining:
            base = self.param(
                "canonical_base_memory",
                nn.initializers.normal(stddev=0.02),
                (1, 1, remaining, self.width),
                jnp.float32,
            )
            base = jnp.tile(base, (batch, states, 1, 1))
            memory = jnp.concatenate((field_tokens, base), axis=2)
        else:
            memory = field_tokens
        return nn.LayerNorm(name="canonical_memory_ln", dtype=self.dtype_mm)(memory)


class UnifiedRoboMMEGTTeacher(nn.Module):
    """Map a GT goal/event sequence to recurrent latent state and readouts."""

    width: int = 64
    num_memory_tokens: int = 128
    memory_depth: int = 2
    memory_heads: int = 4
    readout_heads: int = 4
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(
        self,
        task_ids: jnp.ndarray,
        goal_color_ids: jnp.ndarray,
        required_counts: jnp.ndarray,
        queried_ordinals: jnp.ndarray,
        num_regions: jnp.ndarray,
        event_ids: jnp.ndarray,
        entity_ids: jnp.ndarray,
        region_a_ids: jnp.ndarray,
        region_b_ids: jnp.ndarray,
        step_mask: jnp.ndarray,
        teacher_state_targets: jnp.ndarray | None = None,
        teacher_state_field_mask: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        batch = task_ids.shape[0]
        steps = event_ids.shape[1]
        expected_vectors = {
            "required_counts": required_counts,
            "queried_ordinals": queried_ordinals,
            "num_regions": num_regions,
        }
        if task_ids.shape != (batch,):
            raise ValueError(f"Expected task_ids [B], got {task_ids.shape}")
        if goal_color_ids.shape != (batch, 2):
            raise ValueError(f"Expected goal_color_ids [B,2], got {goal_color_ids.shape}")
        for name, value in expected_vectors.items():
            if value.shape != (batch,):
                raise ValueError(f"Expected {name} [B], got {value.shape}")
        for name, value in {
            "entity_ids": entity_ids,
            "region_a_ids": region_a_ids,
            "region_b_ids": region_b_ids,
            "step_mask": step_mask,
        }.items():
            if value.shape != (batch, steps):
                raise ValueError(f"Expected {name} {(batch, steps)}, got {value.shape}")

        task_embed = nn.Embed(len(TASKS), self.width, name="task_embedding", dtype=self.dtype_mm)
        color_embed = nn.Embed(len(COLORS), self.width, name="color_embedding", dtype=self.dtype_mm)
        count_embed = nn.Embed(6, self.width, name="count_embedding", dtype=self.dtype_mm)
        ordinal_embed = nn.Embed(5, self.width, name="ordinal_embedding", dtype=self.dtype_mm)
        num_regions_embed = nn.Embed(5, self.width, name="num_regions_embedding", dtype=self.dtype_mm)
        event_embed = nn.Embed(len(EVENTS), self.width, name="event_embedding", dtype=self.dtype_mm)
        entity_embed = nn.Embed(len(COLORS), self.width, name="entity_embedding", dtype=self.dtype_mm)
        region_embed = nn.Embed(len(REGIONS), self.width, name="region_embedding", dtype=self.dtype_mm)

        goal_type = self.param(
            "goal_token_type",
            nn.initializers.normal(stddev=0.02),
            (1, 6, self.width),
            jnp.float32,
        )
        goal_tokens = jnp.stack(
            (
                task_embed(task_ids),
                color_embed(goal_color_ids[:, 0]),
                color_embed(goal_color_ids[:, 1]),
                count_embed(required_counts),
                ordinal_embed(queried_ordinals),
                num_regions_embed(num_regions),
            ),
            axis=1,
        )
        goal_tokens = goal_tokens + goal_type.astype(goal_tokens.dtype)

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.width),
            jnp.float32,
        )
        base_memory = jnp.tile(base_memory, (batch, 1, 1))
        initial_memory, _ = memory_core.RecurrentMemoryUpdater(
            name="goal_memory_initializer",
            width=self.width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            dtype_mm=self.dtype_mm,
        )(base_memory, goal_tokens[:, None, :, :])

        event_type = self.param(
            "event_token_type",
            nn.initializers.normal(stddev=0.02),
            (1, 1, 4, self.width),
            jnp.float32,
        )
        event_tokens = jnp.stack(
            (
                event_embed(event_ids),
                entity_embed(entity_ids),
                region_embed(region_a_ids),
                region_embed(region_b_ids),
            ),
            axis=2,
        )
        event_tokens = event_tokens + event_type.astype(event_tokens.dtype)
        state_changing = (event_ids >= 1) & (event_ids <= STATE_CHANGING_EVENT_COUNT)
        update_mask = step_mask.astype(jnp.bool_) & state_changing
        final_memory, event_memories = memory_core.RecurrentMemoryUpdater(
            name="event_memory_updater",
            width=self.width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            dtype_mm=self.dtype_mm,
        )(initial_memory, event_tokens, step_mask=update_mask)

        all_memories = jnp.concatenate((initial_memory[:, None], event_memories), axis=1)
        flat = all_memories.reshape(batch * (steps + 1), self.num_memory_tokens, self.width)
        readout = UnifiedStateReadout(
            name="unified_state_readout",
            width=self.width,
            num_heads=self.readout_heads,
            dtype_mm=self.dtype_mm,
        )
        flat_logits = readout(flat)
        state_logits = flat_logits.reshape(
            batch,
            steps + 1,
            len(STATE_FIELDS),
            MAX_FIELD_CLASSES,
        )
        outputs = {
            "initial_memory": initial_memory,
            "event_memories": event_memories,
            "all_memories": all_memories,
            "final_memory": final_memory,
            "state_logits": state_logits,
            "update_mask": update_mask,
        }
        if (teacher_state_targets is None) != (teacher_state_field_mask is None):
            raise ValueError("teacher state targets and field mask must be supplied together")
        if teacher_state_targets is not None and teacher_state_field_mask is not None:
            expected_states = (batch, steps + 1, len(STATE_FIELDS))
            if (
                teacher_state_targets.shape != expected_states
                or teacher_state_field_mask.shape != expected_states
            ):
                raise ValueError(
                    f"Expected teacher states {expected_states}, got {teacher_state_targets.shape} "
                    f"and {teacher_state_field_mask.shape}"
                )
            gt_state_memories = GTStateMemoryEncoder(
                name="gt_state_memory_encoder",
                width=self.width,
                num_memory_tokens=self.num_memory_tokens,
                dtype_mm=self.dtype_mm,
            )(teacher_state_targets, teacher_state_field_mask)
            gt_flat = gt_state_memories.reshape(
                batch * (steps + 1), self.num_memory_tokens, self.width
            )
            gt_state_logits = readout(gt_flat).reshape(
                batch,
                steps + 1,
                len(STATE_FIELDS),
                MAX_FIELD_CLASSES,
            )
            outputs.update(
                {
                    "gt_state_memories": gt_state_memories,
                    "gt_state_logits": gt_state_logits,
                }
            )
        return outputs
