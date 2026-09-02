"""Ceiling-decomposed recurrent semantic region memory for RoboMME."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.tasks.robomme import unified_gt_teacher as contract
from openpi.tasks.robomme.unified_visual_student import VisualWindowEncoder


SWAP_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _straight_through_one_hot(logits: jnp.ndarray) -> jnp.ndarray:
    soft = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    hard = jax.nn.one_hot(jnp.argmax(logits, axis=-1), logits.shape[-1])
    return soft + jax.lax.stop_gradient(hard - soft)


class DecomposedRegionRecurrentMemory(nn.Module):
    """Predict local semantic operations and recurrently update a region table.

    The recurrent state is a 7x5 semantic table: red/green/blue and four
    ordinal entities, each storing none or one of four episode-local regions.
    It is deterministically encoded into the first seven tokens of a 128x64
    latent memory so the action-facing contract remains latent-token based.
    """

    max_steps: int = 96
    frames: int = 12
    spatial_tokens: int = 16
    input_width: int = 1152
    width: int = 64
    memory_tokens: int = 128
    encoder_width: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 8
    micro_events: int = 2
    gate_temperature: float = 0.25

    @nn.compact
    def __call__(
        self,
        patch_tokens: jnp.ndarray,
        sequence_mask: jnp.ndarray,
        task_ids: jnp.ndarray,
        goal_color_ids: jnp.ndarray,
        queried_ordinals: jnp.ndarray,
        num_regions: jnp.ndarray,
        *,
        train: bool = False,
    ) -> dict[str, jnp.ndarray]:
        batch = patch_tokens.shape[0]
        expected = (
            batch,
            self.max_steps,
            self.frames,
            self.spatial_tokens,
            self.input_width,
        )
        if patch_tokens.shape != expected:
            raise ValueError(f"Expected patch tokens {expected}, got {patch_tokens.shape}")
        if sequence_mask.shape != (batch, self.max_steps):
            raise ValueError(f"Invalid sequence mask {sequence_mask.shape}")

        flat = patch_tokens.reshape(
            batch * self.max_steps,
            self.frames,
            self.spatial_tokens,
            self.input_width,
        )
        evidence = VisualWindowEncoder(
            name="visual_window_encoder",
            frames=self.frames,
            spatial_tokens=self.spatial_tokens,
            input_width=self.input_width,
            width=self.width,
            encoder_width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            dtype_mm="bfloat16",
        )(flat, train=train).reshape(
            batch, self.max_steps, self.frames, self.spatial_tokens, self.width
        )
        early = evidence[:, :, : self.frames // 2].mean(axis=(2, 3))
        late = evidence[:, :, self.frames // 2 :].mean(axis=(2, 3))
        mean = evidence.mean(axis=(2, 3))
        event_visual = jnp.concatenate(
            (mean, early, late, late - early, jnp.abs(late - early)), axis=-1
        ).astype(jnp.float32)

        task = nn.Embed(len(contract.TASKS), self.width, name="task_embedding")(
            task_ids
        )
        color_embed = nn.Embed(
            len(contract.COLORS), self.width, name="color_embedding"
        )
        color_0 = color_embed(goal_color_ids[:, 0])
        color_1 = color_embed(goal_color_ids[:, 1])
        ordinal = nn.Embed(5, self.width, name="ordinal_embedding")(
            queried_ordinals
        )
        regions = nn.Embed(5, self.width, name="num_regions_embedding")(
            num_regions
        )
        goal = jnp.concatenate((task, color_0, color_1, ordinal, regions), axis=-1)
        goal = jnp.broadcast_to(
            goal[:, None, None],
            (batch, self.max_steps, self.micro_events, goal.shape[-1]),
        )
        visual = jnp.broadcast_to(
            event_visual[:, :, None],
            (
                batch,
                self.max_steps,
                self.micro_events,
                event_visual.shape[-1],
            ),
        )
        micro_embedding = self.param(
            "micro_event_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.micro_events, self.width),
        )
        hidden = jnp.concatenate((visual, goal, jnp.broadcast_to(
            micro_embedding, (batch, self.max_steps, self.micro_events, self.width)
        )), axis=-1)
        hidden = nn.LayerNorm(name="operation_input_ln")(hidden)
        hidden = nn.gelu(nn.Dense(self.width * 2, name="operation_hidden")(hidden))
        hidden = nn.LayerNorm(name="operation_hidden_ln")(
            nn.Dense(self.width, name="operation_out")(hidden)
        )
        event_type_logits = nn.Dense(3, name="event_type_logits")(hidden)
        write_entity_logits = nn.Dense(7, name="write_entity_logits")(hidden)
        write_region_logits = nn.Dense(4, name="write_region_logits")(hidden)
        swap_pair_logits = nn.Dense(len(SWAP_PAIRS), name="swap_pair_logits")(hidden)

        initial = jnp.zeros((batch, 7, 5), dtype=jnp.float32)
        initial = initial.at[:, :, 0].set(1.0)

        def apply_micro(
            table: jnp.ndarray,
            type_logits: jnp.ndarray,
            entity_logits: jnp.ndarray,
            region_logits: jnp.ndarray,
            pair_logits: jnp.ndarray,
        ) -> jnp.ndarray:
            gates = jax.nn.softmax(
                type_logits.astype(jnp.float32) / self.gate_temperature, axis=-1
            )
            entity = _straight_through_one_hot(entity_logits)
            region_4 = _straight_through_one_hot(region_logits)
            region = jnp.concatenate(
                (jnp.zeros((batch, 1), dtype=jnp.float32), region_4), axis=-1
            )
            write = (
                (1.0 - entity[:, :, None]) * table
                + entity[:, :, None] * region[:, None, :]
            )

            pair = _straight_through_one_hot(pair_logits)
            swapped = jnp.zeros_like(table)
            for pair_index, (region_a, region_b) in enumerate(SWAP_PAIRS):
                permutation = table
                column_a = table[:, :, region_a + 1]
                column_b = table[:, :, region_b + 1]
                permutation = permutation.at[:, :, region_a + 1].set(column_b)
                permutation = permutation.at[:, :, region_b + 1].set(column_a)
                swapped = swapped + pair[:, pair_index, None, None] * permutation
            return (
                gates[:, 0, None, None] * table
                + gates[:, 1, None, None] * write
                + gates[:, 2, None, None] * swapped
            )

        def scan_step(table: jnp.ndarray, inputs):
            type_logits, entity_logits, region_logits, pair_logits, valid = inputs
            candidate = table
            for micro in range(self.micro_events):
                candidate = apply_micro(
                    candidate,
                    type_logits[:, micro],
                    entity_logits[:, micro],
                    region_logits[:, micro],
                    pair_logits[:, micro],
                )
            next_table = jnp.where(valid[:, None, None], candidate, table)
            return next_table, next_table

        _, tables = jax.lax.scan(
            scan_step,
            initial,
            (
                jnp.swapaxes(event_type_logits, 0, 1),
                jnp.swapaxes(write_entity_logits, 0, 1),
                jnp.swapaxes(write_region_logits, 0, 1),
                jnp.swapaxes(swap_pair_logits, 0, 1),
                jnp.swapaxes(sequence_mask, 0, 1),
            ),
        )
        tables = jnp.swapaxes(tables, 0, 1)
        all_tables = jnp.concatenate((initial[:, None], tables), axis=1)

        region_code = self.param(
            "semantic_region_code",
            nn.initializers.normal(stddev=0.02),
            (5, self.width),
        )
        entity_code = self.param(
            "semantic_entity_code",
            nn.initializers.normal(stddev=0.02),
            (7, self.width),
        )
        semantic_tokens = jnp.einsum("bsec,cd->bsed", all_tables, region_code)
        semantic_tokens = semantic_tokens + entity_code[None, None]
        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.memory_tokens, self.width),
        )
        memories = jnp.broadcast_to(
            base_memory,
            (batch, self.max_steps + 1, self.memory_tokens, self.width),
        )
        memories = memories.at[:, :, :7].set(semantic_tokens)
        return {
            "all_tables": all_tables,
            "all_memories": memories,
            "event_type_logits": event_type_logits,
            "write_entity_logits": write_entity_logits,
            "write_region_logits": write_region_logits,
            "swap_pair_logits": swap_pair_logits,
            "write_gates": jax.nn.softmax(
                event_type_logits.astype(jnp.float32) / self.gate_temperature,
                axis=-1,
            )[..., 1],
            "swap_gates": jax.nn.softmax(
                event_type_logits.astype(jnp.float32) / self.gate_temperature,
                axis=-1,
            )[..., 2],
        }

