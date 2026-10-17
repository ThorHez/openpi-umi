"""Anchor-conditioned decomposed recurrent semantic memory for RoboMME."""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.tasks.robomme import unified_gt_teacher as contract
from openpi.tasks.robomme.decomposed_region_recurrent_memory import SWAP_PAIRS


def _straight_through_one_hot(logits: jnp.ndarray) -> jnp.ndarray:
    soft = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    hard = jax.nn.one_hot(jnp.argmax(logits, axis=-1), logits.shape[-1])
    return soft + jax.lax.stop_gradient(hard - soft)


def _sample_anchor_tokens(
    patch_tokens: jnp.ndarray,
    anchor_yx: jnp.ndarray,
) -> jnp.ndarray:
    """Bilinearly sample a square token grid at normalized anchor coordinates.

    Args:
        patch_tokens: [B, S, F, P, D], where P is a square number.
        anchor_yx: [B, A, 2], normalized to [-1, 1] in the source image.

    Returns:
        [B, S, F, A, D] sampled tokens.
    """

    batch, steps, frames, spatial, width = patch_tokens.shape
    grid_size = math.isqrt(spatial)
    if grid_size * grid_size != spatial:
        raise ValueError(f"Expected a square grid, got {spatial} tokens")
    grid = patch_tokens.reshape(batch, steps, frames, grid_size, grid_size, width)
    spacing = 2.0 / grid_size
    centers = -1.0 + spacing * (jnp.arange(grid_size, dtype=jnp.float32) + 0.5)
    y = anchor_yx[..., 0]
    x = anchor_yx[..., 1]
    # Triangular interpolation kernels. Coordinates at the image edge are
    # clipped to the nearest grid center.
    y = jnp.clip(y, centers[0], centers[-1])
    x = jnp.clip(x, centers[0], centers[-1])
    wy = jnp.clip(1.0 - jnp.abs(y[..., None] - centers) / spacing, 0.0, 1.0)
    wx = jnp.clip(1.0 - jnp.abs(x[..., None] - centers) / spacing, 0.0, 1.0)
    weights = wy[..., :, None] * wx[..., None, :]
    weights = weights / jnp.maximum(weights.sum(axis=(-2, -1), keepdims=True), 1e-6)
    return jnp.einsum("bsfhwd,bahw->bsfad", grid, weights)


class AnchorConditionedDecomposedMemory(nn.Module):
    """Shared anchor-pointer operations followed by an explicit recurrent table."""

    max_steps: int = 96
    frames: int = 12
    spatial_tokens: int = 16
    input_width: int = 1152
    max_anchors: int = 4
    width: int = 64
    hidden_width: int = 128
    memory_tokens: int = 128
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
        anchor_yx: jnp.ndarray,
        anchor_mask: jnp.ndarray,
        *,
        train: bool = False,
    ) -> dict[str, jnp.ndarray]:
        del train
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
        if anchor_yx.shape != (batch, self.max_anchors, 2):
            raise ValueError(f"Invalid anchor coordinates {anchor_yx.shape}")
        if anchor_mask.shape != (batch, self.max_anchors):
            raise ValueError(f"Invalid anchor mask {anchor_mask.shape}")

        sampled = _sample_anchor_tokens(patch_tokens, anchor_yx).astype(jnp.float32)
        sampled = nn.Dense(self.width, name="anchor_visual_projection")(
            nn.LayerNorm(name="anchor_visual_ln")(sampled)
        )
        y, x = anchor_yx[..., 0], anchor_yx[..., 1]
        coordinate_features = jnp.stack(
            (
                y,
                x,
                y * y,
                x * x,
                y * x,
                jnp.sin(jnp.pi * y),
                jnp.cos(jnp.pi * y),
                jnp.sin(jnp.pi * x),
                jnp.cos(jnp.pi * x),
            ),
            axis=-1,
        )
        coordinate = nn.Dense(self.width, name="anchor_coordinate_out")(
            nn.gelu(
                nn.Dense(self.hidden_width, name="anchor_coordinate_hidden")(
                    coordinate_features
                )
            )
        )
        sampled = sampled + coordinate[:, None, None]
        early = sampled[:, :, : self.frames // 2].mean(axis=2)
        late = sampled[:, :, self.frames // 2 :].mean(axis=2)
        mean = sampled.mean(axis=2)
        temporal = jnp.concatenate(
            (mean, early, late, late - early, jnp.abs(late - early)), axis=-1
        )
        anchor_evidence = nn.LayerNorm(name="anchor_evidence_ln")(
            nn.Dense(self.width, name="anchor_evidence_out")(
                nn.gelu(
                    nn.Dense(self.hidden_width, name="anchor_evidence_hidden")(temporal)
                )
            )
        )
        valid_anchor = anchor_mask[:, None, :, None].astype(jnp.float32)
        pooled = jnp.sum(anchor_evidence * valid_anchor, axis=2) / jnp.maximum(
            jnp.sum(valid_anchor, axis=2), 1.0
        )

        task = nn.Embed(len(contract.TASKS), self.width, name="task_embedding")(task_ids)
        color_embed = nn.Embed(len(contract.COLORS), self.width, name="color_embedding")
        color_0 = color_embed(goal_color_ids[:, 0])
        color_1 = color_embed(goal_color_ids[:, 1])
        ordinal = nn.Embed(5, self.width, name="ordinal_embedding")(queried_ordinals)
        regions = nn.Embed(5, self.width, name="num_regions_embedding")(num_regions)
        goal = jnp.concatenate((task, color_0, color_1, ordinal, regions), axis=-1)
        goal = jnp.broadcast_to(
            goal[:, None, None],
            (batch, self.max_steps, self.micro_events, goal.shape[-1]),
        )
        visual = jnp.broadcast_to(
            pooled[:, :, None],
            (batch, self.max_steps, self.micro_events, self.width),
        )
        micro_embedding = self.param(
            "micro_event_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.micro_events, self.width),
        )
        hidden = jnp.concatenate(
            (
                visual,
                goal,
                jnp.broadcast_to(
                    micro_embedding,
                    (batch, self.max_steps, self.micro_events, self.width),
                ),
            ),
            axis=-1,
        )
        hidden = nn.LayerNorm(name="operation_input_ln")(hidden)
        hidden = nn.gelu(nn.Dense(self.hidden_width, name="operation_hidden")(hidden))
        hidden = nn.LayerNorm(name="operation_hidden_ln")(
            nn.Dense(self.width, name="operation_out")(hidden)
        )
        event_type_logits = nn.Dense(3, name="event_type_logits")(hidden)
        write_entity_logits = nn.Dense(7, name="write_entity_logits")(hidden)

        write_query = nn.Dense(self.width, name="write_query")(hidden)
        write_keys = nn.Dense(self.width, name="write_anchor_keys")(anchor_evidence)
        write_region_logits = jnp.einsum(
            "bsmw,bsaw->bsma", write_query, write_keys
        ) / jnp.sqrt(float(self.width))
        write_region_logits = jnp.where(
            anchor_mask[:, None, None], write_region_logits, -1e9
        )

        pair_keys = []
        pair_masks = []
        for region_a, region_b in SWAP_PAIRS:
            a = anchor_evidence[:, :, region_a]
            b = anchor_evidence[:, :, region_b]
            pair_feature = jnp.concatenate((a + b, jnp.abs(a - b), a * b), axis=-1)
            pair_keys.append(pair_feature)
            pair_masks.append(anchor_mask[:, region_a] & anchor_mask[:, region_b])
        pair_keys = jnp.stack(pair_keys, axis=2)
        pair_keys = nn.Dense(self.width, name="swap_pair_keys")(pair_keys)
        swap_query = nn.Dense(self.width, name="swap_query")(hidden)
        swap_pair_logits = jnp.einsum(
            "bsmw,bspw->bsmp", swap_query, pair_keys
        ) / jnp.sqrt(float(self.width))
        pair_mask = jnp.stack(pair_masks, axis=1)
        swap_pair_logits = jnp.where(
            pair_mask[:, None, None], swap_pair_logits, -1e9
        )

        initial = jnp.zeros((batch, 7, 5), dtype=jnp.float32)
        initial = initial.at[:, :, 0].set(1.0)

        def apply_micro(table, type_logits, entity_logits, region_logits, pair_logits):
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

        def scan_step(table, inputs):
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
        all_tables = jnp.concatenate(
            (initial[:, None], jnp.swapaxes(tables, 0, 1)), axis=1
        )
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
        }
