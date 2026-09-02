"""Visual operation parser probes for RoboMME spatial/temporal ablations."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from openpi.tasks.robomme import unified_gt_teacher as contract
from openpi.tasks.robomme.anchor_conditioned_decomposed_memory import _sample_anchor_tokens
from openpi.tasks.robomme.decomposed_region_recurrent_memory import SWAP_PAIRS


class MaskedEventStateCell(nn.Module):
    """Small gated causal state used only by the recurrent parser variant."""

    width: int

    @nn.compact
    def __call__(self, carry, visual, valid):
        joined = jnp.concatenate((carry, visual), axis=-1)
        proposal = jnp.tanh(nn.Dense(self.width, name="proposal")(joined))
        gate = nn.sigmoid(nn.Dense(self.width, name="gate")(joined))
        candidate = gate * proposal + (1.0 - gate) * carry
        next_carry = jnp.where(valid[:, None], candidate, carry)
        return next_carry, next_carry


class CausalVisualOperationParser(nn.Module):
    """Shared local or causal parser with no persistent-table update inside."""

    max_steps: int = 96
    frames: int = 12
    spatial_tokens: int = 16
    input_width: int = 1152
    max_anchors: int = 4
    width: int = 64
    hidden_width: int = 128
    micro_events: int = 2
    recurrent_event_state: bool = False

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
        previous_tables: jnp.ndarray,
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
        if previous_tables.shape != (batch, self.max_steps, 7):
            raise ValueError(f"Invalid previous table shape {previous_tables.shape}")

        sampled = _sample_anchor_tokens(patch_tokens, anchor_yx).astype(jnp.float32)
        sampled = nn.Dense(self.width, name="anchor_visual_projection")(
            nn.LayerNorm(name="anchor_visual_ln")(sampled)
        )
        y, x = anchor_yx[..., 0], anchor_yx[..., 1]
        coordinates = jnp.stack(
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
                    coordinates
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
                    nn.Dense(self.hidden_width, name="anchor_evidence_hidden")(
                        temporal
                    )
                )
            )
        )
        valid_anchor = anchor_mask[:, None, :, None].astype(jnp.float32)
        pooled = jnp.sum(anchor_evidence * valid_anchor, axis=2) / jnp.maximum(
            jnp.sum(valid_anchor, axis=2), 1.0
        )

        table = jnp.eye(5, dtype=jnp.float32)[previous_tables]
        region_code = self.param(
            "feedback_region_code",
            nn.initializers.normal(stddev=0.02),
            (5, self.width),
        )
        entity_code = self.param(
            "feedback_entity_code",
            nn.initializers.normal(stddev=0.02),
            (7, self.width),
        )
        entity_state = jnp.einsum("bsec,cw->bsew", table, region_code)
        entity_state = entity_state + entity_code[None, None]
        state_summary = nn.Dense(self.width, name="state_summary")(
            entity_state.reshape(batch, self.max_steps, -1)
        )

        task = nn.Embed(len(contract.TASKS), self.width, name="task_embedding")(task_ids)
        color_embed = nn.Embed(len(contract.COLORS), self.width, name="color_embedding")
        color_0 = color_embed(goal_color_ids[:, 0])
        color_1 = color_embed(goal_color_ids[:, 1])
        ordinal = nn.Embed(5, self.width, name="ordinal_embedding")(queried_ordinals)
        regions = nn.Embed(5, self.width, name="num_regions_embedding")(num_regions)
        goal = nn.Dense(self.width, name="goal_projection")(
            jnp.concatenate((task, color_0, color_1, ordinal, regions), axis=-1)
        )
        goal = jnp.broadcast_to(goal[:, None], (batch, self.max_steps, self.width))
        visual = nn.LayerNorm(name="visual_state_input_ln")(
            nn.Dense(self.width, name="visual_state_input")(
                jnp.concatenate((pooled, goal, state_summary), axis=-1)
            )
        )

        if self.recurrent_event_state:
            scanned_cell = nn.scan(
                MaskedEventStateCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=(1, 1),
                out_axes=1,
            )(
                name="causal_event_state",
                width=self.width,
            )
            initial = jnp.zeros((batch, self.width), dtype=jnp.float32)
            _, step_hidden = scanned_cell(initial, visual, sequence_mask)
        else:
            step_hidden = nn.gelu(
                nn.Dense(self.width, name="local_step_hidden")(visual)
            )

        micro_embedding = self.param(
            "micro_event_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.micro_events, self.width),
        )
        hidden = jnp.broadcast_to(
            step_hidden[:, :, None],
            (batch, self.max_steps, self.micro_events, self.width),
        )
        hidden = hidden + micro_embedding
        hidden = nn.LayerNorm(name="operation_input_ln")(hidden)
        hidden = nn.gelu(nn.Dense(self.hidden_width, name="operation_hidden")(hidden))
        hidden = nn.LayerNorm(name="operation_hidden_ln")(
            nn.Dense(self.width, name="operation_out")(hidden)
        )
        phase_logits = nn.Dense(4, name="phase_logits")(step_hidden)
        completion_logits = nn.Dense(1, name="completion_logits")(hidden)[..., 0]
        event_kind_logits = nn.Dense(2, name="event_kind_logits")(hidden)
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

        pair_features = []
        pair_masks = []
        for region_a, region_b in SWAP_PAIRS:
            a = anchor_evidence[:, :, region_a]
            b = anchor_evidence[:, :, region_b]
            pair_features.append(
                jnp.concatenate((a + b, jnp.abs(a - b), a * b), axis=-1)
            )
            pair_masks.append(anchor_mask[:, region_a] & anchor_mask[:, region_b])
        pair_keys = nn.Dense(self.width, name="swap_pair_keys")(
            jnp.stack(pair_features, axis=2)
        )
        swap_query = nn.Dense(self.width, name="swap_query")(hidden)
        swap_pair_logits = jnp.einsum(
            "bsmw,bspw->bsmp", swap_query, pair_keys
        ) / jnp.sqrt(float(self.width))
        pair_mask = jnp.stack(pair_masks, axis=1)
        swap_pair_logits = jnp.where(
            pair_mask[:, None, None], swap_pair_logits, -1e9
        )
        return {
            "phase_logits": phase_logits,
            "completion_logits": completion_logits,
            "event_kind_logits": event_kind_logits,
            "event_type_logits": event_type_logits,
            "write_entity_logits": write_entity_logits,
            "write_region_logits": write_region_logits,
            "swap_pair_logits": swap_pair_logits,
        }
