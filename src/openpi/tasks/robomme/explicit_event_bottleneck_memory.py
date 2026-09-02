"""Explicit visual event bottleneck with a deterministic semantic executor."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.tasks.robomme import unified_gt_teacher as contract
from openpi.tasks.robomme.anchor_conditioned_decomposed_memory import _sample_anchor_tokens
from openpi.tasks.robomme.decomposed_region_recurrent_memory import SWAP_PAIRS


def _straight_through_one_hot(logits: jnp.ndarray) -> jnp.ndarray:
    soft = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    hard = jax.nn.one_hot(jnp.argmax(logits, axis=-1), logits.shape[-1])
    return soft + jax.lax.stop_gradient(hard - soft)


def _straight_through_table(table: jnp.ndarray) -> jnp.ndarray:
    normalized = table / jnp.maximum(table.sum(axis=-1, keepdims=True), 1e-6)
    hard = jax.nn.one_hot(jnp.argmax(normalized, axis=-1), normalized.shape[-1])
    return normalized + jax.lax.stop_gradient(hard - normalized)


class CausalAnchorEvidenceCell(nn.Module):
    """Carry unresolved per-anchor visual evidence across fixed chunks."""

    width: int

    @nn.compact
    def __call__(self, carry, current, valid):
        joined = jnp.concatenate((carry, current), axis=-1)
        proposal = jnp.tanh(nn.Dense(self.width, name="proposal")(joined))
        gate = nn.sigmoid(nn.Dense(self.width, name="gate")(joined))
        candidate = gate * proposal + (1.0 - gate) * carry
        next_carry = jnp.where(valid[:, None, None], candidate, carry)
        output = nn.LayerNorm(name="output_ln")(
            nn.Dense(self.width, name="output")(
                jnp.concatenate((current, next_carry, next_carry - carry), axis=-1)
            )
        )
        return next_carry, output


class ExplicitSemanticExecutorCell(nn.Module):
    """Predict a discrete event tuple and execute its semantic state delta."""

    width: int = 64
    hidden_width: int = 128
    micro_events: int = 2
    gate_temperature: float = 0.25
    deterministic_updater: bool = True

    @nn.compact
    def __call__(
        self,
        carry_table: jnp.ndarray,
        anchor_evidence: jnp.ndarray,
        pooled_visual: jnp.ndarray,
        goal: jnp.ndarray,
        anchor_mask: jnp.ndarray,
        teacher_previous_table: jnp.ndarray,
        teacher_force: jnp.ndarray,
        valid: jnp.ndarray,
    ):
        batch = carry_table.shape[0]
        teacher_table = jax.nn.one_hot(teacher_previous_table, 5).astype(jnp.float32)
        feedback = jnp.where(
            teacher_force[:, None, None], teacher_table, carry_table
        )
        # The forward interface is categorical even though the straight-through
        # path still lets recurrent losses train the visual event parser.
        feedback = _straight_through_table(feedback)

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
        entity_state = jnp.einsum("bec,cw->bew", feedback, region_code)
        entity_state = entity_state + entity_code[None]
        state_summary = nn.Dense(self.width, name="state_summary")(
            entity_state.reshape(batch, -1)
        )

        micro_embedding = self.param(
            "micro_event_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.micro_events, self.width),
        )
        shared = jnp.concatenate((pooled_visual, goal, state_summary), axis=-1)
        shared = jnp.broadcast_to(
            shared[:, None], (batch, self.micro_events, shared.shape[-1])
        )
        hidden = jnp.concatenate(
            (shared, jnp.broadcast_to(micro_embedding, (batch, self.micro_events, self.width))),
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
            "bmw,baw->bma", write_query, write_keys
        ) / jnp.sqrt(float(self.width))
        write_region_logits = jnp.where(
            anchor_mask[:, None], write_region_logits, -1e9
        )

        pair_features = []
        pair_masks = []
        for region_a, region_b in SWAP_PAIRS:
            a = anchor_evidence[:, region_a]
            b = anchor_evidence[:, region_b]
            pair_features.append(
                jnp.concatenate((a + b, jnp.abs(a - b), a * b), axis=-1)
            )
            pair_masks.append(anchor_mask[:, region_a] & anchor_mask[:, region_b])
        pair_keys = nn.Dense(self.width, name="swap_pair_keys")(
            jnp.stack(pair_features, axis=1)
        )
        swap_query = nn.Dense(self.width, name="swap_query")(hidden)
        swap_pair_logits = jnp.einsum(
            "bmw,bpw->bmp", swap_query, pair_keys
        ) / jnp.sqrt(float(self.width))
        pair_mask = jnp.stack(pair_masks, axis=1)
        swap_pair_logits = jnp.where(pair_mask[:, None], swap_pair_logits, -1e9)

        def execute(table, type_logits, entity_logits, region_logits, pair_logits):
            scaled_type = type_logits.astype(jnp.float32) / self.gate_temperature
            if self.deterministic_updater:
                event = _straight_through_one_hot(scaled_type)
            else:
                event = jax.nn.softmax(scaled_type, axis=-1)
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
                event[:, 0, None, None] * table
                + event[:, 1, None, None] * write
                + event[:, 2, None, None] * swapped
            ), event

        candidate = feedback
        event_gates = []
        for micro in range(self.micro_events):
            candidate, gates = execute(
                candidate,
                event_type_logits[:, micro],
                write_entity_logits[:, micro],
                write_region_logits[:, micro],
                swap_pair_logits[:, micro],
            )
            event_gates.append(gates)
        next_table = jnp.where(valid[:, None, None], candidate, carry_table)
        return next_table, (
            next_table,
            event_type_logits,
            write_entity_logits,
            write_region_logits,
            swap_pair_logits,
            jnp.stack(event_gates, axis=1),
        )


class ExplicitEventBottleneckMemory(nn.Module):
    """Factorial probe for temporal evidence and deterministic event execution."""

    max_steps: int = 96
    frames: int = 12
    spatial_tokens: int = 64
    input_width: int = 1152
    max_anchors: int = 4
    width: int = 64
    hidden_width: int = 128
    memory_tokens: int = 128
    micro_events: int = 2
    temporal_encoder: str = "pooled"
    temporal_depth: int = 2
    temporal_heads: int = 4
    deterministic_updater: bool = True
    causal_evidence_state: bool = False
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
        teacher_previous_tables: jnp.ndarray,
        teacher_force_mask: jnp.ndarray,
        proprio_tokens: jnp.ndarray | None = None,
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
        if self.temporal_encoder not in ("pooled", "relational"):
            raise ValueError(f"Unknown temporal encoder: {self.temporal_encoder}")

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
            nn.gelu(nn.Dense(self.hidden_width, name="anchor_coordinate_hidden")(coordinates))
        )
        sampled = sampled + coordinate[:, None, None]

        if self.temporal_encoder == "pooled":
            early = sampled[:, :, : self.frames // 2].mean(axis=2)
            late = sampled[:, :, self.frames // 2 :].mean(axis=2)
            mean = sampled.mean(axis=2)
            temporal = jnp.concatenate(
                (mean, early, late, late - early, jnp.abs(late - early)), axis=-1
            )
        else:
            temporal_position = self.param(
                "temporal_position",
                nn.initializers.normal(stddev=0.02),
                (1, self.frames, self.width),
            )
            values = jnp.transpose(sampled, (0, 1, 3, 2, 4)).reshape(
                batch * self.max_steps * self.max_anchors,
                self.frames,
                self.width,
            )
            values = values + temporal_position
            for layer in range(self.temporal_depth):
                residual = values
                normalized = nn.LayerNorm(name=f"temporal_ln_{layer}")(values)
                values = residual + nn.SelfAttention(
                    num_heads=self.temporal_heads,
                    qkv_features=self.width,
                    out_features=self.width,
                    name=f"temporal_attention_{layer}",
                )(normalized)
                residual = values
                normalized = nn.LayerNorm(name=f"temporal_mlp_ln_{layer}")(values)
                hidden = nn.gelu(
                    nn.Dense(self.hidden_width, name=f"temporal_mlp_in_{layer}")(
                        normalized
                    )
                )
                values = residual + nn.Dense(
                    self.width, name=f"temporal_mlp_out_{layer}"
                )(hidden)
            start = values[:, 0]
            end = values[:, -1]
            temporal = jnp.concatenate(
                (values.mean(axis=1), values.max(axis=1), start, end, end - start),
                axis=-1,
            ).reshape(batch, self.max_steps, self.max_anchors, -1)

        anchor_evidence = nn.LayerNorm(name="anchor_evidence_ln")(
            nn.Dense(self.width, name="anchor_evidence_out")(
                nn.gelu(nn.Dense(self.hidden_width, name="anchor_evidence_hidden")(temporal))
            )
        )
        if proprio_tokens is not None:
            expected_proprio = (batch, self.max_steps, self.frames, 8)
            if proprio_tokens.shape != expected_proprio:
                raise ValueError(
                    f"Expected proprio tokens {expected_proprio}, got {proprio_tokens.shape}"
                )
            proprio_tokens = proprio_tokens.astype(jnp.float32)
            proprio_early = proprio_tokens[:, :, : self.frames // 2].mean(axis=2)
            proprio_late = proprio_tokens[:, :, self.frames // 2 :].mean(axis=2)
            proprio_mean = proprio_tokens.mean(axis=2)
            proprio_temporal = jnp.concatenate(
                (
                    proprio_mean,
                    proprio_early,
                    proprio_late,
                    proprio_late - proprio_early,
                    jnp.abs(proprio_late - proprio_early),
                    proprio_tokens.min(axis=2),
                    proprio_tokens.max(axis=2),
                ),
                axis=-1,
            )
            proprio_evidence = nn.LayerNorm(name="proprio_evidence_ln")(
                nn.Dense(self.width, name="proprio_evidence_out")(
                    nn.gelu(
                        nn.Dense(
                            self.hidden_width, name="proprio_evidence_hidden"
                        )(proprio_temporal)
                    )
                )
            )
            # Broadcast robot phase evidence to every region anchor.  Region
            # identity still comes exclusively from the anchor-local RGB
            # stream; proprio decides whether an observed contact/release is a
            # completed event rather than visual motion or a hold interval.
            anchor_evidence = anchor_evidence + proprio_evidence[:, :, None, :]
        valid_anchor = anchor_mask[:, None, :, None].astype(jnp.float32)
        global_evidence = jnp.sum(anchor_evidence * valid_anchor, axis=2) / jnp.maximum(
            jnp.sum(valid_anchor, axis=2), 1.0
        )
        relation = jnp.concatenate(
            (
                anchor_evidence,
                global_evidence[:, :, None].repeat(self.max_anchors, axis=2),
                anchor_evidence - global_evidence[:, :, None],
                anchor_evidence * global_evidence[:, :, None],
            ),
            axis=-1,
        )
        anchor_evidence = nn.LayerNorm(name="cross_anchor_relation_ln")(
            nn.Dense(self.width, name="cross_anchor_relation")(relation)
        )

        if self.causal_evidence_state:
            scanned_evidence = nn.scan(
                CausalAnchorEvidenceCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=(1, 1),
                out_axes=1,
            )(name="causal_anchor_evidence", width=self.width)
            evidence_initial = jnp.zeros(
                (batch, self.max_anchors, self.width), dtype=jnp.float32
            )
            _, anchor_evidence = scanned_evidence(
                evidence_initial, anchor_evidence, sequence_mask
            )

        pooled = jnp.sum(anchor_evidence * valid_anchor, axis=2) / jnp.maximum(
            jnp.sum(valid_anchor, axis=2), 1.0
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

        initial = jnp.zeros((batch, 7, 5), dtype=jnp.float32)
        initial = initial.at[:, :, 0].set(1.0)
        scanned_executor = nn.scan(
            ExplicitSemanticExecutorCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=(1, 1, 1, nn.broadcast, 1, 1, 1),
            out_axes=1,
        )(
            name="explicit_semantic_executor",
            width=self.width,
            hidden_width=self.hidden_width,
            micro_events=self.micro_events,
            gate_temperature=self.gate_temperature,
            deterministic_updater=self.deterministic_updater,
        )
        _, outputs = scanned_executor(
            initial,
            anchor_evidence,
            pooled,
            goal,
            anchor_mask,
            teacher_previous_tables,
            teacher_force_mask,
            sequence_mask,
        )
        tables, event, entity, region, pair, event_gates = outputs
        all_tables = jnp.concatenate((initial[:, None], tables), axis=1)

        semantic_region_code = self.param(
            "semantic_region_code", nn.initializers.normal(stddev=0.02), (5, self.width)
        )
        semantic_entity_code = self.param(
            "semantic_entity_code", nn.initializers.normal(stddev=0.02), (7, self.width)
        )
        semantic_tokens = jnp.einsum("bsec,cd->bsed", all_tables, semantic_region_code)
        semantic_tokens = semantic_tokens + semantic_entity_code[None, None]
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
            "event_type_logits": event,
            "write_entity_logits": entity,
            "write_region_logits": region,
            "swap_pair_logits": pair,
            "event_bottleneck": event_gates,
        }
