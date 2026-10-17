"""Fixed-rate, trigger-free recurrent visual student for RoboMME."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import siglip_mem_semantic as memory_core
from openpi.tasks.robomme import unified_gt_teacher as teacher_contract
from openpi.tasks.robomme.unified_visual_student import VisualWindowEncoder


class CausalEvidenceScanCell(nn.Module):
    """Carry one shared visual-evidence state across fixed chunks."""

    width: int = 64
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(
        self,
        carry: jnp.ndarray,
        evidence: jnp.ndarray,
        valid: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        current = jnp.mean(
            nn.LayerNorm(name="current_ln", dtype=self.dtype_mm)(evidence), axis=1
        )
        joined = jnp.concatenate((carry, current), axis=-1)
        proposal = jnp.tanh(
            nn.Dense(self.width, name="proposal", dtype=self.dtype_mm)(joined)
        )
        gate = jax.nn.sigmoid(
            nn.Dense(self.width, name="gate", dtype=self.dtype_mm)(joined)
        )
        candidate = gate * proposal + (1.0 - gate) * carry
        next_carry = jnp.where(valid[:, None], candidate, carry)
        token = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(
            nn.Dense(self.width, name="output", dtype=self.dtype_mm)(
                jnp.concatenate((current, next_carry, next_carry - carry), axis=-1)
            )
        )
        token = jnp.where(valid[:, None], token, 0.0)
        return next_carry, token


class MemoryUpdateScanCell(nn.Module):
    """One shared memory transition used by ``nn.scan`` at every fixed chunk."""

    width: int = 64
    depth: int = 2
    num_heads: int = 4
    dtype_mm: str = "float32"
    use_write_gate: bool = False
    write_gate_bias: float = -2.0
    use_event_gate: bool = False
    event_gate_bias: float = 0.0
    event_gate_modulation_strength: float = 0.0
    event_gate_reference: float = 0.2
    event_gate_modulation_min: float = 0.75
    event_gate_modulation_max: float = 1.25
    use_event_update_routing: bool = False
    event_update_routing_temperature: float = 1.0
    event_update_routing_reference: float = 0.5
    use_event_correction: bool = False
    use_oracle_event_correction: bool = False
    use_recurrent_memory: bool = True

    @nn.compact
    def __call__(
        self,
        memory: jnp.ndarray,
        evidence: jnp.ndarray,
        valid: jnp.ndarray,
        oracle_event: jnp.ndarray,
        reset_memory: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        carry_memory = memory
        source_memory = memory if self.use_recurrent_memory else reset_memory
        candidate = source_memory
        for index in range(self.depth):
            candidate = memory_core.MemoryUpdateBlock(
                name=f"update_block_{index}",
                width=self.width,
                num_heads=self.num_heads,
                dtype_mm=self.dtype_mm,
            )(candidate, evidence)
        candidate = nn.LayerNorm(name="state_output_ln", dtype=self.dtype_mm)(candidate)
        if self.use_write_gate:
            memory_summary = jnp.mean(
                nn.LayerNorm(name="gate_memory_ln", dtype=self.dtype_mm)(source_memory), axis=1
            )
            evidence_summary = jnp.mean(
                nn.LayerNorm(name="gate_evidence_ln", dtype=self.dtype_mm)(evidence), axis=1
            )
            gate_hidden = nn.gelu(
                nn.Dense(self.width, name="gate_hidden", dtype=self.dtype_mm)(
                    jnp.concatenate((memory_summary, evidence_summary), axis=-1)
                )
            )
            gate = jax.nn.sigmoid(
                nn.Dense(
                    1,
                    name="gate_out",
                    dtype=self.dtype_mm,
                    kernel_init=nn.initializers.zeros_init(),
                    bias_init=nn.initializers.constant(self.write_gate_bias),
                )(gate_hidden)
            )[:, 0]
        else:
            gate = jnp.ones((memory.shape[0],), dtype=memory.dtype)
        gate = jnp.where(valid, gate, 0.0)

        if self.use_event_gate:
            # The event head is deliberately independent from ``gate_*``.  The
            # existing gate has learned an integration step size, while this
            # head estimates whether the current visual chunk contains a
            # semantic transition.  Early/late evidence differences retain
            # motion information that full-window mean pooling discards.
            event_memory = jnp.mean(
                nn.LayerNorm(name="event_memory_ln", dtype=self.dtype_mm)(memory), axis=1
            )
            event_evidence_tokens = nn.LayerNorm(
                name="event_evidence_ln", dtype=self.dtype_mm
            )(evidence)
            split = event_evidence_tokens.shape[1] // 2
            early_evidence = jnp.mean(event_evidence_tokens[:, :split], axis=1)
            late_evidence = jnp.mean(event_evidence_tokens[:, split:], axis=1)
            event_evidence = jnp.mean(event_evidence_tokens, axis=1)
            temporal_delta = late_evidence - early_evidence
            event_features = jnp.concatenate(
                (
                    event_memory,
                    event_evidence,
                    temporal_delta,
                    jnp.abs(temporal_delta),
                    event_evidence - event_memory,
                ),
                axis=-1,
            )
            event_hidden = nn.gelu(
                nn.Dense(self.width, name="event_hidden", dtype=self.dtype_mm)(event_features)
            )
            event_gate = jax.nn.sigmoid(
                nn.Dense(
                    1,
                    name="event_out",
                    dtype=self.dtype_mm,
                    kernel_init=nn.initializers.zeros_init(),
                    bias_init=nn.initializers.constant(self.event_gate_bias),
                )(event_hidden)
            )[:, 0]
            modulation = 1.0 + self.event_gate_modulation_strength * (
                event_gate - self.event_gate_reference
            )
            modulation = jnp.clip(
                modulation,
                self.event_gate_modulation_min,
                self.event_gate_modulation_max,
            )
        else:
            event_gate = jnp.zeros_like(gate)
            modulation = jnp.ones_like(gate)
        event_gate = jnp.where(valid, event_gate, 0.0)
        modulation = jnp.where(valid, modulation, 1.0)

        if self.use_event_update_routing:
            # Route update *content* without changing the integration step size.
            # Both residual branches start at exactly zero, so enabling routing
            # is functionally identical to the old checkpoint before training.
            route_features = nn.LayerNorm(
                name="route_input_ln", dtype=self.dtype_mm
            )(candidate - source_memory)
            event_update_residual = nn.Dense(
                self.width,
                name="route_event_out",
                dtype=self.dtype_mm,
                kernel_init=nn.initializers.zeros_init(),
                bias_init=nn.initializers.zeros_init(),
            )(route_features)
            hold_update_residual = nn.Dense(
                self.width,
                name="route_hold_out",
                dtype=self.dtype_mm,
                kernel_init=nn.initializers.zeros_init(),
                bias_init=nn.initializers.zeros_init(),
            )(route_features)
            event_logit = jnp.log(jnp.clip(event_gate, 1e-6, 1.0 - 1e-6)) - jnp.log(
                jnp.clip(1.0 - event_gate, 1e-6, 1.0 - 1e-6)
            )
            reference_logit = jnp.log(self.event_update_routing_reference) - jnp.log(
                1.0 - self.event_update_routing_reference
            )
            routing_probability = jax.nn.sigmoid(
                (event_logit - reference_logit) / self.event_update_routing_temperature
            )
            routed_update_residual = (
                routing_probability[:, None, None] * event_update_residual
                + (1.0 - routing_probability[:, None, None]) * hold_update_residual
            )
            candidate = candidate + routed_update_residual
            event_update_residual_norm = jnp.sqrt(
                jnp.mean(jnp.square(event_update_residual.astype(jnp.float32)), axis=(-2, -1))
            )
            hold_update_residual_norm = jnp.sqrt(
                jnp.mean(jnp.square(hold_update_residual.astype(jnp.float32)), axis=(-2, -1))
            )
            routed_update_residual_norm = jnp.sqrt(
                jnp.mean(jnp.square(routed_update_residual.astype(jnp.float32)), axis=(-2, -1))
            )
        else:
            routing_probability = jnp.zeros_like(gate)
            event_update_residual_norm = jnp.zeros_like(gate)
            hold_update_residual_norm = jnp.zeros_like(gate)
            routed_update_residual_norm = jnp.zeros_like(gate)
        event_update_residual_norm = jnp.where(valid, event_update_residual_norm, 0.0)
        hold_update_residual_norm = jnp.where(valid, hold_update_residual_norm, 0.0)
        routed_update_residual_norm = jnp.where(valid, routed_update_residual_norm, 0.0)
        routing_probability = jnp.where(valid, routing_probability, 0.0)
        effective_gate = gate * modulation
        base_memory = source_memory + effective_gate[:, None, None] * (
            candidate - source_memory
        )
        if self.use_event_correction:
            correction_features = nn.LayerNorm(
                name="correction_input_ln", dtype=self.dtype_mm
            )(
                jnp.concatenate(
                    (source_memory, candidate, candidate - source_memory), axis=-1
                )
            )
            correction_hidden = nn.gelu(
                nn.Dense(
                    self.width * 2,
                    name="correction_hidden",
                    dtype=self.dtype_mm,
                )(correction_features)
            )
            event_correction = nn.Dense(
                self.width,
                name="correction_out",
                dtype=self.dtype_mm,
                kernel_init=nn.initializers.zeros_init(),
                bias_init=nn.initializers.zeros_init(),
            )(correction_hidden)
            correction_gate = oracle_event if self.use_oracle_event_correction else event_gate
            correction_gate = jnp.where(valid, correction_gate, 0.0)
            memory = base_memory + correction_gate[:, None, None] * event_correction
            event_correction_norm = jnp.sqrt(
                jnp.mean(jnp.square(event_correction.astype(jnp.float32)), axis=(-2, -1))
            )
        else:
            event_correction = jnp.zeros_like(base_memory)
            correction_gate = jnp.zeros_like(gate)
            event_correction_norm = jnp.zeros_like(gate)
            memory = base_memory
        memory = jnp.where(valid[:, None, None], memory, carry_memory)
        event_correction_norm = jnp.where(valid, event_correction_norm, 0.0)
        return memory, (
            memory,
            base_memory,
            gate,
            event_gate,
            modulation,
            effective_gate,
            event_update_residual_norm,
            hold_update_residual_norm,
            routed_update_residual_norm,
            routing_probability,
            event_correction,
            correction_gate,
            event_correction_norm,
        )


class UnifiedFixedChunkRecurrentStudent(nn.Module):
    """Update memory at a fixed non-overlapping video rate, with no detector."""

    max_steps: int = 96
    frames: int = 12
    spatial_tokens: int = 16
    input_width: int = 1152
    proprio_dim: int = 0
    width: int = 64
    num_memory_tokens: int = 128
    encoder_width: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 8
    memory_depth: int = 2
    memory_heads: int = 4
    dtype_mm: str = "bfloat16"
    use_write_gate: bool = False
    write_gate_bias: float = -2.0
    use_event_gate: bool = False
    event_gate_bias: float = 0.0
    event_gate_modulation_strength: float = 0.0
    event_gate_reference: float = 0.2
    event_gate_modulation_min: float = 0.75
    event_gate_modulation_max: float = 1.25
    use_event_update_routing: bool = False
    event_update_routing_temperature: float = 1.0
    event_update_routing_reference: float = 0.5
    use_event_correction: bool = False
    use_oracle_event_correction: bool = False
    use_causal_evidence_state: bool = False
    use_recurrent_memory: bool = True

    @nn.compact
    def __call__(
        self,
        patch_tokens: jnp.ndarray,
        task_ids: jnp.ndarray,
        goal_color_ids: jnp.ndarray,
        required_counts: jnp.ndarray,
        queried_ordinals: jnp.ndarray,
        num_regions: jnp.ndarray,
        sequence_mask: jnp.ndarray,
        proprio: jnp.ndarray | None = None,
        oracle_event_mask: jnp.ndarray | None = None,
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
            raise ValueError(f"Expected fixed chunks {expected}, got {patch_tokens.shape}")
        if sequence_mask.shape != (batch, self.max_steps):
            raise ValueError(
                f"Expected sequence_mask {(batch, self.max_steps)}, got {sequence_mask.shape}"
            )
        if oracle_event_mask is None:
            oracle_event_mask = jnp.zeros_like(sequence_mask, dtype=jnp.float32)
        if oracle_event_mask.shape != (batch, self.max_steps):
            raise ValueError(
                f"Expected oracle_event_mask {(batch, self.max_steps)}, got "
                f"{oracle_event_mask.shape}"
            )
        flat = patch_tokens.reshape(
            batch * self.max_steps,
            self.frames,
            self.spatial_tokens,
            self.input_width,
        )
        visual_evidence = VisualWindowEncoder(
            name="visual_window_encoder",
            frames=self.frames,
            spatial_tokens=self.spatial_tokens,
            input_width=self.input_width,
            width=self.width,
            encoder_width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            dtype_mm=self.dtype_mm,
        )(flat, train=train).reshape(
            batch,
            self.max_steps,
            self.frames * self.spatial_tokens,
            self.width,
        )
        if self.proprio_dim:
            expected_proprio = (
                batch,
                self.max_steps,
                self.frames,
                self.proprio_dim,
            )
            if proprio is None or proprio.shape != expected_proprio:
                actual = None if proprio is None else proprio.shape
                raise ValueError(f"Expected proprio {expected_proprio}, got {actual}")
            proprio_flat = proprio.astype(jnp.float32).reshape(
                batch * self.max_steps, self.frames, self.proprio_dim
            )
            proprio_evidence = nn.Dense(
                self.width, name="proprio_input", dtype="float32"
            )(proprio_flat)
            proprio_position = self.param(
                "proprio_position",
                nn.initializers.normal(stddev=0.02),
                (1, self.frames, self.width),
                jnp.float32,
            )
            proprio_evidence = proprio_evidence + proprio_position
            proprio_residual = nn.gelu(
                nn.Dense(self.width * 2, name="proprio_hidden", dtype="float32")(
                    proprio_evidence
                )
            )
            proprio_evidence = nn.LayerNorm(
                name="proprio_output_ln", dtype="float32"
            )(
                proprio_evidence
                + nn.Dense(self.width, name="proprio_out", dtype="float32")(
                    proprio_residual
                )
            ).reshape(batch, self.max_steps, self.frames, self.width)
            evidence = jnp.concatenate(
                (visual_evidence.astype(jnp.float32), proprio_evidence), axis=2
            )
        else:
            if proprio is not None and proprio.shape[-1] != 0:
                raise ValueError("Received proprio but proprio_dim is disabled")
            proprio_evidence = jnp.zeros(
                (batch, self.max_steps, 0, self.width), dtype=jnp.float32
            )
            evidence = visual_evidence

        if self.use_causal_evidence_state:
            causal_cell = nn.scan(
                CausalEvidenceScanCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=1,
                out_axes=1,
            )(
                name="shared_causal_evidence_state",
                width=self.width,
                dtype_mm="float32",
            )
            initial_evidence_state = jnp.zeros(
                (batch, self.width), dtype=jnp.float32
            )
            final_evidence_state, causal_evidence_tokens = causal_cell(
                initial_evidence_state,
                evidence.astype(jnp.float32),
                sequence_mask,
            )
            evidence = jnp.concatenate(
                (evidence, causal_evidence_tokens[:, :, None, :]), axis=2
            )
        else:
            causal_evidence_tokens = jnp.zeros(
                (batch, self.max_steps, 0, self.width), dtype=jnp.float32
            )
            final_evidence_state = jnp.zeros(
                (batch, self.width), dtype=jnp.float32
            )

        task_embed = nn.Embed(len(teacher_contract.TASKS), self.width, name="task_embedding")
        color_embed = nn.Embed(len(teacher_contract.COLORS), self.width, name="color_embedding")
        count_embed = nn.Embed(6, self.width, name="count_embedding")
        ordinal_embed = nn.Embed(5, self.width, name="ordinal_embedding")
        regions_embed = nn.Embed(5, self.width, name="num_regions_embedding")
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
                regions_embed(num_regions),
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
            dtype_mm="float32",
        )(base_memory, goal_tokens[:, None])

        scanned_cell = nn.scan(
            MemoryUpdateScanCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(
            name="shared_visual_memory_updater",
            width=self.width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            dtype_mm="float32",
            use_write_gate=self.use_write_gate,
            write_gate_bias=self.write_gate_bias,
            use_event_gate=self.use_event_gate,
            event_gate_bias=self.event_gate_bias,
            event_gate_modulation_strength=self.event_gate_modulation_strength,
            event_gate_reference=self.event_gate_reference,
            event_gate_modulation_min=self.event_gate_modulation_min,
            event_gate_modulation_max=self.event_gate_modulation_max,
            use_event_update_routing=self.use_event_update_routing,
            event_update_routing_temperature=self.event_update_routing_temperature,
            event_update_routing_reference=self.event_update_routing_reference,
            use_event_correction=self.use_event_correction,
            use_oracle_event_correction=self.use_oracle_event_correction,
            use_recurrent_memory=self.use_recurrent_memory,
        )
        final_memory, (
            chunk_memories,
            base_chunk_memories,
            write_gates,
            event_gates,
            gate_modulations,
            effective_write_gates,
            event_update_residual_norms,
            hold_update_residual_norms,
            routed_update_residual_norms,
            event_update_routing_probabilities,
            event_corrections,
            event_correction_gates,
            event_correction_norms,
        ) = scanned_cell(
            initial_memory,
            evidence,
            sequence_mask,
            oracle_event_mask.astype(jnp.float32),
            jnp.broadcast_to(
                initial_memory[:, None],
                (batch, self.max_steps, *initial_memory.shape[1:]),
            ),
        )
        all_memories = jnp.concatenate((initial_memory[:, None], chunk_memories), axis=1)
        return {
            "initial_memory": initial_memory,
            "chunk_memories": chunk_memories,
            "base_chunk_memories": base_chunk_memories,
            "all_memories": all_memories,
            "final_memory": final_memory,
            "visual_evidence": evidence,
            "causal_evidence_tokens": causal_evidence_tokens,
            "final_evidence_state": final_evidence_state,
            "proprio_evidence": proprio_evidence,
            "write_gates": write_gates,
            "event_gates": event_gates,
            "gate_modulations": gate_modulations,
            "effective_write_gates": effective_write_gates,
            "event_update_residual_norms": event_update_residual_norms,
            "hold_update_residual_norms": hold_update_residual_norms,
            "routed_update_residual_norms": routed_update_residual_norms,
            "event_update_routing_probabilities": event_update_routing_probabilities,
            "event_corrections": event_corrections,
            "event_correction_gates": event_correction_gates,
            "event_correction_norms": event_correction_norms,
        }


def weighted_memory_distillation_loss(
    student_memory: jnp.ndarray,
    teacher_memory: jnp.ndarray,
    state_weights: jnp.ndarray,
    *,
    semantic_token_weight: float = 4.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Teacher-memory alignment with separate semantic-token/state weights."""

    if student_memory.shape != teacher_memory.shape:
        raise ValueError(f"Memory shape mismatch: {student_memory.shape} != {teacher_memory.shape}")
    if state_weights.shape != student_memory.shape[:2]:
        raise ValueError(f"State weights {state_weights.shape} do not match {student_memory.shape}")
    student = student_memory.astype(jnp.float32)
    teacher = jax.lax.stop_gradient(teacher_memory.astype(jnp.float32))
    student_unit = student / jnp.maximum(jnp.linalg.norm(student, axis=-1, keepdims=True), 1e-6)
    teacher_unit = teacher / jnp.maximum(jnp.linalg.norm(teacher, axis=-1, keepdims=True), 1e-6)
    cosine_per_token = 1.0 - jnp.sum(student_unit * teacher_unit, axis=-1)
    mse_per_token = jnp.mean(jnp.square(student - teacher), axis=-1)
    token_weights = jnp.ones((student.shape[-2],), dtype=jnp.float32)
    token_weights = token_weights.at[: len(teacher_contract.STATE_FIELDS)].set(semantic_token_weight)
    weights = state_weights.astype(jnp.float32)[..., None] * token_weights
    denominator = jnp.maximum(jnp.sum(weights), 1.0)
    cosine = jnp.sum(cosine_per_token * weights) / denominator
    mse = jnp.sum(mse_per_token * weights) / denominator
    loss = cosine + 0.1 * mse
    return loss, {
        "memory_distill_loss": loss,
        "memory_cosine_loss": cosine,
        "memory_mse_loss": mse,
    }


def oracle_event_correction_delta_loss(
    event_corrections: jnp.ndarray,
    base_chunk_memories: jnp.ndarray,
    teacher_after_memories: jnp.ndarray,
    sequence_mask: jnp.ndarray,
    state_change_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Directly supervise post-base correction on privileged event chunks."""

    expected = base_chunk_memories.shape
    if event_corrections.shape != expected or teacher_after_memories.shape != expected:
        raise ValueError(
            "Expected correction, base, and teacher memories to match: "
            f"{event_corrections.shape}, {base_chunk_memories.shape}, "
            f"{teacher_after_memories.shape}"
        )
    if sequence_mask.shape != expected[:2] or state_change_mask.shape != expected[:2]:
        raise ValueError(
            f"Expected masks {expected[:2]}, got {sequence_mask.shape}, "
            f"{state_change_mask.shape}"
        )
    target = jax.lax.stop_gradient(
        teacher_after_memories.astype(jnp.float32)
        - base_chunk_memories.astype(jnp.float32)
    )
    prediction = event_corrections.astype(jnp.float32)
    per_chunk_mse = jnp.mean(jnp.square(prediction - target), axis=(-2, -1))
    prediction_norm = jnp.sqrt(jnp.mean(jnp.square(prediction), axis=(-2, -1)))
    target_norm = jnp.sqrt(jnp.mean(jnp.square(target), axis=(-2, -1)))
    event_mask = (sequence_mask & state_change_mask).astype(jnp.float32)
    denominator = jnp.maximum(jnp.sum(event_mask), 1.0)
    loss = jnp.sum(per_chunk_mse * event_mask) / denominator
    return loss, {
        "event_correction_delta_loss": loss,
        "event_correction_prediction_rms": jnp.sum(prediction_norm * event_mask)
        / denominator,
        "event_correction_target_rms": jnp.sum(target_norm * event_mask) / denominator,
    }


def direct_teacher_delta_loss(
    student_memories: jnp.ndarray,
    teacher_memories: jnp.ndarray,
    sequence_mask: jnp.ndarray,
    state_change_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Supervise the deployed recurrent update itself with the teacher delta."""

    if student_memories.shape != teacher_memories.shape:
        raise ValueError(
            f"Memory shape mismatch: {student_memories.shape} != {teacher_memories.shape}"
        )
    expected = student_memories.shape[:2]
    if expected != (sequence_mask.shape[0], sequence_mask.shape[1] + 1):
        raise ValueError(
            f"Expected memories [B,T+1,...] for {sequence_mask.shape}, got {expected}"
        )
    if state_change_mask.shape != sequence_mask.shape:
        raise ValueError(
            f"Expected state_change_mask {sequence_mask.shape}, got {state_change_mask.shape}"
        )
    prediction = student_memories[:, 1:].astype(jnp.float32) - student_memories[
        :, :-1
    ].astype(jnp.float32)
    target = jax.lax.stop_gradient(
        teacher_memories[:, 1:].astype(jnp.float32)
        - teacher_memories[:, :-1].astype(jnp.float32)
    )
    per_chunk = jnp.mean(jnp.square(prediction - target), axis=(-2, -1))
    event_mask = (sequence_mask & state_change_mask).astype(jnp.float32)
    denominator = jnp.maximum(jnp.sum(event_mask), 1.0)
    loss = jnp.sum(per_chunk * event_mask) / denominator
    prediction_rms = jnp.sqrt(jnp.mean(jnp.square(prediction), axis=(-2, -1)))
    target_rms = jnp.sqrt(jnp.mean(jnp.square(target), axis=(-2, -1)))
    return loss, {
        "direct_teacher_delta_loss": loss,
        "direct_teacher_delta_prediction_rms": jnp.sum(prediction_rms * event_mask)
        / denominator,
        "direct_teacher_delta_target_rms": jnp.sum(target_rms * event_mask)
        / denominator,
    }


def weighted_state_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    field_mask: jnp.ndarray,
    state_weights: jnp.ndarray,
    field_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Masked state CE that prevents 87% no-change chunks dominating training."""

    if logits.shape[:-1] != targets.shape or targets.shape != field_mask.shape:
        raise ValueError(
            f"State target mismatch: logits={logits.shape}, targets={targets.shape}, mask={field_mask.shape}"
        )
    if state_weights.shape != targets.shape[:2]:
        raise ValueError(f"Expected state weights {targets.shape[:2]}, got {state_weights.shape}")
    token_losses = -jnp.take_along_axis(
        jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1),
        targets[..., None],
        axis=-1,
    )[..., 0]
    fields = field_mask.astype(jnp.float32)
    if field_weights is not None:
        if field_weights.shape != (targets.shape[-1],):
            raise ValueError(
                f"Expected field weights {(targets.shape[-1],)}, got {field_weights.shape}"
            )
        fields = fields * field_weights.astype(jnp.float32)
    fields_per_state = jnp.sum(fields, axis=-1)
    loss_per_state = jnp.sum(token_losses * fields, axis=-1) / jnp.maximum(fields_per_state, 1.0)
    weights = state_weights.astype(jnp.float32)
    return jnp.sum(loss_per_state * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def no_change_memory_consistency_loss(
    all_memories: jnp.ndarray,
    sequence_mask: jnp.ndarray,
    state_change_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Preserve the preceding memory on valid chunks whose teacher state does not change."""

    if all_memories.shape[:2] != (sequence_mask.shape[0], sequence_mask.shape[1] + 1):
        raise ValueError(
            f"Expected memories [B,T+1,...] for mask {sequence_mask.shape}, got {all_memories.shape}"
        )
    if state_change_mask.shape != sequence_mask.shape:
        raise ValueError(
            f"Expected state_change_mask {sequence_mask.shape}, got {state_change_mask.shape}"
        )
    current = all_memories[:, 1:].astype(jnp.float32)
    preceding = jax.lax.stop_gradient(all_memories[:, :-1].astype(jnp.float32))
    per_state = jnp.mean(jnp.square(current - preceding), axis=(-2, -1))
    keep_mask = sequence_mask & ~state_change_mask
    weights = keep_mask.astype(jnp.float32)
    return jnp.sum(per_state * weights) / jnp.maximum(jnp.sum(weights), 1.0)
