"""Unified fixed-chunk student with explicit recurrent semantic-state feedback."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.tasks.robomme import unified_gt_teacher as teacher_contract
from openpi.tasks.robomme.unified_visual_student import VisualWindowEncoder


def _mask_invalid_classes(logits: jnp.ndarray) -> jnp.ndarray:
    class_ids = jnp.arange(teacher_contract.MAX_FIELD_CLASSES)[None, :]
    valid = class_ids < jnp.asarray(teacher_contract.FIELD_CLASS_COUNTS)[:, None]
    return jnp.where(valid[None], logits, jnp.asarray(-1e9, dtype=logits.dtype))


def targets_to_logits(targets: jnp.ndarray) -> jnp.ndarray:
    """Convert categorical semantic states to stable normalized feedback logits."""

    one_hot = jax.nn.one_hot(targets, teacher_contract.MAX_FIELD_CLASSES)
    logits = jnp.where(one_hot > 0, 0.0, -12.0)
    return _mask_invalid_classes(logits)


class SemanticFeedbackScanCell(nn.Module):
    """Update shared semantic field logits from state feedback and observation evidence."""

    width: int = 64
    straight_through_hard_feedback: bool = False

    @nn.compact
    def __call__(
        self,
        carry_logits: jnp.ndarray,
        evidence: jnp.ndarray,
        valid: jnp.ndarray,
        field_mask: jnp.ndarray,
        teacher_previous_logits: jnp.ndarray,
        teacher_force: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        feedback_logits = jnp.where(
            teacher_force[:, None, None], teacher_previous_logits, carry_logits
        )
        if self.straight_through_hard_feedback:
            hard_targets = jnp.argmax(feedback_logits, axis=-1)
            hard_logits = targets_to_logits(hard_targets)
            # Forward with a discrete semantic state while retaining the identity
            # gradient with respect to the student's soft recurrent logits.
            feedback_logits = feedback_logits + jax.lax.stop_gradient(
                hard_logits - feedback_logits
            )
        feedback_log_probs = jax.nn.log_softmax(feedback_logits, axis=-1)
        probabilities = jax.nn.softmax(feedback_logits, axis=-1)
        value_embedding = self.param(
            "semantic_value_embedding",
            nn.initializers.normal(stddev=0.02),
            (
                len(teacher_contract.STATE_FIELDS),
                teacher_contract.MAX_FIELD_CLASSES,
                self.width,
            ),
            jnp.float32,
        )
        field_type = self.param(
            "semantic_field_type",
            nn.initializers.normal(stddev=0.02),
            (1, len(teacher_contract.STATE_FIELDS), self.width),
            jnp.float32,
        )
        semantic_tokens = jnp.einsum("bfc,fcw->bfw", probabilities, value_embedding)
        semantic_tokens = (semantic_tokens + field_type) * field_mask[..., None]
        global_state = jnp.sum(semantic_tokens, axis=1) / jnp.maximum(
            jnp.sum(field_mask, axis=1, keepdims=True), 1.0
        )
        observation = nn.gelu(nn.Dense(self.width * 2, name="observation_in")(evidence))
        observation = nn.gelu(nn.Dense(self.width, name="observation_out")(observation))
        shared = jnp.concatenate(
            (
                semantic_tokens,
                jnp.broadcast_to(global_state[:, None], semantic_tokens.shape),
                jnp.broadcast_to(observation[:, None], semantic_tokens.shape),
            ),
            axis=-1,
        )
        hidden = nn.gelu(nn.Dense(self.width * 4, name="transition_hidden")(shared))
        hidden = nn.LayerNorm(name="transition_ln")(
            hidden
            + nn.Dense(self.width * 4, name="transition_residual")(
                nn.gelu(nn.Dense(self.width * 4, name="transition_residual_hidden")(hidden))
            )
        )
        delta_logits = nn.Dense(
            teacher_contract.MAX_FIELD_CLASSES,
            name="shared_semantic_delta",
            kernel_init=nn.initializers.zeros_init(),
        )(hidden)
        next_logits = _mask_invalid_classes(feedback_log_probs + delta_logits)
        next_logits = jnp.where(valid[:, None, None], next_logits, carry_logits)
        return next_logits, next_logits


class UnifiedSemanticFeedbackStudent(nn.Module):
    """Consume fixed RGB/proprio chunks while recurrently feeding back semantic state."""

    max_steps: int = 96
    frames: int = 12
    spatial_tokens: int = 16
    input_width: int = 1152
    proprio_dim: int = 0
    width: int = 64
    encoder_width: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 8
    dtype_mm: str = "bfloat16"
    straight_through_hard_feedback: bool = False

    @nn.compact
    def __call__(
        self,
        patch_tokens: jnp.ndarray,
        proprio: jnp.ndarray,
        sequence_mask: jnp.ndarray,
        initial_state_targets: jnp.ndarray,
        state_field_mask: jnp.ndarray,
        teacher_previous_targets: jnp.ndarray,
        teacher_force_mask: jnp.ndarray,
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
            raise ValueError(f"Invalid sequence mask {sequence_mask.shape}")
        if initial_state_targets.shape != (batch, len(teacher_contract.STATE_FIELDS)):
            raise ValueError(f"Invalid initial semantic state {initial_state_targets.shape}")
        expected_states = (batch, self.max_steps, len(teacher_contract.STATE_FIELDS))
        if teacher_previous_targets.shape != expected_states:
            raise ValueError(f"Invalid teacher previous states {teacher_previous_targets.shape}")
        if state_field_mask.shape != expected_states:
            raise ValueError(f"Invalid state field mask {state_field_mask.shape}")
        if teacher_force_mask.shape != (batch, self.max_steps):
            raise ValueError(f"Invalid teacher force mask {teacher_force_mask.shape}")
        expected_proprio = (batch, self.max_steps, self.frames, self.proprio_dim)
        if proprio.shape != expected_proprio:
            raise ValueError(f"Expected proprio {expected_proprio}, got {proprio.shape}")

        flat = patch_tokens.reshape(
            batch * self.max_steps,
            self.frames,
            self.spatial_tokens,
            self.input_width,
        )
        visual = VisualWindowEncoder(
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
            batch, self.max_steps, self.frames, self.spatial_tokens, self.width
        )
        visual = visual.astype(jnp.float32)
        visual_early = jnp.mean(visual[:, :, :6], axis=(2, 3))
        visual_late = jnp.mean(visual[:, :, 6:], axis=(2, 3))
        visual_delta = visual_late - visual_early
        visual_summary = jnp.concatenate(
            (
                jnp.mean(visual, axis=(2, 3)),
                visual_early,
                visual_late,
                visual_delta,
                jnp.abs(visual_delta),
            ),
            axis=-1,
        )
        proprio_flat = proprio.reshape(batch * self.max_steps, self.frames, self.proprio_dim)
        proprio_tokens = nn.Dense(self.width, name="proprio_input")(proprio_flat)
        proprio_position = self.param(
            "proprio_position",
            nn.initializers.normal(stddev=0.02),
            (1, self.frames, self.width),
            jnp.float32,
        )
        proprio_tokens = proprio_tokens + proprio_position
        proprio_tokens = nn.LayerNorm(name="proprio_ln")(
            proprio_tokens
            + nn.Dense(self.width, name="proprio_out")(
                nn.gelu(nn.Dense(self.width * 2, name="proprio_hidden")(proprio_tokens))
            )
        )
        proprio_tokens = proprio_tokens.reshape(
            batch, self.max_steps, self.frames, self.width
        )
        proprio_early = jnp.mean(proprio_tokens[:, :, :6], axis=2)
        proprio_late = jnp.mean(proprio_tokens[:, :, 6:], axis=2)
        proprio_delta = proprio_late - proprio_early
        proprio_summary = jnp.concatenate(
            (
                jnp.mean(proprio_tokens, axis=2),
                proprio_early,
                proprio_late,
                proprio_delta,
                jnp.abs(proprio_delta),
            ),
            axis=-1,
        )
        evidence = nn.gelu(
            nn.Dense(self.width * 2, name="multimodal_summary")(
                jnp.concatenate((visual_summary, proprio_summary), axis=-1)
            )
        )
        initial_logits = targets_to_logits(initial_state_targets)
        teacher_logits = targets_to_logits(teacher_previous_targets)
        scanned_cell = nn.scan(
            SemanticFeedbackScanCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(
            name="shared_semantic_updater",
            width=self.width,
            straight_through_hard_feedback=self.straight_through_hard_feedback,
        )
        final_logits, chunk_logits = scanned_cell(
            initial_logits,
            evidence,
            sequence_mask,
            state_field_mask,
            teacher_logits,
            teacher_force_mask,
        )
        all_logits = jnp.concatenate((initial_logits[:, None], chunk_logits), axis=1)
        probabilities = jax.nn.softmax(all_logits, axis=-1)
        return {
            "state_logits": all_logits,
            "state_probabilities": probabilities,
            "initial_state_logits": initial_logits,
            "chunk_state_logits": chunk_logits,
            "final_state_logits": final_logits,
            "multimodal_evidence": evidence,
        }
