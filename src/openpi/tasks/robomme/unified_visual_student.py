"""Unified direct-visual recurrent student for four RoboMME memory tasks."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import siglip_mem_semantic as memory_core
from openpi.tasks.robomme import unified_gt_teacher as teacher_contract


class VisualWindowEncoder(nn.Module):
    """Encode one causal 12-frame frozen-SigLIP window into evidence tokens."""

    frames: int = 12
    spatial_tokens: int = 16
    input_width: int = 1152
    width: int = 64
    encoder_width: int = 128
    depth: int = 2
    num_heads: int = 8
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        expected = (self.frames, self.spatial_tokens, self.input_width)
        if patch_tokens.ndim != 4 or patch_tokens.shape[1:] != expected:
            raise ValueError(f"Expected visual windows [B,{expected}], got {patch_tokens.shape}")
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(patch_tokens)
        x = nn.Dense(self.encoder_width, name="input_projection", dtype=self.dtype_mm)(x)
        temporal_position = self.param(
            "relative_temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.frames, 1, self.encoder_width),
            jnp.float32,
        )
        spatial_position = self.param(
            "spatial_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.spatial_tokens, self.encoder_width),
            jnp.float32,
        )
        x = x + temporal_position.astype(x.dtype) + spatial_position.astype(x.dtype)
        for index in range(self.depth):
            x = memory_core.FactorizedSpaceTimeBlock(
                name=f"block_{index}",
                width=self.encoder_width,
                num_heads=self.num_heads,
                dropout=0.0,
                dtype_mm=self.dtype_mm,
            )(x, train=train)
        x = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(x)
        x = nn.Dense(self.width, name="memory_projection", dtype=jnp.float32)(x.astype(jnp.float32))
        return x.reshape(x.shape[0], self.frames * self.spatial_tokens, self.width)


class UnifiedVisualRecurrentStudent(nn.Module):
    """Goal-initialize memory, then recurrently consume causal visual windows."""

    max_steps: int = 12
    frames: int = 12
    spatial_tokens: int = 16
    input_width: int = 1152
    width: int = 64
    num_memory_tokens: int = 128
    encoder_width: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 8
    memory_depth: int = 2
    memory_heads: int = 4
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(
        self,
        patch_tokens: jnp.ndarray,
        task_ids: jnp.ndarray,
        goal_color_ids: jnp.ndarray,
        required_counts: jnp.ndarray,
        queried_ordinals: jnp.ndarray,
        num_regions: jnp.ndarray,
        step_mask: jnp.ndarray,
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
            raise ValueError(f"Expected patch_tokens {expected}, got {patch_tokens.shape}")
        if step_mask.shape != (batch, self.max_steps):
            raise ValueError(f"Expected step_mask {(batch, self.max_steps)}, got {step_mask.shape}")
        if goal_color_ids.shape != (batch, 2):
            raise ValueError(f"Expected goal_color_ids {(batch, 2)}, got {goal_color_ids.shape}")

        flat_windows = patch_tokens.reshape(
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
            dtype_mm=self.dtype_mm,
        )(flat_windows, train=train)
        evidence = evidence.reshape(
            batch,
            self.max_steps,
            self.frames * self.spatial_tokens,
            self.width,
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
        final_memory, event_memories = memory_core.RecurrentMemoryUpdater(
            name="shared_visual_memory_updater",
            width=self.width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            dtype_mm="float32",
        )(initial_memory, evidence, step_mask=step_mask)
        all_memories = jnp.concatenate((initial_memory[:, None], event_memories), axis=1)
        return {
            "initial_memory": initial_memory,
            "event_memories": event_memories,
            "all_memories": all_memories,
            "final_memory": final_memory,
            "visual_evidence": evidence,
        }


def memory_distillation_loss(
    student_memory: jnp.ndarray,
    teacher_memory: jnp.ndarray,
    valid_state_mask: jnp.ndarray,
    *,
    semantic_token_weight: float = 4.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Token-aligned cosine/MSE, upweighting the 19 state-bearing tokens."""

    if student_memory.shape != teacher_memory.shape:
        raise ValueError(f"Memory shape mismatch: {student_memory.shape} != {teacher_memory.shape}")
    if valid_state_mask.shape != student_memory.shape[:2]:
        raise ValueError(f"Invalid state mask {valid_state_mask.shape} for {student_memory.shape}")
    student = student_memory.astype(jnp.float32)
    teacher = jax.lax.stop_gradient(teacher_memory.astype(jnp.float32))
    student_unit = student / jnp.maximum(jnp.linalg.norm(student, axis=-1, keepdims=True), 1e-6)
    teacher_unit = teacher / jnp.maximum(jnp.linalg.norm(teacher, axis=-1, keepdims=True), 1e-6)
    cosine_per_token = 1.0 - jnp.sum(student_unit * teacher_unit, axis=-1)
    mse_per_token = jnp.mean(jnp.square(student - teacher), axis=-1)
    token_weights = jnp.ones((student.shape[-2],), dtype=jnp.float32)
    token_weights = token_weights.at[: len(teacher_contract.STATE_FIELDS)].set(semantic_token_weight)
    weights = valid_state_mask.astype(jnp.float32)[..., None] * token_weights
    denominator = jnp.maximum(jnp.sum(weights), 1.0)
    cosine = jnp.sum(cosine_per_token * weights) / denominator
    mse = jnp.sum(mse_per_token * weights) / denominator
    loss = cosine + 0.1 * mse
    return loss, {"memory_distill_loss": loss, "memory_cosine_loss": cosine, "memory_mse_loss": mse}
