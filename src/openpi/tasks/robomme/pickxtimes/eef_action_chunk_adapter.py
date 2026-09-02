"""Frozen event-memory to EEF7 action-chunk adapter for PickXtimes."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from openpi.models import pi0_mem_semantic_action
from openpi.tasks.robomme.pickxtimes import eef_action_adapter


class PickXtimesEEFActionChunkAdapter(nn.Module):
    """Predict a short absolute-EEF7 chunk from observation and memory."""

    action_horizon: int = 8
    hidden_width: int = 256
    depth: int = 3
    memory_query_tokens: int = 8
    use_memory: bool = True
    spatial_visual_tokens: int = 0

    @nn.compact
    def __call__(self, visual_features, robot_goal, memory, *, train: bool = False):
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if self.spatial_visual_tokens:
            expected_visual = (self.spatial_visual_tokens, eef_action_adapter.VISUAL_FEATURE_DIM)
            if visual_features.ndim != 3 or visual_features.shape[1:] != expected_visual:
                raise ValueError(f"Expected visual features [B,{expected_visual}], got {visual_features.shape}")
        elif visual_features.ndim != 2 or visual_features.shape[-1] != eef_action_adapter.VISUAL_FEATURE_DIM:
            raise ValueError(f"Expected visual features [B,{eef_action_adapter.VISUAL_FEATURE_DIM}], got {visual_features.shape}")
        if robot_goal.ndim != 2 or robot_goal.shape[-1] != eef_action_adapter.ROBOT_GOAL_DIM:
            raise ValueError(
                f"Expected robot-goal features [B,{eef_action_adapter.ROBOT_GOAL_DIM}], got {robot_goal.shape}"
            )
        expected_memory = (eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH)
        if memory.ndim != 3 or memory.shape[1:] != expected_memory:
            raise ValueError(f"Expected memory [B,{expected_memory}], got {memory.shape}")

        visual = nn.LayerNorm(name="visual_ln", dtype=jnp.float32)(visual_features.astype(jnp.float32))
        visual = nn.gelu(nn.Dense(self.hidden_width, name="visual_projection", dtype=jnp.float32)(visual))
        if self.spatial_visual_tokens:
            visual_queries = self.param(
                "visual_queries",
                nn.initializers.normal(stddev=0.02),
                (1, 4, self.hidden_width),
                jnp.float32,
            )
            visual_queries = jnp.tile(visual_queries, (visual.shape[0], 1, 1))
            visual_queries = nn.MultiHeadDotProductAttention(
                name="visual_cross_attention",
                num_heads=4,
                dropout_rate=0.0,
                deterministic=True,
                dtype=jnp.float32,
            )(visual_queries, visual)
            visual = visual_queries.reshape(visual.shape[0], -1)
            visual = nn.gelu(
                nn.Dense(self.hidden_width, name="visual_query_projection", dtype=jnp.float32)(visual)
            )
        state = nn.gelu(
            nn.Dense(self.hidden_width // 2, name="state_projection", dtype=jnp.float32)(
                robot_goal.astype(jnp.float32)
            )
        )
        x = nn.gelu(
            nn.Dense(self.hidden_width, name="observation_projection", dtype=jnp.float32)(
                jnp.concatenate((visual, state), axis=-1)
            )
        )
        if self.use_memory:
            action_tokens, _ = pi0_mem_semantic_action.MemoryActionInterface(
                name="memory_action_interface",
                memory_width=eef_action_adapter.MEMORY_WIDTH,
                memory_tokens=eef_action_adapter.MEMORY_TOKENS,
                query_width=self.hidden_width // 2,
                query_tokens=self.memory_query_tokens,
                query_depth=1,
                query_heads=4,
                action_width=self.hidden_width,
                action_heads=4,
                dtype_mm="float32",
            )(x[:, None, :], memory.astype(jnp.float32))
            x = action_tokens[:, 0]

        for index in range(self.depth):
            residual = x
            x = nn.LayerNorm(name=f"block_{index}_ln", dtype=jnp.float32)(x)
            x = nn.Dense(self.hidden_width * 2, name=f"block_{index}_expand", dtype=jnp.float32)(x)
            x = nn.gelu(x)
            x = nn.Dropout(0.05, name=f"block_{index}_dropout")(x, deterministic=not train)
            x = nn.Dense(self.hidden_width, name=f"block_{index}_project", dtype=jnp.float32)(x)
            x = residual + x
        x = nn.LayerNorm(name="output_ln", dtype=jnp.float32)(x)
        poses = nn.Dense(
            self.action_horizon * eef_action_adapter.EEF_POSE_DIM,
            name="pose_chunk_head",
            dtype=jnp.float32,
        )(x).reshape(x.shape[0], self.action_horizon, eef_action_adapter.EEF_POSE_DIM)
        close_logits = nn.Dense(
            self.action_horizon,
            name="gripper_chunk_head",
            dtype=jnp.float32,
        )(x)
        return {
            "normalized_poses": poses,
            "close_logits": close_logits,
            "phase_logits": nn.Dense(
                eef_action_adapter.NUM_PHASES,
                name="phase_head",
                dtype=jnp.float32,
            )(x),
        }
