"""Frozen event-memory to next-step EEF action adapter for PickXtimes."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from openpi.models import pi0_mem_semantic_action

VISUAL_FEATURE_DIM = 1152
ROBOT_GOAL_DIM = 19
MEMORY_TOKENS = 128
MEMORY_WIDTH = 64
EEF_POSE_DIM = 6
NUM_PHASES = 3


class PickXtimesEEFActionAdapter(nn.Module):
    """Predict a next absolute EEF7 command from observation and frozen memory.

    ``use_memory`` is static so the action-only control really has no memory
    path.  The two memory controls share the same architecture and differ only
    in whether their frozen tokens came from teacher events or Round-9 causal
    detections.
    """

    hidden_width: int = 256
    depth: int = 3
    memory_query_tokens: int = 8
    use_memory: bool = True

    @nn.compact
    def __call__(self, visual_features, robot_goal, memory, *, train: bool = False):
        if visual_features.ndim != 2 or visual_features.shape[-1] != VISUAL_FEATURE_DIM:
            raise ValueError(f"Expected visual features [B,{VISUAL_FEATURE_DIM}], got {visual_features.shape}")
        if robot_goal.ndim != 2 or robot_goal.shape[-1] != ROBOT_GOAL_DIM:
            raise ValueError(f"Expected robot-goal features [B,{ROBOT_GOAL_DIM}], got {robot_goal.shape}")
        expected_memory = (MEMORY_TOKENS, MEMORY_WIDTH)
        if memory.ndim != 3 or memory.shape[1:] != expected_memory:
            raise ValueError(f"Expected memory [B,{expected_memory}], got {memory.shape}")

        visual = nn.LayerNorm(name="visual_ln", dtype=jnp.float32)(visual_features.astype(jnp.float32))
        visual = nn.gelu(nn.Dense(self.hidden_width, name="visual_projection", dtype=jnp.float32)(visual))
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
                memory_width=MEMORY_WIDTH,
                memory_tokens=MEMORY_TOKENS,
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
        return {
            "normalized_pose": nn.Dense(EEF_POSE_DIM, name="pose_head", dtype=jnp.float32)(x),
            "close_logit": nn.Dense(1, name="gripper_head", dtype=jnp.float32)(x)[..., 0],
            "phase_logits": nn.Dense(NUM_PHASES, name="phase_head", dtype=jnp.float32)(x),
        }
