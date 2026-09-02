"""Goal-conditioned recurrent visual memory for RoboMME VideoUnmask."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from openpi.models import siglip_mem_semantic as memory_core
from openpi.models import siglip_mem_semantic_goal as goal_memory

NUM_DEMO_FRAMES = 12
NUM_PATCHES = 256
NUM_SPATIAL_TOKENS = 64
NUM_COLORS = 3


class MemoryPointReadout(nn.Module):
    width: int = 64

    @nn.compact
    def __call__(self, memories):
        if memories.ndim != 4 or memories.shape[-1] != self.width:
            raise ValueError(f"Expected memories [B,S,M,{self.width}], got {memories.shape}")
        x = nn.LayerNorm(name="memory_ln", dtype=jnp.float32)(memories)
        weights = nn.softmax(nn.Dense(1, name="attention", dtype=jnp.float32)(x), axis=2)
        pooled = jnp.sum(weights * x, axis=2)
        hidden = nn.gelu(nn.Dense(self.width, name="hidden", dtype=jnp.float32)(pooled))
        return nn.sigmoid(nn.Dense(2, name="point_head", dtype=jnp.float32)(hidden))


class VideoUnmaskSemanticMemory(nn.Module):
    input_width: int = 1152
    prompt_width: int = 2048
    encoder_width: int = 64
    encoder_depth: int = 2
    encoder_heads: int = 4
    memory_width: int = 64
    memory_depth: int = 1
    memory_heads: int = 4
    num_memory_tokens: int = 32
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, demo_patch_tokens, prompt_tokens, prompt_mask, frame_mask, *, train: bool = False):
        expected = (NUM_DEMO_FRAMES, NUM_PATCHES, self.input_width)
        if demo_patch_tokens.ndim != 4 or demo_patch_tokens.shape[1:] != expected:
            raise ValueError(f"Expected demo_patch_tokens [B,{expected}], got {demo_patch_tokens.shape}")
        if frame_mask.shape != demo_patch_tokens.shape[:2]:
            raise ValueError(f"Expected frame_mask {demo_patch_tokens.shape[:2]}, got {frame_mask.shape}")

        x = memory_core.pool_fixed_grid(demo_patch_tokens, pool_factor=2)
        x = nn.LayerNorm(name="visual_input_ln", dtype=self.dtype_mm)(x)
        x = nn.Dense(self.encoder_width, name="visual_projection", dtype=self.dtype_mm)(x)
        temporal_pos = self.param(
            "temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, NUM_DEMO_FRAMES, 1, self.encoder_width),
            x.dtype,
        )
        spatial_pos = self.param(
            "spatial_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, NUM_SPATIAL_TOKENS, self.encoder_width),
            x.dtype,
        )
        x = x + temporal_pos + spatial_pos
        for index in range(self.encoder_depth):
            x = memory_core.FactorizedSpaceTimeBlock(
                name=f"visual_block_{index}",
                width=self.encoder_width,
                num_heads=self.encoder_heads,
                dropout=0.0,
                dtype_mm=self.dtype_mm,
            )(x, train=train)
        x = nn.LayerNorm(name="visual_output_ln", dtype=jnp.float32)(x)
        evidence = nn.Dense(self.memory_width, name="evidence_projection", dtype=jnp.float32)(x)
        evidence = jnp.where(frame_mask[:, :, None, None], evidence, jnp.zeros_like(evidence))

        final_memory, stage_memories, goal_tokens, initial_memory = goal_memory.GoalConditionedRecurrentMemory(
            name="goal_conditioned_recurrent_memory",
            prompt_width=self.prompt_width,
            memory_width=self.memory_width,
            num_memory_tokens=self.num_memory_tokens,
            num_goal_tokens=1,
            goal_heads=4,
            memory_depth=self.memory_depth,
            memory_heads=self.memory_heads,
            dtype_mm="float32",
        )(prompt_tokens, evidence, prompt_mask=prompt_mask, step_mask=frame_mask)
        stage_points = MemoryPointReadout(name="memory_point_readout", width=self.memory_width)(stage_memories)
        goal = nn.LayerNorm(name="goal_readout_ln", dtype=jnp.float32)(goal_tokens[:, 0])
        color_logits = nn.Dense(NUM_COLORS, name="goal_color_head", dtype=jnp.float32)(goal)
        return {
            "target_point": stage_points[:, -1],
            "stage_target_points": stage_points,
            "target_color_logits": color_logits,
            "memory": final_memory,
            "stage_memories": stage_memories,
            "initial_memory": initial_memory,
            "goal_tokens": goal_tokens,
        }
