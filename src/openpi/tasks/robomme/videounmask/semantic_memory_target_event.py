"""ShellGame-style semantic-event memory for RoboMME VideoUnmask."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from openpi.models import siglip_mem_semantic as memory_core
from openpi.models import siglip_mem_semantic_goal as goal_memory

NUM_DEMO_FRAMES = 12
NUM_VISIBLE_FRAMES = 6
NUM_PATCHES = 256
NUM_SPATIAL_TOKENS = 64
NUM_CELLS = 64
NUM_COLORS = 3


class MemoryTargetReadout(nn.Module):
    width: int = 64

    @nn.compact
    def __call__(self, memory):
        x = nn.LayerNorm(name="memory_ln", dtype=jnp.float32)(memory)
        weights = nn.softmax(nn.Dense(1, name="attention", dtype=jnp.float32)(x), axis=1)
        pooled = jnp.sum(weights * x, axis=1)
        hidden = nn.gelu(nn.Dense(self.width, name="hidden", dtype=jnp.float32)(pooled))
        return {
            "memory_cell_logits": nn.Dense(NUM_CELLS, name="cell_head", dtype=jnp.float32)(hidden),
            "target_point": nn.sigmoid(nn.Dense(2, name="point_head", dtype=jnp.float32)(hidden)),
        }


class VideoUnmaskTargetEventMemory(nn.Module):
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

        # In VideoUnmask the first six sampled frames show all colored cubes;
        # later frames contain the same locations after opaque bins appear.
        visible = demo_patch_tokens[:, :NUM_VISIBLE_FRAMES]
        x = memory_core.pool_fixed_grid(visible, pool_factor=2)
        x = nn.LayerNorm(name="visual_input_ln", dtype=self.dtype_mm)(x)
        x = nn.Dense(self.encoder_width, name="visual_projection", dtype=self.dtype_mm)(x)
        temporal_pos = self.param(
            "visible_temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, NUM_VISIBLE_FRAMES, 1, self.encoder_width),
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
                name=f"visible_visual_block_{index}",
                width=self.encoder_width,
                num_heads=self.encoder_heads,
                dropout=0.0,
                dtype_mm=self.dtype_mm,
            )(x, train=train)
        # Cube locations are static before masking; temporal averaging improves
        # signal-to-noise while retaining the 8x8 spatial topology.
        spatial = nn.LayerNorm(name="visible_output_ln", dtype=jnp.float32)(jnp.mean(x, axis=1))

        locator_goal = goal_memory.GoalTokenEncoder(
            name="locator_goal_encoder",
            input_width=self.prompt_width,
            width=self.encoder_width,
            num_goal_tokens=1,
            num_heads=4,
            dtype_mm="float32",
        )(prompt_tokens, prompt_mask=prompt_mask)[:, 0]
        query = nn.Dense(self.encoder_width, name="locator_query", dtype=jnp.float32)(locator_goal)
        keys = nn.Dense(self.encoder_width, name="locator_keys", dtype=jnp.float32)(spatial)
        locator_cell_logits = jnp.einsum("bd,bnd->bn", query, keys) / jnp.sqrt(self.encoder_width)
        locator_weights = nn.softmax(locator_cell_logits, axis=-1)
        target_semantic = jnp.sum(locator_weights[..., None] * spatial, axis=1)
        locator_hidden = nn.gelu(
            nn.Dense(self.encoder_width, name="locator_hidden", dtype=jnp.float32)(target_semantic)
        )
        locator_point = nn.sigmoid(
            nn.Dense(2, name="locator_point_head", dtype=jnp.float32)(locator_hidden)
        )

        evidence = nn.Dense(self.memory_width, name="semantic_event_projection", dtype=jnp.float32)(
            target_semantic
        )[:, None, None]
        final_memory, stage_memories, memory_goal, initial_memory = goal_memory.GoalConditionedRecurrentMemory(
            name="goal_conditioned_recurrent_memory",
            prompt_width=self.prompt_width,
            memory_width=self.memory_width,
            num_memory_tokens=self.num_memory_tokens,
            num_goal_tokens=1,
            goal_heads=4,
            memory_depth=self.memory_depth,
            memory_heads=self.memory_heads,
            dtype_mm="float32",
        )(
            prompt_tokens,
            evidence,
            prompt_mask=prompt_mask,
            step_mask=jnp.ones((demo_patch_tokens.shape[0], 1), dtype=jnp.bool_),
        )
        memory_outputs = MemoryTargetReadout(name="memory_target_readout", width=self.memory_width)(final_memory)
        color_logits = nn.Dense(NUM_COLORS, name="goal_color_head", dtype=jnp.float32)(locator_goal)
        return {
            "locator_cell_logits": locator_cell_logits,
            "locator_point": locator_point,
            "target_color_logits": color_logits,
            "memory": final_memory,
            "stage_memories": stage_memories,
            "initial_memory": initial_memory,
            "goal_tokens": memory_goal,
            **memory_outputs,
        }
