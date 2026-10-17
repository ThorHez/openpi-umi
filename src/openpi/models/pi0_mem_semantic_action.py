"""Task-agnostic interfaces from compact memory to Pi action tokens.

The modules in this file only define tensor transformations.  They do not
encode a task's event vocabulary, frame schedule, action schema, loss weights,
or dataset conventions.  A task adapter supplies compact memory, while these
components resample it and expose it to an action expert.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class RawMemoryQueryResampler(nn.Module):
    """Resample compact memory into action-expert-width tokens."""

    input_width: int = 64
    width: int = 256
    output_width: int = 1024
    input_tokens: int = 128
    query_tokens: int = 16
    depth: int = 2
    num_heads: int = 4
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, memory):
        if memory.ndim != 3 or memory.shape[1:] != (self.input_tokens, self.input_width):
            raise ValueError(f"Expected memory [B,{self.input_tokens},{self.input_width}], got {memory.shape}")
        batch_size = memory.shape[0]
        position = self.param(
            "memory_position_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.input_tokens, self.input_width),
            jnp.float32,
        )
        memory = nn.LayerNorm(name="memory_input_ln", dtype=jnp.float32)(
            memory.astype(jnp.float32) + position
        )
        memory = nn.Dense(self.width, name="memory_projection", dtype=self.dtype_mm)(memory)

        learned_queries = self.param(
            "learned_queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.query_tokens, self.width),
            jnp.float32,
        )
        queries = jnp.tile(learned_queries, (batch_size, 1, 1)).astype(memory.dtype)
        for layer in range(self.depth):
            query_norm = nn.LayerNorm(name=f"query_ln_{layer}", dtype=self.dtype_mm)(queries)
            memory_norm = nn.LayerNorm(name=f"memory_ln_{layer}", dtype=self.dtype_mm)(memory)
            update = nn.MultiHeadDotProductAttention(
                name=f"query_cross_attention_{layer}",
                num_heads=self.num_heads,
                dropout_rate=0.0,
                deterministic=True,
                dtype=self.dtype_mm,
            )(query_norm, memory_norm)
            queries = queries + update
            mlp_input = nn.LayerNorm(name=f"mlp_ln_{layer}", dtype=self.dtype_mm)(queries)
            hidden = nn.Dense(self.width * 4, name=f"mlp_in_{layer}", dtype=self.dtype_mm)(mlp_input)
            hidden = nn.gelu(hidden)
            queries = queries + nn.Dense(
                self.width, name=f"mlp_out_{layer}", dtype=self.dtype_mm
            )(hidden)

        queries = nn.LayerNorm(name="query_output_ln", dtype=self.dtype_mm)(queries)
        queries = nn.Dense(
            self.output_width, name="action_width_projection", dtype=self.dtype_mm
        )(queries)
        return nn.LayerNorm(name="action_width_ln", dtype=self.dtype_mm)(queries)


class ActionMemoryCrossAttention(nn.Module):
    """Give action tokens a direct memory read through a gated residual."""

    width: int = 1024
    num_heads: int = 8
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, action_tokens, memory_tokens):
        if action_tokens.shape[-1] != self.width or memory_tokens.shape[-1] != self.width:
            raise ValueError(f"Expected width {self.width}, got {action_tokens.shape} and {memory_tokens.shape}")
        action_norm = nn.LayerNorm(name="action_ln", dtype=self.dtype_mm)(action_tokens)
        memory_norm = nn.LayerNorm(name="memory_ln", dtype=self.dtype_mm)(memory_tokens)
        update = nn.MultiHeadDotProductAttention(
            name="cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(action_norm, memory_norm)
        gate_delta = self.param("gate_delta", nn.initializers.zeros_init(), (1,), jnp.float32)
        gate = (1.0 + jnp.tanh(gate_delta)).astype(update.dtype)
        conditioned = action_tokens + gate * update
        mlp_input = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(conditioned)
        hidden = nn.Dense(self.width * 2, name="mlp_in", dtype=self.dtype_mm)(mlp_input)
        hidden = nn.gelu(hidden)
        mlp_update = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(hidden)
        return conditioned + gate * mlp_update


class MemoryActionInterface(nn.Module):
    """Task-neutral composition of memory resampling and action conditioning."""

    memory_width: int = 64
    memory_tokens: int = 128
    query_width: int = 256
    query_tokens: int = 16
    query_depth: int = 2
    query_heads: int = 4
    action_width: int = 1024
    action_heads: int = 8
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, action_tokens, memory):
        memory_tokens = RawMemoryQueryResampler(
            name="memory_resampler",
            input_width=self.memory_width,
            width=self.query_width,
            output_width=self.action_width,
            input_tokens=self.memory_tokens,
            query_tokens=self.query_tokens,
            depth=self.query_depth,
            num_heads=self.query_heads,
            dtype_mm=self.dtype_mm,
        )(memory)
        conditioned = ActionMemoryCrossAttention(
            name="action_cross_attention",
            width=self.action_width,
            num_heads=self.action_heads,
            dtype_mm=self.dtype_mm,
        )(action_tokens, memory_tokens)
        return conditioned, memory_tokens
