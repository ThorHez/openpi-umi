"""Continuous waypoint bridge from compact semantic memory to Pi action tokens.

Unlike a task-specific class head, the bridge is supervised directly by
continuous coordinates already present in a future action chunk.  It keeps the
existing memory cross-attention path and adds two deliberately explicit paths:

* a small decoder predicts a continuous future waypoint from compact memory;
* an embedding of that waypoint is added to every diffusion action token.

The decoded waypoint can also be used as an inference-time anchor for absolute
EEF actions.  This prevents a diffusion expert from reading the correct goal
semantics but attenuating them to a very small positional bias.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class SemanticMemoryWaypointConditioner(nn.Module):
    """Condition action tokens and decode a continuous goal from memory."""

    memory_tokens: int = 128
    memory_width: int = 64
    query_tokens: int = 8
    hidden_width: int = 256
    action_width: int = 1024
    num_heads: int = 4
    waypoint_dim: int = 2
    waypoint_injection_scale: float = 1.0
    use_memory_statistics: bool = False
    dtype_mm: str = "bfloat16"
    use_learned_null_memory: bool = False
    residual_gate_init: float = 1.0
    residual_dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        action_tokens,
        semantic_memory,
        *,
        train: bool = False,
        dropout_rng=None,
    ):
        expected_action = (action_tokens.shape[0], action_tokens.shape[1], self.action_width)
        if action_tokens.shape != expected_action:
            raise ValueError(f"Expected action tokens [B,H,{self.action_width}], got {action_tokens.shape}")
        expected_memory = (action_tokens.shape[0], self.memory_tokens, self.memory_width)
        if semantic_memory.shape != expected_memory:
            raise ValueError(f"Expected semantic memory {expected_memory}, got {semantic_memory.shape}")

        if self.use_learned_null_memory:
            null_memory = self.param(
                "null_memory",
                nn.initializers.normal(stddev=0.02),
                (1, self.memory_tokens, self.memory_width),
                jnp.float32,
            )
            is_null = jnp.all(jnp.abs(semantic_memory) < 1e-8, axis=(1, 2), keepdims=True)
            semantic_memory = jnp.where(
                is_null,
                jnp.broadcast_to(null_memory, semantic_memory.shape),
                semantic_memory.astype(jnp.float32),
            )

        # Keep these names identical to SemanticMemoryActionConditioner so an
        # existing action checkpoint initializes the established path exactly.
        normalized_memory = nn.LayerNorm(name="memory_ln", dtype=jnp.float32)(semantic_memory)
        memory = nn.Dense(self.hidden_width, name="memory_in", dtype=self.dtype_mm)(normalized_memory)
        queries = self.param(
            "queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.query_tokens, self.hidden_width),
            jnp.float32,
        )
        queries = jnp.broadcast_to(
            queries, (action_tokens.shape[0], self.query_tokens, self.hidden_width)
        ).astype(memory.dtype)
        queries = queries + nn.MultiHeadDotProductAttention(
            name="query_cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(
            nn.LayerNorm(name="query_ln", dtype=self.dtype_mm)(queries),
            nn.LayerNorm(name="memory_key_ln", dtype=self.dtype_mm)(memory),
        )

        # Decode in float32: the auxiliary target is a normalized continuous
        # action coordinate and benefits from more precision than bfloat16.
        pooled = jnp.mean(
            nn.LayerNorm(name="waypoint_query_ln", dtype=jnp.float32)(queries.astype(jnp.float32)),
            axis=1,
        )
        if self.use_memory_statistics:
            raw_memory = semantic_memory.astype(jnp.float32)
            memory_mean = jnp.mean(raw_memory, axis=1)
            memory_std = jnp.sqrt(
                jnp.mean(jnp.square(raw_memory - memory_mean[:, None, :]), axis=1) + 1e-6
            )
            memory_statistics = jnp.concatenate((memory_mean, memory_std), axis=-1)
            # Zero init makes the statistics branch an exact no-op when an old
            # V1 checkpoint is loaded, then lets training learn only the
            # residual correction supported by these proven linear features.
            statistics_update = nn.Dense(
                self.hidden_width,
                kernel_init=nn.initializers.zeros_init(),
                bias_init=nn.initializers.zeros_init(),
                name="waypoint_statistics_in",
                dtype=jnp.float32,
            )(memory_statistics)
            pooled = pooled + statistics_update
        waypoint_hidden = nn.Dense(
            self.hidden_width,
            name="waypoint_hidden",
            dtype=jnp.float32,
        )(pooled)
        waypoint_hidden = nn.gelu(waypoint_hidden)
        waypoint = nn.Dense(
            self.waypoint_dim,
            name="waypoint_out",
            dtype=jnp.float32,
        )(waypoint_hidden)

        queries = nn.Dense(self.action_width, name="query_out", dtype=self.dtype_mm)(
            nn.LayerNorm(name="query_out_ln", dtype=self.dtype_mm)(queries)
        )
        update = nn.MultiHeadDotProductAttention(
            name="action_cross_attention",
            num_heads=8,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(
            nn.LayerNorm(name="action_ln", dtype=self.dtype_mm)(action_tokens),
            nn.LayerNorm(name="query_key_ln", dtype=self.dtype_mm)(queries),
        )
        if train and self.residual_dropout_rate > 0:
            if dropout_rng is None:
                raise ValueError("dropout_rng is required when training with MEM residual dropout")
            keep_probability = 1.0 - self.residual_dropout_rate
            keep = jax.random.bernoulli(
                dropout_rng,
                keep_probability,
                (action_tokens.shape[0], 1, 1),
            )
            update = update * keep.astype(update.dtype) / keep_probability
        gate_delta = self.param("gate_delta", nn.initializers.zeros_init(), (1,), jnp.float32)
        gate = (self.residual_gate_init + jnp.tanh(gate_delta)).astype(update.dtype)

        # Zero initialization leaves the loaded action policy unchanged before
        # training, while allowing flow loss to learn how strongly to use the
        # explicitly decoded waypoint.
        waypoint_embedding = nn.Dense(
            self.action_width,
            use_bias=False,
            kernel_init=nn.initializers.zeros_init(),
            name="waypoint_embed",
            dtype=self.dtype_mm,
        )(waypoint.astype(self.dtype_mm))
        conditioned = action_tokens + gate * update
        conditioned = conditioned + self.waypoint_injection_scale * waypoint_embedding[:, None, :]
        return conditioned, waypoint
