"""Goal-conditioned extensions for compact recurrent semantic memory.

This module deliberately builds on :mod:`openpi.models.siglip_mem_semantic`
instead of changing that task-agnostic core.  A task adapter can embed an
instruction with the existing PaliGemma language model, compress the language
sequence into a small number of goal tokens, and use those tokens to
initialize recurrent memory before visual evidence is consumed.

The goal is injected only once at the start of an episode.  Static language
therefore does not need to be repeated at every recurrent update, while the
resulting memory remains compatible with readers that expect a fixed tensor
of shape ``[B, M, D]``.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from openpi.models import siglip_mem_semantic as memory_core


class GoalTokenEncoder(nn.Module):
    """Compress masked language embeddings into compact learned goal tokens.

    The input is expected to come from an existing language embedder, for
    example ``PaliGemma.llm(tokenized_prompt, method="embed")``.  Learned
    queries cross-attend to the prompt, so the output length is independent of
    the prompt length and can be kept small (one token is the default).
    """

    input_width: int = 2048
    width: int = 64
    num_goal_tokens: int = 1
    num_heads: int = 4
    mlp_ratio: int = 4
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, prompt_tokens, *, prompt_mask=None):
        if prompt_tokens.ndim != 3 or prompt_tokens.shape[-1] != self.input_width:
            raise ValueError(f"Expected prompt_tokens [B,L,{self.input_width}], got {prompt_tokens.shape}")
        if prompt_tokens.shape[1] < 1:
            raise ValueError("prompt_tokens must contain at least one token")
        if self.num_goal_tokens < 1:
            raise ValueError(f"num_goal_tokens must be positive, got {self.num_goal_tokens}")
        if self.width % self.num_heads != 0:
            raise ValueError(f"Goal width {self.width} must be divisible by num_heads={self.num_heads}")

        batch, prompt_length, _ = prompt_tokens.shape
        if prompt_mask is None:
            prompt_mask = jnp.ones((batch, prompt_length), dtype=jnp.bool_)
        elif prompt_mask.shape != (batch, prompt_length):
            raise ValueError(f"Expected prompt_mask {(batch, prompt_length)}, got {prompt_mask.shape}")
        else:
            prompt_mask = prompt_mask.astype(jnp.bool_)

        prompt = nn.LayerNorm(name="prompt_ln", dtype=self.dtype_mm)(prompt_tokens)
        prompt = nn.Dense(self.width, name="prompt_projection", dtype=self.dtype_mm)(prompt)
        prompt = jnp.where(prompt_mask[..., None], prompt, jnp.zeros_like(prompt))

        goal_queries = self.param(
            "goal_queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_goal_tokens, self.width),
            jnp.float32,
        )
        goal_queries = jnp.tile(goal_queries, (batch, 1, 1))

        # Attention with no valid keys is ill-defined.  For an empty mask, make
        # the first (already zeroed) prompt position a harmless dummy key.  The
        # learned query still provides a well-defined fallback goal token.
        has_prompt = jnp.any(prompt_mask, axis=1)
        safe_prompt_mask = prompt_mask.at[:, 0].set(prompt_mask[:, 0] | ~has_prompt)
        attention_mask = safe_prompt_mask[:, None, None, :]

        query_norm = nn.LayerNorm(name="query_ln", dtype=self.dtype_mm)(goal_queries)
        attended = nn.MultiHeadDotProductAttention(
            name="prompt_cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(query_norm, prompt, mask=attention_mask)
        goal = goal_queries + attended

        y = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(goal)
        y = nn.Dense(self.width * self.mlp_ratio, name="mlp_in", dtype=self.dtype_mm)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(y)
        return nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(goal + y)


class GoalConditionedMemoryInitializer(nn.Module):
    """Inject goal tokens into reserved slots of fixed-size initial memory."""

    width: int = 64
    num_memory_tokens: int = 128
    num_goal_tokens: int = 1

    @nn.compact
    def __call__(self, goal_tokens):
        expected = (self.num_goal_tokens, self.width)
        if goal_tokens.ndim != 3 or goal_tokens.shape[1:] != expected:
            raise ValueError(f"Expected goal_tokens [B,{expected}], got {goal_tokens.shape}")
        if self.num_memory_tokens < self.num_goal_tokens:
            raise ValueError(
                "num_memory_tokens must be at least num_goal_tokens, got "
                f"{self.num_memory_tokens} and {self.num_goal_tokens}"
            )

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.width),
            jnp.float32,
        )
        memory = jnp.tile(base_memory, (goal_tokens.shape[0], 1, 1))
        return memory.at[:, : self.num_goal_tokens].add(goal_tokens.astype(memory.dtype))


class GoalConditionedRecurrentMemory(nn.Module):
    """Initialize memory from a prompt, then recurrently ingest evidence.

    ``evidence_steps`` uses the same contract as
    :class:`siglip_mem_semantic.RecurrentMemoryUpdater`: ``[B,S,N,D]``.  The
    returned tuple contains ``(final_memory, memory_states, goal_tokens,
    initial_memory)`` so pretraining code can attach auxiliary diagnostics or
    losses without recomputing the language path.
    """

    prompt_width: int = 2048
    memory_width: int = 64
    num_memory_tokens: int = 128
    num_goal_tokens: int = 1
    goal_heads: int = 4
    goal_mlp_ratio: int = 4
    memory_depth: int = 2
    memory_heads: int = 4
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(
        self,
        prompt_tokens,
        evidence_steps,
        *,
        prompt_mask=None,
        step_mask=None,
    ):
        if evidence_steps.ndim != 4 or evidence_steps.shape[-1] != self.memory_width:
            raise ValueError(f"Expected evidence_steps [B,S,N,{self.memory_width}], got {evidence_steps.shape}")
        if evidence_steps.shape[0] != prompt_tokens.shape[0]:
            raise ValueError(
                "prompt_tokens and evidence_steps must have the same batch size, got "
                f"{prompt_tokens.shape[0]} and {evidence_steps.shape[0]}"
            )

        goal_tokens = GoalTokenEncoder(
            name="goal_token_encoder",
            input_width=self.prompt_width,
            width=self.memory_width,
            num_goal_tokens=self.num_goal_tokens,
            num_heads=self.goal_heads,
            mlp_ratio=self.goal_mlp_ratio,
            dtype_mm=self.dtype_mm,
        )(prompt_tokens, prompt_mask=prompt_mask)
        initial_memory = GoalConditionedMemoryInitializer(
            name="goal_conditioned_initializer",
            width=self.memory_width,
            num_memory_tokens=self.num_memory_tokens,
            num_goal_tokens=self.num_goal_tokens,
        )(goal_tokens)
        final_memory, memory_states = memory_core.RecurrentMemoryUpdater(
            name="recurrent_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            dtype_mm=self.dtype_mm,
        )(initial_memory, evidence_steps, step_mask=step_mask)
        return final_memory, memory_states, goal_tokens, initial_memory
