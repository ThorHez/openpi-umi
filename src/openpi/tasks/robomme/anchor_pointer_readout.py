"""Shared episode-local anchor pointer for RoboMME semantic memory."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class AnchorPointerReadout(nn.Module):
    """Point from recurrent memory to one episode-local visual anchor.

    The module is deliberately shared by all tasks. Task, queried color and
    ordinal are query tokens; no routed task-specific heads are used.
    """

    width: int = 128
    num_heads: int = 4
    memory_width: int = 64
    anchor_width: int = 1152
    max_anchors: int = 4

    @nn.compact
    def __call__(
        self,
        memory: jnp.ndarray,
        anchor_tokens: jnp.ndarray,
        anchor_yx: jnp.ndarray,
        anchor_mask: jnp.ndarray,
        task_ids: jnp.ndarray,
        query_color_ids: jnp.ndarray,
        queried_ordinals: jnp.ndarray,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        batch = memory.shape[0]
        if memory.shape[-1] != self.memory_width:
            raise ValueError(f"Expected memory width {self.memory_width}, got {memory.shape}")
        expected_anchor = (batch, self.max_anchors, self.anchor_width)
        if anchor_tokens.shape != expected_anchor:
            raise ValueError(f"Expected anchor tokens {expected_anchor}, got {anchor_tokens.shape}")
        if anchor_yx.shape != (batch, self.max_anchors, 2):
            raise ValueError(f"Invalid anchor coordinates: {anchor_yx.shape}")
        if anchor_mask.shape != (batch, self.max_anchors):
            raise ValueError(f"Invalid anchor mask: {anchor_mask.shape}")

        memory_tokens = nn.Dense(self.width, name="memory_projection")(
            nn.LayerNorm(name="memory_input_ln")(memory.astype(jnp.float32))
        )
        task_token = nn.Embed(4, self.width, name="task_embedding")(task_ids)
        color_token = nn.Embed(4, self.width, name="color_embedding")(query_color_ids)
        ordinal_token = nn.Embed(5, self.width, name="ordinal_embedding")(queried_ordinals)
        goal_tokens = jnp.stack((task_token, color_token, ordinal_token), axis=1)
        context = jnp.concatenate((memory_tokens, goal_tokens), axis=1)

        query_seed = self.param(
            "query_seed",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.width),
            jnp.float32,
        )
        query = jnp.tile(query_seed, (batch, 1, 1))
        query = query + task_token[:, None] + color_token[:, None] + ordinal_token[:, None]
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            name="memory_query_attention",
        )(query, context, deterministic=not train)
        query = nn.LayerNorm(name="query_attention_ln")(query + attended)
        query_hidden = nn.gelu(nn.Dense(self.width * 2, name="query_hidden")(query))
        query = nn.LayerNorm(name="query_output_ln")(
            query + nn.Dense(self.width, name="query_out")(query_hidden)
        )[:, 0]

        visual_keys = nn.Dense(self.width, name="anchor_visual_projection")(
            nn.LayerNorm(name="anchor_visual_ln")(anchor_tokens.astype(jnp.float32))
        )
        y, x = anchor_yx[..., 0], anchor_yx[..., 1]
        coordinate_features = jnp.stack(
            (
                y,
                x,
                y * y,
                x * x,
                y * x,
                jnp.sin(jnp.pi * y),
                jnp.cos(jnp.pi * y),
                jnp.sin(jnp.pi * x),
                jnp.cos(jnp.pi * x),
            ),
            axis=-1,
        )
        coordinate_keys = nn.Dense(self.width, name="anchor_coordinate_out")(
            nn.gelu(
                nn.Dense(self.width * 2, name="anchor_coordinate_hidden")(
                    coordinate_features
                )
            )
        )
        keys = nn.LayerNorm(name="anchor_key_ln")(visual_keys + coordinate_keys)
        logits = jnp.einsum("bd,bkd->bk", query, keys) / jnp.sqrt(float(self.width))
        return jnp.where(anchor_mask, logits, jnp.finfo(jnp.float32).min)
