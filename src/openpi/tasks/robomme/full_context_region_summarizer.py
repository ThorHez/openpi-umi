"""Bidirectional full-demonstration region summarizer for RoboMME probes."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class FullContextRegionSummarizer(nn.Module):
    """Summarize all visual chunks once and classify an episode-local region.

    This deliberately has no recurrent bottleneck or causal deployment
    constraint.  It is an observation-sufficiency ceiling, not the final MEM.
    """

    width: int = 64
    num_heads: int = 4
    depth: int = 2
    feature_width: int = 3456
    max_chunks: int = 96
    spatial_tokens: int = 16
    max_regions: int = 4

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        chunk_mask: jnp.ndarray,
        task_ids: jnp.ndarray,
        query_color_ids: jnp.ndarray,
        queried_ordinals: jnp.ndarray,
        num_regions: jnp.ndarray,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        del train
        batch = features.shape[0]
        expected = (batch, self.max_chunks, self.spatial_tokens, self.feature_width)
        if features.shape != expected:
            raise ValueError(f"Expected features {expected}, got {features.shape}")
        if chunk_mask.shape != (batch, self.max_chunks):
            raise ValueError(f"Invalid chunk mask {chunk_mask.shape}")

        x = nn.Dense(self.width, name="visual_projection")(
            nn.LayerNorm(name="visual_input_ln")(features.astype(jnp.float32))
        )
        temporal = self.param(
            "temporal_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.max_chunks, 1, self.width),
        )
        spatial = self.param(
            "spatial_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.spatial_tokens, self.width),
        )
        x = x + temporal + spatial

        # Temporal attention is bidirectional and is applied independently at
        # each pooled 4x4 spatial cell before global goal-conditioned pooling.
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(
            batch * self.spatial_tokens, self.max_chunks, self.width
        )
        temporal_mask = jnp.repeat(chunk_mask, self.spatial_tokens, axis=0)
        attention_mask = nn.make_attention_mask(temporal_mask, temporal_mask)
        for layer in range(self.depth):
            residual = x
            normalized = nn.LayerNorm(name=f"temporal_ln_{layer}")(x)
            x = residual + nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.width,
                out_features=self.width,
                name=f"temporal_attention_{layer}",
            )(normalized, mask=attention_mask)
            residual = x
            normalized = nn.LayerNorm(name=f"temporal_mlp_ln_{layer}")(x)
            hidden = nn.gelu(
                nn.Dense(self.width * 2, name=f"temporal_mlp_in_{layer}")(
                    normalized
                )
            )
            x = residual + nn.Dense(
                self.width, name=f"temporal_mlp_out_{layer}"
            )(hidden)

        x = x.reshape(batch, self.spatial_tokens, self.max_chunks, self.width)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(
            batch, self.max_chunks * self.spatial_tokens, self.width
        )
        visual_mask = jnp.repeat(chunk_mask, self.spatial_tokens, axis=1)

        task = nn.Embed(4, self.width, name="task_embedding")(task_ids)
        color = nn.Embed(4, self.width, name="color_embedding")(query_color_ids)
        ordinal = nn.Embed(5, self.width, name="ordinal_embedding")(
            queried_ordinals
        )
        query_seed = self.param(
            "query_seed",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.width),
        )
        query = jnp.tile(query_seed, (batch, 1, 1))
        query = query + task[:, None] + color[:, None] + ordinal[:, None]
        cross_mask = nn.make_attention_mask(
            jnp.ones((batch, 1), dtype=jnp.bool_), visual_mask
        )
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.width,
            out_features=self.width,
            name="goal_visual_attention",
        )(query, x, mask=cross_mask)
        query = nn.LayerNorm(name="goal_visual_ln")(query + attended)[:, 0]
        hidden = nn.gelu(nn.Dense(self.width * 2, name="readout_hidden")(query))
        logits = nn.Dense(self.max_regions, name="region_logits")(hidden)
        valid = jnp.arange(self.max_regions)[None] < num_regions[:, None]
        return jnp.where(valid, logits, jnp.finfo(jnp.float32).min)

