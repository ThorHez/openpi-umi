"""Fixed-grid temporal MEM with a dedicated action-memory interface.

This variant keeps the generic, topology-preserving history encoder from
``pi0_mem_fixed_grid_temporal`` and changes only how its compact memory reaches
the Pi0 action expert:

    per-stream history memory [B, M, 1152]
      -> mask-weighted stream aggregation
      -> learned-query resampler [B, Q, 1024]
      -> gated cross-attention into noisy action tokens
      -> native Pi0.5 flow action expert

History tokens are never appended to the visual/language prefix.  The module
contains no task labels, class logits, or ShellGame-specific assumptions.
"""

from __future__ import annotations

import dataclasses

import einops
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base
from openpi.models import pi0_mem_fixed_grid_temporal as _fixed_grid
from openpi.models import siglip_mem_fixed_grid_temporal as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils


class HistoryActionQueryResampler(nn.Module):
    """Compress generic visual memory into a few action-width tokens."""

    input_width: int = 1152
    width: int = 256
    output_width: int = 1024
    input_tokens: int = 128
    query_tokens: int = 16
    depth: int = 2
    num_heads: int = 4
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, memory):
        expected = (self.input_tokens, self.input_width)
        if memory.ndim != 3 or memory.shape[1:] != expected:
            raise ValueError(f"Expected memory [B,{expected}], got {memory.shape}")
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
            queries = queries + nn.MultiHeadDotProductAttention(
                name=f"query_cross_attention_{layer}",
                num_heads=self.num_heads,
                dropout_rate=0.0,
                deterministic=True,
                dtype=self.dtype_mm,
            )(query_norm, memory_norm)
            mlp_input = nn.LayerNorm(name=f"mlp_ln_{layer}", dtype=self.dtype_mm)(queries)
            hidden = nn.Dense(
                self.width * 4,
                name=f"mlp_in_{layer}",
                dtype=self.dtype_mm,
            )(mlp_input)
            hidden = nn.gelu(hidden)
            queries = queries + nn.Dense(
                self.width,
                name=f"mlp_out_{layer}",
                dtype=self.dtype_mm,
            )(hidden)

        queries = nn.LayerNorm(name="query_output_ln", dtype=self.dtype_mm)(queries)
        queries = nn.Dense(
            self.output_width,
            name="action_width_projection",
            dtype=self.dtype_mm,
        )(queries)
        return nn.LayerNorm(name="action_width_ln", dtype=self.dtype_mm)(queries)


class ActionMemoryCrossAttention(nn.Module):
    """Inject resampled history directly into the action suffix."""

    width: int = 1024
    num_heads: int = 8
    gate_init: float = 1.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, action_tokens, memory_tokens):
        if action_tokens.shape[-1] != self.width or memory_tokens.shape[-1] != self.width:
            raise ValueError(
                f"Expected width {self.width}, got {action_tokens.shape} and {memory_tokens.shape}"
            )
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
        gate = (self.gate_init + jnp.tanh(gate_delta)).astype(update.dtype)
        conditioned = action_tokens + gate * update

        mlp_input = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(conditioned)
        hidden = nn.Dense(self.width * 2, name="mlp_in", dtype=self.dtype_mm)(mlp_input)
        hidden = nn.gelu(hidden)
        mlp_update = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(hidden)
        return conditioned + gate * mlp_update


@dataclasses.dataclass(frozen=True)
class Pi0MemFixedGridQueryActionConfig(_fixed_grid.Pi0MemFixedGridTemporalConfig):
    action_memory_query_tokens: int = 16
    action_memory_query_width: int = 256
    action_memory_query_depth: int = 2
    action_memory_query_heads: int = 4
    action_memory_cross_attention_heads: int = 8
    action_memory_gate_init: float = 1.0

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0MemFixedGridQueryAction:
        return Pi0MemFixedGridQueryAction(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_action_memory_finetune(self) -> nnx.filterlib.Filter:
        """Train the history encoder, memory interface, and action expert."""
        history_encoder = nnx_utils.PathRegex(
            r".*PaliGemma/img/Transformer/FixedGridTemporalHistory_0.*"
        )
        memory_interface = nnx_utils.PathRegex(
            r".*(HistoryActionQueryResampler|ActionMemoryCrossAttention).*"
        )
        action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm/.*_1.*")
        action_modules = nnx_utils.PathRegex(
            r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|"
            r"state_proj|action_time_mlp_in|action_time_mlp_out).*"
        )
        return nnx.Not(
            nnx.Any(history_encoder, memory_interface, action_expert, action_modules)
        )

    def get_freeze_filter_memory_path_only(self) -> nnx.filterlib.Filter:
        """Train only the fixed-grid encoder and direct action-memory interface."""
        history_encoder = nnx_utils.PathRegex(
            r".*PaliGemma/img/Transformer/FixedGridTemporalHistory_0.*"
        )
        memory_interface = nnx_utils.PathRegex(
            r".*(HistoryActionQueryResampler|ActionMemoryCrossAttention).*"
        )
        return nnx.Not(nnx.Any(history_encoder, memory_interface))


class Pi0MemFixedGridQueryAction(_fixed_grid.Pi0MemFixedGridTemporal):
    """Generic fixed-grid MEM whose action tokens read memory directly."""

    def __init__(self, config: Pi0MemFixedGridQueryActionConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        vision_width = _siglip.decode_variant("So400m/14")["width"]
        action_width = _gemma.get_config(config.action_expert_variant).width
        self.HistoryActionQueryResampler = nnx_bridge.ToNNX(
            HistoryActionQueryResampler(
                input_width=vision_width,
                width=config.action_memory_query_width,
                output_width=action_width,
                input_tokens=config.history_memory_tokens,
                query_tokens=config.action_memory_query_tokens,
                depth=config.action_memory_query_depth,
                num_heads=config.action_memory_query_heads,
                dtype_mm=config.dtype,
            )
        )
        self.HistoryActionQueryResampler.lazy_init(
            jnp.zeros(
                (1, config.history_memory_tokens, vision_width),
                dtype=jnp.bfloat16,
            ),
            rngs=rngs,
        )
        self.ActionMemoryCrossAttention = nnx_bridge.ToNNX(
            ActionMemoryCrossAttention(
                width=action_width,
                num_heads=config.action_memory_cross_attention_heads,
                gate_init=config.action_memory_gate_init,
                dtype_mm=config.dtype,
            )
        )
        self.ActionMemoryCrossAttention.lazy_init(
            jnp.zeros((1, self.action_horizon, action_width), dtype=jnp.bfloat16),
            jnp.zeros(
                (1, config.action_memory_query_tokens, action_width),
                dtype=jnp.bfloat16,
            ),
            rngs=rngs,
        )

    @staticmethod
    def _aggregate_stream_memories(observation: _model.Observation, encoder_auxes):
        """Mask-weighted mean over camera streams, preserving memory slots."""
        if not encoder_auxes:
            raise ValueError("Action-memory conditioning requires at least one image stream")
        memories = jnp.stack(
            [encoder_aux["history_mem"] for encoder_aux in encoder_auxes],
            axis=1,
        )
        masks = jnp.stack(
            [jnp.asarray(observation.image_masks[name], dtype=jnp.float32) for name in observation.images],
            axis=1,
        )
        denom = jnp.maximum(jnp.sum(masks, axis=1, keepdims=True), 1.0)
        return jnp.sum(memories * masks[..., None, None], axis=1) / denom[..., None]

    def _prefix_and_action_memory(self, observation: _model.Observation):
        prefix_tokens, prefix_mask, prefix_ar_mask, history_mem, encoder_auxes = (
            self._embed_prefix_with_history_mem(observation)
        )
        aggregate = self._aggregate_stream_memories(observation, encoder_auxes)
        action_memory = self.HistoryActionQueryResampler(aggregate)
        return (
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            action_memory,
            history_mem,
            encoder_auxes,
        )

    def compute_loss_with_memory_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        (
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            action_memory,
            history_mem,
            encoder_auxes,
        ) = self._prefix_and_action_memory(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
        suffix_tokens = self.ActionMemoryCrossAttention(suffix_tokens, action_memory)
        input_mask = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
        ar_mask = jnp.concatenate((prefix_ar_mask, suffix_ar_mask), axis=0)
        attn_mask = _base.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        squared_error = jnp.square(velocity - u_t)
        if observation.action_loss_mask is not None:
            mask = observation.action_loss_mask[..., None, :]
        elif self.action_loss_mask is not None:
            mask = jnp.asarray(self.action_loss_mask)[None, None, :]
        else:
            mask = jnp.ones((1, 1, actions.shape[-1]), dtype=squared_error.dtype)
        loss_per_timestep = jnp.sum(squared_error * mask, axis=-1) / jnp.maximum(
            jnp.sum(mask, axis=-1), 1e-8
        )
        return loss_per_timestep, {
            "history_mem": history_mem,
            "encoder_auxes": encoder_auxes,
            "history_class_logits": self._history_class_logits(observation, encoder_auxes),
        }

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        loss, _ = self.compute_loss_with_memory_aux(rng, observation, actions, train=train)
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(
                rng,
                (batch_size, self.action_horizon, self.action_dim),
            )

        prefix_tokens, prefix_mask, prefix_ar_mask, action_memory, _, _ = (
            self._prefix_and_action_memory(observation)
        )
        prefix_attn_mask = _base.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
            )
            suffix_tokens = self.ActionMemoryCrossAttention(suffix_tokens, action_memory)
            suffix_attn_mask = _base.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_for_suffix = einops.repeat(
                prefix_mask,
                "b p -> b s p",
                s=suffix_tokens.shape[1],
            )
            full_attn_mask = jnp.concatenate(
                (prefix_for_suffix, suffix_attn_mask),
                axis=-1,
            )
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * velocity, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return actions


@dataclasses.dataclass(frozen=True)
class QueryActionCheckpointLoader:
    """Warm-start from the successful query-action checkpoint.

    Exact shared parameters, including the action expert and action-memory
    cross-attention, are restored.  Fixed-grid history parameters and the
    input-width-dependent part of the query resampler remain initialized.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        source = _model.restore_params(self.params_path, restore_type=np.ndarray)
        source_flat = flax.traverse_util.flatten_dict(source, sep="/")
        target_flat = flax.traverse_util.flatten_dict(params, sep="/")
        probe_query_prefix = "HistoryRawMemoryQueryResampler/"
        query_prefix = "HistoryActionQueryResampler/"
        restored = {}
        counts = {"exact": 0, "mapped_query": 0, "initialized": 0}
        unexpected = []

        allowed_new = (
            "HistoryActionQueryResampler/",
            "PaliGemma/img/Transformer/FixedGridTemporalHistory_0/",
        )
        allowed_new_fragments = (
            "/HistoryLayerNorm_0/",
            "/HistoryMultiHeadDotProductAttention_0/",
            "/history_memory_gate_logit",
        )
        for key, reference in target_flat.items():
            candidate = source_flat.get(key)
            source_kind = "exact"
            if candidate is None and key.startswith(query_prefix):
                candidate = source_flat.get(
                    probe_query_prefix + key.removeprefix(query_prefix)
                )
                source_kind = "mapped_query"
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                restored[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                counts[source_kind] += 1
                continue

            restored[key] = reference
            counts["initialized"] += 1
            if not key.startswith(allowed_new) and not any(
                fragment in key for fragment in allowed_new_fragments
            ):
                unexpected.append(key)

        if unexpected:
            raise ValueError(f"Unexpected missing query-action parameters: {unexpected[:8]}")
        print(
            "QueryActionCheckpointLoader: "
            f"exact={counts['exact']}, mapped_query={counts['mapped_query']}, "
            f"initialized={counts['initialized']}, unexpected=0"
        )
        return flax.traverse_util.unflatten_dict(restored, sep="/")
