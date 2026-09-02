"""Pi0.5 EEF action expert conditioned on frozen PickXtimes MEM tokens."""

from __future__ import annotations

import dataclasses

import einops
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils


class SemanticMemoryActionConditioner(nn.Module):
    """Resample frozen 64-D MEM tokens and cross-attend from action tokens."""

    memory_tokens: int = 128
    memory_width: int = 64
    query_tokens: int = 8
    hidden_width: int = 256
    action_width: int = 1024
    num_heads: int = 4
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

        # A zero tensor is the explicit action-only ablation.  Substitute one
        # shared learned null bank before LayerNorm: this carries no episode or
        # temporal information, while avoiding the ill-conditioned derivative
        # of LayerNorm at an exactly zero-variance input.
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
        memory = nn.LayerNorm(name="memory_ln", dtype=jnp.float32)(semantic_memory)
        memory = nn.Dense(self.hidden_width, name="memory_in", dtype=self.dtype_mm)(memory)
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
        # Drop the entire MEM residual for a subset of training examples.  The
        # action expert must therefore retain a strong visual/state-only path
        # instead of treating a noisy event bank as an unconditional shortcut.
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
        return action_tokens + gate * update


@dataclasses.dataclass(frozen=True)
class Pi0PickXtimesMemoryActionConfig(_base.Pi0MemCompressConfig):
    semantic_memory_tokens: int = 128
    semantic_memory_width: int = 64
    semantic_query_tokens: int = 8
    semantic_hidden_width: int = 256
    gripper_loss_weight: float = 4.0
    real_action_dim: int = 7
    gripper_action_index: int = 6
    use_learned_null_memory: bool = False
    semantic_residual_gate_init: float = 1.0
    semantic_residual_dropout_rate: float = 0.0

    def create(self, rng: at.KeyArrayLike) -> Pi0PickXtimesMemoryAction:
        return Pi0PickXtimesMemoryAction(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_action_finetune(self) -> nnx.filterlib.Filter:
        memory_adapter = nnx_utils.PathRegex(r".*SemanticMemoryActionConditioner.*")
        action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm/.*_1.*")
        action_modules = nnx_utils.PathRegex(
            r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"
        )
        return nnx.Not(nnx.Any(memory_adapter, action_expert, action_modules))


class Pi0PickXtimesMemoryAction(_base.Pi0MemCompress):
    """Current-image Pi0.5 policy with a direct frozen-MEM action interface."""

    def __init__(self, config: Pi0PickXtimesMemoryActionConfig, rngs: nnx.Rngs):
        if config.num_frames != 1:
            raise ValueError("PickXtimes action training requires one current frame")
        if not 0 <= config.gripper_action_index < config.real_action_dim <= config.action_dim:
            raise ValueError("Invalid real action/gripper dimensions")
        super().__init__(config, rngs)
        self.semantic_memory_tokens = int(config.semantic_memory_tokens)
        self.semantic_memory_width = int(config.semantic_memory_width)
        self.gripper_loss_weight = float(config.gripper_loss_weight)
        self.real_action_dim = int(config.real_action_dim)
        self.gripper_action_index = int(config.gripper_action_index)
        self.SemanticMemoryActionConditioner = nnx_bridge.ToNNX(
            SemanticMemoryActionConditioner(
                memory_tokens=config.semantic_memory_tokens,
                memory_width=config.semantic_memory_width,
                query_tokens=config.semantic_query_tokens,
                hidden_width=config.semantic_hidden_width,
                dtype_mm=config.dtype,
                use_learned_null_memory=config.use_learned_null_memory,
                residual_gate_init=config.semantic_residual_gate_init,
                residual_dropout_rate=config.semantic_residual_dropout_rate,
            )
        )
        self.SemanticMemoryActionConditioner.lazy_init(
            jnp.zeros((1, config.action_horizon, 1024), dtype=jnp.bfloat16),
            jnp.zeros(
                (1, config.semantic_memory_tokens, config.semantic_memory_width),
                dtype=jnp.float32,
            ),
            rngs=rngs,
        )

    def _condition_action_tokens(
        self,
        observation,
        suffix_tokens,
        *,
        train: bool = False,
        dropout_rng=None,
    ):
        memory = observation.semantic_memory
        if memory is None:
            raise ValueError("PickXtimes Pi action model requires observation.semantic_memory")
        return self.SemanticMemoryActionConditioner(
            suffix_tokens,
            memory,
            train=train,
            dropout_rng=dropout_rng,
        )

    def _weighted_temporal_loss(self, observation, squared_error):
        if observation.action_loss_mask is not None:
            dim_mask = observation.action_loss_mask[..., None, :]
        else:
            dim_mask = jnp.asarray(self.action_loss_mask)[None, None, :]
        weights = jnp.ones((self.action_dim,), dtype=jnp.float32)
        weights = weights.at[self.gripper_action_index].set(self.gripper_loss_weight)
        weights = weights.at[self.real_action_dim :].set(0.0)
        dim_mask = dim_mask * weights[None, None, :]
        per_timestep = jnp.sum(squared_error * dim_mask, axis=-1) / jnp.maximum(
            jnp.sum(dim_mask, axis=-1), 1e-8
        )
        if observation.frame_index is None or observation.episode_T is None:
            return per_timestep
        frame = jnp.asarray(observation.frame_index, dtype=jnp.int32)
        episode_t = jnp.asarray(observation.episode_T, dtype=jnp.int32)
        valid = frame[..., None] + jnp.arange(self.action_horizon) < episode_t[..., None]
        scale = self.action_horizon / jnp.maximum(jnp.sum(valid, axis=-1, keepdims=True), 1)
        return per_timestep * valid.astype(per_timestep.dtype) * scale

    def compute_loss_with_memory_aux(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng, memory_dropout_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        target_velocity = noise - actions
        prefix, prefix_mask, prefix_ar, history_mem, encoder_auxes = (
            self._embed_prefix_with_history_mem(observation)
        )
        suffix, suffix_mask, suffix_ar, adarms = self.embed_suffix(observation, x_t, time)
        suffix = self._condition_action_tokens(
            observation,
            suffix,
            train=train,
            dropout_rng=memory_dropout_rng,
        )
        input_mask = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
        ar_mask = jnp.concatenate((prefix_ar, suffix_ar), axis=0)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix, suffix],
            mask=_base.make_attn_mask(input_mask, ar_mask),
            positions=positions,
            adarms_cond=[None, adarms],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        loss = self._weighted_temporal_loss(observation, jnp.square(velocity - target_velocity))
        return loss, {
            "history_mem": history_mem,
            "encoder_auxes": encoder_auxes,
            "history_class_logits": None,
        }

    def compute_loss(self, rng, observation, actions, *, train=False):
        loss, _ = self.compute_loss_with_memory_aux(rng, observation, actions, train=train)
        return loss

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(
                rng, (batch_size, self.action_horizon, self.action_dim)
            )
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix, None], mask=_base.make_attn_mask(prefix_mask, prefix_ar), positions=positions
        )

        def step(carry):
            x_t, time = carry
            suffix, suffix_mask, suffix_ar, adarms = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix = self._condition_action_tokens(observation, suffix)
            suffix_mask_2d = _base.make_attn_mask(suffix_mask, suffix_ar)
            prefix_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix.shape[1])
            full_mask = jnp.concatenate((prefix_for_suffix, suffix_mask_2d), axis=-1)
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix],
                mask=full_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms],
            )
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * velocity, time + dt

        def cond(carry):
            return carry[1] >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return actions
