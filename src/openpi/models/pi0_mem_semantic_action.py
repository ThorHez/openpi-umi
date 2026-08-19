"""Pi0.5 policy with validated semantic recurrent memory for ShellGame.

The model consolidates the production path that was previously assembled by
``examples/shellgame`` inheritance:

``60 history frames -> semantic recurrent memory [128,64]``
``-> 16 learned queries [16,1024] -> gated action cross-attention``.

The 61st image is treated as the dynamic current observation and is encoded
only into the PaliGemma visual prefix.  The original compressed-history branch
inside :class:`Pi0MemCompress` is disabled by configuration; the semantic
memory in this file is the sole history-to-action path.
"""

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
from openpi.models import siglip_mem_semantic as _semantic
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils


class SemanticJointActionReadout(nn.Module):
    """Checkpoint-compatible diagnostic action head from the source model."""

    state_dim: int = 8
    hidden_width: int = 256
    action_horizon: int = 16
    action_dim: int = 32

    @nn.compact
    def __call__(self, final_slot_probabilities, state):
        if final_slot_probabilities.shape[-1] != 3:
            raise ValueError(f"Expected three final-slot probabilities, got {final_slot_probabilities.shape}")
        state = state[..., : self.state_dim].astype(jnp.float32)
        features = jnp.concatenate((final_slot_probabilities.astype(jnp.float32), state), axis=-1)
        features = nn.Dense(self.hidden_width, name="input_projection")(features)
        features = nn.gelu(features)
        residual = features
        features = nn.Dense(self.hidden_width, name="hidden_0")(features)
        features = nn.gelu(features)
        features = nn.Dense(self.hidden_width, name="hidden_1")(features)
        features = nn.gelu(features + residual)
        flat = nn.Dense(self.action_horizon * self.action_dim, name="trajectory_output")(features)
        return flat.reshape((-1, self.action_horizon, self.action_dim))


class RawMemoryQueryResampler(nn.Module):
    """Resample compact recurrent memory into action-expert-width tokens."""

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
            queries = queries + nn.Dense(self.width, name=f"mlp_out_{layer}", dtype=self.dtype_mm)(hidden)

        queries = nn.LayerNorm(name="query_output_ln", dtype=self.dtype_mm)(queries)
        queries = nn.Dense(self.output_width, name="action_width_projection", dtype=self.dtype_mm)(queries)
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


@dataclasses.dataclass(frozen=True)
class Pi0MemSemanticActionConfig(_base.Pi0MemCompressConfig):
    """Configuration for the consolidated semantic-memory action policy."""

    history_frames: int = _semantic.HISTORY_FRAMES
    encoder_width: int = 256
    encoder_depth: int = 2
    encoder_heads: int = 8
    semantic_memory_width: int = 64
    semantic_memory_depth: int = 2
    semantic_memory_heads: int = 4
    semantic_memory_tokens: int = 128
    diagnostic_current_tokens: int = 256
    diagnostic_adapter_heads: int = 4
    diagnostic_residual_scale: float = 1.0
    video_mode: str = "normal"
    initial_mode: str = "normal"
    relation_mode: str = "one_hot"
    raw_memory_mode: str = "normal"
    query_tokens: int = 16
    query_width: int = 256
    query_depth: int = 2
    query_heads: int = 4
    action_cross_attention_heads: int = 8
    gripper_loss_weight: float = 4.0
    real_action_dim: int = 7
    gripper_action_index: int = 6
    last_episode_frame: int = 154

    def create(self, rng: at.KeyArrayLike) -> Pi0MemSemanticAction:
        return Pi0MemSemanticAction(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_action_finetune(self) -> nnx.filterlib.Filter:
        """Freeze memory and base perception; train only Pi0.5 action layers."""
        action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm/.*_1.*")
        action_modules = nnx_utils.PathRegex(
            r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"
        )
        return nnx.Not(nnx.Any(action_expert, action_modules))

    def get_freeze_filter_memory_interface_finetune(self) -> nnx.filterlib.Filter:
        """Train the action-memory interface together with the action expert."""
        memory_interface = nnx_utils.PathRegex(
            r".*(HistoryRawMemoryQueryResampler|ActionMemoryCrossAttention).*"
        )
        action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm/.*_1.*")
        action_modules = nnx_utils.PathRegex(
            r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"
        )
        return nnx.Not(nnx.Any(memory_interface, action_expert, action_modules))


class Pi0MemSemanticAction(_base.Pi0MemCompress):
    """Pi0.5 action policy conditioned on compact semantic recurrent memory."""

    def __init__(self, config: Pi0MemSemanticActionConfig, rngs: nnx.Rngs):
        if config.num_frames != config.history_frames + 1:
            raise ValueError(
                "Pi0MemSemanticAction requires fixed history plus one current frame: "
                f"num_frames={config.num_frames}, history_frames={config.history_frames}"
            )
        if config.semantic_memory_width < 3:
            raise ValueError("semantic_memory_width must leave at least three relation-code channels")
        if not 0 <= config.gripper_action_index < config.real_action_dim <= config.action_dim:
            raise ValueError(
                "Expected 0 <= gripper_action_index < real_action_dim <= action_dim, got "
                f"{config.gripper_action_index}, {config.real_action_dim}, {config.action_dim}"
            )
        super().__init__(config, rngs)
        self.history_frames = int(config.history_frames)
        self.video_mode = config.video_mode
        self.initial_mode = config.initial_mode
        self.raw_memory_mode = config.raw_memory_mode
        self.gripper_loss_weight = float(config.gripper_loss_weight)
        self.real_action_dim = int(config.real_action_dim)
        self.gripper_action_index = int(config.gripper_action_index)
        self.last_episode_frame = int(config.last_episode_frame)

        self.HistoryFrame0InitialCupClassifier = nnx_bridge.ToNNX(
            _semantic.FrozenFrame0InitialCupClassifier(input_width=1152)
        )
        self.HistoryFrame0InitialCupClassifier.lazy_init(
            jnp.zeros((1, 256, 1152), dtype=jnp.bfloat16), rngs=rngs
        )
        self.HistoryThreeSwapVisualRelationMemoryTracker = nnx_bridge.ToNNX(
            _semantic.ThreeSwapVisualRelationMemoryTracker(
                num_frames=config.history_frames,
                input_width=1152,
                encoder_width=config.encoder_width,
                encoder_depth=config.encoder_depth,
                encoder_heads=config.encoder_heads,
                memory_width=config.semantic_memory_width,
                memory_depth=config.semantic_memory_depth,
                memory_heads=config.semantic_memory_heads,
                adapter_heads=config.diagnostic_adapter_heads,
                num_memory_tokens=config.semantic_memory_tokens,
                num_current_tokens=config.diagnostic_current_tokens,
                current_width=1152,
                residual_scale=config.diagnostic_residual_scale,
                relation_mode=config.relation_mode,
                dtype_mm=config.dtype,
            )
        )
        self.HistoryThreeSwapVisualRelationMemoryTracker.lazy_init(
            jnp.zeros((1, config.history_frames, 256, 1152), dtype=jnp.bfloat16),
            jnp.zeros((1,), dtype=jnp.int32),
            rngs=rngs,
        )
        self.HistorySemanticJointActionReadout = nnx_bridge.ToNNX(
            SemanticJointActionReadout(
                state_dim=8,
                hidden_width=256,
                action_horizon=config.action_horizon,
                action_dim=config.action_dim,
            )
        )
        self.HistorySemanticJointActionReadout.lazy_init(
            jnp.zeros((1, 3), dtype=jnp.float32),
            jnp.zeros((1, config.action_dim), dtype=jnp.float32),
            rngs=rngs,
        )
        self.HistoryRawMemoryQueryResampler = nnx_bridge.ToNNX(
            RawMemoryQueryResampler(
                input_width=config.semantic_memory_width,
                width=config.query_width,
                output_width=1024,
                input_tokens=config.semantic_memory_tokens,
                query_tokens=config.query_tokens,
                depth=config.query_depth,
                num_heads=config.query_heads,
                dtype_mm=config.dtype,
            )
        )
        self.HistoryRawMemoryQueryResampler.lazy_init(
            jnp.zeros(
                (1, config.semantic_memory_tokens, config.semantic_memory_width),
                dtype=jnp.bfloat16,
            ),
            rngs=rngs,
        )
        self.ActionMemoryCrossAttention = nnx_bridge.ToNNX(
            ActionMemoryCrossAttention(
                width=1024,
                num_heads=config.action_cross_attention_heads,
                dtype_mm=config.dtype,
            )
        )
        self.ActionMemoryCrossAttention.lazy_init(
            jnp.zeros((1, self.action_horizon, 1024), dtype=jnp.bfloat16),
            jnp.zeros((1, config.query_tokens, 1024), dtype=jnp.bfloat16),
            rngs=rngs,
        )

    def _track_history(self, observation: _model.Observation):
        image = observation.images.get("base_rgb")
        if image is None:
            raise ValueError("Pi0MemSemanticAction requires a 'base_rgb' image stream")
        expected_frames = self.history_frames + 1
        if image.ndim != 5 or image.shape[1] != expected_frames:
            raise ValueError(f"Expected base_rgb [B,{expected_frames},H,W,C], got {image.shape}")
        history = image[:, : self.history_frames]

        _, initial_encoder_out = self.PaliGemma.img(history[:, :1], train=False)
        frame0_features = initial_encoder_out["encoded"]
        if self.initial_mode == "shuffle_batch":
            frame0_features = jnp.roll(frame0_features, 1, axis=0)
        elif self.initial_mode == "zero":
            frame0_features = jnp.zeros_like(frame0_features)
        elif self.initial_mode != "normal":
            raise ValueError(f"Unknown initial_mode={self.initial_mode!r}")
        initial_logits = self.HistoryFrame0InitialCupClassifier(frame0_features)
        initial_ids = jnp.argmax(initial_logits, axis=-1)

        _, history_encoder_out = self.PaliGemma.img(history, train=False)
        history_patches = history_encoder_out["with_posemb"][:, : self.history_frames]
        if self.video_mode == "shuffle_swaps":
            start, end = _semantic.SWAP_SLICES[0][0], _semantic.SWAP_SLICES[-1][1]
            history_patches = history_patches.at[:, start:end].set(
                jnp.roll(history_patches[:, start:end], 1, axis=0)
            )
        elif self.video_mode == "zero_swaps":
            start, end = _semantic.SWAP_SLICES[0][0], _semantic.SWAP_SLICES[-1][1]
            history_patches = history_patches.at[:, start:end].set(0)
        elif self.video_mode != "normal":
            raise ValueError(f"Unknown video_mode={self.video_mode!r}")

        joint_logits, stage_logits, stage_memories, relation_logits, relation_ids = (
            self.HistoryThreeSwapVisualRelationMemoryTracker(history_patches, initial_ids)
        )
        return {
            "joint_logits": joint_logits,
            "stage_logits": stage_logits,
            "stage_memories": stage_memories,
            "initial_logits": initial_logits,
            "initial_ids": initial_ids,
            "relation_logits": relation_logits,
            "relation_ids": relation_ids,
        }

    def _raw_and_resampled_memory(self, observation: _model.Observation):
        tracked = self._track_history(observation)
        raw_memory = jax.lax.stop_gradient(tracked["stage_memories"][:, -1])
        if self.raw_memory_mode == "shuffle_batch":
            raw_memory = jnp.roll(raw_memory, 1, axis=0)
        elif self.raw_memory_mode == "zero":
            raw_memory = jnp.zeros_like(raw_memory)
        elif self.raw_memory_mode != "normal":
            raise ValueError(f"Unknown raw_memory_mode={self.raw_memory_mode!r}")
        memory_tokens = self.HistoryRawMemoryQueryResampler(raw_memory)
        return raw_memory, memory_tokens, tracked

    def _embed_current_prefix(self, observation: _model.Observation):
        tokens = []
        input_mask = []
        ar_mask: list[bool] = []
        for name, video in observation.images.items():
            image = video[:, -1] if video.ndim == 5 else video
            image_tokens, _ = self.PaliGemma.img(image[:, None], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(observation.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]
        if observation.tokenized_prompt is not None:
            prompt_tokens = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            tokens.append(prompt_tokens)
            input_mask.append(observation.tokenized_prompt_mask)
            ar_mask += [False] * prompt_tokens.shape[1]
        return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1), jnp.asarray(ar_mask)

    def compute_history_classification(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ):
        observation = _model.preprocess_observation(rng, observation, train=train)
        tracked = self._track_history(observation)
        return tracked["joint_logits"], {
            "history_mem": tracked["stage_memories"][:, -1],
            "stage_logits": tracked["stage_logits"],
            "initial_logits": tracked["initial_logits"],
            "initial_ids": tracked["initial_ids"],
            "relation_logits": tracked["relation_logits"],
            "relation_ids": tracked["relation_ids"],
            "encoder_auxes": (),
        }

    def compute_loss_with_memory_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        del train
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=False)
        if observation.frame_index is None:
            raise ValueError("Semantic action temporal masking requires frame_index")

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        raw_memory, memory_tokens, tracked = self._raw_and_resampled_memory(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_current_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_tokens = self.ActionMemoryCrossAttention(suffix_tokens, memory_tokens)
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
            dim_mask = observation.action_loss_mask[..., None, :]
        else:
            dim_mask = jnp.asarray(self.action_loss_mask)[None, None, :]
        dimension_weights = jnp.ones((self.action_dim,), dtype=jnp.float32)
        dimension_weights = dimension_weights.at[self.gripper_action_index].set(self.gripper_loss_weight)
        dimension_weights = dimension_weights.at[self.real_action_dim :].set(0.0)
        dim_mask = dim_mask * dimension_weights[None, None, :]
        loss_per_timestep = jnp.sum(squared_error * dim_mask, axis=-1) / jnp.maximum(
            jnp.sum(dim_mask, axis=-1), 1e-8
        )

        frame_index = jnp.asarray(observation.frame_index, dtype=jnp.int32)
        future_offsets = 1 + jnp.arange(self.action_horizon, dtype=jnp.int32)
        temporal_valid = frame_index[..., None] + future_offsets <= self.last_episode_frame
        valid_count = jnp.sum(temporal_valid, axis=-1, keepdims=True)
        temporal_scale = self.action_horizon / jnp.maximum(valid_count, 1)
        loss_per_timestep = (
            loss_per_timestep * temporal_valid.astype(loss_per_timestep.dtype) * temporal_scale
        )
        return loss_per_timestep, {
            "history_mem": raw_memory,
            "encoder_auxes": (),
            "history_class_logits": tracked["joint_logits"],
            "temporal_valid_fraction": jnp.mean(temporal_valid),
        }

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
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        _, memory_tokens, _ = self._raw_and_resampled_memory(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_current_prefix(observation)
        prefix_attn_mask = _base.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_tokens = self.ActionMemoryCrossAttention(suffix_tokens, memory_tokens)
            suffix_attn_mask = _base.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate((prefix_for_suffix, suffix_attn_mask), axis=-1)
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
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
