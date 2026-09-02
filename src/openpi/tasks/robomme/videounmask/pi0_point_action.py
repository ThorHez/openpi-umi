"""Pi0.5 EEF action expert conditioned on a VideoUnmask target point.

The first action experiment deliberately does not run the learned visual
memory inside the policy.  Instead, the last two normalized state dimensions
carry an oracle image-space target point.  A small trainable conditioner
injects that point directly into every action token.  This isolates action
learning from memory-selection errors while retaining current front/wrist
visual feedback and Pi0.5's 16-step flow-matching action chunks.
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
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils


class TargetPointConditioner(nn.Module):
    """Add a learned target-point residual to Pi action tokens."""

    hidden_width: int = 256
    output_width: int = 1024
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, action_tokens, target_point):
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.output_width:
            raise ValueError(
                f"Expected action tokens [B,H,{self.output_width}], got {action_tokens.shape}"
            )
        if target_point.ndim != 2 or target_point.shape[-1] != 2:
            raise ValueError(f"Expected target point [B,2], got {target_point.shape}")
        point = nn.Dense(self.hidden_width, name="point_in", dtype=self.dtype_mm)(
            target_point.astype(jnp.float32)
        )
        point = nn.gelu(point)
        point = nn.Dense(self.output_width, name="point_out", dtype=self.dtype_mm)(point)
        point = nn.LayerNorm(name="point_out_ln", dtype=self.dtype_mm)(point)
        gate_delta = self.param("gate_delta", nn.initializers.zeros_init(), (1,), jnp.float32)
        gate = (1.0 + jnp.tanh(gate_delta)).astype(action_tokens.dtype)
        return action_tokens + gate * point[:, None, :]


class PhaseGoalConditioner(nn.Module):
    """Inject continuous XY error, EEF height, and manipulation phase."""

    hidden_width: int = 256
    output_width: int = 1024
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, action_tokens, phase_goal):
        if phase_goal.ndim != 2 or phase_goal.shape[-1] != 7:
            raise ValueError(f"Expected phase-goal context [B,7], got {phase_goal.shape}")
        update = nn.Dense(self.hidden_width, name="context_in", dtype=self.dtype_mm)(
            phase_goal.astype(jnp.float32)
        )
        update = nn.gelu(update)
        update = nn.Dense(self.output_width, name="context_out", dtype=self.dtype_mm)(update)
        update = nn.LayerNorm(name="context_out_ln", dtype=self.dtype_mm)(update)
        gate_delta = self.param("gate_delta", nn.initializers.zeros_init(), (1,), jnp.float32)
        gate = (1.0 + jnp.tanh(gate_delta)).astype(action_tokens.dtype)
        return action_tokens + gate * update[:, None, :]


@dataclasses.dataclass(frozen=True)
class Pi0VideoUnmaskPointActionConfig(_base.Pi0MemCompressConfig):
    """Current-image Pi0.5 policy with explicit target-point conditioning."""

    target_point_state_start: int = 7
    target_point_hidden_width: int = 256
    target_point_relative_to_eef: bool = False
    phase_goal_conditioner: bool = False
    gripper_loss_weight: float = 4.0
    real_action_dim: int = 7
    gripper_action_index: int = 6

    def create(self, rng: at.KeyArrayLike) -> Pi0VideoUnmaskPointAction:
        return Pi0VideoUnmaskPointAction(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_action_finetune(self) -> nnx.filterlib.Filter:
        """Train the point adapter, Pi action expert, and action/time projections."""
        target_adapter = nnx_utils.PathRegex(
            r".*(TargetPointConditioner|PhaseGoalConditioner).*"
        )
        action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm/.*_1.*")
        action_modules = nnx_utils.PathRegex(
            r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"
        )
        return nnx.Not(nnx.Any(target_adapter, action_expert, action_modules))


class Pi0VideoUnmaskPointAction(_base.Pi0MemCompress):
    """Point-conditioned absolute EEF7 flow policy for VideoUnmask."""

    def __init__(self, config: Pi0VideoUnmaskPointActionConfig, rngs: nnx.Rngs):
        if config.num_frames != 1:
            raise ValueError("The action-only gate requires exactly one current image frame")
        if config.target_point_state_start < 0:
            raise ValueError("target_point_state_start must be non-negative")
        if not 0 <= config.gripper_action_index < config.real_action_dim <= config.action_dim:
            raise ValueError(
                "Expected 0 <= gripper_action_index < real_action_dim <= action_dim, got "
                f"{config.gripper_action_index}, {config.real_action_dim}, {config.action_dim}"
            )
        super().__init__(config, rngs)
        self.target_point_state_start = int(config.target_point_state_start)
        self.target_point_relative_to_eef = bool(config.target_point_relative_to_eef)
        self.phase_goal_conditioner = bool(config.phase_goal_conditioner)
        self.gripper_loss_weight = float(config.gripper_loss_weight)
        self.real_action_dim = int(config.real_action_dim)
        self.gripper_action_index = int(config.gripper_action_index)
        self.TargetPointConditioner = nnx_bridge.ToNNX(
            TargetPointConditioner(
                hidden_width=config.target_point_hidden_width,
                output_width=1024,
                dtype_mm=config.dtype,
            )
        )
        self.TargetPointConditioner.lazy_init(
            jnp.zeros((1, config.action_horizon, 1024), dtype=jnp.bfloat16),
            jnp.zeros((1, 2), dtype=jnp.float32),
            rngs=rngs,
        )
        if self.phase_goal_conditioner:
            self.PhaseGoalConditioner = nnx_bridge.ToNNX(
                PhaseGoalConditioner(
                    hidden_width=config.target_point_hidden_width,
                    output_width=1024,
                    dtype_mm=config.dtype,
                )
            )
            self.PhaseGoalConditioner.lazy_init(
                jnp.zeros((1, config.action_horizon, 1024), dtype=jnp.bfloat16),
                jnp.zeros((1, 7), dtype=jnp.float32),
                rngs=rngs,
            )

    def _target_point(self, observation: _model.Observation):
        start = self.target_point_state_start
        if observation.state.shape[-1] < start + 2:
            raise ValueError(
                f"State width {observation.state.shape[-1]} does not contain target point at {start}:{start + 2}"
            )
        target = observation.state[..., start : start + 2]
        if self.target_point_relative_to_eef:
            # Data transforms normalize each state coordinate before this
            # point.  With world-XY target conditioning, this exposes a direct
            # continuous closed-loop error to the action-token adapter.
            target = target - observation.state[..., :2]
        return target

    def _condition_action_tokens(self, observation, suffix_tokens):
        suffix_tokens = self.TargetPointConditioner(suffix_tokens, self._target_point(observation))
        if self.phase_goal_conditioner:
            state = observation.state
            start = self.target_point_state_start
            if state.shape[-1] < start + 6:
                raise ValueError("Phase-goal conditioning requires target XY and four phase values")
            xy_error = state[..., start : start + 2] - state[..., :2]
            context = jnp.concatenate(
                (xy_error, state[..., 2:3], state[..., start + 2 : start + 6]), axis=-1
            )
            suffix_tokens = self.PhaseGoalConditioner(suffix_tokens, context)
        return suffix_tokens

    def _weighted_temporal_loss(self, observation, squared_error):
        if observation.action_loss_mask is not None:
            dim_mask = observation.action_loss_mask[..., None, :]
        else:
            dim_mask = jnp.asarray(self.action_loss_mask)[None, None, :]
        dimension_weights = jnp.ones((self.action_dim,), dtype=jnp.float32)
        dimension_weights = dimension_weights.at[self.gripper_action_index].set(
            self.gripper_loss_weight
        )
        dimension_weights = dimension_weights.at[self.real_action_dim :].set(0.0)
        dim_mask = dim_mask * dimension_weights[None, None, :]
        loss_per_timestep = jnp.sum(squared_error * dim_mask, axis=-1) / jnp.maximum(
            jnp.sum(dim_mask, axis=-1), 1e-8
        )

        if observation.frame_index is None or observation.episode_T is None:
            return loss_per_timestep
        frame_index = jnp.asarray(observation.frame_index, dtype=jnp.int32)
        episode_t = jnp.asarray(observation.episode_T, dtype=jnp.int32)
        future_offsets = jnp.arange(self.action_horizon, dtype=jnp.int32)
        temporal_valid = frame_index[..., None] + future_offsets < episode_t[..., None]
        valid_count = jnp.sum(temporal_valid, axis=-1, keepdims=True)
        temporal_scale = self.action_horizon / jnp.maximum(valid_count, 1)
        return (
            loss_per_timestep
            * temporal_valid.astype(loss_per_timestep.dtype)
            * temporal_scale
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

        prefix_tokens, prefix_mask, prefix_ar_mask, history_mem, encoder_auxes = (
            self._embed_prefix_with_history_mem(observation)
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
        suffix_tokens = self._condition_action_tokens(observation, suffix_tokens)
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
        loss_per_timestep = self._weighted_temporal_loss(
            observation, jnp.square(velocity - u_t)
        )
        return loss_per_timestep, {
            "history_mem": history_mem,
            "encoder_auxes": encoder_auxes,
            "history_class_logits": None,
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
            noise = jax.random.normal(
                rng, (batch_size, self.action_horizon, self.action_dim)
            )

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
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
            suffix_tokens = self._condition_action_tokens(observation, suffix_tokens)
            suffix_attn_mask = _base.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate((prefix_for_suffix, suffix_attn_mask), axis=-1)
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
