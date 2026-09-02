"""Absolute-EEF policy that predicts local XY residuals around a MEM waypoint.

The frozen semantic-memory bridge supplies a global normalized XY waypoint.
During training the diffusion target is rewritten from absolute XY to
``absolute_xy - waypoint_xy``.  During inference the bounded residual is added
back to the waypoint.  Z, rotation and gripper dimensions retain their normal
absolute action semantics.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _pi0
from openpi.shared import array_typing as at
from openpi.tasks.shellgame import pi0_qwen_event_memory_action as _raw_action
from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_action as _waypoint


def absolute_to_waypoint_residual(actions, waypoint, action_dims: tuple[int, ...]):
    result = actions
    for waypoint_dim, action_dim in enumerate(action_dims):
        result = result.at[:, :, action_dim].set(
            actions[:, :, action_dim] - waypoint[:, waypoint_dim, None]
        )
    return result


def waypoint_residual_to_absolute(
    residual_actions,
    waypoint,
    action_dims: tuple[int, ...],
    normalized_limits: tuple[float, ...],
):
    if len(action_dims) != len(normalized_limits):
        raise ValueError("Each waypoint action dimension requires one residual limit")
    result = residual_actions
    for waypoint_dim, (action_dim, limit) in enumerate(
        zip(action_dims, normalized_limits, strict=True)
    ):
        residual = jnp.clip(residual_actions[:, :, action_dim], -limit, limit)
        result = result.at[:, :, action_dim].set(
            waypoint[:, waypoint_dim, None] + residual
        )
    return result


@dataclasses.dataclass(frozen=True)
class Pi0QwenEventMemoryWaypointResidualActionConfig(
    _waypoint.Pi0QwenEventMemoryWaypointActionConfig
):
    # Nominal action min/max normalization maps 30 mm to approximately 0.395
    # in X and 0.246 in Y.  Keeping limits in normalized coordinates makes the
    # model/train transform exactly invertible before physical unnormalization.
    waypoint_residual_normalized_limits: tuple[float, ...] = (0.395, 0.246)

    def create(self, rng: at.KeyArrayLike) -> Pi0QwenEventMemoryWaypointResidualAction:
        return Pi0QwenEventMemoryWaypointResidualAction(self, rngs=nnx.Rngs(rng))


class Pi0QwenEventMemoryWaypointResidualAction(
    _waypoint.Pi0QwenEventMemoryWaypointAction
):
    """Diffuse bounded local XY corrections while preserving global selection."""

    def __init__(
        self,
        config: Pi0QwenEventMemoryWaypointResidualActionConfig,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, rngs)
        limits = tuple(float(value) for value in config.waypoint_residual_normalized_limits)
        if len(limits) != len(self.waypoint_action_dims) or any(value <= 0 for value in limits):
            raise ValueError("Residual limits must be positive and match waypoint_action_dims")
        self.waypoint_residual_normalized_limits = limits

    def _decode_waypoint(self, observation):
        dummy_tokens = jnp.zeros(
            (observation.state.shape[0], self.action_horizon, 1024),
            dtype=jnp.bfloat16,
        )
        _, waypoint = self._condition_action_tokens_with_waypoint(observation, dummy_tokens)
        return waypoint

    def compute_loss_with_memory_aux(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng, memory_dropout_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        waypoint = self._decode_waypoint(observation)
        residual_actions = absolute_to_waypoint_residual(
            actions, waypoint, self.waypoint_action_dims
        )

        batch_shape = residual_actions.shape[:-2]
        noise = jax.random.normal(noise_rng, residual_actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * residual_actions
        target_velocity = noise - residual_actions

        prefix, prefix_mask, prefix_ar, history_mem, encoder_auxes = (
            self._embed_prefix_with_history_mem(observation)
        )
        suffix, suffix_mask, suffix_ar, adarms = self.embed_suffix(observation, x_t, time)
        suffix, conditioned_waypoint = self._condition_action_tokens_with_waypoint(
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
            mask=_pi0.make_attn_mask(input_mask, ar_mask),
            positions=positions,
            adarms_cond=[None, adarms],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow_loss = self._weighted_temporal_loss(
            observation, jnp.square(velocity - target_velocity)
        )

        target_waypoint = jnp.stack(
            [actions[:, self.waypoint_action_index, dim] for dim in self.waypoint_action_dims],
            axis=-1,
        )
        waypoint_loss = jnp.mean(jnp.square(conditioned_waypoint - target_waypoint), axis=-1)
        total_loss = flow_loss + self.waypoint_aux_weight * waypoint_loss[:, None]
        return total_loss, {
            "history_mem": history_mem,
            "encoder_auxes": encoder_auxes,
            "history_class_logits": None,
            "predicted_waypoint": conditioned_waypoint,
            "waypoint_loss": waypoint_loss,
        }

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        residual_actions = _raw_action.Pi0QwenEventMemoryAction.sample_actions(
            self,
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
        )
        processed = _model.preprocess_observation(None, observation, train=False)
        waypoint = self._decode_waypoint(processed)
        return waypoint_residual_to_absolute(
            residual_actions,
            waypoint,
            self.waypoint_action_dims,
            self.waypoint_residual_normalized_limits,
        )
