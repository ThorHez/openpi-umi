"""Inference-only bounded XY release around a frozen semantic waypoint.

Unlike the trainable residual-action model, this wrapper keeps the checkpoint's
absolute-action semantics.  It first samples the ordinary absolute Pi action,
then clips only its difference from the decoded waypoint.  No parameters or
training targets change, so an established hard-waypoint checkpoint loads
exactly and the only controlled variable is post-diffusion XY composition.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.tasks.shellgame import pi0_qwen_event_memory_action as _raw_action
from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_action as _waypoint


def apply_bounded_absolute_xy(
    actions,
    waypoint,
    action_dims: tuple[int, ...],
    normalized_limits: tuple[float, ...],
):
    """Keep absolute Pi XY commands inside a trust region around the waypoint."""
    if len(action_dims) != len(normalized_limits):
        raise ValueError("Each waypoint action dimension requires one residual limit")
    result = actions
    for waypoint_dim, (action_dim, limit) in enumerate(
        zip(action_dims, normalized_limits, strict=True)
    ):
        goal = waypoint[:, waypoint_dim, None]
        correction = jnp.clip(actions[:, :, action_dim] - goal, -limit, limit)
        result = result.at[:, :, action_dim].set(goal + correction)
    return result


@dataclasses.dataclass(frozen=True)
class Pi0QwenEventMemoryWaypointBoundedActionConfig(
    _waypoint.Pi0QwenEventMemoryWaypointActionConfig
):
    # In the nominal min/max normalization, these correspond to about 10 mm.
    waypoint_bounded_normalized_limits: tuple[float, ...] = (0.395 / 3.0, 0.246 / 3.0)

    def create(self, rng: at.KeyArrayLike) -> Pi0QwenEventMemoryWaypointBoundedAction:
        return Pi0QwenEventMemoryWaypointBoundedAction(self, rngs=nnx.Rngs(rng))


class Pi0QwenEventMemoryWaypointBoundedAction(
    _waypoint.Pi0QwenEventMemoryWaypointAction
):
    """Release bounded live-image XY corrections without adding parameters."""

    def __init__(
        self,
        config: Pi0QwenEventMemoryWaypointBoundedActionConfig,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, rngs)
        limits = tuple(float(value) for value in config.waypoint_bounded_normalized_limits)
        if len(limits) != len(self.waypoint_action_dims) or any(value <= 0 for value in limits):
            raise ValueError("Bounded limits must be positive and match waypoint_action_dims")
        self.waypoint_bounded_normalized_limits = limits

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        # Bypass the parent's hard overwrite while retaining the identical
        # waypoint-conditioned diffusion expert.
        actions = _raw_action.Pi0QwenEventMemoryAction.sample_actions(
            self,
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
        )
        processed = _model.preprocess_observation(None, observation, train=False)
        dummy_tokens = jnp.zeros(
            (processed.state.shape[0], self.action_horizon, 1024),
            dtype=jnp.bfloat16,
        )
        _, waypoint = self._condition_action_tokens_with_waypoint(processed, dummy_tokens)
        return apply_bounded_absolute_xy(
            actions,
            waypoint,
            self.waypoint_action_dims,
            self.waypoint_bounded_normalized_limits,
        )
