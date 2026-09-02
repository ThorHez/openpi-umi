"""Inference-only release schedule for the continuous MEM waypoint anchor.

The trained parameter tree is identical to
``Pi0QwenEventMemoryWaypointAction``.  Only post-diffusion XY blending changes:
the global memory waypoint is enforced during approach, then progressively
released so the current-image action expert can perform local visual
corrections during descent and grasping.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.tasks.shellgame import pi0_qwen_event_memory_action as _raw_action
from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_action as _waypoint


def scheduled_anchor_strength(
    frame_index,
    *,
    start_frame: int,
    end_frame: int,
    initial_strength: float,
    final_strength: float,
):
    """Linearly interpolate anchor strength and clamp outside the interval."""
    if end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")
    frame = jnp.asarray(frame_index, dtype=jnp.float32)
    progress = jnp.clip((frame - start_frame) / (end_frame - start_frame), 0.0, 1.0)
    return initial_strength + progress * (final_strength - initial_strength)


@dataclasses.dataclass(frozen=True)
class Pi0QwenEventMemoryWaypointScheduledActionConfig(
    _waypoint.Pi0QwenEventMemoryWaypointActionConfig
):
    waypoint_anchor_start_frame: int = 91
    waypoint_anchor_end_frame: int = 107
    waypoint_anchor_initial_strength: float = 1.0
    waypoint_anchor_final_strength: float = 0.2

    def create(self, rng: at.KeyArrayLike) -> Pi0QwenEventMemoryWaypointScheduledAction:
        return Pi0QwenEventMemoryWaypointScheduledAction(self, rngs=nnx.Rngs(rng))


class Pi0QwenEventMemoryWaypointScheduledAction(
    _waypoint.Pi0QwenEventMemoryWaypointAction
):
    """Use a strong global anchor early and release it for local correction."""

    def __init__(
        self,
        config: Pi0QwenEventMemoryWaypointScheduledActionConfig,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, rngs)
        if not 0.0 <= config.waypoint_anchor_initial_strength <= 1.0:
            raise ValueError("waypoint_anchor_initial_strength must lie in [0, 1]")
        if not 0.0 <= config.waypoint_anchor_final_strength <= 1.0:
            raise ValueError("waypoint_anchor_final_strength must lie in [0, 1]")
        if config.waypoint_anchor_end_frame <= config.waypoint_anchor_start_frame:
            raise ValueError("waypoint anchor end frame must follow start frame")
        self.waypoint_anchor_start_frame = int(config.waypoint_anchor_start_frame)
        self.waypoint_anchor_end_frame = int(config.waypoint_anchor_end_frame)
        self.waypoint_anchor_initial_strength = float(
            config.waypoint_anchor_initial_strength
        )
        self.waypoint_anchor_final_strength = float(config.waypoint_anchor_final_strength)

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        # Bypass the parent's fixed post-diffusion anchor while retaining the
        # identical diffusion expert and waypoint-conditioned action tokens.
        actions = _raw_action.Pi0QwenEventMemoryAction.sample_actions(
            self,
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
        )
        if observation.frame_index is None:
            raise ValueError("Scheduled waypoint inference requires frame_index")

        processed = _model.preprocess_observation(None, observation, train=False)
        dummy_tokens = jnp.zeros(
            (processed.state.shape[0], self.action_horizon, 1024),
            dtype=jnp.bfloat16,
        )
        _, waypoint = self._condition_action_tokens_with_waypoint(processed, dummy_tokens)
        strength = scheduled_anchor_strength(
            observation.frame_index,
            start_frame=self.waypoint_anchor_start_frame,
            end_frame=self.waypoint_anchor_end_frame,
            initial_strength=self.waypoint_anchor_initial_strength,
            final_strength=self.waypoint_anchor_final_strength,
        ).astype(actions.dtype)
        strength = strength[..., None]
        for waypoint_dim, action_dim in enumerate(self.waypoint_action_dims):
            goal = waypoint[:, waypoint_dim, None]
            anchored = (1.0 - strength) * actions[:, :, action_dim] + strength * goal
            actions = actions.at[:, :, action_dim].set(anchored)
        return actions
