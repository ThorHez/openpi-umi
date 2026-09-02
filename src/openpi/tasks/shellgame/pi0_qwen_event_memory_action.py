"""ShellGame Pi0 action policy conditioned on frozen recurrent-memory tokens.

Qwen and the recurrent updater run outside the high-frequency action model.
They publish a fixed ``Observation.semantic_memory`` bank, while this policy
uses the current camera images and robot state for closed-loop EEF control.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.tasks.robomme.pickxtimes import pi0_memory_action as _external_memory
from openpi.shared import array_typing as at


def shellgame_temporal_valid(frame, *, action_horizon: int, last_episode_frame: int):
    """Return the legacy ShellGame next-action validity mask."""
    frame = jnp.asarray(frame, dtype=jnp.int32)
    future_offsets = 1 + jnp.arange(action_horizon, dtype=jnp.int32)
    return frame[..., None] + future_offsets <= last_episode_frame


@dataclasses.dataclass(frozen=True)
class Pi0QwenEventMemoryActionConfig(_external_memory.Pi0PickXtimesMemoryActionConfig):
    """Absolute EEF7 action config with the ShellGame terminal-frame contract."""

    last_episode_frame: int = 154

    def create(self, rng: at.KeyArrayLike) -> Pi0QwenEventMemoryAction:
        return Pi0QwenEventMemoryAction(self, rngs=nnx.Rngs(rng))


class Pi0QwenEventMemoryAction(_external_memory.Pi0PickXtimesMemoryAction):
    """Direct-MEM action policy; Qwen remains frozen in a separate process."""

    def __init__(self, config: Pi0QwenEventMemoryActionConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.last_episode_frame = int(config.last_episode_frame)

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
        if observation.frame_index is None:
            return per_timestep
        valid = shellgame_temporal_valid(
            observation.frame_index,
            action_horizon=self.action_horizon,
            last_episode_frame=self.last_episode_frame,
        )
        scale = self.action_horizon / jnp.maximum(jnp.sum(valid, axis=-1, keepdims=True), 1)
        return per_timestep * valid.astype(per_timestep.dtype) * scale
