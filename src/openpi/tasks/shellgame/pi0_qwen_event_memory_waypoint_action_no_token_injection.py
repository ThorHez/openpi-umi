"""Ablate semantic-memory injection while retaining waypoint anchoring.

This variant keeps the complete trained ``SemanticMemoryActionConditioner``
parameter tree and still runs its waypoint decoder.  It deliberately discards
the conditioned action tokens, so the Pi diffusion expert receives its raw
suffix tokens and semantic memory can affect inference only through the final
hard XY waypoint anchor.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx

from openpi.shared import array_typing as at
from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_action as _base


@dataclasses.dataclass(frozen=True)
class Pi0QwenEventMemoryWaypointNoTokenInjectionConfig(_base.Pi0QwenEventMemoryWaypointActionConfig):
    def create(self, rng: at.KeyArrayLike) -> Pi0QwenEventMemoryWaypointNoTokenInjection:
        return Pi0QwenEventMemoryWaypointNoTokenInjection(self, rngs=nnx.Rngs(rng))


class Pi0QwenEventMemoryWaypointNoTokenInjection(_base.Pi0QwenEventMemoryWaypointAction):
    """Decode memory to XY, but do not inject it into diffusion action tokens."""

    def _condition_action_tokens_with_waypoint(
        self,
        observation,
        suffix_tokens,
        *,
        train: bool = False,
        dropout_rng=None,
    ):
        _, waypoint = super()._condition_action_tokens_with_waypoint(
            observation,
            suffix_tokens,
            train=train,
            dropout_rng=dropout_rng,
        )
        return suffix_tokens, waypoint


def from_waypoint_config(
    config: _base.Pi0QwenEventMemoryWaypointActionConfig,
) -> Pi0QwenEventMemoryWaypointNoTokenInjectionConfig:
    """Convert a normal waypoint config without changing its parameter tree."""
    values = {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    return Pi0QwenEventMemoryWaypointNoTokenInjectionConfig(**values)
