"""ShellGame action policy with a generic continuous-memory waypoint bridge."""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_mem_semantic_waypoint_action as _waypoint
from openpi.tasks.shellgame import pi0_qwen_event_memory_action as _base
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi0QwenEventMemoryWaypointActionConfig(_base.Pi0QwenEventMemoryActionConfig):
    """Configure continuous future-action supervision and inference anchoring."""

    waypoint_action_dims: tuple[int, ...] = (0, 1)
    waypoint_action_index: int = 0
    waypoint_aux_weight: float = 1.0
    waypoint_injection_scale: float = 1.0
    waypoint_anchor_strength: float = 1.0
    waypoint_use_memory_statistics: bool = False

    def create(self, rng: at.KeyArrayLike) -> Pi0QwenEventMemoryWaypointAction:
        return Pi0QwenEventMemoryWaypointAction(self, rngs=nnx.Rngs(rng))


class Pi0QwenEventMemoryWaypointAction(_base.Pi0QwenEventMemoryAction):
    """Decode a continuous memory goal and expose it strongly to EEF actions."""

    def __init__(self, config: Pi0QwenEventMemoryWaypointActionConfig, rngs: nnx.Rngs):
        if not config.waypoint_action_dims:
            raise ValueError("waypoint_action_dims cannot be empty")
        if any(dim < 0 or dim >= config.real_action_dim for dim in config.waypoint_action_dims):
            raise ValueError("waypoint_action_dims must index real action dimensions")
        if not 0 <= config.waypoint_action_index < config.action_horizon:
            raise ValueError("waypoint_action_index must fall inside the action horizon")
        super().__init__(config, rngs)
        self.waypoint_action_dims = tuple(int(dim) for dim in config.waypoint_action_dims)
        self.waypoint_action_index = int(config.waypoint_action_index)
        self.waypoint_aux_weight = float(config.waypoint_aux_weight)
        self.waypoint_anchor_strength = float(config.waypoint_anchor_strength)

        # Retain the attribute name so all established conditioner parameters
        # load from the previous checkpoint; only waypoint_* leaves are new.
        self.SemanticMemoryActionConditioner = nnx_bridge.ToNNX(
            _waypoint.SemanticMemoryWaypointConditioner(
                memory_tokens=config.semantic_memory_tokens,
                memory_width=config.semantic_memory_width,
                query_tokens=config.semantic_query_tokens,
                hidden_width=config.semantic_hidden_width,
                waypoint_dim=len(config.waypoint_action_dims),
                waypoint_injection_scale=config.waypoint_injection_scale,
                use_memory_statistics=config.waypoint_use_memory_statistics,
                dtype_mm=config.dtype,
                use_learned_null_memory=config.use_learned_null_memory,
                residual_gate_init=config.semantic_residual_gate_init,
                residual_dropout_rate=config.semantic_residual_dropout_rate,
            )
        )
        self.SemanticMemoryActionConditioner.lazy_init(
            jnp.zeros((1, config.action_horizon, 1024), dtype=jnp.bfloat16),
            jnp.zeros((1, config.semantic_memory_tokens, config.semantic_memory_width), dtype=jnp.float32),
            rngs=rngs,
        )

    def _condition_action_tokens_with_waypoint(
        self,
        observation,
        suffix_tokens,
        *,
        train: bool = False,
        dropout_rng=None,
    ):
        memory = observation.semantic_memory
        if memory is None:
            raise ValueError("Waypoint action model requires observation.semantic_memory")
        return self.SemanticMemoryActionConditioner(
            suffix_tokens,
            memory,
            train=train,
            dropout_rng=dropout_rng,
        )

    def _condition_action_tokens(
        self,
        observation,
        suffix_tokens,
        *,
        train: bool = False,
        dropout_rng=None,
    ):
        conditioned, _ = self._condition_action_tokens_with_waypoint(
            observation,
            suffix_tokens,
            train=train,
            dropout_rng=dropout_rng,
        )
        return conditioned

    def compute_loss_with_memory_aux(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng, memory_dropout_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        target_velocity = noise - actions
        prefix, prefix_mask, prefix_ar, history_mem, encoder_auxes = self._embed_prefix_with_history_mem(
            observation
        )
        suffix, suffix_mask, suffix_ar, adarms = self.embed_suffix(observation, x_t, time)
        suffix, waypoint = self._condition_action_tokens_with_waypoint(
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
            mask=self._make_attention_mask(input_mask, ar_mask),
            positions=positions,
            adarms_cond=[None, adarms],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow_loss = self._weighted_temporal_loss(observation, jnp.square(velocity - target_velocity))

        target_waypoint = jnp.stack(
            [actions[:, self.waypoint_action_index, dim] for dim in self.waypoint_action_dims],
            axis=-1,
        )
        waypoint_loss = jnp.mean(jnp.square(waypoint - target_waypoint), axis=-1)
        total_loss = flow_loss + self.waypoint_aux_weight * waypoint_loss[:, None]
        return total_loss, {
            "history_mem": history_mem,
            "encoder_auxes": encoder_auxes,
            "history_class_logits": None,
            "predicted_waypoint": waypoint,
            "waypoint_loss": waypoint_loss,
        }

    @staticmethod
    def _make_attention_mask(input_mask, ar_mask):
        # Local import avoids exposing an implementation detail in the config.
        from openpi.models import pi0_mem_compress as _pi0

        return _pi0.make_attn_mask(input_mask, ar_mask)

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        actions = super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)
        if self.waypoint_anchor_strength <= 0:
            return actions

        processed = _model.preprocess_observation(None, observation, train=False)
        dummy_tokens = jnp.zeros(
            (processed.state.shape[0], self.action_horizon, 1024),
            dtype=jnp.bfloat16,
        )
        _, waypoint = self._condition_action_tokens_with_waypoint(processed, dummy_tokens)
        strength = jnp.asarray(self.waypoint_anchor_strength, dtype=actions.dtype)
        # Absolute-EEF demonstrations keep the selected planar goal constant
        # throughout the chunk while Z/rotation/gripper carry the local motion.
        # Blend every planar command toward the decoded goal; merely shifting
        # the chunk would preserve the exact lateral drift this bridge is meant
        # to remove.
        for waypoint_dim, action_dim in enumerate(self.waypoint_action_dims):
            goal = waypoint[:, waypoint_dim, None]
            anchored = (1.0 - strength) * actions[:, :, action_dim] + strength * goal
            actions = actions.at[:, :, action_dim].set(anchored)
        return actions
