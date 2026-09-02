"""Frozen waypoint policy with a small current-visual XY correction head.

The proven action expert keeps producing Z, rotation, and gripper commands.
The frozen semantic-memory bridge supplies the global planar waypoint.  A
separate lightweight head predicts one bounded local XY correction from the
current base/wrist images, robot state, frozen memory, and waypoint.  The same
correction is applied to the complete action chunk; this deliberately avoids
learning the future trajectory shape as an XY residual.
"""

from __future__ import annotations

import dataclasses

import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.tasks.shellgame import pi0_qwen_event_memory_action as _raw_action
from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_action as _waypoint


def clip_correction(correction, normalized_limits: tuple[float, ...]):
    limits = jnp.asarray(normalized_limits, dtype=correction.dtype)
    return jnp.clip(correction, -limits, limits)


def first_action_correction_target(
    actions,
    waypoint,
    action_dims: tuple[int, ...],
    normalized_limits: tuple[float, ...],
):
    """Return only the immediate oracle correction, never a future trajectory."""
    target = jnp.stack(
        [actions[:, 0, action_dim] - waypoint[:, waypoint_dim]
         for waypoint_dim, action_dim in enumerate(action_dims)],
        axis=-1,
    )
    return clip_correction(target, normalized_limits)


def apply_waypoint_correction(
    actions,
    waypoint,
    correction,
    action_dims: tuple[int, ...],
    normalized_limits: tuple[float, ...],
):
    """Overwrite planar chunk commands with waypoint plus one bounded delta."""
    correction = clip_correction(correction, normalized_limits)
    result = actions
    for waypoint_dim, action_dim in enumerate(action_dims):
        command = waypoint[:, waypoint_dim] + correction[:, waypoint_dim]
        result = result.at[:, :, action_dim].set(command[:, None])
    return result


class CurrentVisualXYCorrectionHead(nn.Module):
    """Small CNN/MLP head; the zero-init output preserves hard anchoring."""

    state_dim: int = 10
    memory_width: int = 64
    hidden_width: int = 256
    normalized_limits: tuple[float, float] = (0.132, 0.082)

    @nn.compact
    def __call__(self, images, state, semantic_memory, waypoint):
        visual_features = []
        for image_key in ("base_rgb", "wrist_rgb"):
            if image_key not in images:
                raise ValueError(f"Visual correction requires image key {image_key!r}")
            image = images[image_key]
            if image.ndim == 5:
                image = image[:, -1]
            if image.ndim != 4:
                raise ValueError(f"Expected {image_key} [B,H,W,C], got {image.shape}")
            x = image.astype(jnp.float32)
            for layer, width in enumerate((32, 64, 96, 128)):
                x = nn.Conv(
                    width,
                    kernel_size=(5, 5) if layer == 0 else (3, 3),
                    strides=(2, 2),
                    padding="SAME",
                    name=f"{image_key}_conv_{layer}",
                )(x)
                x = nn.gelu(x)
            visual_features.append(jnp.mean(x, axis=(1, 2)))

        memory = semantic_memory.astype(jnp.float32)
        memory_mean = jnp.mean(memory, axis=1)
        memory_std = jnp.sqrt(
            jnp.mean(jnp.square(memory - memory_mean[:, None, :]), axis=1) + 1e-6
        )
        features = jnp.concatenate(
            (
                *visual_features,
                state[..., : self.state_dim].astype(jnp.float32),
                memory_mean,
                memory_std,
                waypoint.astype(jnp.float32),
            ),
            axis=-1,
        )
        features = nn.LayerNorm(name="input_ln")(features)
        residual = nn.gelu(nn.Dense(self.hidden_width, name="input_projection")(features))
        update = nn.gelu(nn.Dense(self.hidden_width, name="hidden_0")(residual))
        update = nn.Dense(self.hidden_width, name="hidden_1")(update)
        features = nn.gelu(residual + update)
        # Exact zero initialization makes a newly attached head bitwise equal
        # to the established hard-waypoint policy before any optimization.
        raw = nn.Dense(
            len(self.normalized_limits),
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="correction_out",
        )(features)
        limits = jnp.asarray(self.normalized_limits, dtype=jnp.float32)
        return jnp.tanh(raw) * limits


@dataclasses.dataclass(frozen=True)
class Pi0QwenEventMemoryWaypointVisualCorrectionActionConfig(
    _waypoint.Pi0QwenEventMemoryWaypointActionConfig
):
    correction_state_dim: int = 10
    correction_hidden_width: int = 256
    # Nominal min/max normalization maps these values to about 10 mm in X/Y.
    correction_normalized_limits: tuple[float, float] = (0.132, 0.082)
    correction_scale: float = 1.0

    def create(self, rng: at.KeyArrayLike) -> Pi0QwenEventMemoryWaypointVisualCorrectionAction:
        return Pi0QwenEventMemoryWaypointVisualCorrectionAction(self, rngs=nnx.Rngs(rng))


class Pi0QwenEventMemoryWaypointVisualCorrectionAction(
    _waypoint.Pi0QwenEventMemoryWaypointAction
):
    """Preserve the base policy and learn only a trust-region XY correction."""

    def __init__(
        self,
        config: Pi0QwenEventMemoryWaypointVisualCorrectionActionConfig,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, rngs)
        limits = tuple(float(value) for value in config.correction_normalized_limits)
        if len(limits) != len(self.waypoint_action_dims) or any(value <= 0 for value in limits):
            raise ValueError("Correction limits must be positive and match waypoint_action_dims")
        if not 0.0 <= config.correction_scale <= 1.0:
            raise ValueError("correction_scale must lie in [0, 1]")
        self.correction_normalized_limits = limits
        self.correction_scale = float(config.correction_scale)
        self.CurrentVisualXYCorrectionHead = nnx_bridge.ToNNX(
            CurrentVisualXYCorrectionHead(
                state_dim=config.correction_state_dim,
                memory_width=config.semantic_memory_width,
                hidden_width=config.correction_hidden_width,
                normalized_limits=limits,
            )
        )
        dummy_images = {
            "base_rgb": jnp.zeros((1, 224, 224, 3), dtype=jnp.float32),
            "wrist_rgb": jnp.zeros((1, 224, 224, 3), dtype=jnp.float32),
        }
        self.CurrentVisualXYCorrectionHead.lazy_init(
            dummy_images,
            jnp.zeros((1, config.action_dim), dtype=jnp.float32),
            jnp.zeros(
                (1, config.semantic_memory_tokens, config.semantic_memory_width),
                dtype=jnp.float32,
            ),
            jnp.zeros((1, len(self.waypoint_action_dims)), dtype=jnp.float32),
            rngs=rngs,
        )

    def _decode_waypoint(self, observation):
        dummy_tokens = jnp.zeros(
            (observation.state.shape[0], self.action_horizon, 1024),
            dtype=jnp.bfloat16,
        )
        _, waypoint = self._condition_action_tokens_with_waypoint(observation, dummy_tokens)
        return waypoint

    def _predict_correction(self, observation, waypoint):
        if observation.semantic_memory is None:
            raise ValueError("Visual correction requires observation.semantic_memory")
        return self.CurrentVisualXYCorrectionHead(
            observation.images,
            observation.state,
            observation.semantic_memory,
            waypoint,
        )

    def compute_loss_with_memory_aux(self, rng, observation, actions, *, train=False):
        observation = _model.preprocess_observation(rng, observation, train=train)
        waypoint = jax.lax.stop_gradient(self._decode_waypoint(observation))
        predicted = self._predict_correction(observation, waypoint)
        target = first_action_correction_target(
            actions,
            waypoint,
            self.waypoint_action_dims,
            self.correction_normalized_limits,
        )
        limits = jnp.asarray(self.correction_normalized_limits, dtype=jnp.float32)
        correction_loss = jnp.mean(jnp.square((predicted - target) / limits), axis=-1)
        # The shared trainer expects a temporal loss. Broadcasting changes no
        # gradient while retaining its established logging/reduction contract.
        loss = jnp.broadcast_to(correction_loss[:, None], (actions.shape[0], self.action_horizon))
        return loss, {
            # Shape-stable neutral value for the shared trainer's legacy MEM
            # diagnostics; the external semantic memory itself stays frozen.
            "history_mem": jnp.zeros((actions.shape[0], 1, 1), dtype=jnp.float32),
            "encoder_auxes": (),
            "history_class_logits": None,
            "predicted_waypoint": waypoint,
            "predicted_xy_correction": predicted,
            "target_xy_correction": target,
            "xy_correction_loss": correction_loss,
        }

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        # Bypass the parent's fixed anchor, then restore it together with the
        # independently predicted correction.  Non-XY dimensions are untouched.
        actions = _raw_action.Pi0QwenEventMemoryAction.sample_actions(
            self,
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
        )
        processed = _model.preprocess_observation(None, observation, train=False)
        waypoint = self._decode_waypoint(processed)
        correction = self._predict_correction(processed, waypoint) * self.correction_scale
        return apply_waypoint_correction(
            actions,
            waypoint,
            correction,
            self.waypoint_action_dims,
            self.correction_normalized_limits,
        )
