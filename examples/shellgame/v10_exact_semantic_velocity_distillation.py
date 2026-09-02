"""Distill V10's executed diffusion velocity into semantic memory.

Teacher and student receive the exact same observation, noisy action ``x_t``
and diffusion time ``t``.  The frozen teacher uses V10's established memory
conditioner, while the student uses only the replacement semantic-memory
adapter.  Unlike token-residual distillation, the auxiliary target here is the
final normalized velocity emitted by the frozen Pi0.5 action expert.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from examples.shellgame import v10_exact_parallel_semantic_adapter as _base
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _pi_mem
from openpi.shared import array_typing as at


def velocity_distillation_terms(
    teacher_velocity,
    student_velocity,
    dimension_weights,
    *,
    eps: float = 1e-6,
):
    """Return direct MSE plus scale/direction diagnostics per action step."""
    teacher = jax.lax.stop_gradient(jnp.asarray(teacher_velocity, dtype=jnp.float32))
    student = jnp.asarray(student_velocity, dtype=jnp.float32)
    weights = jnp.broadcast_to(
        jnp.asarray(dimension_weights, dtype=jnp.float32), teacher.shape
    )
    denominator = jnp.maximum(jnp.sum(weights, axis=-1), eps)
    squared_error = jnp.square(student - teacher)
    mse = jnp.sum(squared_error * weights, axis=-1) / denominator
    teacher_energy = jnp.sum(jnp.square(teacher) * weights, axis=-1) / denominator
    normalized_mse = mse / jnp.maximum(teacher_energy, eps)

    sqrt_weights = jnp.sqrt(weights)
    teacher_weighted = teacher * sqrt_weights
    student_weighted = student * sqrt_weights
    dot = jnp.sum(teacher_weighted * student_weighted, axis=-1)
    teacher_norm = jnp.linalg.norm(teacher_weighted, axis=-1)
    student_norm = jnp.linalg.norm(student_weighted, axis=-1)
    cosine_distance = 1.0 - dot / jnp.maximum(teacher_norm * student_norm, eps)
    return mse, normalized_mse, cosine_distance


@dataclasses.dataclass(frozen=True)
class V10ExactSemanticVelocityDistillationConfig(
    _base.V10ExactParallelSemanticAdapterConfig
):
    velocity_distillation_weight: float = 1.0

    def create(self, rng: at.KeyArrayLike) -> V10ExactSemanticVelocityDistillationModel:
        return V10ExactSemanticVelocityDistillationModel(self, rngs=nnx.Rngs(rng))


class V10ExactSemanticVelocityDistillationModel(
    _base.V10ExactParallelSemanticAdapterModel
):
    """Use the complete frozen V10 policy as a final-output teacher."""

    def __init__(
        self,
        config: V10ExactSemanticVelocityDistillationConfig,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, rngs)
        if config.old_memory_condition_strength != 0.0:
            raise ValueError("Velocity distillation requires old memory strength 0 at inference")
        if config.velocity_distillation_weight < 0.0:
            raise ValueError("velocity_distillation_weight must be nonnegative")
        self.velocity_distillation_weight = float(config.velocity_distillation_weight)

    def _velocity_from_suffix(
        self,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        suffix_tokens,
        suffix_mask,
        suffix_ar_mask,
        adarms_cond,
    ):
        input_mask = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
        ar_mask = jnp.concatenate((prefix_ar_mask, suffix_ar_mask), axis=0)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=_pi_mem.make_attn_mask(input_mask, ar_mask),
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

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
            raise ValueError("Velocity distillation requires frame_index")

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        target_velocity = noise - actions

        raw_memory, memory_tokens, tracked = self._raw_and_resampled_memory(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_current_prefix(observation)
        raw_suffix, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        # Both branches see exactly the same raw suffix.  Gradients from the
        # teacher output are stopped; at inference the old branch is absent.
        teacher_suffix = self.ActionMemoryCrossAttention(raw_suffix, memory_tokens)
        teacher_velocity = jax.lax.stop_gradient(
            self._velocity_from_suffix(
                prefix_tokens,
                prefix_mask,
                prefix_ar_mask,
                teacher_suffix,
                suffix_mask,
                suffix_ar_mask,
                adarms_cond,
            )
        )
        student_suffix = self._apply_parallel_semantic_adapter(
            observation, raw_suffix, train=True
        )
        student_velocity = self._velocity_from_suffix(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            student_suffix,
            suffix_mask,
            suffix_ar_mask,
            adarms_cond,
        )

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

        action_squared_error = jnp.square(student_velocity - target_velocity)
        action_loss = jnp.sum(action_squared_error * dim_mask, axis=-1) / jnp.maximum(
            jnp.sum(dim_mask, axis=-1), 1e-8
        )
        velocity_mse, velocity_normalized_mse, velocity_cosine = (
            velocity_distillation_terms(
                teacher_velocity,
                student_velocity,
                dim_mask,
            )
        )
        loss_per_timestep = (
            action_loss + self.velocity_distillation_weight * velocity_mse
        )

        frame_index = jnp.asarray(observation.frame_index, dtype=jnp.int32)
        future_offsets = 1 + jnp.arange(self.action_horizon, dtype=jnp.int32)
        temporal_valid = frame_index[..., None] + future_offsets <= self.last_episode_frame
        valid_count = jnp.sum(temporal_valid, axis=-1, keepdims=True)
        temporal_scale = self.action_horizon / jnp.maximum(valid_count, 1)
        loss_per_timestep = (
            loss_per_timestep
            * temporal_valid.astype(loss_per_timestep.dtype)
            * temporal_scale
        )
        valid_float = temporal_valid.astype(jnp.float32)
        metric_denominator = jnp.maximum(jnp.sum(valid_float), 1.0)

        def valid_mean(value):
            return jnp.sum(value * valid_float) / metric_denominator

        return loss_per_timestep, {
            "history_mem": raw_memory,
            "encoder_auxes": (),
            "history_class_logits": tracked["joint_logits"],
            "temporal_valid_fraction": jnp.mean(temporal_valid),
            "extra_metrics": {
                "pure_action_loss": valid_mean(action_loss),
                "velocity_distillation_mse": valid_mean(velocity_mse),
                "velocity_distillation_normalized_mse": valid_mean(
                    velocity_normalized_mse
                ),
                "velocity_distillation_cosine": valid_mean(velocity_cosine),
            },
        }


def make_config_from_adapter(
    base: _base.V10ExactParallelSemanticAdapterConfig,
    *,
    velocity_distillation_weight: float = 1.0,
) -> V10ExactSemanticVelocityDistillationConfig:
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    return V10ExactSemanticVelocityDistillationConfig(
        **values,
        velocity_distillation_weight=velocity_distillation_weight,
    )
