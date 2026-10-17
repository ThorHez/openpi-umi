"""Distill V10 old-memory action-token residuals into semantic memory."""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from examples.shellgame import v10_exact_parallel_semantic_adapter as _base
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _pi_mem
from openpi.shared import array_typing as at


def residual_distillation_terms(raw_tokens, teacher_tokens, student_tokens, eps: float = 1e-6):
    """Return scale-normalized MSE and cosine distance for residual tokens."""
    teacher_delta = jax.lax.stop_gradient(teacher_tokens - raw_tokens).astype(jnp.float32)
    student_delta = (student_tokens - raw_tokens).astype(jnp.float32)
    teacher_energy = jnp.mean(jnp.square(teacher_delta), axis=-1)
    mse = jnp.mean(jnp.square(student_delta - teacher_delta), axis=-1)
    normalized_mse = mse / jnp.maximum(teacher_energy, eps)
    dot = jnp.sum(student_delta * teacher_delta, axis=-1)
    student_norm = jnp.sqrt(jnp.sum(jnp.square(student_delta), axis=-1))
    teacher_norm = jnp.sqrt(jnp.sum(jnp.square(teacher_delta), axis=-1))
    cosine_distance = 1.0 - dot / jnp.maximum(student_norm * teacher_norm, eps)
    return normalized_mse, cosine_distance


@dataclasses.dataclass(frozen=True)
class V10ExactSemanticResidualDistillationConfig(_base.V10ExactParallelSemanticAdapterConfig):
    residual_distillation_weight: float = 0.1
    residual_distillation_mse_fraction: float = 0.5

    def create(self, rng: at.KeyArrayLike) -> V10ExactSemanticResidualDistillationModel:
        return V10ExactSemanticResidualDistillationModel(self, rngs=nnx.Rngs(rng))


class V10ExactSemanticResidualDistillationModel(_base.V10ExactParallelSemanticAdapterModel):
    """Use frozen old V10 conditioning as a training-only token teacher."""

    def __init__(
        self,
        config: V10ExactSemanticResidualDistillationConfig,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, rngs)
        if config.old_memory_condition_strength != 0.0:
            raise ValueError("Residual distillation requires old memory strength 0 at inference")
        if config.residual_distillation_weight < 0.0:
            raise ValueError("residual_distillation_weight must be nonnegative")
        if not 0.0 <= config.residual_distillation_mse_fraction <= 1.0:
            raise ValueError("residual_distillation_mse_fraction must lie in [0,1]")
        self.residual_distillation_weight = float(config.residual_distillation_weight)
        self.residual_distillation_mse_fraction = float(config.residual_distillation_mse_fraction)

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
            raise ValueError("Residual distillation requires frame_index")

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        target_velocity = noise - actions

        raw_memory, memory_tokens, tracked = self._raw_and_resampled_memory(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_current_prefix(observation)
        raw_suffix, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        # Teacher and student see the exact same noisy action tokens. The old
        # branch is frozen and stop-gradient; it is absent from sample_actions.
        teacher_suffix = jax.lax.stop_gradient(self.ActionMemoryCrossAttention(raw_suffix, memory_tokens))
        student_suffix = self._apply_parallel_semantic_adapter(observation, raw_suffix, train=True)
        input_mask = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
        ar_mask = jnp.concatenate((prefix_ar_mask, suffix_ar_mask), axis=0)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, student_suffix],
            mask=_pi_mem.make_attn_mask(input_mask, ar_mask),
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        squared_error = jnp.square(velocity - target_velocity)

        if observation.action_loss_mask is not None:
            dim_mask = observation.action_loss_mask[..., None, :]
        else:
            dim_mask = jnp.asarray(self.action_loss_mask)[None, None, :]
        dimension_weights = jnp.ones((self.action_dim,), dtype=jnp.float32)
        dimension_weights = dimension_weights.at[self.gripper_action_index].set(self.gripper_loss_weight)
        dimension_weights = dimension_weights.at[self.real_action_dim :].set(0.0)
        dim_mask = dim_mask * dimension_weights[None, None, :]
        action_loss = jnp.sum(squared_error * dim_mask, axis=-1) / jnp.maximum(jnp.sum(dim_mask, axis=-1), 1e-8)

        mse, cosine = residual_distillation_terms(
            raw_suffix[:, -self.action_horizon :],
            teacher_suffix[:, -self.action_horizon :],
            student_suffix[:, -self.action_horizon :],
        )
        fraction = self.residual_distillation_mse_fraction
        distillation_loss = fraction * mse + (1.0 - fraction) * cosine
        loss_per_timestep = action_loss + self.residual_distillation_weight * distillation_loss

        frame_index = jnp.asarray(observation.frame_index, dtype=jnp.int32)
        future_offsets = 1 + jnp.arange(self.action_horizon, dtype=jnp.int32)
        temporal_valid = frame_index[..., None] + future_offsets <= self.last_episode_frame
        valid_count = jnp.sum(temporal_valid, axis=-1, keepdims=True)
        temporal_scale = self.action_horizon / jnp.maximum(valid_count, 1)
        loss_per_timestep = loss_per_timestep * temporal_valid.astype(loss_per_timestep.dtype) * temporal_scale
        return loss_per_timestep, {
            "history_mem": raw_memory,
            "encoder_auxes": (),
            "history_class_logits": tracked["joint_logits"],
            "temporal_valid_fraction": jnp.mean(temporal_valid),
            "extra_metrics": {
                "residual_distillation_mse": jnp.mean(mse),
                "residual_distillation_cosine": jnp.mean(cosine),
                "pure_action_loss": jnp.mean(action_loss),
            },
        }


def make_config_from_adapter(
    base: _base.V10ExactParallelSemanticAdapterConfig,
    *,
    distillation_weight: float = 0.1,
    mse_fraction: float = 0.5,
) -> V10ExactSemanticResidualDistillationConfig:
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    return V10ExactSemanticResidualDistillationConfig(
        **values,
        residual_distillation_weight=distillation_weight,
        residual_distillation_mse_fraction=mse_fraction,
    )
