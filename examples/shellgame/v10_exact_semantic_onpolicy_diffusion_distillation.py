"""Distill V10 velocities on states visited by the semantic student sampler.

The student starts from pure Gaussian action noise and is unrolled with the
same Euler schedule used at inference.  At every student-visited ``x_t``, the
frozen V10 old-memory policy supplies the target velocity.  The rollout state
is stopped between denoising steps, yielding a DAgger-style local vector-field
objective without backpropagating through the whole sampler trajectory.
"""

from __future__ import annotations

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from examples.shellgame import v10_exact_parallel_semantic_adapter as _base
from examples.shellgame import v10_exact_semantic_velocity_distillation as _velocity
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _pi_mem
from openpi.shared import array_typing as at


def inference_diffusion_times(num_steps: int) -> jax.Array:
    """Return the exact time values queried by the Euler inference loop."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    return 1.0 - jnp.arange(num_steps, dtype=jnp.float32) / float(num_steps)


@dataclasses.dataclass(frozen=True)
class V10ExactSemanticOnPolicyDiffusionDistillationConfig(
    _base.V10ExactParallelSemanticAdapterConfig
):
    onpolicy_num_steps: int = 4
    onpolicy_velocity_weight: float = 1.0

    def create(
        self, rng: at.KeyArrayLike
    ) -> V10ExactSemanticOnPolicyDiffusionDistillationModel:
        return V10ExactSemanticOnPolicyDiffusionDistillationModel(
            self, rngs=nnx.Rngs(rng)
        )


class V10ExactSemanticOnPolicyDiffusionDistillationModel(
    _base.V10ExactParallelSemanticAdapterModel
):
    """Query frozen V10 at every state induced by the student sampler."""

    def __init__(
        self,
        config: V10ExactSemanticOnPolicyDiffusionDistillationConfig,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, rngs)
        if config.old_memory_condition_strength != 0.0:
            raise ValueError("On-policy distillation requires old memory strength 0 at inference")
        if config.onpolicy_num_steps < 1:
            raise ValueError("onpolicy_num_steps must be positive")
        if config.onpolicy_velocity_weight <= 0.0:
            raise ValueError("onpolicy_velocity_weight must be positive")
        self.onpolicy_num_steps = int(config.onpolicy_num_steps)
        self.onpolicy_velocity_weight = float(config.onpolicy_velocity_weight)

    def _prefix_cache(self, observation):
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_current_prefix(
            observation
        )
        prefix_attn_mask = _pi_mem.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        return prefix_mask, kv_cache

    def _velocity_from_cached_prefix(
        self,
        prefix_mask,
        kv_cache,
        suffix_tokens,
        suffix_mask,
        suffix_ar_mask,
        adarms_cond,
    ):
        suffix_attn_mask = _pi_mem.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_for_suffix = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate(
            (prefix_for_suffix, suffix_attn_mask), axis=-1
        )
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
        preprocess_rng, noise_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=False
        )
        if observation.frame_index is None:
            raise ValueError("On-policy diffusion distillation requires frame_index")

        raw_memory, memory_tokens, tracked = self._raw_and_resampled_memory(observation)
        prefix_mask, kv_cache = self._prefix_cache(observation)

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

        batch_shape = actions.shape[:-2]
        student_x = jax.random.normal(noise_rng, actions.shape)
        dt = -1.0 / self.onpolicy_num_steps
        diffusion_times = inference_diffusion_times(self.onpolicy_num_steps)
        step_mse = []
        step_normalized_mse = []
        step_cosine = []

        for step_index in range(self.onpolicy_num_steps):
            time = jnp.full(
                batch_shape,
                diffusion_times[step_index],
                dtype=jnp.float32,
            )
            raw_suffix, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, student_x, time
            )
            teacher_suffix = self.ActionMemoryCrossAttention(
                raw_suffix, memory_tokens
            )
            teacher_velocity = jax.lax.stop_gradient(
                self._velocity_from_cached_prefix(
                    prefix_mask,
                    kv_cache,
                    teacher_suffix,
                    suffix_mask,
                    suffix_ar_mask,
                    adarms_cond,
                )
            )
            student_suffix = self._apply_parallel_semantic_adapter(
                observation, raw_suffix, train=True
            )
            student_velocity = self._velocity_from_cached_prefix(
                prefix_mask,
                kv_cache,
                student_suffix,
                suffix_mask,
                suffix_ar_mask,
                adarms_cond,
            )
            mse, normalized_mse, cosine = _velocity.velocity_distillation_terms(
                teacher_velocity,
                student_velocity,
                dim_mask,
            )
            step_mse.append(mse)
            step_normalized_mse.append(normalized_mse)
            step_cosine.append(cosine)

            # The next training state is produced by the student exactly as at
            # inference.  Stop-gradient makes each visited state a local
            # DAgger sample instead of differentiating through the full ODE.
            student_x = jax.lax.stop_gradient(student_x + dt * student_velocity)

        mse_by_step = jnp.stack(step_mse, axis=0)
        normalized_mse_by_step = jnp.stack(step_normalized_mse, axis=0)
        cosine_by_step = jnp.stack(step_cosine, axis=0)
        loss_per_timestep = self.onpolicy_velocity_weight * jnp.mean(
            mse_by_step, axis=0
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

        extra_metrics = {
            "onpolicy_velocity_mse": valid_mean(jnp.mean(mse_by_step, axis=0)),
            "onpolicy_velocity_normalized_mse": valid_mean(
                jnp.mean(normalized_mse_by_step, axis=0)
            ),
            "onpolicy_velocity_cosine": valid_mean(
                jnp.mean(cosine_by_step, axis=0)
            ),
        }
        for step_index in range(self.onpolicy_num_steps):
            extra_metrics[f"onpolicy_velocity_mse_step{step_index}"] = valid_mean(
                mse_by_step[step_index]
            )
            extra_metrics[
                f"onpolicy_velocity_normalized_mse_step{step_index}"
            ] = valid_mean(normalized_mse_by_step[step_index])
            extra_metrics[f"onpolicy_velocity_cosine_step{step_index}"] = valid_mean(
                cosine_by_step[step_index]
            )

        return loss_per_timestep, {
            "history_mem": raw_memory,
            "encoder_auxes": (),
            "history_class_logits": tracked["joint_logits"],
            "temporal_valid_fraction": jnp.mean(temporal_valid),
            "extra_metrics": extra_metrics,
        }


def make_config_from_adapter(
    base: _base.V10ExactParallelSemanticAdapterConfig,
    *,
    onpolicy_num_steps: int = 4,
    onpolicy_velocity_weight: float = 1.0,
) -> V10ExactSemanticOnPolicyDiffusionDistillationConfig:
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    return V10ExactSemanticOnPolicyDiffusionDistillationConfig(
        **values,
        onpolicy_num_steps=onpolicy_num_steps,
        onpolicy_velocity_weight=onpolicy_velocity_weight,
    )
