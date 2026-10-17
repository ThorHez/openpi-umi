"""Exact V10 action policy with a zero-init parallel semantic-memory adapter.

The established V10 path remains intact:

    frames 0..59 -> old tracker -> old query resampler
    -> old ActionMemoryCrossAttention -> Pi0.5 action expert

An external semantic memory is added only as a parallel residual on the
already-conditioned action tokens.  Its effective gate is exactly zero at
initialization.  Therefore a model restored from a complete V10 checkpoint is
numerically identical before the adapter is trained; unlike the earlier
"action transplant", no V10 conditioning component is removed or replaced.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from examples.shellgame import train_old_tracker_full_joint_grasp as _v10
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _pi_mem
from openpi.shared import array_typing as at
from openpi.tasks.robomme.pickxtimes import pi0_memory_action as _semantic

PARALLEL_ADAPTER_PREFIX = "ParallelSemanticMemoryActionConditioner/"


def blend_old_memory_condition(action_tokens, old_conditioned_tokens, strength: float):
    """Continuously retain or remove only V10's old-memory residual."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("old memory strength must lie in [0, 1]")
    return action_tokens + strength * (old_conditioned_tokens - action_tokens)


@dataclasses.dataclass(frozen=True)
class V10ExactParallelSemanticAdapterConfig(_v10.OldTrackerFullJointGraspConfig):
    semantic_memory_tokens: int = 128
    semantic_memory_width: int = 64
    semantic_query_tokens: int = 8
    semantic_hidden_width: int = 256
    semantic_residual_gate_init: float = 0.0
    # The strict baseline bypasses the adapter at Python trace time and calls
    # the original V10 methods. Training must explicitly enable the branch.
    parallel_semantic_adapter_enabled: bool = False
    # 1.0 is original V10; 0.0 removes only the old-memory residual while
    # preserving tracker computation, current-image prefix and action expert.
    old_memory_condition_strength: float = 1.0

    def create(self, rng: at.KeyArrayLike) -> V10ExactParallelSemanticAdapterModel:
        return V10ExactParallelSemanticAdapterModel(self, rngs=nnx.Rngs(rng))


class V10ExactParallelSemanticAdapterModel(_v10.OldTrackerFullJointGraspModel):
    """Preserve every V10 operation and add one initially inactive branch."""

    def __init__(self, config: V10ExactParallelSemanticAdapterConfig, rngs: nnx.Rngs):
        if config.semantic_residual_gate_init != 0.0:
            raise ValueError("The strict V10 equivalence model requires a zero initial gate")
        super().__init__(config, rngs)
        self.parallel_semantic_adapter_enabled = bool(
            config.parallel_semantic_adapter_enabled
        )
        if not 0.0 <= config.old_memory_condition_strength <= 1.0:
            raise ValueError("old_memory_condition_strength must lie in [0, 1]")
        self.old_memory_condition_strength = float(config.old_memory_condition_strength)
        self.ParallelSemanticMemoryActionConditioner = nnx_bridge.ToNNX(
            _semantic.SemanticMemoryActionConditioner(
                memory_tokens=config.semantic_memory_tokens,
                memory_width=config.semantic_memory_width,
                query_tokens=config.semantic_query_tokens,
                hidden_width=config.semantic_hidden_width,
                dtype_mm=config.dtype,
                residual_gate_init=0.0,
                residual_dropout_rate=0.0,
            )
        )
        self.ParallelSemanticMemoryActionConditioner.lazy_init(
            jnp.zeros((1, config.action_horizon, 1024), dtype=jnp.bfloat16),
            jnp.zeros(
                (1, config.semantic_memory_tokens, config.semantic_memory_width),
                dtype=jnp.float32,
            ),
            rngs=rngs,
        )

    def _apply_parallel_semantic_adapter(
        self, observation, action_tokens, *, train: bool = False
    ):
        if observation.semantic_memory is None:
            raise ValueError("The parallel V10 experiment requires semantic_memory")
        return self.ParallelSemanticMemoryActionConditioner(
            action_tokens,
            observation.semantic_memory,
            train=train,
        )

    def _apply_old_memory_condition(self, action_tokens, memory_tokens):
        old_conditioned = self.ActionMemoryCrossAttention(action_tokens, memory_tokens)
        return blend_old_memory_condition(
            action_tokens,
            old_conditioned,
            self.old_memory_condition_strength,
        )

    def compute_loss_with_memory_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        if not self.parallel_semantic_adapter_enabled:
            return super().compute_loss_with_memory_aux(
                rng, observation, actions, train=train
            )
        del train
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=False)
        if observation.frame_index is None:
            raise ValueError("Full-action temporal masking requires frame_index")

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        target_velocity = noise - actions

        raw_memory, memory_tokens, tracked = self._raw_and_resampled_memory(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_current_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
        # The complete established V10 condition remains first and unchanged.
        suffix_tokens = self._apply_old_memory_condition(suffix_tokens, memory_tokens)
        suffix_tokens = self._apply_parallel_semantic_adapter(
            observation, suffix_tokens, train=True
        )
        input_mask = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
        ar_mask = jnp.concatenate((prefix_ar_mask, suffix_ar_mask), axis=0)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
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
        dimension_weights = dimension_weights.at[self.gripper_action_index].set(
            self.gripper_loss_weight
        )
        dimension_weights = dimension_weights.at[self.real_action_dim :].set(0.0)
        dim_mask = dim_mask * dimension_weights[None, None, :]
        loss_per_timestep = jnp.sum(squared_error * dim_mask, axis=-1) / jnp.maximum(
            jnp.sum(dim_mask, axis=-1), 1e-8
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
        return loss_per_timestep, {
            "history_mem": raw_memory,
            "encoder_auxes": (),
            "history_class_logits": tracked["joint_logits"],
            "temporal_valid_fraction": jnp.mean(temporal_valid),
        }

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        # Preserve the byte-for-byte original V10 inference path whenever no
        # ablation or parallel adapter is requested.  A zero old-memory
        # strength deliberately takes the explicit path below so the tracker
        # residual can be removed while retaining V10's current-image prefix,
        # state conditioning, action expert, and checkpoint weights.
        if (
            not self.parallel_semantic_adapter_enabled
            and self.old_memory_condition_strength == 1.0
        ):
            return super().sample_actions(
                rng, observation, num_steps=num_steps, noise=noise
            )
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(
                rng, (batch_size, self.action_horizon, self.action_dim)
            )

        memory_tokens = None
        if self.old_memory_condition_strength != 0.0:
            _, memory_tokens, _ = self._raw_and_resampled_memory(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_current_prefix(observation)
        prefix_attn_mask = _pi_mem.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            if self.old_memory_condition_strength != 0.0:
                assert memory_tokens is not None
                suffix_tokens = self._apply_old_memory_condition(
                    suffix_tokens, memory_tokens
                )
            if self.parallel_semantic_adapter_enabled:
                suffix_tokens = self._apply_parallel_semantic_adapter(
                    observation, suffix_tokens
                )
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
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * velocity, time + dt

        def cond(carry):
            return carry[1] >= -dt / 2

        sampled, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return sampled


def merge_exact_v10_with_fresh_parallel_adapter(
    target_params: dict[str, Any],
    v10_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Restore every original leaf exactly; preserve only new adapter leaves."""
    target = flax.traverse_util.flatten_dict(target_params, sep="/")
    source = flax.traverse_util.flatten_dict(v10_params, sep="/")
    merged = {}
    counts = {"v10": 0, "parallel_adapter": 0}
    missing = []
    mismatched = []
    unexpected_v10 = []

    for path, reference in target.items():
        if path.startswith(PARALLEL_ADAPTER_PREFIX):
            merged[path] = reference
            counts["parallel_adapter"] += 1
            continue
        candidate = source.get(path)
        if candidate is None:
            missing.append(path)
            continue
        if np.shape(candidate) != np.shape(reference):
            mismatched.append((path, np.shape(reference), np.shape(candidate)))
            continue
        # Training validates the configured model dtype strictly. This cast is
        # the same restore-time dtype normalization used by the normal policy
        # loader; paths, shapes and numerical values remain checkpoint-derived.
        reference_dtype = getattr(reference, "dtype", None)
        merged[path] = np.asarray(candidate, dtype=reference_dtype)
        counts["v10"] += 1

    unexpected_v10 = [path for path in source if path not in target]
    if missing or mismatched or unexpected_v10:
        raise ValueError(
            "Exact V10 restore failed: "
            f"missing={missing[:8]} mismatched={mismatched[:8]} "
            f"unexpected_v10={unexpected_v10[:8]}"
        )
    return flax.traverse_util.unflatten_dict(merged, sep="/"), counts


def make_config_from_v10(
    base: _v10.OldTrackerFullJointGraspConfig,
) -> V10ExactParallelSemanticAdapterConfig:
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    return V10ExactParallelSemanticAdapterConfig(**values)
