"""Client-managed RTC adapted from kai0's ``src/openpi/models/pi0_rtc.py``.

Source revision: 9d93078c757840f50e75248c5c5a94ab7b41e13a. The model inherits
the unified Pi0 architecture, training loss and observation preprocessing, so
ordinary Pi0/Pi05 JAX checkpoints remain compatible. Previous actions must be
in model coordinates and already aligned with the new observation's time.
"""

import einops
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.shared import array_typing as at


def get_prefix_weights(start: int, end: int, total: int, schedule: str = "exp"):
    """Return kai0's prefix weights for the overlap ending at ``end``.

    ``start`` is the predicted inference delay. ``end`` is the number of
    aligned previous actions to guide, rather than an index into the original
    unshifted chunk. Unlike the server-managed mask, kai0's exponential ramp
    uses ``w * expm1(w) / expm1(1)``.
    """
    start = jnp.minimum(start, end)
    indices = jnp.arange(total)
    if schedule == "ones":
        weights = jnp.ones(total, dtype=jnp.float32)
    elif schedule == "zeros":
        weights = (indices < start).astype(jnp.float32)
    elif schedule in ("linear", "exp"):
        weights = jnp.clip((start - 1 - indices) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            weights = weights * jnp.expm1(weights) / jnp.expm1(1.0)
    else:
        raise ValueError(f"Invalid RTC prefix schedule: {schedule}")
    return jnp.where(indices >= end, 0, weights)


class Pi0RTC(pi0.Pi0):
    """Pi0/Pi05 with kai0's client-managed, inference-time RTC interface."""

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        prev_action_chunk: at.Float[at.Array, "b ph pd"] | None = None,
        inference_delay: int | None = None,
        execute_horizon: int | None = None,
        mask_prefix_delay: bool = False,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 0.5,
        enable_rtc: bool = True,
        prev_action_dim: int | None = None,
    ) -> _model.Actions:
        if prev_action_chunk is None:
            return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)

        observation = self.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        previous = jnp.asarray(prev_action_chunk, dtype=noise.dtype)
        if previous.ndim == 2:
            previous = previous[None, ...]
        if previous.ndim != 3 or previous.shape[0] != batch_size:
            raise ValueError("prev_action_chunk must have shape (batch, prefix_horizon, action_dim)")
        provided_horizon, provided_dim = previous.shape[1:]
        if not 0 < provided_horizon <= self.action_horizon or not 0 < provided_dim <= self.action_dim:
            raise ValueError("prev_action_chunk must fit the model action horizon and dimensions")
        previous = jnp.pad(
            previous,
            ((0, 0), (0, self.action_horizon - provided_horizon), (0, self.action_dim - provided_dim)),
        )
        end = jnp.clip(provided_horizon if execute_horizon is None else execute_horizon, 1, provided_horizon)
        delay = jnp.clip(0 if inference_delay is None else inference_delay, 0, end)
        weights = get_prefix_weights(delay, end, self.action_horizon, prefix_attention_schedule)
        # Policy pads wire prefixes to a fixed (H, D), retaining this dynamic
        # valid-width scalar so varying client prefixes reuse the same JIT.
        valid_dim = provided_dim if prev_action_dim is None else jnp.clip(prev_action_dim, 0, provided_dim)
        dimension_mask = (jnp.arange(self.action_dim) < valid_dim)[None, None, :]
        guidance_mask = weights[None, :, None] * dimension_mask
        prefix_mask = (jnp.arange(self.action_horizon) < delay)[None, :, None] & dimension_mask
        use_guidance = jnp.asarray(enable_rtc) & (max_guidance_weight > 0) & (valid_dim > 0)

        prefix_tokens, prefix_valid, prefix_ar = self.embed_prefix(observation)
        prefix_attention = pi0.make_attn_mask(prefix_valid, prefix_ar)
        positions = jnp.cumsum(prefix_valid, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attention, positions=positions)

        def velocity(actions, time):
            suffix, suffix_valid, suffix_ar, adarms_cond = self.embed_suffix(
                observation, actions, jnp.broadcast_to(time, batch_size)
            )
            suffix_attention = pi0.make_attn_mask(suffix_valid, suffix_ar)
            cross_attention = einops.repeat(prefix_valid, "b p -> b s p", s=suffix.shape[1])
            attention = jnp.concatenate([cross_attention, suffix_attention], axis=-1)
            suffix_positions = jnp.sum(prefix_valid, axis=-1)[:, None] + jnp.cumsum(suffix_valid, axis=-1) - 1
            (_, suffix_output), _ = self.PaliGemma.llm(
                [None, suffix],
                mask=attention,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return self.action_out_proj(suffix_output[:, -self.action_horizon :])

        dt = -1.0 / num_steps

        def step(carry):
            actions, time = carry

            def guided_velocity():
                denoiser_input = jnp.where(jnp.asarray(mask_prefix_delay) & prefix_mask, previous, actions)

                def denoiser(value):
                    predicted_velocity = velocity(value, time)
                    return value - time * predicted_velocity, predicted_velocity

                estimate, pullback, predicted_velocity = jax.vjp(denoiser, denoiser_input, has_aux=True)
                correction = pullback((previous - estimate) * guidance_mask)[0]
                # kai0 follows the paper's forward time tau = 1 - openpi_time.
                # Bound the denominator at tau=0 before clipping the guidance.
                tau = jnp.clip(1.0 - time, 1e-3, 1.0)
                noise_time = 1.0 - tau
                numerator = jnp.square(noise_time) + jnp.square(tau)
                raw_weight = numerator / jnp.maximum(tau * noise_time, 1e-8)
                weight = jnp.minimum(jnp.maximum(max_guidance_weight, 0.0), raw_weight)
                return predicted_velocity - weight * correction

            predicted_velocity = jax.lax.cond(use_guidance, guided_velocity, lambda: velocity(actions, time))
            return actions + dt * predicted_velocity, time + dt

        actions, _ = jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, step, (noise, 1.0))
        return actions
