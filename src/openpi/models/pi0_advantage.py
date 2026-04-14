"""Pi0Advantage: Pi0 flow matching model with an auxiliary value head.

Extends Pi0 to jointly train a flow matching action predictor and a scalar
value predictor (normalized return). The value head is a 3-layer MLP applied
to the state-token representation from the action expert suffix.
"""

import dataclasses
import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, make_attn_mask
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi0AdvantageConfig(pi0_config.Pi0Config):
    """Pi0 + value head config for advantage estimation."""

    loss_value_weight: float = 0.0
    loss_action_weight: float = 1.0
    gamma: float = 1.0
    c_fail_mult: float = 1.0
    task_max_steps: int = 1500

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Advantage":
        return Pi0Advantage(self, rngs=nnx.Rngs(rng))


class Pi0Advantage(Pi0):
    """Pi0 with an auxiliary value head for advantage estimation.

    During training, ``compute_loss`` returns (combined_loss, aux_dict) where
    combined_loss has shape [B, AH] (action loss + broadcast value loss) and
    aux_dict contains per-component scalar losses for logging.

    The value head predicts a normalized return in [-1, 0] using the pi0.6
    shaped reward formulation.
    """

    def __init__(self, config: Pi0AdvantageConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        self.loss_value_weight = config.loss_value_weight
        self.loss_action_weight = config.loss_action_weight
        self.gamma = config.gamma
        self.c_fail_mult = config.c_fail_mult
        self.task_max_steps = config.task_max_steps

        action_expert_config = _gemma.get_config(config.action_expert_variant)
        width = action_expert_config.width

        # 3-layer MLP value head: Linear -> SiLU -> Linear -> SiLU -> Linear -> Tanh
        self.value_linear1 = nnx.Linear(width, width, rngs=rngs)
        self.value_linear2 = nnx.Linear(width, width, rngs=rngs)
        self.value_linear3 = nnx.Linear(width, 1, rngs=rngs)

    def _value_head(self, x: jnp.ndarray) -> jnp.ndarray:
        """3-layer MLP with SiLU activations and Tanh output."""
        x = self.value_linear1(x)
        x = nnx.swish(x)
        x = self.value_linear2(x)
        x = nnx.swish(x)
        x = self.value_linear3(x)
        x = jnp.tanh(x)
        return x

    def _paper_return_target_scalar(self, obs: _model.Observation) -> at.Float[at.Array, "b"]:
        """Normalized scalar target R_norm(t0) in [-1, 0] for current step t0.

        Uses the pi0.6 shaped reward:
          r_t = -1 for t < T
          r_T = 0 if success else -C_fail
          R = sum_{t'=t}^{T} r_{t'}
          R_norm = clip(R / task_max_steps, -1, 0)
        """
        if obs.terminal_reward is None:
            raise ValueError("Observation must have `terminal_reward` for value target.")
        if obs.step_index is None:
            raise ValueError("Observation must have `step_index` for value target.")
        if obs.episode_T is None:
            raise ValueError("Observation must have `episode_T` for value target.")

        terminal_reward = obs.terminal_reward
        step_index = obs.step_index
        episode_T = obs.episode_T

        success = terminal_reward > 0
        t0 = step_index.astype(jnp.int32)
        T = episode_T.astype(jnp.int32)
        task_max = float(self.task_max_steps)

        C_fail = self.c_fail_mult * task_max
        r_T = jnp.where(success, 0.0, -C_fail)

        remaining = jnp.maximum((T - t0).astype(jnp.float32), 0.0)
        R = -remaining + r_T

        R_norm = R / (task_max + 1e-8)
        R_norm = jnp.clip(R_norm, -1.0, 0.0)
        return R_norm

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, jnp.ndarray]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )

        # --- Action loss ---
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        squared_error = jnp.square(v_t - u_t)

        if observation.action_loss_mask is not None:
            mask = observation.action_loss_mask[..., None, :]
            squared_error_masked = squared_error * mask
            mask_sum = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1e-8)
            loss_action = jnp.sum(squared_error_masked, axis=-1) / jnp.squeeze(mask_sum, axis=-1)
        elif self.action_loss_mask is not None:
            mask = jnp.asarray(self.action_loss_mask)
            squared_error_masked = squared_error * mask
            loss_action = jnp.sum(squared_error_masked, axis=-1) / jnp.sum(mask)
        else:
            loss_action = jnp.mean(squared_error, axis=-1)  # [B, AH]

        # --- Value loss ---
        # State token is the first suffix token (Pi0 non-pi05 mode)
        deep_rep = suffix_out[:, 0, :].astype(jnp.float32)
        value_pred = self._value_head(deep_rep)  # [B, 1]

        R_norm = self._paper_return_target_scalar(observation)  # [B]
        value_target = R_norm[:, None]  # [B, 1]

        value_loss = jnp.square(value_pred - value_target)  # [B, 1]

        # --- Combined loss ---
        # loss_action: [B, AH], value_loss: [B, 1] -> broadcasts to [B, AH]
        loss = loss_action * self.loss_action_weight + value_loss * self.loss_value_weight

        aux_dict = {
            "loss_action": jnp.mean(loss_action),
            "loss_value": jnp.mean(value_loss),
        }

        return loss, aux_dict

    def sample_values(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ) -> at.Float[at.Array, "b 1"]:
        """Forward pass to predict value (normalized return) for the given observation."""
        observation = _model.preprocess_observation(None, observation, train=False)

        batch_size = observation.state.shape[0]
        actions_shape = (batch_size, self.action_horizon, self.action_dim)

        noise_rng, time_rng = jax.random.split(rng)
        noise_action = jax.random.normal(noise_rng, actions_shape)
        time = jax.random.uniform(time_rng, (batch_size,))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, noise_action, time)

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        deep_rep = suffix_out[:, 0, :].astype(jnp.float32)
        value_pred = self._value_head(deep_rep)
        return value_pred

    def compute_advantage(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "b"], at.Float[at.Array, "b"], at.Float[at.Array, "b"]]:
        """Compute advantage A = R - V for the given observation.

        Returns:
            (advantage, return_target, value_pred) each of shape [B].
        """
        preprocess_rng, sample_rng = jax.random.split(rng)
        obs = _model.preprocess_observation(preprocess_rng, observation, train=train)

        R_norm = self._paper_return_target_scalar(obs)

        value_pred = self.sample_values(sample_rng, obs)[:, 0]  # [B]
        value_pred = jax.lax.stop_gradient(value_pred)

        A_norm = R_norm - value_pred
        return A_norm, R_norm, value_pred
