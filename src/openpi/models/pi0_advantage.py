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
    """Pi0 + deterministic 201-bin distributional value head."""

    loss_value_weight: float = 0.0
    loss_action_weight: float = 1.0

    # return shaping
    gamma: float = 1.0
    c_fail_mult: float = 1.0
    task_max_steps: int = 1500
    episode_T_is_length: bool = False

    # distributional critic
    num_value_bins: int = 201
    value_min: float = -1.0
    value_max: float = 0.0

    # True: two-hot soft targets (more stable)
    # False: hard one-hot targets (closer to strict paper-style CE)
    soft_value_targets: bool = True

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Advantage":
        return Pi0Advantage(self, rngs=nnx.Rngs(rng))


class Pi0Advantage(Pi0):
    """Pi0 with a deterministic auxiliary distributional value head.

    Main differences from the earlier scalar version:
      - value head outputs logits over 201 bins in [value_min, value_max]
      - value prediction is the expectation over the bin distribution
      - value branch is deterministic and observation-only
    """

    def __init__(self, config: Pi0AdvantageConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        self.loss_value_weight = config.loss_value_weight
        self.loss_action_weight = config.loss_action_weight

        self.gamma = config.gamma
        self.c_fail_mult = config.c_fail_mult
        self.task_max_steps = config.task_max_steps
        self.episode_T_is_length = config.episode_T_is_length

        self.num_value_bins = config.num_value_bins
        self.value_min = config.value_min
        self.value_max = config.value_max
        self.soft_value_targets = config.soft_value_targets

        action_expert_config = _gemma.get_config(config.action_expert_variant)
        width = action_expert_config.width

        # pooled prefix feature -> MLP -> logits[num_value_bins]
        self.value_linear1 = nnx.Linear(width, width, rngs=rngs)
        self.value_linear2 = nnx.Linear(width, width, rngs=rngs)
        self.value_linear3 = nnx.Linear(width, self.num_value_bins, rngs=rngs)

    # -------------------------------------------------------------------------
    # Distributional critic helpers
    # -------------------------------------------------------------------------

    def _value_head_logits(self, x: jnp.ndarray) -> jnp.ndarray:
        """MLP head producing logits over value bins."""
        x = self.value_linear1(x)
        x = nnx.swish(x)
        x = self.value_linear2(x)
        x = nnx.swish(x)
        x = self.value_linear3(x)
        return x  # [B, num_value_bins]

    def _bin_centers(self) -> jnp.ndarray:
        """Bin centers in [value_min, value_max]. Shape: [num_bins]."""
        return jnp.linspace(
            float(self.value_min),
            float(self.value_max),
            int(self.num_value_bins),
            dtype=jnp.float32,
        )

    def _expected_value_from_logits(self, logits: jnp.ndarray) -> jnp.ndarray:
        """E[V] from categorical logits. Returns shape [B, 1]."""
        probs = jax.nn.softmax(logits, axis=-1)  # [B, K]
        centers = self._bin_centers()[None, :]   # [1, K]
        value = jnp.sum(probs * centers, axis=-1, keepdims=True)  # [B, 1]
        return value

    def _value_targets_to_probs(self, values: jnp.ndarray) -> jnp.ndarray:
        """Convert scalar value targets in [value_min, value_max] to target probs.

        If soft_value_targets=True:
            use two-hot interpolation over the two nearest bins.
        Else:
            use hard one-hot over the nearest bin.
        """
        v = jnp.clip(values, self.value_min, self.value_max)  # [B]

        # scaled in [0, K-1]
        scaled = (v - self.value_min) / (self.value_max - self.value_min + 1e-8)
        scaled = scaled * float(self.num_value_bins - 1)

        if not self.soft_value_targets:
            idx = jnp.rint(scaled).astype(jnp.int32)
            idx = jnp.clip(idx, 0, self.num_value_bins - 1)
            return jax.nn.one_hot(idx, self.num_value_bins, dtype=jnp.float32)

        lo = jnp.floor(scaled).astype(jnp.int32)
        hi = jnp.clip(lo + 1, 0, self.num_value_bins - 1)
        lo = jnp.clip(lo, 0, self.num_value_bins - 1)

        hi_w = scaled - lo.astype(jnp.float32)
        lo_w = 1.0 - hi_w

        lo_oh = jax.nn.one_hot(lo, self.num_value_bins, dtype=jnp.float32)
        hi_oh = jax.nn.one_hot(hi, self.num_value_bins, dtype=jnp.float32)

        probs = lo_w[:, None] * lo_oh + hi_w[:, None] * hi_oh
        return probs  # [B, K]

    def _distribution_ce_loss(self, logits: jnp.ndarray, target_probs: jnp.ndarray) -> jnp.ndarray:
        """Cross-entropy between target categorical probs and predicted logits.

        Returns shape [B, 1].
        """
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(target_probs * log_probs, axis=-1, keepdims=True)
        return loss

    # -------------------------------------------------------------------------
    # Target construction
    # -------------------------------------------------------------------------

    def _get_episode_success(self, obs: _model.Observation) -> jnp.ndarray:
        """Prefer explicit episode_success; fallback to terminal_reward > 0."""
        episode_success = getattr(obs, "episode_success", None)
        if episode_success is not None:
            return jnp.asarray(episode_success).astype(jnp.bool_)

        terminal_reward = getattr(obs, "terminal_reward", None)
        if terminal_reward is None:
            raise ValueError(
                "Observation must provide either `episode_success` or `terminal_reward` "
                "to build value targets."
            )
        return jnp.asarray(terminal_reward > 0).astype(jnp.bool_)

    def _get_task_max_steps(self, obs: _model.Observation) -> jnp.ndarray:
        """Use per-sample task_max_steps if available, else fallback to config."""
        step_index = getattr(obs, "step_index", None)
        if step_index is None:
            raise ValueError("Observation must have `step_index`.")

        task_max_steps = getattr(obs, "task_max_steps", None)
        if task_max_steps is None:
            task_max = jnp.full_like(
                jnp.asarray(step_index, dtype=jnp.float32),
                fill_value=float(self.task_max_steps),
                dtype=jnp.float32,
            )
        else:
            task_max = jnp.asarray(task_max_steps, dtype=jnp.float32)
            task_max = jnp.broadcast_to(task_max, jnp.shape(step_index))
        return task_max

    def _paper_return_target_scalar(self, obs: _model.Observation) -> at.Float[at.Array, "b"]:
        """Normalized scalar target R_norm in [value_min, value_max].

        Reward shaping:
          r_t = -1 for intermediate steps
          r_T = 0 if success else -C_fail
          R_t = sum_{t' = t}^T r_{t'}
          R_norm = clip(R_t / task_max_steps, value_min, value_max)

        If gamma != 1:
          R_t = -sum_{k=0}^{n-1} gamma^k + gamma^n * r_T
        where n is the number of non-terminal steps before terminal.
        """
        step_index = getattr(obs, "step_index", None)
        episode_T = getattr(obs, "episode_T", None)

        if step_index is None:
            raise ValueError("Observation must have `step_index`.")
        if episode_T is None:
            raise ValueError("Observation must have `episode_T`.")

        success = self._get_episode_success(obs)
        task_max = self._get_task_max_steps(obs)

        t0 = jnp.asarray(step_index, dtype=jnp.int32)
        T_raw = jnp.asarray(episode_T, dtype=jnp.int32)

        if self.episode_T_is_length:
            last_idx = jnp.maximum(T_raw - 1, 0)
        else:
            last_idx = T_raw

        n = jnp.maximum(last_idx - t0, 0).astype(jnp.float32)

        C_fail = self.c_fail_mult * task_max
        r_T = jnp.where(success, 0.0, -C_fail)

        gamma = jnp.asarray(self.gamma, dtype=jnp.float32)
        if float(self.gamma) == 1.0:
            R = -n + r_T
        else:
            neg_part = -(1.0 - jnp.power(gamma, n)) / (1.0 - gamma + 1e-8)
            R = neg_part + jnp.power(gamma, n) * r_T

        R_norm = R / (task_max + 1e-8)
        R_norm = jnp.clip(R_norm, self.value_min, self.value_max)
        return R_norm  # [B]

    # -------------------------------------------------------------------------
    # Deterministic observation-only value features
    # -------------------------------------------------------------------------

    def _encode_value_features(self, obs: _model.Observation) -> jnp.ndarray:
        """Deterministic observation-only features for the critic.

        Strategy:
          - run the normal prefix encoding
          - attach a dummy suffix only for API compatibility
          - take only prefix_out
          - do masked mean pooling over prefix tokens

        Under the usual causal mask, prefix tokens do not depend on later suffix
        tokens, so this stays deterministic and observation-only.
        """
        batch_size = obs.state.shape[0]

        dummy_x = jnp.zeros(
            (batch_size, self.action_horizon, self.action_dim),
            dtype=jnp.float32,
        )
        dummy_t = jnp.zeros((batch_size,), dtype=jnp.float32)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            obs, dummy_x, dummy_t
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        prefix_out = prefix_out.astype(jnp.float32)               # [B, Lp, D]
        prefix_mask_f = prefix_mask.astype(jnp.float32)           # [B, Lp]

        denom = jnp.maximum(jnp.sum(prefix_mask_f, axis=1, keepdims=True), 1.0)
        pooled = jnp.sum(prefix_out * prefix_mask_f[..., None], axis=1) / denom
        return pooled  # [B, D]

    def _predict_value_logits_from_processed_obs(self, obs: _model.Observation) -> jnp.ndarray:
        feat = self._encode_value_features(obs)
        logits = self._value_head_logits(feat)
        return logits  # [B, K]

    # -------------------------------------------------------------------------
    # Action loss helper
    # -------------------------------------------------------------------------

    def _compute_action_loss(
        self,
        squared_error: jnp.ndarray,
        observation: _model.Observation,
    ) -> jnp.ndarray:
        """Return action loss with shape [B, AH]."""
        if observation.action_loss_mask is not None:
            mask = jnp.asarray(observation.action_loss_mask, dtype=squared_error.dtype)
        elif self.action_loss_mask is not None:
            mask = jnp.asarray(self.action_loss_mask, dtype=squared_error.dtype)
        else:
            return jnp.mean(squared_error, axis=-1)

        while mask.ndim < squared_error.ndim:
            mask = jnp.expand_dims(mask, axis=-2)

        masked = squared_error * mask
        denom = jnp.maximum(jnp.sum(mask, axis=-1), 1e-8)
        loss_action = jnp.sum(masked, axis=-1) / denom
        return loss_action

    # -------------------------------------------------------------------------
    # Main APIs
    # -------------------------------------------------------------------------

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, jnp.ndarray]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        obs = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # -------------------------------
        # Action branch (same as Pi0 FM)
        # -------------------------------
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(obs, x_t, time)

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

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])  # [B, AH, AD]
        squared_error = jnp.square(v_t - u_t)
        loss_action = self._compute_action_loss(squared_error, obs)  # [B, AH]

        # -------------------------------
        # Distributional critic branch
        # -------------------------------
        value_logits = self._predict_value_logits_from_processed_obs(obs)    # [B, K]
        value_target = self._paper_return_target_scalar(obs)                 # [B]
        value_target_probs = self._value_targets_to_probs(value_target)      # [B, K]
        value_loss = self._distribution_ce_loss(value_logits, value_target_probs)  # [B, 1]

        value_pred = self._expected_value_from_logits(value_logits)          # [B, 1]

        # -------------------------------
        # Combined loss
        # -------------------------------
        # Keep [B, AH] output shape for compatibility with the Pi0 training loop.
        # Divide the broadcasted value term by AH so its effective weight does not
        # silently scale with action_horizon.
        loss = (
            self.loss_action_weight * loss_action
            + self.loss_value_weight * (value_loss / float(self.action_horizon))
        )

        # useful monitoring stats
        pred_probs = jax.nn.softmax(value_logits, axis=-1)
        pred_entropy = -jnp.sum(pred_probs * jnp.log(pred_probs + 1e-8), axis=-1)

        aux_dict = {
            "loss_action": jnp.mean(loss_action),
            "loss_value": jnp.mean(value_loss),
            "value_pred_mean": jnp.mean(value_pred),
            "value_target_mean": jnp.mean(value_target),
            "value_entropy": jnp.mean(pred_entropy),
        }
        return loss, aux_dict

    def sample_value_logits(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ) -> at.Float[at.Array, "b k"]:
        """Deterministic logits over value bins."""
        del rng
        obs = _model.preprocess_observation(None, observation, train=False)
        return self._predict_value_logits_from_processed_obs(obs)

    def sample_values(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ) -> at.Float[at.Array, "b 1"]:
        """Deterministic scalar E[V] from the distributional critic."""
        logits = self.sample_value_logits(rng, observation)
        return self._expected_value_from_logits(logits)

    def compute_advantage(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "b"], at.Float[at.Array, "b"], at.Float[at.Array, "b"]]:
        """Compute A = R_norm - E[V] for offline relabeling / analysis."""
        del rng, train

        obs = _model.preprocess_observation(None, observation, train=False)

        R_norm = self._paper_return_target_scalar(obs)                   # [B]
        value_logits = self._predict_value_logits_from_processed_obs(obs)
        value_pred = self._expected_value_from_logits(value_logits)[:, 0]  # [B]
        value_pred = jax.lax.stop_gradient(value_pred)

        A_norm = R_norm - value_pred
        return A_norm, R_norm, value_pred