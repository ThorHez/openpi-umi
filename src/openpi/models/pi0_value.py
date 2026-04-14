"""Pi0 value model: value-only head on top of Pi0 backbone (pi0.py only, no Pi0Advantage).

Architecture follows Pistar06Model: pooled prefix from Pi0 -> LayerNorm (final_norm)
-> value_head (Linear -> GELU -> Dropout -> Linear) -> logits over num_bins.
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
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi0ValueConfig(pi0_config.Pi0Config):
    """Config for Pi0Value: Pi0 backbone + value head. No action branch training."""

    num_value_bins: int = 201
    value_min: float = -1.0
    value_max: float = 0.0
    value_head_dropout: float = 0.1
    soft_value_targets: bool = True

    def get_freeze_filter_value_head_only(self) -> nnx.filterlib.Filter:
        """Freeze Pi0 backbone; only value branch (final_norm, value_fc1, value_fc2, value_dropout) trainable."""
        value_branch_regex = nnx_utils.PathRegex(
            ".*(final_norm|value_fc1|value_fc2|value_dropout).*"
        )
        return nnx.Not(value_branch_regex)

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Value":
        return Pi0Value(self, rngs=nnx.Rngs(rng))


class Pi0Value(Pi0):
    """Pi0 backbone + distributional value head only (no action branch training).

    Based solely on pi0.py (Pi0). Encodes observation via prefix + dummy suffix,
    pools prefix tokens, then final_norm -> value_head -> logits over value bins.
    """

    def __init__(self, config: Pi0ValueConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.num_value_bins = config.num_value_bins
        self.value_min = config.value_min
        self.value_max = config.value_max
        self.value_head_dropout = config.value_head_dropout
        self.soft_value_targets = config.soft_value_targets

        # Pooled prefix tokens are outputs of the first LLM (paligemma / VLM), not the action expert.
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        prefix_width = paligemma_config.width

        self.final_norm = nnx.LayerNorm(prefix_width, rngs=rngs)
        self.value_fc1 = nnx.Linear(prefix_width, prefix_width, rngs=rngs)
        self.value_dropout = nnx.Dropout(rate=config.value_head_dropout, rngs=rngs)
        self.value_fc2 = nnx.Linear(prefix_width, config.num_value_bins, rngs=rngs)

    # -------------------------------------------------------------------------
    # Pooled prefix encoding (same idea as Pi0 forward, observation-only)
    # -------------------------------------------------------------------------

    def _encode_value_features(self, obs: _model.Observation) -> jnp.ndarray:
        """Run prefix only through LLM (no action/suffix) and return masked mean-pooled prefix."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        prefix_out = prefix_out.astype(jnp.float32)
        prefix_mask_f = prefix_mask.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(prefix_mask_f, axis=1, keepdims=True), 1.0)
        pooled = jnp.sum(prefix_out * prefix_mask_f[..., None], axis=1) / denom
        return pooled  # [B, paligemma width]

    def _value_head_logits(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pistar06-style: Linear -> GELU -> Dropout -> Linear -> logits."""
        x = self.value_fc1(x)
        x = jax.nn.gelu(x)
        x = self.value_dropout(x, deterministic=not self.deterministic)
        x = self.value_fc2(x)
        return x  # [B, num_value_bins]

    def _value_targets_to_probs(self, values: jnp.ndarray) -> jnp.ndarray:
        """Scalar targets -> two-hot bin probs. Matches project_values_to_bins (uniform bin_centers).

        values: [B], in [value_min, value_max].
        Returns: [B, num_value_bins], two-hot with weights summing to 1.
        """
        v = jnp.clip(values, self.value_min, self.value_max)
        # Uniform bins: step = (max - min) / (num_bins - 1), scaled = (v - min) / step
        step = (self.value_max - self.value_min) / jnp.maximum(
            float(self.num_value_bins - 1), 1.0
        )
        scaled = (v - self.value_min) / (step + 1e-8)

        if not self.soft_value_targets:
            idx = jnp.rint(scaled).astype(jnp.int32)
            idx = jnp.clip(idx, 0, self.num_value_bins - 1)
            return jax.nn.one_hot(idx, self.num_value_bins, dtype=jnp.float32)

        lo = jnp.floor(scaled).astype(jnp.int32)
        lo = jnp.clip(lo, 0, self.num_value_bins - 1)
        hi = jnp.clip(lo + 1, 0, self.num_value_bins - 1)
        hi_w = jnp.clip(scaled - lo.astype(jnp.float32), 0.0, 1.0)
        lo_w = 1.0 - hi_w
        lo_oh = jax.nn.one_hot(lo, self.num_value_bins, dtype=jnp.float32)
        hi_oh = jax.nn.one_hot(hi, self.num_value_bins, dtype=jnp.float32)
        return lo_w[:, None] * lo_oh + hi_w[:, None] * hi_oh

    def _distribution_ce_loss(
        self, logits: jnp.ndarray, target_probs: jnp.ndarray
    ) -> jnp.ndarray:
        """Cross-entropy between target probs and predicted logits. Shape [B, 1]."""
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(target_probs * log_probs, axis=-1, keepdims=True)
        return loss

    # -------------------------------------------------------------------------
    # Public API for value training
    # -------------------------------------------------------------------------

    def forward_value_logits(self, obs: _model.Observation) -> jnp.ndarray:
        """Observation -> pooled prefix -> final_norm -> value_head -> logits [B, num_bins]."""
        obs = _model.preprocess_observation(None, obs, train=False)
        feat = self._encode_value_features(obs)
        feat = self.final_norm(feat)
        return self._value_head_logits(feat)

    def compute_value_loss_from_targets(
        self,
        obs: _model.Observation,
        value_targets_scalar: at.Float[at.Array, " b"],
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "b 1"]:
        """Compute distributional CE loss from precomputed scalar value targets.

        value_targets_scalar: shape [B], in [value_min, value_max] (e.g. from
        compute_normalized_value_targets).
        """
        obs = _model.preprocess_observation(rng, obs, train=train)
        feat = self._encode_value_features(obs)
        feat = self.final_norm(feat)
        logits = self._value_head_logits(feat)
        target_probs = self._value_targets_to_probs(value_targets_scalar)
        return self._distribution_ce_loss(logits, target_probs)

    def expected_value_from_logits(self, logits: jnp.ndarray) -> jnp.ndarray:
        """E[V] from categorical logits. Returns shape [B, 1]."""
        centers = jnp.linspace(
            float(self.value_min),
            float(self.value_max),
            int(self.num_value_bins),
            dtype=jnp.float32,
        )
        probs = jax.nn.softmax(logits, axis=-1)
        value = jnp.sum(probs * centers[None, :], axis=-1, keepdims=True)
        return value
