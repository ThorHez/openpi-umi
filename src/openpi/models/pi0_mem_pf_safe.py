"""Conservative Pi0.5 Past-Future temporal-bottleneck variant.

This module intentionally leaves :mod:`openpi.models.pi0_mem_pf` unchanged.
It reuses the original PF architecture and checkpoint parameter names while
changing only the defaults and training contract that are important for
preserving the pretrained Pi0.5 policy:

* temporal branches are learned through small, non-fixed residual gates;
* temporal injection starts at the final SigLIP block by default;
* only PF-specific parameters are trainable during the first adaptation stage;
* latent alignment is scale-invariant and the regularizer prevents feature
  collapse instead of minimizing the latent norm.

The class is deliberately a strict subclass of ``Pi0MemPF`` so the existing
PF-aware checkpoint loader and inference path remain compatible.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import pi0_mem_pf
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

# These are the parameters introduced by the PF architecture. Everything else
# belongs to the pretrained Pi0.5 image/VLM/action path and is frozen by the
# stage-1 filter below.
_TEMPORAL_PARAMETER_PATTERN = (
    r".*(UTR_0|"
    r"HistoryLayerNorm_0|HistoryMultiHeadDotProductAttention_0|HistoryOutProj|"
    r"history_memory_gate_logit|"
    r"FutureLayerNorm_0|FutureMultiHeadDotProductAttention_0|FutureOutProj|"
    r"future_memory_gate_logit|"
    r"FuturePrior|PredictiveState|BeliefFusion|align_proj).*"
)


def cosine_alignment_and_variance_loss(
    z_prior,
    z_post,
    *,
    variance_target: float,
    eps: float = 1e-6,
):
    """Return stop-gradient cosine alignment and a non-collapse loss.

    The original PF implementation regularized ``||z||^2``. Together with an
    alignment loss, that objective admits a zero-valued solution. Here the
    alignment is performed after per-token L2 normalization, and the
    regularizer penalizes feature dimensions whose standard deviation across
    batch and token positions falls below ``variance_target``.
    """
    z_prior = jnp.asarray(z_prior, dtype=jnp.float32)
    z_post = jnp.asarray(z_post, dtype=jnp.float32)

    prior_norm = jnp.maximum(jnp.linalg.norm(z_prior, axis=-1, keepdims=True), eps)
    post_norm = jnp.maximum(jnp.linalg.norm(z_post, axis=-1, keepdims=True), eps)
    prior_unit = z_prior / prior_norm
    post_unit = z_post / post_norm

    align = jnp.mean(
        1.0
        - jnp.sum(
            prior_unit * jax.lax.stop_gradient(post_unit),
            axis=-1,
        )
    )

    def variance_floor(z):
        flat = z.reshape((-1, z.shape[-1]))
        std = jnp.sqrt(jnp.var(flat, axis=0) + eps)
        return jnp.mean(jnp.square(jax.nn.relu(variance_target - std)))

    noncollapse = 0.5 * (variance_floor(z_prior) + variance_floor(z_post))
    return align, noncollapse


@dataclasses.dataclass(frozen=True)
class Pi0MemPFSafeConfig(pi0_mem_pf.Pi0MemPFConfig):
    """First conservative PF configuration for Pi0.5 adaptation.

    ``lambda_reg`` is retained for trainer compatibility, but in this variant
    it weights the feature-variance floor returned as ``reg_loss``; it no
    longer minimizes the latent norm.
    """

    # So400m/14 contains 27 blocks. The safe default injects temporal residuals
    # only into the final block; later experiments may lower this to 9 or 3.
    memory_every: int = 27
    history_memory_tokens: int = 64
    future_latent_tokens: int = 32

    # Never force randomly initialized branches fully open in the safe model.
    history_gate_init: float = -6.9
    history_gate_fixed: float | None = None
    future_gate_init: float = -6.9
    future_gate_fixed: float | None = None

    # Conservative auxiliary-loss defaults. Alignment is ramped by the safe
    # trainer instead of being applied at full strength from step zero.
    lambda_align: float = 0.01
    lambda_reg: float = 1e-3
    align_warmup_steps: int = 2_000
    align_ramp_steps: int = 3_000
    latent_variance_target: float = 0.5

    # Separate projectors plus a stopped posterior target require their own
    # teacher update rule. The first safe version therefore aligns the latents
    # directly and rejects projector-based configurations.
    align_proj_dim: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.align_proj_dim is not None:
            raise ValueError("Pi0MemPFSafeConfig requires align_proj_dim=None")
        if self.memory_every <= 0:
            raise ValueError("memory_every must be positive")
        if self.history_memory_tokens <= 0 or self.future_latent_tokens <= 0:
            raise ValueError("safe PF requires positive history and future token counts")
        if self.align_warmup_steps < 0 or self.align_ramp_steps < 0:
            raise ValueError("alignment warmup/ramp steps must be non-negative")
        if self.latent_variance_target <= 0:
            raise ValueError("latent_variance_target must be positive")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0MemPFSafe":
        return Pi0MemPFSafe(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_temporal_only(self) -> nnx.filterlib.Filter:
        """Freeze the pretrained Pi0.5 path and train only PF parameters.

        ``TrainConfig.freeze_filter`` selects parameters to freeze, hence the
        negation: every parameter not matching the PF-specific pattern is
        frozen. The PF trainer logs the resulting parameter counts at startup.
        """
        temporal_params = nnx_utils.PathRegex(_TEMPORAL_PARAMETER_PATTERN)
        return nnx.Not(temporal_params)


class Pi0MemPFSafe(pi0_mem_pf.Pi0MemPF):
    """Pi0MemPF with scale-invariant alignment and non-collapse regularization."""

    def __init__(self, config: Pi0MemPFSafeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.latent_variance_target = config.latent_variance_target

    @override
    def _align_and_reg_losses(self, z_prior, z_post):
        if self.align_proj_dim is not None:
            raise ValueError("safe PF aligns Zprior and Zpost directly")
        return cosine_alignment_and_variance_loss(
            z_prior,
            z_post,
            variance_target=self.latent_variance_target,
        )
