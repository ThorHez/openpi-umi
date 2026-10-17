import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_mem_pf_safe


def test_safe_defaults_keep_temporal_residuals_conservative():
    config = pi0_mem_pf_safe.Pi0MemPFSafeConfig()

    assert config.memory_every == 27
    assert config.future_latent_tokens == 32
    assert config.history_gate_fixed is None
    assert config.future_gate_fixed is None
    assert np.isclose(float(jax.nn.sigmoid(config.history_gate_init)), 1e-3, rtol=0.02)
    assert np.isclose(float(jax.nn.sigmoid(config.future_gate_init)), 1e-3, rtol=0.02)


def test_safe_config_rejects_independent_alignment_projectors():
    with pytest.raises(ValueError, match="align_proj_dim=None"):
        pi0_mem_pf_safe.Pi0MemPFSafeConfig(align_proj_dim=64)


def test_temporal_only_filter_freezes_backbone_and_action_expert():
    config = pi0_mem_pf_safe.Pi0MemPFSafeConfig()
    is_frozen = nnx.filterlib.to_predicate(config.get_freeze_filter_temporal_only())
    leaf = object()

    assert is_frozen(("PaliGemma", "llm", "layers", "0", "kernel"), leaf)
    assert is_frozen(("action_in_proj", "kernel"), leaf)
    assert not is_frozen(("PaliGemma", "img", "Transformer", "UTR_0", "future_queries"), leaf)
    assert not is_frozen(("PaliGemma", "img", "FutureMultiHeadDotProductAttention_0", "kernel"), leaf)
    assert not is_frozen(("FuturePrior", "prior_queries"), leaf)


def test_cosine_alignment_stops_posterior_target_gradient():
    z_prior = jnp.asarray(
        [
            [[1.0, 2.0, -1.0], [-2.0, 1.0, 0.5]],
            [[0.5, -1.0, 2.0], [2.0, 0.0, -0.5]],
        ]
    )
    z_post = z_prior * 3.0

    align, _ = pi0_mem_pf_safe.cosine_alignment_and_variance_loss(
        z_prior,
        z_post,
        variance_target=0.5,
    )
    post_grad = jax.grad(
        lambda target: pi0_mem_pf_safe.cosine_alignment_and_variance_loss(
            z_prior,
            target,
            variance_target=0.5,
        )[0]
    )(z_post)

    assert np.isclose(float(align), 0.0, atol=1e-6)
    np.testing.assert_array_equal(post_grad, jnp.zeros_like(post_grad))


def test_variance_regularizer_penalizes_collapse_not_large_norm():
    collapsed = jnp.ones((2, 4, 3)) * 10.0
    varied = jnp.asarray(
        [
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]],
            [[-1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, -1.0]],
        ]
    )

    _, collapsed_reg = pi0_mem_pf_safe.cosine_alignment_and_variance_loss(
        collapsed,
        collapsed,
        variance_target=0.5,
    )
    _, varied_reg = pi0_mem_pf_safe.cosine_alignment_and_variance_loss(
        varied * 10.0,
        varied * 10.0,
        variance_target=0.5,
    )

    assert float(collapsed_reg) > 0.2
    assert np.isclose(float(varied_reg), 0.0, atol=1e-6)
