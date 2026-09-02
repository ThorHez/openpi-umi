import jax.numpy as jnp
import numpy as np
import pytest

from examples.shellgame import v10_exact_semantic_onpolicy_diffusion_distillation as distill


def test_inference_diffusion_times_match_four_step_sampler():
    np.testing.assert_allclose(
        distill.inference_diffusion_times(4),
        jnp.asarray([1.0, 0.75, 0.5, 0.25]),
        atol=1e-7,
    )


def test_inference_diffusion_times_reject_invalid_count():
    with pytest.raises(ValueError, match="positive"):
        distill.inference_diffusion_times(0)
