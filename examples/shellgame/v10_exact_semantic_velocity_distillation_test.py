import jax.numpy as jnp
import numpy as np

from examples.shellgame import v10_exact_semantic_velocity_distillation as distill


def test_velocity_distillation_is_zero_for_exact_student():
    teacher = jnp.asarray([[[1.0, -2.0, 9.0]]])
    weights = jnp.asarray([[[1.0, 2.0, 0.0]]])
    mse, normalized_mse, cosine = distill.velocity_distillation_terms(
        teacher, teacher, weights
    )
    np.testing.assert_allclose(mse, 0.0, atol=1e-7)
    np.testing.assert_allclose(normalized_mse, 0.0, atol=1e-7)
    np.testing.assert_allclose(cosine, 0.0, atol=1e-7)


def test_velocity_distillation_masks_padding_and_penalizes_wrong_direction():
    teacher = jnp.asarray([[[1.0, 0.0, 100.0]]])
    student = jnp.asarray([[[-1.0, 0.0, -100.0]]])
    weights = jnp.asarray([[[1.0, 1.0, 0.0]]])
    mse, normalized_mse, cosine = distill.velocity_distillation_terms(
        teacher, student, weights
    )
    np.testing.assert_allclose(mse, 2.0, atol=1e-6)
    np.testing.assert_allclose(normalized_mse, 4.0, atol=1e-6)
    np.testing.assert_allclose(cosine, 2.0, atol=1e-6)
