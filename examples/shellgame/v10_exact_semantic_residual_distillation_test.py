import jax.numpy as jnp
import numpy as np

from examples.shellgame import v10_exact_semantic_residual_distillation as distill


def test_residual_distillation_is_zero_for_exact_student():
    raw = jnp.zeros((2, 3, 4))
    teacher = jnp.ones((2, 3, 4))
    mse, cosine = distill.residual_distillation_terms(raw, teacher, teacher)
    np.testing.assert_allclose(mse, 0.0, atol=1e-7)
    np.testing.assert_allclose(cosine, 0.0, atol=1e-7)


def test_residual_distillation_penalizes_wrong_direction():
    raw = jnp.zeros((1, 1, 2))
    teacher = jnp.asarray([[[1.0, 0.0]]])
    student = jnp.asarray([[[-1.0, 0.0]]])
    mse, cosine = distill.residual_distillation_terms(raw, teacher, student)
    np.testing.assert_allclose(mse, 4.0, atol=1e-6)
    np.testing.assert_allclose(cosine, 2.0, atol=1e-6)
