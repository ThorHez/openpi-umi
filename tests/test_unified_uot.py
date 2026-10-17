"""CPU checks for the optional UOT objective and its policy feature boundary."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model
from openpi.models.pi0 import Pi0
from openpi.training.uot import compute_uot_alignment_loss
from openpi.training.uot import unbalanced_sinkhorn_plan


def loss(features, labels):
    return compute_uot_alignment_loss(
        features, jnp.array([[0.2, 0.4], [1.2, 1.4]]), labels, jnp.array([[0.0], [1.0]]),
        embedding_cost_scale=1.0, label_cost_scale=1.0, entropy_reg=0.5,
        marginal_relaxation=1.0, sinkhorn_iterations=50,
    )[0]


def test_uot_has_finite_visual_gradient_and_stops_label_gradient():
    features = jnp.array([[0.0, 0.1], [1.0, 1.1]])
    labels = jnp.array([[0.0], [1.0]])
    value, gradient = jax.jit(jax.value_and_grad(loss))(features, labels)
    assert np.isfinite(value) and value > 0
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0
    np.testing.assert_array_equal(jax.grad(loss, argnums=1)(features, labels), 0)


def test_sinkhorn_prefers_low_cost_correspondences():
    plan = unbalanced_sinkhorn_plan(
        jnp.array([[0.0, 4.0], [4.0, 0.0]]),
        entropy_reg=0.5, marginal_relaxation=1.0, iterations=50,
    )
    assert np.isfinite(plan).all()
    assert plan[0, 0] > plan[0, 1] and plan[1, 1] > plan[1, 0]
    with pytest.raises(ValueError, match="must be positive"):
        unbalanced_sinkhorn_plan(jnp.ones((2, 2)), entropy_reg=0, marginal_relaxation=1, iterations=2)


def test_uot_features_use_every_valid_camera_without_loading_weights():
    observation = model.Observation(
        images={"left_wrist_0_rgb": jnp.ones((2, 8, 8, 3)), "right_wrist_0_rgb": jnp.zeros((2, 8, 8, 3))},
        image_masks={"left_wrist_0_rgb": jnp.array([True, True]), "right_wrist_0_rgb": jnp.array([True, False])},
        state=jnp.zeros((2, 4)),
    )
    fake = SimpleNamespace(
        depth_keys=(),
        preprocess_observation=lambda rng, obs, train: model.preprocess_observation(
            rng, obs, train=train, image_resolution=(8, 8)
        ),
        PaliGemma=SimpleNamespace(img=lambda images, train: (images.reshape(2, -1, 3), None)),
    )
    features = Pi0.encode_visual_features(fake, None, observation, train=False)
    np.testing.assert_allclose(features, [[0.5] * 3, [1.0] * 3])


def test_uot_does_not_claim_mem_video_support():
    observation = model.Observation(
        images={"wrist_rgb": jnp.ones((2, 3, 8, 8, 3))},
        image_masks={"wrist_rgb": jnp.ones((2,), dtype=bool)}, state=jnp.zeros((2, 4)),
    )
    fake = SimpleNamespace(depth_keys=(), preprocess_observation=lambda rng, obs, train: obs)
    with pytest.raises(ValueError, match="MEM video UOT is not implemented"):
        Pi0.encode_visual_features(fake, None, observation, train=False)


def test_uot_depth_uses_the_same_contract_as_preprocessing():
    observation = model.Observation(
        images={"camera": jnp.ones((2, 8, 8, 3)), "camera_depth": jnp.full((2, 8, 8, 3), 99.0)},
        image_masks={"camera": jnp.array([True, True]), "camera_depth": jnp.array([True, True])},
        state=jnp.zeros((2, 4)),
    )
    fake = SimpleNamespace(
        depth_keys=None,
        preprocess_observation=lambda rng, obs, train: model.preprocess_observation(
            rng, obs, train=train, image_resolution=(8, 8)
        ),
        PaliGemma=SimpleNamespace(img=lambda images, train: (images.reshape(2, -1, 3), None)),
    )
    np.testing.assert_allclose(Pi0.encode_visual_features(fake, None, observation, train=False), 1)
    observation.image_masks["camera"] = jnp.array([True, False])
    with pytest.raises(ValueError, match="no valid image view"):
        Pi0.encode_visual_features(fake, None, observation, train=False)
