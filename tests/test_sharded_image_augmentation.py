"""Compile the actual image augmentation in a four-device JAX gradient step.

The probe also runs directly with JAX_PLATFORMS=cpu and
XLA_FLAGS=--xla_force_host_platform_device_count=4 in a training environment.
"""

import dataclasses
import os
import pathlib
import subprocess
import sys
from unittest import mock


def check_sharded_augmentation_gradients():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from openpi.models import model
    from openpi.training import sharding

    assert jax.default_backend() == "cpu" and jax.device_count() == 4
    mesh = sharding.make_mesh(4)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    kernel_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, sharding.FSDP_AXIS))

    # Use a tiny trainable layer after the actual augmentation preprocessing.
    views = {"base_0_rgb": 3, "left_wrist_0_rgb": 2, "right_wrist_0_rgb": 2}
    layer = nnx.Linear(3 * len(views), 4, rngs=nnx.Rngs(0))
    graph, params = nnx.split(layer, nnx.Param)
    param_sharding = jax.tree.map(lambda value: kernel_sharding if value.ndim == 2 else replicated, params)
    params = jax.device_put(params, param_sharding)
    image = jnp.asarray(np.random.default_rng(42).uniform(-0.8, 0.8, (4, 24, 24, 3)), dtype=jnp.float32)
    observation = model.Observation(
        images={key: image for key in views},
        image_masks={key: jnp.ones(4, dtype=bool) for key in views},
        state=jnp.zeros((4, 7)),
    )

    def routed(codes):
        selected = dataclasses.replace(
            observation, image_augmentation={key: jnp.full(4, code, dtype=jnp.int32) for key, code in codes.items()}
        )
        return jax.device_put(selected, data_sharding)

    def train_step(key, params, step, observation, fixed_image_augmentation):
        layer = nnx.merge(graph, params)
        step_key = jax.random.fold_in(key, step)

        def loss_fn(layer, key, observation):
            preprocess_key, _, _ = jax.random.split(key, 3)
            processed = model.preprocess_observation(
                preprocess_key,
                observation,
                train=True,
                image_resolution=(24, 24),
                fixed_image_augmentation=fixed_image_augmentation,
            )
            pooled = jnp.concatenate([processed.images[view].mean(axis=(1, 2)) for view in views], axis=-1)
            pooled = sharding.activation_sharding_constraint(pooled)
            return jnp.mean(jnp.square(layer(pooled)))

        return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(layer, step_key, observation)

    train = jax.jit(
        train_step,
        in_shardings=(replicated, param_sharding, replicated, data_sharding),
        out_shardings=(replicated, param_sharding),
        static_argnums=(4,),
    )
    key = jax.device_put(jax.random.key(42), replicated)
    step = jax.device_put(jnp.int32(0), replicated)
    sharded_observation = jax.device_put(observation, data_sharding)
    disabled = dict.fromkeys(views, 1)
    with sharding.set_mesh(mesh):
        default = train(key, params, step, sharded_observation, ())
        explicit = train(key, params, step, routed(views), ())
        disabled_dynamic = train(key, params, step, routed(disabled), ())
        # The startup policy is static even while the training step is traced.
        with mock.patch.object(model, "_augment_rgb_with_policy", side_effect=AssertionError("dynamic route used")):
            fixed = train(key, params, step, sharded_observation, tuple(views.items()))
            disabled_fixed = train(key, params, step, sharded_observation, tuple(disabled.items()))

    # The chosen head and wrist policies are equivalent to the unrouted
    # defaults. Compare loss and gradients to protect their existing behavior.
    assert not np.isclose(disabled_dynamic[0], default[0])
    for reference, result in ((default, explicit), (default, fixed), (disabled_dynamic, disabled_fixed)):
        for expected, actual in zip(jax.tree.leaves(reference), jax.tree.leaves(result), strict=True):
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    print(f"JAX {jax.__version__}: dynamic and fixed augmentation gradients match on {jax.device_count()} CPU devices")


def test_typed_key_augmentation_compiles_with_sharded_gradients():
    # Set the device count before importing JAX in a dedicated process.
    result = subprocess.run(
        [sys.executable, "-B", str(pathlib.Path(__file__).resolve())],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=4"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    check_sharded_augmentation_gradients()
