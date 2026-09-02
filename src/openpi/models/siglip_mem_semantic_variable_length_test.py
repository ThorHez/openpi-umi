import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip_mem_semantic


def _model():
    return siglip_mem_semantic.RecurrentMemoryUpdater(
        width=8,
        depth=1,
        num_heads=2,
        dtype_mm="float32",
    )


def test_padded_steps_strictly_carry_previous_memory():
    model = _model()
    memory = jax.random.normal(jax.random.key(0), (2, 4, 8))
    evidence = jax.random.normal(jax.random.key(1), (2, 3, 5, 8))
    params = model.init(jax.random.key(2), memory, evidence)
    reference, _ = model.apply(params, memory, evidence)

    padding = jax.random.normal(jax.random.key(3), (2, 2, 5, 8))
    padded = jnp.concatenate((evidence, padding), axis=1)
    step_mask = jnp.array([[True, True, True, False, False]] * 2)
    actual, states = model.apply(params, memory, padded, step_mask=step_mask)

    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(states[:, 2], states[:, 3], rtol=0, atol=0)
    np.testing.assert_allclose(states[:, 3], states[:, 4], rtol=0, atol=0)


def test_full_unroll_matches_cross_call_stateful_updates():
    model = _model()
    memory = jax.random.normal(jax.random.key(10), (2, 4, 8))
    evidence = jax.random.normal(jax.random.key(11), (2, 4, 5, 8))
    params = model.init(jax.random.key(12), memory, evidence)
    full_final, full_states = model.apply(params, memory, evidence)

    stateful_memory = memory
    stateful_states = []
    for step in range(evidence.shape[1]):
        stateful_memory, state = model.apply(
            params,
            stateful_memory,
            evidence[:, step : step + 1],
        )
        stateful_states.append(state[:, 0])
    stateful_states = jnp.stack(stateful_states, axis=1)

    np.testing.assert_allclose(stateful_memory, full_final, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(stateful_states, full_states, rtol=1e-5, atol=1e-5)

