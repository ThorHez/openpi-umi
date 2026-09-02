import jax
import jax.numpy as jnp

from openpi.models import pi0_mem_semantic_event_action


def test_event_memory_action_interface_shapes_and_no_event_state():
    module = pi0_mem_semantic_event_action.EventDrivenMemoryActionInterface(
        memory_width=8,
        memory_tokens=6,
        memory_depth=1,
        memory_heads=2,
        query_width=16,
        query_tokens=3,
        query_depth=1,
        query_heads=2,
        action_width=16,
        action_heads=2,
        dtype_mm="float32",
    )
    action_tokens = jax.random.normal(jax.random.key(0), (2, 5, 16))
    memory = jax.random.normal(jax.random.key(1), (2, 6, 8))
    evidence = jax.random.normal(jax.random.key(2), (2, 4, 3, 8))
    event_logits = -jnp.ones((2, 4), dtype=jnp.float32)
    variables = module.init(
        jax.random.key(3),
        action_tokens,
        memory,
        evidence,
        event_logits,
    )
    outputs = module.apply(
        variables,
        action_tokens,
        memory,
        evidence,
        event_logits,
    )
    assert outputs["conditioned_action_tokens"].shape == (2, 5, 16)
    assert outputs["resampled_memory_tokens"].shape == (2, 3, 16)
    assert outputs["memory"].shape == (2, 6, 8)
    assert jnp.allclose(outputs["memory"], memory)
    assert jnp.array_equal(outputs["trigger_count"], jnp.zeros((2,), dtype=jnp.int32))
