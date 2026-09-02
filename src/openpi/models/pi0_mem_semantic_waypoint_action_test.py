import flax.traverse_util
import jax
import jax.numpy as jnp

from openpi.models import pi0_mem_semantic_waypoint_action as waypoint_action


def test_waypoint_conditioner_shapes_and_zero_init_compatibility():
    module = waypoint_action.SemanticMemoryWaypointConditioner(
        memory_tokens=4,
        memory_width=8,
        query_tokens=2,
        hidden_width=16,
        action_width=32,
        num_heads=2,
        waypoint_dim=3,
        dtype_mm="float32",
    )
    action = jnp.ones((2, 5, 32), dtype=jnp.float32)
    memory = jax.random.normal(jax.random.key(1), (2, 4, 8))
    variables = module.init(jax.random.key(0), action, memory)
    conditioned, waypoint = module.apply(variables, action, memory)

    assert conditioned.shape == action.shape
    assert waypoint.shape == (2, 3)
    assert jnp.all(jnp.isfinite(conditioned))
    assert jnp.all(jnp.isfinite(waypoint))
    flat = flax.traverse_util.flatten_dict(variables["params"])
    waypoint_kernel = next(value for key, value in flat.items() if key[-2:] == ("waypoint_embed", "kernel"))
    assert jnp.all(waypoint_kernel == 0)
