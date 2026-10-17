import jax.numpy as jnp
import numpy as np

from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_residual_action as residual


def test_waypoint_residual_round_trip_and_clip():
    waypoint = jnp.asarray([[0.2, -0.4]], dtype=jnp.float32)
    actions = jnp.zeros((1, 2, 7), dtype=jnp.float32)
    actions = actions.at[:, :, 0].set(jnp.asarray([[0.3, 0.7]]))
    actions = actions.at[:, :, 1].set(jnp.asarray([[-0.5, -0.2]]))
    delta = residual.absolute_to_waypoint_residual(actions, waypoint, (0, 1))
    restored = residual.waypoint_residual_to_absolute(
        delta, waypoint, (0, 1), (0.2, 0.15)
    )
    np.testing.assert_allclose(restored[0, 0, :2], actions[0, 0, :2], atol=1e-6)
    np.testing.assert_allclose(restored[0, 1, :2], [0.4, -0.25], atol=1e-6)
    np.testing.assert_allclose(restored[:, :, 2:], actions[:, :, 2:], atol=1e-6)
