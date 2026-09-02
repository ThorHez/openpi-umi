import jax.numpy as jnp
import numpy as np

from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_bounded_action as bounded


def test_apply_bounded_absolute_xy_preserves_non_xy_and_clips_about_waypoint():
    actions = jnp.zeros((1, 3, 7), dtype=jnp.float32)
    actions = actions.at[0, :, 0].set(jnp.asarray([-2.0, 0.8, 3.0]))
    actions = actions.at[0, :, 1].set(jnp.asarray([-3.0, 0.1, 4.0]))
    actions = actions.at[0, :, 2].set(jnp.asarray([0.2, 0.3, 0.4]))
    waypoint = jnp.asarray([[1.0, -1.0]], dtype=jnp.float32)

    actual = bounded.apply_bounded_absolute_xy(
        actions, waypoint, (0, 1), (0.5, 0.25)
    )

    np.testing.assert_allclose(actual[0, :, 0], [0.5, 0.8, 1.5])
    np.testing.assert_allclose(actual[0, :, 1], [-1.25, -0.75, -0.75])
    np.testing.assert_allclose(actual[0, :, 2], [0.2, 0.3, 0.4])
