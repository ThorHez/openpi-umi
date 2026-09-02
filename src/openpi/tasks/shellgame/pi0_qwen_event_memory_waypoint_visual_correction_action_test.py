import jax.numpy as jnp
import numpy as np

from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_visual_correction_action as correction


def test_target_uses_only_first_action_and_clips():
    waypoint = jnp.asarray([[0.2, -0.4]], dtype=jnp.float32)
    actions = jnp.zeros((1, 3, 7), dtype=jnp.float32)
    actions = actions.at[0, 0, 0].set(0.25)
    actions = actions.at[0, 0, 1].set(-0.8)
    # Later trajectory points must have no effect on the correction label.
    actions = actions.at[0, 1:, :2].set(9.0)
    target = correction.first_action_correction_target(
        actions, waypoint, (0, 1), (0.1, 0.2)
    )
    np.testing.assert_allclose(target, [[0.05, -0.2]], atol=1e-6)


def test_apply_correction_overwrites_only_xy_for_whole_chunk():
    waypoint = jnp.asarray([[0.2, -0.4]], dtype=jnp.float32)
    delta = jnp.asarray([[0.05, -0.3]], dtype=jnp.float32)
    actions = jnp.arange(21, dtype=jnp.float32).reshape(1, 3, 7)
    result = correction.apply_waypoint_correction(
        actions, waypoint, delta, (0, 1), (0.1, 0.2)
    )
    np.testing.assert_allclose(result[0, :, 0], 0.25, atol=1e-6)
    np.testing.assert_allclose(result[0, :, 1], -0.6, atol=1e-6)
    np.testing.assert_allclose(result[0, :, 2:], actions[0, :, 2:], atol=1e-6)
