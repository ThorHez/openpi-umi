import jax.numpy as jnp
import numpy as np

from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_scheduled_action as scheduled


def test_scheduled_anchor_strength_releases_linearly():
    actual = scheduled.scheduled_anchor_strength(
        jnp.asarray([59, 91, 99, 107, 130]),
        start_frame=91,
        end_frame=107,
        initial_strength=1.0,
        final_strength=0.2,
    )
    np.testing.assert_allclose(actual, [1.0, 1.0, 0.6, 0.2, 0.2], atol=1e-6)
