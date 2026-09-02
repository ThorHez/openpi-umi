import numpy as np

from openpi.tasks.robomme.pickxtimes import eef_action_adapter
from openpi.training.mem import robomme_pickxtimes_action_dataset as action_data


def test_build_robot_goal_layout():
    result = action_data.build_robot_goal(
        np.arange(6),
        np.arange(2),
        np.arange(7),
        target_color="green",
        required_count=4,
    )
    assert result.shape == (eef_action_adapter.ROBOT_GOAL_DIM,)
    assert np.array_equal(result[15:18], [0.0, 1.0, 0.0])
    assert np.isclose(result[-1], 0.8)


def test_latest_memory_offsets_are_causal():
    visible = np.asarray([10, 20, 30])
    timesteps = np.asarray([0, 9, 10, 19, 20, 31])
    assert np.array_equal(action_data.latest_memory_offsets(visible, timesteps), [0, 0, 1, 1, 2, 3])


def test_phase_at_timestep():
    events = [
        {"start": 0, "end": 4, "event_type_id": 0},
        {"start": 4, "end": 8, "event_type_id": 1},
        {"start": 8, "end": 10, "event_type_id": 2},
    ]
    assert [action_data.phase_at_timestep(events, value) for value in (0, 4, 9)] == [0, 1, 2]
