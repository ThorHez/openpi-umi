import numpy as np
import pytest

from examples.umi.sampler_v2 import SequenceSampler


def _make_replay_buffer(gripper_ndims: tuple[int, ...], *, explicit_action: bool) -> dict[str, np.ndarray]:
    steps = np.arange(12, dtype=np.float32)
    widths = np.array([0.12, 0.11, 0.06, 0.10, 0.09, 0.13, 0.14, 0.12, 0.06, 0.11, 0.13, 0.12], dtype=np.float32)
    replay_buffer = {}
    actions = []
    for robot_id, ndim in enumerate(gripper_ndims):
        position = np.column_stack((steps, steps * 0.1, steps * 0.2)) + robot_id
        rotation = np.column_stack((steps * 0, steps * 0, steps * 0.01))
        gripper = widths - robot_id * 0.015
        replay_buffer[f"robot{robot_id}_eef_pos"] = position
        replay_buffer[f"robot{robot_id}_eef_rot_axis_angle"] = rotation
        replay_buffer[f"robot{robot_id}_gripper_width"] = gripper if ndim == 1 else gripper[:, None]
        actions.extend((position, rotation, gripper[:, None] + 0.2))
    if explicit_action:
        replay_buffer["action"] = np.concatenate(actions, axis=-1)
    return replay_buffer


def _make_sampler(replay_buffer: dict[str, np.ndarray], *, repeat_frame_prob: float = 0.0) -> SequenceSampler:
    robot_ids = [
        int(key.removeprefix("robot").removesuffix("_eef_pos")) for key in replay_buffer if key.endswith("_eef_pos")
    ]
    return SequenceSampler(
        replay_buffer=replay_buffer,
        episode_ends=[6, 12],
        repeat_frame_prob=repeat_frame_prob,
        dataset_config={
            "dataset": {"low_dim_obs_horizon": 2, "action_horizon": 3, "obs_down_sample_steps": 1},
            "robots": [{"id": robot_id, "enabled": True} for robot_id in robot_ids],
            "load_keys": [key for key in replay_buffer if key != "action"],
        },
    )


@pytest.mark.parametrize(
    ("gripper_ndims", "explicit_action"),
    [
        ((1,), False),
        ((2,), False),
        ((1,), True),
        ((2,), True),
        ((1, 1), False),
        ((2, 2), True),
        ((1, 2), False),
        ((2, 1), True),
    ],
)
def test_gripper_shapes_preserve_observations_actions_and_input(gripper_ndims, explicit_action):
    replay_buffer = _make_replay_buffer(gripper_ndims, explicit_action=explicit_action)
    original_arrays = dict(replay_buffer)
    original_values = {key: value.copy() for key, value in replay_buffer.items()}
    if explicit_action:
        expected_actions = original_values["action"]
    else:
        expected_actions = np.concatenate(
            [
                original_values[f"robot{robot_id}_{field}"].reshape(12, -1)
                for robot_id in range(len(gripper_ndims))
                for field in ("eef_pos", "eef_rot_axis_angle", "gripper_width")
            ],
            axis=-1,
        )

    sampler = _make_sampler(replay_buffer)
    assert len(sampler) == 12
    for index in range(len(sampler)):
        start, end = (0, 6) if index < 6 else (6, 12)
        observation_indices = [max(start, index - 1), index]
        action_indices = np.minimum(np.arange(index, index + 3), end - 1)
        sampled = sampler.sample_sequence(index)
        for robot_id in range(len(gripper_ndims)):
            for field in ("eef_pos", "eef_rot_axis_angle", "gripper_width"):
                key = f"robot{robot_id}_{field}"
                expected = original_values[key].reshape(12, -1)[observation_indices]
                assert sampled[key].shape == expected.shape
                assert sampled[key].dtype == np.float32
                np.testing.assert_allclose(sampled[key], expected, atol=1e-7)
        assert sampled["action"].shape == (3, 7 * len(gripper_ndims))
        np.testing.assert_allclose(sampled["action"], expected_actions[action_indices], atol=1e-7)

    assert replay_buffer.keys() == original_arrays.keys()
    for key, original_array in original_arrays.items():
        assert replay_buffer[key] is original_array
        assert replay_buffer[key].shape == original_values[key].shape
        np.testing.assert_array_equal(replay_buffer[key], original_values[key])


@pytest.mark.parametrize("gripper_ndim", [1, 2])
def test_repeat_observations_stops_after_grasp_and_resets_per_episode(gripper_ndim):
    replay_buffer = _make_replay_buffer((gripper_ndim,), explicit_action=False)
    sampler = _make_sampler(replay_buffer, repeat_frame_prob=1.0)
    for index, observation_indices in ((1, [1, 1]), (3, [2, 3]), (7, [7, 7]), (9, [8, 9])):
        sampled = sampler.sample_sequence(index)
        np.testing.assert_allclose(sampled["robot0_eef_pos"], replay_buffer["robot0_eef_pos"][observation_indices])
        expected_widths = replay_buffer["robot0_gripper_width"].reshape(12, 1)[observation_indices]
        np.testing.assert_allclose(sampled["robot0_gripper_width"], expected_widths)


@pytest.mark.parametrize(("robot_id", "shape"), [(0, ()), (0, (12, 2)), (1, (12, 1, 1))])
def test_invalid_gripper_shape_reports_field_and_shape(robot_id, shape):
    replay_buffer = _make_replay_buffer((2, 2), explicit_action=True)
    key = f"robot{robot_id}_gripper_width"
    replay_buffer[key] = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match=key) as error:
        _make_sampler(replay_buffer)
    assert str(shape) in str(error.value)
