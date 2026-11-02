"""Verify the elec_meta inference adapter against the existing training path."""

import numpy as np
from openpi_client import umi_pose
import pytest

from openpi.policies import elec_meta_policy
from openpi.policies import umi_policy
from openpi.utils import pose_repr_utils
from openpi.utils import pose_utils


def prepared_observation():
    return {
        "robot0_eef_pos": np.arange(6, dtype=np.float32).reshape(2, 3) / 10,
        "robot0_eef_rot_axis_angle": np.tile([1, 0, 0, 0, 1, 0], (2, 1)),
        "robot0_eef_rot_axis_angle_wrt_start": np.tile([0, -1, 0, 1, 0, 0], (2, 1)),
        "robot0_gripper_width": np.array([[0.02], [0.07]], dtype=np.float32),
        "camera0_rgb": np.stack(
            [
                np.zeros((3, 224, 224), dtype=np.float32),
                np.full((3, 224, 224), 0.5, dtype=np.float32),
            ]
        ),
        "prompt": "sort electronic components",
    }


def test_inference_inputs_match_umi_v4_training_state_and_image():
    source = prepared_observation()
    actual = elec_meta_policy.ElecMetaInputs()(source)
    training_sample = {
        **source,
        "left_wrist_0_rgb_1": source["camera0_rgb"][-1],
        "actions": np.zeros((16, 10), dtype=np.float32),
    }
    expected = umi_policy.UmiInputsV4()(training_sample)
    np.testing.assert_array_equal(actual["state"], expected["state"])
    assert actual["state"].shape == (2, 16)
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        np.testing.assert_array_equal(actual["image"][key], expected["image"][key])
        assert actual["image_mask"][key] == expected["image_mask"][key]
    assert not actual["image_mask"]["base_0_rgb"]
    assert actual["image_mask"]["left_wrist_0_rgb"]
    assert not actual["image_mask"]["right_wrist_0_rgb"]
    np.testing.assert_array_equal(actual["actions"], 0)
    assert "actions" not in source
    assert "left_wrist_0_rgb_1" not in source
    np.testing.assert_array_equal(source["camera0_rgb"][-1], 0.5)


def test_explicit_previous_actions_are_preserved_for_rtc_normalization():
    source = prepared_observation()
    actions = np.arange(160, dtype=np.float32).reshape(16, 10)
    result = elec_meta_policy.ElecMetaInputs()({**source, "actions": actions})
    np.testing.assert_array_equal(result["actions"], actions)
    result["actions"][0, 0] = -100
    assert actions[0, 0] == 0


def test_prepared_wrist_alias_and_uint8_images_are_supported():
    source = prepared_observation()
    source["left_wrist_0_rgb"] = (source.pop("camera0_rgb") * 255).astype(np.uint8)
    result = elec_meta_policy.ElecMetaInputs()(source)
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], 127)


@pytest.mark.parametrize(
    "replacement",
    [
        {"robot1_eef_pos": np.zeros((2, 3))},
        {"robot0_eef_pos": np.zeros((3,))},
        {"robot0_eef_rot_axis_angle": np.full((2, 6), np.nan)},
        {"robot0_eef_rot_axis_angle_wrt_start": np.zeros((2, 3))},
        {"robot0_gripper_width": np.zeros((2,))},
        {"camera0_rgb": np.zeros((2, 224, 224, 3), dtype=np.uint8)},
        {"camera0_rgb": np.full((2, 3, 224, 224), 1.1)},
        {"actions": np.zeros((16, 32))},
    ],
)
def test_bad_observations_fail_before_model_execution(replacement):
    with pytest.raises(ValueError, match=r"elec_meta|robot0_|Wrist images"):
        elec_meta_policy.ElecMetaInputs()({**prepared_observation(), **replacement})


def test_outputs_crop_before_unnormalization_and_drop_flattened_state():
    model_actions = np.arange(16 * 32).reshape(16, 32)
    result = elec_meta_policy.ElecMetaOutputs()({"actions": model_actions, "state": np.zeros(32)})
    assert set(result) == {"actions"}
    np.testing.assert_array_equal(result["actions"], model_actions[:, :10])
    result["actions"][0, 0] = -1
    assert model_actions[0, 0] == 0


@pytest.mark.parametrize("pose_repr", ["abs", "rel", "relative"])
def test_client_coordinates_match_the_existing_umi_pose_converter(pose_repr):
    base = np.array([0.2, -0.3, 0.4, 0.5, 0.1, -0.2])
    world = np.array(
        [
            [0.3, 0.1, 0.5, 0.2, -0.1, 0.6, 0.02],
            [0.5, 0.4, 0.1, -0.3, 0.5, 0.2, 0.07],
        ]
    )
    expected_pose = pose_repr_utils.convert_pose_mat_rep(
        pose_utils.pose_to_mat(world[:, :6]),
        base_pose_mat=pose_utils.pose_to_mat(base),
        pose_rep=pose_repr,
        backward=False,
    )
    expected = np.concatenate((pose_utils.mat_to_pose10d(expected_pose), world[:, 6:7]), axis=-1)
    actual = umi_pose.world_actions_to_model_actions(world, base, pose_repr=pose_repr)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
