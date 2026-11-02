"""Prove UMI frame rebasing and timestamp alignment without a robot runtime."""

import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi_client import umi_pose


def world_plan():
    return np.array(
        [
            [0.3, -0.2, 0.5, 0.2, -0.1, 0.4, 0.02],
            [0.5, 0.1, 0.3, -0.3, 0.5, 0.2, 0.07],
            [0.1, 0.4, 0.6, 0.7, 0.2, -0.4, 0.04],
        ]
    )


def assert_world_equal(actual, expected):
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_rotvec(actual[:, 3:6]).as_matrix(),
        Rotation.from_rotvec(expected[:, 3:6]).as_matrix(),
        atol=1e-12,
    )
    np.testing.assert_array_equal(actual[:, 6], expected[:, 6])


@pytest.mark.parametrize("pose_repr", ["abs", "rel", "relative"])
@pytest.mark.parametrize("base", [np.zeros(6), np.array([0.2, -0.1, 0.4, 0.3, 0.5, -0.2])])
def test_world_round_trip_and_row_rotation_convention(pose_repr, base):
    world = world_plan()
    actions = umi_pose.world_actions_to_model_actions(world, base, pose_repr=pose_repr)
    recovered = umi_pose.model_actions_to_world_actions(actions, base, pose_repr=pose_repr)
    assert_world_equal(recovered, world)
    assert actions.shape == (3, 10)
    np.testing.assert_array_equal(actions[:, 9], world[:, 6])


@pytest.mark.parametrize("pose_repr", ["abs", "rel", "relative"])
def test_rebase_remaining_prefix_preserves_world_target(pose_repr):
    old_base = np.array([0.2, 0.3, 0.1, -0.4, 0.2, 0.5])
    new_base = np.array([-0.3, 0.1, 0.4, 0.2, -0.5, 0.7])
    world = world_plan()
    old_actions = umi_pose.world_actions_to_model_actions(world, old_base, pose_repr=pose_repr)
    absolute = umi_pose.model_actions_to_world_actions(old_actions, old_base, pose_repr=pose_repr)
    rebased = umi_pose.world_actions_to_model_actions(absolute[1:], new_base, pose_repr=pose_repr)
    recovered = umi_pose.model_actions_to_world_actions(rebased, new_base, pose_repr=pose_repr)
    assert_world_equal(recovered, world[1:])
    if pose_repr != "abs":
        assert not np.allclose(old_actions[1:, :9], rebased[:, :9])


def test_relative_and_legacy_rel_are_distinct_under_rotated_reference():
    base = np.array([0.2, 0.1, 0.0, 0, 0, np.pi / 2])
    world = np.array([[1.2, 0.1, 0.0, 0, 0, 0, 0.04]])
    relative = umi_pose.world_actions_to_model_actions(world, base, pose_repr="relative")
    legacy = umi_pose.world_actions_to_model_actions(world, base, pose_repr="rel")
    np.testing.assert_allclose(relative[0, :3], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(legacy[0, :3], [1, 0, 0], atol=1e-12)


@pytest.mark.parametrize("pose_repr", ["delta", "unknown"])
def test_unconfirmed_pose_conventions_are_rejected(pose_repr):
    with pytest.raises(ValueError, match="pose_repr"):
        umi_pose.world_actions_to_model_actions(world_plan(), np.zeros(6), pose_repr=pose_repr)


def test_degenerate_predicted_rotation_is_rejected():
    with pytest.raises(ValueError, match="rotation rows"):
        umi_pose.model_actions_to_world_actions(np.zeros((1, 10)), np.zeros(6), pose_repr="relative")


def test_world_interpolation_uses_rotation_slerp_and_linear_gripper():
    plan = np.array([[0, 0, 0, 0, 0, 0, 0.02], [2, 4, 6, 0, 0, np.pi, 0.08]])
    result = umi_pose.interpolate_world_actions(plan, [10.0, 12.0], [10.0, 11.0, 12.0])
    np.testing.assert_allclose(result[:, :3], [[0, 0, 0], [1, 2, 3], [2, 4, 6]])
    np.testing.assert_allclose(result[:, 6], [0.02, 0.05, 0.08])
    np.testing.assert_allclose(result[1, 3:6], [0, 0, np.pi / 2], atol=1e-12)


def test_world_interpolation_takes_short_rotation_path_across_rotvec_wrap():
    plan = np.array([[0, 0, 0, 0, 0, np.deg2rad(179), 0.02], [0, 0, 0, 0, 0, np.deg2rad(-179), 0.02]])
    result = umi_pose.interpolate_world_actions(plan, [0, 2], [1])
    np.testing.assert_allclose(Rotation.from_rotvec(result[0, 3:6]).as_matrix(), np.diag([-1, -1, 1]), atol=1e-12)


@pytest.mark.parametrize("targets", [[9.9], [12.1]])
def test_interpolation_never_extrapolates(targets):
    with pytest.raises(ValueError, match="extrapolation"):
        umi_pose.interpolate_world_actions(world_plan()[:2], [10, 12], targets)


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0, np.nan]])
def test_interpolation_requires_a_valid_time_grid(times):
    with pytest.raises(ValueError, match="timestamps"):
        umi_pose.interpolate_world_actions(world_plan()[:2], times, [0])


def test_single_remaining_waypoint_only_supports_its_own_time():
    np.testing.assert_array_equal(umi_pose.interpolate_world_actions(world_plan()[:1], [4], [4]), world_plan()[:1])


def test_client_geometry_module_does_not_import_model_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from openpi_client import umi_pose; import sys; assert 'jax' not in sys.modules; "
                "assert 'openpi' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
