from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from openpi.training.config_pi0_mem import UmiInputsV4ShellgameRealWristVideo
from scripts.mem.convert_real_shellgame_stage2_epfirst import build_episode_contract

UMI_ARX_ROOT = Path("/data2/hzl_workspace_for_pi_mem/umi-arx-kian")
sys.path.insert(0, str(UMI_ARX_ROOT))
from utils.shellgame_openpi import ACTION_DIM  # noqa: E402
from utils.shellgame_openpi import ACTION_HORIZON  # noqa: E402
from utils.shellgame_openpi import ShellGameHistory  # noqa: E402
from utils.shellgame_openpi import relative_chunk_to_world  # noqa: E402


def _pose_series(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.zeros((length, 3), dtype=np.float64)
    position[:, 0] = 0.1 * np.arange(length)
    rotation = np.zeros((length, 3), dtype=np.float64)
    gripper = np.linspace(0.1, 0.07, length, dtype=np.float64)
    return position, rotation, gripper


def test_future_targets_share_current_frame_anchor() -> None:
    position, rotation, gripper = _pose_series(20)
    raw_action = np.concatenate((position, rotation, gripper[:, None]), axis=-1)

    contract = build_episode_contract(position, rotation, gripper, raw_action)

    # At current frame 2, every future target is relative to frame 2, matching
    # eval_arx5_pi_hzl.py's one-anchor relative_chunk_to_world decoder.
    np.testing.assert_allclose(contract.actions[2, 0, :3], [0.1, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(contract.actions[2, 4, :3], [0.5, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(contract.state[2, :3], [0.2, 0.0, 0.0], atol=1e-6)
    assert contract.actions.shape == (20, 16, 10)
    assert contract.state.shape == (20, 10)


def test_invalid_or_stale_command_falls_back_to_measured_pose() -> None:
    position, rotation, gripper = _pose_series(20)
    raw_action = np.concatenate((position, rotation, gripper[:, None]), axis=-1)
    raw_action[3, :3] = [3.0, 0.0, 0.0]

    contract = build_episode_contract(
        position,
        rotation,
        gripper,
        raw_action,
        max_command_position_error_m=0.05,
    )

    assert not contract.command_valid[3]
    np.testing.assert_allclose(
        contract.actions[2, 0, :3],
        position[3] - position[2],
        atol=1e-6,
    )
    # Tail padding repeats the final target relative to the same current pose.
    np.testing.assert_allclose(contract.actions[-1, 0], contract.actions[-1, -1])
    assert contract.max_roundtrip_position_error_m < 1e-12
    assert contract.max_roundtrip_rotation_matrix_error < 1e-12


def test_action_horizon_32_uses_the_same_current_frame_anchor() -> None:
    position, rotation, gripper = _pose_series(40)
    raw_action = np.concatenate((position, rotation, gripper[:, None]), axis=-1)

    contract = build_episode_contract(
        position,
        rotation,
        gripper,
        raw_action,
        action_horizon=32,
    )

    assert contract.actions.shape == (40, 32, 10)
    np.testing.assert_allclose(contract.actions[2, 0, :3], [0.1, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(contract.actions[2, 31, :3], [3.2, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(contract.actions[-1, 0], contract.actions[-1, -1])


def test_real_wrist_transform_accepts_its_configured_action_horizon() -> None:
    common = {
        "left_wrist_0_rgb_0_video": np.zeros((2, 224, 224, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_rot_axis_angle": np.zeros(6, dtype=np.float32),
        "robot0_gripper_width": np.zeros(1, dtype=np.float32),
    }
    for horizon in (16, 32):
        transform = UmiInputsV4ShellgameRealWristVideo(num_frames=2, action_horizon=horizon)
        result = transform({**common, "actions": np.zeros((horizon, 10), dtype=np.float32)})
        assert result["actions"].shape == (horizon, 10)


def test_training_contract_roundtrips_through_exact_robot_inference_decoder() -> None:
    """Guard training I/O against eval_arx5_pi_hzl.py contract drift."""
    length = 20
    position = np.stack(
        [
            np.linspace(0.10, 0.13, length),
            np.linspace(-0.04, 0.02, length),
            np.linspace(0.20, 0.26, length),
        ],
        axis=-1,
    )
    rotation = np.stack(
        [
            np.linspace(0.05, 0.15, length),
            np.linspace(-0.10, 0.08, length),
            np.linspace(0.02, -0.04, length),
        ],
        axis=-1,
    )
    gripper = np.linspace(0.11, 0.02, length)
    raw_action = np.concatenate((position, rotation, gripper[:, None]), axis=-1)
    contract = build_episode_contract(position, rotation, gripper, raw_action)

    current_index = 2
    anchor_pose6 = np.concatenate((position[current_index], rotation[current_index]))
    decoded = relative_chunk_to_world(contract.actions[current_index], anchor_pose6)
    future_indices = np.minimum(
        current_index + 1 + np.arange(ACTION_HORIZON),
        length - 1,
    )
    np.testing.assert_allclose(decoded[:, :3], position[future_indices], atol=2e-7)
    # Compare rotations geometrically because equivalent rotvecs need not be
    # component-identical after scipy canonicalization.
    from scipy.spatial.transform import Rotation

    decoded_rotation = Rotation.from_rotvec(decoded[:, 3:6])
    expected_rotation = Rotation.from_rotvec(rotation[future_indices])
    np.testing.assert_allclose(
        (decoded_rotation.inv() * expected_rotation).magnitude(),
        0.0,
        atol=2e-7,
    )
    np.testing.assert_allclose(decoded[:, 6], gripper[future_indices], atol=1e-7)

    # Input state and all 242 wrist keys are exactly those built by the robot
    # inference client: 241 fixed history frames plus the current frame.
    history = ShellGameHistory()
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    start_pose6 = np.concatenate((position[0], rotation[0]))
    history.seed_dummy_history(image, start_pose6)
    observation = history.build_policy_observation(
        current_frame=image,
        current_pose6=anchor_pose6,
        gripper_width=float(gripper[current_index]),
        prompt="The shell game has ended. Grasp and lift the cup containing the ball.",
    )
    inference_state = np.concatenate(
        [
            observation["robot0_eef_pos"],
            observation["robot0_eef_rot_axis_angle"],
            observation["robot0_gripper_width"],
        ]
    )
    np.testing.assert_allclose(inference_state, contract.state[current_index], atol=2e-7)
    assert len([key for key in observation if key.startswith("left_wrist_0_rgb_")]) == 242
    assert contract.actions.shape[1:] == (ACTION_HORIZON, ACTION_DIM)
