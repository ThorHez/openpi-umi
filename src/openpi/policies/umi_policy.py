import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

from openpi.utils.pose_utils import pose_to_mat, mat_to_pose10d, pose10d_to_mat
from openpi.utils.pose_repr_utils import convert_pose_mat_rep
from openpi.utils.coordinate_transform import pose6d_to_9d, pose9d_to_6d


def make_umi_example() -> dict:
    """Creates a random input example for the UMI policy."""
    return {
        "robot0_eef_pos": np.random.rand(3),
        "robot0_eef_rot_axis_angle": np.random.rand(3),
        "robot0_gripper_width": np.random.rand(1),
        "camera0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(7),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format.

    LeRobot automatically stores images as float32 (C,H,W), so we need to:
    1. Convert float32 [0, 1] to uint8 [0, 255]
    2. Rearrange from CHW to HWC format
    """
    image = np.asarray(image)

    # Convert float32 [0, 1] to uint8 [0, 255]
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    # Rearrange from CHW to HWC
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UmiArxInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    """

    def __call__(self, data: dict) -> dict:
        robot0_eef_pos = data["robot0_eef_pos"]
        robot0_eef_rot_axis_angle = data["robot0_eef_rot_axis_angle"]
        robot0_gripper_width = data["robot0_gripper_width"]
        robot0_eef_rot_axis_angle_wrt_start = data["robot0_eef_rot_axis_angle_wrt_start"]
        camera0_rgb = data["camera0_rgb"]
        left_wrist_0_rgb_1 = data["camera0_rgb"][1]

        assert camera0_rgb.shape == (2, 3, 224, 224)
        assert robot0_eef_pos.shape == (2, 3)
        assert robot0_eef_rot_axis_angle.shape == (2, 6)
        assert robot0_eef_rot_axis_angle_wrt_start.shape == (2, 6)
        assert robot0_gripper_width.shape == (2, 1)

        state = np.concatenate(
            [robot0_eef_pos, robot0_eef_rot_axis_angle, robot0_eef_rot_axis_angle_wrt_start, robot0_gripper_width],
            axis=-1)
        data["state"] = state
        data["image"] = {
            "base_0_rgb": _parse_image(np.zeros_like(left_wrist_0_rgb_1).astype(np.uint8)),
            "left_wrist_0_rgb": _parse_image(left_wrist_0_rgb_1),
            "right_wrist_0_rgb": _parse_image(np.zeros_like(left_wrist_0_rgb_1).astype(np.uint8)),
        }
        data["image_mask"] = {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        }
        data[
            "prompt"] = "pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box"
        return data


@dataclasses.dataclass(frozen=True)
class UmiArxInputs_Bimanual(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    """

    def __call__(self, data: dict) -> dict:
        robot0_eef_pos = data["robot0_eef_pos"]
        robot0_eef_rot_axis_angle = data["robot0_eef_rot_axis_angle"]
        robot0_gripper_width = data["robot0_gripper_width"]
        robot0_eef_rot_axis_angle_wrt_start = data["robot0_eef_rot_axis_angle_wrt_start"]
        robot0_eef_pos_wrt1 = data["robot0_eef_pos_wrt1"]
        robot0_eef_rot_axis_angle_wrt1 = data["robot0_eef_rot_axis_angle_wrt1"]
        camera0_rgb = data["camera0_rgb"]

        # left hand data
        robot1_eef_pos = data["robot1_eef_pos"]
        robot1_eef_rot_axis_angle = data["robot1_eef_rot_axis_angle"]
        robot1_gripper_width = data["robot1_gripper_width"]
        robot1_eef_rot_axis_angle_wrt_start = data["robot1_eef_rot_axis_angle_wrt_start"]
        robot1_eef_pos_wrt0 = data["robot1_eef_pos_wrt0"]
        robot1_eef_rot_axis_angle_wrt0 = data["robot1_eef_rot_axis_angle_wrt0"]
        camera1_rgb = data["camera1_rgb"]

        assert camera0_rgb.shape == (2, 3, 224, 224)
        assert robot0_eef_pos.shape == (2, 3)
        assert robot0_eef_rot_axis_angle.shape == (2, 6)
        assert robot0_eef_rot_axis_angle_wrt_start.shape == (2, 6)
        assert robot0_eef_pos_wrt1.shape == (2, 3)
        assert robot0_eef_rot_axis_angle_wrt1.shape == (2, 6)
        assert robot0_gripper_width.shape == (2, 1)
        assert robot1_eef_pos.shape == (2, 3)
        assert robot1_eef_rot_axis_angle.shape == (2, 6)
        assert robot1_gripper_width.shape == (2, 1)
        assert robot1_eef_rot_axis_angle_wrt_start.shape == (2, 6)
        assert robot1_eef_pos_wrt0.shape == (2, 3)
        assert robot1_eef_rot_axis_angle_wrt0.shape == (2, 6)
        assert camera1_rgb.shape == (2, 3, 224, 224)

        assert camera0_rgb.max() > 0

        state = np.concatenate([robot0_eef_pos,
                                robot0_eef_rot_axis_angle,
                                robot0_eef_rot_axis_angle_wrt_start,
                                robot0_eef_pos_wrt1,
                                robot0_eef_rot_axis_angle_wrt1,
                                robot0_gripper_width,
                                robot1_eef_pos,
                                robot1_eef_rot_axis_angle,
                                robot1_eef_rot_axis_angle_wrt_start,
                                robot1_eef_pos_wrt0,
                                robot1_eef_rot_axis_angle_wrt0,
                                robot1_gripper_width,
                                ], axis=-1)
        data["state"] = state
        data["image"] = {
            "base_0_rgb": _parse_image(np.zeros_like(camera0_rgb[1]).astype(np.uint8)),
            "left_wrist_0_rgb": _parse_image(camera1_rgb[1]),
            "right_wrist_0_rgb": _parse_image(camera0_rgb[1]),
        }
        data["image_mask"] = {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }
        data["prompt"] = "fold the clothes"
        return data


@dataclasses.dataclass(frozen=True)
class UmiArxOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = data["actions"]
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class UmiInputsV4(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    """

    def __call__(self, data: dict) -> dict:
        robot0_eef_pos = data["robot0_eef_pos"]
        robot0_eef_rot_axis_angle = data["robot0_eef_rot_axis_angle"]
        robot0_gripper_width = data["robot0_gripper_width"]
        robot0_eef_rot_axis_angle_wrt_start = data["robot0_eef_rot_axis_angle_wrt_start"]
        left_wrist_0_rgb_1 = data["left_wrist_0_rgb_1"]
        # actions = data["actions"].reshape(16, 10)
        actions = data["actions"]

        assert left_wrist_0_rgb_1.shape == (3, 224, 224)
        assert robot0_eef_pos.shape == (2, 3)
        assert robot0_eef_rot_axis_angle.shape == (2, 6)
        assert robot0_gripper_width.shape == (2, 1)
        assert robot0_eef_rot_axis_angle_wrt_start.shape == (2, 6)
        assert actions.shape == (16, 10)

        state = np.concatenate(
            [robot0_eef_pos, robot0_eef_rot_axis_angle, robot0_eef_rot_axis_angle_wrt_start, robot0_gripper_width],
            axis=-1)
        data["state"] = state
        data["image"] = {
            "base_0_rgb": _parse_image(np.zeros_like(left_wrist_0_rgb_1).astype(np.uint8)),
            "left_wrist_0_rgb": _parse_image(left_wrist_0_rgb_1),
            "right_wrist_0_rgb": _parse_image(np.zeros_like(left_wrist_0_rgb_1).astype(np.uint8)),
        }
        data["image_mask"] = {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        }
        data["actions"] = actions
        # print(f"prompt: {data['prompt']}")
        return data


@dataclasses.dataclass(frozen=True)
class UmiInputsV4_Bimanual(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    """

    def __call__(self, data: dict) -> dict:
        # right hand data
        robot0_eef_pos = data["robot0_eef_pos"]
        robot0_eef_rot_axis_angle = data["robot0_eef_rot_axis_angle"]
        robot0_gripper_width = data["robot0_gripper_width"]
        robot0_eef_rot_axis_angle_wrt_start = data["robot0_eef_rot_axis_angle_wrt_start"]
        robot0_eef_pos_wrt1 = data["robot0_eef_pos_wrt1"]
        robot0_eef_rot_axis_angle_wrt1 = data["robot0_eef_rot_axis_angle_wrt1"]
        right_wrist_0_rgb_1 = data["right_wrist_0_rgb_1"]

        # bimanual data
        # left hand data
        robot1_eef_pos = data["robot1_eef_pos"]
        robot1_eef_rot_axis_angle = data["robot1_eef_rot_axis_angle"]
        robot1_gripper_width = data["robot1_gripper_width"]
        robot1_eef_rot_axis_angle_wrt_start = data["robot1_eef_rot_axis_angle_wrt_start"]
        robot1_eef_pos_wrt0 = data["robot1_eef_pos_wrt0"]
        robot1_eef_rot_axis_angle_wrt0 = data["robot1_eef_rot_axis_angle_wrt0"]
        left_wrist_0_rgb_1 = data["left_wrist_0_rgb_1"]

        # actions = data["actions"].reshape(16, 10)
        actions = data["actions"]

        assert left_wrist_0_rgb_1.shape == (3, 224, 224)
        assert right_wrist_0_rgb_1.shape == (3, 224, 224)
        assert robot0_eef_pos.shape == (2, 3)
        assert robot0_eef_rot_axis_angle.shape == (2, 6)
        assert robot0_gripper_width.shape == (2, 1)
        assert robot0_eef_rot_axis_angle_wrt_start.shape == (2, 6)
        assert robot0_eef_pos_wrt1.shape == (2, 3)
        assert robot0_eef_rot_axis_angle_wrt1.shape == (2, 6)
        assert robot1_eef_pos.shape == (2, 3)
        assert robot1_eef_rot_axis_angle.shape == (2, 6)
        assert robot1_gripper_width.shape == (2, 1)
        assert robot1_eef_rot_axis_angle_wrt_start.shape == (2, 6)
        assert robot1_eef_pos_wrt0.shape == (2, 3)
        assert robot1_eef_rot_axis_angle_wrt0.shape == (2, 6)
        assert actions.shape == (16, 20)

        state = np.concatenate([robot0_eef_pos,
                                robot0_eef_rot_axis_angle,
                                robot0_eef_rot_axis_angle_wrt_start,
                                robot0_eef_pos_wrt1,
                                robot0_eef_rot_axis_angle_wrt1,
                                robot0_gripper_width,
                                robot1_eef_pos,
                                robot1_eef_rot_axis_angle,
                                robot1_eef_rot_axis_angle_wrt_start,
                                robot1_eef_pos_wrt0,
                                robot1_eef_rot_axis_angle_wrt0,
                                robot1_gripper_width,
                                ], axis=-1)
        data["state"] = state
        data["image"] = {
            "base_0_rgb": _parse_image(np.zeros_like(left_wrist_0_rgb_1).astype(np.uint8)),
            "left_wrist_0_rgb": _parse_image(left_wrist_0_rgb_1),
            "right_wrist_0_rgb": _parse_image(right_wrist_0_rgb_1),
        }
        data["image_mask"] = {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }
        data["actions"] = actions
        # print(f"prompt: {data['prompt']}")
        return data


@dataclasses.dataclass(frozen=True)
class UmiOutputsV4(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        return data


@dataclasses.dataclass(frozen=True)
class UmiInputsV2(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    """
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    training_mode: bool

    def __call__(self, data: dict) -> dict:
        left_wrist_camera_image_0 = _parse_image(data["camera0_rgb_0"])
        left_wrist_camera_image_1 = _parse_image(data["camera0_rgb_1"])

        pose_mat = pose_to_mat(np.asarray(np.concatenate([
            data["state_sequence"][:, :3],
            data["state_sequence"][:, 3: 6]
        ], axis=-1), dtype=np.float32))

        # solve reltaive obs
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat,
            base_pose_mat=pose_mat[-1],
            pose_rep="relative",
            backward=False)

        obs_pose = mat_to_pose10d(obs_pose_mat)

        # get start pose
        if self.training_mode:
            start_pose = data["base_state"][: 6]
            # add noise to the start pose when training
            start_pose = np.asarray(start_pose, dtype=np.float32) + np.random.normal(
                scale=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05], size=start_pose.shape)
        else:
            start_pose = data["base_state"][: 6]
        start_pose_mat = pose_to_mat(np.asarray(start_pose, dtype=np.float32))
        rel_obs_pose_mat = convert_pose_mat_rep(
            pose_mat,
            base_pose_mat=start_pose_mat,
            pose_rep='relative',
            backward=False)

        rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)[:, 3:]
        obs_gripper_width = np.expand_dims(data["state_sequence"][:, 6], axis=-1)

        # print(f"obs_pose: {obs_pose.shape}, rel_obs_pose: {rel_obs_pose.shape}, obs_gripper_width: {obs_gripper_width.shape}")

        # 拼接 obs_pose 和 rel_obs_pose
        # obs_pose shape: (sequence_length, 10)
        # rel_obs_pose shape: (sequence_length, 7)
        # obs_gripper_width shape: (sequence_length, 1)
        # state shape: (sequence_length, 18)
        state = np.concatenate([obs_pose, rel_obs_pose, obs_gripper_width], axis=-1)
        # Flatten to 1D so batch processing will make it (batch_size, state_dim)
        # state = state.flatten()

        if self.training_mode:
            action_mat = pose_to_mat(np.asarray(data["actions"][:, :6]).astype(np.float32))
            action_gripper_width = np.expand_dims(np.asarray(data["actions"][:, 6]).astype(np.float32), axis=-1)
            action_pose_mat = convert_pose_mat_rep(
                action_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep="relative",
                backward=False)
            action_pose = mat_to_pose10d(action_pose_mat)
            actions = np.concatenate([action_pose, action_gripper_width], axis=-1)

            # print(f"state: {state.shape}, actions: {actions.shape}")

            inputs = {
                "state": state,
                "image": {
                    # "base_0_rgb": np.zeros_like(left_wrist_camera_image),
                    "left_wrist_0_rgb": left_wrist_camera_image_0,
                    "left_wrist_1_rgb": left_wrist_camera_image_1,
                },
                "image_mask": {
                    # "base_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                    # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                    "left_wrist_0_rgb": np.True_,
                    # "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                    "left_wrist_1_rgb": np.True_,
                },
                "actions": actions,
                "prompt": "pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box",
            }
        else:
            inputs = {
                "state": state,
                "image": {
                    # "base_0_rgb": np.zeros_like(left_wrist_camera_image),
                    "left_wrist_0_rgb": left_wrist_camera_image_0,
                    # "right_wrist_0_rgb": np.zeros_like(left_wrist_camera_image),
                    "left_wrist_1_rgb": left_wrist_camera_image_1,
                },
                "image_mask": {
                    # "base_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                    # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                    "left_wrist_0_rgb": np.True_,
                    # "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                    "left_wrist_1_rgb": np.True_,
                },
                "prompt": "pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box",
            }

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOutputsV2(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        action_pose10d = data["actions"][..., :9]
        action_gripper_width = data["actions"][..., 9]
        action_pose_mat = pose10d_to_mat(action_pose10d)

        pose_mat = pose_to_mat(np.concatenate([
            data["state_sequence"][-1, :3],
            data["state_sequence"][-1, 3: 6]
        ], axis=-1))

        action_mat = convert_pose_mat_rep(
            action_pose_mat,
            base_pose_mat=pose_mat,
            pose_rep="relative",
            backward=True)

        action_pose = mat_to_pose10d(action_mat)
        actions = np.concatenate([action_pose, action_gripper_width], axis=-1)
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class UmiInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    use_10d_pose: bool = False

    def __call__(self, data: dict) -> dict:
        # Get the robot state from the dataset.
        # For UMI, the state is 7-dimensional: [eef_pos (3D), eef_rot (3D), gripper_width (1D)]
        state = np.asarray(data["state"])
        if self.use_10d_pose:
            pose9d = pose6d_to_9d(state[:6])
            state = np.concatenate([pose9d, state[6:]], axis=-1)

        # Parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W).
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "camera0_rgb",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # wrist images below.
        left_wrist_camera_image = _parse_image(data["camera0_rgb"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros_like(left_wrist_camera_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                # UMI only has one camera, so we pad the wrist images.
                "left_wrist_0_rgb": left_wrist_camera_image,
                "right_wrist_0_rgb": np.zeros_like(left_wrist_camera_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
            "base_state": data["base_state"],
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            if self.use_10d_pose:
                actions = np.asarray(data["actions"])
                actions = np.concatenate([pose6d_to_9d(actions[:, :6]), actions[:, 6:]], axis=-1)
                inputs["actions"] = actions
            else:
                inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        # if "prompt" in data:
        #     inputs["prompt"] = data["prompt"]
        inputs[
            "prompt"] = "pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box"

        # print(f"inputs: {inputs}")

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    use_10d_pose: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For UMI, we only return the first 7 actions: [eef_pos (3D), eef_rot (3D), gripper_width (1D)].
        # For your own dataset, replace `7` with the action dimension of your dataset.
        if self.use_10d_pose:
            actions = data["actions"]
            actions = np.concatenate([pose9d_to_6d(actions[:, :9]), actions[:, 9:]], axis=-1)
            return {"actions": np.asarray(actions)}
        else:
            return {"actions": np.asarray(data["actions"][:, :7])}
