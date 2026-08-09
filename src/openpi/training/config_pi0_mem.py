"""Pi0Mem training paradigm helpers.

This module hosts the two pieces of glue that distinguish Pi0Mem training from
the standard OpenPI pipeline:

1. ``UmiInputsV4_Bimanual_Video`` — a data transform that mirrors
   ``UmiInputsV4_Bimanual_Horizon1`` but consumes stacked video tensors
   ``(T, H, W, C)`` produced by ``transforms_video.BuildVideoTensor`` and
   emits ``data["image"][<key>]`` of shape ``(T, 224, 224, 3)`` ready for
   Pi0Mem's siglip_mem encoder.

2. ``create_pi0_mem_data_loader`` — a data-loader factory that wraps the
   underlying ``LeRobotDataset`` with ``VideoFrameDataset`` (dynamic per-
   ``__getitem__`` historical frame loading) before the standard
   transform / collate / sharding pipeline runs.

The ``DataConfigFactory`` and ``TrainConfig`` entries live in
``openpi.training.config`` (registered in the central ``_CONFIGS`` list);
this module is consumed lazily from there to avoid an import cycle.

The Pi0Mem trainer (``scripts/mem/train_pi0_mem.py``) is byte-for-byte
identical to ``scripts/train.py`` except that it calls
``create_pi0_mem_data_loader(config, ...)`` in place of
``openpi.training.data_loader.create_data_loader(config, ...)``.
"""

from __future__ import annotations

import dataclasses
import logging

import einops
import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms
from openpi.training.mem.video_dataset import VideoFrameConfig, VideoFrameDataset


# ---------------------------------------------------------------------------
# Pi0Mem-specific data transform
# ---------------------------------------------------------------------------


def _parse_video(video: np.ndarray) -> np.ndarray:
    """Parse a video tensor to uint8 ``(T, H, W, C)`` format.

    LeRobot stores images as float32 ``(C, H, W)``. After
    ``BuildVideoTensor`` stacks ``T`` frames into a video, the array can be
    ``(T, C, H, W)`` (float32 from LeRobot) or already ``(T, H, W, C)``.
    Normalize to uint8 HWC per frame.
    """

    video = np.asarray(video)

    if video.ndim != 4:
        raise ValueError(
            f"Expected video with shape (T, H, W, C) or (T, C, H, W); got {video.shape}"
        )

    if np.issubdtype(video.dtype, np.floating):
        video = (255.0 * video).astype(np.uint8)

    # ``(T, C, H, W)`` -> ``(T, H, W, C)``.
    if video.shape[1] == 3 and video.shape[-1] != 3:
        video = einops.rearrange(video, "t c h w -> t h w c")

    return video


def _build_bimanual_state(data: dict) -> np.ndarray:
    """Concatenate the 38-dim bimanual state vector used by all Pi0Mem UMI factories.

    Order matches ``UmiInputsV4_Bimanual_Horizon1`` /
    ``UmiInputsV4_Bimanual_HeadView_Depth_Horizon1`` byte-for-byte so the
    state norm stats and ``normalize_masks`` carry over unchanged.
    """

    return np.concatenate(
        [
            data["robot0_eef_pos"],
            data["robot0_eef_pos_wrt_start"],
            data["robot0_eef_rot_axis_angle"],
            data["robot0_eef_rot_axis_angle_wrt_start"],
            data["robot0_eef_pos_wrt1"],
            data["robot0_eef_rot_axis_angle_wrt1"],
            data["robot0_gripper_width"],
            data["robot1_eef_pos"],
            data["robot1_eef_pos_wrt_start"],
            data["robot1_eef_rot_axis_angle"],
            data["robot1_eef_rot_axis_angle_wrt_start"],
            data["robot1_eef_pos_wrt0"],
            data["robot1_eef_rot_axis_angle_wrt0"],
            data["robot1_gripper_width"],
        ],
        axis=-1,
    )

def _build_wbcd_bimanual_state(data: dict) -> np.ndarray:
    """Concatenate the 38-dim bimanual state vector used by all Pi0Mem UMI factories.

    Order matches ``UmiInputsV4_Bimanual_Horizon1`` /
    ``UmiInputsV4_Bimanual_HeadView_Depth_Horizon1`` byte-for-byte so the
    state norm stats and ``normalize_masks`` carry over unchanged.
    """

    return np.concatenate(
        [
            data["robot0_eef_pos"],
            data["robot0_eef_pos_wrt_start"],
            data["robot0_eef_rot_axis_angle"],
            data["robot0_eef_rot_axis_angle_wrt_start"],
            # data["robot0_eef_pos_wrt1"],
            # data["robot0_eef_rot_axis_angle_wrt1"],
            data["robot0_gripper_width"],
            data["robot1_eef_pos"],
            data["robot1_eef_pos_wrt_start"],
            data["robot1_eef_rot_axis_angle"],
            data["robot1_eef_rot_axis_angle_wrt_start"],
            # data["robot1_eef_pos_wrt0"],
            # data["robot1_eef_rot_axis_angle_wrt0"],
            data["robot1_gripper_width"],
        ],
        axis=-1,
    )


@dataclasses.dataclass(frozen=True)
class UmiInputsV4_Bimanual_Video(_transforms.DataTransformFn):
    """Pi0Mem video twin of ``UmiInputsV4_Bimanual_Horizon1``.

    Reads stacked video tensors at ``left_wrist_0_rgb_0_video`` and
    ``right_wrist_0_rgb_0_video`` (each ``(T, 3, 224, 224)`` from LeRobot's
    CHW float storage), builds the 38-dim concatenated bimanual state vector
    (identical concatenation order to ``UmiInputsV4_Bimanual_Horizon1``),
    and emits the ``image`` / ``image_mask`` dicts that Pi0Mem expects.
    """

    num_frames: int

    def __call__(self, data: dict) -> dict:
        # --- video tensors ---
        left_video = _parse_video(data["left_wrist_0_rgb_0_video"])
        right_video = _parse_video(data["right_wrist_0_rgb_0_video"])

        actions = data["actions"]

        T = self.num_frames
        assert left_video.shape == (T, 224, 224, 3), (
            f"left_wrist_0_rgb_0_video shape {left_video.shape} != (T={T}, 224, 224, 3)"
        )
        assert right_video.shape == (T, 224, 224, 3), (
            f"right_wrist_0_rgb_0_video shape {right_video.shape} != (T={T}, 224, 224, 3)"
        )
        assert actions.shape == (16, 20), f"actions shape {actions.shape} != (16, 20)"

        data["state"] = _build_bimanual_state(data)

        # Pi0Mem.embed_prefix accepts ``(B, T, H, W, C)`` per image stream.
        # We also synthesize a zero base_0_rgb video with mask=False so the
        # model's three expected image keys are always populated.
        zero_video = np.zeros_like(left_video)
        data["image"] = {
            "base_0_rgb": zero_video,
            "left_wrist_0_rgb": left_video,
            "right_wrist_0_rgb": right_video,
        }
        data["image_mask"] = {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class WBCD_V1_Bimanual_Video(_transforms.DataTransformFn):
    """Pi0Mem video twin of ``UmiInputsV4_Bimanual_Horizon1``.

    Reads stacked video tensors at ``left_wrist_0_rgb_0_video`` and
    ``right_wrist_0_rgb_0_video`` (each ``(T, 3, 224, 224)`` from LeRobot's
    CHW float storage), builds the 38-dim concatenated bimanual state vector
    (identical concatenation order to ``UmiInputsV4_Bimanual_Horizon1``),
    and emits the ``image`` / ``image_mask`` dicts that Pi0Mem expects.
    """

    num_frames: int

    def __call__(self, data: dict) -> dict:
        # --- video tensors ---
        left_video_0 = _parse_video(data["left_wrist_0_rgb_0_video"])
        right_video_0 = _parse_video(data["right_wrist_0_rgb_0_video"])
        left_video_1 = _parse_video(data["left_wrist_1_rgb_0_video"])
        right_video_1 = _parse_video(data["right_wrist_1_rgb_0_video"])

        actions = data["actions"]

        T = self.num_frames
        assert left_video_0.shape == (T, 224, 224, 3), (
            f"left_wrist_0_rgb_0_video shape {left_video_0.shape} != (T={T}, 224, 224, 3)"
        )
        assert right_video_0.shape == (T, 224, 224, 3), (
            f"right_wrist_0_rgb_0_video shape {right_video_0.shape} != (T={T}, 224, 224, 3)"
        )
        assert left_video_1.shape == (T, 224, 224, 3), (
            f"left_wrist_1_rgb_0_video shape {left_video_1.shape} != (T={T}, 224, 224, 3)"
        )
        assert right_video_1.shape == (T, 224, 224, 3), (
            f"right_wrist_1_rgb_0_video shape {right_video_1.shape} != (T={T}, 224, 224, 3)"
        )
        # assert actions.shape == (32, 20), f"actions shape {actions.shape} != (16, 20)"

        data["state"] = _build_wbcd_bimanual_state(data)

        data["image"] = {
            "left_wrist_0_rgb": left_video_0,
            "right_wrist_0_rgb": right_video_0,
            "left_wrist_1_rgb": left_video_1,
            "right_wrist_1_rgb": right_video_1,
        }
        data["image_mask"] = {
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
            "left_wrist_1_rgb": np.True_,
            "right_wrist_1_rgb": np.True_,
        }
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class UmiInputsV4_Bimanual_HeadView_Depth_Video(_transforms.DataTransformFn):
    """Pi0Mem video twin of ``UmiInputsV4_Bimanual_HeadView_Depth_Horizon1``.

    Reads four stacked video tensors:
        - ``left_wrist_0_rgb_0_video``    (T, 3, 224, 224) uint8/float
        - ``right_wrist_0_rgb_0_video``   (T, 3, 224, 224) uint8/float
        - ``base_0_rgb_0_video``          (T, 3, 224, 224) uint8/float
        - ``base_0_depth_0_video``        (T, 224, 224, 3) uint8 (after
                                          per-frame depth-to-3ch conversion
                                          applied earlier in the pipeline)

    Builds the same 38-d concatenated bimanual state used elsewhere and
    emits a 4-stream ``data["image"]`` / ``data["image_mask"]`` dict.

    Pi0Mem dispatches over whatever image keys are in ``obs.images`` (its
    ``inputs_spec`` only declares 3 streams but the params are independent
    of the obs structure — only the encoder needs lazy_init, which uses
    one sample frame).
    """

    num_frames: int

    def __call__(self, data: dict) -> dict:
        left_video = _parse_video(data["left_wrist_0_rgb_0_video"])
        right_video = _parse_video(data["right_wrist_0_rgb_0_video"])
        base_rgb_video = _parse_video(data["base_0_rgb_0_video"])
        base_depth_video = _parse_video(data["base_0_depth_0_video"])

        actions = data["actions"]

        T = self.num_frames
        for name, video in [
            ("left_wrist_0_rgb_0_video", left_video),
            ("right_wrist_0_rgb_0_video", right_video),
            ("base_0_rgb_0_video", base_rgb_video),
            ("base_0_depth_0_video", base_depth_video),
        ]:
            assert video.shape == (T, 224, 224, 3), (
                f"{name} shape {video.shape} != (T={T}, 224, 224, 3)"
            )
        assert actions.shape == (16, 20), f"actions shape {actions.shape} != (16, 20)"

        data["state"] = _build_bimanual_state(data)

        data["image"] = {
            "base_0_rgb": base_rgb_video,
            "base_0_depth": base_depth_video,
            "left_wrist_0_rgb": left_video,
            "right_wrist_0_rgb": right_video,
        }
        data["image_mask"] = {
            "base_0_rgb": np.True_,
            "base_0_depth": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class UmiInputsV4_Shellgame_Video(_transforms.DataTransformFn):
    """Pi0Mem video twin of ``UmiInputsV4_Bimanual_HeadView_Depth_Horizon1``.

    Reads four stacked video tensors:
        - ``left_wrist_0_rgb_0_video``    (T, 3, 224, 224) uint8/float
        - ``right_wrist_0_rgb_0_video``   (T, 3, 224, 224) uint8/float
        - ``base_0_rgb_0_video``          (T, 3, 224, 224) uint8/float
        - ``base_0_depth_0_video``        (T, 224, 224, 3) uint8 (after
                                          per-frame depth-to-3ch conversion
                                          applied earlier in the pipeline)

    Builds the same 38-d concatenated bimanual state used elsewhere and
    emits a 4-stream ``data["image"]`` / ``data["image_mask"]`` dict.

    Pi0Mem dispatches over whatever image keys are in ``obs.images`` (its
    ``inputs_spec`` only declares 3 streams but the params are independent
    of the obs structure — only the encoder needs lazy_init, which uses
    one sample frame).
    """

    num_frames: int

    def _build_shellgame_state(self, data: dict) -> np.ndarray:
        return np.concatenate(
            [
                data["robot0_eef_pos"],
                data["robot0_eef_rot_axis_angle"],
                data["robot0_gripper_width"],
            ],
            axis=-1,
        )

    def __call__(self, data: dict) -> dict:
        wrist_video = _parse_video(data["left_wrist_0_rgb_0_video"])
        base_video = _parse_video(data["left_wrist_0_rgb_1_video"])
        # right_video = _parse_video(data["right_wrist_0_rgb_0_video"])
        # base_rgb_video = _parse_video(data["base_0_rgb_0_video"])
        # base_depth_video = _parse_video(data["base_0_depth_0_video"])

        actions = data["actions"]

        T = self.num_frames
        for name, video in [
            ("wrist_video", wrist_video),
            ("base_video", base_video),
            # ("base_0_rgb_0_video", base_rgb_video),
            # ("base_0_depth_0_video", base_depth_video),
        ]:
            assert video.shape == (T, 224, 224, 3), (
                f"{name} shape {video.shape} != (T={T}, 224, 224, 3)"
            )
        # assert actions.shape == (16, 10), f"actions shape {actions.shape} != (16, 10)"

        data["state"] = self._build_shellgame_state(data)

        data["image"] = {
            # "base_0_rgb": base_rgb_video,
            # "base_0_depth": base_depth_video,
            # "left_wrist_0_rgb": left_video,
            # "right_wrist_0_rgb": right_video,
            "base_rgb": base_video,
            "wrist_rgb": wrist_video
        }
        data["image_mask"] = {
            # "base_0_rgb": np.True_,
            # "base_0_depth": np.True_,
            # "left_wrist_0_rgb": np.True_,
            # "right_wrist_0_rgb": np.True_,


            "base_rgb": np.True_,
            "wrist_rgb": np.True_
        }
        frame_valid_mask = data.get("video_frame_valid_mask")
        if frame_valid_mask is not None:
            data["frame_valid_mask"] = {
                "base_rgb": np.asarray(frame_valid_mask.get("left_wrist_0_rgb_1", np.ones(T, dtype=np.bool_))),
                "wrist_rgb": np.asarray(frame_valid_mask.get("left_wrist_0_rgb_0", np.ones(T, dtype=np.bool_))),
            }
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class UmiInputsV4_Shellgame_Video_Joint(_transforms.DataTransformFn):
    """Pi0Mem video twin of ``UmiInputsV4_Bimanual_HeadView_Depth_Horizon1``.

    Reads four stacked video tensors:
        - ``left_wrist_0_rgb_0_video``    (T, 3, 224, 224) uint8/float
        - ``right_wrist_0_rgb_0_video``   (T, 3, 224, 224) uint8/float
        - ``base_0_rgb_0_video``          (T, 3, 224, 224) uint8/float
        - ``base_0_depth_0_video``        (T, 224, 224, 3) uint8 (after
                                          per-frame depth-to-3ch conversion
                                          applied earlier in the pipeline)

    Builds the same 38-d concatenated bimanual state used elsewhere and
    emits a 4-stream ``data["image"]`` / ``data["image_mask"]`` dict.

    Pi0Mem dispatches over whatever image keys are in ``obs.images`` (its
    ``inputs_spec`` only declares 3 streams but the params are independent
    of the obs structure — only the encoder needs lazy_init, which uses
    one sample frame).
    """

    num_frames: int

    def _build_shellgame_state(self, data: dict) -> np.ndarray:
        return np.concatenate(
            [
                data["robot0_joint_pos"],
                data["robot0_gripper_width"],
            ],
            axis=-1,
        )

    def __call__(self, data: dict) -> dict:
        wrist_video = _parse_video(data["left_wrist_0_rgb_0_video"])
        base_video = _parse_video(data["left_wrist_0_rgb_1_video"])
        # right_video = _parse_video(data["right_wrist_0_rgb_0_video"])
        # base_rgb_video = _parse_video(data["base_0_rgb_0_video"])
        # base_depth_video = _parse_video(data["base_0_depth_0_video"])

        actions = data["actions"]

        T = self.num_frames
        for name, video in [
            ("wrist_video", wrist_video),
            ("base_video", base_video),
            # ("base_0_rgb_0_video", base_rgb_video),
            # ("base_0_depth_0_video", base_depth_video),
        ]:
            assert video.shape == (T, 224, 224, 3), (
                f"{name} shape {video.shape} != (T={T}, 224, 224, 3)"
            )
        # assert actions.shape == (16, 10), f"actions shape {actions.shape} != (16, 10)"

        data["state"] = self._build_shellgame_state(data)

        data["image"] = {
            # "base_0_rgb": base_rgb_video,
            # "base_0_depth": base_depth_video,
            # "left_wrist_0_rgb": left_video,
            # "right_wrist_0_rgb": right_video,
            "base_rgb": base_video,
            "wrist_rgb": wrist_video
        }
        data["image_mask"] = {
            # "base_0_rgb": np.True_,
            # "base_0_depth": np.True_,
            # "left_wrist_0_rgb": np.True_,
            # "right_wrist_0_rgb": np.True_,


            "base_rgb": np.True_,
            "wrist_rgb": np.True_
        }
        frame_valid_mask = data.get("video_frame_valid_mask")
        if frame_valid_mask is not None:
            data["frame_valid_mask"] = {
                "base_rgb": np.asarray(frame_valid_mask.get("left_wrist_0_rgb_1", np.ones(T, dtype=np.bool_))),
                "wrist_rgb": np.asarray(frame_valid_mask.get("left_wrist_0_rgb_0", np.ones(T, dtype=np.bool_))),
            }
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class UmiInputsV4_Shellgame_Base(_transforms.DataTransformFn):
    """Pi0Mem video twin of ``UmiInputsV4_Bimanual_HeadView_Depth_Horizon1``.

    Reads four stacked video tensors:
        - ``left_wrist_0_rgb_0_video``    (T, 3, 224, 224) uint8/float
        - ``right_wrist_0_rgb_0_video``   (T, 3, 224, 224) uint8/float
        - ``base_0_rgb_0_video``          (T, 3, 224, 224) uint8/float
        - ``base_0_depth_0_video``        (T, 224, 224, 3) uint8 (after
                                          per-frame depth-to-3ch conversion
                                          applied earlier in the pipeline)

    Builds the same 38-d concatenated bimanual state used elsewhere and
    emits a 4-stream ``data["image"]`` / ``data["image_mask"]`` dict.

    Pi0Mem dispatches over whatever image keys are in ``obs.images`` (its
    ``inputs_spec`` only declares 3 streams but the params are independent
    of the obs structure — only the encoder needs lazy_init, which uses
    one sample frame).
    """

    def _build_shellgame_state(self, data: dict) -> np.ndarray:
        return np.concatenate(
            [
                data["robot0_eef_pos"],
                data["robot0_eef_rot_axis_angle"],
                data["robot0_gripper_width"],
            ],
            axis=-1,
        )

    def _parse_image(self, image) -> np.ndarray:
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

    def __call__(self, data: dict) -> dict:
        wrist_image = self._parse_image(data["left_wrist_0_rgb_0"])
        base_image = self._parse_image(data["left_wrist_0_rgb_1"])
        # right_video = _parse_video(data["right_wrist_0_rgb_0_video"])
        # base_rgb_video = _parse_video(data["base_0_rgb_0_video"])
        # base_depth_video = _parse_video(data["base_0_depth_0_video"])

        actions = data["actions"]

        # T = self.num_frames
        # for name, video in [
        #     ("wrist_video", wrist_video),
        #     ("base_video", base_video),
        #     # ("base_0_rgb_0_video", base_rgb_video),
        #     # ("base_0_depth_0_video", base_depth_video),
        # ]:
        #     assert video.shape == (T, 224, 224, 3), (
        #         f"{name} shape {video.shape} != (T={T}, 224, 224, 3)"
        #     )
        # assert actions.shape == (16, 10), f"actions shape {actions.shape} != (16, 10)"

        data["state"] = self._build_shellgame_state(data)

        data["image"] = {
            # "base_0_rgb": base_rgb_video,
            # "base_0_depth": base_depth_video,
            # "left_wrist_0_rgb": left_video,
            # "right_wrist_0_rgb": right_video,
            "base_rgb": base_image,
            "wrist_rgb": wrist_image
        }
        data["image_mask"] = {
            # "base_0_rgb": np.True_,
            # "base_0_depth": np.True_,
            # "left_wrist_0_rgb": np.True_,
            # "right_wrist_0_rgb": np.True_,


            "base_rgb": np.True_,
            "wrist_rgb": np.True_
        }
        # frame_valid_mask = data.get("video_frame_valid_mask")
        # if frame_valid_mask is not None:
        #     data["frame_valid_mask"] = {
        #         "base_rgb": np.asarray(frame_valid_mask.get("left_wrist_0_rgb_1", np.ones(T, dtype=np.bool_))),
        #         "wrist_rgb": np.asarray(frame_valid_mask.get("left_wrist_0_rgb_0", np.ones(T, dtype=np.bool_))),
        #     }
        data["actions"] = actions
        return data


# ---------------------------------------------------------------------------
# Pi0Mem data loader
# ---------------------------------------------------------------------------


def _build_pi0_mem_dataset(
    data_config: _config.DataConfig,
    video_frame_config: VideoFrameConfig,
    action_horizon: int,
    *,
    skip_norm_stats: bool = False,
) -> _data_loader.Dataset:
    """LeRobotDataset -> (optional) PromptFromLeRobotTask -> VideoFrameDataset -> transform_dataset."""

    repo_id = data_config.repo_id
    if repo_id is None or repo_id == "fake":
        raise ValueError(
            "Pi0Mem training requires a real LeRobot dataset (data.repo_id); "
            f"got repo_id={repo_id!r}."
        )

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    base_ds = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    # Match the prompt injection done by data_loader.create_torch_dataset.
    if data_config.prompt_from_task:
        base_ds = _data_loader.TransformedDataset(
            base_ds, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    # The defining piece of the Pi0Mem paradigm: wrap the dataset so each
    # __getitem__ resolves T historical frames from the underlying HF dataset.
    video_ds = VideoFrameDataset(base_ds, video_frame_config)

    # Reuse the standard transform pipeline (repack -> data_transforms ->
    # Normalize -> model_transforms -> action_loss_mask) so we stay in lock
    # step with create_data_loader's normalization / robot_type / loss-mask
    # handling.
    return _data_loader.transform_dataset(video_ds, data_config, skip_norm_stats=skip_norm_stats)


def _is_pi0_mem_aware(factory) -> bool:
    """A DataConfigFactory is Pi0Mem-aware iff it exposes ``video_frame_config()``."""
    return hasattr(factory, "video_frame_config")


def create_pi0_mem_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: str = "jax",
) -> _data_loader.DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Pi0Mem analogue of ``data_loader.create_data_loader``.

    Identical contract to the standard data loader (yields
    ``(Observation, Actions)`` batches on the given sharding) — the only
    behavioral difference is the ``VideoFrameDataset`` wrapping step
    inserted between each raw LeRobot dataset and the transform pipeline.

    Dispatches to a multi-dataset path when ``config.data`` is a
    :class:`openpi.training.config.MultiDataConfigFactory` (mirrors how
    :mod:`scripts.train_multi_dataset` handles single vs. multi).
    """

    if isinstance(config.data, _config.MultiDataConfigFactory):
        return _create_pi0_mem_multi_data_loader(
            config,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )

    if not _is_pi0_mem_aware(config.data):
        raise ValueError(
            "create_pi0_mem_data_loader requires a Pi0Mem-aware DataConfigFactory "
            "(must expose .video_frame_config()); got "
            f"{type(config.data).__name__}."
        )

    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    video_frame_config = config.data.video_frame_config()
    logging.info(
        f"Pi0Mem video frame loading: num_frames={video_frame_config.num_frames}, "
        f"frame_stride={video_frame_config.frame_stride}, "
        f"padding_mode={video_frame_config.padding_mode}, "
        f"image_keys={video_frame_config.image_keys}"
    )

    dataset = _build_pi0_mem_dataset(
        data_config,
        video_frame_config,
        action_horizon=config.model.action_horizon,
        skip_norm_stats=skip_norm_stats,
    )

    # Same per-process batch-size handling as create_torch_data_loader. Pi0Mem
    # uses JAX FSDP exclusively, so we don't bother with the PyTorch-DDP path.
    local_batch_size = config.batch_size // jax.process_count()
    logging.info(f"local_batch_size: {local_batch_size}")

    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework=framework,
        # Video loading is CPU-heavy: each sample fetches up to num_frames
        # rows from the underlying LeRobotDataset. Prefetch ~4 batches per
        # worker so the GPU isn't starved while workers decode frames.
        prefetch_factor=8 if config.num_workers > 0 else None,
    )

    return _data_loader.DataLoaderImpl(data_config, torch_loader)


def _create_pi0_mem_multi_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None,
    shuffle: bool,
    num_batches: int | None,
    skip_norm_stats: bool,
    framework: str,
) -> _data_loader.DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Pi0Mem multi-dataset analogue of ``multi_data_loader.create_multi_data_loader``.

    Each child of ``config.data`` (a ``MultiDataConfigFactory``) must be
    Pi0Mem-aware (expose ``video_frame_config()``). For each child we:

    1. Use ``MultiDataConfigFactory.create_all`` to materialize per-child
       DataConfigs with merged norm_stats and ``state_pad_dim`` applied.
    2. Build a Pi0Mem dataset (LeRobotDataset → optional PromptFromLeRobotTask
       → VideoFrameDataset → transform_dataset).
    3. Concat with ``WeightedConcatDataset`` and use a
       ``WeightedRandomSampler`` when per-dataset weights differ.

    Returns a :class:`MultiDataLoaderImpl` so checkpoint saving can iterate
    norm stats per dataset (identical contract to the standard multi
    loader).
    """

    from openpi.training import multi_data_loader as _multi_loader
    import torch

    multi_factory: _config.MultiDataConfigFactory = config.data  # type: ignore[assignment]
    if not multi_factory.datasets:
        raise ValueError("MultiDataConfigFactory.datasets must be non-empty for Pi0Mem.")

    for i, child in enumerate(multi_factory.datasets):
        if not _is_pi0_mem_aware(child):
            raise ValueError(
                f"MultiDataConfigFactory.datasets[{i}] is "
                f"{type(child).__name__}, which is not Pi0Mem-aware. "
                "Every child factory must expose .video_frame_config()."
            )

    all_configs = multi_factory.create_all(config.assets_dirs, config.model)
    weights_list = multi_factory.weights or [1.0] * len(all_configs)
    if len(weights_list) != len(all_configs):
        raise ValueError(
            f"MultiDataConfigFactory.weights has {len(weights_list)} entries but "
            f"datasets has {len(all_configs)}."
        )

    datasets = []
    for i, (dc, child) in enumerate(zip(all_configs, multi_factory.datasets)):
        vfc = child.video_frame_config()
        logging.info(
            f"  Pi0Mem multi-dataset[{i}]: repo_id={dc.repo_id}, asset_id={dc.asset_id}, "
            f"weight={weights_list[i]}, num_frames={vfc.num_frames}, "
            f"image_keys={vfc.image_keys}"
        )
        ds = _build_pi0_mem_dataset(
            dc,
            vfc,
            action_horizon=config.model.action_horizon,
            skip_norm_stats=skip_norm_stats,
        )
        datasets.append(ds)

    use_weights = len(set(weights_list)) > 1
    concat = _multi_loader.WeightedConcatDataset(
        datasets,
        weights=weights_list if use_weights else None,
    )

    for i, (dc, ds) in enumerate(zip(all_configs, datasets)):
        logging.info(
            f"  Pi0Mem multi-dataset[{i}] size: repo_id={dc.repo_id}, len={len(ds)}, "
            f"weight={weights_list[i]}"
        )

    sampler = None
    if use_weights:
        index_weights = torch.tensor(
            concat.get_dataset_weights_for_sampler(), dtype=torch.double
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            index_weights,
            num_samples=len(concat),
            replacement=True,
        )

    local_batch_size = config.batch_size // jax.process_count()
    if len(concat) < local_batch_size:
        raise ValueError(
            f"Concatenated dataset size ({len(concat)}) is smaller than "
            f"local_batch_size ({local_batch_size})."
        )

    torch_loader = _data_loader.TorchDataLoader(
        concat,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework=framework,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )

    return _multi_loader.MultiDataLoaderImpl(all_configs, torch_loader)
