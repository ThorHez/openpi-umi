"""Transforms for video frame loading and processing.

This module provides transforms for dynamically loading multiple frames
from LeRobot datasets during training, instead of pre-storing expanded frames.
This significantly reduces dataset storage requirements.
"""

import dataclasses
from typing import Any

import numpy as np
from openpi_client import image_tools

from openpi import transforms as _transforms


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format.

    LeRobot automatically stores images as float32 (C, H, W), so we need to:
    1. Convert float32 [0, 1] to uint8 [0, 255]
    2. Rearrange from CHW to HWC format
    """
    import einops

    image = np.asarray(image)

    # Convert float32 [0, 1] to uint8 [0, 255]
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    # Rearrange from CHW to HWC
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LoadVideoFrames(_transforms.DataTransformFn):
    """Dynamically load multiple video frames from LeRobot dataset during training.

    This transform loads historical frames on-the-fly instead of relying on
    pre-expanded frames in the dataset, significantly reducing storage requirements.

    The transform expects the data dict to contain:
    - 'index': global frame index in the dataset
    - 'episode_index': episode index
    - Single-frame image keys (e.g., 'left_wrist_0_rgb', 'right_wrist_0_rgb')

    And produces:
    - Time-expanded image keys (e.g., 'left_wrist_0_rgb_0', 'left_wrist_0_rgb_1')
      where the index indicates the temporal position (0 = oldest, last = current)

    Args:
        dataset: The underlying LeRobot dataset to load frames from
        image_keys: List of image keys to expand (e.g., ['left_wrist_0_rgb', 'right_wrist_0_rgb'])
        num_frames: Number of frames to load for each image key (temporal horizon)
        frame_stride: Stride between consecutive frames (default: 1)
        padding_mode: How to handle frames at the beginning of an episode:
            - 'repeat': Repeat the first available frame (default)
            - 'zero': Pad with zeros
    """

    # The LeRobot dataset - needed to load frames by index
    dataset: Any
    # Image keys to expand (e.g., ['left_wrist_0_rgb', 'right_wrist_0_rgb'])
    image_keys: tuple[str, ...]
    # Number of frames to load for each image key
    num_frames: int
    # Stride between frames (default: 1)
    frame_stride: int = 1
    # Padding mode for start of episode: 'repeat' or 'zero'
    padding_mode: str = "repeat"

    def __post_init__(self):
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {self.frame_stride}")
        if self.padding_mode not in ("repeat", "zero"):
            raise ValueError(f"padding_mode must be 'repeat' or 'zero', got {self.padding_mode}")

    def __call__(self, data: dict) -> dict:
        # Get the current frame's index and episode info
        current_index = int(data.get("index", -1))
        current_episode = int(data.get("episode_index", -1))
        current_frame_idx = int(data.get("frame_index", -1))

        if current_index < 0:
            # No index information - assume data already has expanded frames
            # This handles inference cases where dataset is not available
            return data

        # Access the underlying HuggingFace dataset
        hf_dataset = self.dataset.hf_dataset

        # Compute the indices we need to load
        # We want to load frames from (current - (num_frames-1)*stride) to current
        target_indices = [
            current_index - i * self.frame_stride
            for i in range(self.num_frames - 1, -1, -1)
        ]

        # Load frames for each image key
        for img_key in self.image_keys:
            frames = []

            for idx in target_indices:
                # Check if we're at the start of an episode
                if idx < 0:
                    # Use padding (will be handled below)
                    frame = None
                else:
                    # Load the frame from dataset
                    try:
                        row = hf_dataset[int(idx)]

                        # Check if we crossed episode boundary
                        row_episode = int(row.get("episode_index", -1))
                        if row_episode != current_episode:
                            # Crossed episode boundary - use padding
                            frame = None
                        else:
                            # Get the image
                            frame = row.get(img_key)
                            if frame is not None:
                                frame = _parse_image(frame)
                    except (IndexError, KeyError, Exception):
                        frame = None

                frames.append(frame)

            # Handle padding for None frames
            valid_frames = [f for f in frames if f is not None]
            if not valid_frames:
                # No valid frames - create a zero frame
                # Try to get shape from current data if available
                if img_key in data and data[img_key] is not None:
                    template = _parse_image(data[img_key])
                    zero_frame = np.zeros_like(template)
                else:
                    # Default shape: (224, 224, 3) uint8
                    zero_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                frames = [zero_frame] * self.num_frames
            else:
                # Fill in None frames
                first_valid = valid_frames[0]
                filled_frames = []
                for f in frames:
                    if f is None:
                        if self.padding_mode == "repeat":
                            filled_frames.append(first_valid)
                        else:  # zero
                            filled_frames.append(np.zeros_like(first_valid))
                    else:
                        filled_frames.append(f)
                frames = filled_frames

            # Store frames with indexed keys (e.g., left_wrist_0_rgb_0, _1, etc.)
            for i, frame in enumerate(frames):
                key = f"{img_key}_{i}"
                data[key] = frame

        return data


@dataclasses.dataclass(frozen=True)
class BuildVideoTensor(_transforms.DataTransformFn):
    """Build video tensor from individual frames for Pi0Mem model.

    This transform stacks individual frame images into a video tensor
    with shape [T, H, W, C] suitable for the Pi0Mem model.

    Args:
        image_keys: List of base image keys (e.g., ['left_wrist_0_rgb', 'right_wrist_0_rgb'])
        num_frames: Number of frames per key
        output_keys: Optional mapping from input key to output key name.
                     If not provided, uses the base key name.
    """

    image_keys: tuple[str, ...]
    num_frames: int
    output_keys: dict[str, str] | None = None

    def __call__(self, data: dict) -> dict:
        output_keys = self.output_keys or {}

        for img_key in self.image_keys:
            # Collect frames
            frames = []
            for i in range(self.num_frames):
                key = f"{img_key}_{i}"
                if key not in data:
                    raise KeyError(f"Missing frame key: {key}")
                frame = data[key]
                # Ensure frame is numpy array in HWC format
                frame = np.asarray(frame)
                if frame.ndim == 3 and frame.shape[0] == 3:
                    # Convert CHW to HWC if needed
                    import einops

                    frame = einops.rearrange(frame, "c h w -> h w c")
                frames.append(frame)

            # Stack into video tensor [T, H, W, C]
            video = np.stack(frames, axis=0)  # Shape: [num_frames, H, W, C]

            # Store with output key name
            out_key = output_keys.get(img_key, img_key)
            data[out_key] = video

            # Clean up individual frame keys to save memory
            for i in range(self.num_frames):
                key = f"{img_key}_{i}"
                if key in data:
                    del data[key]

        return data


@dataclasses.dataclass(frozen=True)
class LoadAndBuildVideo(_transforms.DataTransformFn):
    """Convenience transform that combines LoadVideoFrames and BuildVideoTensor.

    This is a single transform that:
    1. Loads multiple historical frames from the dataset
    2. Stacks them into video tensors [T, H, W, C]

    Args:
        dataset: The underlying LeRobot dataset
        image_keys: List of image keys to process
        num_frames: Number of frames to load per key
        frame_stride: Stride between frames (default: 1)
        padding_mode: 'repeat' or 'zero'
        output_keys: Optional mapping from input to output key names
    """

    dataset: Any
    image_keys: tuple[str, ...]
    num_frames: int
    frame_stride: int = 1
    padding_mode: str = "repeat"
    output_keys: dict[str, str] | None = None

    def __call__(self, data: dict) -> dict:
        # First load the frames
        loader = LoadVideoFrames(
            dataset=self.dataset,
            image_keys=self.image_keys,
            num_frames=self.num_frames,
            frame_stride=self.frame_stride,
            padding_mode=self.padding_mode,
        )
        data = loader(data)

        # Then build video tensors
        builder = BuildVideoTensor(
            image_keys=self.image_keys,
            num_frames=self.num_frames,
            output_keys=self.output_keys,
        )
        return builder(data)


@dataclasses.dataclass(frozen=True)
class FormatPi0MemVideoInput(_transforms.DataTransformFn):
    """Format video tensors for Pi0Mem model input.

    Pi0Mem expects images in a dict with shape [T, H, W, C] and creates
    image masks automatically.

    Args:
        image_key_mapping: Mapping from dataset image keys to model image keys.
                          e.g., {"left_wrist_0_rgb": "left_wrist_0_rgb",
                                 "right_wrist_0_rgb": "right_wrist_0_rgb"}
        base_image_key: Key to use for the base (third-person) camera view.
                       If the dataset doesn't have this view, it will be created
                       as zeros with appropriate mask=False.
    """

    image_key_mapping: dict[str, str]
    base_image_key: str | None = None

    def __call__(self, data: dict) -> dict:
        images = {}
        image_masks = {}

        for src_key, dst_key in self.image_key_mapping.items():
            if src_key not in data:
                raise KeyError(f"Missing image key: {src_key}")

            video = data[src_key]  # Shape: [T, H, W, C]

            # Ensure video is numpy array
            video = np.asarray(video)

            # Verify shape
            if video.ndim != 4:
                raise ValueError(
                    f"Video for {src_key} should have shape [T, H, W, C], got {video.shape}"
                )

            # Store video tensor
            images[dst_key] = video

            # All frames in the video are valid (no masking within video)
            # Create a single mask value for this video stream
            image_masks[dst_key] = np.True_

        # Handle base camera if specified
        if self.base_image_key and self.base_image_key not in images:
            # Get shape from first available video
            first_video = next(iter(images.values()))
            base_video = np.zeros_like(first_video)
            images[self.base_image_key] = base_video
            image_masks[self.base_image_key] = np.False_

        data["images"] = images
        data["image_masks"] = image_masks

        # Clean up original image keys to save memory
        for src_key in self.image_key_mapping.keys():
            if src_key in data and src_key not in images:
                del data[src_key]

        return data
