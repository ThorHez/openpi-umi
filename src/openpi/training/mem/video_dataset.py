"""Video dataset wrapper for dynamic frame loading.

This module provides a dataset wrapper that enables dynamic loading of
multiple video frames during training, significantly reducing storage
requirements compared to pre-expanded datasets.
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np

from openpi import transforms as _transforms
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass(frozen=True)
class VideoFrameConfig:
    """Configuration for video frame loading.

    Args:
        image_keys: List of image keys to expand (e.g., ['left_wrist_0_rgb', 'right_wrist_0_rgb'])
        num_frames: Number of frames to load per key (temporal horizon)
        frame_stride: Stride between consecutive frames (default: 1)
        padding_mode: How to handle start of episode: 'repeat' or 'zero'
    """

    image_keys: tuple[str, ...]
    num_frames: int = 2
    frame_stride: int = 1
    padding_mode: str = "repeat"


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    import einops

    image = np.asarray(image)

    # Convert float32 [0, 1] to uint8 [0, 255]
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    # Rearrange from CHW to HWC
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


class VideoFrameDataset(_data_loader.Dataset):
    """Dataset wrapper that loads video frames dynamically.

    This wrapper takes a base dataset (e.g., LeRobotDataset) and loads
    multiple historical frames on-the-fly during __getitem__, instead
    of relying on pre-expanded frames.

    This significantly reduces dataset storage requirements at the cost
    of slightly higher data loading time.

    Usage:
        base_dataset = LeRobotDataset(...)
        video_config = VideoFrameConfig(
            image_keys=('left_wrist_0_rgb', 'right_wrist_0_rgb'),
            num_frames=2,
        )
        video_dataset = VideoFrameDataset(base_dataset, video_config)
    """

    def __init__(
        self,
        dataset: _data_loader.Dataset,
        config: VideoFrameConfig,
    ):
        self._dataset = dataset
        self._config = config

        # Try to get access to underlying HuggingFace dataset
        self._hf_dataset = None
        if hasattr(dataset, "hf_dataset"):
            self._hf_dataset = dataset.hf_dataset
        elif hasattr(dataset, "_dataset") and hasattr(dataset._dataset, "hf_dataset"):
            self._hf_dataset = dataset._dataset.hf_dataset

        if self._hf_dataset is None:
            raise ValueError(
                "VideoFrameDataset requires a LeRobotDataset with hf_dataset attribute"
            )

    def __getitem__(self, index: int) -> dict:
        """Get a sample with dynamically loaded video frames.

        Performance note:
            Earlier implementations called ``hf_dataset[idx]`` once per
            ``(image_key, frame_offset)`` pair (i.e. ``len(image_keys) * num_frames``
            times). Each such call decodes ALL image columns of that row, so the
            CPU cost grew as ``len(image_keys)**2 * num_frames``.

            This version caches each unique ``idx`` row at most once and reuses
            the base dataset's already-loaded current frame, reducing the work to
            at most ``num_frames`` row reads per sample (4x fewer reads for the
            4-image-stream HeadView+Depth Pi0Mem setup).
        """
        # Load the current frame (this also decodes the current row's images once).
        data = dict(self._dataset[index])

        if "index" not in data:
            data["index"] = index

        current_index = int(data.get("index", index))
        current_episode = int(data.get("episode_index", -1))
        current_frame_idx = int(data.get("frame_index", -1))

        # If we don't have episode info, try to get it from dataset
        if current_episode < 0 or current_frame_idx < 0:
            try:
                row = self._hf_dataset[int(current_index)]
                current_episode = int(row.get("episode_index", -1))
                current_frame_idx = int(row.get("frame_index", -1))
                data["episode_index"] = current_episode
                data["frame_index"] = current_frame_idx
            except Exception:
                pass

        # Compute target indices for historical frames
        num_frames = self._config.num_frames
        stride = self._config.frame_stride

        # We want to load: current, current-stride, current-2*stride, ...
        # Then reverse so that index 0 is the oldest, last is current
        target_indices = [
            current_index - i * stride for i in range(num_frames - 1, -1, -1)
        ]

        # ------------------------------------------------------------------
        # Fetch each unique historical row at most once and cache it.
        # The current index is intentionally NOT fetched here — we reuse
        # the already-loaded ``data`` from ``self._dataset[index]`` instead.
        # ------------------------------------------------------------------
        unique_offset_indices = {idx for idx in target_indices if idx >= 0 and idx != current_index}
        rows_cache: dict[int, dict | None] = {}
        for idx in unique_offset_indices:
            try:
                row = self._hf_dataset[int(idx)]
                row_episode = int(row.get("episode_index", -1))
                if row_episode != current_episode and current_episode >= 0:
                    # Crossed episode boundary -> treat as missing so we fall back
                    # to padding (repeat first valid or zero) below.
                    rows_cache[idx] = None
                else:
                    rows_cache[idx] = row
            except (IndexError, KeyError, Exception):
                rows_cache[idx] = None

        # Load frames for each image key, reusing the cached rows.
        for img_key in self._config.image_keys:
            frames: list[np.ndarray | None] = []

            for idx in target_indices:
                if idx < 0:
                    frames.append(None)
                    continue
                # Reuse the already-loaded current frame to avoid an extra
                # row decode / image deserialization for the current step.
                if idx == current_index and img_key in data and data[img_key] is not None:
                    frames.append(_parse_image(data[img_key]))
                    continue
                row = rows_cache.get(idx)
                if row is None:
                    frames.append(None)
                    continue
                frame = row.get(img_key)
                if frame is None:
                    frames.append(None)
                    continue
                frames.append(_parse_image(frame))

            # Handle padding for None frames
            valid_frames = [f for f in frames if f is not None]
            if not valid_frames:
                # Create a zero frame
                if img_key in data and data[img_key] is not None:
                    template = _parse_image(data[img_key])
                    zero_frame = np.zeros_like(template)
                else:
                    zero_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                frames = [zero_frame] * num_frames
            else:
                first_valid = valid_frames[0]
                filled_frames = []
                for f in frames:
                    if f is None:
                        if self._config.padding_mode == "repeat":
                            filled_frames.append(first_valid)
                        else:  # zero
                            filled_frames.append(np.zeros_like(first_valid))
                    else:
                        filled_frames.append(f)
                frames = filled_frames

            for i, frame in enumerate(frames):
                key = f"{img_key}_{i}"
                data[key] = frame

        return data

    def __len__(self) -> int:
        return len(self._dataset)


class VideoFrameTransformDataset(_data_loader.Dataset):
    """Dataset that applies transforms after dynamic frame loading.

    This combines VideoFrameDataset with transform application in a single
    dataset class, making it easier to use with the standard data loader.
    """

    def __init__(
        self,
        dataset: _data_loader.Dataset,
        config: VideoFrameConfig,
        transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._video_dataset = VideoFrameDataset(dataset, config)
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: int) -> dict:
        data = self._video_dataset[index]
        return self._transform(data)

    def __len__(self) -> int:
        return len(self._video_dataset)


def create_video_frame_transform(
    dataset: Any,
    image_keys: Sequence[str],
    num_frames: int = 2,
    frame_stride: int = 1,
    padding_mode: str = "repeat",
    build_video_tensor: bool = False,
    output_keys: dict[str, str] | None = None,
) -> _transforms.DataTransformFn:
    """Create a transform that loads video frames dynamically.

    This is a factory function that creates the appropriate transform
    based on whether you want individual frames or stacked video tensors.

    Args:
        dataset: The LeRobot dataset to load frames from
        image_keys: List of image keys to expand
        num_frames: Number of frames per key
        frame_stride: Stride between frames
        padding_mode: 'repeat' or 'zero'
        build_video_tensor: If True, stack frames into [T, H, W, C] tensors
        output_keys: Optional mapping for output key names

    Returns:
        A DataTransformFn that loads video frames
    """
    from openpi import transforms_video as _transforms_video

    if build_video_tensor:
        return _transforms_video.LoadAndBuildVideo(
            dataset=dataset,
            image_keys=tuple(image_keys),
            num_frames=num_frames,
            frame_stride=frame_stride,
            padding_mode=padding_mode,
            output_keys=output_keys,
        )
    else:
        return _transforms_video.LoadVideoFrames(
            dataset=dataset,
            image_keys=tuple(image_keys),
            num_frames=num_frames,
            frame_stride=frame_stride,
            padding_mode=padding_mode,
        )
