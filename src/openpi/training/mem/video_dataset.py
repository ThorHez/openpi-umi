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
        num_frames: Number of past+current frames to load per key (temporal horizon).
            The current frame is always the LAST of these.
        frame_stride: Stride between consecutive past frames (default: 1)
        padding_mode: How to handle episode boundaries: 'repeat' or 'zero'
        num_future_frames: Number of future frames appended AFTER the current
            frame (default 0, i.e. past-only — the original behavior). The
            resulting per-key frame layout is
            ``[oldest_past, ..., current, future_1, ..., future_F]``.
        future_frame_stride: Stride between consecutive future frames.
    """

    image_keys: tuple[str, ...]
    num_frames: int = 2
    frame_stride: int = 1
    padding_mode: str = "repeat"
    num_future_frames: int = 0
    future_frame_stride: int = 1
    layout: str = "sliding"
    fixed_prefix_frames: int = 0
    min_frame_index: int | None = None
    max_frame_index: int | None = None

    @property
    def total_frames(self) -> int:
        """Total frames emitted per image key (past + current + future)."""
        return self.num_frames + self.num_future_frames


class FixedPrefixCurrentVideoDataset(_data_loader.Dataset):
    """Emit an episode's fixed prefix followed by one dynamic current frame.

    This is the production dataset contract for semantic-memory policies.  A
    sample at episode frame ``t`` becomes ``[frame 0, ..., frame P-1, frame t]``.
    Optional frame bounds are applied once at construction so early rows that
    cannot supply the complete prefix never reach the training sampler.
    """

    def __init__(self, dataset: _data_loader.Dataset, config: VideoFrameConfig):
        if config.layout != "fixed_prefix_current":
            raise ValueError(f"Expected layout='fixed_prefix_current', got {config.layout!r}")
        if config.fixed_prefix_frames <= 0:
            raise ValueError("fixed_prefix_frames must be positive")
        if config.num_frames != config.fixed_prefix_frames + 1:
            raise ValueError(
                "Fixed-prefix layout requires num_frames=fixed_prefix_frames+1; "
                f"got {config.num_frames} and {config.fixed_prefix_frames}"
            )
        if config.num_future_frames != 0:
            raise ValueError("Fixed-prefix layout does not support future image frames")

        self._dataset = dataset
        self._config = config
        self._hf_dataset = None
        if hasattr(dataset, "hf_dataset"):
            self._hf_dataset = dataset.hf_dataset
        else:
            inner_dataset = getattr(dataset, "_dataset", None)
            if inner_dataset is not None and hasattr(inner_dataset, "hf_dataset"):
                self._hf_dataset = inner_dataset.hf_dataset
        if self._hf_dataset is None:
            raise ValueError("FixedPrefixCurrentVideoDataset requires a LeRobot hf_dataset")

        columns = set(getattr(self._hf_dataset, "column_names", ()) or ())
        if "frame_index" not in columns:
            raise ValueError("Fixed-prefix layout requires a frame_index column")
        frame_indices = np.asarray(self._hf_dataset["frame_index"], dtype=np.int64)
        min_frame = config.min_frame_index
        if min_frame is None:
            min_frame = config.fixed_prefix_frames - 1
        max_frame = config.max_frame_index
        eligible = frame_indices >= min_frame
        if max_frame is not None:
            eligible &= frame_indices <= max_frame
        self._sample_indices = np.flatnonzero(eligible).astype(np.int64)
        if self._sample_indices.size == 0:
            raise ValueError(
                f"Fixed-prefix frame range [{min_frame}, {max_frame}] selected no rows"
            )

        self._source_keys = {
            key: _resolve_image_source_key(key, columns) or key for key in config.image_keys
        }
        history_columns = list(dict.fromkeys(("episode_index", *self._source_keys.values())))
        self._history_hf_dataset = self._hf_dataset
        if columns and all(column in columns for column in history_columns):
            self._history_hf_dataset = self._hf_dataset.select_columns(history_columns)

    @property
    def sample_indices(self) -> np.ndarray:
        """Indices into the underlying full LeRobot dataset."""
        return self._sample_indices

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = int(self._sample_indices[int(index)])
        data = dict(self._dataset[source_index])
        data.setdefault("index", source_index)
        current_index = _to_int(data.get("index"), default=source_index)
        current_episode = _to_int(data.get("episode_index"), default=-1)
        current_frame = _to_int(data.get("frame_index"), default=-1)
        if current_episode < 0 or current_frame < 0:
            row = self._hf_dataset[current_index]
            current_episode = _to_int(row.get("episode_index"), default=-1)
            current_frame = _to_int(row.get("frame_index"), default=-1)
            data["episode_index"] = current_episode
            data["frame_index"] = current_frame
        if current_frame < self._config.fixed_prefix_frames - 1:
            raise ValueError(
                "Fixed-prefix sample is earlier than the complete history: "
                f"frame_index={current_frame}, prefix={self._config.fixed_prefix_frames}"
            )

        episode_start = current_index - current_frame
        prefix_indices = [
            episode_start + offset for offset in range(self._config.fixed_prefix_frames)
        ]
        target_indices = [*prefix_indices, current_index]
        batch_rows = self._history_hf_dataset[prefix_indices]
        rows = {
            row_index: {key: _batch_item(value, position) for key, value in batch_rows.items()}
            for position, row_index in enumerate(prefix_indices)
        }
        for row_index, row in rows.items():
            row_episode = _to_int(row.get("episode_index"), default=-1)
            if row_episode != current_episode:
                raise RuntimeError(
                    f"Fixed prefix crossed an episode boundary at row {row_index}: "
                    f"expected {current_episode}, got {row_episode}"
                )

        frame_valid_masks = {}
        for image_key in self._config.image_keys:
            source_key = self._source_keys[image_key]
            current_source = _resolve_image_source_key(image_key, data) or source_key
            frames = []
            for row_index in prefix_indices:
                raw = rows[row_index].get(source_key)
                if raw is None:
                    resolved = _resolve_image_source_key(image_key, rows[row_index])
                    raw = rows[row_index].get(resolved) if resolved is not None else None
                if raw is None:
                    raise KeyError(f"Missing {image_key!r} at global row {row_index}")
                frames.append(_parse_image(raw))

            current_raw = data.get(current_source)
            if current_raw is None:
                current_raw = self._hf_dataset[current_index].get(source_key)
            if current_raw is None:
                raise KeyError(f"Missing current {image_key!r} at global row {current_index}")
            frames.append(_parse_image(current_raw))

            for offset, frame in enumerate(frames):
                data[f"{image_key}_{offset}"] = frame
            frame_valid_masks[image_key] = np.ones(len(target_indices), dtype=np.bool_)

        data["video_frame_valid_mask"] = frame_valid_masks
        return data

    def __len__(self) -> int:
        return int(self._sample_indices.size)


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


def _to_int(value, default: int = -1) -> int:
    """Convert scalar-like dataset values, including torch/np scalars, to int."""
    if value is None:
        return default
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except Exception:
            return default
    try:
        return int(value)
    except Exception:
        return default


def _batch_item(values, position: int):
    """Extract one row value from a batched HuggingFace dataset column."""
    if isinstance(values, (list, tuple)):
        return values[position]
    try:
        return values[position]
    except Exception:
        return values


def _image_key_candidates(img_key: str) -> tuple[str, ...]:
    """Possible dataset column / sample keys for a configured image stream."""
    candidates = [img_key]
    if img_key.startswith("observation."):
        candidates.append(img_key[len("observation.") :])
    else:
        candidates.append(f"observation.{img_key}")
    # LeRobot camera convention used by some datasets.
    short = img_key.split(".")[-1]
    candidates.append(f"observation.images.{short}")
    if short != img_key:
        candidates.append(f"observation.images.{img_key}")
    # Preserve order while dropping duplicates.
    return tuple(dict.fromkeys(candidates))


def _resolve_image_source_key(img_key: str, available) -> str | None:
    """Resolve ``img_key`` against a sample dict or HF column-name set."""
    for candidate in _image_key_candidates(img_key):
        if candidate in available:
            return candidate
    return None


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

        # Map configured output stream names -> actual keys present in the
        # LeRobot / HF sample. Shellgame (and many LeRobot v2 datasets) store
        # cameras as ``observation.<name>`` while configs often use bare
        # ``<name>``; without this remap every frame is treated as missing and
        # padded to zeros (which silently collapses past/future to identical
        # blank images).
        column_names = set(getattr(self._hf_dataset, "column_names", ()) or ())
        self._source_keys: dict[str, str] = {}
        for img_key in self._config.image_keys:
            source = _resolve_image_source_key(img_key, column_names)
            if source is None:
                # Fall back to the configured name; __getitem__ may still find
                # it on the transformed LeRobot sample even if HF columns differ.
                source = img_key
            self._source_keys[img_key] = source

        self._history_hf_dataset = self._hf_dataset
        try:
            history_columns = list(
                dict.fromkeys(("episode_index", *self._source_keys.values()))
            )
            if column_names and all(column in column_names for column in history_columns):
                # Historical-frame lookup only needs episode_index and image streams.
                # Keeping extra state/action/prompt columns out avoids unnecessary
                # HuggingFace transform work for each random historical row.
                self._history_hf_dataset = self._hf_dataset.select_columns(history_columns)
        except Exception:
            self._history_hf_dataset = self._hf_dataset

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

        current_index = _to_int(data.get("index", index), default=int(index))
        current_episode = _to_int(data.get("episode_index"), default=-1)
        current_frame_idx = _to_int(data.get("frame_index"), default=-1)

        # If we don't have episode info, try to get it from dataset
        if current_episode < 0 or current_frame_idx < 0:
            try:
                row = self._hf_dataset[int(current_index)]
                current_episode = _to_int(row.get("episode_index"), default=-1)
                current_frame_idx = _to_int(row.get("frame_index"), default=-1)
                data["episode_index"] = current_episode
                data["frame_index"] = current_frame_idx
            except Exception:
                pass

        # Compute target indices for historical (and optionally future) frames
        num_frames = self._config.num_frames
        stride = self._config.frame_stride
        num_future = self._config.num_future_frames
        future_stride = self._config.future_frame_stride
        total_frames = self._config.total_frames

        # Past+current: current, current-stride, current-2*stride, ...
        # reversed so that index 0 is the oldest and index num_frames-1 is
        # current. Future frames (if any) are appended after the current one:
        # [oldest_past, ..., current, current+fs, current+2*fs, ...]
        target_indices = [
            current_index - i * stride for i in range(num_frames - 1, -1, -1)
        ]
        target_indices += [
            current_index + (j + 1) * future_stride for j in range(num_future)
        ]

        # ------------------------------------------------------------------
        # Fetch each unique offset row at most once and cache it.
        # The current index is intentionally NOT fetched here — we reuse
        # the already-loaded ``data`` from ``self._dataset[index]`` instead.
        # Indices past the end of the dataset are treated as missing (they
        # fall back to padding below), same as negative past indices.
        # ------------------------------------------------------------------
        num_rows = len(self._hf_dataset)
        unique_offset_indices = sorted(
            {idx for idx in target_indices if 0 <= idx < num_rows and idx != current_index}
        )
        rows_cache: dict[int, dict | None] = {}

        try:
            batch_rows = self._history_hf_dataset[[int(idx) for idx in unique_offset_indices]]
            for position, idx in enumerate(unique_offset_indices):
                row = {key: _batch_item(value, position) for key, value in batch_rows.items()}
                row_episode = _to_int(row.get("episode_index"))
                if row_episode != current_episode and current_episode >= 0:
                    # Crossed episode boundary -> treat as missing so we fall back
                    # to padding (repeat first valid or zero) below.
                    rows_cache[idx] = None
                else:
                    rows_cache[idx] = row
        except Exception:
            rows_cache.clear()

        for idx in unique_offset_indices:
            if idx in rows_cache:
                continue
            try:
                row = self._history_hf_dataset[int(idx)]
                row_episode = _to_int(row.get("episode_index"))
                if row_episode != current_episode and current_episode >= 0:
                    # Crossed episode boundary -> treat as missing so we fall back
                    # to padding (repeat first valid or zero) below.
                    rows_cache[idx] = None
                else:
                    rows_cache[idx] = row
            except (IndexError, KeyError, Exception):
                rows_cache[idx] = None

        frame_valid_masks: dict[str, np.ndarray] = {}

        # Load frames for each image key, reusing the cached rows.
        for img_key in self._config.image_keys:
            source_key = self._source_keys[img_key]
            # LeRobot __getitem__ may expose a different key spelling than the
            # raw HF columns (e.g. after a repack). Prefer the live sample.
            current_source = _resolve_image_source_key(img_key, data) or source_key
            frames: list[np.ndarray | None] = []

            for idx in target_indices:
                if idx < 0 or idx >= num_rows:
                    frames.append(None)
                    continue
                # Reuse the already-loaded current frame to avoid an extra
                # row decode / image deserialization for the current step.
                if (
                    idx == current_index
                    and current_source in data
                    and data[current_source] is not None
                ):
                    frames.append(_parse_image(data[current_source]))
                    continue
                row = rows_cache.get(idx)
                if row is None:
                    frames.append(None)
                    continue
                frame = row.get(source_key)
                if frame is None and source_key != current_source:
                    frame = row.get(current_source)
                if frame is None:
                    # Last resort: any candidate present on this row.
                    resolved = _resolve_image_source_key(img_key, row)
                    frame = row.get(resolved) if resolved is not None else None
                if frame is None:
                    frames.append(None)
                    continue
                frames.append(_parse_image(frame))

            valid_mask = np.asarray([frame is not None for frame in frames], dtype=np.bool_)
            frame_valid_masks[img_key] = valid_mask

            # Handle padding for None frames
            valid_frames = [f for f in frames if f is not None]
            if not valid_frames:
                # Create a zero frame
                if current_source in data and data[current_source] is not None:
                    template = _parse_image(data[current_source])
                    zero_frame = np.zeros_like(template)
                else:
                    zero_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                frames = [zero_frame] * total_frames
            else:
                template = valid_frames[0]
                filled_frames = []
                for i, f in enumerate(frames):
                    if f is not None:
                        filled_frames.append(f)
                        continue
                    if self._config.padding_mode != "repeat":  # zero
                        filled_frames.append(np.zeros_like(template))
                        continue
                    # 'repeat': fill with the nearest valid frame. For leading
                    # (past) gaps the nearest valid frame is the first valid
                    # one (identical to the original past-only behavior); for
                    # trailing (future) gaps past the episode end it is the
                    # last valid frame (usually the current one).
                    prev_valid = next(
                        (frames[j] for j in range(i - 1, -1, -1) if frames[j] is not None), None
                    )
                    next_valid = next(
                        (frames[j] for j in range(i + 1, len(frames)) if frames[j] is not None), None
                    )
                    filled_frames.append(prev_valid if prev_valid is not None else next_valid)
                frames = filled_frames

            for i, frame in enumerate(frames):
                key = f"{img_key}_{i}"
                data[key] = frame

        data["video_frame_valid_mask"] = frame_valid_masks

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
