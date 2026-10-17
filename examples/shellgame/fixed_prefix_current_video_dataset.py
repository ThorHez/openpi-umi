"""Video wrapper emitting a fixed episode prefix plus one dynamic current frame.

For a ShellGame row at frame ``t >= 59``, each image stream is emitted as

    [frame 0, frame 1, ..., frame 59, frame t]

The first 60 frames therefore preserve the proven visual tracker's exact
temporal contract, while the final frame supplies the action expert with the
live observation needed for closed-loop replanning.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openpi.training import data_loader as _data_loader
from openpi.training.mem import video_dataset as _video


class FixedPrefixCurrentVideoDataset(_data_loader.Dataset):
    """Load frames 0..59 from the row's episode and append the current row."""

    PREFIX_FRAMES = 60

    def __init__(self, dataset: _data_loader.Dataset, config: _video.VideoFrameConfig):
        if config.num_frames != self.PREFIX_FRAMES + 1:
            raise ValueError(
                "FixedPrefixCurrentVideoDataset requires num_frames=61; "
                f"got {config.num_frames}"
            )
        if config.num_future_frames != 0:
            raise ValueError("Fixed-prefix action training does not support future image frames")
        self._dataset = dataset
        self._config = config
        self._hf_dataset = None
        if hasattr(dataset, "hf_dataset"):
            self._hf_dataset = dataset.hf_dataset
        elif hasattr(dataset, "_dataset") and hasattr(dataset._dataset, "hf_dataset"):
            self._hf_dataset = dataset._dataset.hf_dataset
        if self._hf_dataset is None:
            raise ValueError("FixedPrefixCurrentVideoDataset requires a LeRobot hf_dataset")

        columns = set(getattr(self._hf_dataset, "column_names", ()) or ())
        self._source_keys = {
            key: _video._resolve_image_source_key(key, columns) or key  # noqa: SLF001
            for key in config.image_keys
        }
        history_columns = list(dict.fromkeys(("episode_index", *self._source_keys.values())))
        self._history_hf_dataset = self._hf_dataset
        if columns and all(column in columns for column in history_columns):
            self._history_hf_dataset = self._hf_dataset.select_columns(history_columns)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = dict(self._dataset[index])
        data.setdefault("index", index)
        current_index = _video._to_int(data.get("index"), default=int(index))  # noqa: SLF001
        current_episode = _video._to_int(data.get("episode_index"), default=-1)  # noqa: SLF001
        current_frame = _video._to_int(data.get("frame_index"), default=-1)  # noqa: SLF001
        if current_episode < 0 or current_frame < 0:
            row = self._hf_dataset[current_index]
            current_episode = _video._to_int(row.get("episode_index"), default=-1)  # noqa: SLF001
            current_frame = _video._to_int(row.get("frame_index"), default=-1)  # noqa: SLF001
            data["episode_index"] = current_episode
            data["frame_index"] = current_frame
        if current_frame < self.PREFIX_FRAMES - 1:
            raise ValueError(
                f"Fixed-prefix action sample requires frame_index>=59, got {current_frame}"
            )

        episode_start = current_index - current_frame
        target_indices = [episode_start + offset for offset in range(self.PREFIX_FRAMES)]
        target_indices.append(current_index)
        unique_history = sorted(set(target_indices[:-1]))
        batch_rows = self._history_hf_dataset[unique_history]
        rows = {
            row_index: {
                key: _video._batch_item(value, position)  # noqa: SLF001
                for key, value in batch_rows.items()
            }
            for position, row_index in enumerate(unique_history)
        }
        for row_index, row in rows.items():
            row_episode = _video._to_int(row.get("episode_index"), default=-1)  # noqa: SLF001
            if row_episode != current_episode:
                raise RuntimeError(
                    f"Fixed prefix crossed episode boundary at global row {row_index}: "
                    f"expected episode {current_episode}, got {row_episode}"
                )

        frame_valid_masks = {}
        for image_key in self._config.image_keys:
            source_key = self._source_keys[image_key]
            current_source = _video._resolve_image_source_key(image_key, data) or source_key  # noqa: SLF001
            frames = []
            for row_index in target_indices[:-1]:
                raw = rows[row_index].get(source_key)
                if raw is None:
                    resolved = _video._resolve_image_source_key(image_key, rows[row_index])  # noqa: SLF001
                    raw = rows[row_index].get(resolved) if resolved is not None else None
                if raw is None:
                    raise KeyError(f"Missing {image_key!r} at global row {row_index}")
                frames.append(_video._parse_image(raw))  # noqa: SLF001

            current_raw = data.get(current_source)
            if current_raw is None:
                current_row = self._hf_dataset[current_index]
                current_raw = current_row.get(source_key)
            if current_raw is None:
                raise KeyError(f"Missing current {image_key!r} at global row {current_index}")
            frames.append(_video._parse_image(current_raw))  # noqa: SLF001

            if len(frames) != self.PREFIX_FRAMES + 1:
                raise RuntimeError(f"Expected 61 frames, constructed {len(frames)}")
            for offset, frame in enumerate(frames):
                data[f"{image_key}_{offset}"] = frame
            frame_valid_masks[image_key] = np.ones(len(frames), dtype=np.bool_)

        data["video_frame_valid_mask"] = frame_valid_masks
        return data

    def __len__(self) -> int:
        return len(self._dataset)
