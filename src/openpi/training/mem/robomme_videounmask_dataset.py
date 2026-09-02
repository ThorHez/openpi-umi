"""Frozen-feature dataset for VideoUnmask semantic-memory pretraining."""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import torch

COLOR_TO_ID = {"red": 0, "green": 1, "blue": 2}


class VideoUnmaskFeatureDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        feature_h5_path: str | pathlib.Path,
        labels_path: str | pathlib.Path,
        *,
        episode_indices: list[int],
    ):
        self.feature_h5_path = pathlib.Path(feature_h5_path).expanduser().resolve()
        self.labels_path = pathlib.Path(labels_path).expanduser().resolve()
        if not self.feature_h5_path.is_file():
            raise FileNotFoundError(self.feature_h5_path)
        payload = json.loads(self.labels_path.read_text(encoding="utf-8"))
        by_index = {int(item["episode_index"]): item for item in payload["episodes"]}
        missing = [index for index in episode_indices if index not in by_index]
        if missing:
            raise ValueError(f"Unknown episode indices: {missing[:10]}")
        self.episodes = [by_index[int(index)] for index in episode_indices]
        self._features: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.episodes)

    def _handle(self) -> h5py.File:
        if self._features is None:
            self._features = h5py.File(self.feature_h5_path, "r")
        return self._features

    def __getitem__(self, item: int):
        metadata = self.episodes[item]
        feature_episode = self._handle()[metadata["episode_name"]]
        return {
            "episode_index": np.int32(metadata["episode_index"]),
            "demo_patch_tokens": feature_episode["demo_patch_tokens"][()],
            "prompt_tokens": feature_episode["prompt_tokens"][0],
            "prompt_mask": feature_episode["prompt_mask"][0],
            "frame_mask": np.ones((len(metadata["demo_indices"]),), dtype=np.bool_),
            "target_point": np.asarray(metadata["target_point_normalized_yx"], dtype=np.float32),
            "target_cell": np.int32(
                min(int(metadata["target_point_yx"][0]) // 32, 7) * 8
                + min(int(metadata["target_point_yx"][1]) // 32, 7)
            ),
            "target_color": np.int32(COLOR_TO_ID[metadata["target_color"]]),
        }
