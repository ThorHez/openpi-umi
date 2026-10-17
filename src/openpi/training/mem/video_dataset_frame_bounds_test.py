import numpy as np

from openpi.training.mem.video_dataset import VideoFrameConfig
from openpi.training.mem.video_dataset import VideoFrameDataset


class _FakeHF:
    column_names = ("episode_index", "frame_index", "camera")

    def __init__(self):
        self.rows = [
            {
                "episode_index": episode,
                "frame_index": frame,
                "camera": np.full((2, 2, 3), episode * 10 + frame, dtype=np.uint8),
            }
            for episode in range(2)
            for frame in range(5)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        if isinstance(index, list):
            return {key: [self.rows[i][key] for i in index] for key in self.column_names}
        return self.rows[index]

    def select_columns(self, _columns):
        return self


class _FakeDataset:
    def __init__(self):
        self.hf_dataset = _FakeHF()

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        return dict(self.hf_dataset[index])


def test_sliding_frame_bounds_filter_rows_and_preserve_global_indices():
    dataset = VideoFrameDataset(
        _FakeDataset(),
        VideoFrameConfig(
            image_keys=("camera",),
            num_frames=1,
            min_frame_index=2,
            max_frame_index=3,
        ),
    )
    assert len(dataset) == 4
    assert dataset.sample_indices.tolist() == [2, 3, 7, 8]
    rows = [dataset[index] for index in range(len(dataset))]
    assert [row["frame_index"] for row in rows] == [2, 3, 2, 3]
    assert [row["index"] for row in rows] == [2, 3, 7, 8]
