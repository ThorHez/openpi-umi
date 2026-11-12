"""LeRobot frame indexing and same-episode pairs for stage-progress learning."""

from pathlib import Path
import random

import numpy as np

from openpi.models import model
from openpi.training import data_loader


_SOURCE_COLUMNS = ("index", "episode_index", "frame_index", "task_index", "stage_progress_gt")


def _integer_column(frames, key, size):
    values = np.asarray(frames[key])
    if (values.shape not in {(size,), (size, 1)} or values.dtype.kind not in "iu"
            or np.any(values < 0) or np.any(values > np.iinfo(np.int64).max)):
        raise ValueError(f"阶段进度数据的 {key} 必须是非负整数标量")
    return values.astype(np.int64, copy=False).reshape(-1)


class StageProgressSource:
    """Random access to transformed observations and the original parquet row IDs."""

    def __init__(self, config, *, skip_norm_stats=False):
        if hasattr(config.data, "datasets"):
            raise ValueError("阶段进度训练只支持一组 LeRobot 数据")
        resolved = config.data.create(config.assets_dirs, config.model)
        self.data_config = data_loader._apply_global_normalization(resolved, config)
        if not self.data_config.repo_id or self.data_config.rlds_data_dir is not None:
            raise ValueError("阶段进度训练需要本地 LeRobot 数据集")
        self.root = Path(self.data_config.repo_id).resolve(strict=True)
        dataset = data_loader.create_torch_dataset(self.data_config, config.model.action_horizon, config.model)
        raw_dataset = dataset._dataset if isinstance(dataset, data_loader.TransformedDataset) else dataset
        frames = raw_dataset.hf_dataset
        size = len(frames)
        missing = set(_SOURCE_COLUMNS) - set(frames.column_names)
        if size == 0 or missing:
            raise ValueError(f"阶段进度数据为空或缺少物理列：{', '.join(sorted(missing))}")
        columns = frames.with_format("numpy", columns=list(_SOURCE_COLUMNS))[:]
        self.indices = _integer_column(columns, "index", size)
        self.task_indices = _integer_column(columns, "task_index", size)
        episodes = _integer_column(columns, "episode_index", size)
        frame_indices = _integer_column(columns, "frame_index", size)
        if np.unique(self.indices).size != size:
            raise ValueError("阶段进度数据的全局 index 必须唯一")
        progress = np.asarray(columns["stage_progress_gt"])
        if (progress.shape not in {(size,), (size, 1)} or progress.dtype.kind not in "iuf"
                or not np.all(np.isfinite(progress)) or np.any((progress < 0) | (progress > 1))):
            raise ValueError("stage_progress_gt 必须是 [0, 1] 内的有限数值标量")
        self.progress = progress.astype(np.float32, copy=False).reshape(-1)

        ordered_rows = np.lexsort((frame_indices, episodes))
        boundaries = np.flatnonzero(np.diff(episodes[ordered_rows])) + 1
        self.episodes = tuple(np.split(ordered_rows, boundaries))
        if any(np.any(np.diff(frame_indices[rows]) != 1) for rows in self.episodes):
            raise ValueError("阶段进度数据的每条轨迹必须有连续且唯一的 frame_index")
        self._dataset = data_loader.transform_dataset(dataset, self.data_config, skip_norm_stats=skip_norm_stats)
        self._image_keys = config.model.image_keys
        self._depth_keys = config.model.depth_keys

    def observation(self, row):
        sample = self._dataset[int(row)]
        return self._observation_fields(sample)

    def observation_with_actions(self, row):
        sample = self._dataset[int(row)]
        return self._observation_fields(sample), sample["actions"]

    @staticmethod
    def _observation_fields(sample):
        required = {"image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask"}
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"阶段进度观测缺少字段：{', '.join(sorted(missing))}")
        return {key: sample[key] for key in sorted(required)}

    def observation_batch(self, rows, observations):
        batch = data_loader._collate_fn(
            [observations[int(row)] for row in rows], image_keys=self._image_keys, depth_keys=self._depth_keys
        )
        return model.Observation.from_dict(batch)


class StageProgressPairs:
    """Sample a uniform, distinct reference within each current frame's episode."""

    def __init__(self, source, *, include_actions=False):
        self.source = source
        self._include_actions = include_actions
        episodes = [rows for rows in source.episodes if len(rows) > 1]
        if not episodes:
            raise ValueError("阶段进度训练至少需要一条含两帧的轨迹")
        self._rows = np.concatenate(episodes)
        sizes = np.asarray([len(rows) for rows in episodes])
        self._sizes = np.repeat(sizes, sizes)
        self._starts = np.repeat(np.cumsum(sizes) - sizes, sizes)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, position):
        first = int(self._starts[position])
        size = int(self._sizes[position])
        offset = random.randrange(size - 1)
        reference_position = first + offset
        if reference_position >= position:
            reference_position += 1
        reference_row = self._rows[reference_position]
        current_row = self._rows[position]
        pair = {
            "reference": self.source.observation(reference_row),
            "target": self.source.progress[current_row] - self.source.progress[reference_row],
        }
        if self._include_actions:
            pair["current"], pair["actions"] = self.source.observation_with_actions(current_row)
        else:
            pair["current"] = self.source.observation(current_row)
        return pair


class StageProgressDataLoader:
    def __init__(self, source, config, sharding):
        self._source = source
        self._loader = data_loader.TorchDataLoader(
            StageProgressPairs(source),
            image_keys=config.model.image_keys,
            depth_keys=config.model.depth_keys,
            local_batch_size=config.batch_size,
            sharding=sharding,
            shuffle=True,
            num_workers=config.num_workers,
            seed=config.seed,
        )

    def data_config(self):
        return self._source.data_config

    def __iter__(self):
        for batch in self._loader:
            yield (
                model.Observation.from_dict(batch["reference"]),
                model.Observation.from_dict(batch["current"]),
                batch["target"],
            )
