"""Statistics-only Parquet projection; never construct a LeRobot/image loader.

Only explicitly declared numeric state/action leaves and, for per-frame actions,
three scalar temporal indices are read. Packed action chunks and state history
retain their stored axes; per-frame actions clamp at each episode's last row.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import dataclasses
import hashlib
import math
from pathlib import Path
import re

from openpi.training.composed_processing import (
    _read_lerobot_metadata, composition_runtime_contract, normalize_composition,
)


def _fail(message):
    raise ValueError("NORM_NUMERIC_ONLY_UNSUPPORTED: " + message)


@dataclasses.dataclass(frozen=True)
class NumericStatisticsConfig:
    repo_id: str
    composition: dict
    action_horizon: int
    model_action_dim: int
    effective_fps: float
    total_frames: int
    packed_actions: bool

    @property
    def columns(self):
        return list(dict.fromkeys([field["key"] for field in self.composition["state_fields"]]
                    + [self.composition["action_key"]] + self.temporal_columns))

    @property
    def temporal_columns(self):
        return [] if self.packed_actions else ["episode_index", "frame_index", "timestamp"]


def build_numeric_statistics_config(factory, model_config, *, effective_fps=None):
    # A generic create()/repack factory cannot prove which fields its transforms
    # need. Reject it before executing user transforms or any dataset loader.
    if type(factory).__name__ != "ComposableDataConfig":
        _fail("该处理器没有可证明的纯数值统计接口，请使用字段组合处理器；不会回退读取图像")
    composition = normalize_composition(factory.composition, norm_stats=True)
    composition_runtime_contract(factory, model_config, norm_stats=True)
    info, _ = _read_lerobot_metadata(factory.repo_id)
    for key in [field["key"] for field in composition["state_fields"]] + [composition["action_key"]]:
        feature = info["features"].get(key)
        if not isinstance(feature, dict) or re.fullmatch(r"(?:float|int|uint)(?:8|16|32|64)?|double|bfloat16", str(feature.get("dtype", ""))) is None:
            _fail("统计字段必须在元信息中明确登记为数值，不能选择图像或视频：" + key)
    frames = info.get("total_frames")
    if type(frames) is not int or frames < 1:
        _fail("数据集缺少有效的 total_frames")
    fps = info.get("fps") if effective_fps is None else effective_fps
    if type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 1000:
        _fail("统计数据集需要有效的登记或元信息 FPS")
    width = sum(row["width"] for row in composition["action_layout"])
    shape = info["features"][composition["action_key"]].get("shape")
    packed = shape == [model_config.action_horizon, width]
    if not packed and shape != [width]:
        _fail("动作元信息必须是匹配预测长度的 [H,D] 或逐帧 [D]")
    for field in composition["state_fields"]:
        shape = info["features"][field["key"]].get("shape")
        if shape != [composition["state_history"], field["width"]] and not (
                composition["state_history"] == 1 and shape == [field["width"]]):
            _fail("状态历史和字段宽度必须与原始数值字段一致：" + field["key"])
    return NumericStatisticsConfig(factory.repo_id, composition, model_config.action_horizon,
        model_config.action_dim, float(fps), frames, packed)


def numeric_io_evidence(config):
    source = Path(__file__)
    return {"schema_version": 1, "reader": "pyarrow_parquet_projection", "columns": config.columns,
        "state_columns": [field["key"] for field in config.composition["state_fields"]],
        "action_column": config.composition["action_key"], "temporal_columns": config.temporal_columns,
        "action_sequence": "packed" if config.packed_actions else "episode_clamp",
        "image_columns": [], "video_decode": False,
        "source": {"path": "src/openpi/training/numeric_stats.py", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}


class NumericParquetDataset:
    """Read numeric row groups lazily with a bounded two-group in-memory cache."""

    def __init__(self, config):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.config = config
        self._groups, self._starts, self._cache, self._dtypes = [], [], OrderedDict(), {}
        self._size = 0
        root = (Path(config.repo_id) / "data").resolve(strict=True)
        files = sorted(root.rglob("*.parquet"))
        if not files:
            _fail("数据集缺少 Parquet 数值数据")
        for path in files:
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
                _fail("Parquet 数据路径不能逃逸登记目录")
            parquet = pq.ParquetFile(path)
            try:
                schema = parquet.schema_arrow
                for column in config.columns:
                    if column not in schema.names:
                        _fail("Parquet 缺少显式投影字段：" + column)
                    dtype = schema.field(column).type
                    while pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
                        dtype = dtype.value_type
                    if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
                        _fail("Parquet 投影字段必须为实数，不能读取图像/视频结构或二进制：" + column)
                    self._dtypes[column] = dtype.to_pandas_dtype()
                for group in range(parquet.num_row_groups):
                    rows = parquet.metadata.row_group(group).num_rows
                    if rows:
                        self._starts.append(self._size)
                        self._groups.append((str(path), group, rows))
                        self._size += rows
            finally:
                parquet.close()
        if self._size != config.total_frames:
            _fail("Parquet 行数与登记 total_frames 不一致，不能遗漏数据")
        self._episode_end = self._temporal_indices() if not config.packed_actions else None

    def __getstate__(self):
        return {**self.__dict__, "_cache": OrderedDict()}

    def _read_group(self, index, columns):
        import pyarrow.parquet as pq
        path, group, _ = self._groups[index]
        parquet = pq.ParquetFile(path)
        try:
            table = parquet.read_row_group(group, columns=list(columns), use_threads=False)
        finally:
            parquet.close()
        if set(table.column_names) != set(columns) or len(table.column_names) != len(columns):
            _fail("Parquet 投影未严格返回声明的数值列")
        return table.select(list(columns))

    def _temporal_indices(self):
        import numpy as np
        episodes, frames, timestamps = [], [], []
        for index in range(len(self._groups)):
            table = self._read_group(index, self.config.temporal_columns)
            episodes.extend(table["episode_index"].to_pylist())
            frames.extend(table["frame_index"].to_pylist())
            timestamps.extend(table["timestamp"].to_pylist())
        end = np.empty(self._size, dtype=np.int64)
        start, seen = 0, set()
        for index in range(self._size):
            episode, frame, timestamp = episodes[index], frames[index], timestamps[index]
            if type(episode) is not int or episode < 0 or type(frame) is not int or frame < 0 or type(timestamp) not in (int, float) or not math.isfinite(timestamp):
                _fail("动作采样需要有效的 episode_index/frame_index/timestamp 数值")
            if index == 0 or episode != episodes[index - 1]:
                if episode in seen:
                    _fail("同一 episode 的行必须在原始索引中连续，不能猜测或重排")
                if frame != 0:
                    _fail("episode 的逐帧索引必须从 0 开始")
                if index:
                    end[start:index] = index - 1
                seen.add(episode); start = index
            elif frame != frames[index - 1] + 1 or not math.isclose(timestamp - timestamps[index - 1], 1 / self.config.effective_fps, rel_tol=0, abs_tol=1e-4):
                _fail("逐帧动作索引或时间戳与生效 FPS 不一致，不自动重采样")
        end[start:] = self._size - 1
        return end

    def _row(self, index, columns=None):
        group = bisect_right(self._starts, index) - 1
        if group not in self._cache:
            self._cache[group] = self._read_group(group, self.config.columns)
            if len(self._cache) > 2:
                self._cache.popitem(last=False)
        table = self._cache[group]
        self._cache.move_to_end(group)
        offset = index - self._starts[group]
        return {key: table[key][offset].as_py() for key in (self.config.columns if columns is None else columns)}

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        import numpy as np
        if not 0 <= index < self._size:
            raise IndexError(index)
        config, spec = self.config, self.config.composition
        row = self._row(index)
        states = []
        for field in spec["state_fields"]:
            value = np.asarray(row[field["key"]], dtype=self._dtypes[field["key"]])
            if spec["state_history"] == 1 and value.shape == (field["width"],):
                value = value.reshape(1, field["width"])
            if value.shape != (spec["state_history"], field["width"]) or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                _fail("状态必须是匹配声明历史长度的有限数值：" + field["key"])
            if field["kind"] == "gripper":
                self._check_gripper(value)
            states.append(value)
        if config.packed_actions:
            actions = np.asarray(row[spec["action_key"]], dtype=self._dtypes[spec["action_key"]])
        else:
            actions = np.asarray([self._row(min(index + step, int(self._episode_end[index])), [spec["action_key"]])[spec["action_key"]]
                                  for step in range(config.action_horizon)], dtype=self._dtypes[spec["action_key"]])
        width = sum(block["width"] for block in spec["action_layout"])
        if actions.shape != (config.action_horizon, width) or actions.dtype.kind not in "iuf" or not np.isfinite(actions).all():
            _fail("动作必须是匹配预测长度/完整宽度的有限数值")
        start = 0
        for block in spec["action_layout"]:
            if block["kind"] == "gripper":
                self._check_gripper(actions[:, start:start + block["width"]])
            start += block["width"]
        return {"state": np.concatenate(states, axis=-1), "actions": actions}

    def _check_gripper(self, value):
        import numpy as np
        limits = self.config.composition["gripper_range"]
        if limits is not None and (np.any(value < limits[0]) or np.any(value > limits[1])):
            _fail("显式夹爪字段超出声明范围")
