"""Statistics-only Parquet projection; never construct a LeRobot/image loader.

Only explicitly declared numeric state/action leaves and, for per-frame actions,
three scalar temporal indices are read. Packed action chunks and state history
retain their stored axes; per-frame actions clamp at each episode's last row.
"""
from __future__ import annotations

from collections import OrderedDict
import dataclasses
import hashlib
import math
from pathlib import Path
import re


MAX_NUMERIC_CHUNK_BYTES = 8 * 1024 * 1024
MAX_NUMERIC_CACHE_BYTES = 32 * 1024 * 1024
MAX_NUMERIC_BATCH_BYTES = 128 * 1024 * 1024

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


def build_numeric_statistics_config(factory, model_config, *, effective_fps=None, validate_model=True):
    # A generic create()/repack factory cannot prove which fields its transforms
    # need. Reject it before executing user transforms or any dataset loader.
    if type(factory).__name__ != "ComposableDataConfig":
        _fail("该处理器没有可证明的纯数值统计接口，请使用字段组合处理器；不会回退读取图像")
    composition = normalize_composition(factory.composition, norm_stats=True)
    if validate_model:
        composition_runtime_contract(factory, model_config, norm_stats=True)
    else:
        # Metadata-only automatic-statistics planning must not import model
        # implementations (and transitively initialize JAX). Full model-type
        # validation remains the ordinary training preflight's responsibility.
        horizon, action_dim = model_config.action_horizon, model_config.action_dim
        if type(horizon) is not int or horizon <= 0 or type(action_dim) is not int or action_dim <= 0:
            _fail("模型动作维度/预测长度必须为正整数")
        if sum(row["width"] for row in composition["action_layout"]) > action_dim:
            _fail("模型动作维度小于真实动作宽度")
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
    """Read bounded numeric Arrow chunks and gather complete batches in NumPy."""

    def __init__(self, config):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.config = config
        self._groups, self._starts, self._cache, self._dtypes = [], [], OrderedDict(), {}
        self._streams, self._cache_bytes = OrderedDict(), 0
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
                    # Reading numeric columns does not require Pandas. Arrow's
                    # to_pandas_dtype() imports it even for primitive numbers.
                    kind = "float" if pa.types.is_floating(dtype) else "uint" if pa.types.is_unsigned_integer(dtype) else "int"
                    self._dtypes[column] = kind + str(dtype.bit_width)
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
        import numpy as np
        self._numeric_columns = list(dict.fromkeys([row["key"] for row in config.composition["state_fields"]]
            + [config.composition["action_key"]]))
        self._shapes = {row["key"]: (config.composition["state_history"], row["width"])
                       for row in config.composition["state_fields"]}
        width = sum(row["width"] for row in config.composition["action_layout"])
        self._shapes[config.composition["action_key"]] = ((config.action_horizon, width) if config.packed_actions else (width,))
        row_bytes = sum(int(np.prod(self._shapes[key])) * np.dtype(self._dtypes[key]).itemsize for key in self._numeric_columns)
        if row_bytes > MAX_NUMERIC_CHUNK_BYTES:
            _fail("单行数值字段超过受限内存大小，请检查字段形状")
        self._chunk_rows = max(1, min(4096, MAX_NUMERIC_CHUNK_BYTES // max(1, row_bytes)))

    def __getstate__(self):
        return {**self.__dict__, "_cache": OrderedDict(), "_streams": OrderedDict(), "_cache_bytes": 0}

    def clear_cache(self):
        """Release Arrow readers and decoded arrays when a worker changes source."""
        self._cache.clear()
        self._cache_bytes = 0
        for stream in self._streams.values():
            stream["parquet"].close()
        self._streams.clear()

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

    def _decode_chunk(self, batch):
        import numpy as np
        arrays = {}
        rows = batch.num_rows
        for key in self._numeric_columns:
            try:
                value = np.asarray(batch.column(batch.schema.get_field_index(key)).to_pylist(), dtype=self._dtypes[key])
            except (TypeError, ValueError, OverflowError) as exc:
                _fail("数值字段形状与声明不一致：" + key + ": " + str(exc)[:200])
            shape = self._shapes[key]
            if (key != self.config.composition["action_key"] and shape[0] == 1
                    and value.shape == (rows, shape[1])):
                value = value[:, None, :]
            if value.shape != (rows, *shape) or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                _fail("数值字段必须是匹配声明形状的有限数值：" + key)
            arrays[key] = value
        for field in self.config.composition["state_fields"]:
            if field["kind"] == "gripper":
                self._check_gripper(arrays[field["key"]])
        actions = arrays[self.config.composition["action_key"]]
        start = 0
        for block in self.config.composition["action_layout"]:
            if block["kind"] == "gripper":
                self._check_gripper(actions[..., start:start + block["width"]])
            start += block["width"]
        return arrays

    def _chunk(self, group, offset):
        import pyarrow.parquet as pq
        for key, (count, arrays, _) in reversed(self._cache.items()):
            if key[0] == group and key[1] <= offset < key[1] + count:
                self._cache.move_to_end(key)
                return key[1], count, arrays
        stream = self._streams.get(group)
        if stream is None or offset < stream["offset"]:
            if stream is not None:
                stream["parquet"].close()
            path, row_group, _ = self._groups[group]
            parquet = pq.ParquetFile(path)
            stream = {"parquet": parquet, "offset": 0, "batches": iter(parquet.iter_batches(
                batch_size=self._chunk_rows, row_groups=[row_group], columns=self._numeric_columns, use_threads=False))}
            self._streams[group] = stream
        self._streams.move_to_end(group)
        while len(self._streams) > 2:
            _, previous = self._streams.popitem(last=False)
            previous["parquet"].close()
        while True:
            try:
                batch = next(stream["batches"])
            except StopIteration:
                _fail("Parquet 数值块行数与文件元信息不一致")
            start, count = stream["offset"], batch.num_rows
            stream["offset"] += count
            if not start <= offset < start + count:
                continue
            if set(batch.schema.names) != set(self._numeric_columns):
                _fail("Parquet 投影未严格返回声明的数值列")
            arrays = self._decode_chunk(batch)
            size = sum(value.nbytes for value in arrays.values())
            key = (group, start)
            self._cache[key] = (count, arrays, size)
            self._cache_bytes += size
            while self._cache_bytes > MAX_NUMERIC_CACHE_BYTES or len(self._cache) > 256:
                _, (_, _, previous_size) = self._cache.popitem(last=False)
                self._cache_bytes -= previous_size
            return start, count, arrays

    def _take(self, indices, key):
        import numpy as np
        result = np.empty((len(indices), *self._shapes[key]), dtype=self._dtypes[key])
        groups = np.searchsorted(self._starts, indices, side="right") - 1
        for group in np.unique(groups):
            positions = np.flatnonzero(groups == group)
            offsets = indices[positions] - self._starts[group]
            while len(positions):
                start, count, arrays = self._chunk(int(group), int(offsets.min()))
                selected = (offsets >= start) & (offsets < start + count)
                result[positions[selected]] = arrays[key][offsets[selected] - start]
                positions, offsets = positions[~selected], offsets[~selected]
        return result

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        batch = self.get_batch([index])
        return {key: value[0] for key, value in batch.items()}

    def __getitems__(self, indices):
        batch = self.get_batch(indices)
        return [{key: value[index] for key, value in batch.items()} for index in range(len(batch["state"]))]

    def iter_batches(self, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            _fail("统计批次大小必须是正整数")
        for start in range(0, len(self), batch_size):
            yield self.get_batch(range(start, min(start + batch_size, len(self))))

    def get_batch(self, indices):
        """Gather one ordered batch; no per-row Arrow scalar or H-step loop."""
        import numpy as np
        values = list(indices)
        if not values or any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                             or not 0 <= index < self._size for index in values):
            raise IndexError("Numeric batch indices must be nonempty valid integer rows")
        indices = np.asarray(values, dtype=np.int64)
        config, spec = self.config, self.config.composition
        width = sum(block["width"] for block in spec["action_layout"])
        state_size = sum(int(np.prod(self._shapes[field["key"]])) * np.dtype(self._dtypes[field["key"]]).itemsize
                         for field in spec["state_fields"])
        action_size = config.action_horizon * width * np.dtype(self._dtypes[spec["action_key"]]).itemsize
        if len(indices) * (state_size + action_size + config.action_horizon * 8) > MAX_NUMERIC_BATCH_BYTES:
            _fail("统计批次超过受限内存大小，请降低 batch_size")
        states = [self._take(indices, field["key"]) for field in spec["state_fields"]]
        if config.packed_actions:
            actions = self._take(indices, spec["action_key"])
        else:
            action_indices = np.minimum(indices[:, None] + np.arange(config.action_horizon)[None, :], self._episode_end[indices, None])
            actions = self._take(action_indices.reshape(-1), spec["action_key"]).reshape(len(indices), config.action_horizon, width)
        return {"state": np.concatenate(states, axis=-1), "actions": actions}

    def _check_gripper(self, value):
        import numpy as np
        limits = self.config.composition["gripper_range"]
        if limits is not None and (np.any(value < limits[0]) or np.any(value > limits[1])):
            _fail("显式夹爪字段超出声明范围")
