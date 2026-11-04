"""Bounded batch decoding preserves the previous per-row numeric semantics."""
import importlib.util
from pathlib import Path
import pickle
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openpi.training import numeric_stats as numeric


def data(tmp_path, *, packed=False, history=2, horizon=50, group_size=3, counts=(3, 2, 8, 1)):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    size = sum(counts)
    states = np.arange(size * history * 3, dtype=np.float32).reshape(size, history, 3) / 50
    gripper = np.full((size, history, 1), .03, dtype=np.float32)
    actions = np.arange(size * (horizon if packed else 1) * 4, dtype=np.float32).reshape(
        (size, horizon, 4) if packed else (size, 4)) / 100
    actions[..., -1] = .04
    episodes = np.repeat(np.arange(len(counts)), counts)
    frames = np.concatenate([np.arange(count) for count in counts])
    columns = {"joints": pa.array((states[:, 0] if history == 1 else states).tolist(), type=
                   pa.list_(pa.float32(), 3) if history == 1 else pa.list_(pa.list_(pa.float32(), 3), history)),
        "gripper": pa.array((gripper[:, 0] if history == 1 else gripper).tolist(), type=
                   pa.list_(pa.float32(), 1) if history == 1 else pa.list_(pa.list_(pa.float32(), 1), history)),
        "action": pa.array(actions.tolist(), type=pa.list_(pa.list_(pa.float32(), 4), horizon) if packed else pa.list_(pa.float32(), 4)),
        "episode_index": pa.array(episodes), "frame_index": pa.array(frames), "timestamp": pa.array(frames / 20),
        "observation.images.broken": pa.array([{"bytes": b"BROKEN_NOT_RGB", "path": "absent.jpg"}] * size)}
    table = pa.table(columns)
    pq.write_table(table.slice(0, 4), root / "data/000.parquet", row_group_size=group_size)
    pq.write_table(table.slice(4), root / "data/001.parquet", row_group_size=group_size)
    spec = {"state_fields": [{"key": "joints", "width": 3, "kind": "joint"},
                             {"key": "gripper", "width": 1, "kind": "gripper"}],
        "state_history": history, "action_key": "action", "action_layout": [
            {"kind": "joint", "width": 3}, {"kind": "gripper", "width": 1}], "gripper_range": [0., .1]}
    config = numeric.NumericStatisticsConfig(str(root), spec, horizon, 32, 20., size, packed)
    ends = np.repeat(np.cumsum(counts) - 1, counts)
    return config, np.concatenate([states, gripper], axis=-1), actions, ends


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("history", [1, 2])
@pytest.mark.parametrize("group_size", [1, 3])
def test_batches_are_bitwise_equal_across_files_groups_and_episode_boundaries(tmp_path, packed, history, group_size):
    config, states, actions, ends = data(tmp_path, packed=packed, history=history, group_size=group_size)
    reader = numeric.NumericParquetDataset(config)
    for indices in ([0, 1, 2, 3, 4, 5, 6], [13, 0, 3, 3, 12, 1], [7, 8, 9, 10, 11, 12, 13]):
        batch = reader.get_batch(indices)
        expected = actions[indices] if packed else actions[np.minimum(
            np.asarray(indices)[:, None] + np.arange(config.action_horizon)[None, :], ends[indices, None])]
        np.testing.assert_array_equal(batch["state"], states[indices])
        np.testing.assert_array_equal(batch["actions"], expected)
        assert batch["state"].dtype == batch["actions"].dtype == np.float32
        items = reader.__getitems__(indices)
        np.testing.assert_array_equal(np.stack([row["state"] for row in items]), batch["state"])
        np.testing.assert_array_equal(np.stack([row["actions"] for row in items]), batch["actions"])
    assert reader._cache_bytes <= numeric.MAX_NUMERIC_CACHE_BYTES and len(reader._streams) <= 2


def test_tail_batches_no_scalar_arrow_reads_and_bounded_decode_reuse(tmp_path, monkeypatch):
    config, states, actions, ends = data(tmp_path, group_size=1)
    reader = numeric.NumericParquetDataset(config)
    reads = []
    decode = reader._decode_chunk
    def tracked(batch):
        reads.append(batch.num_rows)
        assert not any('image' in name for name in batch.schema.names)
        return decode(batch)
    monkeypatch.setattr(reader, '_decode_chunk', tracked)
    batches = list(reader.iter_batches(6))
    assert list(map(lambda batch: len(batch['state']), batches)) == [6, 6, 2]
    np.testing.assert_array_equal(np.concatenate([batch['state'] for batch in batches]), states)
    assert sum(reads) == len(reader)  # Even H50 with one-row groups is decoded once, not 50 times.
    reader.clear_cache()
    assert not reader._streams and not reader._cache and reader._cache_bytes == 0


def test_decoded_blocks_and_returned_batches_obey_memory_limits(tmp_path, monkeypatch):
    config, _, _, _ = data(tmp_path, packed=True, group_size=20, counts=(50,))
    monkeypatch.setattr(numeric, 'MAX_NUMERIC_CHUNK_BYTES', 3000)
    monkeypatch.setattr(numeric, 'MAX_NUMERIC_CACHE_BYTES', 6000)
    reader = numeric.NumericParquetDataset(config)
    assert reader._chunk_rows == 3
    for start in range(0, len(reader), 3):
        reader.get_batch(range(start, min(start + 3, len(reader))))
        assert reader._cache_bytes <= 6000
        assert all(size <= 3000 for _, _, size in reader._cache.values())
    monkeypatch.setattr(numeric, 'MAX_NUMERIC_BATCH_BYTES', 100)
    with pytest.raises(ValueError, match='批次超过受限内存'):
        reader.get_batch([0, 1])


@pytest.mark.parametrize('packed', [False, True])
def test_spawn_snapshot_drops_live_arrow_handles_and_numpy_cache(tmp_path, packed):
    config, _, _, _ = data(tmp_path, packed=packed)
    reader = numeric.NumericParquetDataset(config)
    expected = reader.get_batch([3, 4, 7, 13])
    assert reader._cache
    restored = pickle.loads(pickle.dumps(reader))
    assert not restored._streams and not restored._cache and restored._cache_bytes == 0
    actual = restored.get_batch([3, 4, 7, 13])
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize('bad', ['state_nan', 'action_inf', 'gripper'])
def test_invalid_numeric_values_are_rejected_in_bulk(tmp_path, bad):
    config, _, _, _ = data(tmp_path)
    path = Path(config.repo_id) / 'data/000.parquet'
    table = pq.read_table(path)
    key = 'joints' if bad == 'state_nan' else 'action'
    values = table[key].to_pylist()
    if bad == 'state_nan':
        values[1][0][0] = float('nan')
    elif bad == 'action_inf':
        values[1][1] = float('inf')
    else:
        values[1][-1] = 4.
    table = table.set_column(table.schema.get_field_index(key), key, pa.array(values, type=table.schema.field(key).type))
    pq.write_table(table, path)
    reader = numeric.NumericParquetDataset(config)
    with pytest.raises(ValueError, match='有限数值|夹爪'):
        reader.get_batch([0, 1, 2])


def test_template_wrapper_delegates_bulk_indices_and_retains_frame_limit(tmp_path):
    config, _, _, _ = data(tmp_path)
    reader = numeric.NumericParquetDataset(config)
    spec = importlib.util.spec_from_file_location('batch_compute_norm_stats', ROOT / 'scripts/compute_norm_stats.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    wrapper = module._NumericDataset(reader, limit=5)
    def forbidden(*args):
        pytest.fail('Template wrapper must use the bulk interface')
    reader.__class__ = type('BulkOnly', (reader.__class__,), {'__getitem__': forbidden})
    items = wrapper.__getitems__([0, 1, 4])
    assert len(items) == 3 and set(items[0]) == {'state', 'actions'}
    with pytest.raises(IndexError, match='frame limit'):
        wrapper.__getitems__([5])


@pytest.mark.parametrize('indices', [[], [-1], [14], [True], [1.5]])
def test_invalid_batch_indices_are_rejected_without_io(tmp_path, indices):
    config, _, _, _ = data(tmp_path)
    reader = numeric.NumericParquetDataset(config)
    with pytest.raises(IndexError):
        reader.get_batch(indices)
    assert not reader._cache
