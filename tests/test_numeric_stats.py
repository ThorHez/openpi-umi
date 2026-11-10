"""Real Parquet projection of single-step statistics without model dependencies."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
numeric = importlib.import_module("openpi.training.numeric_stats")


def composition(history=1):
    return {"schema_version": 1, "robot_model": "explicit", "arm_mode": "bimanual",
        "state_representation": "joint", "state_fields": [{"key": "observation.state", "width": 2,
            "arm": "shared", "kind": "other", "normalize": False}], "state_history": history,
        "action_representation": "joint", "action_key": "actions", "action_layout": [
            {"arm": "shared", "kind": "shared", "width": 2, "normalize": False}],
        "action_mode": "delta", "rotation_representation": "rotation_6d", "position_unit": "",
        "joint_unit": "", "coordinate_frame": "", "gripper_range": [0., .1],
        "cameras": [], "prompt_key": "missing_task", "normalization": "quantile",
        "model_transform": "standard", "aux_keys": ["is_positive"]}


def dataset(tmp_path, *, packed=False, history=1):
    root = tmp_path / "dataset"; (root / "data/chunk-000").mkdir(parents=True); (root / "meta").mkdir()
    states = np.arange(5 * history * 2, dtype=np.float32).reshape(5, history, 2)
    actions = np.arange(5 * (3 if packed else 1) * 2, dtype=np.float32).reshape((5, 3, 2) if packed else (5, 2))
    feature_shape = [history, 2] if history > 1 else [2]
    columns = {"observation.state": pa.array(states.tolist() if history > 1 else states[:, 0].tolist(),
            type=pa.list_(pa.list_(pa.float32(), 2), history) if history > 1 else pa.list_(pa.float32(), 2)),
        "actions": pa.array(actions.tolist(), type=pa.list_(pa.list_(pa.float32(), 2), 3) if packed else pa.list_(pa.float32(), 2)),
        "observation.images.invalid": pa.array([{"bytes": b"NOT_AN_IMAGE", "path": "missing.jpg"}] * 5),
        "unused_numeric": pa.array([90.] * 5)}
    table = pa.table(columns)
    pq.write_table(table.slice(0, 3), root / "data/chunk-000/episode_000.parquet", row_group_size=2)
    pq.write_table(table.slice(3), root / "data/chunk-000/episode_001.parquet", row_group_size=1)
    info = {"fps": 10., "total_frames": 5, "total_episodes": 2, "features": {
        "observation.state": {"dtype": "float32", "shape": feature_shape},
        "actions": {"dtype": "float32", "shape": [3, 2] if packed else [2]},
        "observation.images.invalid": {"dtype": "image", "shape": [999]},
        "observation.images.missing_video": {"dtype": "video", "shape": []}}}
    (root / "meta/info.json").write_text(json.dumps(info))
    config = numeric.NumericStatisticsConfig(str(root), composition(history), 3, 32, 10., 5, packed)
    return config, states, actions


@pytest.mark.parametrize("packed", [False, True])
def test_source_projection_never_reads_embedded_images_or_missing_videos(tmp_path, monkeypatch, packed):
    config, states, actions = dataset(tmp_path, packed=packed, history=2)
    batches = pq.ParquetFile.iter_batches
    projections = []

    def projected_batches(self, *args, columns=None, **kwargs):
        projections.append(list(columns))
        assert columns == ["observation.state", "actions"]
        return batches(self, *args, columns=columns, **kwargs)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", projected_batches)
    scan = Path.rglob

    def numeric_scan(self, pattern):
        assert self.name == "data" and pattern == "*.parquet"
        return scan(self, pattern)

    monkeypatch.setattr(Path, "rglob", numeric_scan)
    source = numeric.NumericParquetDataset(config)
    assert len(source) == 5
    for index in range(5):
        result = source[index]
        np.testing.assert_array_equal(result["state"], states[index])
        np.testing.assert_array_equal(result["actions"], actions[index])
        assert result["state"].dtype == np.float32 and result["actions"].dtype == np.float32
        assert set(result) == {"state", "actions"}
    assert projections and source._cache_bytes <= numeric.MAX_NUMERIC_CACHE_BYTES
    assert len(source._streams) <= 2 and len(source._cache) <= 256
    evidence = numeric.numeric_io_evidence(config)
    assert evidence["columns"] == ["observation.state", "actions"]
    assert evidence["schema_version"] == 3
    assert evidence["action_sequence"] == ("packed_all_steps" if packed else "per_frame")
    assert evidence["temporal_columns"] == []
    assert evidence["video_decode"] is False


def test_per_frame_statistics_read_one_action_per_row_regardless_of_training_horizon(tmp_path):
    config, _, actions = dataset(tmp_path)
    for horizon in (1, 50):
        source = numeric.NumericParquetDataset(numeric.dataclasses.replace(config, action_horizon=horizon))
        batches = list(source.iter_batches(2))
        np.testing.assert_array_equal(np.concatenate([batch["actions"] for batch in batches]), actions)


def test_stored_action_chunk_must_still_match_training_horizon(tmp_path):
    config, _, _ = dataset(tmp_path, packed=True)
    factory = type("ComposableDataConfig", (), {})()
    factory.repo_id = config.repo_id
    factory.composition = config.composition
    with pytest.raises(ValueError, match="匹配预测长度"):
        numeric.build_numeric_statistics_config(
            factory, SimpleNamespace(action_horizon=50, action_dim=32), validate_model=False,
        )


def test_invalid_source_numeric_selection_rejected_before_any_parquet_payload_read(tmp_path, monkeypatch):
    config, _, _ = dataset(tmp_path, packed=True)
    monkeypatch.setattr(numeric, "composition_runtime_contract", lambda *args, **kwargs: {})
    factory_type = type("ComposableDataConfig", (), {})
    factory = factory_type(); factory.repo_id = config.repo_id; factory.composition = deepcopy(config.composition)
    factory.composition["state_fields"][0]["key"] = "observation.images.invalid"
    monkeypatch.setattr(pq, "ParquetFile", lambda *args, **kwargs: pytest.fail("Do not read a Parquet payload for an image selected as numeric"))
    with pytest.raises(ValueError, match="数值.*图像"):
        numeric.build_numeric_statistics_config(factory, SimpleNamespace(action_horizon=3, action_dim=32))
    with pytest.raises(ValueError, match="不会回退读取图像"):
        numeric.build_numeric_statistics_config(SimpleNamespace(), SimpleNamespace())


def test_arrow_schema_rejects_visual_struct_even_when_metadata_lies(tmp_path):
    config, _, _ = dataset(tmp_path, packed=True)
    spec = deepcopy(config.composition)
    spec["state_fields"][0]["key"] = "observation.images.invalid"
    config = numeric.dataclasses.replace(config, composition=spec)
    with pytest.raises(ValueError, match="不能读取图像"):
        numeric.NumericParquetDataset(config)


def test_numeric_sampling_does_not_apply_normalization_flags_or_guess_shared_grippers(tmp_path):
    config, _, actions = dataset(tmp_path, packed=True)
    assert actions.max() > config.composition["gripper_range"][1]
    first = numeric.NumericParquetDataset(config)[2]
    spec = deepcopy(config.composition)
    spec["state_fields"][0]["normalize"] = True
    spec["action_layout"][0]["normalize"] = True
    second = numeric.NumericParquetDataset(numeric.dataclasses.replace(config, composition=spec))[2]
    np.testing.assert_array_equal(first["state"], second["state"])
    np.testing.assert_array_equal(first["actions"], second["actions"])


@pytest.mark.parametrize("packed", [False, True])
def test_explicit_composite_numeric_config_preserves_raw_state_and_actions_without_representation_conversion(tmp_path, monkeypatch, packed):
    original, states, actions = dataset(tmp_path, packed=packed, history=2)
    spec = deepcopy(original.composition)
    spec.update(state_representation="composite", action_representation="composite")
    factory_type = type("ComposableDataConfig", (), {})
    factory = factory_type(); factory.repo_id = original.repo_id; factory.composition = spec
    # Model-class imports are a separate boundary; exercise the real schema,
    # metadata validation, projection and numeric sampler here.
    monkeypatch.setattr(numeric, "composition_runtime_contract", lambda *args, **kwargs: {})
    config = numeric.build_numeric_statistics_config(factory, SimpleNamespace(action_horizon=3, action_dim=32))
    assert config.composition["state_representation"] == config.composition["action_representation"] == "composite"
    source = numeric.NumericParquetDataset(config)
    for index in range(len(source)):
        np.testing.assert_array_equal(source[index]["state"], states[index])
        np.testing.assert_array_equal(source[index]["actions"], actions[index])
