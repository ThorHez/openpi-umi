"""Real numeric Parquet scans and task-local artifacts without JAX or Torch."""
from copy import deepcopy
import ast
import dataclasses
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openpi.training import training_normalization as automatic
from openpi.training import numeric_stats
from openpi.training.shared_norm import load_shared_norm_document


def composition():
    return {"schema_version": 1, "robot_model": "robot", "arm_mode": "bimanual",
        "state_representation": "joint", "state_fields": [{"key": "observation.state", "width": 2,
            "arm": "shared", "kind": "other", "normalize": True}], "state_history": 1,
        "action_representation": "joint", "action_key": "action", "action_layout": [
            {"arm": "shared", "kind": "shared", "width": 2, "normalize": True}],
        "action_mode": "absolute", "rotation_representation": "", "position_unit": "",
        "joint_unit": "", "coordinate_frame": "", "gripper_range": None,
        "cameras": [], "prompt_key": "", "normalization": "quantile", "model_transform": "standard", "aux_keys": []}


@dataclasses.dataclass(frozen=True)
class Assets:
    assets_dir: str
    asset_id: str = "."


@dataclasses.dataclass(frozen=True)
class ComposableDataConfig:
    repo_id: str
    assets: Assets
    composition: dict


@dataclasses.dataclass(frozen=True)
class Multi:
    datasets: list
    weights: list | None = None
    auto_shared_norm: bool = True
    use_merged_norm_stats: bool = True
    shared_norm_stats_path: str | None = None


@dataclasses.dataclass(frozen=True)
class Config:
    data: object
    checkpoint_dir: Path
    assets_dirs: Path
    model: object
    batch_size: int = 2
    normalization_method: str = "quantile"
    resume: bool = False


def source(root, values, *, packed=False, metadata_fps=10.):
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    raw = np.asarray([[value, value + 1] for value in values], dtype=np.float32)
    actions = np.repeat(raw[:, None], 3, axis=1) if packed else raw
    table = pa.table({"observation.state": pa.array(raw.tolist(), type=pa.list_(pa.float32(), 2)),
        "action": pa.array(actions.tolist(), type=pa.list_(pa.list_(pa.float32(), 2), 3) if packed else pa.list_(pa.float32(), 2)),
        "observation.images.invalid": pa.array([{"bytes": b"BROKEN_IMAGE"}] * len(raw)),
        "episode_index": pa.array([0] * len(raw)), "frame_index": pa.array(list(range(len(raw)))),
        "timestamp": pa.array([index / 10 for index in range(len(raw))])})
    pq.write_table(table, root / "data/episode.parquet", row_group_size=2)
    info = {"fps": metadata_fps, "total_frames": len(raw), "total_episodes": 1, "features": {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [3, 2] if packed else [2]},
        "observation.images.invalid": {"dtype": "image", "shape": [1]}}}
    (root / "meta/info.json").write_text(json.dumps(info))
    assets = root.parent / (root.name + "-old-norm")
    assets.mkdir()
    (assets / "norm_stats.json").write_text('{"old": "unchanged"}')
    return ComposableDataConfig(str(root), Assets(str(assets)), composition()), raw


@pytest.fixture
def config(tmp_path):
    first, _ = source(tmp_path / "one", [0, 2])
    second, _ = source(tmp_path / "two", [100, 102, 104])
    return Config(Multi([first, second], [1., 2.]), tmp_path / "run/checkpoints/recipe/experiment",
        tmp_path / "defaults", SimpleNamespace(action_horizon=3, action_dim=32, pi05=True))


def test_planning_does_not_import_backends_scan_payloads_or_publish(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Planning must not touch numeric payloads")
    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    monkeypatch.setattr(numeric_stats, "NumericParquetDataset", forbidden)
    before = {name for name in sys.modules if name.split('.')[0] in {'torch', 'jax'}}
    plan = automatic.planned_training_normalization(config)
    assert plan["mode"] == "auto_shared" and plan["action_horizon"] == 3
    assert plan["effective_fps"] == [10., 10.]
    assert all(row["video_decode"] is False for row in plan["numeric_io"])
    assert {name for name in sys.modules if name.split('.')[0] in {'torch', 'jax'}} == before
    assert not config.checkpoint_dir.parent.exists()
    result = subprocess.run([sys.executable, '-c',
        f'import sys;sys.path.insert(0,{str(ROOT / "src")!r});import openpi.training.training_normalization;'
        'assert not any(name in sys.modules for name in ("numpy","torch","jax"))'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_real_weighted_shared_statistic_is_generated_before_padding_and_reused(config, monkeypatch):
    read = pq.ParquetFile.read_row_group
    projections = []
    def projected(self, index, columns=None, **kwargs):
        assert columns is not None and not any('image' in column for column in columns)
        projections.append(columns)
        return read(self, index, columns=columns, **kwargs)
    monkeypatch.setattr(pq.ParquetFile, 'read_row_group', projected)
    before = [(Path(child.assets.assets_dir) / 'norm_stats.json').read_bytes() for child in config.data.datasets]
    plan = automatic.planned_training_normalization(config)
    updated = automatic.prepare_training_normalization(config)
    assert config.data.shared_norm_stats_path is None and updated is not config
    assert updated.data.use_merged_norm_stats is True
    assert updated.data.shared_norm_stats_path == plan['output_path']
    document = load_shared_norm_document(plan['output_path'], config.data.datasets, config.data.weights)
    expected_state = np.concatenate([np.array([[0., 1.], [2., 3.]]), np.repeat(
        np.array([[100., 101.], [102., 103.], [104., 105.]]), 2, axis=0)])
    np.testing.assert_allclose(document['norm_stats']['state']['mean'], expected_state.mean(axis=0))
    assert len(document['norm_stats']['actions']['mean']) == 2
    assert [row['rows'] for row in document['shared_normalization']['sources']] == [2, 3]
    assert [row['action_vectors'] for row in document['shared_normalization']['sources']] == [6, 9]
    assert document['training_normalization']['fingerprint'] == plan['fingerprint']
    receipt = config.checkpoint_dir.parent / 'experiment-normalization.json'
    receipt_doc = json.loads(receipt.read_text())
    assert receipt_doc['norm_path'] == plan['output_path'] and receipt_doc['checkpoint_asset_id'] == 'shared'
    assert receipt_doc['statistics_widths'] == {'state': 2, 'actions': 2}
    assert receipt_doc['normalization_method'] == 'quantile'
    assert projections
    monkeypatch.setattr(automatic, '_batches', lambda *args: pytest.fail('Complete matching statistic should be reused'))
    assert automatic.prepare_training_normalization(config).data.shared_norm_stats_path == plan['output_path']
    automatic.prepare_training_normalization(updated)
    assert json.loads(receipt.read_text())['fingerprint'] == plan['fingerprint']
    assert before == [(Path(child.assets.assets_dir) / 'norm_stats.json').read_bytes() for child in config.data.datasets]


@pytest.mark.parametrize('change', ['weight', 'horizon', 'fps', 'data'])
def test_changes_cannot_reuse_the_previous_statistics(config, change):
    first = automatic.planned_training_normalization(config)
    automatic.prepare_training_normalization(config)
    if change == 'weight':
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, weights=[2., 1.]))
    elif change == 'horizon':
        config = dataclasses.replace(config, model=SimpleNamespace(action_horizon=2, action_dim=32, pi05=True))
    elif change == 'fps':
        object.__setattr__(config.data.datasets[0], '_norm_effective_fps', 20.)
    else:
        path = Path(config.data.datasets[0].repo_id) / 'data/episode.parquet'
        path.touch()
    second = automatic.planned_training_normalization(config)
    assert first['fingerprint'] != second['fingerprint'] and first['output_path'] != second['output_path']
    assert Path(first['output_path']).is_file() and not Path(second['output_path']).exists()
    with pytest.raises(ValueError, match='resuming cannot change'):
        automatic.prepare_training_normalization(dataclasses.replace(config, resume=True))


def test_formula_can_reuse_statistics_but_not_change_a_resumed_model(config):
    first = automatic.prepare_training_normalization(config)
    changed = dataclasses.replace(config, normalization_method='minmax')
    second = automatic.prepare_training_normalization(changed)
    assert first.data.shared_norm_stats_path == second.data.shared_norm_stats_path
    receipt = json.loads((config.checkpoint_dir.parent / 'experiment-normalization.json').read_text())
    assert receipt['normalization_method'] == 'minmax'
    with pytest.raises(ValueError, match='resuming cannot change'):
        automatic.prepare_training_normalization(dataclasses.replace(config, resume=True))


def test_packed_action_horizon_is_preserved_and_not_chunked_twice(tmp_path):
    first, _ = source(tmp_path / 'one', [0, 2], packed=True)
    second, _ = source(tmp_path / 'two', [10, 12, 14], packed=True)
    config = Config(Multi([first, second], [1., 2.]), tmp_path / 'output/checkpoints/run',
        tmp_path / 'defaults', SimpleNamespace(action_horizon=3, action_dim=32))
    updated = automatic.prepare_training_normalization(config)
    document = json.loads(Path(updated.data.shared_norm_stats_path).read_text())
    assert [row['action_vectors'] for row in document['shared_normalization']['sources']] == [6, 9]
    assert len(document['norm_stats']['actions']['q01']) == 2


def test_single_dataset_and_explicit_legacy_modes_are_unchanged(config):
    for data in (dataclasses.replace(config.data, datasets=config.data.datasets[:1], weights=[1.]),
                 dataclasses.replace(config.data, auto_shared_norm=False, use_merged_norm_stats=False)):
        old = dataclasses.replace(config, data=data)
        assert automatic.prepare_training_normalization(old) is old
    assert not config.checkpoint_dir.parent.exists()


def test_native_shared_default_computes_without_an_extra_opt_in(config):
    native = dataclasses.replace(config, data=dataclasses.replace(config.data, auto_shared_norm=False))
    ready = automatic.prepare_training_normalization(native)
    assert ready.data.auto_shared_norm is True
    assert Path(ready.data.shared_norm_stats_path).is_file()


def test_conflicting_existing_artifact_is_rejected_without_overwrite(config):
    path = Path(automatic.planned_training_normalization(config)['output_path'])
    path.parent.mkdir(parents=True)
    path.write_text('{"do_not_touch":true}')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='SHARED_NORM_INVALID'):
        automatic.prepare_training_normalization(config)
    assert path.read_bytes() == before


def test_bad_timestamps_fail_before_a_statistic_is_published(config):
    object.__setattr__(config.data.datasets[0], '_norm_effective_fps', 20.)
    plan = automatic.planned_training_normalization(config)
    with pytest.raises(ValueError, match='FPS'):
        automatic.prepare_training_normalization(config)
    assert not Path(plan['output_path']).exists()


def test_changed_execution_context_cannot_reuse_an_already_bound_runtime_config(config):
    ready = automatic.prepare_training_normalization(config)
    ready = dataclasses.replace(ready, model=SimpleNamespace(action_horizon=2, action_dim=32))
    with pytest.raises(ValueError, match='execution context'):
        automatic.prepare_training_normalization(ready)


def test_incompatible_layouts_fail_at_planning(config):
    altered = deepcopy(config.data.datasets[1].composition)
    altered['action_mode'] = 'delta'
    data = dataclasses.replace(config.data, datasets=[config.data.datasets[0],
        dataclasses.replace(config.data.datasets[1], composition=altered)])
    with pytest.raises(ValueError, match='layouts, ordering'):
        automatic.planned_training_normalization(dataclasses.replace(config, data=data))


def test_training_entrypoint_prepares_statistics_before_the_first_device_operation(config, monkeypatch):
    tree = ast.parse((ROOT / 'scripts/train_multi_dataset.py').read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    program = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), main], type_ignores=[])
    calls = []
    class ReachedDevice(Exception):
        pass
    def prepare(value):
        assert value is config
        calls.append('statistics_ready')
        return value
    def device_count():
        assert calls == ['statistics_ready']
        raise ReachedDevice
    monkeypatch.setattr(automatic, 'prepare_training_normalization', prepare)
    monkeypatch.setenv('HF_DATASETS_CACHE', '/tmp/test-cache')
    monkeypatch.setenv('TMPDIR', '/tmp')
    namespace = {'init_logging': lambda: None, 'logging': logging, 'platform': SimpleNamespace(node=lambda: 'cpu-test'),
        'os': os, 'jax': SimpleNamespace(device_count=device_count)}
    exec(compile(ast.fix_missing_locations(program), 'train_multi_dataset.py', 'exec'), namespace)
    with pytest.raises(ReachedDevice):
        namespace['main'](config)
