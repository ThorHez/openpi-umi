"""Exercise actual shared-statistic runtime definitions on CPU without JAX imports."""

from __future__ import annotations

import abc
import ast
import dataclasses
from functools import partial
import json
import logging
import math
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openpi.shared.weighted_stats import compute_weighted_statistics
from openpi.training import shared_norm


def actual_definitions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == names
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def flatten(tree, sep="/"):
    result = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            result.update({key + sep + subkey: leaf for subkey, leaf in flatten(value, sep).items()})
        else:
            result[key] = value
    return result


def unflatten(tree, sep="/"):
    result = {}
    for key, value in tree.items():
        target = result
        *parents, leaf = key.split(sep)
        for parent in parents:
            target = target.setdefault(parent, {})
        target[leaf] = value
    return result


@pytest.fixture
def runtime(monkeypatch):
    """Only dependency boundaries are fixtures; runtime algorithms come from AST."""
    module = ModuleType("shared_norm_runtime_definitions")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = module.__dict__
    namespace.update(dataclasses=dataclasses, pathlib=pathlib, json=json, np=np, partial=partial,
                     abc=abc, math=math, logging=logging, DataTransformFn=object,
                     override=lambda fn: fn, tyro=SimpleNamespace(MISSING=object()))
    normalization = ModuleType("shared_norm_runtime_serialization")
    monkeypatch.setitem(sys.modules, normalization.__name__, normalization)
    normalization.__dict__.update(
        # pydantic/numpydantic are serialization boundaries, not required to
        # exercise the real normalization formulas or checkpoint writer.
        pydantic=SimpleNamespace(dataclasses=dataclasses), pathlib=pathlib, json=json,
    )
    actual_definitions(ROOT / "src/openpi/shared/normalize.py", {"NormStats"}, normalization.__dict__)

    class StatsEnvelope:
        def __init__(self, norm_stats, **_metadata):
            self.norm_stats = {
                key: values if isinstance(values, normalization.NormStats) else normalization.NormStats(**{
                    name: np.asarray(value, dtype=np.float64)
                    if name != "quantile_method" and value is not None else value
                    for name, value in values.items()
                }) for key, values in norm_stats.items()
            }

        def model_dump_json(self, **kwargs):
            return json.dumps({"norm_stats": {key: dataclasses.asdict(value)
                                             for key, value in self.norm_stats.items()}},
                              default=lambda value: value.tolist(), **kwargs)

    normalization._NormStatsDict = StatsEnvelope
    actual_definitions(ROOT / "src/openpi/shared/normalize.py",
                       {"serialize_json", "deserialize_json", "save", "load", "load_method", "load_with_method"},
                       normalization.__dict__)
    namespace.update(_normalize=normalization, NormStats=normalization.NormStats,
                     traverse_util=SimpleNamespace(flatten_dict=flatten, unflatten_dict=unflatten))
    actual_definitions(ROOT / "src/openpi/transforms.py", {
        "Group", "Normalize", "Unnormalize", "PadStateOnly", "flatten_dict", "unflatten_dict",
        "apply_tree", "pad_to_dim", "_assert_quantile_stats", "_assert_normalization_stats",
    }, namespace)
    namespace["_transforms"] = SimpleNamespace(Group=namespace["Group"], PadStateOnly=namespace["PadStateOnly"])
    actual_definitions(ROOT / "src/openpi/training/config.py", {
        "AssetsConfig", "DataConfig", "DataConfigFactory", "MultiDataConfigFactory",
    }, namespace)
    actual_definitions(ROOT / "src/openpi/training/checkpoints.py", {"_save_norm_assets"}, namespace)
    module.normalization = normalization
    return module


def stats(runtime, *, marker="q01_q99", shift=0.):
    return {key: runtime.NormStats(
        mean=np.array([5., 5., 0.]) + shift, std=np.array([2., 0., 1.]),
        q01=np.array([0., 5., -2.]) + shift, q99=np.array([10., 5., 2.]) + shift,
        min=np.array([-100., -100., -100.]), max=np.array([100., 100., 100.]),
        quantile_method=marker,
    ) for key in ("state", "actions")}


@pytest.fixture
def sources(runtime, tmp_path):
    @dataclasses.dataclass(frozen=True)
    class LocalSource(runtime.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            del model_config
            norm_root = pathlib.Path(self.assets.assets_dir or assets_dirs) / (self.assets.asset_id or self.repo_id)
            return runtime.DataConfig(
                repo_id=self.repo_id, asset_id=self.assets.asset_id or self.repo_id,
                norm_stats=runtime.normalization.load(norm_root),
                normalize_masks={"state": (True, True, False), "actions": (True, True, False)},
            )

    factories, arrays = [], []
    for index in range(3):
        root = tmp_path / f"source-{index}"
        (root / "meta").mkdir(parents=True)
        (root / "data").mkdir()
        (root / "meta/info.json").write_text(json.dumps({
            "total_frames": 3, "total_episodes": 1,
            "features": {"state": {"dtype": "float64", "shape": [3]},
                         "actions": {"dtype": "float64", "shape": [2, 3]}},
        }))
        (root / "data/episode_0.parquet").write_bytes(b"PAR1-local-source-manifest-PAR1")
        runtime.normalization.save(root, stats(runtime, shift=float(index)))
        factories.append(LocalSource(repo_id=str(root), assets=runtime.AssetsConfig(str(root), ".")))
        raw = np.array([[0., 5., -2.], [5., 5., 0.], [10., 5., 2.]]) + [index, 0., index]
        arrays.append({"state": raw, "actions": np.repeat(raw[:, None, :], 2, axis=1)})
    weights = [1., 2., 2.]
    numeric, counts = compute_weighted_statistics(
        [lambda sample=sample: iter([sample]) for sample in arrays], weights,
    )
    shared = {key: runtime.NormStats(**values, quantile_method="q01_q99") for key, values in numeric.items()}
    metadata = {
        "schema_version": 1, "method": "weighted_raw_samples", "weight_semantics": "per_sample_repeat",
        "coverage": "full", "quantile_method": "weighted_histogram", "quantile_bins": 5000,
        "sources": [{**descriptor, **count} for descriptor, count in
                    zip(shared_norm.source_descriptors(factories, weights), counts, strict=True)],
    }
    shared_dir = tmp_path / "weighted-shared"
    runtime.normalization.save(shared_dir, shared, shared_normalization=metadata)
    document = json.loads((shared_dir / "norm_stats.json").read_text())
    assert shared_norm.validate_shared_metadata(document, factories, weights) == metadata
    return factories, weights, shared_dir / "norm_stats.json", metadata


def multi(runtime, sources, **changes):
    factories, weights, path, _ = sources
    return runtime.MultiDataConfigFactory(datasets=factories, weights=weights,
                                         state_pad_dim=8, shared_norm_stats_path=str(path), **changes)


def test_valid_shared_artifact_is_one_object_and_checkpoint_identity_for_three_sources(runtime, sources, tmp_path):
    factory = multi(runtime, sources)
    configs = factory.create_all(tmp_path, object())
    assert len(configs) == 3
    assert all(config.norm_stats is configs[0].norm_stats for config in configs)
    assert [config.asset_id for config in configs] == ["shared"] * 3
    assert all(config.norm_stats_metadata == sources[3] for config in configs)
    first = factory.create(tmp_path, object())
    assert first.asset_id == "shared" and first.norm_stats_metadata == sources[3]
    assert runtime.normalization.serialize_json(first.norm_stats) == runtime.normalization.serialize_json(configs[0].norm_stats)
    assert first.normalize_masks == configs[0].normalize_masks
    for config in configs:
        assert isinstance(config.model_transforms.inputs[-1], runtime.PadStateOnly)
        assert config.model_transforms.inputs[-1].target_dim == 8
    physical = np.array([[2., 5., .025]])
    results = [runtime.Normalize(config.norm_stats, use_quantiles=True, key_masks=config.normalize_masks)(
        {"state": physical.copy(), "actions": physical.copy()}) for config in configs]
    for result in results[1:]:
        np.testing.assert_array_equal(result["state"], results[0]["state"])
        np.testing.assert_array_equal(result["actions"], results[0]["actions"])


def test_missing_common_shared_path_refuses_to_average_existing_quantiles(runtime, sources, tmp_path):
    factories, weights, _, _ = sources
    factory = runtime.MultiDataConfigFactory(datasets=factories, weights=weights, state_pad_dim=8)
    with pytest.raises(ValueError, match="SHARED_NORM_REQUIRED.*existing quantiles cannot be merged"):
        factory.create_all(tmp_path, object())


def test_legacy_per_source_dot_assets_get_distinct_ids_with_consistent_first_policy(runtime, sources, tmp_path):
    factories, weights, _, _ = sources
    factory = runtime.MultiDataConfigFactory(datasets=factories, weights=weights, state_pad_dim=8,
                                             use_merged_norm_stats=False)
    configs = factory.create_all(tmp_path, object())
    assert len({config.asset_id for config in configs}) == 3
    assert configs[0].asset_id == factory.create(tmp_path, object()).asset_id == "."
    np.testing.assert_array_equal(configs[1].norm_stats["state"].mean, [6., 6., 1.])
    assert runtime.normalization.serialize_json(configs[0].norm_stats) == runtime.normalization.serialize_json(
        factory.create(tmp_path, object()).norm_stats)


def test_checkpoint_conflicting_asset_ids_fail_before_any_write(runtime, tmp_path, monkeypatch):
    writes = []
    monkeypatch.setattr(runtime.normalization, "save", lambda *args, **kwargs: writes.append((args, kwargs)))
    configs = [runtime.DataConfig(asset_id="same", norm_stats=stats(runtime, shift=shift)) for shift in (0., 1.)]
    with pytest.raises(ValueError, match="Conflicting normalization statistics"):
        runtime._save_norm_assets(tmp_path / "assets", configs)
    assert writes == [] and not (tmp_path / "assets").exists()


def test_checkpoint_shared_asset_is_written_once_with_original_provenance(runtime, sources, tmp_path, monkeypatch):
    configs = multi(runtime, sources).create_all(tmp_path, object())
    writes = []
    save = runtime.normalization.save
    def tracked_save(directory, norm_stats, **kwargs):
        writes.append(directory)
        save(directory, norm_stats, **kwargs)
    monkeypatch.setattr(runtime.normalization, "save", tracked_save)
    runtime._save_norm_assets(tmp_path / "checkpoint/assets", configs)
    assert writes == [tmp_path / "checkpoint/assets/shared"]
    document = json.loads((writes[0] / "norm_stats.json").read_text())
    assert document["shared_normalization"] == sources[3]
    assert document["norm_stats"]["state"]["quantile_method"] == "q01_q99"
    assert document["norm_stats"]["actions"]["quantile_method"] == "q01_q99"


@pytest.mark.parametrize("asset_id", ["/absolute", "../escape", "nested/../../escape", "back\\slash", "bad\nname", ""])
def test_checkpoint_unsafe_asset_ids_fail_before_writing(runtime, tmp_path, asset_id, monkeypatch):
    writes = []
    monkeypatch.setattr(runtime.normalization, "save", lambda *args, **kwargs: writes.append(args))
    configs = [runtime.DataConfig(asset_id="safe", norm_stats=stats(runtime)),
               runtime.DataConfig(asset_id=asset_id, norm_stats=stats(runtime))]
    with pytest.raises(ValueError, match="asset_id"):
        runtime._save_norm_assets(tmp_path / "assets", configs)
    assert writes == []


def test_marked_formula_uses_quantiles_even_with_minmax_and_roundtrips_masks_constants_and_padding(runtime):
    values = stats(runtime)
    physical = np.array([[0., 5., .025], [2., 5., .05], [12., 5., .075]])
    masks = {key: (True, True, False) for key in values}
    source = {key: physical.copy() for key in values}
    normalized = runtime.Normalize(values, use_quantiles=True, key_masks=masks)(source)
    for key, entry in values.items():
        expected = (physical - entry.q01) / (entry.q99 - entry.q01 + 1e-6) * 2 - 1
        expected[:, 2] = physical[:, 2]
        np.testing.assert_allclose(normalized[key], expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(normalized[key][:, 1], -np.ones(3))
        np.testing.assert_array_equal(normalized[key][:, 2], physical[:, 2])
    suffix = np.array([[23., 71.], [24., 72.], [25., 73.]])
    padded = {key: np.concatenate([value, suffix], axis=-1) for key, value in normalized.items()}
    restored = runtime.Unnormalize(values, use_quantiles=True, key_masks=masks)(padded)
    for key in restored:
        np.testing.assert_allclose(restored[key][:, :3], physical, rtol=0, atol=1e-13)
        np.testing.assert_array_equal(restored[key][:, 3:], suffix)


def test_unmarked_legacy_quantiles_retain_their_minmax_behavior(runtime):
    values = stats(runtime, marker=None)
    physical = np.array([[0., 5., .025], [20., 7., .05]])
    normalized = runtime.Normalize(values, use_quantiles=True)({"actions": physical.copy()})["actions"]
    np.testing.assert_allclose(normalized, physical / 100., rtol=0, atol=1e-15)
    restored = runtime.Unnormalize(values, use_quantiles=True)({"actions": normalized})["actions"]
    np.testing.assert_allclose(restored, physical, rtol=0, atol=1e-14)
