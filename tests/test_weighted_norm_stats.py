"""Numerical tests run on CPU with NumPy; training backends are not needed."""

from __future__ import annotations

import importlib.util
import dataclasses
import ast
import json
import math
import numbers
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openpi.shared.weighted_stats import _histogram_counts,compute_weighted_statistics


def sources(arrays, batch_size=4):
    return [
        lambda sample=sample: (
            {key: value[index:index + batch_size] for key, value in sample.items()}
            for index in range(0, len(sample["state"]), batch_size)
        )
        for sample in arrays
    ]


def sample(count, offset=0):
    return {
        "state": np.arange(count * 4, dtype=np.float64).reshape(count, 2, 2) + offset,
        "actions": np.arange(count * 6, dtype=np.float64).reshape(count, 3, 2) / 7 + offset,
    }


def test_three_sources_match_explicit_one_two_two_repeated_raw_vectors():
    arrays = [sample(2), sample(7, 20), sample(5, -10)]
    result, counts = compute_weighted_statistics(sources(arrays), [1, 2, 2])
    assert counts == [
        {"rows": count, "state_vectors": count * 2, "action_vectors": count * 3}
        for count in (2, 7, 5)
    ]
    for key in ("state", "actions"):
        repeated = np.concatenate([
            np.repeat(array[key], repeats, axis=0).reshape(-1, 2)
            for array, repeats in zip(arrays, (1, 2, 2), strict=True)
        ])
        np.testing.assert_allclose(result[key]["mean"], repeated.mean(axis=0), rtol=1e-13)
        np.testing.assert_allclose(result[key]["std"], repeated.std(axis=0), rtol=1e-13)
        np.testing.assert_array_equal(result[key]["min"], repeated.min(axis=0))
        np.testing.assert_array_equal(result[key]["max"], repeated.max(axis=0))
        tolerance = np.ptp(repeated, axis=0) / 5000 + 1e-12
        for name, q in (("q01", 0.01), ("q99", 0.99)):
            expected = np.quantile(repeated, q, axis=0, method="inverted_cdf")
            assert np.all(np.abs(result[key][name] - expected) <= tolerance)


def test_progress_preserves_statistics_and_reports_full_ordered_coverage():
    arrays=[sample(5),sample(7,20)]
    before,counts=compute_weighted_statistics(sources(arrays,batch_size=3),[1.,2.])
    events=[]
    after,other=compute_weighted_statistics(sources(arrays,batch_size=3),[1.,2.],
        progress=events.append,source_rows=[5,7])
    assert counts==other
    for key in before:
        for name in before[key]:
            np.testing.assert_array_equal(before[key][name],after[key][name])
    completed=[event for event in events if event['event']=='source_done']
    assert [(event['pass'],event['source_index'],event['rows_done'],event['pass_rows_done']) for event in completed]==[
        (1,0,5,5),(1,1,7,12),(2,0,5,5),(2,1,7,12)]
    assert all(event['rows_total'] in (5,7) and event['pass_rows_total']==12 for event in events)


def test_declared_row_coverage_cannot_silently_omit_tail():
    with pytest.raises(ValueError,match='declared full coverage'):
        compute_weighted_statistics(sources([sample(5)],batch_size=3),[1.],source_rows=[6])


@pytest.mark.parametrize('bounds',[(-3.,7.),(1.,np.nextafter(1.,2.)),(-1e308,1e308)])
def test_histogram_fast_path_exactly_matches_numpy_at_every_edge(bounds):
    low,high=bounds
    fractions=np.linspace(0.,1.,5001)
    edges=(1-fractions)*low+fractions*high
    values=np.concatenate([edges,edges[::-1],np.array([low,high,(low/2+high/2)])])
    if np.any(edges[:-1]>edges[1:]):
        with pytest.raises(ValueError,match='monotonically'):
            np.histogram(values,bins=edges)
        with pytest.raises(ValueError,match='monotonically'):
            _histogram_counts(values,edges,5000)
        return
    np.testing.assert_array_equal(_histogram_counts(values,edges,5000),np.histogram(values,bins=edges)[0])


def test_fast_histogram_preserves_complete_statistics_bit_for_bit(monkeypatch):
    import openpi.shared.weighted_stats as statistics
    arrays=[sample(5),sample(7,20)]
    optimized,counts=compute_weighted_statistics(sources(arrays,batch_size=3),[1.,2.])
    monkeypatch.setattr(statistics,'_histogram_counts',lambda values,edges,bins:np.histogram(values,bins=edges)[0])
    reference,coverage=compute_weighted_statistics(sources(arrays,batch_size=3),[1.,2.])
    assert counts==coverage
    for key in reference:
        for name in reference[key]:
            np.testing.assert_array_equal(optimized[key][name],reference[key][name])


def test_quantiles_pool_raw_values_instead_of_averaging_existing_quantiles():
    arrays = [{key: np.full((100, 1), value) for key in ("state", "actions")} for value in (0.0, 100.0)]
    result, _ = compute_weighted_statistics(sources(arrays), [1, 2])
    for stats in result.values():
        np.testing.assert_allclose(stats["q01"], [0])
        np.testing.assert_allclose(stats["q99"], [100], atol=0.021)
        assert not np.allclose(stats["q01"], [200 / 3])


def test_fractional_weights_keep_their_mass_without_rounding():
    arrays = [sample(3), sample(8, 40)]
    fractional, counts = compute_weighted_statistics(sources(arrays), [0.5, 1.5])
    repeated, other_counts = compute_weighted_statistics(sources(arrays), [1, 3])
    assert counts == other_counts
    for key in fractional:
        for name in fractional[key]:
            np.testing.assert_allclose(fractional[key][name], repeated[key][name])


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_common_weight_scaling_preserves_all_statistics(scale):
    arrays = [sample(2), sample(7, 20), sample(5, -10)]
    expected, counts = compute_weighted_statistics(sources(arrays), [1, 2, 2])
    actual, other_counts = compute_weighted_statistics(sources(arrays), [scale, 2 * scale, 2 * scale])
    assert counts == other_counts
    for key in actual:
        for name in actual[key]:
            np.testing.assert_array_equal(actual[key][name], expected[key][name])


def test_unrepresentable_relative_weight_is_rejected():
    with pytest.raises(ValueError, match="dynamic range"):
        compute_weighted_statistics(sources([sample(2), sample(2)]), [1e-300, 1e300])


def test_deterministic_two_passes_without_repeating_source_reads_by_weight():
    calls = [0, 0]
    arrays = [sample(3), sample(5, 1e9)]

    def factory(index):
        def iterate():
            calls[index] += 1
            return iter([arrays[index]])
        return iterate

    first, _ = compute_weighted_statistics([factory(0), factory(1)], [1, 200])
    assert calls == [2, 2]
    second, _ = compute_weighted_statistics(sources(arrays, batch_size=8), [1, 200])
    for key in first:
        for name in first[key]:
            np.testing.assert_array_equal(first[key][name], second[key][name])


def test_large_offset_population_variance_is_stable():
    arrays = [{key: (1e12 + np.arange(4)).reshape(4, 1) for key in ("state", "actions")}]
    result, _ = compute_weighted_statistics(sources(arrays), [0.1])
    for stats in result.values():
        np.testing.assert_allclose(stats["std"], [np.std(np.arange(4))], rtol=1e-13)


@pytest.mark.parametrize("weights", [[], [1, 2], [0], [-1], [np.nan], [np.inf], [True], ["1"]])
def test_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weight"):
        compute_weighted_statistics(sources([sample(2)]), weights)


def test_rejects_empty_sources():
    with pytest.raises(ValueError, match="one source"):
        compute_weighted_statistics([], [])
    with pytest.raises(ValueError, match="at least one row"):
        compute_weighted_statistics([lambda: iter(())], [1])


@pytest.mark.parametrize("mutation", ["nan", "inf", "complex", "text", "empty", "missing", "rows", "rank"])
def test_rejects_bad_raw_batches(mutation):
    raw = sample(2)
    if mutation in ("nan", "inf"):
        raw["actions"][0, 0, 0] = float(mutation)
    elif mutation == "complex":
        raw["state"] = raw["state"].astype(complex)
    elif mutation == "text":
        raw["state"] = raw["state"].astype(str)
    elif mutation == "empty":
        raw["state"] = np.empty((0, 2))
    elif mutation == "missing":
        del raw["actions"]
    elif mutation == "rows":
        raw["actions"] = raw["actions"][:1]
    else:
        raw["state"] = np.array([1, 2])
    with pytest.raises(ValueError):
        compute_weighted_statistics([lambda: iter([raw])], [1])


def test_rejects_incompatible_feature_dimensions():
    other = sample(3)
    other["actions"] = np.ones((3, 3, 7))
    with pytest.raises(ValueError, match="feature dimension"):
        compute_weighted_statistics(sources([sample(2), other]), [1, 2])


def test_rejects_drift_between_passes():
    calls = 0

    def iterate():
        nonlocal calls
        calls += 1
        return iter([sample(3, offset=0 if calls == 1 else 0.01)])

    with pytest.raises(ValueError, match="changed between"):
        compute_weighted_statistics([iterate], [1])


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("weighted_compute_norm", ROOT / "scripts/compute_norm_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("row_count", [2, 5])
def test_statistics_loader_keeps_small_sources_and_tail_after_actual_transforms(script, monkeypatch, row_count):
    import openpi.training

    raw = [dict(state=np.array([index]), actions=np.array([[index]]), image="unused") for index in range(row_count)]

    class Transformed:
        def __init__(self, dataset, transforms):
            self.dataset, self.transforms = dataset, transforms
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, index):
            value = self.dataset[index]
            for transform in self.transforms:
                value = transform(value)
            return value

    class Loader:
        def __init__(self, dataset, **kwargs):
            assert kwargs["drop_last"] is False and kwargs["shuffle"] is False
            self.dataset, self.kwargs = dataset, kwargs
        def __iter__(self):
            size = self.kwargs["batch_size"]
            for start in range(0, len(self.dataset), size):
                yield self.kwargs["collate_fn"]([self.dataset[i] for i in range(start, min(start + size, len(self.dataset)))])

    fake_torch = ModuleType("torch")
    fake_torch.utils = SimpleNamespace(data=SimpleNamespace(DataLoader=Loader))
    fake_loader = ModuleType("openpi.training.data_loader")
    fake_loader.create_torch_dataset = lambda *_: raw
    fake_loader.TransformedDataset = Transformed
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "openpi.training.data_loader", fake_loader)
    config = SimpleNamespace(repo_id="fixture", repack_transforms=SimpleNamespace(inputs=[lambda x: dict(x, state=x["state"] + 10)]),
        data_transforms=SimpleNamespace(inputs=[lambda x: dict(x, actions=x["actions"] * 2)]))
    loader, count = script.create_torch_dataloader(config, 1, 4, object(), 0)
    batches = list(loader)
    assert len(batches) == count == (row_count + 3) // 4
    assert sum(len(batch["state"]) for batch in batches) == row_count
    np.testing.assert_array_equal(np.concatenate([batch["state"] for batch in batches])[:, 0], np.arange(row_count) + 10)
    np.testing.assert_array_equal(np.concatenate([batch["actions"] for batch in batches])[:, 0, 0], np.arange(row_count) * 2)
    assert all(set(batch) == {"state", "actions"} for batch in batches)


def test_output_respects_assets_and_protects_input_data_and_old_norms(script, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    asset = tmp_path / "old-assets"
    factory = SimpleNamespace(repo_id=str(dataset), assets=SimpleNamespace(assets_dir=str(asset), asset_id="robot"))
    config = SimpleNamespace(assets_dirs=tmp_path / "default-assets")
    assert script._assets_directory(factory, config) == asset / "robot"
    with pytest.raises(ValueError, match="input dataset"):
        script._check_output(dataset / "new-norm", [factory], config, multi=True)
    with pytest.raises(ValueError, match="input statistics"):
        script._check_output(asset / "robot", [factory], config, multi=True)
    script._check_output(tmp_path / "shared", [factory], config, multi=True)
    with pytest.raises(ValueError, match="empty"):
        script._local_path("")


def test_atomic_publication_refuses_overwrite_without_explicit_option(script, tmp_path):
    output = tmp_path / "output"
    first = {"norm_stats": {"fixture": 1}, "shared_normalization": {"schema_version": 1}}
    script._publish(output, first, overwrite=False)
    with pytest.raises(FileExistsError):
        script._publish(output, {"changed": True}, overwrite=False)
    assert json.loads((output / "norm_stats.json").read_text()) == first
    script._publish(output, {"changed": True}, overwrite=True)
    assert json.loads((output / "norm_stats.json").read_text()) == {"changed": True}
    assert not list(output.glob("*.tmp"))


@pytest.fixture
def multi_script(script, monkeypatch, tmp_path):
    import openpi.training

    @dataclasses.dataclass
    class MultiFactory:
        datasets: list
        weights: list
        shared_norm_stats_path: str | None = None
        use_merged_norm_stats: bool = True

        def create(self, *_):
            pytest.fail("Shared statistics must never select only the first source")

        def create_all(self, *_):
            assert self.shared_norm_stats_path is None and self.use_merged_norm_stats is False
            return [SimpleNamespace(repo_id=child.repo_id, rlds_data_dir=None) for child in self.datasets]

    children = [SimpleNamespace(repo_id=str(tmp_path / f"data-{i}"),
        assets=SimpleNamespace(assets_dir=str(tmp_path / "old-assets"), asset_id=f"old-{i}")) for i in range(3)]
    factory = MultiFactory(children, [1, 2, 2])
    config = SimpleNamespace(data=factory, assets_dirs=tmp_path / "default-assets",
        model=SimpleNamespace(action_horizon=3), batch_size=4, num_workers=0)
    module = ModuleType("openpi.training.config")
    module.MultiDataConfigFactory = MultiFactory
    module.get_config = lambda _: config
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime = ModuleType("openpi.shared.runtime_paths")
    runtime.configure_training_cache = lambda: None
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    normalization = ModuleType("openpi.shared.normalize")
    normalization.NormStats = lambda **values: values
    normalization.serialize_json = lambda stats: json.dumps({"norm_stats": {
        key: {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in values.items()}
        for key, values in stats.items()}})
    monkeypatch.setitem(sys.modules, normalization.__name__, normalization)
    shared = ModuleType("openpi.training.shared_norm")
    descriptors = [dict(repo_id=child.repo_id, weight=weight, info_sha256="info", data_manifest_digest="data",
        processor_sha256="processor", compatibility_sha256="compatible")
        for child, weight in zip(children, factory.weights, strict=True)]
    calls = []

    def describe(factories, weights):
        assert factories == children and weights == [1, 2, 2]
        calls.append("describe")
        return [dict(source) for source in descriptors]

    shared.source_descriptors = describe
    shared.validate_shared_metadata = lambda document, *_: calls.append(document["shared_normalization"])
    monkeypatch.setitem(sys.modules, shared.__name__, shared)
    arrays = [sample(count, index * 20) for index, count in enumerate((2, 3, 5))]
    loaders = dict(zip([child.repo_id for child in children], sources(arrays), strict=True))
    monkeypatch.setattr(script, "create_torch_dataloader", lambda dc, *_: (list(loaders[dc.repo_id]()), None))
    return SimpleNamespace(script=script, config=config, arrays=arrays, calls=calls, shared=shared,
        descriptors=descriptors, output=tmp_path / "shared")


def test_multi_script_scans_all_sources_and_publishes_shared_provenance(multi_script):
    fixture = multi_script
    fixture.script.main("fixture", output_dir=str(fixture.output))
    document = json.loads((fixture.output / "norm_stats.json").read_text())
    metadata = document["shared_normalization"]
    assert metadata["method"] == "weighted_raw_samples" and metadata["coverage"] == "full"
    assert metadata["weight_semantics"] == "per_sample_repeat"
    assert metadata["quantile_method"] == "weighted_histogram" and metadata["quantile_bins"] == 5000
    assert [source["rows"] for source in metadata["sources"]] == [2, 3, 5]
    assert [source["state_vectors"] for source in metadata["sources"]] == [4, 6, 10]
    assert [source["action_vectors"] for source in metadata["sources"]] == [6, 9, 15]
    for key in ("state", "actions"):
        pooled = np.concatenate([np.repeat(raw[key], repeat, axis=0).reshape(-1, 2)
            for raw, repeat in zip(fixture.arrays, (1, 2, 2), strict=True)])
        np.testing.assert_allclose(document["norm_stats"][key]["mean"], pooled.mean(axis=0))
        assert document["norm_stats"][key]["quantile_method"] == "q01_q99"
    assert fixture.calls[:2] == ["describe", "describe"]


def test_multi_script_accepts_configured_shared_file_destination(multi_script):
    fixture = multi_script
    fixture.config.data.shared_norm_stats_path = str(fixture.output / "norm_stats.json")
    fixture.script.main("fixture")
    assert (fixture.output / "norm_stats.json").exists()


def test_multi_script_rejects_partial_scan_or_missing_output(multi_script):
    fixture = multi_script
    with pytest.raises(ValueError, match="full coverage"):
        fixture.script.main("fixture", max_frames=10, output_dir=str(fixture.output))
    with pytest.raises(ValueError, match="independent --output-dir"):
        fixture.script.main("fixture")
    assert not fixture.output.exists()


def test_multi_script_rejects_processor_incompatibility_before_scanning(multi_script):
    fixture = multi_script
    fixture.descriptors[1]["compatibility_sha256"] = "different"
    with pytest.raises(ValueError, match="compatible"):
        fixture.script.main("fixture", output_dir=str(fixture.output))
    assert not fixture.output.exists()


def test_multi_script_detects_source_drift_without_publishing(multi_script):
    fixture = multi_script
    describe = fixture.shared.source_descriptors

    def drift(*args):
        result = describe(*args)
        if fixture.calls.count("describe") == 2:
            result[0]["data_manifest_digest"] = "changed"
        return result

    fixture.shared.source_descriptors = drift
    with pytest.raises(ValueError, match="changed during"):
        fixture.script.main("fixture", output_dir=str(fixture.output))
    assert not fixture.output.exists()


def test_config_file_uses_exported_builder_exact_bindings_without_calling_training(multi_script, monkeypatch, tmp_path):
    fixture = multi_script
    bridge = ModuleType("_weighted_exported_config_fixture")
    bridge.config = fixture.config
    bridge.calls = []
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    config_file = tmp_path / "training_config.py"
    config_file.write_text(
        "from _weighted_exported_config_fixture import config, calls\n"
        "def build_config(output_dir):\n"
        "    calls.append(output_dir)\n"
        "    return config\n"
        "def main(*args):\n"
        "    raise AssertionError('Training must not start')\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    monkeypatch.setattr(sys.modules["openpi.training.config"], "get_config",
        lambda *_: pytest.fail("A config file must not fall back to a static named recipe"))
    fixture.script.main(config_file=str(config_file), output_dir=str(fixture.output))
    assert bridge.calls == [str(fixture.output / ".config-only-output")]
    assert not (fixture.output / ".config-only-output").exists()
    document = json.loads((fixture.output / "norm_stats.json").read_text())
    metadata = document["shared_normalization"]
    assert [source["repo_id"] for source in metadata["sources"]] == [child.repo_id for child in fixture.config.data.datasets]
    assert [source["weight"] for source in metadata["sources"]] == [1, 2, 2]
    assert [source["rows"] for source in metadata["sources"]] == [2, 3, 5]
    for key in ("state", "actions"):
        pooled = np.concatenate([np.repeat(raw[key], repeat, axis=0).reshape(-1, 2)
            for raw, repeat in zip(fixture.arrays, (1, 2, 2), strict=True)])
        np.testing.assert_allclose(document["norm_stats"][key]["mean"], pooled.mean(axis=0))
        np.testing.assert_allclose(document["norm_stats"][key]["std"], pooled.std(axis=0))


def test_named_config_entry_remains_supported(multi_script, monkeypatch):
    fixture = multi_script
    requested = []

    def named(name):
        requested.append(name)
        return fixture.config

    monkeypatch.setattr(sys.modules["openpi.training.config"], "get_config", named)
    fixture.script.main(config_name="named-three-datasets", output_dir=str(fixture.output))
    assert requested == ["named-three-datasets"]


@pytest.mark.parametrize("arguments", [{}, {"config_name": "recipe", "config_file": "training_config.py"}])
def test_config_name_and_config_file_are_exclusive(script, arguments):
    with pytest.raises(ValueError, match="exactly one"):
        script.main(**arguments)


def test_config_file_requires_explicit_output_directory(script):
    with pytest.raises(ValueError, match="requires an independent --output-dir"):
        script.main(config_file="training_config.py")


@pytest.mark.parametrize("source, message", [
    ("def main(): raise AssertionError('not a builder')\n", "callable build_config"),
    ("build_config = 7\n", "callable build_config"),
    ("async def build_config(output_dir): return None\n", "synchronous callable"),
    ("def build_config(): return None\n", "one output_dir argument"),
    ("def build_config(output_dir, required): return None\n", "one output_dir argument"),
    ("def build_config(output_dir): return None\n", "did not return a training configuration"),
])
def test_config_file_rejects_invalid_builder(script, tmp_path, source, message):
    config_file = tmp_path / "invalid_config.py"
    config_file.write_text(source)
    with pytest.raises(ValueError, match=message):
        script._load_file_config(str(config_file), str(tmp_path / "stats"))
    assert not (tmp_path / "stats").exists()


@pytest.mark.parametrize("path", ["https://example.invalid/training_config.py", "absent.py", "dataset.json"])
def test_config_file_only_loads_existing_local_python(script, tmp_path, path):
    if "://" not in path:
        path = str(tmp_path / path)
    with pytest.raises(ValueError, match="local Python"):
        script._load_file_config(path, str(tmp_path / "stats"))


@pytest.fixture
def legacy_merge():
    """Run the real functions without importing optional schema dependencies."""
    @dataclasses.dataclass
    class Stats:
        mean: np.ndarray
        std: np.ndarray
        min: np.ndarray | None = None
        max: np.ndarray | None = None
        q01: np.ndarray | None = None
        q99: np.ndarray | None = None

    tree = ast.parse((ROOT / "src/openpi/shared/normalize.py").read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name in {"merge_norm_stats", "merge_norm_stats_dict"}]
    namespace = {"np": np, "NormStats": Stats}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "normalize.py", "exec"), namespace)
    return SimpleNamespace(stats=Stats, merge=namespace["merge_norm_stats"], merge_dict=namespace["merge_norm_stats_dict"])


@pytest.mark.parametrize("quantile", ["q01", "q99"])
@pytest.mark.parametrize("source_index", [0, 1])
def test_legacy_merge_rejects_precomputed_quantiles(legacy_merge, quantile, source_index):
    values = [legacy_merge.stats(np.array([0.0]), np.array([1.0])) for _ in range(2)]
    setattr(values[source_index], quantile, np.array([0.0]))
    with pytest.raises(ValueError, match="original samples.*compute_norm_stats.py"):
        legacy_merge.merge(values, weights=[1, 2])
    with pytest.raises(ValueError, match="original samples.*compute_norm_stats.py"):
        legacy_merge.merge_dict([{"state": values[0]}, {"state": values[1]}], weights=[1, 2])


def test_legacy_moment_only_merge_remains_compatible(legacy_merge):
    first, second = np.array([[0.0], [2.0]]), np.array([[10.0], [12.0]])
    stats = [legacy_merge.stats(array.mean(axis=0), array.std(axis=0),
        min=array.min(axis=0), max=array.max(axis=0)) for array in (first, second)]
    expected = np.concatenate([first, np.repeat(second, 2, axis=0)])
    for result in (legacy_merge.merge(stats, weights=[1, 2]),
                   legacy_merge.merge_dict([{"state": stats[0]}, {"state": stats[1]}], weights=[1, 2])["state"]):
        np.testing.assert_allclose(result.mean, expected.mean(axis=0))
        np.testing.assert_allclose(result.std, expected.std(axis=0))
        np.testing.assert_array_equal(result.min, expected.min(axis=0))
        np.testing.assert_array_equal(result.max, expected.max(axis=0))
        assert result.q01 is None and result.q99 is None


@pytest.fixture
def weighted_concat():
    """Execute the actual sampler dataset class without loading Torch or JAX."""
    tree = ast.parse((ROOT / "src/openpi/training/multi_data_loader.py").read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WeightedConcatDataset")
    fake_torch = SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(Dataset=object)))
    namespace = {"torch": fake_torch, "math": math, "numbers": numbers}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "multi_data_loader.py", "exec"), namespace)
    return namespace["WeightedConcatDataset"]


def test_sampler_expands_repeat_weights_per_row_and_respects_different_source_sizes(weighted_concat):
    datasets = [list(range(size)) for size in (2, 3, 5)]
    concat = weighted_concat(datasets, [1, 2, 2])
    weights = np.asarray(concat.get_dataset_weights_for_sampler())
    np.testing.assert_array_equal(weights, [0.5] * 2 + [1.0] * 3 + [1.0] * 5)
    mass = np.asarray([weights[:2].sum(), weights[2:5].sum(), weights[5:].sum()])
    np.testing.assert_allclose(mass / mass.sum(), np.array([2, 6, 10]) / 18)
    assert not np.allclose(mass / mass.sum(), [0.2, 0.4, 0.4])
    assert len(concat) == 10 and concat[2] == datasets[1][0] and concat[9] == datasets[2][-1]


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_sampler_common_weight_scaling_preserves_distribution(weighted_concat, scale):
    datasets = [list(range(size)) for size in (2, 3, 5)]
    base = weighted_concat(datasets, [1, 2, 2]).get_dataset_weights_for_sampler()
    scaled = weighted_concat(datasets, [scale, 2 * scale, 2 * scale]).get_dataset_weights_for_sampler()
    np.testing.assert_array_equal(base, scaled)
    assert np.isfinite(np.asarray(scaled)).all() and max(scaled) == 1.0


def test_sampler_none_remains_uniform_per_row(weighted_concat):
    np.testing.assert_array_equal(weighted_concat([[1, 2], [3]], None).get_dataset_weights_for_sampler(), [1, 1, 1])


@pytest.mark.parametrize("weights", [[], [1], [1, 2, 3], [0, 1], [0, 0], [-1, 1], [np.nan, 1],
    [np.nan, np.nan], [np.inf, 1], [True, 1], [False, False], [np.bool_(True), 1], ["1", 2], [1e-300, 1e300]])
def test_sampler_invalid_weights_are_rejected_instead_of_silent_uniform_fallback(weighted_concat, weights):
    with pytest.raises(ValueError, match="weight"):
        weighted_concat([[0], [1, 2]], weights)
