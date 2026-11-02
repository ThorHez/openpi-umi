"""Consumer wiring and source proofs run without model, Torch or JAX imports."""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import math
import numbers
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_shared_norm_runtime import runtime as numeric_runtime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openpi.shared.weighted_stats import compute_weighted_statistics
from openpi.training import shared_norm


def actual(path, names, namespace):
    nodes = [node for node in ast.parse((ROOT / path).read_text()).body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)


@dataclasses.dataclass(frozen=True)
class Group:
    inputs: tuple = ()
    outputs: tuple = ()


@dataclasses.dataclass(frozen=True)
class Data:
    repo_id: str = "source"
    asset_id: str = "robot"
    normalization_method: str | None = None
    use_quantile_norm: bool = True
    norm_stats: object = dataclasses.field(default_factory=dict)
    normalize_masks: object = None
    repack_transforms: Group = dataclasses.field(default_factory=Group)
    data_transforms: Group = dataclasses.field(default_factory=Group)
    model_transforms: Group = dataclasses.field(default_factory=Group)
    action_loss_mask: object = None
    robot_type: object = None
    rlds_data_dir: object = None


class CaptureNormalize:
    """Capture constructor wiring only; formula mathematics has separate tests."""
    def __init__(self, stats, *, use_quantiles=False, key_masks=None, method=None):
        self.stats, self.use_quantiles, self.key_masks, self.method = stats, use_quantiles, key_masks, method

    def __call__(self, value):
        return value


class CaptureUnnormalize(CaptureNormalize):
    pass


@pytest.fixture
def loaders():
    transforms = SimpleNamespace(Normalize=CaptureNormalize, InjectActionLossMask=lambda value: value)
    namespace = dict(dataclasses=dataclasses, logging=logging, _transforms=transforms,
        _config=SimpleNamespace(_set_robot_type=lambda data, _: data),
        TransformedDataset=lambda dataset, pipeline: SimpleNamespace(dataset=dataset, pipeline=pipeline),
        IterableTransformedDataset=lambda dataset, pipeline, **kwargs: SimpleNamespace(dataset=dataset, pipeline=pipeline),
        create_torch_data_loader=lambda data, **_: data, create_rlds_data_loader=lambda data, **_: data)
    actual("src/openpi/training/data_loader.py",
        {"_apply_global_normalization", "transform_dataset", "transform_iterable_dataset", "create_data_loader"}, namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("method", ["minmax", "quantile", "zscore", None])
@pytest.mark.parametrize("iterable", [False, True])
def test_core_loader_passes_final_explicit_method_without_reinterpreting_old_bool(loaders, method, iterable):
    original = Data(normalization_method="quantile")
    data = loaders._apply_global_normalization(original, SimpleNamespace(normalization_method=method))
    assert data.normalization_method == (method or "quantile")
    assert original.normalization_method == "quantile"
    transform = loaders.transform_iterable_dataset if iterable else loaders.transform_dataset
    result = transform([], data)
    normalizer = next(item for item in result.pipeline if isinstance(item, CaptureNormalize))
    assert normalizer.method == (method or "quantile") and normalizer.use_quantiles is True


@pytest.mark.parametrize("rlds", [False, True])
def test_single_loader_applies_train_config_after_data_factory(loaders, rlds):
    original = Data(normalization_method="zscore", rlds_data_dir="rlds" if rlds else None)
    config = SimpleNamespace(data=SimpleNamespace(create=lambda *_: original), assets_dirs=(),
        model=SimpleNamespace(action_horizon=2), batch_size=2, num_workers=0, seed=0, normalization_method="minmax")
    result = loaders.create_data_loader(config)
    assert result.normalization_method == "minmax"


@pytest.mark.parametrize("method", ["bad", True, []])
def test_global_loader_rejects_invalid_explicit_method(loaders, method):
    with pytest.raises(ValueError, match="normalization_method"):
        loaders._apply_global_normalization(Data(), SimpleNamespace(normalization_method=method))


@pytest.mark.parametrize("method", ["minmax", None])
def test_multi_loader_keeps_resolved_global_method_in_each_checkpoint_data_config(loaders, method):
    seen = []
    fake_torch = SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(
        Dataset=object, WeightedRandomSampler=lambda weights, **kwargs: weights)),
        tensor=lambda weights, **_: np.asarray(weights), double=np.float64)
    data_loader = SimpleNamespace(_apply_global_normalization=loaders._apply_global_normalization,
        create_torch_dataset=lambda *_: [0, 1],
        transform_dataset=lambda dataset, dc, **_: (seen.append(dc) or dataset),
        TorchDataLoader=lambda dataset, **kwargs: SimpleNamespace(dataset=dataset, options=kwargs))
    namespace = dict(math=math, numbers=numbers, logging=logging, torch=fake_torch,
        jax=SimpleNamespace(process_count=lambda: 1), _data_loader=data_loader,
        MultiDataLoaderImpl=lambda configs, loader: SimpleNamespace(configs=configs, loader=loader))
    actual("src/openpi/training/multi_data_loader.py",
        {"WeightedConcatDataset", "create_multi_data_loader"}, namespace)
    config = SimpleNamespace(model=SimpleNamespace(action_horizon=2, image_keys=None, depth_keys=None),
        batch_size=2, num_workers=0, seed=0, normalization_method=method)
    data = [Data(repo_id=str(i), normalization_method="zscore") for i in range(3)]
    result = namespace["create_multi_data_loader"](config, list(zip(data, [1, 2, 2])))
    assert len(seen) == 3 and result.configs == seen
    assert all(dc.normalization_method == (method or "zscore") for dc in result.configs)


@pytest.fixture
def policy_runtime():
    reads, model_loads, composed = [], [], []

    def read_document(directory):
        return json.loads((Path(directory) / "norm_stats.json").read_text())

    def load_with_method(directory):
        reads.append("with_method")
        document = read_document(directory)
        return document["norm_stats"], document.get("normalization_method")

    def load_method(directory):
        reads.append("method_only")
        return read_document(directory).get("normalization_method")

    def compose(pipeline):
        composed.append(pipeline)
        return lambda value: value

    class ModelConfig:
        def load(self, parameters):
            model_loads.append(parameters)
            return SimpleNamespace(action_horizon=2)

    transforms = SimpleNamespace(Group=Group, Normalize=CaptureNormalize, Unnormalize=CaptureUnnormalize,
        InjectDefaultPrompt=lambda _: object(), compose=compose)
    namespace = dict(dataclasses=dataclasses, pathlib=SimpleNamespace(Path=Path), os=os, logging=logging, np=np,
        download=SimpleNamespace(maybe_download=lambda path: Path(path)),
        _normalize=SimpleNamespace(load_with_method=load_with_method, load_method=load_method),
        pi0_config=SimpleNamespace(Pi0Config=ModelConfig, Pi0RTCConfig=type("RTC", (), {})),
        _model=SimpleNamespace(restore_params=lambda *_, **__: "parameters"),
        jnp=SimpleNamespace(bfloat16=object()), transforms=transforms,
        _policy=SimpleNamespace(Policy=lambda model, **kwargs: SimpleNamespace(_model=model, **kwargs)),
        _config=SimpleNamespace(_set_robot_type=lambda data, _: data))
    actual("src/openpi/policies/policy_config.py",
        {"_with_checkpoint_normalization_method", "create_trained_policy"}, namespace)

    def config(data=None, method=None):
        return SimpleNamespace(data=SimpleNamespace(create=lambda *_: data or Data()),
            assets_dirs=(), model=ModelConfig(), policy_metadata={}, normalization_method=method)

    return SimpleNamespace(**namespace, reads=reads, model_loads=model_loads, composed=composed, config=config)


def checkpoint(tmp_path, method):
    directory = tmp_path / "assets/robot"
    directory.mkdir(parents=True)
    document = {"norm_stats": {"state": {}, "actions": {}}}
    if method is not None:
        document["normalization_method"] = method
    (directory / "norm_stats.json").write_text(json.dumps(document))
    return tmp_path


@pytest.mark.parametrize("recorded", ["minmax", "quantile", "zscore", None])
def test_standard_policy_and_rtc_forward_inverse_adopt_checkpoint_method_once(policy_runtime, tmp_path, recorded):
    runtime = policy_runtime
    result = runtime.create_trained_policy(runtime.config(), checkpoint(tmp_path, recorded))
    forward = next(item for item in result.transforms if isinstance(item, CaptureNormalize))
    inverse = next(item for item in result.output_transforms if isinstance(item, CaptureUnnormalize))
    assert forward.method == inverse.method == recorded
    assert any(forward is item for item in runtime.composed[0])  # The RTC guide uses this same normalizer.
    assert runtime.reads == ["with_method"]


def test_explicit_checkpoint_conflict_is_rejected_before_loading_model(policy_runtime, tmp_path):
    runtime = policy_runtime
    with pytest.raises(ValueError, match="NORMALIZATION_METHOD_CONFLICT"):
        runtime.create_trained_policy(runtime.config(method="minmax"), checkpoint(tmp_path, "quantile"))
    assert not runtime.model_loads


def test_global_policy_request_overrides_dataset_default_and_matches_checkpoint(policy_runtime, tmp_path):
    runtime = policy_runtime
    result = runtime.create_trained_policy(runtime.config(Data(normalization_method="zscore"), "minmax"),
        checkpoint(tmp_path, "minmax"))
    assert next(item for item in result.transforms if isinstance(item, CaptureNormalize)).method == "minmax"


def test_explicit_stats_work_without_checkpoint_assets_and_keep_request(policy_runtime, tmp_path):
    runtime = policy_runtime
    result = runtime.create_trained_policy(runtime.config(method="minmax"), tmp_path, norm_stats={"state": {}, "actions": {}})
    assert next(item for item in result.transforms if isinstance(item, CaptureNormalize)).method == "minmax"
    assert runtime.reads == []


def test_explicit_stats_still_reject_conflicting_existing_checkpoint_contract(policy_runtime, tmp_path):
    runtime = policy_runtime
    with pytest.raises(ValueError, match="NORMALIZATION_METHOD_CONFLICT"):
        runtime.create_trained_policy(runtime.config(method="zscore"), checkpoint(tmp_path, "minmax"),
            norm_stats={"state": {}, "actions": {}})


def test_cached_policy_view_uses_same_checkpoint_method_and_single_document_read(policy_runtime, tmp_path):
    runtime = policy_runtime
    namespace = dict(policy_config=runtime, _normalize=runtime._normalize, _policy=runtime._policy,
        _transforms=runtime.transforms, _training_config=runtime._config)
    actual("scripts/mem/serve_shellgame_real_m6_cached.py", {"_shared_model_policy_view"}, namespace)
    result = namespace["_shared_model_policy_view"](SimpleNamespace(_model=object(), metadata={}),
        runtime.config(), checkpoint(tmp_path, "minmax"))
    assert next(item for item in result.transforms if isinstance(item, CaptureNormalize)).method == "minmax"
    assert next(item for item in result.output_transforms if isinstance(item, CaptureUnnormalize)).method == "minmax"
    assert runtime.reads == ["with_method"]


@pytest.mark.parametrize("method", ["minmax", "quantile", "zscore"])
def test_training_and_policy_consumers_use_real_identical_numeric_transforms(
    loaders, policy_runtime, numeric_runtime, tmp_path, method
):
    values = {key: numeric_runtime.NormStats(
        mean=np.array([5.]), std=np.array([2.]), min=np.array([0.]), max=np.array([20.]),
        q01=np.array([4.]), q99=np.array([16.]), quantile_method="q01_q99",
    ) for key in ("state", "actions")}
    loaders._transforms.Normalize = numeric_runtime.Normalize
    policy_runtime.transforms.Normalize = numeric_runtime.Normalize
    policy_runtime.transforms.Unnormalize = numeric_runtime.Unnormalize
    data = Data(norm_stats=values)
    train_data = loaders._apply_global_normalization(data, SimpleNamespace(normalization_method=method))
    pipeline = loaders.transform_dataset([], train_data).pipeline
    training = next(item for item in pipeline if isinstance(item, numeric_runtime.Normalize))
    policy = policy_runtime.create_trained_policy(policy_runtime.config(data, method), tmp_path, norm_stats=values)
    forward = next(item for item in policy.transforms if isinstance(item, numeric_runtime.Normalize))
    inverse = next(item for item in policy.output_transforms if isinstance(item, numeric_runtime.Unnormalize))
    physical = np.array([[6.]])
    training_result = training({"state": physical.copy(), "actions": physical.copy()})
    policy_result = forward({"state": physical.copy(), "actions": physical.copy()})
    expected = {
        "minmax": (physical - 0) / (20 + 1e-6) * 2 - 1,
        "quantile": (physical - 4) / (12 + 1e-6) * 2 - 1,
        "zscore": (physical - 5) / (2 + 1e-6),
    }[method]
    for key in ("state", "actions"):
        np.testing.assert_allclose(training_result[key], expected)
        np.testing.assert_array_equal(policy_result[key], training_result[key])
    restored = inverse(policy_result)
    np.testing.assert_allclose(restored["actions"], physical, atol=1e-12)


@dataclasses.dataclass(frozen=True)
class ProofBase:
    normalization_method: str | None = None
    use_quantile_norm: bool = True
    normalize_masks: dict = dataclasses.field(default_factory=lambda: {"state": (True, False)})


@dataclasses.dataclass(frozen=True)
class ProofFactory:
    repo_id: str
    base_config: ProofBase


@dataclasses.dataclass(frozen=True)
class ComposableDataConfig:
    repo_id: str
    composition: dict
    base_config: ProofBase


def source_directory(tmp_path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "meta/info.json").write_text(json.dumps({
        "total_frames": 3, "features": {"state": {"dtype": "float64", "shape": [2]},
            "actions": {"dtype": "float64", "shape": [2]}}}))
    (source / "data/part.parquet").write_bytes(b"PAR1-fixture-PAR1")
    return source


@pytest.mark.parametrize("composable", [False, True])
def test_formula_switch_reuses_the_same_raw_statistics_source_proof(tmp_path, composable):
    source = source_directory(tmp_path)
    before = ProofFactory(str(source), ProofBase("minmax", True))
    if composable:
        before = ComposableDataConfig(str(source), {
            "state_fields": [{"key": "state"}], "action_key": "actions",
            "normalization": "quantile", "cameras": [], "prompt_key": "task",
        }, before.base_config)
    after = dataclasses.replace(before, base_config=ProofBase("quantile", False))
    if composable:
        after = dataclasses.replace(after, composition=dict(after.composition, normalization="zscore"))
    descriptors = shared_norm.source_descriptors([before], [1.0])
    assert shared_norm.source_descriptors([after], [1.0]) == descriptors
    raw = {key: np.arange(6, dtype=np.float64).reshape(3, 2) for key in ("state", "actions")}
    numeric, coverage = compute_weighted_statistics([lambda: iter([raw])], [1.0])
    document = {"norm_stats": {key: {**{name: value.tolist() for name, value in values.items()},
        "quantile_method": "q01_q99"} for key, values in numeric.items()}, "shared_normalization": {
        "schema_version": 1, "method": "weighted_raw_samples", "weight_semantics": "per_sample_repeat",
        "coverage": "full", "sources": [dict(descriptors[0], **coverage[0])],
        "quantile_method": "weighted_histogram", "quantile_bins": 5000,
    }}
    shared_norm.validate_shared_metadata(document, [after], [1.0])
    changed_mask = dataclasses.replace(after, base_config=dataclasses.replace(after.base_config, normalize_masks={}))
    with pytest.raises(ValueError, match="SHARED_NORM_INVALID"):
        shared_norm.validate_shared_metadata(document, [changed_mask], [1.0])
