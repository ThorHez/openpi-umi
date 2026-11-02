"""Numerical formula choices and checkpoint persistence, without model backends."""

import dataclasses
import json

import numpy as np
import pytest

from test_shared_norm_runtime import runtime, stats  # noqa: F401


@pytest.mark.parametrize("method", ["minmax", "quantile", "zscore"])
@pytest.mark.parametrize("use_quantiles", [False, True])
@pytest.mark.parametrize("marker", [None, "q01_q99"])
def test_explicit_formula_overrides_legacy_bool_and_file_marker(runtime, method, use_quantiles, marker):
    statistics = stats(runtime, marker=marker)
    sample = np.array([[2., 17., .5], [7., -3., -1.]], dtype=np.float64)
    mask = (True, False, True)
    selected = statistics["actions"]
    if method == "zscore":
        expected = (sample - selected.mean) / (selected.std + 1e-6)
    else:
        low, high = (selected.min, selected.max) if method == "minmax" else (selected.q01, selected.q99)
        expected = 2 * (sample - low) / (high - low + 1e-6) - 1
    expected[:, 1] = sample[:, 1]
    transform = runtime.Normalize(statistics, use_quantiles=use_quantiles,
                                  key_masks={key: mask for key in statistics}, method=method)
    result = transform({key: sample.copy() for key in statistics})
    for key in statistics:
        np.testing.assert_allclose(result[key], expected, rtol=1e-13, atol=1e-13)
    # Model-only padding has no physical units and must not acquire bounds.
    padding = np.array([[2.5, -7.], [9., .125]])
    padded = {key: np.concatenate([value, padding], axis=-1) for key, value in result.items()}
    inverse = runtime.Unnormalize(statistics, use_quantiles=not use_quantiles,
                                   key_masks={key: mask for key in statistics}, method=method)
    restored = inverse(padded)
    for value in restored.values():
        np.testing.assert_allclose(value[:, :3], sample, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(value[:, 3:], padding)


@pytest.mark.parametrize("method,fields", [("minmax", ("min", "max")), ("quantile", ("q01", "q99")),
                                          ("zscore", ("mean", "std"))])
def test_only_the_selected_bounds_are_required_and_missing_values_fail(runtime, method, fields):
    selected = stats(runtime)["state"]
    others = {"mean", "std", "min", "max", "q01", "q99"} - set(fields)
    selective = dataclasses.replace(selected, **{name: None for name in others})
    sample = np.ones((2, 3))
    runtime.Normalize({"state": selective}, use_quantiles=True, method=method)({"state": sample})
    for missing in fields:
        broken = dataclasses.replace(selective, **{missing: None})
        for transform in (runtime.Normalize, runtime.Unnormalize):
            with pytest.raises(ValueError, match="requires"):
                transform({"state": broken}, method=method)


@pytest.mark.parametrize("method", ["minmax", "quantile", "zscore"])
def test_short_statistics_cannot_broadcast_across_raw_dimensions(runtime, method):
    selected = stats(runtime)["state"]
    short = dataclasses.replace(selected, **{
        name: getattr(selected, name)[:1] for name in ("mean", "std", "min", "max", "q01", "q99")
    })
    with pytest.raises(ValueError, match="shorter than the raw input"):
        runtime.Normalize({"state": short}, method=method)({"state": np.ones((2, 3))})


@pytest.mark.parametrize("method", ["minmax", "quantile"])
def test_constant_dimension_uses_epsilon_and_no_clipping(runtime, method):
    statistics = stats(runtime)
    statistic = statistics["state"]
    if method == "minmax":
        statistics["state"] = dataclasses.replace(statistic, min=statistic.q01, max=statistic.q99)
    sample = np.array([[15., 5., 8.]])
    normalized = runtime.Normalize(statistics, method=method)({"state": sample})["state"]
    assert normalized[0, 1] == -1
    assert normalized[0, 0] > 1 and normalized[0, 2] > 1
    restored = runtime.Unnormalize(statistics, method=method)({"state": normalized})["state"]
    np.testing.assert_allclose(restored, sample, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("invalid", ["maxmin", "q01_q99", "", True, 1])
def test_invalid_method_never_silently_falls_back(runtime, invalid):
    for transform in (runtime.Normalize, runtime.Unnormalize):
        with pytest.raises(ValueError, match="method must"):
            transform(stats(runtime), method=invalid)


@pytest.mark.parametrize("method", ["minmax", "quantile", "zscore"])
def test_checkpoint_records_and_reloads_explicit_formula(runtime, tmp_path, method):
    config = runtime.DataConfig(asset_id="shared", norm_stats=stats(runtime), normalization_method=method)
    runtime._save_norm_assets(tmp_path, [config, config])
    document = json.loads((tmp_path / "shared/norm_stats.json").read_text())
    assert document["normalization_method"] == method
    loaded, chosen = runtime.normalization.load_with_method(tmp_path / "shared")
    assert chosen == method == runtime.normalization.load_method(tmp_path / "shared")
    np.testing.assert_array_equal(loaded["state"].q01, config.norm_stats["state"].q01)


def test_legacy_checkpoint_keeps_absent_formula_and_conflicting_methods_cannot_overwrite(runtime, tmp_path):
    config = runtime.DataConfig(asset_id="shared", norm_stats=stats(runtime))
    runtime._save_norm_assets(tmp_path, [config])
    assert "normalization_method" not in json.loads((tmp_path / "shared/norm_stats.json").read_text())
    assert runtime.normalization.load_method(tmp_path / "shared") is None
    conflicting = [dataclasses.replace(config, normalization_method=method) for method in ("minmax", "quantile")]
    with pytest.raises(ValueError, match="Conflicting"):
        runtime._save_norm_assets(tmp_path / "unpublished", conflicting)
    assert not (tmp_path / "unpublished").exists()
