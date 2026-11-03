import numpy as np
import pytest

import openpi.shared.normalize as normalize


def test_normalize_update():
    arr = np.arange(12).reshape(4, 3)  # 4 vectors of length 3

    stats = normalize.RunningStats()
    for i in range(len(arr)):
        stats.update(arr[i : i + 1])  # Update with one vector at a time
    results = stats.get_statistics()

    assert np.allclose(results.mean, np.mean(arr, axis=0))
    assert np.allclose(results.std, np.std(arr, axis=0))


def test_serialize_deserialize():
    stats = normalize.RunningStats()
    stats.update(np.arange(12).reshape(4, 3))  # 4 vectors of length 3

    norm_stats = {"test": stats.get_statistics()}
    norm_stats2 = normalize.deserialize_json(normalize.serialize_json(norm_stats))
    assert np.allclose(norm_stats["test"].mean, norm_stats2["test"].mean)
    assert np.allclose(norm_stats["test"].std, norm_stats2["test"].std)


def test_multiple_batch_dimensions():
    # Test with multiple batch dimensions: (2, 3, 4) where 4 is vector dimension
    batch_shape = (2, 3, 4)
    arr = np.random.rand(*batch_shape)

    stats = normalize.RunningStats()
    stats.update(arr)  # Should handle (2, 3, 4) -> reshape to (6, 4)
    results = stats.get_statistics()

    # Flatten batch dimensions and compute expected stats
    flattened = arr.reshape(-1, arr.shape[-1])  # (6, 4)
    expected_mean = np.mean(flattened, axis=0)
    expected_std = np.std(flattened, axis=0)

    assert np.allclose(results.mean, expected_mean)
    assert np.allclose(results.std, expected_std)


@pytest.mark.parametrize("batch_size", [1, 7, 32, 128])
def test_float32_nearly_constant_vectors_keep_stable_population_variance(batch_size):
    offsets = np.arange(67) % 7 - 3
    values = np.stack(
        [4096 + offsets / 1024, -8192 + offsets / 2048, np.full(67, 2048)], axis=-1
    ).astype(np.float32)
    original = values.copy()
    expected = values.astype(np.float64)
    stats = normalize.RunningStats()
    for start in range(0, len(values), batch_size):
        # The extra dimension mirrors action chunks or state history.
        stats.update(values[start : start + batch_size, None, :])
    results = stats.get_statistics()

    np.testing.assert_allclose(results.mean, expected.mean(axis=0), rtol=0, atol=1e-10)
    np.testing.assert_allclose(results.std, expected.std(axis=0), rtol=0, atol=1e-10)
    assert results.std[0] > 0 and results.std[1] > 0 and results.std[2] == 0
    np.testing.assert_array_equal(results.min, expected.min(axis=0))
    np.testing.assert_array_equal(results.max, expected.max(axis=0))
    assert np.isfinite(results.q01).all() and np.isfinite(results.q99).all()
    assert np.all(results.q01 <= results.q99)
    np.testing.assert_array_equal(values, original)

    restored = normalize.deserialize_json(normalize.serialize_json({"state": results}))["state"]
    np.testing.assert_array_equal(restored.std, results.std)
    assert restored.quantile_method is None
