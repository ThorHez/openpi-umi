"""Two-pass statistics of weighted raw samples, independent of training libraries.

Weights apply to every row from a source, rather than to a source's already
computed statistics. Quantiles use fixed, global histogram edges in the second
pass and are approximate to one histogram bin. No random sampling is involved.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import math
import numbers

import numpy as np


KEYS = ("state", "actions")


def _histogram_counts(values,edges,bins):
    """Exact explicit-edge histogram without sorting every small batch.

    The caller already checked global bounds. Match numpy.histogram's left
    inclusive bins and inclusive final endpoint, including repeated edges.
    """
    if np.any(edges[:-1]>edges[1:]):
        raise ValueError('Histogram bins must increase monotonically')
    indices=np.searchsorted(edges,values,side='right')-1
    np.minimum(indices,bins-1,out=indices)
    return np.bincount(indices,minlength=bins)


def _weights(values: Sequence[float], count: int) -> list[float]:
    if len(values) != count:
        raise ValueError("weights length must match the number of sources")
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError("Each source weight must be a positive finite number")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Each source weight must be a positive finite number")
        result.append(value)
    scale = max(result)
    scaled = [value / scale for value in result]
    if any(value == 0 for value in scaled):
        raise ValueError("Source weight dynamic range exceeds finite float64 precision")
    return scaled


def _batches(factory, widths, digest):
    for batch in factory():
        arrays = {}
        rows = None
        for key in KEYS:
            if key not in batch:
                raise ValueError(f"Statistics batch is missing {key!r}")
            raw = np.asarray(batch[key])
            if raw.ndim < 2 or any(size <= 0 for size in raw.shape):
                raise ValueError(f"{key} must contain a nonempty batch and feature dimension")
            if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
                raise ValueError(f"{key} must contain real numeric values")
            array = np.asarray(raw, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"{key} contains non-finite values")
            if rows is not None and rows != array.shape[0]:
                raise ValueError("state and actions must have the same number of rows")
            rows = array.shape[0]
            width = array.shape[-1]
            if key in widths and widths[key] != width:
                raise ValueError(f"Inconsistent {key} feature dimension: {width} != {widths[key]}")
            widths[key] = width
            digest.update(key.encode())
            digest.update(str(array.shape).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
            arrays[key] = array.reshape(-1, width)
        yield rows, arrays


def compute_weighted_statistics(
    source_factories: Sequence[Callable[[], Iterable[Mapping[str, np.ndarray]]]],
    weights: Sequence[float],
    *,
    quantile_bins: int = 5000,
    progress: Callable[[Mapping], None] | None = None,
    source_rows: Sequence[int] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, int]]]:
    """Return pooled statistics and unweighted source coverage counts.

    Each factory must produce the same finite batches on both calls. The first
    dimension is dataset rows and the last dimension is features; intermediate
    dimensions (for example action horizons) contribute additional vectors.
    A positive fractional weight is supported without rounding. Weights are
    rescaled by their common maximum to preserve the distribution for uniformly
    very large or small weights. Unrepresentable relative weights are rejected.
    Returned std
    is the population standard deviation. Source counts exclude the weights.
    """
    if not source_factories:
        raise ValueError("At least one source is required")
    if source_rows is not None and (len(source_rows) != len(source_factories)
            or any(isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0 for value in source_rows)):
        raise ValueError("source_rows must contain one positive integer per source")
    weights = _weights(weights, len(source_factories))
    if isinstance(quantile_bins, bool) or not isinstance(quantile_bins, int) or quantile_bins < 2:
        raise ValueError("quantile_bins must be an integer of at least two")

    widths = {}
    moments = {}
    counts = []
    digests = []
    def report(pass_index, source_index, rows_done, event):
        if progress is not None:
            totals = source_rows if source_rows is not None else [value['rows'] for value in counts] if pass_index == 2 else None
            preceding = sum(value['rows'] for value in counts[:source_index])
            progress({'pass':pass_index, 'passes':2, 'pass_name':'moments' if pass_index==1 else 'quantiles',
                'source_index':source_index, 'sources_total':len(source_factories), 'rows_done':rows_done,
                'rows_total':int(totals[source_index]) if totals is not None else None,
                'pass_rows_done':preceding+rows_done,
                'pass_rows_total':int(sum(totals)) if totals is not None else None, 'event':event})

    for index, (factory, weight) in enumerate(zip(source_factories, weights, strict=True)):
        source_count = {"rows": 0, "state_vectors": 0, "action_vectors": 0}
        digest = hashlib.sha256()
        report(1,index,0,'source_start')
        for rows, arrays in _batches(factory, widths, digest):
            source_count["rows"] += rows
            for key, array in arrays.items():
                source_count["state_vectors" if key == "state" else "action_vectors"] += len(array)
                mass = weight * len(array)
                if not math.isfinite(mass):
                    raise ValueError("Weighted vector count is not finite")
                mean = np.mean(array, axis=0, dtype=np.float64)
                centered = array - mean
                m2 = np.sum(centered * centered, axis=0, dtype=np.float64) * weight
                low, high = np.min(array, axis=0), np.max(array, axis=0)
                if key not in moments:
                    moments[key] = [mass, mean, m2, low, high]
                else:
                    old_mass, old_mean, old_m2, old_low, old_high = moments[key]
                    total = old_mass + mass
                    if not math.isfinite(total):
                        raise ValueError("Total weighted vector count is not finite")
                    delta = mean - old_mean
                    moments[key] = [
                        total,
                        old_mean + delta * (mass / total),
                        old_m2 + m2 + delta * delta * (old_mass / total) * mass,
                        np.minimum(old_low, low),
                        np.maximum(old_high, high),
                    ]
                if not all(np.isfinite(value).all() for value in moments[key][1:]):
                    raise ValueError(f"{key} statistics exceed finite float64 range")
            report(1,index,source_count['rows'],'batch')
        if source_count["rows"] == 0:
            raise ValueError("Every source must contain at least one row")
        if source_rows is not None and source_count['rows'] != source_rows[index]:
            raise ValueError("Statistics source row count differs from declared full coverage")
        report(1,index,source_count['rows'],'source_done')
        counts.append(source_count)
        digests.append(digest.hexdigest())

    edges = {}
    histograms = {}
    fractions = np.linspace(0.0, 1.0, quantile_bins + 1)
    for key, (_, _, _, low, high) in moments.items():
        # Convex interpolation avoids overflowing (high - low) for wide ranges.
        edges[key] = [
            (1 - fractions) * lo + fractions * hi if lo != hi else np.linspace(-0.5, 0.5, quantile_bins + 1)
            for lo, hi in zip(low, high, strict=True)
        ]
        histograms[key] = np.zeros((widths[key], quantile_bins), dtype=np.float64)

    for index, (factory, weight) in enumerate(zip(source_factories, weights, strict=True)):
        observed = {"rows": 0, "state_vectors": 0, "action_vectors": 0}
        digest = hashlib.sha256()
        report(2,index,0,'source_start')
        for rows, arrays in _batches(factory, widths, digest):
            observed["rows"] += rows
            for key, array in arrays.items():
                observed["state_vectors" if key == "state" else "action_vectors"] += len(array)
                low, high = moments[key][3:]
                if np.any(array < low) or np.any(array > high):
                    raise ValueError("Source values changed between statistics passes")
                for dim in range(widths[key]):
                    if low[dim] == high[dim]:
                        histograms[key][dim, 0] += len(array) * weight
                    else:
                        hist = _histogram_counts(array[:, dim],edges[key][dim],quantile_bins)
                        histograms[key][dim] += hist * weight
            report(2,index,observed['rows'],'batch')
        if observed != counts[index] or digest.hexdigest() != digests[index]:
            raise ValueError("Source rows or values changed between statistics passes")
        report(2,index,observed['rows'],'source_done')

    result = {}
    for key, (mass, mean, m2, low, high) in moments.items():
        quantiles = []
        for quantile in (0.01, 0.99):
            values = []
            for dim, histogram in enumerate(histograms[key]):
                if low[dim] == high[dim]:
                    values.append(low[dim])
                    continue
                bin_index = int(np.searchsorted(np.cumsum(histogram), quantile * mass))
                values.append(edges[key][dim][min(bin_index, quantile_bins - 1)])
            quantiles.append(np.asarray(values))
        result[key] = {
            "mean": mean,
            "std": np.sqrt(np.maximum(0, m2 / mass)),
            "min": low,
            "max": high,
            "q01": quantiles[0],
            "q99": quantiles[1],
        }
    return result, counts
