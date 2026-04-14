import json
import pathlib

import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    max: numpydantic.NDArray | None = None
    min: numpydantic.NDArray | None = None
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99, max=self._max, min=self._min)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())


def merge_norm_stats(
    stats_list: list[NormStats],
    weights: list[float] | None = None,
) -> NormStats:
    """Merge multiple NormStats into one (e.g. for multi-dataset training).

    Uses weighted average for mean, and combined variance for std:
    Var(X) = E[X^2] - E[X]^2, E[X^2] = Var(X) + E[X]^2.
    min/max are element-wise min/max across all; q01/q99 are weighted averages.
    If weights is None, uses equal weights.
    """
    if not stats_list:
        raise ValueError("stats_list must be non-empty")
    n = len(stats_list)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError("weights length must match stats_list")
    w_sum = sum(weights)
    if w_sum <= 0:
        raise ValueError("weights must sum to a positive value")

    mean = np.asarray(stats_list[0].mean)
    std = np.asarray(stats_list[0].std)
    merged_mean = weights[0] * mean
    # E[X^2] = std^2 + mean^2
    merged_ex2 = weights[0] * (std**2 + mean**2)
    merged_min = np.asarray(stats_list[0].min).copy() if stats_list[0].min is not None else None
    merged_max = np.asarray(stats_list[0].max).copy() if stats_list[0].max is not None else None
    q01_sum = weights[0] * np.asarray(stats_list[0].q01) if stats_list[0].q01 is not None else None
    q99_sum = weights[0] * np.asarray(stats_list[0].q99) if stats_list[0].q99 is not None else None
    q01_w = weights[0] if stats_list[0].q01 is not None else 0.0
    q99_w = weights[0] if stats_list[0].q99 is not None else 0.0

    for i in range(1, n):
        s = stats_list[i]
        m = np.asarray(s.mean)
        v = np.asarray(s.std)
        merged_mean += weights[i] * m
        merged_ex2 += weights[i] * (v**2 + m**2)
        if s.min is not None:
            merged_min = np.minimum(merged_min, np.asarray(s.min)) if merged_min is not None else np.asarray(s.min).copy()
        if s.max is not None:
            merged_max = np.maximum(merged_max, np.asarray(s.max)) if merged_max is not None else np.asarray(s.max).copy()
        if s.q01 is not None:
            arr = np.asarray(s.q01)
            q01_sum = q01_sum + weights[i] * arr if q01_sum is not None else weights[i] * arr
            q01_w += weights[i]
        if s.q99 is not None:
            arr = np.asarray(s.q99)
            q99_sum = q99_sum + weights[i] * arr if q99_sum is not None else weights[i] * arr
            q99_w += weights[i]

    merged_mean = merged_mean / w_sum
    merged_var = np.maximum(0.0, merged_ex2 / w_sum - merged_mean**2)
    merged_std = np.sqrt(merged_var)
    merged_q01 = (q01_sum / q01_w) if q01_sum is not None and q01_w > 0 else None
    merged_q99 = (q99_sum / q99_w) if q99_sum is not None and q99_w > 0 else None

    return NormStats(
        mean=merged_mean,
        std=merged_std,
        min=merged_min,
        max=merged_max,
        q01=merged_q01,
        q99=merged_q99,
    )


def merge_norm_stats_dict(
    stats_dict_list: list[dict[str, NormStats]],
    weights: list[float] | None = None,
) -> dict[str, NormStats]:
    """Merge per-key norm_stats from multiple datasets into one dict.

    For each key (e.g. 'state', 'actions'), merges the NormStats from all dicts
    that have that key. Returns a single dict[str, NormStats] suitable for
    use as norm_stats for all datasets in multi-dataset training.
    """
    if not stats_dict_list:
        raise ValueError("stats_dict_list must be non-empty")
    n = len(stats_dict_list)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError("weights length must match stats_dict_list")

    all_keys = set()
    for d in stats_dict_list:
        all_keys.update(d.keys())
    result = {}
    for key in all_keys:
        per_key = [d[key] for d in stats_dict_list if key in d and d[key] is not None]
        if not per_key:
            continue
        # Use same weights for each key (by dataset index)
        indices = [i for i, d in enumerate(stats_dict_list) if key in d and d[key] is not None]
        w = [weights[i] for i in indices]
        result[key] = merge_norm_stats(per_key, weights=w)
    return result
