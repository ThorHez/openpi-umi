"""Framework-independent reward and advantage calculations for ReCap."""

import numpy as np


def compute_dense_rewards_from_targets(
    targets: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Compute per-frame rewards as the difference in consecutive value targets.

    reward[i] = target[i] - target[i+1]  (if i+1 is next frame in same episode)
    reward[i] = target[i]                (at episode boundary / last frame)
    """
    n = targets.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    same_ep = episode_indices[:-1] == episode_indices[1:]
    contiguous = frame_indices[1:] == frame_indices[:-1] + 1
    is_next = same_ep & contiguous

    rewards = targets.copy().astype(np.float32)
    rewards[:-1] = np.where(is_next, targets[:-1] - targets[1:], targets[:-1])
    return rewards

def compute_n_step_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    n_step: int,
) -> np.ndarray:
    """N-step advantage: A(i) = sum_{k=0}^{n-1} r(i+k) + V(i+n) - V(i).

    Bootstrap with V(i+n) only if i+n is in the same episode and contiguous.
    """
    if n_step <= 0:
        raise ValueError("n_step must be > 0.")

    n = rewards.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # Precompute contiguity mask between consecutive frames.
    same_ep = np.empty(n, dtype=np.bool_)
    same_ep[:-1] = (episode_indices[:-1] == episode_indices[1:]) & (frame_indices[1:] == frame_indices[:-1] + 1)
    same_ep[-1] = False

    # Build cumulative contiguity: how many contiguous frames follow frame i within its episode.
    # max_reach[i] = max k such that frames i..i+k-1 are all contiguous and same episode.
    max_reach = np.zeros(n, dtype=np.int64)
    max_reach[-1] = 1
    for i in range(n - 2, -1, -1):
        if same_ep[i]:
            max_reach[i] = max_reach[i + 1] + 1
        else:
            max_reach[i] = 1

    # Prefix sum of rewards for sliding-window sums.
    cum_rewards = np.zeros(n + 1, dtype=np.float64)
    cum_rewards[1:] = np.cumsum(rewards.astype(np.float64))

    # Clamp effective step count to available contiguous frames.
    effective_n = np.minimum(n_step, max_reach).astype(np.int64)
    idx = np.arange(n, dtype=np.int64)

    reward_sum = (cum_rewards[idx + effective_n] - cum_rewards[idx]).astype(np.float32)

    # Bootstrap: use V(i+n_step) only if we had the full n_step contiguous frames AND i+n_step is still contiguous.
    can_bootstrap = (effective_n == n_step) & ((idx + n_step) < n)
    can_bootstrap &= (episode_indices[np.minimum(idx + n_step, n - 1)] == episode_indices[idx])
    can_bootstrap &= (frame_indices[np.minimum(idx + n_step, n - 1)] == frame_indices[idx] + n_step)

    bootstrap = np.where(can_bootstrap, values[np.minimum(idx + n_step, n - 1)], 0.0).astype(np.float32)

    return reward_sum + bootstrap - values.astype(np.float32)

def compute_task_thresholds(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    positive_ratio: float,
) -> dict[int, float]:
    """Per-task quantile threshold: frames above threshold are 'positive'."""
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("positive_ratio must be in [0, 1].")

    quantile = 1.0 - positive_ratio
    thresholds: dict[int, float] = {}
    for task_idx in np.unique(task_indices):
        task_adv = advantages[task_indices == task_idx]
        if task_adv.size == 0:
            thresholds[int(task_idx)] = float("inf")
        else:
            thresholds[int(task_idx)] = float(np.quantile(task_adv, quantile))
    return thresholds

def binarize_advantages(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    thresholds: dict[int, float],
) -> np.ndarray:
    """Assign indicator=1 (positive) if advantage >= threshold, else 0."""
    threshold_arr = np.full(advantages.shape[0], np.inf, dtype=np.float32)
    for task_idx, thr in thresholds.items():
        mask = task_indices == task_idx
        threshold_arr[mask] = thr
    return (advantages >= threshold_arr).astype(np.int64)

def label_value_advantages(values, targets, episodes, frames, tasks, *, n_step, positive_ratio):
    """Score windows in trajectory order, then return labels in source row order."""
    if (values.ndim != 1 or not len(values) or not all(array.shape == values.shape for array in
            (targets, episodes, frames, tasks)) or not np.isfinite(values).all() or not np.isfinite(targets).all()):
        raise ValueError("价值标注要求每帧具有有限的预测值、目标值和对应索引")
    order = np.lexsort((frames, episodes))
    sorted_episodes, sorted_frames = episodes[order], frames[order]
    if np.any((sorted_episodes[1:] == sorted_episodes[:-1]) & (sorted_frames[1:] == sorted_frames[:-1])):
        raise ValueError("价值标注数据包含重复的 episode_index / frame_index")
    rewards = compute_dense_rewards_from_targets(targets[order], sorted_episodes, sorted_frames)
    advantages = np.empty(len(values), dtype=np.float32)
    advantages[order] = compute_n_step_advantages(rewards, values[order], sorted_episodes, sorted_frames, n_step)
    thresholds = compute_task_thresholds(tasks, advantages, positive_ratio)
    return advantages, binarize_advantages(tasks, advantages, thresholds), thresholds


def load_existing_value_targets(
    raw_frames,
    field_name: str,
) -> np.ndarray:
    """
    Load existing value_target from hf dataset.

    Supports common storage layouts:
    - scalar column: shape [N]
    - vector column with one element: shape [N, 1]
    """
    column_names = getattr(raw_frames, "column_names", None)
    if column_names is None:
        try:
            column_names = list(raw_frames.features.keys())
        except Exception:
            column_names = []

    if field_name not in column_names:
        raise KeyError(
            f"Dataset does not contain precomputed '{field_name}' column. "
            f"Available columns: {column_names}"
        )

    arr = np.asarray(raw_frames[field_name], dtype=np.float32)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].astype(np.float32, copy=False)

    # Fallback: flatten if user stored shape (N, 1, 1) etc.
    if arr.shape[0] > 0:
        reshaped = arr.reshape(arr.shape[0], -1)
        if reshaped.shape[1] == 1:
            return reshaped[:, 0].astype(np.float32, copy=False)

    raise ValueError(
        f"Unsupported shape for '{field_name}': {arr.shape}. "
        "Expected [N] or [N,1]."
    )
