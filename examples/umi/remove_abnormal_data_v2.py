#!/usr/bin/env python3
"""
Remove abnormal frames from UMI LeRobot dataset + (IMPROVED) action outlier detection/smoothing
based on HORIZON-dimension variability.

Your dataset facts:
- actions column name: "actions"
- actions dtype in parquet: list<element: list<element: float>>
- actions shape per episode: [B, H, 20]  (B=frames in episode, H=fixed horizon, D=20 dual-arm)

Key change vs previous version:
- We DO NOT judge outliers by comparing along B for each horizon step.
- We judge outliers by comparing along horizon H *within each frame*:
    For each time frame t, actions[t] is a sequence over k=0..H-1.
    We detect "too much variation" along k (horizon) via robust MAD thresholds on
    first-differences along k, separately for:
        - translation delta xyz (per arm)
        - rotation rot6d (per arm, using SO(3) geodesic angle)
        - gripper (per arm)
    Then we can smooth short abnormal horizon segments by interpolation along k.

Behavior:
- If too many frames in an episode have high horizon-bad-ratio, episode can be dropped
  (configurable).
- Otherwise, we smooth per-frame horizon spikes.

Smoothing method along horizon k:
- delta xyz: linear interpolation
- rot6d: SO(3) slerp (rot6d -> R -> quat -> slerp -> R -> rot6d)
- gripper: linear interpolation

Usage example:
  python remove_abnormal_data.py --input_dir dataset --output_dir cleaned_dataset \
    --features observation.robot0_eef_pos --threshold 1.5 \
    --fix_action=true --action_column actions \
    --action_k_mad 10 --action_max_gap 3 \
    --episode_bad_ratio_thr 0.5 --frame_bad_ratio_thr 0.6 --action_on_bad_episode drop_episode

Notes:
- If you don't want to drop episodes at all, set: --action_on_bad_episode keep
- If you want only smoothing and never dropping, set:
    --action_on_bad_episode keep --episode_bad_ratio_thr 1.0
"""

import argparse
import io
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq


# ===================== Episode stats =====================
def compute_episode_stats(table: pa.Table) -> Dict[str, Any]:
    stats = {}
    for col_name in table.column_names:
        col = table.column(col_name)
        try:
            data = np.array([row.as_py() for row in col])
        except (ValueError, TypeError):
            continue

        if data.dtype == object:
            continue

        if len(data.shape) == 1:
            stats[col_name] = {
                "min": [float(np.min(data))],
                "max": [float(np.max(data))],
                "mean": [float(np.mean(data))],
                "std": [float(np.std(data))],
                "count": [int(len(data))],
            }
        elif len(data.shape) == 2:
            stats[col_name] = {
                "min": data.min(axis=0).tolist(),
                "max": data.max(axis=0).tolist(),
                "mean": data.mean(axis=0).tolist(),
                "std": data.std(axis=0).tolist(),
                "count": [int(len(data))],
            }
        elif len(data.shape) == 3:
            min_vals, max_vals, mean_vals, std_vals = [], [], [], []
            for t in range(data.shape[1]):
                min_vals.append(data[:, t, :].min(axis=0).tolist())
                max_vals.append(data[:, t, :].max(axis=0).tolist())
                mean_vals.append(data[:, t, :].mean(axis=0).tolist())
                std_vals.append(data[:, t, :].std(axis=0).tolist())
            stats[col_name] = {
                "min": min_vals,
                "max": max_vals,
                "mean": mean_vals,
                "std": std_vals,
                "count": [int(len(data))],
            }
        elif len(data.shape) == 4:
            min_vals, max_vals, mean_vals, std_vals = [], [], [], []
            for c in range(data.shape[-1]):
                channel_data = data[..., c]
                min_vals.append([[float(np.min(channel_data))]])
                max_vals.append([[float(np.max(channel_data))]])
                mean_vals.append([[float(np.mean(channel_data))]])
                std_vals.append([[float(np.std(channel_data))]])
            stats[col_name] = {
                "min": min_vals,
                "max": max_vals,
                "mean": mean_vals,
                "std": std_vals,
                "count": [int(len(data))],
            }
    return stats


# ===================== Dataset loaders =====================
def load_minmax_stats(dataset_path: Path) -> Dict[str, Any]:
    p = dataset_path / "minmax_stats.json"
    if not p.exists():
        raise FileNotFoundError(f"minmax_stats.json not found at {p}. Run compute_minmax_stats.py first.")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_info(dataset_path: Path) -> Dict[str, Any]:
    p = dataset_path / "meta" / "info.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_episodes(dataset_path: Path) -> List[Dict[str, Any]]:
    p = dataset_path / "meta" / "episodes.jsonl"
    eps = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eps.append(json.loads(line))
    return eps


def load_tasks(dataset_path: Path) -> List[Dict[str, Any]]:
    p = dataset_path / "meta" / "tasks.jsonl"
    tasks = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def _flatten_nested(nested):
    out = []
    if isinstance(nested, (list, tuple)):
        for item in nested:
            out.extend(_flatten_nested(item))
    else:
        out.append(nested)
    return out


# ===================== Empty image detection =====================
def get_image_columns(table: pa.Table) -> List[str]:
    image_columns = []
    image_patterns = ["rgb", "image", "camera", "wrist"]
    for col_name in table.column_names:
        if any(p in col_name.lower() for p in image_patterns):
            try:
                sample = table.column(col_name)[0].as_py()
                if isinstance(sample, dict) and "bytes" in sample:
                    image_columns.append(col_name)
            except (IndexError, TypeError):
                continue
    return image_columns


def check_episode_has_empty_images(table: pa.Table, image_columns: List[str] = None) -> Tuple[bool, Dict]:
    if image_columns is None:
        image_columns = get_image_columns(table)
    if not image_columns:
        return False, {"message": "No image columns found"}

    empty_columns = []
    column_stats = {}

    for col_name in image_columns:
        col = table.column(col_name)
        max_values = []
        for row_idx in range(table.num_rows):
            try:
                img_data = col[row_idx].as_py()
                if img_data is not None and "bytes" in img_data:
                    img = Image.open(io.BytesIO(img_data["bytes"]))
                    arr = np.array(img)
                    mv = int(arr.max())
                    max_values.append(mv)
                    if mv != 0:
                        break
            except Exception:
                continue

        if max_values:
            overall_max = max(max_values)
            column_stats[col_name] = {"checked_frames": len(max_values), "overall_max": overall_max}
            if overall_max == 0:
                empty_columns.append(col_name)

    is_empty = len(empty_columns) > 0
    return is_empty, {
        "empty_columns": empty_columns,
        "column_stats": column_stats,
        "total_image_columns": len(image_columns),
    }


# ===================== Frame abnormal detection (q01/q99) =====================
def detect_abnormal_frames(
    table: pa.Table,
    feature_stats: Dict[str, Dict[str, Any]],
    threshold_multiplier: float = 1.5,
) -> Tuple[List[int], List[Dict]]:
    abnormal_indices = []
    abnormal_info = []

    for row_idx in range(table.num_rows):
        is_abnormal = False
        abnormal_details = []

        for feature_name, stats in feature_stats.items():
            if feature_name not in table.column_names:
                continue

            col = table.column(feature_name)
            data = col[row_idx].as_py()

            q01_per_dim = stats.get("q01_per_dim")
            q99_per_dim = stats.get("q99_per_dim")
            if q01_per_dim is None:
                q01_per_dim = [[stats.get("q01", 0)]]
                q99_per_dim = [[stats.get("q99", 0)]]

            flat_data = _flatten_nested(data)
            flat_q01 = _flatten_nested(q01_per_dim)
            flat_q99 = _flatten_nested(q99_per_dim)

            for dim_idx, value in enumerate(flat_data):
                if dim_idx >= len(flat_q01) or dim_idx >= len(flat_q99):
                    continue

                q01_val = flat_q01[dim_idx]
                q99_val = flat_q99[dim_idx]

                if q99_val >= 0:
                    upper_threshold = q99_val * threshold_multiplier
                else:
                    upper_threshold = q99_val / threshold_multiplier

                if q01_val <= 0:
                    lower_threshold = q01_val * threshold_multiplier
                else:
                    lower_threshold = q01_val / threshold_multiplier

                if value > upper_threshold:
                    is_abnormal = True
                    abnormal_details.append({
                        "feature": feature_name,
                        "dim": dim_idx,
                        "value": value,
                        "type": "too_high",
                        "threshold": upper_threshold,
                        "q99": q99_val,
                    })
                elif value < lower_threshold:
                    is_abnormal = True
                    abnormal_details.append({
                        "feature": feature_name,
                        "dim": dim_idx,
                        "value": value,
                        "type": "too_low",
                        "threshold": lower_threshold,
                        "q01": q01_val,
                    })

        if is_abnormal:
            abnormal_indices.append(row_idx)
            abnormal_info.append({"frame_index": row_idx, "abnormal_details": abnormal_details})

    return abnormal_indices, abnormal_info


# ===================== Action horizon variability: detect + smooth =====================
EPS = 1e-8

def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return np.median(np.abs(x - med)) + EPS

def _thr_mad(x: np.ndarray, k_mad: float) -> float:
    return float(np.median(x) + k_mad * _mad(x))

def _np_norm(x, axis=-1):
    return np.sqrt(np.sum(np.square(x), axis=axis))

def rot6d_to_R(r6):
    """Zhou 6D -> rotation matrix (3,3). r6 shape (6,)"""
    a = r6[:3]
    b = r6[3:6]
    u1 = a / (np.linalg.norm(a) + EPS)
    b_orth = b - np.dot(u1, b) * u1
    u2 = b_orth / (np.linalg.norm(b_orth) + EPS)
    u3 = np.cross(u1, u2)
    return np.stack([u1, u2, u3], axis=1)  # columns

def R_to_rot6d(R):
    return np.concatenate([R[:, 0], R[:, 1]], axis=0)

def R_to_quat(R):
    """(3,3) -> quat [w,x,y,z]"""
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + EPS)

def quat_to_R(q):
    q = q / (np.linalg.norm(q) + EPS)
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=np.float64)

def quat_slerp(q0, q1, a):
    q0 = q0 / (np.linalg.norm(q0) + EPS)
    q1 = q1 / (np.linalg.norm(q1) + EPS)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / (np.linalg.norm(q) + EPS)
    omega = np.arccos(np.clip(dot, -1.0, 1.0))
    so = np.sin(omega)
    w0 = np.sin((1.0 - a) * omega) / (so + EPS)
    w1 = np.sin(a * omega) / (so + EPS)
    return w0 * q0 + w1 * q1

def rot6d_slerp(r6_a, r6_b, alpha):
    qa = R_to_quat(rot6d_to_R(np.asarray(r6_a, dtype=np.float64)))
    qb = R_to_quat(rot6d_to_R(np.asarray(r6_b, dtype=np.float64)))
    q = quat_slerp(qa, qb, float(alpha))
    return R_to_rot6d(quat_to_R(q))

def quat_geodesic_angle(q0, q1):
    q0 = q0 / (np.linalg.norm(q0) + EPS)
    q1 = q1 / (np.linalg.norm(q1) + EPS)
    d = abs(float(np.dot(q0, q1)))
    d = np.clip(d, -1.0, 1.0)
    return 2.0 * np.arccos(d)

def _extract_arm(a_k: np.ndarray, arm: int, arm_dim: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
    off = arm * arm_dim
    dp = a_k[off:off+3]
    r6 = a_k[off+3:off+9]
    g = float(a_k[off+9])
    return dp, r6, g

def detect_bad_horizon_steps_for_frame(
    A: np.ndarray,         # (H,D)
    D: int,
    k_mad: float,
    arm_dim: int = 10,
) -> np.ndarray:
    """
    Return bad_k mask for k=1..H-1 (length H-1) based on horizon-variation spikes.
    bad_k[idx] corresponds to step k=idx+1 in horizon.
    """
    H = A.shape[0]
    if H < 2:
        return np.zeros(0, dtype=bool)
    if D not in (arm_dim, 2 * arm_dim):
        return np.zeros(H-1, dtype=bool)

    n_arms = D // arm_dim

    # compute horizon-differences metrics per arm, then combine by max across arms
    v_pos = np.zeros(H-1, dtype=np.float64)
    v_rot = np.zeros(H-1, dtype=np.float64)
    v_grip = np.zeros(H-1, dtype=np.float64)

    for arm in range(n_arms):
        dp_prev, r6_prev, g_prev = _extract_arm(A[0], arm, arm_dim)
        q_prev = R_to_quat(rot6d_to_R(r6_prev))

        v_pos_arm = np.zeros(H-1, dtype=np.float64)
        v_rot_arm = np.zeros(H-1, dtype=np.float64)
        v_grip_arm = np.zeros(H-1, dtype=np.float64)

        for k in range(1, H):
            dp_k, r6_k, g_k = _extract_arm(A[k], arm, arm_dim)
            q_k = R_to_quat(rot6d_to_R(r6_k))

            v_pos_arm[k-1] = float(np.linalg.norm(dp_k - dp_prev))
            v_rot_arm[k-1] = float(quat_geodesic_angle(q_prev, q_k))
            v_grip_arm[k-1] = float(abs(g_k - g_prev))

            dp_prev, q_prev, g_prev = dp_k, q_k, g_k

        v_pos = np.maximum(v_pos, v_pos_arm)
        v_rot = np.maximum(v_rot, v_rot_arm)
        v_grip = np.maximum(v_grip, v_grip_arm)

    thr_pos = _thr_mad(v_pos, k_mad)
    thr_rot = _thr_mad(v_rot, k_mad)
    thr_grip = _thr_mad(v_grip, k_mad)

    bad = (v_pos > thr_pos) | (v_rot > thr_rot) | (v_grip > thr_grip)
    return bad

def smooth_frame_actions_along_horizon(
    A: np.ndarray,         # (H,D)
    bad_k: np.ndarray,     # (H-1,) bad for steps k=1..H-1
    max_gap: int,
    arm_dim: int = 10,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Smooth short contiguous bad segments along horizon axis (k).
    We treat bad_k indices as boundaries at steps k=idx+1.
    We'll smooth the action rows A[k] for those k in bad segments, using neighbors.
    """
    A = A.copy()
    H, D = A.shape
    if H < 2 or bad_k.size != H-1:
        return A, {"repaired_segments": 0, "dropped_segments": 0}

    n_arms = D // arm_dim if D in (arm_dim, 2*arm_dim) else 0
    if n_arms == 0:
        return A, {"repaired_segments": 0, "dropped_segments": 0}

    # convert bad_k (for k>=1) into bad_step_mask over k=0..H-1 (we don't mark k=0)
    bad_step = np.zeros(H, dtype=bool)
    bad_step[1:] = bad_k

    repaired = 0
    dropped = 0

    k = 1
    while k < H:
        if not bad_step[k]:
            k += 1
            continue
        j = k
        while j < H and bad_step[j]:
            j += 1
        seg_len = j - k  # bad steps are k..j-1

        L = k - 1
        R = j  # first good after segment
        # need both sides valid, and segment short
        if L >= 0 and R < H and (not bad_step[L]) and (not bad_step[R]) and seg_len <= max_gap:
            for kk in range(k, j):
                alpha = (kk - L) / (R - L)
                for arm in range(n_arms):
                    off = arm * arm_dim
                    aL = A[L, off:off+arm_dim]
                    aR = A[R, off:off+arm_dim]

                    dp = (1 - alpha) * aL[:3] + alpha * aR[:3]
                    r6 = rot6d_slerp(aL[3:9], aR[3:9], alpha)
                    g = (1 - alpha) * aL[9] + alpha * aR[9]

                    A[kk, off:off+3] = dp
                    A[kk, off+3:off+9] = r6
                    A[kk, off+9] = g
            repaired += 1
        else:
            dropped += 1

        k = j

    return A, {"repaired_segments": repaired, "dropped_segments": dropped}


def fix_actions_horizon_variability(
    actions: np.ndarray,     # (B,H,D)
    k_mad: float,
    max_gap: int,
    frame_bad_ratio_thr: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    For each frame t, detect horizon spikes and smooth short segments.
    Returns fixed actions and per-episode stats:
      - per_frame_bad_ratio (len B)
      - episode_bad_ratio (mean)
      - num_frames_high_bad (count frame_bad_ratio > frame_bad_ratio_thr)
    """
    actions = actions.copy()
    B, H, D = actions.shape

    per_frame_bad_ratio = np.zeros(B, dtype=np.float64)
    repaired_segments_total = 0
    dropped_segments_total = 0

    for t in range(B):
        A = actions[t]  # (H,D)
        if not np.isfinite(A).all():
            # if NaN/inf, mark frame as fully bad; don't try to repair
            per_frame_bad_ratio[t] = 1.0
            continue

        bad_k = detect_bad_horizon_steps_for_frame(A, D=D, k_mad=k_mad)
        if bad_k.size == 0:
            per_frame_bad_ratio[t] = 0.0
            continue

        per_frame_bad_ratio[t] = float(bad_k.mean())  # / (H-1)

        # Smooth only if this frame isn't overwhelmingly bad; otherwise skip smoothing
        # (you can change this policy if you want)
        A_fixed, rep = smooth_frame_actions_along_horizon(A, bad_k, max_gap=max_gap)
        actions[t] = A_fixed
        repaired_segments_total += int(rep.get("repaired_segments", 0))
        dropped_segments_total += int(rep.get("dropped_segments", 0))

    episode_bad_ratio = float(per_frame_bad_ratio.mean()) if B > 0 else 0.0
    num_frames_high_bad = int(np.sum(per_frame_bad_ratio > frame_bad_ratio_thr))

    return actions, {
        "episode_bad_ratio": episode_bad_ratio,
        "per_frame_bad_ratio_mean": float(per_frame_bad_ratio.mean()) if B > 0 else 0.0,
        "per_frame_bad_ratio_max": float(per_frame_bad_ratio.max()) if B > 0 else 0.0,
        "num_frames_high_bad": num_frames_high_bad,
        "repaired_segments_total": repaired_segments_total,
        "dropped_segments_total": dropped_segments_total,
    }


def _make_actions_array_like_3d_listlistfloat(original_type: pa.DataType, data_3d: np.ndarray) -> pa.Array:
    # Your confirmed type: list<list<float>>
    return pa.array(data_3d.astype(np.float32).tolist(), type=original_type)


# ===================== Main cleaning function =====================
def remove_abnormal_frames(
    input_dir: str,
    output_dir: str,
    features: List[str],
    threshold_multiplier: float = 1.5,
    min_episode_length: int = 50,
    remove_mode: str = "frame",
    remove_empty_images: bool = False,
    image_columns: List[str] = None,
    verbose: bool = True,
    # action fix (horizon variability)
    action_column: str = "actions",
    fix_action: bool = True,
    action_k_mad: float = 10.0,
    action_max_gap: int = 3,
    frame_bad_ratio_thr: float = 0.6,
    episode_bad_ratio_thr: float = 0.5,
    action_on_bad_episode: str = "drop_episode",  # keep|drop_episode
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    minmax_stats = load_minmax_stats(input_path)
    info = load_info(input_path)
    episodes = load_episodes(input_path)
    tasks = load_tasks(input_path)

    fps = info.get("fps", 20)

    feature_stats = {}
    for feat in features:
        if feat in minmax_stats:
            feature_stats[feat] = minmax_stats[feat]
        else:
            print(f"⚠️  Warning: Feature '{feat}' not found in minmax_stats.json, skipping...")

    if not feature_stats and not remove_empty_images and not fix_action:
        raise ValueError("No valid features to check and action fix disabled. Nothing to do.")

    if verbose:
        print(f"Dataset: {input_path}")
        print(f"Total episodes: {len(episodes)}")
        print(f"FPS: {fps}")
        print(f"Remove mode: {remove_mode}")
        print(f"Threshold multiplier: {threshold_multiplier}")
        print(f"Remove empty images: {remove_empty_images}")
        print(
            f"Fix action(horizon): {fix_action} (col='{action_column}', k_mad={action_k_mad}, "
            f"max_gap={action_max_gap}, frame_bad_thr={frame_bad_ratio_thr}, episode_bad_thr={episode_bad_ratio_thr}, "
            f"on_bad_episode={action_on_bad_episode})"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    total_abnormal_frames = 0
    total_original_frames = 0
    removed_episodes = 0
    removed_empty_image_episodes = 0
    all_abnormal_info: List[Dict] = []
    empty_image_episodes_info: List[Dict] = []
    action_fix_info: List[Dict] = []

    new_episode_index = 0
    new_global_index = 0
    new_episodes = []

    if verbose:
        print("\n🔄 Processing episodes...")

    for episode in tqdm(episodes, desc="Processing episodes", disable=not verbose):
        old_episode_idx = episode["episode_index"]
        chunk_idx = old_episode_idx // 1000
        parquet_path = input_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{old_episode_idx:06d}.parquet"
        if not parquet_path.exists():
            continue

        table = pq.read_table(parquet_path)
        original_num_rows = table.num_rows
        total_original_frames += original_num_rows

        # 1) empty images
        if remove_empty_images:
            is_empty, empty_info = check_episode_has_empty_images(table, image_columns)
            if is_empty:
                removed_empty_image_episodes += 1
                empty_image_episodes_info.append({
                    "episode_index": old_episode_idx,
                    "empty_columns": empty_info.get("empty_columns", []),
                    "column_stats": empty_info.get("column_stats", {}),
                })
                if verbose:
                    print(f"  [空图片] Episode {old_episode_idx}: 空列={empty_info.get('empty_columns', [])}")
                continue

        # 2) action fix (horizon variability)
        if fix_action and (action_column in table.column_names):
            act_col = table.column(action_column)
            act_list = [act_col[i].as_py() for i in range(table.num_rows)]
            actions = np.asarray(act_list, dtype=np.float64)

            if actions.ndim == 3 and actions.shape[2] in (10, 20):
                B, H, D = actions.shape
                original_type = table.schema.field(action_column).type

                actions_fixed, ainfo = fix_actions_horizon_variability(
                    actions,
                    k_mad=action_k_mad,
                    max_gap=action_max_gap,
                    frame_bad_ratio_thr=frame_bad_ratio_thr,
                )

                if (ainfo["episode_bad_ratio"] > episode_bad_ratio_thr) and (action_on_bad_episode == "drop_episode"):
                    removed_episodes += 1
                    if verbose:
                        print(f"  [Action坏] Episode {old_episode_idx}: episode_bad_ratio={ainfo['episode_bad_ratio']:.3f} -> drop episode")
                    action_fix_info.append({
                        "episode_index": old_episode_idx,
                        "shape": list(actions.shape),
                        **ainfo,
                        "dropped_episode": True,
                    })
                    continue

                # write back
                col_idx = table.schema.get_field_index(action_column)
                new_act_arr = _make_actions_array_like_3d_listlistfloat(original_type, actions_fixed)
                table = table.set_column(col_idx, action_column, new_act_arr)

                if verbose and (ainfo["repaired_segments_total"] > 0 or ainfo["dropped_segments_total"] > 0):
                    print(
                        f"  [Action修复] Episode {old_episode_idx}: shape={actions.shape}, "
                        f"episode_bad_ratio={ainfo['episode_bad_ratio']:.3f}, "
                        f"frames_high_bad={ainfo['num_frames_high_bad']}, "
                        f"repaired_segs={ainfo['repaired_segments_total']}, dropped_segs={ainfo['dropped_segments_total']}"
                    )

                action_fix_info.append({
                    "episode_index": old_episode_idx,
                    "shape": list(actions.shape),
                    **ainfo,
                    "dropped_episode": False,
                })
            else:
                if verbose:
                    print(f"⚠️  Skip action fix: '{action_column}' shape {actions.shape} (expect [B,H,10] or [B,H,20])")

        # 3) detect abnormal frames by q01/q99 and remove
        abnormal_indices, abnormal_info = (
            detect_abnormal_frames(table, feature_stats, threshold_multiplier)
            if feature_stats
            else ([], [])
        )

        if abnormal_indices:
            total_abnormal_frames += len(abnormal_indices)
            all_abnormal_info.append({
                "episode_index": old_episode_idx,
                "num_abnormal_frames": len(abnormal_indices),
                "abnormal_frames": abnormal_info,
            })
            if remove_mode == "episode":
                removed_episodes += 1
                continue

        abnormal_set = set(abnormal_indices)
        normal_indices = [i for i in range(original_num_rows) if i not in abnormal_set]

        if len(normal_indices) < min_episode_length:
            removed_episodes += 1
            continue

        if abnormal_indices and remove_mode == "frame":
            table = table.take(normal_indices)

        num_rows = table.num_rows

        # 4) update indices/timestamps
        schema = table.schema
        new_columns = {}
        for i, field in enumerate(schema):
            col_name = field.name
            col = table.column(i)
            if col_name == "episode_index":
                new_columns[col_name] = pa.array([new_episode_index] * num_rows, type=pa.int64())
            elif col_name == "frame_index":
                new_columns[col_name] = pa.array(range(num_rows), type=pa.int64())
            elif col_name == "index":
                new_columns[col_name] = pa.array(
                    range(new_global_index, new_global_index + num_rows),
                    type=pa.int64(),
                )
            elif col_name == "timestamp":
                new_timestamps = [i / fps for i in range(num_rows)]
                new_columns[col_name] = pa.array(new_timestamps, type=pa.float32())
            else:
                new_columns[col_name] = col

        arrays = [new_columns[field.name] for field in schema]
        new_table = pa.table(dict(zip([f.name for f in schema], arrays)), schema=schema)

        # 5) write parquet
        new_chunk_idx = new_episode_index // 1000
        new_chunk_dir = output_path / "data" / f"chunk-{new_chunk_idx:03d}"
        new_chunk_dir.mkdir(exist_ok=True)
        new_parquet_path = new_chunk_dir / f"episode_{new_episode_index:06d}.parquet"
        pq.write_table(new_table, new_parquet_path)

        # 6) update meta
        new_episodes.append({
            "episode_index": new_episode_index,
            "tasks": episode.get("tasks", ["unknown task"]),
            "length": num_rows,
        })
        new_global_index += num_rows
        new_episode_index += 1

    # ===================== Write metadata =====================
    if verbose:
        print("\n💾 Writing metadata...")

    new_info = info.copy()
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_frames"] = new_global_index
    new_info["total_chunks"] = (len(new_episodes) // 1000) + 1
    new_info["splits"] = {"train": f"0:{len(new_episodes)}"}

    with open(output_path / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(new_info, f, indent=4)

    with open(output_path / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    with open(output_path / "meta" / "tasks.jsonl", "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    if verbose:
        print("📊 Computing episode statistics...")

    with open(output_path / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for ep in tqdm(new_episodes, desc="Computing stats", disable=not verbose):
            ep_idx = ep["episode_index"]
            chunk_idx = ep_idx // 1000
            p = output_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            if p.exists():
                t = pq.read_table(p)
                f.write(json.dumps({"episode_index": ep_idx, "stats": compute_episode_stats(t)}) + "\n")

    report_path = output_path / "abnormal_frames_report.json"
    feature_thresholds = {}
    for feat_name, st in feature_stats.items():
        feature_thresholds[feat_name] = {
            "shape": st.get("shape"),
            "q01": st.get("global_q01", st.get("q01")),
            "q99": st.get("global_q99", st.get("q99")),
            "threshold_multiplier": threshold_multiplier,
        }

    report = {
        "remove_mode": remove_mode,
        "threshold_multiplier": threshold_multiplier,
        "features_checked": list(feature_stats.keys()),
        "feature_thresholds": feature_thresholds,
        "remove_empty_images": remove_empty_images,
        "fix_action_horizon": fix_action,
        "action_column": action_column,
        "action_k_mad": action_k_mad,
        "action_max_gap": action_max_gap,
        "frame_bad_ratio_thr": frame_bad_ratio_thr,
        "episode_bad_ratio_thr": episode_bad_ratio_thr,
        "action_on_bad_episode": action_on_bad_episode,
        "total_original_episodes": len(episodes),
        "total_original_frames": total_original_frames,
        "total_abnormal_frames": total_abnormal_frames,
        "removed_episodes": removed_episodes,
        "removed_empty_image_episodes": removed_empty_image_episodes,
        "final_episodes": len(new_episodes),
        "final_frames": new_global_index,
        "episodes_with_abnormal_frames": all_abnormal_info,
        "action_fix_summary": action_fix_info,
    }
    if remove_empty_images:
        report["empty_image_episodes"] = empty_image_episodes_info

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print("\n✅ Cleaning complete!")
        print(f"  Original episodes: {len(episodes)}")
        print(f"  Original frames: {total_original_frames}")
        print(f"  Removed episodes (too short / dropped): {removed_episodes}")
        print(f"  Final episodes: {len(new_episodes)}")
        print(f"  Final frames: {new_global_index}")
        print(f"  Output directory: {output_path}")
        print(f"  Report: {report_path}")


# ===================== CLI =====================
def main():
    parser = argparse.ArgumentParser(
        description="Remove abnormal frames/episodes from UMI LeRobot dataset (action smoothing via horizon variability)"
    )

    parser.add_argument("--input_dir", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cleaned dataset")

    parser.add_argument("--features", type=str, nargs="+", default=["observation.robot0_eef_pos"],
                        help="Feature names to check for abnormality.")
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="Threshold multiplier for q01/q99 (default: 1.5).")
    parser.add_argument("--min_episode_length", type=int, default=600,
                        help="Minimum episode length to keep (default: 600).")
    parser.add_argument("--remove_mode", type=str, choices=["frame", "episode"], default="frame",
                        help="Remove mode: 'frame' or 'episode'.")

    parser.add_argument("--remove_empty_images", type=lambda x: x.lower() != "false", default=True,
                        help="Remove episodes where any image column has all-zero images (default: True). Use --remove_empty_images=false to disable.")
    parser.add_argument("--image_columns", type=str, nargs="+", default=None,
                        help="Image column names to check for empty images. If not specified, auto-detect.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    # action smoothing (horizon)
    parser.add_argument("--action_column", type=str, default="actions",
                        help="Action column name in parquet (default: actions). Expect [B,H,10] or [B,H,20].")
    parser.add_argument("--fix_action", type=lambda x: x.lower() != "false", default=True,
                        help="Detect horizon-variability spikes and smooth actions (default: True). Use --fix_action=false to disable.")
    parser.add_argument("--action_k_mad", type=float, default=10.0,
                        help="MAD multiplier for horizon-variability spike detection per frame (default: 10.0). Larger -> fewer spikes.")
    parser.add_argument("--action_max_gap", type=int, default=3,
                        help="Max length of contiguous bad horizon steps to interpolate (default: 3).")
    parser.add_argument("--frame_bad_ratio_thr", type=float, default=0.6,
                        help="Count frames whose horizon-bad-ratio exceeds this (default: 0.6). Used for reporting.")
    parser.add_argument("--episode_bad_ratio_thr", type=float, default=0.5,
                        help="Drop episode if mean(frame_bad_ratio) exceeds this (default: 0.5).")
    parser.add_argument("--action_on_bad_episode", type=str, choices=["keep", "drop_episode"], default="drop_episode",
                        help="What to do if episode is judged bad by horizon-variability (default: drop_episode).")

    args = parser.parse_args()

    remove_abnormal_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        features=args.features,
        threshold_multiplier=args.threshold,
        min_episode_length=args.min_episode_length,
        remove_mode=args.remove_mode,
        remove_empty_images=args.remove_empty_images,
        image_columns=args.image_columns,
        verbose=not args.quiet,
        action_column=args.action_column,
        fix_action=args.fix_action,
        action_k_mad=args.action_k_mad,
        action_max_gap=args.action_max_gap,
        frame_bad_ratio_thr=args.frame_bad_ratio_thr,
        episode_bad_ratio_thr=args.episode_bad_ratio_thr,
        action_on_bad_episode=args.action_on_bad_episode,
    )


if __name__ == "__main__":
    main()
