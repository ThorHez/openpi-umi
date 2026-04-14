#!/usr/bin/env python3
"""
Remove abnormal frames from UMI LeRobot dataset.

- Removes frames where specified features exceed q99 or fall below q01 (by threshold).
- Optionally removes episodes with all-zero (empty) images.
- Episode-level: removes entire episodes where eef_pos_wrt_start first-to-last distance > threshold (始末位置过滤).
- Episode-level: removes episodes where arms are out of sync (一臂先动时 1 秒窗内另一臂未动, 双臂同步过滤), default on; use --no_remove_out_of_sync_episodes to disable.
- Clip is applied only to the actions column (by default), to [clip_min, clip_max] per dimension (default -1, 1; configurable via --clip_min / --clip_max; certain dims can be excluded).
- Can remove frames where any dimension of given column(s) is outside ±threshold: --column_above_threshold_columns col1 col2 --column_above_threshold_value 0.2 (default: robot0_eef_pos robot1_eef_pos, ±0.2)
- Does NOT perform any action filtering or horizon-based smoothing (see remove_abnormal_data_v2.py for that).
- When --features is used and minmax_stats.json is missing under input_dir, the script automatically runs analyze/compute_minmax_stats.py on that dataset first.

Usage example:
  python remove_abnormal_data.py --input_dir dataset --output_dir cleaned_dataset \
    --features observation.robot0_eef_pos --threshold 1.5 \
    --remove_empty_images --image_columns right_wrist_0_rgb_0 left_wrist_0_rgb_0

  # Remove frames where any dim of robot0/robot1_eef_pos is outside ±0.2 (default)
  python remove_abnormal_data.py --input_dir dataset --output_dir cleaned_dataset
  # Or use a different range: --column_above_threshold_value 0.5  (then ±0.5)

  # Remove mode: drop only abnormal frames, or drop entire episode
  python remove_abnormal_data.py --input_dir dataset --output_dir cleaned_dataset \
    --features observation.robot0_eef_pos --remove_mode episode

  # Disable bimanual sync filter (enabled by default)
  python remove_abnormal_data.py --input_dir dataset --output_dir cleaned_dataset --no_remove_out_of_sync_episodes
"""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq


# ===================== Episode stats =====================
def compute_episode_stats(table: pa.Table, skip_columns: set | None = None) -> Dict[str, Any]:
    stats = {}
    _skip = skip_columns or set()
    for col_name in table.column_names:
        if col_name in _skip:
            continue
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
def _ensure_minmax_stats(dataset_path: Path, verbose: bool = True) -> None:
    """
    若 dataset_path/minmax_stats.json 不存在，则自动运行 analyze/compute_minmax_stats.py
    生成该文件（与手动先跑 compute_minmax_stats.py --dataset <dataset_path> 等价）。
    """
    p = dataset_path / "minmax_stats.json"
    if p.exists():
        return
    script_path = Path(__file__).resolve().parent / "analyze" / "compute_minmax_stats.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"minmax_stats.json not found at {p} and compute_minmax_stats.py not found at {script_path}. "
            "Run: python examples/umi/analyze/compute_minmax_stats.py --dataset <dataset>"
        )
    if verbose:
        print(f"📊 minmax_stats.json not found at {p}; running compute_minmax_stats.py ...")
    subprocess.run(
        [sys.executable, str(script_path), "--dataset", str(dataset_path)],
        check=True,
    )
    if not p.exists():
        raise RuntimeError(f"compute_minmax_stats finished but {p} still not found.")


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


def get_heavy_columns(info: Dict[str, Any]) -> set:
    """Return column names that are images or large spatial arrays (depth maps etc.)
    based on feature metadata in info.json.  These should be skipped in
    row-by-row numeric processing to avoid extreme slowness."""
    heavy = set()
    for fname, finfo in info.get("features", {}).items():
        fdtype = finfo.get("dtype", "")
        fshape = tuple(finfo.get("shape", []))
        if fdtype == "image" or (len(fshape) >= 2 and fshape[0] >= 64 and fshape[1] >= 64):
            heavy.add(fname)
    return heavy


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
        if col_name not in table.column_names:
            continue
        col = table.column(col_name)
        max_values = []
        for row_idx in range(min(table.num_rows, 5)):
            try:
                img_data = col[row_idx].as_py()
                if img_data is None:
                    continue
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img = Image.open(io.BytesIO(img_data["bytes"]))
                    arr = np.array(img)
                elif isinstance(img_data, (list, np.ndarray)):
                    arr = np.asarray(img_data)
                else:
                    continue
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


# ===================== Start-end distance check (eef_pos_wrt_start) =====================
def _eef_pos_wrt_start_to_vec(row_value):
    """将 parquet 一行的 eef_pos_wrt_start 转为 3D 向量，支持 (2,3) 取当前帧。"""
    arr = np.asarray(row_value, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[-1]
    return np.asarray(arr).ravel()[:3]


def check_episode_start_end_distance(
    table: pa.Table,
    threshold_m: float = 1.0,
    pos_wrt_start_keys: List[str] = None,
) -> Tuple[bool, Dict]:
    """
    检查 episode 首帧与末帧的 eef_pos_wrt_start 欧氏距离。
    若任一手臂距离 > threshold_m，则视为未回到起始位置，应删除整个 episode。
    返回 (should_remove, info_dict)。
    """
    if pos_wrt_start_keys is None:
        pos_wrt_start_keys = ["robot0_eef_pos_wrt_start", "robot1_eef_pos_wrt_start"]
    if table.num_rows < 2:
        return False, {"robot0_dist": None, "robot1_dist": None}
    info = {}
    should_remove = False
    for key in pos_wrt_start_keys:
        if key not in table.column_names:
            info[key.replace("_eef_pos_wrt_start", "_dist")] = None
            continue
        col = table.column(key)
        first = col[0].as_py()
        last = col[table.num_rows - 1].as_py()
        v0 = _eef_pos_wrt_start_to_vec(first)
        v1 = _eef_pos_wrt_start_to_vec(last)
        dist = float(np.linalg.norm(v1 - v0))
        short_key = key.replace("_eef_pos_wrt_start", "_dist")
        info[short_key] = dist
        if dist > threshold_m:
            should_remove = True
    return should_remove, info


# ===================== Bimanual sync check (一臂先动时 1 秒窗内另一臂未动则删除 episode) =====================
def check_episode_bimanual_sync(
    table: pa.Table,
    fps: float,
    sync_min_movement: float = 0.03,
    sync_max_still: float = 0.01,
    follow_window_sec: float = 1.0,
    pos_wrt_start_keys: List[str] | None = None,
) -> Tuple[bool, Dict]:
    """
    判断双臂是否同步：当一臂「首次开始移动」时，在随后 follow_window_sec 秒内若另一臂位移 < sync_max_still 则视为不同步。
    返回 (should_remove=True 表示应删除该 episode, info_dict)。
    """
    if pos_wrt_start_keys is None:
        pos_wrt_start_keys = ["robot0_eef_pos_wrt_start", "robot1_eef_pos_wrt_start"]
    if pos_wrt_start_keys[0] not in table.column_names or pos_wrt_start_keys[1] not in table.column_names:
        return False, {"reason": "missing_columns"}
    n = table.num_rows
    if n == 0:
        return False, {}

    col0 = table.column(pos_wrt_start_keys[0])
    col1 = table.column(pos_wrt_start_keys[1])
    p0 = np.array([_eef_pos_wrt_start_to_vec(col0[i].as_py()) for i in range(n)])
    p1 = np.array([_eef_pos_wrt_start_to_vec(col1[i].as_py()) for i in range(n)])
    disp0_from_start = np.linalg.norm(p0 - p0[0], axis=1)
    disp1_from_start = np.linalg.norm(p1 - p1[0], axis=1)

    start_frame_0 = None
    start_frame_1 = None
    for i in range(1, n):
        if start_frame_0 is None and disp0_from_start[i] > sync_min_movement:
            start_frame_0 = i
        if start_frame_1 is None and disp1_from_start[i] > sync_min_movement:
            start_frame_1 = i

    follow_frames = max(1, int(fps * follow_window_sec))
    out_of_sync = False
    info = {"start_frame_0": start_frame_0, "start_frame_1": start_frame_1}

    if start_frame_0 is not None:
        end_idx = min(start_frame_0 + follow_frames, n - 1)
        disp1_when_0_moved = float(np.linalg.norm(p1[end_idx] - p1[start_frame_0]))
        info["disp1_when_0_moved"] = disp1_when_0_moved
        if disp1_when_0_moved < sync_max_still:
            out_of_sync = True

    if start_frame_1 is not None:
        end_idx = min(start_frame_1 + follow_frames, n - 1)
        disp0_when_1_moved = float(np.linalg.norm(p0[end_idx] - p0[start_frame_1]))
        info["disp0_when_1_moved"] = disp0_when_1_moved
        if disp0_when_1_moved < sync_max_still:
            out_of_sync = True

    return out_of_sync, info


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


# ===================== Frames where any dimension of a column is outside ±threshold =====================
def detect_frames_column_above_threshold(
    table: pa.Table,
    column_name: str,
    threshold: float,
) -> Tuple[List[int], List[Dict]]:
    """
    指定一列和一个阈值：若该帧在该列上任一维度的值超出 [-threshold, threshold]（即 > threshold 或 < -threshold），则剔除该帧。
    返回 (indices_to_remove, details_per_frame).
    """
    if column_name not in table.column_names:
        return [], []
    indices = []
    details_list = []
    col = table.column(column_name)
    for row_idx in range(table.num_rows):
        data = col[row_idx].as_py()
        flat = _flatten_nested(data)
        if not flat:
            continue
        for dim_idx, v in enumerate(flat):
            val = float(v)
            if val > threshold or val < -threshold:
                indices.append(row_idx)
                details_list.append({
                    "frame_index": row_idx,
                    "column": column_name,
                    "dim": dim_idx,
                    "value": val,
                    "threshold": threshold,
                })
                break
    return indices, details_list


# ===================== Clip to [clip_min, clip_max] for actions (skip certain dims) =====================
# 多机械臂时每个机器人动作维度（UMI 配置中通常为 10：3 pos + 6 rot + 1 gripper）
DIMS_PER_ROBOT = 10
POS_DIMS_PER_ROBOT = 3  # 仅对前三维（位置）做 clip


def clip_to_range(
    table: pa.Table,
    column_name: str,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    exclude_dims: List[int] | None = None,
) -> pa.Table:
    """
    Clip dimensions of *column_name* to [clip_min, clip_max]. For the actions column,
    when *exclude_dims* is None, only the first 3 dimensions (position) of each robot
    are clipped (dims 0,1,2 for robot0; 10,11,12 for robot1 when action_dim=20); rotation
    and gripper are not clipped.  Supports 2-D (num_frames, dim) and 3-D
    (num_frames, horizon, dim) data.
    """
    if column_name not in table.column_names:
        return table
    col = table.column(column_name)
    data_list = [col[i].as_py() for i in range(table.num_rows)]
    data = np.asarray(data_list, dtype=np.float64)
    if data.ndim not in (2, 3):
        return table

    action_dim = data.shape[-1]
    if exclude_dims is None and column_name == "actions":
        # 仅对多机械臂的前三维（pos）做 clip，其余维度不 clip
        exclude_dims = [d for d in range(action_dim) if d % DIMS_PER_ROBOT >= POS_DIMS_PER_ROBOT]
    exclude_set = set(exclude_dims) if exclude_dims else set()

    lower = np.full(action_dim, float(clip_min))
    upper = np.full(action_dim, float(clip_max))
    for d in exclude_set:
        if 0 <= d < action_dim:
            lower[d] = -np.inf
            upper[d] = np.inf

    data_clipped = np.clip(data, lower, upper)

    original_type = table.schema.field(column_name).type
    new_col = pa.array(data_clipped.astype(np.float32).tolist(), type=original_type)
    col_idx = table.schema.get_field_index(column_name)
    return table.set_column(col_idx, column_name, new_col)


# ===================== Main cleaning function =====================
def remove_abnormal_frames(
    input_dir: str,
    output_dir: str,
    features: List[str],
    threshold_multiplier: float = 1.5,
    min_episode_length: int = 50,
    remove_mode: str = "frame",
    remove_empty_images: bool = False,
    image_columns: List[str] | None = None,
    verbose: bool = True,
    clip_unit_range: bool = False,
    clip_unit_range_columns: List[str] | None = None,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    clip_unit_range_exclude_dims: List[int] | None = None,
    remove_start_end_distance: bool = True,
    start_end_distance_threshold: float = 1.0,
    remove_out_of_sync_episodes: bool = True,
    sync_follow_window_sec: float = 1.0,
    sync_min_movement: float = 0.03,
    sync_max_still: float = 0.01,
    column_above_threshold_columns: List[str] | None = None,
    column_above_threshold_value: float | None = None,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    info = load_info(input_path)
    episodes = load_episodes(input_path)
    tasks = load_tasks(input_path)
    heavy_columns = get_heavy_columns(info)

    minmax_stats = {}
    if features:
        _ensure_minmax_stats(input_path, verbose=verbose)
        minmax_stats = load_minmax_stats(input_path)

    fps = info.get("fps", 20)

    feature_stats = {}
    for feat in features:
        if feat in minmax_stats:
            feature_stats[feat] = minmax_stats[feat]
        else:
            if verbose and features:
                print(f"⚠️  Warning: Feature '{feat}' not found in minmax_stats.json, skipping...")

    if not feature_stats and not remove_empty_images and not remove_start_end_distance and not remove_out_of_sync_episodes and not (column_above_threshold_columns and column_above_threshold_value is not None):
        raise ValueError(
            "Nothing to do: no valid features, remove_empty_images off, remove_start_end_distance off, remove_out_of_sync_episodes off, and no column_above_threshold."
        )

    if verbose:
        print(f"Dataset: {input_path}")
        print(f"Total episodes: {len(episodes)}")
        print(f"FPS: {fps}")
        print(f"Remove mode: {remove_mode}")
        print(f"Threshold multiplier: {threshold_multiplier}")
        print(f"Remove empty images: {remove_empty_images}")
        print(f"Remove start-end distance > {start_end_distance_threshold}m: {remove_start_end_distance}")
        print(f"Remove out-of-sync episodes (bimanual): {remove_out_of_sync_episodes}"
              f" (follow_window={sync_follow_window_sec}s, min_movement={sync_min_movement}, max_still={sync_max_still})")
        print(f"Clip to [{clip_min}, {clip_max}] (actions only): {clip_unit_range}"
              f" columns={clip_unit_range_columns or ['actions']}"
              f" exclude_dims={clip_unit_range_exclude_dims or []}")
        if column_above_threshold_columns and column_above_threshold_value is not None:
            print(f"Remove frames where any dim of {column_above_threshold_columns} outside ±{column_above_threshold_value}")

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    total_abnormal_frames = 0
    total_original_frames = 0
    removed_episodes = 0
    removed_empty_image_episodes = 0
    removed_start_end_distance_episodes = 0
    removed_out_of_sync_episodes = 0
    all_abnormal_info: List[Dict] = []
    empty_image_episodes_info: List[Dict] = []
    start_end_distance_episodes_info: List[Dict] = []
    out_of_sync_episodes_info: List[Dict] = []

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
            effective_image_cols = image_columns or [c for c in heavy_columns if c in table.column_names]
            is_empty, empty_info = check_episode_has_empty_images(table, effective_image_cols or None)
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

        # 2) 起始帧与结束帧距离：eef_pos_wrt_start 首末帧欧氏距离 > 阈值则删除整个 episode
        if remove_start_end_distance:
            should_remove, dist_info = check_episode_start_end_distance(
                table, threshold_m=start_end_distance_threshold
            )
            if should_remove:
                removed_start_end_distance_episodes += 1
                start_end_distance_episodes_info.append({
                    "episode_index": old_episode_idx,
                    "robot0_dist": dist_info.get("robot0_dist"),
                    "robot1_dist": dist_info.get("robot1_dist"),
                    "threshold_m": start_end_distance_threshold,
                })
                if verbose:
                    print(
                        f"  [起止距离] Episode {old_episode_idx}: "
                        f"robot0={dist_info.get('robot0_dist')}, robot1={dist_info.get('robot1_dist')} "
                        f"(阈值 {start_end_distance_threshold}m)"
                    )
                continue

        # 2b) 双臂同步：一臂先动时 1 秒窗内另一臂未动则删除整个 episode
        if remove_out_of_sync_episodes:
            out_of_sync, sync_info = check_episode_bimanual_sync(
                table,
                fps=fps,
                sync_min_movement=sync_min_movement,
                sync_max_still=sync_max_still,
                follow_window_sec=sync_follow_window_sec,
            )
            if out_of_sync:
                removed_out_of_sync_episodes += 1
                out_of_sync_episodes_info.append({
                    "episode_index": old_episode_idx,
                    **sync_info,
                })
                if verbose:
                    print(
                        f"  [双臂不同步] Episode {old_episode_idx}: "
                        f"sync_info={sync_info}"
                    )
                continue

        # 3) clip 仅针对 actions（或显式指定的列）到 [clip_min, clip_max]，可排除部分维度
        if clip_unit_range:
            if clip_unit_range_columns is not None:
                cols_to_clip = clip_unit_range_columns
            else:
                cols_to_clip = ["actions"] if "actions" in table.column_names else []
            for col_name in cols_to_clip:
                if col_name in table.column_names:
                    table = clip_to_range(
                        table, col_name,
                        clip_min=clip_min, clip_max=clip_max,
                        exclude_dims=clip_unit_range_exclude_dims,
                    )

        # 4) detect abnormal frames by q01/q99 and remove
        abnormal_indices, abnormal_info = (
            detect_abnormal_frames(table, feature_stats, threshold_multiplier)
            if feature_stats
            else ([], [])
        )

        # 4b) 指定若干列：若该帧在任一一列上任一维度 > 阈值则剔除该帧
        if column_above_threshold_columns and column_above_threshold_value is not None:
            col_above_set = set()
            col_above_info = []
            for col_name in column_above_threshold_columns:
                idx, info_list = detect_frames_column_above_threshold(
                    table, col_name, column_above_threshold_value
                )
                col_above_set.update(idx)
                col_above_info.extend(info_list)
            if col_above_set:
                abnormal_indices = list(set(abnormal_indices) | col_above_set)
                abnormal_info = list(abnormal_info) + col_above_info

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

        # 5) update indices/timestamps
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

        # 6) write parquet
        new_chunk_idx = new_episode_index // 1000
        new_chunk_dir = output_path / "data" / f"chunk-{new_chunk_idx:03d}"
        new_chunk_dir.mkdir(exist_ok=True)
        new_parquet_path = new_chunk_dir / f"episode_{new_episode_index:06d}.parquet"
        pq.write_table(new_table, new_parquet_path)

        # 7) update meta
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
                f.write(json.dumps({"episode_index": ep_idx, "stats": compute_episode_stats(t, heavy_columns)}) + "\n")

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
        "clip_unit_range": clip_unit_range,
        "clip_unit_range_columns": clip_unit_range_columns if clip_unit_range else None,
        "clip_min": clip_min if clip_unit_range else None,
        "clip_max": clip_max if clip_unit_range else None,
        "clip_unit_range_exclude_dims": clip_unit_range_exclude_dims if clip_unit_range else None,
        "total_original_episodes": len(episodes),
        "total_original_frames": total_original_frames,
        "total_abnormal_frames": total_abnormal_frames,
        "removed_episodes": removed_episodes,
        "removed_empty_image_episodes": removed_empty_image_episodes,
        "removed_start_end_distance_episodes": removed_start_end_distance_episodes,
        "removed_out_of_sync_episodes": removed_out_of_sync_episodes,
        "final_episodes": len(new_episodes),
        "final_frames": new_global_index,
        "episodes_with_abnormal_frames": all_abnormal_info,
    }
    if column_above_threshold_columns and column_above_threshold_value is not None:
        report["column_above_threshold_columns"] = column_above_threshold_columns
        report["column_above_threshold_value"] = column_above_threshold_value
    if remove_empty_images:
        report["empty_image_episodes"] = empty_image_episodes_info
    if remove_start_end_distance:
        report["remove_start_end_distance"] = True
        report["start_end_distance_threshold_m"] = start_end_distance_threshold
        report["start_end_distance_episodes"] = start_end_distance_episodes_info
    if remove_out_of_sync_episodes:
        report["remove_out_of_sync_episodes"] = True
        report["sync_follow_window_sec"] = sync_follow_window_sec
        report["sync_min_movement"] = sync_min_movement
        report["sync_max_still"] = sync_max_still
        report["out_of_sync_episodes"] = out_of_sync_episodes_info

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print("\n✅ Cleaning complete!")
        print(f"  Original episodes: {len(episodes)}")
        print(f"  Original frames: {total_original_frames}")
        print(f"  Removed episodes (too short / dropped): {removed_episodes}")
        print(f"  Removed episodes (empty images): {removed_empty_image_episodes}")
        print(f"  Removed episodes (start-end distance > {start_end_distance_threshold}m): {removed_start_end_distance_episodes}")
        print(f"  Removed episodes (bimanual out-of-sync): {removed_out_of_sync_episodes}")
        print(f"  Final episodes: {len(new_episodes)}")
        print(f"  Final frames: {new_global_index}")
        print(f"  Output directory: {output_path}")
        print(f"  Report: {report_path}")


# ===================== CLI =====================
def main():
    parser = argparse.ArgumentParser(
        description="Remove abnormal frames/episodes from UMI LeRobot dataset (no action filtering)."
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
    parser.add_argument("--clip_to_unit_range", type=lambda x: x.lower() != "false", default=True,
                        help="Clip specified columns to [clip_min, clip_max] per dim (default: True). Use --clip_to_unit_range=false to disable.")
    parser.add_argument("--clip_unit_range_columns", type=str, nargs="+", default=None,
                        help="Column names to clip. Defaults to 'actions' only.")
    parser.add_argument("--clip_min", type=float, default=-0.5,
                        help="Lower bound for clip range (default: -1.0).")
    parser.add_argument("--clip_max", type=float, default=0.5,
                        help="Upper bound for clip range (default: 1.0).")
    parser.add_argument("--clip_unit_range_exclude_dims", type=int, nargs="+", default=None,
                        help="Dimension indices to EXCLUDE from clipping (0-based); those dims are not clipped.")
    parser.add_argument("--column_above_threshold_columns", type=str, nargs="+",
                        default=["robot0_eef_pos", "robot1_eef_pos"],
                        help="Columns to check: if any dimension in a frame is outside ±threshold, remove that frame. Default: robot0_eef_pos robot1_eef_pos")
    parser.add_argument("--column_above_threshold_value", type=float, default=0.2,
                        help="Threshold (symmetric): frames with any value in these columns outside [-threshold, +threshold] are removed (default: ±0.2).")
    parser.add_argument("--no_remove_start_end_distance", action="store_true",
                        help="Disable removing episodes where eef_pos_wrt_start first-to-last distance > threshold (default: enabled, threshold 1m).")
    parser.add_argument("--start_end_distance_threshold", type=float, default=1.0,
                        help="Threshold in meters for start-end distance; episode removed if either arm exceeds (default: 1.0).")
    parser.add_argument("--no_remove_out_of_sync_episodes", action="store_true",
                        help="Disable removing out-of-sync episodes (default: enabled; one arm moves first, other does not move within 1s window).")
    parser.add_argument("--sync_follow_window_sec", type=float, default=1.0,
                        help="Seconds after one arm starts moving to check the other arm (default: 1.0).")
    parser.add_argument("--sync_min_movement", type=float, default=0.02,
                        help="Min displacement (m) to consider arm 'started moving' (default: 0.03).")
    parser.add_argument("--sync_max_still", type=float, default=0.02,
                        help="Max displacement (m) in follow window to consider other arm 'still' (default: 0.01).")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

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
        clip_unit_range=args.clip_to_unit_range,
        clip_unit_range_columns=args.clip_unit_range_columns,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        clip_unit_range_exclude_dims=args.clip_unit_range_exclude_dims,
        remove_start_end_distance=not args.no_remove_start_end_distance,
        start_end_distance_threshold=args.start_end_distance_threshold,
        remove_out_of_sync_episodes=not args.no_remove_out_of_sync_episodes,
        sync_follow_window_sec=args.sync_follow_window_sec,
        sync_min_movement=args.sync_min_movement,
        sync_max_still=args.sync_max_still,
        column_above_threshold_columns=args.column_above_threshold_columns,
        column_above_threshold_value=args.column_above_threshold_value,
    )


if __name__ == "__main__":
    main()
