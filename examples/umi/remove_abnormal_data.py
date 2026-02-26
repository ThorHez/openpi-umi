#!/usr/bin/env python3
"""
Remove abnormal frames from UMI LeRobot dataset.

- Removes frames where specified features exceed q99 or fall below q01 (by threshold).
- Optionally removes episodes with all-zero (empty) images.
- Optionally clips actions to [q01 * multiplier, q99 * multiplier] per dimension (using minmax_stats.json).
- Does NOT perform any action filtering or horizon-based smoothing (see remove_abnormal_data_v2.py for that).

Usage example:
  python remove_abnormal_data.py --input_dir dataset --output_dir cleaned_dataset \
    --features observation.robot0_eef_pos --threshold 1.5 \
    --remove_empty_images --image_columns right_wrist_0_rgb_0 left_wrist_0_rgb_0

  # Remove mode: drop only abnormal frames, or drop entire episode
  python remove_abnormal_data.py --input_dir dataset --output_dir cleaned_dataset \
    --features observation.robot0_eef_pos --remove_mode episode
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


# ===================== Actions clip =====================
def clip_actions_by_q01_q99(
    table: pa.Table,
    actions_key: str,
    q01_per_dim: List[float],
    q99_per_dim: List[float],
    multiplier: float = 1.5,
) -> pa.Table:
    """
    Clip actions to [q01 * multiplier, q99 * multiplier] per dimension.
    actions in table have shape (num_frames, horizon, action_dim) per row.
    """
    if actions_key not in table.column_names:
        return table
    col = table.column(actions_key)
    act_list = [col[i].as_py() for i in range(table.num_rows)]
    actions = np.asarray(act_list, dtype=np.float64)
    if actions.ndim != 3:
        return table
    _, _, action_dim = actions.shape
    if len(q01_per_dim) != action_dim or len(q99_per_dim) != action_dim:
        return table
    q01 = np.array(q01_per_dim, dtype=np.float64)
    q99 = np.array(q99_per_dim, dtype=np.float64)
    lower = np.minimum(q01 * multiplier, q99 * multiplier)
    upper = np.maximum(q01 * multiplier, q99 * multiplier)
    actions_clipped = np.clip(actions, lower, upper)
    original_type = table.schema.field(actions_key).type
    new_col = pa.array(actions_clipped.astype(np.float32).tolist(), type=original_type)
    col_idx = table.schema.get_field_index(actions_key)
    return table.set_column(col_idx, actions_key, new_col)


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
    clip_actions: bool = True,
    clip_actions_multiplier: float = 1.5,
    actions_column: str = "actions",
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
            if verbose:
                print(f"⚠️  Warning: Feature '{feat}' not found in minmax_stats.json, skipping...")

    if not feature_stats and not remove_empty_images:
        raise ValueError("No valid features to check and remove_empty_images disabled. Nothing to do.")

    # Actions clip: get q01/q99 from minmax_stats if present
    actions_clip_q01: List[float] | None = None
    actions_clip_q99: List[float] | None = None
    if clip_actions and "actions" in minmax_stats:
        act_st = minmax_stats["actions"]
        if isinstance(act_st.get("q01_per_dim"), list) and isinstance(act_st.get("q99_per_dim"), list):
            actions_clip_q01 = [float(x) for x in act_st["q01_per_dim"]]
            actions_clip_q99 = [float(x) for x in act_st["q99_per_dim"]]
        elif "global_q01" in act_st and "global_q99" in act_st:
            actions_clip_q01 = [float(act_st["global_q01"])]
            actions_clip_q99 = [float(act_st["global_q99"])]
    if clip_actions and (actions_clip_q01 is None or actions_clip_q99 is None):
        if verbose:
            print("⚠️  clip_actions=True but 'actions' q01_per_dim/q99_per_dim not in minmax_stats; skipping action clip.")
        clip_actions = False

    if verbose:
        print(f"Dataset: {input_path}")
        print(f"Total episodes: {len(episodes)}")
        print(f"FPS: {fps}")
        print(f"Remove mode: {remove_mode}")
        print(f"Threshold multiplier: {threshold_multiplier}")
        print(f"Remove empty images: {remove_empty_images}")
        print(f"Clip actions: {clip_actions} (multiplier={clip_actions_multiplier})")

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    total_abnormal_frames = 0
    total_original_frames = 0
    removed_episodes = 0
    removed_empty_image_episodes = 0
    all_abnormal_info: List[Dict] = []
    empty_image_episodes_info: List[Dict] = []

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

        # 2) clip actions to [q01*multiplier, q99*multiplier] per dimension
        if clip_actions and actions_clip_q01 is not None and actions_clip_q99 is not None:
            table = clip_actions_by_q01_q99(
                table,
                actions_key=actions_column,
                q01_per_dim=actions_clip_q01,
                q99_per_dim=actions_clip_q99,
                multiplier=clip_actions_multiplier,
            )

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
        "clip_actions": clip_actions,
        "clip_actions_multiplier": clip_actions_multiplier if clip_actions else None,
        "total_original_episodes": len(episodes),
        "total_original_frames": total_original_frames,
        "total_abnormal_frames": total_abnormal_frames,
        "removed_episodes": removed_episodes,
        "removed_empty_image_episodes": removed_empty_image_episodes,
        "final_episodes": len(new_episodes),
        "final_frames": new_global_index,
        "episodes_with_abnormal_frames": all_abnormal_info,
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
        print(f"  Removed episodes (empty images): {removed_empty_image_episodes}")
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
    parser.add_argument("--clip_actions", type=lambda x: x.lower() != "false", default=True,
                        help="Clip actions to [q01*multiplier, q99*multiplier] per dim using minmax_stats (default: True).")
    parser.add_argument("--clip_actions_multiplier", type=float, default=1.5,
                        help="Multiplier for action clip bounds: lower=q01*this, upper=q99*this (default: 1.5).")
    parser.add_argument("--actions_column", type=str, default="actions",
                        help="Action column name in parquet (default: actions).")
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
        clip_actions=args.clip_actions,
        clip_actions_multiplier=args.clip_actions_multiplier,
        actions_column=args.actions_column,
    )


if __name__ == "__main__":
    main()
