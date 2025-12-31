"""
Remove abnormal frames from UMI LeRobot dataset.

This script reads minmax_stats.json and removes frames where specified features
exceed q99 or fall below q01 by a threshold. It can also remove episodes with
all-zero (empty) images.

Usage:
    # Remove abnormal frames based on feature thresholds
    python remove_abnormal_data.py --input_dir dataset --output_dir cleaned_dataset --threshold 1.5 \
        --features observation.robot0_eef_pos observation.robot0_eef_rot_axis_angle
    
    # Remove episodes with empty images
    python remove_abnormal_data.py --input_dir dataset --output_dir cleaned_dataset \
        --remove_empty_images --image_columns right_wrist_0_rgb_0 left_wrist_0_rgb_0
    
    # Combine both: remove abnormal frames and empty image episodes
    python remove_abnormal_data.py --input_dir dataset --output_dir cleaned_dataset \
        --features observation.robot0_eef_pos --remove_empty_images
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import io

import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import numpy as np
from PIL import Image


def compute_episode_stats(table: pa.Table) -> Dict[str, Any]:
    """
    Compute statistics for an episode (min, max, mean, std, count for each feature).
    
    This matches the format in episodes_stats.jsonl.
    """
    stats = {}
    
    for col_name in table.column_names:
        col = table.column(col_name)
        
        # Convert to numpy for stats computation
        try:
            # For nested arrays (like observations with shape [2, 3])
            data = np.array([row.as_py() for row in col])
        except (ValueError, TypeError):
            # Skip columns that can't be converted
            continue
        
        if data.dtype == object:
            # Handle string or complex nested data
            continue
        
        # Compute statistics based on data shape
        if len(data.shape) == 1:
            # Simple 1D array (e.g., timestamp, frame_index)
            stats[col_name] = {
                "min": [float(np.min(data))],
                "max": [float(np.max(data))],
                "mean": [float(np.mean(data))],
                "std": [float(np.std(data))],
                "count": [int(len(data))]
            }
        elif len(data.shape) == 2:
            # 2D array (e.g., gripper_width with shape [N, 1])
            stats[col_name] = {
                "min": data.min(axis=0).tolist(),
                "max": data.max(axis=0).tolist(),
                "mean": data.mean(axis=0).tolist(),
                "std": data.std(axis=0).tolist(),
                "count": [int(len(data))]
            }
        elif len(data.shape) == 3:
            # 3D array (e.g., observation.robot0_eef_pos with shape [N, 2, 3])
            # Compute stats for each time step separately
            min_vals = []
            max_vals = []
            mean_vals = []
            std_vals = []
            
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
                "count": [int(len(data))]
            }
        elif len(data.shape) == 4:
            # 4D array (e.g., images with shape [N, H, W, C])
            # Compute channel-wise stats
            min_vals = []
            max_vals = []
            mean_vals = []
            std_vals = []
            
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
                "count": [int(len(data))]
            }
        elif len(data.shape) >= 5:
            # Higher dimensional arrays (e.g., video/image sequences)
            # Just compute global stats per channel
            if data.shape[-1] <= 4:  # Likely channels
                min_vals = []
                max_vals = []
                mean_vals = []
                std_vals = []
                
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
                    "count": [int(len(data))]
                }
    
    return stats


def load_minmax_stats(dataset_path: Path) -> Dict[str, Any]:
    """Load minmax_stats.json from dataset."""
    minmax_stats_path = dataset_path / "minmax_stats.json"
    if not minmax_stats_path.exists():
        raise FileNotFoundError(f"minmax_stats.json not found at {minmax_stats_path}. "
                               "Please run compute_minmax_stats.py first.")
    
    with open(minmax_stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_info(dataset_path: Path) -> Dict[str, Any]:
    """Load dataset info.json"""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_episodes(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load episodes from episodes.jsonl"""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def load_tasks(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load tasks from tasks.jsonl"""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    tasks = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def _flatten_nested(nested):
    """Flatten nested list to 1D"""
    result = []
    if isinstance(nested, (list, tuple)):
        for item in nested:
            result.extend(_flatten_nested(item))
    else:
        result.append(nested)
    return result


def get_image_columns(table: pa.Table) -> List[str]:
    """Get list of image columns from table (columns containing image data)."""
    image_columns = []
    # Common image column patterns in UMI dataset
    image_patterns = ['rgb', 'image', 'camera', 'wrist']
    
    for col_name in table.column_names:
        # Check if column name contains image-related patterns
        if any(pattern in col_name.lower() for pattern in image_patterns):
            # Verify it's actually image data (dict with 'bytes' key)
            try:
                sample = table.column(col_name)[0].as_py()
                if isinstance(sample, dict) and 'bytes' in sample:
                    image_columns.append(col_name)
            except (IndexError, TypeError):
                continue
    
    return image_columns


def check_episode_has_empty_images(table: pa.Table, image_columns: List[str] = None) -> Tuple[bool, Dict]:
    """
    Check if an episode has all-zero (empty) images in any image column.
    
    Args:
        table: Arrow table with episode data
        image_columns: List of image column names to check. If None, auto-detect.
    
    Returns:
        Tuple of (is_empty, info_dict) where is_empty is True if ALL images in 
        ANY column have max=0, and info_dict contains details.
    """
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
                if img_data is not None and 'bytes' in img_data:
                    img = Image.open(io.BytesIO(img_data['bytes']))
                    img_array = np.array(img)
                    max_val = int(img_array.max())
                    max_values.append(max_val)
                    
                    # Early exit: if any image has non-zero max, this column is not empty
                    if max_val != 0:
                        break
            except Exception as e:
                continue
        
        if max_values:
            overall_max = max(max_values)
            column_stats[col_name] = {
                "checked_frames": len(max_values),
                "overall_max": overall_max
            }
            
            # If ALL checked images have max=0, column is empty
            if overall_max == 0:
                empty_columns.append(col_name)
    
    # Episode is considered empty if ANY image column has all-zero images
    is_empty = len(empty_columns) > 0
    
    return is_empty, {
        "empty_columns": empty_columns,
        "column_stats": column_stats,
        "total_image_columns": len(image_columns)
    }


def detect_abnormal_frames(
    table: pa.Table,
    feature_stats: Dict[str, Dict[str, Any]],
    threshold_multiplier: float = 1.5,
) -> Tuple[List[int], List[Dict]]:
    """
    Detect abnormal frames in a parquet table based on multiple features.
    
    Args:
        table: Arrow table with episode data
        feature_stats: Dict mapping feature names to their q01/q99 stats
                      e.g., {"observation.robot0_eef_pos": {"q01_per_dim": [...], "q99_per_dim": [...]}}
        threshold_multiplier: Multiplier for determining threshold range
                             value outside [q01 * multiplier, q99 * multiplier] is abnormal
    
    Returns:
        Tuple of (list of abnormal frame indices, list of abnormal info dicts)
    """
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
            
            # Get q01 and q99 values
            q01_per_dim = stats.get("q01_per_dim")
            q99_per_dim = stats.get("q99_per_dim")
            
            # Handle scalar stats (for 1D features)
            if q01_per_dim is None:
                q01_per_dim = [[stats.get("q01", 0)]]
                q99_per_dim = [[stats.get("q99", 0)]]
            
            flat_data = _flatten_nested(data)
            flat_q01 = _flatten_nested(q01_per_dim)
            flat_q99 = _flatten_nested(q99_per_dim)
            
            # Check each dimension
            for dim_idx, value in enumerate(flat_data):
                if dim_idx >= len(flat_q01) or dim_idx >= len(flat_q99):
                    continue
                
                q01_val = flat_q01[dim_idx]
                q99_val = flat_q99[dim_idx]
                
                # Calculate thresholds
                # For positive q99, upper_threshold = q99 * multiplier
                # For negative q01, lower_threshold = q01 * multiplier (makes it more negative)
                if q99_val >= 0:
                    upper_threshold = q99_val * threshold_multiplier
                else:
                    upper_threshold = q99_val / threshold_multiplier
                
                if q01_val <= 0:
                    lower_threshold = q01_val * threshold_multiplier
                else:
                    lower_threshold = q01_val / threshold_multiplier
                
                # Check if value is abnormal
                if value > upper_threshold:
                    is_abnormal = True
                    abnormal_details.append({
                        "feature": feature_name,
                        "dim": dim_idx,
                        "value": value,
                        "type": "too_high",
                        "threshold": upper_threshold,
                        "q99": q99_val
                    })
                elif value < lower_threshold:
                    is_abnormal = True
                    abnormal_details.append({
                        "feature": feature_name,
                        "dim": dim_idx,
                        "value": value,
                        "type": "too_low",
                        "threshold": lower_threshold,
                        "q01": q01_val
                    })
        
        if is_abnormal:
            abnormal_indices.append(row_idx)
            abnormal_info.append({
                "frame_index": row_idx,
                "abnormal_details": abnormal_details
            })
    
    return abnormal_indices, abnormal_info


def remove_abnormal_frames(
    input_dir: str,
    output_dir: str,
    features: List[str],
    threshold_multiplier: float = 1.5,
    min_episode_length: int = 50,
    remove_mode: str = "frame",
    remove_empty_images: bool = False,
    image_columns: List[str] = None,
    verbose: bool = True
) -> None:
    """
    Remove abnormal frames or episodes from the dataset.
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for cleaned dataset
        features: List of feature names to check for abnormality
        threshold_multiplier: Multiplier for q99/q01 thresholds (default: 1.5)
        min_episode_length: Minimum episode length after cleaning (episodes shorter will be removed)
        remove_mode: "frame" to remove only abnormal frames, "episode" to remove entire episode
        remove_empty_images: Whether to remove episodes with all-zero (empty) images
        image_columns: List of image column names to check for empty images. If None, auto-detect.
        verbose: Whether to print progress
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Load dataset info
    minmax_stats = load_minmax_stats(input_path)
    info = load_info(input_path)
    episodes = load_episodes(input_path)
    tasks = load_tasks(input_path)
    
    # Get fps for timestamp recalculation
    fps = info.get("fps", 20)  # Default to 20 fps if not specified
    
    # Build feature stats dict for the specified features
    feature_stats = {}
    for feature in features:
        if feature in minmax_stats:
            feature_stats[feature] = minmax_stats[feature]
        else:
            print(f"⚠️  Warning: Feature '{feature}' not found in minmax_stats.json, skipping...")
    
    if not feature_stats and not remove_empty_images:
        raise ValueError("No valid features to check. Please provide valid feature names or enable --remove_empty_images.")
    
    if verbose:
        print(f"Dataset: {input_path}")
        print(f"Total episodes: {len(episodes)}")
        print(f"FPS: {fps}")
        print(f"Remove mode: {remove_mode} (remove {'entire episode' if remove_mode == 'episode' else 'only abnormal frames'})")
        print(f"Threshold multiplier: {threshold_multiplier}")
        print(f"Remove empty images: {remove_empty_images}")
        if feature_stats:
            print(f"\n📋 Features to check ({len(feature_stats)}):")
            for feat_name, stats in feature_stats.items():
                shape = stats.get("shape", "unknown")
                if "global_q01" in stats:
                    print(f"  • {feat_name}: shape={shape}, q01={stats['global_q01']:.4f}, q99={stats['global_q99']:.4f}")
                else:
                    print(f"  • {feat_name}: shape={shape}, q01={stats.get('q01', 'N/A')}, q99={stats.get('q99', 'N/A')}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_abnormal_frames = 0
    total_original_frames = 0
    removed_episodes = 0
    removed_empty_image_episodes = 0
    all_abnormal_info: List[Dict] = []
    empty_image_episodes_info: List[Dict] = []
    
    if verbose:
        print("\n🔄 Processing episodes and removing abnormal frames...")
    
    new_episode_index = 0
    new_global_index = 0
    new_episodes = []
    
    for episode in tqdm(episodes, desc="Processing episodes", disable=not verbose):
        old_episode_idx = episode["episode_index"]
        chunk_idx = old_episode_idx // 1000
        parquet_path = input_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{old_episode_idx:06d}.parquet"
        
        if not parquet_path.exists():
            continue
        
        # Read table
        table = pq.read_table(parquet_path)
        original_num_rows = table.num_rows
        total_original_frames += original_num_rows
        
        # Check for empty images if enabled
        if remove_empty_images:
            is_empty, empty_info = check_episode_has_empty_images(table, image_columns)
            if is_empty:
                removed_empty_image_episodes += 1
                empty_image_episodes_info.append({
                    "episode_index": old_episode_idx,
                    "empty_columns": empty_info.get("empty_columns", []),
                    "column_stats": empty_info.get("column_stats", {})
                })
                if verbose:
                    print(f"  [空图片] Episode {old_episode_idx}: 空列={empty_info.get('empty_columns', [])}")
                continue
        
        # Detect abnormal frames
        abnormal_indices, abnormal_info = detect_abnormal_frames(
            table, feature_stats, threshold_multiplier
        ) if feature_stats else ([], [])
        
        if abnormal_indices:
            total_abnormal_frames += len(abnormal_indices)
            all_abnormal_info.append({
                "episode_index": old_episode_idx,
                "num_abnormal_frames": len(abnormal_indices),
                "abnormal_frames": abnormal_info
            })
            
            # If remove_mode is "episode", skip this entire episode
            if remove_mode == "episode":
                removed_episodes += 1
                continue
        
        # Get indices of normal frames (frames to keep) - only used in "frame" mode
        abnormal_set = set(abnormal_indices)
        normal_indices = [i for i in range(original_num_rows) if i not in abnormal_set]
        
        # Skip if episode becomes too short after removing abnormal frames
        if len(normal_indices) < min_episode_length:
            removed_episodes += 1
            continue
        
        # Filter table to keep only normal frames (only in "frame" mode)
        if abnormal_indices and remove_mode == "frame":
            # Create a filtered table with only normal frames
            table = table.take(normal_indices)
        
        num_rows = table.num_rows
        
        # Update indices and timestamps
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
                    type=pa.int64()
                )
            elif col_name == "timestamp":
                # Recalculate timestamps to be continuous based on fps
                new_timestamps = [i / fps for i in range(num_rows)]
                new_columns[col_name] = pa.array(new_timestamps, type=pa.float32())
            else:
                new_columns[col_name] = col
        
        # Reconstruct table with original schema
        arrays = [new_columns[field.name] for field in schema]
        new_table = pa.table(dict(zip([f.name for f in schema], arrays)), schema=schema)
        
        # Write to output
        new_chunk_idx = new_episode_index // 1000
        new_chunk_dir = output_path / "data" / f"chunk-{new_chunk_idx:03d}"
        new_chunk_dir.mkdir(exist_ok=True)
        new_parquet_path = new_chunk_dir / f"episode_{new_episode_index:06d}.parquet"
        pq.write_table(new_table, new_parquet_path)
        
        # Update episode metadata
        new_episode = {
            "episode_index": new_episode_index,
            "tasks": episode.get("tasks", ["unknown task"]),
            "length": num_rows
        }
        new_episodes.append(new_episode)
        
        new_global_index += num_rows
        new_episode_index += 1
    
    # Write metadata
    if verbose:
        print("\n💾 Writing metadata...")
    
    # Write info.json
    new_info = info.copy()
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_frames"] = new_global_index
    new_info["total_chunks"] = (len(new_episodes) // 1000) + 1
    new_info["splits"] = {"train": f"0:{len(new_episodes)}"}
    
    with open(output_path / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(new_info, f, indent=4)
    
    # Write episodes.jsonl
    with open(output_path / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for episode in new_episodes:
            f.write(json.dumps(episode) + "\n")
    
    # Write tasks.jsonl (copy from original)
    with open(output_path / "meta" / "tasks.jsonl", "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    
    # Generate and write episodes_stats.jsonl
    if verbose:
        print("📊 Computing episode statistics...")
    
    with open(output_path / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for episode in tqdm(new_episodes, desc="Computing stats", disable=not verbose):
            ep_idx = episode["episode_index"]
            chunk_idx = ep_idx // 1000
            parquet_path = output_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            
            if parquet_path.exists():
                table = pq.read_table(parquet_path)
                episode_stats = compute_episode_stats(table)
                stats_entry = {
                    "episode_index": ep_idx,
                    "stats": episode_stats
                }
                f.write(json.dumps(stats_entry) + "\n")
    
    # Write abnormal frames report
    report_path = output_path / "abnormal_frames_report.json"
    
    # Build feature thresholds info for report
    feature_thresholds = {}
    for feat_name, stats in feature_stats.items():
        feature_thresholds[feat_name] = {
            "shape": stats.get("shape"),
            "q01": stats.get("global_q01", stats.get("q01")),
            "q99": stats.get("global_q99", stats.get("q99")),
            "threshold_multiplier": threshold_multiplier
        }
    
    with open(report_path, "w", encoding="utf-8") as f:
        report_data = {
            "remove_mode": remove_mode,
            "threshold_multiplier": threshold_multiplier,
            "features_checked": list(feature_stats.keys()),
            "feature_thresholds": feature_thresholds,
            "remove_empty_images": remove_empty_images,
            "total_original_episodes": len(episodes),
            "total_original_frames": total_original_frames,
            "total_abnormal_frames": total_abnormal_frames,
            "removed_episodes": removed_episodes,
            "removed_empty_image_episodes": removed_empty_image_episodes,
            "final_episodes": len(new_episodes),
            "final_frames": new_global_index,
            "episodes_with_abnormal_frames": all_abnormal_info
        }
        if remove_empty_images:
            report_data["empty_image_episodes"] = empty_image_episodes_info
        json.dump(report_data, f, indent=2)
    
    if verbose:
        print("\n✅ Cleaning complete!")
        print(f"  Remove mode: {remove_mode}")
        print(f"  Original episodes: {len(episodes)}")
        print(f"  Original frames: {total_original_frames}")
        if remove_mode == "frame":
            print(f"  Removed abnormal frames: {total_abnormal_frames}")
            print(f"  Removed episodes (too short after cleaning): {removed_episodes}")
        else:
            print(f"  Removed episodes (with abnormal frames): {removed_episodes}")
        if remove_empty_images:
            print(f"  Removed episodes (empty images): {removed_empty_image_episodes}")
        print(f"  Final episodes: {len(new_episodes)}")
        print(f"  Final frames: {new_global_index}")
        print(f"  Output directory: {output_path}")
        print(f"  Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove abnormal frames/episodes from UMI LeRobot dataset"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for cleaned dataset"
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=["observation.robot0_eef_pos"],
        help="Feature names to check for abnormality (default: observation.robot0_eef_pos). "
             "Example: --features observation.robot0_eef_pos observation.robot0_eef_rot_axis_angle"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Threshold multiplier for q01/q99 (default: 1.5, value outside [q01*t, q99*t] is abnormal)"
    )
    parser.add_argument(
        "--min_episode_length",
        type=int,
        default=600,
        help="Minimum episode length to keep (default: 50)"
    )
    parser.add_argument(
        "--remove_mode",
        type=str,
        choices=["frame", "episode"],
        default="frame",
        help="Remove mode: 'frame' to remove only abnormal frames, 'episode' to remove entire episode (default: frame)"
    )
    parser.add_argument(
        "--remove_empty_images",
        type=lambda x: x.lower() != 'false',
        default=True,
        help="Remove episodes where any image column has all-zero (empty) images (default: True). "
             "Use --remove_empty_images=false to disable."
    )
    parser.add_argument(
        "--image_columns",
        type=str,
        nargs="+",
        default=None,
        help="Image column names to check for empty images. If not specified, auto-detect. "
             "Example: --image_columns right_wrist_0_rgb_0 left_wrist_0_rgb_0"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
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
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

