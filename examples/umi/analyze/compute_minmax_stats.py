#!/usr/bin/env python3
"""
Compute min/max and percentile (q01, q99) statistics for all dimensions in a LeRobot dataset.
Directly reads parquet files to avoid network dependency.

Usage:
    python compute_minmax_stats.py --dataset /root/openpi-umi/data/umi_lerobot_dataset_v7.2
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def nested_to_numpy(value):
    """
    Recursively convert nested lists/arrays to a proper numpy array.
    Handles parquet's nested list storage format.
    """
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            # Object array - need to convert elements
            try:
                return np.array([nested_to_numpy(v) for v in value])
            except Exception:
                return None
        return value
    elif isinstance(value, (list, tuple)):
        try:
            # Try direct conversion first
            arr = np.array(value)
            if arr.dtype == object:
                # Need recursive conversion
                converted = [nested_to_numpy(v) for v in value]
                if any(v is None for v in converted):
                    return None
                return np.array(converted)
            return arr
        except Exception:
            return None
    else:
        return np.array(value)


def compute_minmax_stats(dataset_path: str, max_episodes: int = None, verbose: bool = False):
    """
    Compute min/max statistics for all features in a LeRobot dataset.
    Directly reads parquet files.
    
    Args:
        dataset_path: Path to the dataset directory
        max_episodes: Maximum number of episodes to process (for faster debugging)
        verbose: Print verbose debug information
    """
    dataset_path = Path(dataset_path)
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Load info.json for metadata
    info_path = dataset_path / "meta" / "info.json"
    feature_shapes = {}
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        print("\nDataset info:")
        print(f"  Total episodes: {info.get('total_episodes', 'N/A')}")
        print(f"  Total frames: {info.get('total_frames', 'N/A')}")
        print(f"  FPS: {info.get('fps', 'N/A')}")
        
        features = info.get('features', {})
        print(f"  Features: {list(features.keys())}")
        
        # Extract expected shapes
        for fname, finfo in features.items():
            if 'shape' in finfo:
                feature_shapes[fname] = tuple(finfo['shape'])
                if verbose:
                    print(f"    {fname}: shape={finfo['shape']}, dtype={finfo.get('dtype', 'N/A')}")
    
    # Find all parquet files
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    
    if not parquet_files:
        print(f"❌ No parquet files found in {data_dir}")
        return None
    
    print(f"\nFound {len(parquet_files)} parquet files")
    
    # Limit episodes if specified
    if max_episodes is not None and max_episodes < len(parquet_files):
        parquet_files = parquet_files[:max_episodes]
        print(f"⚠️  DEBUG MODE: Only processing first {max_episodes} episodes")
    
    # Initialize statistics containers
    # 使用列表收集所有值，以便计算分位数
    stats = defaultdict(lambda: {"min": None, "max": None, "q01": None, "q99": None, "shape": None, "dtype": None, "count": 0, "all_values": []})
    
    print("\nComputing min/max and percentile statistics...")
    
    # Process all parquet files
    for parquet_file in tqdm(parquet_files, desc="Processing episodes"):
        table = pq.read_table(parquet_file)
        
        for column_name in table.column_names:
            # Skip metadata columns
            if column_name in ['episode_index', 'frame_index', 'timestamp', 'index', 'task_index', 
            'observation.left_wrist_0_rgb_0', 'observation.left_wrist_0_rgb_1']:
                continue
            
            column = table.column(column_name)
            
            # Convert column to python/numpy
            try:
                # Use to_pylist() for nested arrays, then convert to numpy
                py_values = column.to_pylist()
                
                if len(py_values) == 0:
                    continue
                
                first_val = py_values[0]
                
                # Skip string columns
                if isinstance(first_val, str):
                    continue
                
                # Handle nested arrays (lists of lists)
                if isinstance(first_val, (list, np.ndarray)):
                    # Convert all values to numpy arrays
                    arrays = []
                    for v in py_values:
                        arr = nested_to_numpy(v)
                        if arr is None:
                            break
                        arrays.append(arr)
                    
                    if len(arrays) != len(py_values):
                        if verbose:
                            print(f"  ⚠️  Column '{column_name}': failed to convert all values")
                        continue
                    
                    # Stack into single array
                    try:
                        stacked = np.stack(arrays)
                    except ValueError as e:
                        if verbose:
                            print(f"  ⚠️  Column '{column_name}': cannot stack - {e}")
                            print(f"      Shapes: {[a.shape for a in arrays[:3]]}...")
                        continue
                    
                    # Skip image data (uint8 with 3+ dimensions, or large arrays)
                    if stacked.dtype == np.uint8 and stacked.ndim >= 3 and stacked.shape[-1] == 3:
                        if stats[column_name]["shape"] is None:
                            stats[column_name]["shape"] = stacked.shape[1:]
                            stats[column_name]["dtype"] = str(stacked.dtype)
                            stats[column_name]["min"] = "N/A (image)"
                            stats[column_name]["max"] = "N/A (image)"
                        continue
                    
                    # Compute per-dimension min/max
                    sample_shape = stacked.shape[1:]
                    
                    # Flatten for min/max computation
                    flat = stacked.reshape(len(stacked), -1).astype(np.float64)
                    current_min = flat.min(axis=0)
                    current_max = flat.max(axis=0)
                    
                    if stats[column_name]["min"] is None:
                        stats[column_name]["min"] = current_min
                        stats[column_name]["max"] = current_max
                        stats[column_name]["shape"] = sample_shape
                        stats[column_name]["dtype"] = str(stacked.dtype)
                    else:
                        stats[column_name]["min"] = np.minimum(stats[column_name]["min"], current_min)
                        stats[column_name]["max"] = np.maximum(stats[column_name]["max"], current_max)
                    
                    # 收集所有值用于计算分位数
                    stats[column_name]["all_values"].append(flat)
                    stats[column_name]["count"] += len(stacked)
                
                # Scalar columns
                elif isinstance(first_val, (int, float, np.number)):
                    arr = np.array(py_values, dtype=np.float64)
                    current_min = arr.min()
                    current_max = arr.max()
                    
                    if stats[column_name]["min"] is None:
                        stats[column_name]["min"] = current_min
                        stats[column_name]["max"] = current_max
                        stats[column_name]["shape"] = ()
                        stats[column_name]["dtype"] = str(arr.dtype)
                    else:
                        stats[column_name]["min"] = min(stats[column_name]["min"], current_min)
                        stats[column_name]["max"] = max(stats[column_name]["max"], current_max)
                    
                    # 收集所有值用于计算分位数
                    stats[column_name]["all_values"].append(arr.reshape(-1, 1))
                    stats[column_name]["count"] += len(arr)
                    
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Column '{column_name}': error - {type(e).__name__}: {e}")
                continue
    
    # 计算分位数
    print("\nComputing percentiles (q01, q99)...")
    for key in tqdm(stats.keys(), desc="Computing percentiles"):
        stat = stats[key]
        if stat["all_values"] and not isinstance(stat["min"], str):
            # 合并所有值
            all_values = np.vstack(stat["all_values"])
            # 计算 q01 和 q99
            stat["q01"] = np.percentile(all_values, 1, axis=0)
            stat["q99"] = np.percentile(all_values, 99, axis=0)
            # 清理内存
            del stat["all_values"]
        else:
            del stat["all_values"]
    
    # Print results
    print("\n" + "=" * 80)
    print("MIN/MAX AND PERCENTILE STATISTICS")
    print("=" * 80)
    
    results = {}
    for key in sorted(stats.keys()):
        stat = stats[key]
        if stat["shape"] is None:
            continue
            
        print(f"\n📊 {key}")
        print(f"   Shape: {stat['shape']}")
        print(f"   Dtype: {stat['dtype']}")
        print(f"   Samples: {stat['count']}")
        
        if isinstance(stat["min"], str):
            print(f"   Min: {stat['min']}")
            print(f"   Max: {stat['max']}")
            results[key] = {
                "shape": list(stat["shape"]) if isinstance(stat["shape"], tuple) else stat["shape"],
                "dtype": stat["dtype"],
                "min": stat["min"],
                "max": stat["max"],
            }
        else:
            min_val = stat["min"]
            max_val = stat["max"]
            q01_val = stat["q01"]
            q99_val = stat["q99"]
            
            if isinstance(min_val, (int, float, np.floating)) or (isinstance(min_val, np.ndarray) and min_val.ndim == 0):
                print(f"   Min: {float(min_val):.6f}")
                print(f"   Q01: {float(q01_val):.6f}")
                print(f"   Q99: {float(q99_val):.6f}")
                print(f"   Max: {float(max_val):.6f}")
                results[key] = {
                    "shape": list(stat["shape"]) if isinstance(stat["shape"], tuple) else stat["shape"],
                    "dtype": stat["dtype"],
                    "min": float(min_val),
                    "q01": float(q01_val),
                    "q99": float(q99_val),
                    "max": float(max_val),
                }
            else:
                # Reshape back to original shape for display
                shape = stat["shape"]
                min_reshaped = min_val.reshape(shape) if shape else min_val
                max_reshaped = max_val.reshape(shape) if shape else max_val
                q01_reshaped = q01_val.reshape(shape) if shape else q01_val
                q99_reshaped = q99_val.reshape(shape) if shape else q99_val
                
                print("   Min (per dim):")
                _print_nested(min_reshaped, indent=6)
                print("   Q01 (per dim):")
                _print_nested(q01_reshaped, indent=6)
                print("   Q99 (per dim):")
                _print_nested(q99_reshaped, indent=6)
                print("   Max (per dim):")
                _print_nested(max_reshaped, indent=6)
                print(f"   Global Min: {float(min_val.min()):.6f}")
                print(f"   Global Q01: {float(q01_val.min()):.6f}")
                print(f"   Global Q99: {float(q99_val.max()):.6f}")
                print(f"   Global Max: {float(max_val.max()):.6f}")
                results[key] = {
                    "shape": list(shape) if isinstance(shape, tuple) else shape,
                    "dtype": stat["dtype"],
                    "min_per_dim": min_reshaped.tolist(),
                    "q01_per_dim": q01_reshaped.tolist(),
                    "q99_per_dim": q99_reshaped.tolist(),
                    "max_per_dim": max_reshaped.tolist(),
                    "global_min": float(min_val.min()),
                    "global_q01": float(q01_val.min()),
                    "global_q99": float(q99_val.max()),
                    "global_max": float(max_val.max()),
                }
    
    # Save results to JSON
    output_file = dataset_path / "minmax_stats.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Statistics saved to: {output_file}")
    
    return results


def _print_nested(arr, indent=0):
    """Pretty print nested array with indentation."""
    prefix = " " * indent
    if arr.ndim == 1:
        formatted = [f"{v:.4f}" for v in arr]
        print(f"{prefix}[{', '.join(formatted)}]")
    else:
        print(f"{prefix}[")
        for row in arr:
            _print_nested(row, indent + 2)
        print(f"{prefix}]")


def main():
    parser = argparse.ArgumentParser(description="Compute min/max statistics for LeRobot dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/root/openpi-umi/data/umi_lerobot_dataset_v7.2",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process (for faster debugging)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose debug information",
    )
    
    args = parser.parse_args()
    
    compute_minmax_stats(
        dataset_path=args.dataset,
        max_episodes=args.max_episodes,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
