"""
Merge multiple UMI LeRobot datasets into a single dataset.

This script merges multiple umi_lerobot_dataset directories into one,
re-encoding frame indices, episode indices, and global indices.

Usage:
    python merge_dataset.py --input_dirs dataset1 dataset2 ... --output_dir merged_dataset
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm


def load_info(dataset_path: Path) -> Dict[str, Any]:
    """Load dataset info.json"""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r") as f:
        return json.load(f)


def load_episodes(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load episodes from episodes.jsonl"""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def load_tasks(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load tasks from tasks.jsonl"""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    tasks = []
    with open(tasks_path, "r") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def load_episodes_stats(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load episode statistics from episodes_stats.jsonl"""
    stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    stats = []
    if stats_path.exists():
        with open(stats_path, "r") as f:
            for line in f:
                if line.strip():
                    stats.append(json.loads(line))
    return stats


def update_arrow_table_indices(
    table: pa.Table,
    new_episode_index: int,
    new_frame_start: int,
    new_global_start: int,
    local_task_to_global: Dict[int, int]
) -> pa.Table:
    """
    Update indices in an Arrow table while preserving schema metadata.
    
    Args:
        table: Original Arrow table
        new_episode_index: New episode index to set
        new_frame_start: Starting frame index
        new_global_start: Starting global index
        local_task_to_global: Mapping from local task indices to global
    
    Returns:
        Updated Arrow table with same schema
    """
    num_rows = table.num_rows
    schema = table.schema
    
    # Build new columns
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
                range(new_global_start, new_global_start + num_rows), 
                type=pa.int64()
            )
        elif col_name == "task_index":
            # Map local task indices to global
            old_values = col.to_pylist()
            new_values = [local_task_to_global.get(v, 0) for v in old_values]
            new_columns[col_name] = pa.array(new_values, type=pa.int64())
        else:
            new_columns[col_name] = col
    
    # Reconstruct table with original schema to preserve metadata
    arrays = [new_columns[field.name] for field in schema]
    new_table = pa.table(dict(zip([f.name for f in schema], arrays)), schema=schema)
    
    return new_table


def merge_datasets(
    input_dirs: List[str],
    output_dir: str,
    verbose: bool = True
) -> None:
    """
    Merge multiple LeRobot datasets into one.
    
    Args:
        input_dirs: List of input dataset directories
        output_dir: Output directory for merged dataset
        verbose: Whether to print progress information
    """
    input_paths = [Path(d) for d in input_dirs]
    output_path = Path(output_dir)
    
    # Validate input directories
    for path in input_paths:
        if not path.exists():
            raise ValueError(f"Input directory does not exist: {path}")
        if not (path / "meta" / "info.json").exists():
            raise ValueError(f"Not a valid LeRobot dataset (missing meta/info.json): {path}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    # Load first dataset's info as template
    base_info = load_info(input_paths[0])
    
    # Get reference schema from first dataset's first episode
    reference_schema: Optional[pa.Schema] = None
    for input_path in input_paths:
        episodes = load_episodes(input_path)
        if episodes:
            first_ep = episodes[0]
            chunk_idx = first_ep["episode_index"] // 1000
            parquet_path = input_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{first_ep['episode_index']:06d}.parquet"
            if parquet_path.exists():
                reference_schema = pq.read_table(parquet_path).schema
                break
    
    if reference_schema is None:
        raise ValueError("Could not find any valid parquet files in input datasets")
    
    # Verify all datasets have compatible structure
    for path in input_paths[1:]:
        info = load_info(path)
        if info["features"] != base_info["features"]:
            print(f"Warning: Feature mismatch between {input_paths[0]} and {path}")
            print("Proceeding anyway, but this may cause issues...")
    
    # Track global counters
    global_episode_index = 0
    global_frame_index = 0
    total_frames = 0
    
    # Collect all episodes and tasks
    all_episodes = []
    all_tasks = {}  # task_text -> task_index
    all_episodes_stats = []
    task_counter = 0
    
    if verbose:
        print(f"Merging {len(input_paths)} datasets into {output_path}")
    
    # First pass: collect all unique tasks
    for input_path in input_paths:
        tasks = load_tasks(input_path)
        for task in tasks:
            task_text = task["task"]
            if task_text not in all_tasks:
                all_tasks[task_text] = task_counter
                task_counter += 1
    
    # Second pass: process each dataset
    for dataset_idx, input_path in enumerate(input_paths):
        if verbose:
            print(f"\nProcessing dataset {dataset_idx + 1}/{len(input_paths)}: {input_path}")
        
        episodes = load_episodes(input_path)
        episodes_stats = load_episodes_stats(input_path)
        tasks = load_tasks(input_path)
        
        # Build task text to new index mapping for this dataset
        local_task_to_global = {}
        for task in tasks:
            local_task_to_global[task["task_index"]] = all_tasks[task["task"]]
        
        # Process each episode
        for local_ep_idx, episode in enumerate(tqdm(episodes, desc=f"Dataset {dataset_idx + 1}", disable=not verbose)):
            old_episode_index = episode["episode_index"]
            episode_length = episode["length"]
            
            # Determine episode parquet file
            chunk_idx = old_episode_index // 1000
            parquet_path = input_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{old_episode_index:06d}.parquet"
            
            if not parquet_path.exists():
                print(f"Warning: Missing parquet file: {parquet_path}")
                continue
            
            # Read parquet file (preserves Arrow schema with metadata)
            table = pq.read_table(parquet_path)
            
            # Verify episode length
            actual_length = table.num_rows
            if actual_length != episode_length:
                print(f"Warning: Episode {old_episode_index} length mismatch: expected {episode_length}, got {actual_length}")
                episode_length = actual_length
            
            # Update indices while preserving schema
            new_table = update_arrow_table_indices(
                table,
                new_episode_index=global_episode_index,
                new_frame_start=0,
                new_global_start=global_frame_index,
                local_task_to_global=local_task_to_global
            )
            
            # Write new parquet file
            new_chunk_idx = global_episode_index // 1000
            new_chunk_dir = output_path / "data" / f"chunk-{new_chunk_idx:03d}"
            new_chunk_dir.mkdir(exist_ok=True)
            new_parquet_path = new_chunk_dir / f"episode_{global_episode_index:06d}.parquet"
            
            # Write with preserved schema
            pq.write_table(new_table, new_parquet_path)
            
            # Update episode metadata
            new_episode = {
                "episode_index": global_episode_index,
                "tasks": episode.get("tasks", ["unknown task"]),
                "length": episode_length
            }
            all_episodes.append(new_episode)
            
            # Update episode stats if available
            if episodes_stats and local_ep_idx < len(episodes_stats):
                stats = episodes_stats[local_ep_idx].copy()
                stats["episode_index"] = global_episode_index
                all_episodes_stats.append(stats)
            
            # Update counters
            global_frame_index += episode_length
            total_frames += episode_length
            global_episode_index += 1
    
    # Write merged metadata
    if verbose:
        print("\nWriting merged metadata...")
    
    # Write info.json
    merged_info = base_info.copy()
    merged_info["total_episodes"] = global_episode_index
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(all_tasks)
    merged_info["total_chunks"] = (global_episode_index // 1000) + 1
    merged_info["splits"] = {"train": f"0:{global_episode_index}"}
    
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(merged_info, f, indent=4)
    
    # Write episodes.jsonl
    with open(output_path / "meta" / "episodes.jsonl", "w") as f:
        for episode in all_episodes:
            f.write(json.dumps(episode) + "\n")
    
    # Write tasks.jsonl
    with open(output_path / "meta" / "tasks.jsonl", "w") as f:
        for task_text, task_index in sorted(all_tasks.items(), key=lambda x: x[1]):
            task_entry = {"task_index": task_index, "task": task_text}
            f.write(json.dumps(task_entry) + "\n")
    
    # Write episodes_stats.jsonl if we have stats
    if all_episodes_stats:
        with open(output_path / "meta" / "episodes_stats.jsonl", "w") as f:
            for stats in all_episodes_stats:
                f.write(json.dumps(stats) + "\n")
    
    if verbose:
        print(f"\nMerge complete!")
        print(f"  Total episodes: {global_episode_index}")
        print(f"  Total frames: {total_frames}")
        print(f"  Total tasks: {len(all_tasks)}")
        print(f"  Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple UMI LeRobot datasets into one"
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="List of input dataset directories to merge"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for merged dataset"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    merge_datasets(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
