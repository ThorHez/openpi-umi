#!/usr/bin/env python3
"""
Convert UMI zarr dataset to LeRobot format with multiprocessing support.

This version uses parallel processing to significantly speed up the conversion:
- Parallel episode data extraction
- Parallel image processing
- Batch frame insertion

Usage:
    python convert_umi_data_to_lerobot_parallel.py \
        --input dataset.zarr.zip \
        --output ./umi_lerobot_dataset \
        --repo-id your_hf_username/umi_dataset \
        --fps 30 \
        --workers 8
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import zarr
from tqdm import tqdm

# Import LeRobot
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("Error: Required packages not installed. Please install them first:")
    print("  pip install lerobot datasets zarr numpy tqdm")
    exit(1)


def load_zarr_dataset(zarr_zip_path: str):
    """Load UMI zarr dataset from zip file."""
    print(f"Loading zarr dataset from: {zarr_zip_path}")
    
    # Extract to temporary directory in /root (where we have space)
    temp_dir = tempfile.mkdtemp(dir="/root")
    print(f"Extracting to temporary directory: {temp_dir}")
    with zipfile.ZipFile(zarr_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Open zarr store
    root = zarr.open(temp_dir, mode='r')
    
    return root, temp_dir


def process_episode(args: Tuple[int, int, int, str, str]) -> List[Dict]:
    """
    Process a single episode and return all frames as a list of dictionaries.
    
    This function runs in a separate process to enable parallel processing.
    
    Args:
        args: Tuple of (episode_idx, episode_start, episode_end, temp_dir, default_task)
    
    Returns:
        List of frame dictionaries ready to be added to the dataset
    """
    _episode_idx, episode_start, episode_end, temp_dir, default_task = args
    
    # Open zarr in this process (each process needs its own handle)
    root = zarr.open(temp_dir, mode='r')
    data = root['data']
    
    frames = []
    
    for frame_idx in range(episode_start, episode_end):
        # Construct action (next state for all but last frame)
        if frame_idx < episode_end - 1:
            action_7d = np.concatenate([
                data['robot0_eef_pos'][frame_idx + 1],
                data['robot0_eef_rot_axis_angle'][frame_idx + 1],
                data['robot0_gripper_width'][frame_idx + 1],
            ]).astype(np.float32)
        else:
            # Last frame: use current state as action
            action_7d = np.concatenate([
                data['robot0_eef_pos'][frame_idx],
                data['robot0_eef_rot_axis_angle'][frame_idx],
                data['robot0_gripper_width'][frame_idx],
            ]).astype(np.float32)
        
        action = action_7d
        
        # Create frame dictionary
        frame_data = {
            "observation.robot0_eef_pos": data['robot0_eef_pos'][frame_idx].astype(np.float32),
            "observation.robot0_eef_rot_axis_angle": data['robot0_eef_rot_axis_angle'][frame_idx].astype(np.float32),
            "observation.robot0_gripper_width": data['robot0_gripper_width'][frame_idx].astype(np.float32).reshape(1),
            "observation.camera0_rgb": np.array(data['camera0_rgb'][frame_idx]),  # Copy to avoid zarr reference
            "actions": action,
            "task": default_task,
        }
        frames.append(frame_data)
    
    return frames


def convert_to_lerobot(
    zarr_path: str,
    output_dir: str,
    repo_id: str,
    fps: int = 30,
    default_task: str = "umi manipulation task",
    push_to_hub: bool = False,
    num_workers: int = None,
    batch_size: int = 100,
):
    """
    Convert UMI zarr dataset to LeRobot format with multiprocessing.
    
    Args:
        zarr_path: Path to the zarr zip file
        output_dir: Output directory for LeRobot dataset
        repo_id: HuggingFace repository ID (e.g., "username/dataset_name")
        fps: Frames per second of the dataset
        default_task: Default task description if not available in data
        push_to_hub: Whether to push to HuggingFace Hub after conversion
        num_workers: Number of worker processes (default: cpu_count())
        batch_size: Number of episodes to process in each batch
    """
    
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"🚀 Using {num_workers} worker processes for parallel conversion")
    
    # Load zarr dataset
    root, temp_dir = load_zarr_dataset(zarr_path)
    
    try:
        data = root['data']
        meta = root['meta']
        
        # Get episode information
        episode_ends = np.array(meta['episode_ends'][:])
        num_episodes = len(episode_ends)
        total_steps = episode_ends[-1]
        
        print("\n📊 Dataset info:")
        print(f"  Total episodes: {num_episodes}")
        print(f"  Total steps: {total_steps}")
        print(f"  Average steps per episode: {total_steps / num_episodes:.1f}")
        
        # Print data structure
        print("\n📁 Data arrays:")
        for key in data.keys():
            arr = data[key]
            print(f"  • {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        # Prepare episode ranges
        episode_starts = [0] + list(episode_ends[:-1])
        episode_args = [
            (idx, int(start), int(end), temp_dir, default_task)
            for idx, (start, end) in enumerate(zip(episode_starts, episode_ends))
        ]
        
        # Create LeRobot dataset with explicit features
        print(f"\n🔨 Creating LeRobot dataset: {repo_id}")
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=output_dir,
            robot_type="umi",
            features={
                "observation.robot0_eef_pos": {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": ["x", "y", "z"],
                },
                "observation.robot0_eef_rot_axis_angle": {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": ["rx", "ry", "rz"],
                },
                "observation.robot0_gripper_width": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["gripper_width"],
                },
                "observation.camera0_rgb": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
                    "names": ["height", "width", "channel"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["actions"],
                },
            },
            use_videos=False,  # Store images as PNG
            image_writer_threads=num_workers,  # Use more threads for image writing
            image_writer_processes=1,
        )
        
        # Process episodes in parallel batches
        print("\n🔄 Processing episodes in parallel...")
        
        with Pool(processes=num_workers) as pool:
            # Process episodes in batches to manage memory
            for batch_start in range(0, num_episodes, batch_size):
                batch_end = min(batch_start + batch_size, num_episodes)
                batch_args = episode_args[batch_start:batch_end]
                
                print(f"\n📦 Processing episodes {batch_start} to {batch_end-1} ({len(batch_args)} episodes)...")
                
                # Process episodes in parallel with progress bar
                episode_frames_list = list(tqdm(
                    pool.imap(process_episode, batch_args),
                    total=len(batch_args),
                    desc="Processing episodes",
                    unit="episode"
                ))
                
                # Add frames to dataset sequentially (LeRobot dataset needs sequential writes)
                print("💾 Writing frames to dataset...")
                for episode_frames in tqdm(episode_frames_list, desc="Writing episodes", unit="episode"):
                    for frame_data in episode_frames:
                        lerobot_dataset.add_frame(frame_data)
                    lerobot_dataset.save_episode()
        
        print("\n✅ Conversion complete!")
        print(f"   Output directory: {output_dir}")
        print(f"   Total episodes: {num_episodes}")
        print(f"   Total frames: {total_steps}")
        
        # Push to HuggingFace Hub if requested
        if push_to_hub:
            print(f"\n📤 Pushing to HuggingFace Hub: {repo_id}")
            print("   Note: Make sure you're logged in with `huggingface-cli login`")
            try:
                lerobot_dataset.push_to_hub(repo_id=repo_id)
                print(f"✅ Successfully pushed to https://huggingface.co/datasets/{repo_id}")
            except Exception as e:  # noqa: BLE001
                print(f"❌ Error pushing to hub: {e}")
                print("   You can manually push later using:")
                print(f"   lerobot_dataset.push_to_hub('{repo_id}')")
        else:
            print("\n💡 To push to HuggingFace Hub later, run:")
            print("   from lerobot.common.datasets.lerobot_dataset import LeRobotDataset")
            print(f"   dataset = LeRobotDataset('{repo_id}', root='{output_dir}')")
            print(f"   dataset.push_to_hub('{repo_id}')")
        
    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert UMI zarr dataset to LeRobot format (parallel version)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input zarr zip file (e.g., dataset.zarr.zip)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./umi_lerobot_dataset",
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'your_username/umi_dataset')",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second of the dataset (default: 30)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="umi manipulation task",
        help="Default task description (default: 'umi manipulation task')",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push to HuggingFace Hub after conversion",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker processes (default: {cpu_count()})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of episodes to process in each batch (default: 100)",
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        exit(1)
    
    print("=" * 60)
    print("UMI to LeRobot Dataset Conversion (Parallel)")
    print("=" * 60)
    print(f"\n📥 Input: {args.input}")
    print(f"📤 Output: {args.output}")
    print(f"🏷️  Repo ID: {args.repo_id}")
    print(f"⏱️  FPS: {args.fps}")
    print(f"📝 Task: {args.task}")
    print(f"☁️  Push to Hub: {args.push_to_hub}")
    print(f"👷 Workers: {args.workers if args.workers else cpu_count()}")
    print(f"📦 Batch size: {args.batch_size}")
    print()
    
    # Convert dataset
    convert_to_lerobot(
        zarr_path=args.input,
        output_dir=args.output,
        repo_id=args.repo_id,
        fps=args.fps,
        default_task=args.task,
        push_to_hub=args.push_to_hub,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

