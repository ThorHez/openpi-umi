#!/usr/bin/env python3
"""
Convert UMI zarr dataset to LeRobot format.

Usage:
    python convert_umi_data_to_lerobot.py \
        --input dataset.zarr.zip \
        --output ./umi_lerobot_dataset \
        --repo-id your_hf_username/umi_dataset \
        --fps 30
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

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


def convert_to_lerobot(
    zarr_path: str,
    output_dir: str,
    repo_id: str,
    fps: int = 30,
    default_task: str = "umi manipulation task",
    push_to_hub: bool = False,
):
    """
    Convert UMI zarr dataset to LeRobot format.
    
    Args:
        zarr_path: Path to the zarr zip file
        output_dir: Output directory for LeRobot dataset
        repo_id: HuggingFace repository ID (e.g., "username/dataset_name")
        fps: Frames per second of the dataset
        default_task: Default task description if not available in data
        push_to_hub: Whether to push to HuggingFace Hub after conversion
    """
    
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
        
        # Create LeRobot dataset with explicit features
        # Following the format used in examples/libero/convert_libero_data_to_lerobot.py
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
            image_writer_threads=4,  # Parallel image writing for faster conversion
            image_writer_processes=1,
        )
        
        # Convert each episode
        episode_start = 0
        for episode_idx in tqdm(range(num_episodes), desc="Converting episodes"):
            episode_end = episode_ends[episode_idx]
            
            # Add frames for this episode
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
                
                # Pad action to 32 dimensions (to match pi05_base pretrained model)
                # action = np.zeros(32, dtype=np.float32)
                # action[:7] = action_7d
                action = action_7d
                
                # Add frame to dataset
                lerobot_dataset.add_frame({
                    "observation.robot0_eef_pos": data['robot0_eef_pos'][frame_idx].astype(np.float32),
                    "observation.robot0_eef_rot_axis_angle": data['robot0_eef_rot_axis_angle'][frame_idx].astype(np.float32),
                    "observation.robot0_gripper_width": data['robot0_gripper_width'][frame_idx].astype(np.float32).reshape(1),  # Ensure shape (1,)
                    "observation.camera0_rgb": data['camera0_rgb'][frame_idx],
                    "actions": action,
                    "task": default_task,
                })
            
            # Mark episode as complete
            lerobot_dataset.save_episode()
            
            episode_start = episode_end
        
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
        description="Convert UMI zarr dataset to LeRobot format"
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
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        exit(1)
    
    print("="*60)
    print("UMI to LeRobot Dataset Conversion")
    print("="*60)
    print(f"\n📥 Input: {args.input}")
    print(f"📤 Output: {args.output}")
    print(f"🏷️  Repo ID: {args.repo_id}")
    print(f"⏱️  FPS: {args.fps}")
    print(f"📝 Task: {args.task}")
    print(f"☁️  Push to Hub: {args.push_to_hub}")
    print()
    
    # Convert dataset
    convert_to_lerobot(
        zarr_path=args.input,
        output_dir=args.output,
        repo_id=args.repo_id,
        fps=args.fps,
        default_task=args.task,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()

