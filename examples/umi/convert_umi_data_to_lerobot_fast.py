#!/usr/bin/env python3
"""
Convert UMI zarr dataset to LeRobot format with optimized multiprocessing.

This version provides maximum performance by:
- Pre-loading data into shared memory
- Parallel frame processing
- Optimized batch insertion
- Memory-efficient processing

Usage:
    python convert_umi_data_to_lerobot_fast.py \
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
from typing import Dict, List
from multiprocessing import Pool, cpu_count
import gc

import numpy as np
import zarr
from tqdm import tqdm

import os
import cv2

import numcodecs  # 你本来可能已经有
import imagecodecs  # 可选，不加也行


from openpi.utils.pose_utils import pose_to_mat, mat_to_pose10d
from openpi.utils.pose_repr_utils import convert_pose_mat_rep



DATASET_CONFIG = {
    'camera_obs_latency': 0.125,
    'robot_obs_latency': 0.0001,
    'gripper_obs_latency': 0.02,
    'dataset_frequeny': 0, #59.94
    'obs_down_sample_steps': 3, # 3, 1
    'low_dim_obs_horizon': 2,
    'img_obs_horizon': 2,
    'action_horizon': 16,
}




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


def preload_episode_data(root, episode_start: int, episode_end: int) -> Dict[str, np.ndarray]:
    """
    Pre-load all data for an episode into memory.
    This is faster than accessing zarr repeatedly.
    """
    data = root['data']
    
    # Load all data for this episode into memory
    episode_data = {
        'robot0_eef_pos': np.array(data['robot0_eef_pos'][episode_start:episode_end]),
        'robot0_eef_rot_axis_angle': np.array(data['robot0_eef_rot_axis_angle'][episode_start:episode_end]),
        'robot0_gripper_width': np.array(data['robot0_gripper_width'][episode_start:episode_end]),
        'camera0_rgb': np.array(data['camera0_rgb'][episode_start:episode_end]),
    }
    
    return episode_data


def generate_robot0_data_frame(robot0_eef_pos: np.ndarray, robot0_eef_rot_axis_angle: np.ndarray, robot0_gripper_width: np.ndarray, images: np.ndarray, robot0_demo_start_pose: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate robot0 data frame.
    """
    # convert pose to mat
    pose_mat = pose_to_mat(np.concatenate([
        robot0_eef_pos,
        robot0_eef_rot_axis_angle
    ], axis=-1))
    # get start pose
    start_pose = robot0_demo_start_pose
    start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
    start_pose_mat = pose_to_mat(start_pose)
    rel_obs_pose_mat = convert_pose_mat_rep(
        pose_mat,
        base_pose_mat=start_pose_mat,
        pose_rep='relative',
        backward=False)
    rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)

    action_mat = pose_to_mat(action[:, :6])
     # solve relative obs
    obs_pose_mat = convert_pose_mat_rep(
        pose_mat, 
        base_pose_mat=pose_mat[-1],
        pose_rep='relative',
        backward=False)
    action_pose_mat = convert_pose_mat_rep(
        action_mat, 
        base_pose_mat=pose_mat[-1],
        pose_rep='relative',
        backward=False)

    # convert pose to pos + rot6d representation
    obs_pose = mat_to_pose10d(obs_pose_mat)

    action_pose = mat_to_pose10d(action_pose_mat)
    action_gripper = action[:, 6]
    action = np.concatenate([action_pose, np.expand_dims(action_gripper, axis=0)], axis=-1)

    result = {
        "observation.robot0_eef_pos": obs_pose[:, :3],
        "observation.robot0_eef_rot_axis_angle": obs_pose[:, 3:],
        "observation.robot0_gripper_width": robot0_gripper_width,
        "observation.robot0_eef_rot_axis_angle_wrt_start": rel_obs_pose[:, 3:],
        "actions": action,
        "task": "pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box",
    }
    # 将图像序列拆分成独立的图像特征，以便使用 "image" dtype 的优化
    for i, img in enumerate(images):
        result[f"observation.left_wrist_0_rgb_{i}"] = img
    return result




def process_episode_fast(episode_idx: int, episode_data: Dict[str, np.ndarray], default_task: str, state_sequence_length: int = 2) -> List[Dict]:  # noqa: ARG001
    """
    Process a pre-loaded episode and return all frames.
    
    Args:
        episode_idx: Episode index (unused but kept for consistent interface)
        episode_data: Pre-loaded episode data
        default_task: Default task description
        state_sequence_length: Length of state sequence (default: 2)
    
    Returns:
        List of frame dictionaries
    """
    frames = []
    num_frames = len(episode_data['robot0_eef_pos'])

    robot0_demo_start_pose = np.concatenate([episode_data['robot0_eef_pos'][0].astype(np.float32), episode_data['robot0_eef_rot_axis_angle'][0].astype(np.float32)], axis=-1).astype(np.float32)
    
    for frame_idx in range(num_frames):
        # Construct action (next state for all but last frame)
        if frame_idx < num_frames - 1:
            action_7d = np.expand_dims(np.concatenate([
                episode_data['robot0_eef_pos'][frame_idx + 1],
                episode_data['robot0_eef_rot_axis_angle'][frame_idx + 1],
                episode_data['robot0_gripper_width'][frame_idx + 1],
            ]), axis=0).astype(np.float32)
        else:
            # Last frame: use current state as action
            action_7d = np.expand_dims(np.concatenate([
                episode_data['robot0_eef_pos'][frame_idx],
                episode_data['robot0_eef_rot_axis_angle'][frame_idx],
                episode_data['robot0_gripper_width'][frame_idx],
            ]), axis=0).astype(np.float32)

        # # 构造当前帧的state
        # current_state = np.concatenate([
        #     episode_data['robot0_eef_pos'][frame_idx],
        #     episode_data['robot0_eef_rot_axis_angle'][frame_idx],
        #     episode_data['robot0_gripper_width'][frame_idx],
        # ]).astype(np.float32)
        current_robot0_eef_pos = episode_data['robot0_eef_pos'][frame_idx].astype(np.float32)
        current_robot0_eef_rot_axis_angle = episode_data['robot0_eef_rot_axis_angle'][frame_idx].astype(np.float32)
        current_robot0_gripper_width = episode_data['robot0_gripper_width'][frame_idx].astype(np.float32).reshape(1)


        current_image = episode_data['camera0_rgb'][frame_idx]
        
        # state_sequence: 包含过去(state_sequence_length-1)帧的历史state + 当前帧state
        # 形状: (state_sequence_length, 7)
        # 例如: state_sequence_length=3 时，序列为 [t-2, t-1, t]
        # state_list = []
        robot0_eef_pos_list = []
        robot0_eef_rot_axis_angle_list = []
        robot0_gripper_width_list = []
        image_list = []
        
        # 获取历史帧 (state_sequence_length - 1 帧)
        for i in range(state_sequence_length - 1):
            # 计算要获取的帧索引（从最早的历史帧开始）
            past_idx = frame_idx - (state_sequence_length - 1 - i)
            
            if past_idx < 0:
                # 如果超出边界，使用第一帧的state填充
                # state = np.concatenate([
                #     episode_data['robot0_eef_pos'][0],
                #     episode_data['robot0_eef_rot_axis_angle'][0],
                #     episode_data['robot0_gripper_width'][0],
                # ]).astype(np.float32)
                robot0_eef_pos = episode_data['robot0_eef_pos'][0].astype(np.float32)
                robot0_eef_rot_axis_angle = episode_data['robot0_eef_rot_axis_angle'][0].astype(np.float32)
                robot0_gripper_width = episode_data['robot0_gripper_width'][0].astype(np.float32).reshape(1)
                image = episode_data['camera0_rgb'][0]
            else:
                # # 正常获取历史帧
                # state = np.concatenate([
                #     episode_data['robot0_eef_pos'][past_idx],
                #     episode_data['robot0_eef_rot_axis_angle'][past_idx],
                #     episode_data['robot0_gripper_width'][past_idx],
                # ]).astype(np.float32)
                robot0_eef_pos = episode_data['robot0_eef_pos'][past_idx].astype(np.float32)
                robot0_eef_rot_axis_angle = episode_data['robot0_eef_rot_axis_angle'][past_idx].astype(np.float32)
                robot0_gripper_width = episode_data['robot0_gripper_width'][past_idx].astype(np.float32).reshape(1)
                image = episode_data['camera0_rgb'][past_idx]
            
            robot0_eef_pos_list.append(robot0_eef_pos)
            robot0_eef_rot_axis_angle_list.append(robot0_eef_rot_axis_angle)
            robot0_gripper_width_list.append(robot0_gripper_width)
            # state_list.append(state)
            image_list.append(image)
        
        # 添加当前帧
        robot0_eef_pos_list.append(current_robot0_eef_pos)
        robot0_eef_rot_axis_angle_list.append(current_robot0_eef_rot_axis_angle)
        robot0_gripper_width_list.append(current_robot0_gripper_width)
        # state_list.append(current_state)
        image_list.append(current_image)
        # state_sequence = np.stack(state_list)
        # image_sequence = np.stack(image_list)

        frame_data = generate_robot0_data_frame(np.array(robot0_eef_pos_list), np.array(robot0_eef_rot_axis_angle_list), np.array(robot0_gripper_width_list), image_list, robot0_demo_start_pose, action_7d)

        # Create frame dictionary
        # frame_data = {
        #     "observation.robot0_eef_pos": episode_data['robot0_eef_pos'][frame_idx].astype(np.float32),
        #     "observation.robot0_eef_rot_axis_angle": episode_data['robot0_eef_rot_axis_angle'][frame_idx].astype(np.float32),
        #     "observation.robot0_gripper_width": episode_data['robot0_gripper_width'][frame_idx].astype(np.float32).reshape(1),
        #     "observation.camera0_rgb": episode_data['camera0_rgb'][frame_idx],
        #     "actions": action_7d,
        #     "task": default_task,
        #     # "state": current_state,
        #     "base_state": np.concatenate([
        #         episode_data['robot0_eef_pos'][0],
        #         episode_data['robot0_eef_rot_axis_angle'][0],
        #         episode_data['robot0_gripper_width'][0],
        #     ]).astype(np.float32),
        #     # "state_sequence": state_sequence,
        #     # "camera0_rgb_0": image_sequence[0],
        #     # "camera0_rgb_1": image_sequence[1],
        # }
        frames.append(frame_data)
    
    return frames


def process_batch_worker(args):
    """Worker function to process a batch of episodes."""
    episode_data_list, default_task, state_sequence_length = args
    results = []
    
    for episode_idx, episode_data in episode_data_list:
        frames = process_episode_fast(episode_idx, episode_data, default_task, state_sequence_length)
        results.append((episode_idx, frames))
    
    return results


def convert_to_lerobot(
    zarr_path: str,
    output_dir: str,
    repo_id: str,
    fps: int = 20,
    default_task: str = "umi manipulation task",
    push_to_hub: bool = False,
    num_workers: int = None,
    load_batch_size: int = 50,
    process_batch_size: int = 10,
    state_sequence_length: int = 2,
):
    """
    Convert UMI zarr dataset to LeRobot format with optimized multiprocessing.
    
    Args:
        zarr_path: Path to the zarr zip file
        output_dir: Output directory for LeRobot dataset
        repo_id: HuggingFace repository ID
        fps: Frames per second of the dataset
        default_task: Default task description
        push_to_hub: Whether to push to HuggingFace Hub
        num_workers: Number of worker processes
        load_batch_size: Number of episodes to load and process at once
        process_batch_size: Number of episodes to process per worker
    """
    
    if num_workers is None:
        num_workers = min(cpu_count(), 16)  # Cap at 16 to avoid memory issues
    
    print(f"🚀 Using {num_workers} worker processes for parallel conversion")
    print(f"📦 Load batch size: {load_batch_size} episodes")
    print(f"⚙️  Process batch size: {process_batch_size} episodes per worker")
    
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
        episode_ranges = list(zip(episode_starts, episode_ends))
        
        # Create LeRobot dataset
        print(f"\n🔨 Creating LeRobot dataset: {repo_id}")
        
        # 动态生成特征定义
        features = {
            "observation.robot0_eef_pos": {
                "dtype": "float32",
                "shape": (state_sequence_length, 3),
                "names": ["x", "y", "z"],
            },
            "observation.robot0_eef_rot_axis_angle": {
                "dtype": "float32",
                "shape": (state_sequence_length, 6),
                "names": ["rx", "ry", "rz"],
            },
            "observation.robot0_eef_rot_axis_angle_wrt_start": {
                "dtype": "float32",
                "shape": (state_sequence_length, 6),
                "names": ["qx", "qy", "qz", "qw"],
            },
            "observation.robot0_gripper_width": {
                "dtype": "float32",
                "shape": (state_sequence_length, 1),
                "names": ["gripper_width"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (1, 10),
                "names": ["actions"],
            }
        }
        # 动态添加图像特征 - 每个时间步一个独立的图像特征，使用 "image" dtype 获得压缩和并行写入优化
        for i in range(state_sequence_length):
            features[f"observation.left_wrist_0_rgb_{i}"] = {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            }
        
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=output_dir,
            robot_type="umi",
            features=features,
            image_writer_threads=num_workers * 2,  # More threads for image writing
            image_writer_processes=1,
        )
        
        print("\n🔄 Processing episodes with optimized pipeline...")
        
        # Process in large batches to balance memory and speed
        with Pool(processes=num_workers) as pool:
            for batch_start in range(0, num_episodes, load_batch_size):
                batch_end = min(batch_start + load_batch_size, num_episodes)
                
                print(f"\n📦 Loading episodes {batch_start} to {batch_end-1}...")
                
                # Pre-load episode data
                episode_data_list = []
                for episode_idx in tqdm(range(batch_start, batch_end), desc="Loading data", unit="episode"):
                    start, end = episode_ranges[episode_idx]
                    episode_data = preload_episode_data(root, int(start), int(end))
                    episode_data_list.append((episode_idx, episode_data))
                
                # Split into worker batches
                worker_batches = []
                for i in range(0, len(episode_data_list), process_batch_size):
                    batch = episode_data_list[i:i+process_batch_size]
                    worker_batches.append((batch, default_task, state_sequence_length))
                
                print(f"⚙️  Processing {len(episode_data_list)} episodes with {len(worker_batches)} worker batches...")
                
                # Process in parallel
                batch_results = list(tqdm(
                    pool.imap(process_batch_worker, worker_batches),
                    total=len(worker_batches),
                    desc="Processing batches",
                    unit="batch"
                ))
                
                # Flatten results and sort by episode index
                all_results = []
                for batch_result in batch_results:
                    all_results.extend(batch_result)
                all_results.sort(key=lambda x: x[0])  # Sort by episode_idx
                
                # Write to dataset
                print(f"💾 Writing {len(all_results)} episodes to dataset...")
                for episode_idx, frames in tqdm(all_results, desc="Writing episodes", unit="episode"):
                    for frame_data in frames:
                        lerobot_dataset.add_frame(frame_data)
                    lerobot_dataset.save_episode()
                
                # Clear memory
                del episode_data_list
                del batch_results
                del all_results
                gc.collect()
        
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
        print("\n🧹 Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert UMI zarr dataset to LeRobot format (optimized version)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/root/openpi-umi/data/dataset_ray_cyrus_Mason.zarr.zip",
        help="Path to input zarr zip file (e.g., dataset.zarr.zip)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/openpi-umi/data/umi_lerobot_dataset_v7.1",
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="mason/umi_lerobot_dataset_v7.1",
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
        default=50,
        help="Number of worker processes (default: min(cpu_count, 16))",
    )
    parser.add_argument(
        "--load-batch-size",
        type=int,
        default=100,
        help="Number of episodes to load at once (default: 50)",
    )
    parser.add_argument(
        "--process-batch-size",
        type=int,
        default=2,
        help="Number of episodes per worker batch (default: 10)",
    )
    parser.add_argument(
        "--state-sequence-length",
        type=int,
        default=2,
        help="Length of state sequence history (default: 2)",
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        exit(1)
    
    print("=" * 60)
    print("UMI to LeRobot Dataset Conversion (Optimized)")
    print("=" * 60)
    print(f"\n📥 Input: {args.input}")
    print(f"📤 Output: {args.output}")
    print(f"🏷️  Repo ID: {args.repo_id}")
    print(f"⏱️  FPS: {args.fps}")
    print(f"📝 Task: {args.task}")
    print(f"☁️  Push to Hub: {args.push_to_hub}")
    print(f"👷 Workers: {args.workers if args.workers else f'min({cpu_count()}, 16)'}")
    print(f"📊 State Sequence Length: {args.state_sequence_length}")
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
        load_batch_size=args.load_batch_size,
        process_batch_size=args.process_batch_size,
        state_sequence_length=args.state_sequence_length,
    )


if __name__ == "__main__":
    main()

