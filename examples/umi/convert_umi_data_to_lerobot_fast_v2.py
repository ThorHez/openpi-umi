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

from sampler_v2 import SequenceSampler
from dataset_config_loader import (
    load_dataset_config,
    build_features_from_config,
    print_config_summary,
    validate_config,
)

LOWDIM_KEYS = [
    'robot0_eef_rot_axis_angle', 
    'robot0_gripper_width', 
    'robot0_eef_pos',
    "robot0_demo_start_pose",
    "robot0_eef_pos_wrt1",
    "robot0_eef_rot_axis_angle_wrt1",


    'robot1_eef_rot_axis_angle', 
    'robot1_gripper_width', 
    'robot1_eef_pos',
    "robot1_demo_start_pose",
    "robot1_eef_pos_wrt0",
    "robot1_eef_rot_axis_angle_wrt0",
]


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


def preload_episode_data(root, episode_start: int, episode_end: int, dataset_config: Dict = None) -> Dict[str, np.ndarray]:
    """
    Pre-load all data for an episode into memory.
    This is faster than accessing zarr repeatedly.
    
    Args:
        root: zarr root
        episode_start: start index of episode
        episode_end: end index of episode
        dataset_config: dataset configuration dict, if provided will use load_keys from config
    """
    data = root['data']
    
    # 从配置中获取要加载的字段，如果没有配置则使用默认值
    if dataset_config is not None:
        load_keys = dataset_config.get('load_keys', [])
    else:
        # 默认加载的字段
        load_keys = [
            'robot0_eef_pos',
            'robot0_eef_rot_axis_angle',
            'robot0_gripper_width',
            'robot0_demo_start_pose',
            'camera0_rgb',
        ]
    
    # Load all data for this episode into memory
    episode_data = {}
    for key in load_keys:
        if key in data:
            episode_data[key] = np.array(data[key][episode_start:episode_end])
        else:
            print(f"⚠️  Warning: key '{key}' not found in dataset, skipping...")
    
    return episode_data


def get_num_enabled_robots(dataset_config: Dict) -> int:
    """从 dataset_config 中获取启用的机器人数量"""
    if dataset_config is None:
        return 1
    num_robot = 0
    for robot in dataset_config.get("robots", []):
        if robot.get("enabled", False):
            num_robot += 1
    return max(num_robot, 1)  # 至少返回1


def generate_robot_data_frame(
    data: Dict[str, np.ndarray],
    task: str,
    dataset_config: Dict = None) -> Dict[str, np.ndarray]:
    """
    Generate robot data frame.
    
    Args:
        data: 原始数据
        task: 任务描述
        dataset_config: 数据集配置，用于获取机器人数量
    """
    num_robot = get_num_enabled_robots(dataset_config)

    for robot_id in range(num_robot):
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            data[f"robot{robot_id}_eef_pos"],
            data[f"robot{robot_id}_eef_rot_axis_angle"]
        ], axis=-1))

        for other_robot_id in range(num_robot):
            if other_robot_id == robot_id:
                continue
            if not f'robot{robot_id}_eef_pos_wrt{other_robot_id}' in LOWDIM_KEYS:
                continue
            other_pose_mat = pose_to_mat(np.concatenate([
                data[f'robot{other_robot_id}_eef_pos'],
                data[f'robot{other_robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=other_pose_mat[-1],
                pose_rep='relative',
                backward=False)
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            data[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = np.array(rel_obs_pose[:,:3]).astype(np.float32)
            data[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = np.array(rel_obs_pose[:,3:]).astype(np.float32)


    for robot_id in range(num_robot):
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([data[f'robot{robot_id}_eef_pos'], data[f'robot{robot_id}_eef_rot_axis_angle']], axis=-1))

        # get start pose
        start_pose = data[f'robot{robot_id}_demo_start_pose']
        start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
        start_pose_mat = pose_to_mat(start_pose)
        rel_obs_pose_mat = convert_pose_mat_rep(
            pose_mat,
            base_pose_mat=start_pose_mat,
            pose_rep='relative',
            backward=False)
        rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
        data[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = np.array(rel_obs_pose[:,3:]).astype(np.float32)

    
    actions = list()
    for robot_id in range(num_robot):
         # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            data[f'robot{robot_id}_eef_pos'],
            data[f'robot{robot_id}_eef_rot_axis_angle']
        ], axis=-1))
        action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6])

        # solve relative obs
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep="relative",
            backward=False)
        action_pose_mat = convert_pose_mat_rep(
            action_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep="relative",
            backward=False)
    
        # convert pose to pos + rot6d representation
        obs_pose = mat_to_pose10d(obs_pose_mat)
        action_pose = mat_to_pose10d(action_pose_mat)
    
        action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
        actions.append(np.concatenate([action_pose, action_gripper], axis=-1).astype(np.float32))

        # generate data
        data[f'robot{robot_id}_eef_pos'] = np.array(obs_pose[:,:3]).astype(np.float32)
        data[f'robot{robot_id}_eef_rot_axis_angle'] = np.array(obs_pose[:,3:]).astype(np.float32)

    data['actions'] = np.concatenate(actions, axis=-1).astype(np.float32)
    data['task'] = task

    # 根据配置生成最终的特征数据
    if dataset_config is not None:
        output_data = {}
        img_obs_horizon = dataset_config.get("dataset", {}).get("img_obs_horizon", 2)
        
        # 处理普通特征 - 根据 features 配置重命名键
        features_config = dataset_config.get("features", {})
        for feature_name in features_config.keys():
            # 从特征名中提取原始键名 (去掉 "observation." 前缀)
            if feature_name.startswith("observation."):
                source_key = feature_name[len("observation."):]
            else:
                source_key = feature_name
            
            if source_key in data:
                output_data[feature_name] = data[source_key]
        
        # 处理图像特征
        images_config = dataset_config.get("images", {})
        for feature_name, image_def in images_config.items():
            source_key = image_def.get("source_key")
            if source_key is None or source_key not in data:
                continue
            
            per_timestep = image_def.get("per_timestep", False)
            source_data = data[source_key]  # shape: (img_obs_horizon, H, W, C)
            
            # 确保图像数据是 uint8 类型，值范围 [0, 255]
            if source_data.dtype != np.uint8:
                if source_data.max() <= 1.0:
                    # 如果是 [0, 1] 范围的 float，转换为 [0, 255]
                    source_data = (source_data * 255).astype(np.uint8)
                else:
                    # 如果是 [0, 255] 范围的 float，直接转换类型
                    source_data = source_data.astype(np.uint8)
            
            if per_timestep:
                # 将图像序列拆分为单帧，添加后缀 id
                for i in range(img_obs_horizon):
                    if i < len(source_data):
                        output_data[f"{feature_name}_{i}"] = source_data[i]
                    else:
                        # 如果数据不足，用最后一帧填充
                        output_data[f"{feature_name}_{i}"] = source_data[-1]
            else:
                # 不拆分，直接使用整个序列
                output_data[feature_name] = source_data
        
        # 添加 task
        output_data['task'] = task
        
        return output_data
    else:
        # 没有配置时，使用默认的键名映射
        output_data = {
            'observation.robot0_eef_pos': data['robot0_eef_pos'],
            'observation.robot0_eef_rot_axis_angle': data['robot0_eef_rot_axis_angle'],
            'observation.robot0_eef_rot_axis_angle_wrt_start': data['robot0_eef_rot_axis_angle_wrt_start'],
            'observation.robot0_gripper_width': data['robot0_gripper_width'],
            'actions': data['actions'],
            'task': task,
        }
        # 添加默认图像特征
        if 'camera0_rgb' in data:
            img_data = data['camera0_rgb']
            # 确保图像数据是 uint8 类型
            if img_data.dtype != np.uint8:
                if img_data.max() <= 1.0:
                    img_data = (img_data * 255).astype(np.uint8)
                else:
                    img_data = img_data.astype(np.uint8)
            for i in range(len(img_data)):
                output_data[f'observation.left_wrist_0_rgb_{i}'] = img_data[i]
        
        return output_data




def process_episode_fast(episode_idx: int, episode_data: Dict[str, np.ndarray], task: str, dataset_config: Dict = None) -> List[Dict]:  # noqa: ARG001
    """
    Process a pre-loaded episode and return all frames.
    
    Args:
        episode_idx: Episode index (unused but kept for consistent interface)
        episode_data: Pre-loaded episode data
        default_task: Default task description
        state_sequence_length: Length of state sequence (default: 2)
        dataset_config: Dataset configuration dictionary
    
    Returns:
        List of frame dictionaries
    """    
    frames = []
    num_frames = len(episode_data['robot0_eef_pos'])
    episode_ends = [num_frames]
    sequence_sampler = SequenceSampler(
        replay_buffer=episode_data,
        episode_ends=episode_ends,
        dataset_config=dataset_config
    )
    for frame_idx in range(num_frames):
        sampled_data = sequence_sampler.sample_sequence(frame_idx)
        frame_data = generate_robot_data_frame(sampled_data, task, dataset_config)
        frames.append(frame_data)
    
    return frames


def process_batch_worker(args):
    """Worker function to process a batch of episodes."""
    episode_data_list, task, dataset_config = args
    results = []
    
    for episode_idx, episode_data in episode_data_list:
        frames = process_episode_fast(episode_idx, episode_data, task, dataset_config)
        results.append((episode_idx, frames))
    
    return results


def convert_to_lerobot(
    zarr_path: str,
    output_dir: str,
    repo_id: str,
    task: str,
    fps: int = 20,
    push_to_hub: bool = False,
    num_workers: int = None,
    load_batch_size: int = 50,
    process_batch_size: int = 10,
    state_sequence_length: int = 2,
    max_episodes: int = None,
    config_path: str = None,
):
    """
    Convert UMI zarr dataset to LeRobot format with optimized multiprocessing.
    
    Args:
        zarr_path: Path to the zarr zip file
        output_dir: Output directory for LeRobot dataset
        repo_id: HuggingFace repository ID
        fps: Frames per second of the dataset
        task: Default task description
        push_to_hub: Whether to push to HuggingFace Hub
        num_workers: Number of worker processes
        load_batch_size: Number of episodes to load and process at once
        process_batch_size: Number of episodes to process per worker
        state_sequence_length: Length of state sequence (default: 2)
        max_episodes: Maximum episodes to process (for debugging)
        config_path: Path to dataset config YAML file for feature definitions
    """
    
    # 加载配置文件（如果提供）
    dataset_config = None
    if config_path:
        print(f"\n📄 Loading dataset config from: {config_path}")
        dataset_config = load_dataset_config(config_path)
        
        # 验证配置
        warnings = validate_config(dataset_config)
        if warnings:
            print("⚠️  配置警告:")
            for w in warnings:
                print(f"    - {w}")
    
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
        total_episodes_in_dataset = len(episode_ends)
        total_steps_in_dataset = episode_ends[-1]
        
        # Limit episodes if max_episodes is set (for debugging)
        if max_episodes is not None and max_episodes < total_episodes_in_dataset:
            num_episodes = max_episodes
            total_steps = episode_ends[max_episodes - 1]
            print(f"\n⚠️  DEBUG MODE: Only processing first {max_episodes} episodes")
        else:
            num_episodes = total_episodes_in_dataset
            total_steps = total_steps_in_dataset
        
        print("\n📊 Dataset info:")
        print(f"  Total episodes in dataset: {total_episodes_in_dataset}")
        print(f"  Episodes to process: {num_episodes}")
        print(f"  Total steps to process: {total_steps}")
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
        
        # 根据配置文件或默认值生成特征定义
        if dataset_config:
            # 使用配置文件生成特征
            features = build_features_from_config(
                dataset_config,
            )
            print_config_summary(dataset_config, features)
        else:
            # 使用默认特征定义 - 基础特征（必选）
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
                    "shape": (16, 10),
                    "names": ["actions"],
                },
            }
            
            # 动态添加图像特征 - 每个时间步一个独立的图像特征
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
                print(f"batch_start: {batch_start}, batch_end: {batch_end}")
                
                print(f"\n📦 Loading episodes {batch_start} to {batch_end-1}...")
                
                # Pre-load episode data
                episode_data_list = []
                for episode_idx in tqdm(range(batch_start, batch_end), desc="Loading data", unit="episode"):
                    start, end = episode_ranges[episode_idx]
                    episode_data = preload_episode_data(root, int(start), int(end), dataset_config)
                    episode_data_list.append((episode_idx, episode_data))
                
                # Split into worker batches
                worker_batches = []
                for i in range(0, len(episode_data_list), process_batch_size):
                    batch = episode_data_list[i:i+process_batch_size]
                    worker_batches.append((batch, task, dataset_config))
                
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
        default="/root/openpi-umi/data/merge_cube_and_box_zarr_20251213.zarr.zip",
        help="Path to input zarr zip file (e.g., dataset.zarr.zip)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/openpi-umi/data/umi_lerobot_dataset_v7.2_ray_cyrus_mason_20251213",
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="mason/umi_lerobot_dataset_v7.2_ray_cyrus_mason_20251213",
        help="HuggingFace repository ID (e.g., 'your_username/umi_dataset')",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second of the dataset (default: 30)",
    )
    parser.add_argument(
        "--task",
        type=str,
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
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process (for debugging). If not set, process all episodes.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to dataset config YAML file for feature definitions",
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        exit(1)
    
    # Check if config file exists (if provided)
    if args.config and not Path(args.config).exists():
        print(f"❌ Error: Config file not found: {args.config}")
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
    print(f"🔢 Max Episodes: {args.max_episodes if args.max_episodes else 'All'}")
    print(f"📄 Config File: {args.config if args.config else 'None (using defaults)'}")
    print()
    
    # Convert dataset
    convert_to_lerobot(
        zarr_path=args.input,
        output_dir=args.output,
        repo_id=args.repo_id,
        fps=args.fps,
        task=args.task,
        push_to_hub=args.push_to_hub,
        num_workers=args.workers,
        load_batch_size=args.load_batch_size,
        process_batch_size=args.process_batch_size,
        state_sequence_length=args.state_sequence_length,
        max_episodes=args.max_episodes,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()