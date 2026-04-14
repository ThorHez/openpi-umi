#!/usr/bin/env python3
"""
Convert UMI zarr dataset to LeRobot format - FAST v3.

v3 optimizations over v2:
- Workers load from shared zarr path (no large pickle of episode data)
- Vectorized generate_robot_data_frame_batch: process all frames of an episode
  in one batch (NumPy vectorized pose math) instead of per-frame Python loop
- Parallel episode loading inside each worker (optional, via zarr multi-thread read)

Usage:
    python convert_umi_data_to_lerobot_fast_v3.py \\
        --input dataset.zarr.zip \\
        --output ./umi_lerobot_dataset \\
        --repo-id your_hf_username/umi_dataset \\
        --task 'fold the clothes' \\
        --config config.yaml \\
        --workers 8
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import threading

import numpy as np
import datasets
import zarr
from tqdm import tqdm

from openpi.utils.pose_utils import pose_to_mat, mat_to_pose10d
from openpi.utils.pose_repr_utils import convert_pose_mat_rep


def batch_pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """
    pose with shape (..., 6). scipy Rotation.from_rotvec only accepts (N, 3),
    so flatten to (N, 6), pose_to_mat -> (N, 4, 4), reshape back to (..., 4, 4).
    """
    orig_shape = pose.shape[:-1]
    n = int(np.prod(orig_shape))
    flat = pose.reshape(n, 6)
    mat = pose_to_mat(flat)
    return mat.reshape(orig_shape + (4, 4))

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

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.image_writer import write_image
    from lerobot.common.datasets.utils import (
        embed_images,
        get_hf_features_from_features,
        load_info,
        write_episode as lerobot_write_episode,
        write_episode_stats as lerobot_write_episode_stats,
        write_info,
        DEFAULT_IMAGE_PATH,
        DEFAULT_PARQUET_PATH,
    )
    from lerobot.common.datasets.compute_stats import compute_episode_stats
except ImportError:
    print("Error: Required packages not installed. Please install them first:")
    print("  pip install lerobot datasets zarr numpy tqdm")
    exit(1)


def load_zarr_dataset(zarr_path: str) -> Tuple[zarr.Group, str, bool]:
    """
    Open UMI zarr dataset from a .zip file or an uncompressed .zarr directory.
    Returns (root, path_for_workers, is_temp_dir).
    - If zarr_path is a directory: open it directly, path_for_workers=zarr_path, is_temp_dir=False.
    - If zarr_path is a zip: extract to temp, path_for_workers=temp_dir, is_temp_dir=True (caller should rmtree).
    """
    zarr_path = str(zarr_path)
    path = Path(zarr_path)
    print(f"Loading zarr dataset from: {zarr_path}")
    if path.is_dir():
        root = zarr.open(zarr_path, mode="r")
        print("Using uncompressed zarr directory.")
        return root, zarr_path, False
    if not path.exists():
        raise FileNotFoundError(f"Zarr path not found: {zarr_path}")
    temp_dir = tempfile.mkdtemp(dir="/root")
    print(f"Extracting zip to temporary directory: {temp_dir}")
    with zipfile.ZipFile(zarr_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    root = zarr.open(temp_dir, mode="r")
    return root, temp_dir, True


def preload_episode_data(root, episode_start: int, episode_end: int, dataset_config: Dict = None) -> Dict[str, np.ndarray]:
    """Pre-load all data for an episode into memory."""
    data = root['data']
    if dataset_config is not None:
        load_keys = dataset_config.get('load_keys', [])
    else:
        load_keys = [
            'robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width',
            'robot0_demo_start_pose', 'camera0_rgb',
        ]
    episode_data = {}
    for key in load_keys:
        if key in data:
            episode_data[key] = np.array(data[key][episode_start:episode_end])
        else:
            print(f"Warning: key '{key}' not found in dataset, skipping...")
    return episode_data


def get_num_enabled_robots(dataset_config: Dict) -> int:
    if dataset_config is None:
        return 1
    num_robot = sum(1 for r in dataset_config.get("robots", []) if r.get("enabled", False))
    return max(num_robot, 1)


def _build_output_frame_dict_batch(
    data_batch: Dict[str, np.ndarray],
    task: str,
    dataset_config: Dict,
    num_robot: int,
) -> List[Dict[str, np.ndarray]]:
    """Build list of per-frame output dicts from batched data. data_batch has arrays with shape (T, ...)."""
    T = data_batch['robot0_eef_pos'].shape[0]
    img_obs_horizon = (dataset_config or {}).get("dataset", {}).get("img_obs_horizon", 2)
    features_config = {
        **(dataset_config or {}).get("features", {}),
        **(dataset_config or {}).get("single_frame_features", {}),
    }
    images_config = (dataset_config or {}).get("images", {})

    frames = []
    for t in range(T):
        output_data = {}
        if dataset_config is not None:
            for feature_name in features_config.keys():
                source_key = feature_name[len("observation."):] if feature_name.startswith("observation.") else feature_name
                if source_key in data_batch:
                    arr = data_batch[source_key]
                    output_data[feature_name] = arr[t] if arr.ndim > 1 else arr
            for feature_name, image_def in images_config.items():
                source_key = image_def.get("source_key")
                if source_key is None or source_key not in data_batch:
                    continue
                source_data = data_batch[source_key][t]
                feat_dtype = image_def.get("dtype", "image")
                if feat_dtype == "image" and source_data.dtype != np.uint8:
                    source_data = (source_data * 255).astype(np.uint8) if source_data.max() <= 1.0 else source_data.astype(np.uint8)
                expected_ndim = len(image_def.get("shape", []))
                per_timestep = image_def.get("per_timestep", False)
                if per_timestep:
                    for i in range(img_obs_horizon):
                        frame = source_data[i] if i < len(source_data) else source_data[-1]
                        if expected_ndim > 0 and frame.ndim < expected_ndim:
                            frame = np.expand_dims(frame, axis=-1)
                        output_data[f"{feature_name}_{i}"] = frame
                else:
                    if expected_ndim > 0 and source_data.ndim < expected_ndim:
                        source_data = np.expand_dims(source_data, axis=-1)
                    output_data[feature_name] = source_data
        else:
            output_data = {
                'observation.robot0_eef_pos': data_batch['robot0_eef_pos'][t],
                'observation.robot0_eef_rot_axis_angle': data_batch['robot0_eef_rot_axis_angle'][t],
                'observation.robot0_eef_rot_axis_angle_wrt_start': data_batch['robot0_eef_rot_axis_angle_wrt_start'][t],
                'observation.robot0_gripper_width': data_batch['robot0_gripper_width'][t],
                'actions': data_batch['actions'][t],
                'task': task,
            }
            if 'camera0_rgb' in data_batch:
                img_data = data_batch['camera0_rgb'][t]
                if img_data.dtype != np.uint8:
                    img_data = (img_data * 255).astype(np.uint8) if img_data.max() <= 1.0 else img_data.astype(np.uint8)
                for i in range(len(img_data)):
                    output_data[f'observation.left_wrist_0_rgb_{i}'] = img_data[i]
        output_data['task'] = task
        frames.append(output_data)
    return frames


def generate_robot_data_frame_batch(
    sampled_data_list: List[Dict[str, np.ndarray]],
    task: str,
    dataset_config: Dict = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Vectorized: compute robot data for all frames in one batch.
    Each element of sampled_data_list is a dict with horizon-shaped arrays (e.g. (2, 3), (2, 6), (16, 7)).
    """
    if not sampled_data_list:
        return []
    num_robot = get_num_enabled_robots(dataset_config)
    T = len(sampled_data_list)
    sample0 = sampled_data_list[0]
    # Stack into (T, horizon, ...) or (T, ...)
    def stack_key(key):
        arrs = [s[key] for s in sampled_data_list]
        return np.stack(arrs, axis=0)

    # Build batched data dict with (T, ...) or (T, H, ...)
    batch = {}
    for key in sample0.keys():
        batch[key] = stack_key(key)

    # Current timestep index (last in horizon)
    last_idx = -1

    # robot_i wrt robot_j
    for robot_id in range(num_robot):
        pose = np.concatenate([
            batch[f"robot{robot_id}_eef_pos"],
            batch[f"robot{robot_id}_eef_rot_axis_angle"]
        ], axis=-1)
        pose_mat = batch_pose_to_mat(pose)
        for other_robot_id in range(num_robot):
            if other_robot_id == robot_id:
                continue
            if f'robot{robot_id}_eef_pos_wrt{other_robot_id}' not in LOWDIM_KEYS:
                continue
            other_pose = np.concatenate([
                batch[f'robot{other_robot_id}_eef_pos'],
                batch[f'robot{other_robot_id}_eef_rot_axis_angle']
            ], axis=-1)
            other_pose_mat = batch_pose_to_mat(other_pose)
            base = other_pose_mat[:, last_idx, :, :]
            rel_mat = np.linalg.inv(base)[:, np.newaxis, :, :] @ pose_mat
            rel_pose = mat_to_pose10d(rel_mat)
            batch[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_pose[:, :, :3].astype(np.float32)
            batch[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_pose[:, :, 3:].astype(np.float32)

    # wrt_start
    rng = np.random.default_rng()
    for robot_id in range(num_robot):
        pose = np.concatenate([
            batch[f'robot{robot_id}_eef_pos'],
            batch[f'robot{robot_id}_eef_rot_axis_angle']
        ], axis=-1)
        pose_mat = batch_pose_to_mat(pose)
        start_pose = batch[f'robot{robot_id}_demo_start_pose'][:, last_idx, :6]
        start_pose = start_pose + rng.normal(scale=0.05, size=start_pose.shape)
        start_pose_mat = pose_to_mat(start_pose)
        base = start_pose_mat
        rel_mat = np.linalg.inv(base)[:, np.newaxis, :, :] @ pose_mat
        rel_pose = mat_to_pose10d(rel_mat)
        batch[f'robot{robot_id}_eef_pos_wrt_start'] = rel_pose[:, :, :3].astype(np.float32)
        batch[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_pose[:, :, 3:].astype(np.float32)

    # actions and obs relative to current
    actions_list = []
    for robot_id in range(num_robot):
        pose = np.concatenate([
            batch[f'robot{robot_id}_eef_pos'],
            batch[f'robot{robot_id}_eef_rot_axis_angle']
        ], axis=-1)
        pose_mat = batch_pose_to_mat(pose)
        base = pose_mat[:, last_idx, :, :]
        action_raw = batch['action'][..., 7 * robot_id : 7 * robot_id + 6]
        action_mat = batch_pose_to_mat(action_raw)
        obs_pose_mat = np.linalg.inv(base)[:, np.newaxis, :, :] @ pose_mat
        action_pose_mat = np.linalg.inv(base)[:, np.newaxis, :, :] @ action_mat
        obs_pose = mat_to_pose10d(obs_pose_mat)
        action_pose = mat_to_pose10d(action_pose_mat)
        action_gripper = batch['action'][..., 7 * robot_id + 6 : 7 * robot_id + 7]
        action_per_robot = np.concatenate([action_pose, action_gripper], axis=-1).astype(np.float32)
        actions_list.append(action_per_robot)
        batch[f'robot{robot_id}_eef_pos'] = obs_pose[:, :, :3].astype(np.float32)
        batch[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:, :, 3:].astype(np.float32)

    batch['actions'] = np.concatenate(actions_list, axis=-1).astype(np.float32)
    batch['task'] = task

    return _build_output_frame_dict_batch(batch, task, dataset_config, num_robot)


def _write_one_episode_parquet(
    root: Path,
    features: Dict,
    hf_features: Any,
    task_to_index: Dict[str, int],
    fps: int,
    ep_idx: int,
    frames: List[Dict],
    start_index: int,
    meta_lock: threading.Lock,
) -> Tuple[int, int, List[str], Dict]:
    """Write one episode: images + parquet. Appends to meta under lock. Returns (ep_idx, length, tasks, stats)."""
    root = Path(root)
    length = len(frames)
    tasks = [f.get("task", "") for f in frames]
    episode_tasks = list(set(tasks))

    # Build episode buffer: same keys as hf_features
    episode_buffer = {}
    episode_buffer["frame_index"] = np.arange(length, dtype=np.int64)
    episode_buffer["timestamp"] = (np.arange(length, dtype=np.float32) / fps)
    episode_buffer["index"] = np.arange(start_index, start_index + length, dtype=np.int64)
    episode_buffer["episode_index"] = np.full((length,), ep_idx, dtype=np.int64)
    episode_buffer["task_index"] = np.array([task_to_index.get(t, 0) for t in tasks], dtype=np.int64)

    for key in hf_features:
        if key in episode_buffer:
            continue
        ft = features.get(key, {})
        if ft.get("dtype") in ["image", "video"]:
            # Write to images/ so embed_images() can read paths; images/ is removed at end to avoid 2x storage.
            paths = []
            for i, frame_data in enumerate(frames):
                if key not in frame_data:
                    continue
                img = frame_data[key]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                rel_path = DEFAULT_IMAGE_PATH.format(
                    image_key=key, episode_index=ep_idx, frame_index=i
                )
                fpath = root / rel_path
                fpath.parent.mkdir(parents=True, exist_ok=True)
                write_image(img, fpath)
                paths.append(str(fpath))
            episode_buffer[key] = paths
        elif ft.get("dtype") not in ("string",):
            arrs = [np.asarray(f[key]) for f in frames if key in f]
            if arrs:
                stacked = np.stack(arrs)
                expected_dtype = ft.get("dtype")
                if stacked.dtype == np.bool_ and expected_dtype and expected_dtype != "bool":
                    stacked = stacked.astype(np.dtype(expected_dtype))
                expected_shape = tuple(ft.get("shape", ()))
                if expected_shape and stacked.shape[1:] != expected_shape:
                    try:
                        stacked = stacked.reshape((-1,) + expected_shape)
                    except ValueError:
                        pass
                episode_buffer[key] = stacked

    episode_dict = {k: episode_buffer[k] for k in hf_features if k in episode_buffer}
    ep_hf_features = datasets.Features({k: v for k, v in hf_features.items() if k in episode_dict})
    ep_dataset = datasets.Dataset.from_dict(
        episode_dict, features=ep_hf_features, split="train"
    )
    ep_dataset = embed_images(ep_dataset)
    chunk = ep_idx // 1000
    parquet_path = root / DEFAULT_PARQUET_PATH.format(
        episode_chunk=chunk, episode_index=ep_idx
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(parquet_path)

    ep_stats = compute_episode_stats(episode_buffer, features)
    episode_meta = {
        "episode_index": ep_idx,
        "tasks": episode_tasks,
        "length": length,
    }
    with meta_lock:
        lerobot_write_episode(episode_meta, root)
        lerobot_write_episode_stats(ep_idx, ep_stats, root)
    return (ep_idx, length, episode_tasks, ep_stats)


def process_episode_fast_vectorized(
    episode_idx: int,
    episode_data: Dict[str, np.ndarray],
    task: str,
    dataset_config: Dict = None,
) -> List[Dict]:
    """Process one episode: sample all frames then run vectorized batch generation."""
    num_frames = len(episode_data['robot0_eef_pos'])
    episode_ends = [num_frames]
    sequence_sampler = SequenceSampler(
        replay_buffer=episode_data,
        episode_ends=episode_ends,
        dataset_config=dataset_config,
    )
    sampled_list = []
    for frame_idx in range(num_frames):
        sampled_data = sequence_sampler.sample_sequence(frame_idx)
        sampled_list.append(sampled_data)
    return generate_robot_data_frame_batch(sampled_list, task, dataset_config)


def worker_process_episodes(args: Tuple) -> List[Tuple[int, List[Dict]]]:
    """
    Worker: load zarr from temp_dir, process assigned episodes, return (episode_idx, frames) list.
    Avoids serializing large episode data across processes.
    """
    temp_dir, config_path, episode_indices, episode_ranges, task = args
    dataset_config = load_dataset_config(config_path) if config_path else None
    root = zarr.open(temp_dir, mode='r')
    results = []
    for ep_idx in episode_indices:
        start, end = episode_ranges[ep_idx]
        episode_data = preload_episode_data(root, int(start), int(end), dataset_config)
        frames = process_episode_fast_vectorized(ep_idx, episode_data, task, dataset_config)
        results.append((ep_idx, frames))
    return results


def convert_to_lerobot(
    zarr_path: str,
    output_dir: str,
    repo_id: str,
    task: str,
    fps: int = 20,
    push_to_hub: bool = False,
    num_workers: int = None,
    episodes_per_worker: int = None,
    write_workers: int = None,
    state_sequence_length: int = 2,
    max_episodes: int = None,
    config_path: str = None,
):
    """
    Convert UMI zarr dataset to LeRobot format (v3: worker-load + vectorized batch).
    """
    dataset_config = None
    if config_path:
        print(f"\nLoading dataset config from: {config_path}")
        dataset_config = load_dataset_config(config_path)
        warnings = validate_config(dataset_config)
        if warnings:
            print("Config warnings:")
            for w in warnings:
                print(f"  - {w}")

    if num_workers is None:
        num_workers = min(cpu_count(), 16)
    print(f"Using {num_workers} worker processes (each loads from zarr, no large data transfer)")
    root, zarr_root_path, is_temp_zarr = load_zarr_dataset(zarr_path)
    try:
        data = root['data']
        meta = root['meta']
        episode_ends_arr = np.array(meta['episode_ends'][:])
        total_episodes = len(episode_ends_arr)
        total_steps = int(episode_ends_arr[-1])
        if max_episodes is not None and max_episodes < total_episodes:
            num_episodes = max_episodes
            total_steps = int(episode_ends_arr[max_episodes - 1])
            print(f"DEBUG: Only processing first {max_episodes} episodes")
        else:
            num_episodes = total_episodes
        episode_starts = [0] + list(episode_ends_arr[:-1])
        episode_ranges = [(int(s), int(e)) for s, e in zip(episode_starts, episode_ends_arr)]
        episode_ranges = episode_ranges[:num_episodes]

        print("\nDataset info:")
        print(f"  Episodes to process: {num_episodes}, total steps: {total_steps}")

        if dataset_config:
            features = build_features_from_config(dataset_config)
            print_config_summary(dataset_config, features)
        else:
            features = {
                "observation.robot0_eef_pos": {"dtype": "float32", "shape": (state_sequence_length, 3), "names": ["x", "y", "z"]},
                "observation.robot0_eef_rot_axis_angle": {"dtype": "float32", "shape": (state_sequence_length, 6), "names": ["rx", "ry", "rz"]},
                "observation.robot0_eef_rot_axis_angle_wrt_start": {"dtype": "float32", "shape": (state_sequence_length, 6), "names": ["qx", "qy", "qz", "qw"]},
                "observation.robot0_gripper_width": {"dtype": "float32", "shape": (state_sequence_length, 1), "names": ["gripper_width"]},
                "actions": {"dtype": "float32", "shape": (16, 10), "names": ["actions"]},
            }
            for i in range(state_sequence_length):
                features[f"observation.left_wrist_0_rgb_{i}"] = {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]}

        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=output_dir,
            robot_type="umi",
            features=features,
            image_writer_threads=num_workers * 2,
            image_writer_processes=1,
        )

        # Distribute episodes across workers (each worker loads its own data from zarr)
        # Use smaller chunks so progress bar updates more often (~every 30-60s instead of 3-4 min)
        if episodes_per_worker is None:
            # Cap at ~6 episodes per chunk so we have many chunks and visible progress
            raw = max(1, (num_episodes + num_workers - 1) // num_workers)
            episodes_per_worker = min(6, raw)
        chunks = []
        for i in range(0, num_episodes, episodes_per_worker):
            chunk = list(range(i, min(i + episodes_per_worker, num_episodes)))
            if chunk:
                chunks.append(chunk)
        worker_args = [
            (zarr_root_path, config_path, chunk, episode_ranges, task)
            for chunk in chunks
        ]

        print(f"\nProcessing episodes (workers load from zarr): {len(chunks)} chunks, ~{episodes_per_worker} ep/chunk ...")
        with Pool(processes=num_workers) as pool:
            all_results = list(tqdm(
                pool.imap(worker_process_episodes, worker_args),
                total=len(worker_args),
                desc="Workers",
                unit="chunk",
            ))

        flat = []
        for chunk_result in all_results:
            flat.extend(chunk_result)
        flat.sort(key=lambda x: x[0])

        # Parallel parquet + image write
        write_workers = write_workers or 8
        print(f"Writing episodes to dataset (parallel, {write_workers} writers)...")
        root_path = Path(output_dir)
        meta = lerobot_dataset.meta
        # Register all tasks first
        all_tasks = set()
        for _ep_idx, frames in flat:
            for fd in frames:
                t = fd.get("task", "")
                if t:
                    all_tasks.add(t)
        for task in sorted(all_tasks):
            if meta.get_task_index(task) is None:
                meta.add_task(task)
        task_to_index = {v: k for k, v in meta.task_to_task_index.items()}
        # Precompute start index per episode
        start_indices = []
        idx = 0
        for _ep_idx, frames in flat:
            start_indices.append(idx)
            idx += len(frames)
        hf_features = get_hf_features_from_features(meta.features)
        meta_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=write_workers) as pool:
            futures = []
            for i, (ep_idx, frames) in enumerate(flat):
                start_index = start_indices[i]
                fut = pool.submit(
                    _write_one_episode_parquet,
                    root_path,
                    meta.features,
                    hf_features,
                    task_to_index,
                    fps,
                    ep_idx,
                    frames,
                    start_index,
                    meta_lock,
                )
                futures.append((ep_idx, fut))
            for _ep_idx, fut in tqdm(futures, desc="Writing", unit="episode"):
                fut.result()
        # Remove images/ to avoid duplicate storage: embed_images() already embedded
        # image bytes into parquet, so images/ is redundant and doubles disk usage.
        images_dir = root_path / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
            print("Removed images/ (images are stored in parquet only, no duplicate).")
        # Update meta info (total_episodes, total_frames, etc.)
        info = load_info(root_path)
        info["total_episodes"] = num_episodes
        info["total_frames"] = total_steps
        chunks_size = info.get("chunks_size", 1000)
        info["total_chunks"] = max(1, (num_episodes + chunks_size - 1) // chunks_size)
        info["splits"] = {"train": f"0:{num_episodes}"}
        write_info(info, root_path)
        # Encode images to videos if required
        if len(meta.video_keys) > 0:
            print("Encoding videos...")
            dataset_for_encode = LeRobotDataset(repo_id, root=output_dir)
            dataset_for_encode.encode_videos()

        print("\nConversion complete.")
        print(f"  Output: {output_dir}, episodes: {num_episodes}, frames: {total_steps}")
        if push_to_hub:
            print(f"\nPushing to Hub: {repo_id}")
            try:
                ds = LeRobotDataset(repo_id, root=output_dir)
                ds.push_to_hub(repo_id=repo_id)
                print(f"Pushed to https://huggingface.co/datasets/{repo_id}")
            except Exception as e:
                print(f"Push failed: {e}")
        else:
            print("\nTo push later: LeRobotDataset(..., root=...).push_to_hub(repo_id)")
    finally:
        if is_temp_zarr:
            print("Cleaning up temporary files...")
            shutil.rmtree(zarr_root_path, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Convert UMI zarr to LeRobot (fast v3)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input zarr: .zip file or uncompressed .zarr directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--episodes-per-worker", type=int, default=None, help="Episodes per worker chunk (default: auto)")
    parser.add_argument("--write-workers", type=int, default=8, help="Number of parallel writers for parquet+images (default: 8)")
    parser.add_argument("--state-sequence-length", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    if args.config and not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        exit(1)

    print("UMI to LeRobot Conversion (Fast v3)")
    print(f"  Input: {args.input}\n  Output: {args.output}\n  Task: {args.task}")
    convert_to_lerobot(
        zarr_path=args.input,
        output_dir=args.output,
        repo_id=args.repo_id,
        fps=args.fps,
        task=args.task,
        push_to_hub=args.push_to_hub,
        num_workers=args.workers,
        episodes_per_worker=args.episodes_per_worker,
        write_workers=args.write_workers,
        state_sequence_length=args.state_sequence_length,
        max_episodes=args.max_episodes,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
