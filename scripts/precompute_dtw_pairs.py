"""为论文 Ours/UOT 离线生成跨场景 DTW 轨迹配对。

每只手使用当前帧的相对起始 SE(3) 位姿和绝对夹爪状态。即使数据列已经标记为
``wrt_start``，这里仍会以当前 episode 的第一帧再次执行 ``T_0^-1 T_t``，从而保证
不同数据集的相对位姿定义一致。DTW 的 translation、rotation 和 gripper 三块采用
等权的、按块维度归一的平方 L2 距离。
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
from numba import njit


def _as_numpy_column(dataset, key: str) -> np.ndarray:
    """优先从 Hugging Face 表直接取列，避免为 DTW 解码所有视频。"""
    if not hasattr(dataset, "hf_dataset"):
        raise AttributeError("LeRobotDataset has no hf_dataset attribute")
    return np.asarray(dataset.hf_dataset[key])


def _current_frame(column: np.ndarray, *, expected_dim: int, key: str) -> np.ndarray:
    """Return only the current (last) state from an optional observation history."""
    column = np.asarray(column, dtype=np.float32)
    if column.ndim >= 3:
        column = column[:, -1]
    column = column.reshape(len(column), -1)
    if column.shape[-1] != expected_dim:
        raise ValueError(f"{key} must end in {expected_dim} values, got {column.shape}")
    return column


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), eps)


def _rot6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert the repository's two-row 6D rotation representation to SO(3)."""
    first = _normalize(rotation_6d[..., :3])
    second = rotation_6d[..., 3:]
    second = _normalize(second - np.sum(first * second, axis=-1, keepdims=True) * first)
    third = np.cross(first, second, axis=-1)
    return np.stack((first, second, third), axis=-2)


def _episode_relative_state(
    positions: list[np.ndarray],
    rotations: list[np.ndarray],
    grippers: list[np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build current-frame state and strictly rebase every arm to episode frame zero."""
    translation_parts = []
    rotation_parts = []
    gripper_parts = []
    for position, rotation, gripper in zip(positions, rotations, grippers, strict=True):
        episode_position = position[indices]
        episode_rotation = _rot6d_to_matrix(rotation[indices])
        initial_rotation_inverse = episode_rotation[0].T
        relative_rotation = np.einsum("ij,tjk->tik", initial_rotation_inverse, episode_rotation)
        relative_translation = np.einsum(
            "ij,tj->ti", initial_rotation_inverse, episode_position - episode_position[0]
        )
        translation_parts.append(relative_translation)
        rotation_parts.append(relative_rotation.reshape(len(indices), 9))
        gripper_parts.append(gripper[indices])
    return {
        "translation": np.concatenate(translation_parts, axis=-1).astype(np.float32),
        "rotation": np.concatenate(rotation_parts, axis=-1).astype(np.float32),
        "gripper": np.concatenate(gripper_parts, axis=-1).astype(np.float32),
    }


def _extract_episodes(
    repo_id: str,
    xyz_keys: list[str],
    rotation_keys: list[str],
    gripper_keys: list[str],
) -> list[dict[str, Any]]:
    dataset = lerobot_dataset.LeRobotDataset(repo_id)
    episode_indices = _as_numpy_column(dataset, "episode_index").reshape(-1)
    frame_indices = _as_numpy_column(dataset, "frame_index").reshape(-1)
    position_columns = [
        _current_frame(_as_numpy_column(dataset, key), expected_dim=3, key=key) for key in xyz_keys
    ]
    rotation_columns = [
        _current_frame(_as_numpy_column(dataset, key), expected_dim=6, key=key) for key in rotation_keys
    ]
    gripper_columns = [
        _current_frame(_as_numpy_column(dataset, key), expected_dim=1, key=key) for key in gripper_keys
    ]

    episodes = []
    for episode_id in np.unique(episode_indices):
        global_indices = np.flatnonzero(episode_indices == episode_id)
        order = np.argsort(frame_indices[global_indices])
        global_indices = global_indices[order]
        state = _episode_relative_state(
            position_columns, rotation_columns, gripper_columns, global_indices
        )
        episodes.append(
            {
                "episode_index": int(episode_id),
                "indices": global_indices.astype(int).tolist(),
                "dtw_state": {name: values.tolist() for name, values in state.items()},
            }
        )
    return episodes


@njit(cache=False)
def _dtw_core(
    source_translation: np.ndarray,
    source_rotation: np.ndarray,
    source_gripper: np.ndarray,
    target_translation: np.ndarray,
    target_rotation: np.ndarray,
    target_gripper: np.ndarray,
):
    """DTW with equally weighted dimension-normalized squared-L2 state blocks."""
    ns, nt = len(source_translation), len(target_translation)
    accumulated = np.full((ns + 1, nt + 1), np.inf, dtype=np.float64)
    accumulated[0, 0] = 0.0
    predecessor = np.zeros((ns + 1, nt + 1), dtype=np.int8)
    for source_t in range(1, ns + 1):
        for target_t in range(1, nt + 1):
            translation_distance = np.mean(
                (source_translation[source_t - 1] - target_translation[target_t - 1]) ** 2
            )
            rotation_distance = np.mean(
                (source_rotation[source_t - 1] - target_rotation[target_t - 1]) ** 2
            )
            gripper_distance = np.mean(
                (source_gripper[source_t - 1] - target_gripper[target_t - 1]) ** 2
            )
            squared_distance = (translation_distance + rotation_distance + gripper_distance) / 3.0
            diagonal = accumulated[source_t - 1, target_t - 1]
            vertical = accumulated[source_t - 1, target_t]
            horizontal = accumulated[source_t, target_t - 1]
            if diagonal <= vertical and diagonal <= horizontal:
                move, previous = 0, diagonal
            elif vertical <= horizontal:
                move, previous = 1, vertical
            else:
                move, previous = 2, horizontal
            accumulated[source_t, target_t] = squared_distance + previous
            predecessor[source_t, target_t] = move

    path_source = np.empty(ns + nt, dtype=np.int64)
    path_target = np.empty(ns + nt, dtype=np.int64)
    count = 0
    source_t, target_t = ns, nt
    while source_t > 0 and target_t > 0:
        path_source[count] = source_t - 1
        path_target[count] = target_t - 1
        count += 1
        move = predecessor[source_t, target_t]
        if move == 0:
            source_t -= 1
            target_t -= 1
        elif move == 1:
            source_t -= 1
        else:
            target_t -= 1
    return accumulated[ns, nt], path_source[:count][::-1], path_target[:count][::-1]


def _dtw(source: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> tuple[float, list[list[int]]]:
    """返回累计距离和 target 时刻到一个或多个 source 时刻的最优路径映射。"""
    distance, path_source, path_target = _dtw_core(
        source["translation"],
        source["rotation"],
        source["gripper"],
        target["translation"],
        target["rotation"],
        target["gripper"],
    )
    ns, nt = len(source["translation"]), len(target["translation"])
    mapping: list[list[int]] = [[] for _ in range(nt)]
    for source_t, target_t in zip(path_source.tolist(), path_target.tolist(), strict=True):
        mapping[target_t].append(source_t)
    for target_t, aligned in enumerate(mapping):
        if not aligned:
            fallback = min(round(target_t * max(ns - 1, 0) / max(nt - 1, 1)), ns - 1)
            aligned.append(int(fallback))
        aligned.sort()
    return float(distance), mapping


def _trajectory_descriptor(state: dict[str, np.ndarray], points: int = 16) -> np.ndarray:
    """仅用于超大轨迹库的候选预筛；不参与最终 DTW 距离或 UOT 损失。"""
    length = len(state["translation"])
    sample_times = np.linspace(0, length - 1, points)
    left = np.floor(sample_times).astype(int)
    right = np.minimum(left + 1, length - 1)
    ratio = (sample_times - left)[:, None]
    descriptor_parts = []
    for name in ("translation", "rotation", "gripper"):
        values = state[name]
        interpolated = (1.0 - ratio) * values[left] + ratio * values[right]
        descriptor_parts.append(interpolated.reshape(-1) / np.sqrt(values.shape[-1]))
    return np.concatenate(descriptor_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute cross-scene DTW pairs for UOT co-training")
    parser.add_argument("--source", action="append", required=True, help="辅助数据集路径；可重复传入")
    parser.add_argument("--target", required=True, help="目标数据集路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument("--xyz-key", action="append", dest="xyz_keys", help="相对起始 XYZ 列；每只手一次")
    parser.add_argument(
        "--rotation-key", action="append", dest="rotation_keys", help="相对起始 rot6d 列；每只手一次"
    )
    parser.add_argument(
        "--gripper-key", action="append", dest="gripper_keys", help="绝对夹爪状态列；每只手一次"
    )
    parser.add_argument("--top-k", type=int, default=0, help="每条目标轨迹仅保留最近 K 条辅助轨迹；0 表示全部")
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=0,
        help="每个 source 数据集先按轨迹描述子保留 K 条再做精确 DTW；0 表示论文式全量比较",
    )
    args = parser.parse_args()
    xyz_keys = args.xyz_keys or ["robot0_eef_pos_wrt_start"]
    rotation_keys = args.rotation_keys or ["robot0_eef_rot_axis_angle_wrt_start"]
    gripper_keys = args.gripper_keys or ["robot0_gripper_width"]
    if not (len(xyz_keys) == len(rotation_keys) == len(gripper_keys)):
        raise ValueError("Pass exactly one xyz, rotation and gripper key per arm")

    sources = [
        {
            "repo_id": repo_id,
            "episodes": _extract_episodes(repo_id, xyz_keys, rotation_keys, gripper_keys),
        }
        for repo_id in args.source
    ]
    target_episodes = _extract_episodes(args.target, xyz_keys, rotation_keys, gripper_keys)
    pairs = []
    for target_episode_index, target_episode in enumerate(target_episodes):
        target_state = {
            name: np.asarray(values, dtype=np.float32) for name, values in target_episode["dtw_state"].items()
        }
        target_descriptor = _trajectory_descriptor(target_state)
        candidates = []
        for source_repo_index, source in enumerate(sources):
            source_episode_indices = list(range(len(source["episodes"])))
            if args.candidate_pool > 0 and len(source_episode_indices) > args.candidate_pool:
                descriptor_distances = []
                for source_episode_index in source_episode_indices:
                    source_state = {
                        name: np.asarray(values, dtype=np.float32)
                        for name, values in source["episodes"][source_episode_index]["dtw_state"].items()
                    }
                    descriptor_distances.append(
                        np.linalg.norm(_trajectory_descriptor(source_state) - target_descriptor)
                    )
                source_episode_indices = np.argsort(descriptor_distances)[: args.candidate_pool].tolist()
            for source_episode_index in source_episode_indices:
                source_episode = source["episodes"][source_episode_index]
                source_state = {
                    name: np.asarray(values, dtype=np.float32)
                    for name, values in source_episode["dtw_state"].items()
                }
                distance, mapping = _dtw(source_state, target_state)
                candidates.append(
                    {
                        "source_repo_index": source_repo_index,
                        "source_episode_list_index": source_episode_index,
                        "normalized_distance": distance
                        / max(len(source_state["translation"]), len(target_state["translation"])),
                        "target_to_source": mapping,
                    }
                )
        candidates.sort(key=lambda item: item["normalized_distance"])
        if args.top_k > 0:
            candidates = candidates[: args.top_k]
        pairs.append({"target_episode_list_index": target_episode_index, "candidates": candidates})

    # 仅保留最终候选引用到的 source 轨迹，并重写索引，避免大数据集的 JSON 携带无关轨迹。
    used_episode_indices = [set() for _ in sources]
    for pair in pairs:
        for candidate in pair["candidates"]:
            used_episode_indices[candidate["source_repo_index"]].add(candidate["source_episode_list_index"])
    remaps = []
    for source_repo_index, source in enumerate(sources):
        selected = sorted(used_episode_indices[source_repo_index])
        remap = {old_index: new_index for new_index, old_index in enumerate(selected)}
        source["episodes"] = [source["episodes"][old_index] for old_index in selected]
        remaps.append(remap)
    for pair in pairs:
        for candidate in pair["candidates"]:
            source_repo_index = candidate["source_repo_index"]
            candidate["source_episode_list_index"] = remaps[source_repo_index][
                candidate["source_episode_list_index"]
            ]

    # dtw_state 仅用于配对计算，输出时删除以减小文件体积。
    for source in sources:
        for episode in source["episodes"]:
            episode.pop("dtw_state", None)
    for episode in target_episodes:
        episode.pop("dtw_state", None)

    output = {
        "schema_version": 2,
        "state_keys": {
            "translation": xyz_keys,
            "rotation": rotation_keys,
            "gripper": gripper_keys,
        },
        "state_definition": "current_frame_episode_rebased_se3_absolute_gripper",
        "state_block_dims": {
            "translation": 3 * len(xyz_keys),
            "rotation": 9 * len(rotation_keys),
            "gripper": len(gripper_keys),
        },
        "sources": sources,
        "target": {"repo_id": args.target, "episodes": target_episodes},
        "pairs": pairs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = (
        gzip.open(output_path, "wt", encoding="utf-8")
        if output_path.suffix == ".gz"
        else output_path.open("w", encoding="utf-8")
    )
    with output_file as file:
        json.dump(output, file, ensure_ascii=False)
    print(f"Wrote {len(target_episodes)} target episodes to {output_path}")


if __name__ == "__main__":
    main()
