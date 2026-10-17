"""为论文 Ours/UOT 离线生成跨场景 DTW 轨迹配对。

逻辑：使用每帧最新的末端位置做 DTW，保存完整的 source/target 轨迹索引、用于
UOT 的两帧位置标签，以及目标时刻到辅助时刻的映射。该脚本只读取数据，不改原数据集。
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


def _extract_episodes(repo_id: str, position_keys: list[str]) -> list[dict[str, Any]]:
    dataset = lerobot_dataset.LeRobotDataset(repo_id)
    episode_indices = _as_numpy_column(dataset, "episode_index").reshape(-1)
    frame_indices = _as_numpy_column(dataset, "frame_index").reshape(-1)
    position_columns = []
    for position_key in position_keys:
        positions = _as_numpy_column(dataset, position_key).astype(np.float32)
        if positions.ndim < 2:
            raise ValueError(f"{position_key} must contain a vector per frame, got {positions.shape}")
        position_columns.append(positions)

    episodes = []
    for episode_id in np.unique(episode_indices):
        global_indices = np.flatnonzero(episode_indices == episode_id)
        order = np.argsort(frame_indices[global_indices])
        global_indices = global_indices[order]
        # 多臂轨迹分别取最新 xyz 后拼接；UOT 标签训练时再从所有原始位置列按索引读取。
        latest_position_parts = []
        for positions in position_columns:
            latest_positions = positions[global_indices]
            if latest_positions.ndim >= 3:
                latest_positions = latest_positions[:, -1]
            latest_position_parts.append(latest_positions.reshape(len(global_indices), -1)[..., :3])
        latest_positions = np.concatenate(latest_position_parts, axis=-1)
        episodes.append(
            {
                "episode_index": int(episode_id),
                "indices": global_indices.astype(int).tolist(),
                "dtw_positions": latest_positions.tolist(),
            }
        )
    return episodes


@njit(cache=True)
def _dtw_core(source: np.ndarray, target: np.ndarray):
    """用 Numba 编译动态规划核心，避免大规模轨迹库上的 Python 双重循环。"""
    ns, nt = len(source), len(target)
    accumulated = np.full((ns + 1, nt + 1), np.inf, dtype=np.float64)
    accumulated[0, 0] = 0.0
    predecessor = np.zeros((ns + 1, nt + 1), dtype=np.int8)
    for source_t in range(1, ns + 1):
        for target_t in range(1, nt + 1):
            squared_distance = 0.0
            for dim in range(source.shape[1]):
                delta = source[source_t - 1, dim] - target[target_t - 1, dim]
                squared_distance += delta * delta
            diagonal = accumulated[source_t - 1, target_t - 1]
            vertical = accumulated[source_t - 1, target_t]
            horizontal = accumulated[source_t, target_t - 1]
            if diagonal <= vertical and diagonal <= horizontal:
                move, previous = 0, diagonal
            elif vertical <= horizontal:
                move, previous = 1, vertical
            else:
                move, previous = 2, horizontal
            accumulated[source_t, target_t] = np.sqrt(squared_distance) + previous
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


def _dtw(source: np.ndarray, target: np.ndarray) -> tuple[float, list[list[int]]]:
    """返回累计距离和 target 时刻到一个或多个 source 时刻的最优路径映射。"""
    distance, path_source, path_target = _dtw_core(source, target)
    ns, nt = len(source), len(target)
    mapping: list[list[int]] = [[] for _ in range(nt)]
    for source_t, target_t in zip(path_source.tolist(), path_target.tolist(), strict=True):
        mapping[target_t].append(source_t)
    for target_t, aligned in enumerate(mapping):
        if not aligned:
            fallback = min(round(target_t * max(ns - 1, 0) / max(nt - 1, 1)), ns - 1)
            aligned.append(int(fallback))
        aligned.sort()
    return float(distance), mapping


def _trajectory_descriptor(positions: np.ndarray, points: int = 16) -> np.ndarray:
    """仅用于超大轨迹库的候选预筛；不参与最终 DTW 距离或 UOT 损失。"""
    sample_times = np.linspace(0, len(positions) - 1, points)
    left = np.floor(sample_times).astype(int)
    right = np.minimum(left + 1, len(positions) - 1)
    ratio = (sample_times - left)[:, None]
    return ((1.0 - ratio) * positions[left] + ratio * positions[right]).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute cross-scene DTW pairs for UOT co-training")
    parser.add_argument("--source", action="append", required=True, help="辅助数据集路径；可重复传入")
    parser.add_argument("--target", required=True, help="目标数据集路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument(
        "--position-key",
        action="append",
        dest="position_keys",
        help="DTW/UOT 使用的位置列；双臂数据重复传入两次。默认 observation.robot0_eef_pos",
    )
    parser.add_argument("--top-k", type=int, default=0, help="每条目标轨迹仅保留最近 K 条辅助轨迹；0 表示全部")
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=0,
        help="每个 source 数据集先按轨迹描述子保留 K 条再做精确 DTW；0 表示论文式全量比较",
    )
    args = parser.parse_args()
    position_keys = args.position_keys or ["observation.robot0_eef_pos"]

    sources = [{"repo_id": repo_id, "episodes": _extract_episodes(repo_id, position_keys)} for repo_id in args.source]
    target_episodes = _extract_episodes(args.target, position_keys)
    pairs = []
    for target_episode_index, target_episode in enumerate(target_episodes):
        target_positions = np.asarray(target_episode["dtw_positions"], dtype=np.float32)
        target_descriptor = _trajectory_descriptor(target_positions)
        candidates = []
        for source_repo_index, source in enumerate(sources):
            source_episode_indices = list(range(len(source["episodes"])))
            if args.candidate_pool > 0 and len(source_episode_indices) > args.candidate_pool:
                descriptor_distances = []
                for source_episode_index in source_episode_indices:
                    source_positions = np.asarray(
                        source["episodes"][source_episode_index]["dtw_positions"], dtype=np.float32
                    )
                    descriptor_distances.append(
                        np.linalg.norm(_trajectory_descriptor(source_positions) - target_descriptor)
                    )
                source_episode_indices = np.argsort(descriptor_distances)[: args.candidate_pool].tolist()
            for source_episode_index in source_episode_indices:
                source_episode = source["episodes"][source_episode_index]
                source_positions = np.asarray(source_episode["dtw_positions"], dtype=np.float32)
                distance, mapping = _dtw(source_positions, target_positions)
                candidates.append(
                    {
                        "source_repo_index": source_repo_index,
                        "source_episode_list_index": source_episode_index,
                        "normalized_distance": distance / max(len(source_positions), len(target_positions)),
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

    # dtw_positions 仅用于配对计算，输出时删除以减小文件体积。
    for source in sources:
        for episode in source["episodes"]:
            episode.pop("dtw_positions", None)
    for episode in target_episodes:
        episode.pop("dtw_positions", None)

    output = {
        "schema_version": 1,
        "position_keys": position_keys,
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
