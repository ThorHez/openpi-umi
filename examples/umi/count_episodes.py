#!/usr/bin/env python3
"""
统计 LeRobot 数据集中 episode 总数。

通过 meta/episodes.jsonl 的行数（每行一个 episode）得到条数；
若存在 meta/info.json 则同时读取其中的 total_episodes 并对比。

用法:
    python count_episodes.py /path/to/dataset1
    python count_episodes.py /path/to/dataset1 /path/to/dataset2 ...
    python count_episodes.py <数据集根目录> [更多数据集...]
"""

import argparse
import json
from pathlib import Path


def count_episodes(dataset_path: Path) -> dict:
    """
    返回数据集 episode 统计。
    - episodes_file: meta/episodes.jsonl 路径
    - count_from_jsonl: 按 jsonl 行数统计的 episode 数
    - total_episodes_from_info: info.json 中的 total_episodes（若存在）
    - total_frames_from_info: info.json 中的 total_frames（若存在）
    """
    dataset_path = Path(dataset_path).resolve()
    episodes_file = dataset_path / "meta" / "episodes.jsonl"
    info_file = dataset_path / "meta" / "info.json"

    result = {
        "dataset_path": str(dataset_path),
        "episodes_file": str(episodes_file),
        "count_from_jsonl": None,
        "total_episodes_from_info": None,
        "total_frames_from_info": None,
    }

    if not episodes_file.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_file}")

    with open(episodes_file, "r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())
    result["count_from_jsonl"] = count

    if info_file.exists():
        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        result["total_episodes_from_info"] = info.get("total_episodes")
        result["total_frames_from_info"] = info.get("total_frames")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="统计 LeRobot 数据集中 episode 总数（依据 meta/episodes.jsonl），支持多个数据集。"
    )
    parser.add_argument(
        "datasets",
        type=str,
        nargs="+",
        help="一个或多个数据集根目录（含 meta/episodes.jsonl）",
    )
    args = parser.parse_args()

    results = []
    for path_str in args.datasets:
        dataset_path = Path(path_str)
        if not dataset_path.exists():
            print(f"⚠️ 路径不存在，跳过: {dataset_path}")
            continue
        try:
            result = count_episodes(dataset_path)
            results.append(result)
        except FileNotFoundError as e:
            print(f"⚠️ 跳过 {dataset_path}: {e}")
            continue

    if not results:
        raise SystemExit("没有成功统计任何数据集。")

    grand_total_episodes = 0
    grand_total_frames = 0
    for i, result in enumerate(results):
        total = result["count_from_jsonl"]
        grand_total_episodes += total
        if result["total_frames_from_info"] is not None:
            grand_total_frames += result["total_frames_from_info"]

        name = Path(result["dataset_path"]).name
        if len(results) > 1:
            print(f"\n[{i + 1}] {name}")
        print(f"  路径: {result['dataset_path']}")
        print(f"  Episode 总数（meta/episodes.jsonl 行数）: {total}")
        if result["total_episodes_from_info"] is not None:
            info_total = result["total_episodes_from_info"]
            if info_total != total:
                print(f"  ⚠️ meta/info.json total_episodes ({info_total}) 与 jsonl 行数不一致")
            if result["total_frames_from_info"] is not None:
                print(f"  total_frames: {result['total_frames_from_info']}")

    if len(results) > 1:
        print("\n" + "=" * 50)
        print(f"合计: {len(results)} 个数据集, {grand_total_episodes} episodes", end="")
        if grand_total_frames > 0:
            print(f", {grand_total_frames} frames", end="")
        print()


if __name__ == "__main__":
    main()
