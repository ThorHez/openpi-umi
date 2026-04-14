#!/usr/bin/env python3
"""
快速统计脚本 - 快速获取数据集关键统计信息
"""

import json
import argparse
from pathlib import Path
import numpy as np
from typing import Dict


def load_dataset_info(dataset_path: Path) -> Dict:
    """加载数据集信息"""
    with open(dataset_path / "meta" / "info.json", 'r') as f:
        return json.load(f)


def load_norm_stats(dataset_path: Path) -> Dict:
    """加载归一化统计信息"""
    norm_stats_path = dataset_path / "norm_stats.json"
    if norm_stats_path.exists():
        with open(norm_stats_path, 'r') as f:
            return json.load(f)
    return None


def load_episode_stats(dataset_path: Path) -> list:
    """加载episode统计信息"""
    episodes_stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    episodes = []
    with open(episodes_stats_path, 'r') as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def print_divider(char='=', length=70):
    """打印分隔线"""
    print(char * length)


def print_section(title):
    """打印节标题"""
    print(f"\n{title}")
    print_divider()


def quick_stats(dataset_path: str, detailed: bool = False):
    """快速统计数据集信息"""
    dataset_path = Path(dataset_path)
    
    print_divider('=', 70)
    print(f"数据集快速统计: {dataset_path.name}")
    print_divider('=', 70)
    
    # 加载数据
    info = load_dataset_info(dataset_path)
    norm_stats = load_norm_stats(dataset_path)
    episodes_stats = load_episode_stats(dataset_path)
    
    # 基本信息
    print_section("📊 基本信息")
    print(f"  数据集路径:        {dataset_path}")
    print(f"  机器人类型:        {info['robot_type']}")
    print(f"  代码版本:          {info['codebase_version']}")
    print(f"  FPS:              {info['fps']}")
    
    print_section("📈 数据量统计")
    print(f"  总Episodes:       {info['total_episodes']}")
    print(f"  总帧数:           {info['total_frames']:,}")
    print(f"  总任务数:         {info['total_tasks']}")
    print(f"  数据分块数:       {info['total_chunks']}")
    
    # Episode长度统计
    episode_lengths = [ep['stats']['actions']['count'][0] for ep in episodes_stats]
    durations = [length / info['fps'] for length in episode_lengths]
    
    print_section("⏱️  Episode长度统计")
    print(f"  平均长度:         {np.mean(episode_lengths):.1f} 帧 "
          f"({np.mean(durations):.2f} 秒)")
    print(f"  中位数:           {np.median(episode_lengths):.1f} 帧 "
          f"({np.median(durations):.2f} 秒)")
    print(f"  最短:             {np.min(episode_lengths)} 帧 "
          f"({np.min(durations):.2f} 秒)")
    print(f"  最长:             {np.max(episode_lengths)} 帧 "
          f"({np.max(durations):.2f} 秒)")
    print(f"  标准差:           {np.std(episode_lengths):.1f} 帧 "
          f"({np.std(durations):.2f} 秒)")
    print(f"  总时长:           {sum(durations) / 3600:.2f} 小时")
    
    # 数据特征
    print_section("🎯 数据特征")
    features = info['features']
    for feature_name, feature_info in features.items():
        if feature_name.startswith('observation') or feature_name in ['actions', 'state']:
            shape_str = 'x'.join(map(str, feature_info['shape']))
            print(f"  {feature_name:40s} {feature_info['dtype']:10s} [{shape_str}]")
    
    # 归一化统计
    if norm_stats and detailed:
        print_section("📐 归一化统计")
        
        state_labels = ['eef_x', 'eef_y', 'eef_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        
        print("\n  State:")
        state_stats = norm_stats['norm_stats']['state']
        for i, label in enumerate(state_labels):
            print(f"    {label:10s}: "
                  f"μ={state_stats['mean'][i]:8.4f}, "
                  f"σ={state_stats['std'][i]:7.4f}, "
                  f"范围=[{state_stats['q01'][i]:7.3f}, {state_stats['q99'][i]:7.3f}]")
        
        print("\n  Actions:")
        action_stats = norm_stats['norm_stats']['actions']
        for i, label in enumerate(state_labels):
            print(f"    {label:10s}: "
                  f"μ={action_stats['mean'][i]:8.6f}, "
                  f"σ={action_stats['std'][i]:7.6f}, "
                  f"范围=[{action_stats['q01'][i]:8.4f}, {action_stats['q99'][i]:8.4f}]")
    
    # 数据存储
    print_section("💾 数据存储")
    
    # 计算数据集大小
    total_size = 0
    data_dir = dataset_path / "data"
    if data_dir.exists():
        for file in data_dir.rglob("*.parquet"):
            total_size += file.stat().st_size
    
    print(f"  Parquet文件总大小: {total_size / (1024**3):.2f} GB")
    print(f"  平均每Episode:     {total_size / info['total_episodes'] / (1024**2):.2f} MB")
    print(f"  平均每帧:          {total_size / info['total_frames'] / 1024:.2f} KB")
    
    # 数据分割
    print_section("✂️  数据分割")
    for split_name, split_range in info['splits'].items():
        print(f"  {split_name:10s}: {split_range}")
    
    print_divider('=', 70)
    print("\n✅ 统计完成！\n")
    
    # 建议
    if not detailed:
        print("💡 提示: 使用 --detailed 参数查看更详细的统计信息")
        print("💡 提示: 使用 analyze_dataset.py 生成完整的分析报告\n")


def main():
    parser = argparse.ArgumentParser(
        description="快速获取数据集统计信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 快速统计
  python quick_stats.py /data/umi_lerobot_dataset_v3
  
  # 详细统计
  python quick_stats.py /data/umi_lerobot_dataset_v3 --detailed
        """
    )
    
    parser.add_argument(
        'dataset_path',
        type=str,
        help='数据集根目录路径'
    )
    
    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='显示详细统计信息'
    )
    
    args = parser.parse_args()
    
    try:
        quick_stats(args.dataset_path, detailed=args.detailed)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

