#!/usr/bin/env python3
"""
数据集切分脚本 - 将数据集按比例切分为训练集和验证集
"""

import json
import shutil
from pathlib import Path
import random
import argparse
import logging
import pyarrow.parquet as pq
import pyarrow as pa

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def split_dataset(
    dataset_path: str,
    val_ratio: float = 0.05,
    seed: int = 42
):
    """
    切分数据集
    
    Args:
        dataset_path: 数据集路径
        val_ratio: 验证集比例 (0-1之间)
        seed: 随机种子
    
    Note:
        切分过程会重新编号 episodes（包括文件名和 parquet 内部的 episode_index），
        因此总是会创建新文件，不会修改原始数据集。
    """
    dataset_path = Path(dataset_path)
    
    # 读取元数据
    info_file = dataset_path / "meta" / "info.json"
    episodes_file = dataset_path / "meta" / "episodes.jsonl"
    episodes_stats_file = dataset_path / "meta" / "episodes_stats.jsonl"
    
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    total_episodes = info['total_episodes']
    logging.info(f"数据集总episodes数: {total_episodes}")
    
    # 读取所有episodes信息
    episodes = []
    with open(episodes_file, 'r') as f:
        for line in f:
            episodes.append(json.loads(line))
    
    episodes_stats = []
    with open(episodes_stats_file, 'r') as f:
        for line in f:
            episodes_stats.append(json.loads(line))
    
    # 计算验证集大小
    num_val = max(1, int(total_episodes * val_ratio))
    num_train = total_episodes - num_val
    
    logging.info(f"验证集比例: {val_ratio * 100:.1f}%")
    logging.info(f"训练集episodes: {num_train}")
    logging.info(f"验证集episodes: {num_val}")
    
    # 随机选择验证集episodes
    random.seed(seed)
    all_indices = list(range(total_episodes))
    random.shuffle(all_indices)
    
    val_indices = sorted(all_indices[:num_val])
    train_indices = sorted(all_indices[num_val:])
    
    logging.info(f"验证集episodes索引: {val_indices[:10]}..." if len(val_indices) > 10 else f"验证集episodes索引: {val_indices}")
    
    # 创建新的目录结构
    train_dir = dataset_path.parent / f"{dataset_path.name}_train"
    val_dir = dataset_path.parent / f"{dataset_path.name}_val"
    
    for d in [train_dir, val_dir]:
        (d / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (d / "meta").mkdir(parents=True, exist_ok=True)
    
    logging.info(f"创建目录结构:")
    logging.info(f"  训练集: {train_dir}")
    logging.info(f"  验证集: {val_dir}")
    
    # 分割数据文件
    data_dir = dataset_path / "data" / "chunk-000"
    
    train_episodes = []
    train_episodes_stats = []
    train_frames = 0
    
    val_episodes = []
    val_episodes_stats = []
    val_frames = 0
    
    # 处理训练集 - 重新编号episodes
    logging.info("处理训练集（重新编号episodes）...")
    for new_idx, old_idx in enumerate(train_indices):
        src_file = data_dir / f"episode_{old_idx:06d}.parquet"
        dst_file = train_dir / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet"
        
        # 读取parquet文件，修改episode_index列
        table = pq.read_table(src_file)
        df = table.to_pandas()
        
        # 修改episode_index列为新的索引
        df['episode_index'] = new_idx
        
        # 写入新文件
        new_table = pa.Table.from_pandas(df, schema=table.schema)
        pq.write_table(new_table, dst_file)
        
        # 重新编号元数据中的 episode_index
        ep = episodes[old_idx].copy()
        ep['episode_index'] = new_idx
        train_episodes.append(ep)
        
        ep_stat = episodes_stats[old_idx].copy()
        ep_stat['episode_index'] = new_idx
        train_episodes_stats.append(ep_stat)
        
        train_frames += episodes[old_idx]['length']
    
    # 处理验证集 - 重新编号episodes
    logging.info("处理验证集（重新编号episodes）...")
    for new_idx, old_idx in enumerate(val_indices):
        src_file = data_dir / f"episode_{old_idx:06d}.parquet"
        dst_file = val_dir / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet"
        
        # 读取parquet文件，修改episode_index列
        table = pq.read_table(src_file)
        df = table.to_pandas()
        
        # 修改episode_index列为新的索引
        df['episode_index'] = new_idx
        
        # 写入新文件
        new_table = pa.Table.from_pandas(df, schema=table.schema)
        pq.write_table(new_table, dst_file)
        
        # 重新编号元数据中的 episode_index
        ep = episodes[old_idx].copy()
        ep['episode_index'] = new_idx
        val_episodes.append(ep)
        
        ep_stat = episodes_stats[old_idx].copy()
        ep_stat['episode_index'] = new_idx
        val_episodes_stats.append(ep_stat)
        
        val_frames += episodes[old_idx]['length']
    
    # 保存训练集元数据
    logging.info("保存训练集元数据...")
    train_info = info.copy()
    train_info['total_episodes'] = num_train
    train_info['total_frames'] = train_frames
    train_info['splits'] = {'train': f"0:{num_train}"}
    
    with open(train_dir / "meta" / "info.json", 'w') as f:
        json.dump(train_info, f, indent=4)
    
    with open(train_dir / "meta" / "episodes.jsonl", 'w') as f:
        for ep in train_episodes:
            f.write(json.dumps(ep) + '\n')
    
    with open(train_dir / "meta" / "episodes_stats.jsonl", 'w') as f:
        for ep_stat in train_episodes_stats:
            f.write(json.dumps(ep_stat) + '\n')
    
    # 保存验证集元数据
    logging.info("保存验证集元数据...")
    val_info = info.copy()
    val_info['total_episodes'] = num_val
    val_info['total_frames'] = val_frames
    val_info['splits'] = {'val': f"0:{num_val}"}
    
    with open(val_dir / "meta" / "info.json", 'w') as f:
        json.dump(val_info, f, indent=4)
    
    with open(val_dir / "meta" / "episodes.jsonl", 'w') as f:
        for ep in val_episodes:
            f.write(json.dumps(ep) + '\n')
    
    with open(val_dir / "meta" / "episodes_stats.jsonl", 'w') as f:
        for ep_stat in val_episodes_stats:
            f.write(json.dumps(ep_stat) + '\n')
    
    # 复制其他文件
    logging.info("复制其他元数据文件...")
    for filename in ['tasks.jsonl', 'norm_stats.json']:
        src = dataset_path / "meta" / filename
        if src.exists():
            shutil.copy2(src, train_dir / "meta" / filename)
            shutil.copy2(src, val_dir / "meta" / filename)
    
    # 复制norm_stats.json到根目录
    norm_stats = dataset_path / "norm_stats.json"
    if norm_stats.exists():
        shutil.copy2(norm_stats, train_dir / "norm_stats.json")
        shutil.copy2(norm_stats, val_dir / "norm_stats.json")
    
    # 保存切分信息
    split_info = {
        'original_dataset': str(dataset_path),
        'train_dataset': str(train_dir),
        'val_dataset': str(val_dir),
        'total_episodes': total_episodes,
        'train_episodes': num_train,
        'val_episodes': num_val,
        'val_ratio': val_ratio,
        'train_frames': train_frames,
        'val_frames': val_frames,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'seed': seed,
    }
    
    split_info_file = dataset_path.parent / f"{dataset_path.name}_split_info.json"
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=4)
    
    logging.info(f"\n{'='*70}")
    logging.info("数据集切分完成！")
    logging.info(f"{'='*70}")
    logging.info(f"原始数据集: {dataset_path}")
    logging.info(f"训练集: {train_dir}")
    logging.info(f"  - Episodes: {num_train}")
    logging.info(f"  - Frames: {train_frames}")
    logging.info(f"验证集: {val_dir}")
    logging.info(f"  - Episodes: {num_val}")
    logging.info(f"  - Frames: {val_frames}")
    logging.info(f"切分信息已保存到: {split_info_file}")
    logging.info(f"{'='*70}\n")
    
    logging.info("注意: 切分过程会修改 episode_index，因此总是从原始数据读取并写入新文件。")
    logging.info("原始数据集保持不变。")


def main():
    parser = argparse.ArgumentParser(
        description='切分数据集为训练集和验证集（会重新编号 episodes）'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/root/openpi-umi/data/umi_lerobot_dataset_v7.1',
        help='数据集路径'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.05,
        help='验证集比例 (默认: 0.05 即 5%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    args = parser.parse_args()
    
    split_dataset(
        dataset_path=args.dataset_path,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

