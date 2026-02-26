#!/usr/bin/env python3
"""
分析 UMI zarr.zip 数据集各维度的分布

用法:
    python analyze_zarr_distribution.py /path/to/dataset.zarr.zip
    python analyze_zarr_distribution.py /path/to/dataset.zarr.zip --output ./output
    python analyze_zarr_distribution.py /path/to/dataset.zarr.zip --max-episodes 100
"""

import argparse
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from tqdm import tqdm


def load_zarr_data(zarr_zip_path: str, tmp_dir: str = None) -> zarr.Group:
    """
    加载 zarr.zip 文件
    
    Args:
        zarr_zip_path: zarr.zip 文件路径
        tmp_dir: 临时解压目录，None则自动创建
    
    Returns:
        zarr.Group 根对象
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="zarr_analysis_")
    
    print(f"解压 zarr 文件到: {tmp_dir}")
    with zipfile.ZipFile(zarr_zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    
    root = zarr.open(tmp_dir, mode='r')
    return root, tmp_dir


def get_episode_indices(meta_group: zarr.Group) -> List[tuple]:
    """
    获取所有 episode 的起止索引
    
    Returns:
        List of (start_idx, end_idx) tuples
    """
    episode_ends = np.array(meta_group['episode_ends'])
    episodes = []
    start = 0
    for end in episode_ends:
        episodes.append((start, end))
        start = end
    return episodes


def analyze_distribution(data: np.ndarray, name: str) -> Dict[str, Any]:
    """
    分析数据分布
    
    Args:
        data: 数据数组
        name: 数据名称
    
    Returns:
        分布统计信息字典
    """
    # 处理多维数据
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    n_dims = data.shape[-1]
    
    stats = {
        'name': name,
        'shape': data.shape,
        'n_dims': n_dims,
        'dims': []
    }
    
    for dim in range(n_dims):
        dim_data = data[..., dim].flatten()
        # 过滤无效值
        valid_data = dim_data[np.isfinite(dim_data)]
        
        if len(valid_data) == 0:
            stats['dims'].append({
                'dim': dim,
                'count': 0,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'q01': np.nan,
                'q25': np.nan,
                'q50': np.nan,
                'q75': np.nan,
                'q99': np.nan,
            })
            continue
        
        stats['dims'].append({
            'dim': dim,
            'count': len(valid_data),
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'q01': float(np.percentile(valid_data, 1)),
            'q25': float(np.percentile(valid_data, 25)),
            'q50': float(np.percentile(valid_data, 50)),
            'q75': float(np.percentile(valid_data, 75)),
            'q99': float(np.percentile(valid_data, 99)),
            'data': valid_data  # 保存数据用于绘图
        })
    
    return stats


def print_stats(stats: Dict[str, Any], dim_names: List[str] = None):
    """打印统计信息"""
    print(f"\n{'='*70}")
    print(f"特征: {stats['name']}")
    print(f"形状: {stats['shape']}, 维度数: {stats['n_dims']}")
    print(f"{'='*70}")
    
    for dim_stat in stats['dims']:
        dim_idx = dim_stat['dim']
        dim_name = dim_names[dim_idx] if dim_names and dim_idx < len(dim_names) else f"dim_{dim_idx}"
        
        print(f"\n  [{dim_name}]")
        print(f"    样本数: {dim_stat['count']}")
        print(f"    范围: [{dim_stat['min']:.6f}, {dim_stat['max']:.6f}]")
        print(f"    均值: {dim_stat['mean']:.6f}, 标准差: {dim_stat['std']:.6f}")
        print(f"    分位数: q01={dim_stat['q01']:.6f}, q50={dim_stat['q50']:.6f}, q99={dim_stat['q99']:.6f}")


def plot_distributions(all_stats: List[Dict[str, Any]], output_dir: Path):
    """
    绘制所有特征的分布图
    
    Args:
        all_stats: 所有特征的统计信息列表
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for stats in all_stats:
        name = stats['name']
        n_dims = stats['n_dims']
        
        if n_dims == 0:
            continue
        
        # 计算子图布局
        n_cols = min(4, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_dims == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for dim_stat in stats['dims']:
            dim_idx = dim_stat['dim']
            ax = axes[dim_idx]
            
            if 'data' not in dim_stat or len(dim_stat.get('data', [])) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'Dim {dim_idx}')
                continue
            
            data = dim_stat['data']
            
            # 绘制直方图
            ax.hist(data, bins=50, edgecolor='black', alpha=0.7, density=True)
            
            # 添加统计线
            ax.axvline(dim_stat['mean'], color='r', linestyle='--', linewidth=1.5, 
                      label=f"mean={dim_stat['mean']:.3f}")
            ax.axvline(dim_stat['q01'], color='g', linestyle=':', linewidth=1, 
                      label=f"q01={dim_stat['q01']:.3f}")
            ax.axvline(dim_stat['q99'], color='g', linestyle=':', linewidth=1, 
                      label=f"q99={dim_stat['q99']:.3f}")
            
            ax.set_title(f'Dim {dim_idx}', fontsize=10)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(n_dims, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{name} Distribution\nShape: {stats["shape"]}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        safe_name = name.replace('/', '_').replace('\\', '_')
        output_path = output_dir / f'{safe_name}_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存: {output_path}")
        plt.close()


def plot_summary(all_stats: List[Dict[str, Any]], output_dir: Path):
    """
    绘制汇总图：所有特征的箱线图
    """
    # 过滤有数据的特征
    valid_stats = [s for s in all_stats if s['n_dims'] > 0 and any('data' in d for d in s['dims'])]
    
    if not valid_stats:
        return
    
    fig, axes = plt.subplots(len(valid_stats), 1, figsize=(12, 3 * len(valid_stats)))
    if len(valid_stats) == 1:
        axes = [axes]
    
    for ax, stats in zip(axes, valid_stats):
        name = stats['name']
        box_data = []
        labels = []
        
        for dim_stat in stats['dims']:
            if 'data' in dim_stat and len(dim_stat['data']) > 0:
                # 采样以避免内存问题
                data = dim_stat['data']
                if len(data) > 10000:
                    data = np.random.choice(data, 10000, replace=False)
                box_data.append(data)
                labels.append(f"d{dim_stat['dim']}")
        
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_title(f'{name}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('All Features Summary (Boxplot)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'summary_boxplot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存汇总图: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='分析 UMI zarr.zip 数据集各维度的分布',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python analyze_zarr_distribution.py /path/to/dataset.zarr.zip
    python analyze_zarr_distribution.py /path/to/dataset.zarr.zip --output ./output
    python analyze_zarr_distribution.py /path/to/dataset.zarr.zip --max-episodes 100
        """
    )
    
    parser.add_argument('zarr_path', type=str, help='zarr.zip 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出目录 (默认: <zarr_name>_distribution)')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='最大分析的 episode 数量')
    parser.add_argument('--keys', type=str, nargs='+', default=None,
                       help='指定要分析的数据键 (默认: 分析所有数值型数据)')
    
    args = parser.parse_args()
    
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"错误: 文件不存在 {zarr_path}")
        return
    
    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = zarr_path.parent / f"{zarr_path.stem}_distribution"
    
    print(f"分析文件: {zarr_path}")
    print(f"输出目录: {output_dir}")
    
    # 加载 zarr 数据
    root, tmp_dir = load_zarr_data(str(zarr_path))
    
    try:
        print(f"\n数据集结构:")
        print(f"  顶层键: {list(root.keys())}")
        
        data_group = root['data']
        meta_group = root['meta']
        
        print(f"  数据键: {list(data_group.keys())}")
        print(f"  元数据键: {list(meta_group.keys())}")
        
        # 获取 episode 信息
        episodes = get_episode_indices(meta_group)
        n_episodes = len(episodes)
        print(f"\n总 Episode 数: {n_episodes}")
        
        if args.max_episodes:
            n_episodes = min(n_episodes, args.max_episodes)
            print(f"分析前 {n_episodes} 个 episodes")
        
        # 确定要分析的键
        if args.keys:
            keys_to_analyze = args.keys
        else:
            # 默认分析所有数值型数据
            keys_to_analyze = []
            skip_keys = ['camera', 'rgb', 'image', 'video']  # 跳过图像数据
            for key in data_group.keys():
                if not any(skip in key.lower() for skip in skip_keys):
                    keys_to_analyze.append(key)
        
        print(f"\n将分析以下数据键: {keys_to_analyze}")
        
        # 分析每个键的分布
        all_stats = []
        
        for key in tqdm(keys_to_analyze, desc="分析各维度分布"):
            if key not in data_group:
                print(f"  警告: 键 '{key}' 不存在，跳过")
                continue
            
            data = np.array(data_group[key])
            
            # 如果限制了 episode 数量，只取对应的数据
            if args.max_episodes and n_episodes < len(episodes):
                end_idx = episodes[n_episodes - 1][1]
                data = data[:end_idx]
            
            print(f"\n处理 {key}: shape={data.shape}, dtype={data.dtype}")
            
            # 跳过非数值型数据
            if not np.issubdtype(data.dtype, np.number):
                print(f"  跳过非数值型数据")
                continue
            
            stats = analyze_distribution(data, key)
            
            # 定义维度名称
            dim_names = None
            if 'eef_pos' in key:
                dim_names = ['x', 'y', 'z']
            elif 'eef_rot' in key or 'axis_angle' in key:
                dim_names = ['rx', 'ry', 'rz']
            elif 'gripper' in key:
                dim_names = ['width']
            elif 'action' in key.lower() and stats['n_dims'] == 7:
                dim_names = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
            elif 'action' in key.lower() and stats['n_dims'] == 14:
                dim_names = ['L_x', 'L_y', 'L_z', 'L_rx', 'L_ry', 'L_rz', 'L_gripper',
                            'R_x', 'R_y', 'R_z', 'R_rx', 'R_ry', 'R_rz', 'R_gripper']
            
            print_stats(stats, dim_names)
            all_stats.append(stats)
        
        # 绘制分布图
        print(f"\n正在生成可视化图表...")
        plot_distributions(all_stats, output_dir)
        plot_summary(all_stats, output_dir)
        
        print(f"\n✓ 分析完成! 结果保存在: {output_dir}")
        
    finally:
        # 清理临时目录
        if tmp_dir:
            print(f"\n清理临时目录: {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
