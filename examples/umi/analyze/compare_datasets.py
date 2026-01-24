#!/usr/bin/env python3
"""
数据集对比脚本 - 对比两个数据集的差异
用于比较不同版本的数据集或评估数据预处理的效果
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

sns.set_style("whitegrid")


class DatasetComparator:
    """数据集对比器"""
    
    def __init__(self, dataset1_path: str, dataset2_path: str):
        self.dataset1_path = Path(dataset1_path)
        self.dataset2_path = Path(dataset2_path)
        
        # 加载数据集信息
        self.info1 = self._load_info(self.dataset1_path)
        self.info2 = self._load_info(self.dataset2_path)
        
        self.name1 = self.dataset1_path.name
        self.name2 = self.dataset2_path.name
        
        print(f"数据集1: {self.name1}")
        print(f"数据集2: {self.name2}")
    
    def _load_info(self, dataset_path: Path) -> Dict:
        """加载数据集信息"""
        with open(dataset_path / "meta" / "info.json", 'r') as f:
            return json.load(f)
    
    def _load_episodes_stats(self, dataset_path: Path) -> list:
        """加载episode统计信息"""
        episodes = []
        with open(dataset_path / "meta" / "episodes_stats.jsonl", 'r') as f:
            for line in f:
                episodes.append(json.loads(line))
        return episodes
    
    def compare_basic_info(self):
        """对比基本信息"""
        print("\n" + "="*70)
        print("基本信息对比")
        print("="*70)
        
        comparisons = [
            ("Episodes数量", 'total_episodes'),
            ("总帧数", 'total_frames'),
            ("任务数", 'total_tasks'),
            ("FPS", 'fps'),
            ("数据分块数", 'total_chunks'),
        ]
        
        print(f"\n{'指标':<20} {self.name1:<25} {self.name2:<25} {'差异':<15}")
        print("-" * 85)
        
        for label, key in comparisons:
            val1 = self.info1[key]
            val2 = self.info2[key]
            
            if isinstance(val1, int) and isinstance(val2, int):
                diff = val2 - val1
                diff_str = f"{diff:+,}"
            elif isinstance(val1, float) and isinstance(val2, float):
                diff = val2 - val1
                diff_str = f"{diff:+.2f}"
            else:
                diff_str = "-"
            
            print(f"{label:<20} {str(val1):<25} {str(val2):<25} {diff_str:<15}")
    
    def compare_episode_lengths(self) -> Tuple[np.ndarray, np.ndarray]:
        """对比Episode长度分布"""
        print("\n" + "="*70)
        print("Episode长度分布对比")
        print("="*70)
        
        episodes1 = self._load_episodes_stats(self.dataset1_path)
        episodes2 = self._load_episodes_stats(self.dataset2_path)
        
        lengths1 = np.array([ep['stats']['actions']['count'][0] for ep in episodes1])
        lengths2 = np.array([ep['stats']['actions']['count'][0] for ep in episodes2])
        
        fps1 = self.info1['fps']
        fps2 = self.info2['fps']
        
        print(f"\n{'统计量':<20} {self.name1:<25} {self.name2:<25}")
        print("-" * 70)
        
        stats = [
            ("平均长度 (帧)", np.mean(lengths1), np.mean(lengths2)),
            ("中位数 (帧)", np.median(lengths1), np.median(lengths2)),
            ("最短 (帧)", np.min(lengths1), np.min(lengths2)),
            ("最长 (帧)", np.max(lengths1), np.max(lengths2)),
            ("标准差 (帧)", np.std(lengths1), np.std(lengths2)),
            ("平均时长 (秒)", np.mean(lengths1) / fps1, np.mean(lengths2) / fps2),
            ("总时长 (小时)", np.sum(lengths1) / fps1 / 3600, np.sum(lengths2) / fps2 / 3600),
        ]
        
        for label, val1, val2 in stats:
            print(f"{label:<20} {val1:<25.2f} {val2:<25.2f}")
        
        return lengths1, lengths2
    
    def compare_normalization_stats(self):
        """对比归一化统计"""
        print("\n" + "="*70)
        print("归一化统计对比")
        print("="*70)
        
        norm1_path = self.dataset1_path / "norm_stats.json"
        norm2_path = self.dataset2_path / "norm_stats.json"
        
        if not norm1_path.exists() or not norm2_path.exists():
            print("⚠ 一个或两个数据集缺少归一化统计文件")
            return
        
        with open(norm1_path, 'r') as f:
            norm1 = json.load(f)['norm_stats']
        
        with open(norm2_path, 'r') as f:
            norm2 = json.load(f)['norm_stats']
        
        state_labels = ['eef_x', 'eef_y', 'eef_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        
        # State对比
        print("\nState均值对比:")
        print(f"{'维度':<12} {self.name1:<15} {self.name2:<15} {'差异':<12}")
        print("-" * 54)
        
        for i, label in enumerate(state_labels):
            mean1 = norm1['state']['mean'][i]
            mean2 = norm2['state']['mean'][i]
            diff = mean2 - mean1
            print(f"{label:<12} {mean1:<15.6f} {mean2:<15.6f} {diff:<+12.6f}")
        
        # Action对比
        print("\nAction标准差对比:")
        print(f"{'维度':<12} {self.name1:<15} {self.name2:<15} {'差异':<12}")
        print("-" * 54)
        
        for i, label in enumerate(state_labels):
            std1 = norm1['actions']['std'][i]
            std2 = norm2['actions']['std'][i]
            diff = std2 - std1
            print(f"{label:<12} {std1:<15.6f} {std2:<15.6f} {diff:<+12.6f}")
    
    def compare_data_sizes(self):
        """对比数据存储大小"""
        print("\n" + "="*70)
        print("数据存储大小对比")
        print("="*70)
        
        def get_dataset_size(dataset_path: Path) -> int:
            total_size = 0
            data_dir = dataset_path / "data"
            if data_dir.exists():
                for file in data_dir.rglob("*.parquet"):
                    total_size += file.stat().st_size
            return total_size
        
        size1 = get_dataset_size(self.dataset1_path)
        size2 = get_dataset_size(self.dataset2_path)
        
        print(f"\n{'指标':<25} {self.name1:<20} {self.name2:<20}")
        print("-" * 65)
        
        print(f"{'总大小 (GB)':<25} {size1/(1024**3):<20.2f} {size2/(1024**3):<20.2f}")
        print(f"{'每Episode (MB)':<25} "
              f"{size1/self.info1['total_episodes']/(1024**2):<20.2f} "
              f"{size2/self.info2['total_episodes']/(1024**2):<20.2f}")
        print(f"{'每帧 (KB)':<25} "
              f"{size1/self.info1['total_frames']/1024:<20.2f} "
              f"{size2/self.info2['total_frames']/1024:<20.2f}")
        
        if size1 > 0:
            compression_ratio = (size2 / size1) * 100
            print(f"\n数据集2相对于数据集1: {compression_ratio:.1f}%")
            if compression_ratio < 100:
                print(f"  节省空间: {(size1 - size2)/(1024**3):.2f} GB "
                      f"({100 - compression_ratio:.1f}%)")
            elif compression_ratio > 100:
                print(f"  增加空间: {(size2 - size1)/(1024**3):.2f} GB "
                      f"({compression_ratio - 100:.1f}%)")
    
    def visualize_comparison(self, lengths1: np.ndarray, lengths2: np.ndarray, 
                           output_path: str = None):
        """可视化对比"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Episode长度直方图对比
        ax1 = axes[0, 0]
        ax1.hist(lengths1, bins=50, alpha=0.5, label=self.name1, edgecolor='black')
        ax1.hist(lengths2, bins=50, alpha=0.5, label=self.name2, edgecolor='black')
        ax1.set_xlabel('Episode Length (frames)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Episode Length Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 箱线图对比
        ax2 = axes[0, 1]
        ax2.boxplot([lengths1, lengths2], labels=[self.name1, self.name2])
        ax2.set_ylabel('Episode Length (frames)')
        ax2.set_title('Episode Length Box Plot Comparison')
        ax2.grid(True, alpha=0.3)
        
        # 累积分布对比
        ax3 = axes[1, 0]
        sorted1 = np.sort(lengths1)
        sorted2 = np.sort(lengths2)
        ax3.plot(sorted1, np.arange(len(sorted1)) / len(sorted1), 
                label=self.name1, linewidth=2)
        ax3.plot(sorted2, np.arange(len(sorted2)) / len(sorted2), 
                label=self.name2, linewidth=2)
        ax3.set_xlabel('Episode Length (frames)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 统计摘要表格
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_data = [
            ['统计量', self.name1, self.name2],
            ['Episodes数', f"{len(lengths1)}", f"{len(lengths2)}"],
            ['平均长度', f"{np.mean(lengths1):.1f}", f"{np.mean(lengths2):.1f}"],
            ['中位数', f"{np.median(lengths1):.1f}", f"{np.median(lengths2):.1f}"],
            ['标准差', f"{np.std(lengths1):.1f}", f"{np.std(lengths2):.1f}"],
            ['最小值', f"{np.min(lengths1)}", f"{np.min(lengths2)}"],
            ['最大值', f"{np.max(lengths1)}", f"{np.max(lengths2)}"],
        ]
        
        table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ 对比图表已保存: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_comparison_report(self, output_dir: str = None):
        """生成完整对比报告"""
        if output_dir is None:
            output_dir = Path("./dataset_comparison")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*70)
        print(f"数据集对比报告")
        print("="*70)
        print(f"数据集1: {self.dataset1_path}")
        print(f"数据集2: {self.dataset2_path}")
        print(f"输出目录: {output_dir}")
        
        # 执行所有对比
        self.compare_basic_info()
        lengths1, lengths2 = self.compare_episode_lengths()
        self.compare_normalization_stats()
        self.compare_data_sizes()
        
        # 生成可视化
        self.visualize_comparison(lengths1, lengths2, 
                                 output_dir / "comparison_visualization.png")
        
        print("\n" + "="*70)
        print("对比报告生成完成！")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="对比两个数据集的差异",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 对比两个数据集
  python compare_datasets.py /data/dataset_v1 /data/dataset_v2
  
  # 指定输出目录
  python compare_datasets.py /data/dataset_v1 /data/dataset_v2 --output ./comparison
        """
    )
    
    parser.add_argument(
        'dataset1',
        type=str,
        help='第一个数据集路径'
    )
    
    parser.add_argument(
        'dataset2',
        type=str,
        help='第二个数据集路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出目录 (默认: ./dataset_comparison)'
    )
    
    args = parser.parse_args()
    
    comparator = DatasetComparator(args.dataset1, args.dataset2)
    comparator.generate_comparison_report(output_dir=args.output)


if __name__ == "__main__":
    main()

