#!/usr/bin/env python3
"""
数据集分析脚本 - 分析 UMI LeRobot 数据集
提供全面的数据集统计、可视化和质量检查功能
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pyarrow.parquet as pq

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False


class DatasetAnalyzer:
    """UMI LeRobot 数据集分析器"""
    
    def __init__(self, dataset_path: str, num_samples: int = 10):
        """
        初始化分析器
        
        Args:
            dataset_path: 数据集根目录路径
        """
        self.dataset_path = Path(dataset_path)
        self.meta_path = self.dataset_path / "meta"
        self.data_path = self.dataset_path / "data"
        
        # 加载元数据
        self.info = self._load_json(self.meta_path / "info.json")
        self.norm_stats = self._load_json(self.dataset_path / "norm_stats.json")
        self.episodes_stats = self._load_episodes_stats()

        self.num_samples = num_samples
        
        print(f"✓ 成功加载数据集: {dataset_path}")
        print(f"  总Episodes: {self.info['total_episodes']}")
        print(f"  总帧数: {self.info['total_frames']}")
        print(f"  FPS: {self.info['fps']}")
    
    def _load_json(self, path: Path) -> Dict:
        """加载JSON文件"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_episodes_stats(self) -> pd.DataFrame:
        """加载episodes统计信息"""
        episodes_stats_path = self.meta_path / "episodes_stats.jsonl"
        episodes = []
        with open(episodes_stats_path, 'r') as f:
            for line in f:
                episodes.append(json.loads(line))
        return episodes
    
    def print_basic_info(self):
        """打印基本数据集信息"""
        print("\n" + "="*60)
        print("数据集基本信息")
        print("="*60)
        
        print(f"\n代码版本: {self.info['codebase_version']}")
        print(f"机器人类型: {self.info['robot_type']}")
        print(f"总Episodes: {self.info['total_episodes']}")
        print(f"总帧数: {self.info['total_frames']}")
        print(f"总任务数: {self.info['total_tasks']}")
        print(f"FPS: {self.info['fps']}")
        print(f"数据分块数: {self.info['total_chunks']}")
        print(f"分块大小: {self.info['chunks_size']}")
        
        print(f"\n数据集分割:")
        for split_name, split_range in self.info['splits'].items():
            print(f"  {split_name}: {split_range}")
    
    def print_features_info(self):
        """打印特征信息"""
        print("\n" + "="*60)
        print("数据特征信息")
        print("="*60)
        
        features = self.info['features']
        for feature_name, feature_info in features.items():
            print(f"\n{feature_name}:")
            print(f"  数据类型: {feature_info['dtype']}")
            print(f"  形状: {feature_info['shape']}")
            if feature_info['names']:
                print(f"  维度名称: {feature_info['names']}")
    
    def analyze_episode_statistics(self) -> pd.DataFrame:
        """分析Episode统计信息"""
        print("\n" + "="*60)
        print("Episode 统计分析")
        print("="*60)
        
        # 提取episode长度
        episode_lengths = []
        for ep in self.episodes_stats:
            ep_idx = ep['episode_index']
            # 从任意特征的count中获取长度
            length = ep['stats']['actions']['count'][0]
            episode_lengths.append({
                'episode_index': ep_idx,
                'length': length,
                'duration_sec': length / self.info['fps']
            })
        
        df = pd.DataFrame(episode_lengths)
        
        print(f"\nEpisode 长度统计:")
        print(f"  平均长度: {df['length'].mean():.2f} 帧 ({df['duration_sec'].mean():.2f} 秒)")
        print(f"  中位数长度: {df['length'].median():.2f} 帧 ({df['duration_sec'].median():.2f} 秒)")
        print(f"  最短: {df['length'].min()} 帧 ({df['duration_sec'].min():.2f} 秒)")
        print(f"  最长: {df['length'].max()} 帧 ({df['duration_sec'].max():.2f} 秒)")
        print(f"  标准差: {df['length'].std():.2f} 帧 ({df['duration_sec'].std():.2f} 秒)")
        
        return df
    
    def analyze_normalization_stats(self):
        """分析归一化统计信息"""
        print("\n" + "="*60)
        print("归一化统计分析")
        print("="*60)
        
        norm_stats = self.norm_stats['norm_stats']
        
        # State 统计
        print("\nState 归一化统计 (7维):")
        state_stats = norm_stats['state']
        print(f"  均值: {np.array(state_stats['mean'])}")
        print(f"  标准差: {np.array(state_stats['std'])}")
        print(f"  第1百分位: {np.array(state_stats['q01'])}")
        print(f"  第99百分位: {np.array(state_stats['q99'])}")
        
        # Actions 统计
        print("\nActions 归一化统计 (7维):")
        action_stats = norm_stats['actions']
        print(f"  均值: {np.array(action_stats['mean'])}")
        print(f"  标准差: {np.array(action_stats['std'])}")
        print(f"  第1百分位: {np.array(action_stats['q01'])}")
        print(f"  第99百分位: {np.array(action_stats['q99'])}")
    
    def analyze_state_action_distribution(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分析状态和动作分布"""
        print("\n" + "="*60)
        print("状态和动作分布分析")
        print("="*60)
        
        # 从episodes_stats中提取统计信息
        state_mins = []
        state_maxs = []
        state_means = []
        state_stds = []
        
        action_mins = []
        action_maxs = []
        action_means = []
        action_stds = []
        
        for ep in self.episodes_stats:
            stats = ep['stats']
            state_mins.append(stats['state']['min'])
            state_maxs.append(stats['state']['max'])
            state_means.append(stats['state']['mean'])
            state_stds.append(stats['state']['std'])
            
            action_mins.append(stats['actions']['min'])
            action_maxs.append(stats['actions']['max'])
            action_means.append(stats['actions']['mean'])
            action_stds.append(stats['actions']['std'])
        
        # 转换为numpy数组
        state_mins = np.array(state_mins)
        state_maxs = np.array(state_maxs)
        state_means = np.array(state_means)
        
        action_mins = np.array(action_mins)
        action_maxs = np.array(action_maxs)
        action_means = np.array(action_means)
        
        # State维度标签
        state_labels = ['eef_x', 'eef_y', 'eef_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        
        print("\nState 分布 (全局):")
        for i, label in enumerate(state_labels):
            global_min = state_mins[:, i].min()
            global_max = state_maxs[:, i].max()
            global_mean = state_means[:, i].mean()
            print(f"  {label:10s}: 范围[{global_min:8.4f}, {global_max:8.4f}], "
                  f"平均={global_mean:8.4f}")
        
        print("\nAction 分布 (全局):")
        for i, label in enumerate(state_labels):
            global_min = action_mins[:, i].min()
            global_max = action_maxs[:, i].max()
            global_mean = action_means[:, i].mean()
            print(f"  {label:10s}: 范围[{global_min:8.4f}, {global_max:8.4f}], "
                  f"平均={global_mean:8.4f}")
        
        # 创建DataFrame用于后续可视化
        state_df = pd.DataFrame({
            'dimension': state_labels * len(self.episodes_stats),
            'episode': np.repeat(range(len(self.episodes_stats)), len(state_labels)),
            'mean': state_means.flatten(),
            'min': state_mins.flatten(),
            'max': state_maxs.flatten()
        })
        
        action_df = pd.DataFrame({
            'dimension': state_labels * len(self.episodes_stats),
            'episode': np.repeat(range(len(self.episodes_stats)), len(state_labels)),
            'mean': action_means.flatten(),
            'min': action_mins.flatten(),
            'max': action_maxs.flatten()
        })
        
        return state_df, action_df
    
    def load_sample_episodes(self, num_episodes: int = 5) -> List[pd.DataFrame]:
        """加载样本episodes数据"""
        print(f"\n加载前 {num_episodes} 个episodes...")
        
        episodes_data = []
        for i in range(min(num_episodes, self.info['total_episodes'])):
            episode_path = self.data_path / f"chunk-000/episode_{i:06d}.parquet"
            if episode_path.exists():
                df = pq.read_table(episode_path).to_pandas()
                episodes_data.append(df)
                print(f"  Episode {i}: {len(df)} 帧")
            else:
                print(f"  Episode {i}: 文件不存在")
        
        return episodes_data
    
    def visualize_episode_lengths(self, episode_df: pd.DataFrame, output_path: str = None):
        """可视化Episode长度分布"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 直方图
        axes[0].hist(episode_df['length'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Episode Length (frames)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Episode Length Distribution')
        axes[0].axvline(episode_df['length'].mean(), color='r', linestyle='--', 
                       label=f'Mean: {episode_df["length"].mean():.1f}')
        axes[0].axvline(episode_df['length'].median(), color='g', linestyle='--', 
                       label=f'Median: {episode_df["length"].median():.1f}')
        axes[0].legend()
        
        # Episode长度序列图
        axes[1].plot(episode_df['episode_index'], episode_df['length'], marker='o', 
                    markersize=2, linewidth=0.5)
        axes[1].set_xlabel('Episode Index')
        axes[1].set_ylabel('Episode Length (frames)')
        axes[1].set_title('Episode Length Sequence')
        axes[1].axhline(episode_df['length'].mean(), color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存图表: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_state_distribution(self, state_df: pd.DataFrame, output_path: str = None):
        """可视化State分布"""
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        axes = axes.flatten()
        
        dimensions = state_df['dimension'].unique()
        
        for i, dim in enumerate(dimensions):
            dim_data = state_df[state_df['dimension'] == dim]
            
            ax = axes[i]
            ax.plot(dim_data['episode'], dim_data['mean'], label='Mean', alpha=0.7)
            ax.fill_between(dim_data['episode'], dim_data['min'], dim_data['max'], 
                           alpha=0.3, label='Min-Max Range')
            ax.set_xlabel('Episode Index')
            ax.set_ylabel(f'{dim} Value')
            ax.set_title(f'State: {dim}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        if len(dimensions) < len(axes):
            for i in range(len(dimensions), len(axes)):
                axes[i].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存图表: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_action_distribution(self, action_df: pd.DataFrame, output_path: str = None):
        """可视化Action分布"""
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        axes = axes.flatten()
        
        dimensions = action_df['dimension'].unique()
        
        for i, dim in enumerate(dimensions):
            dim_data = action_df[action_df['dimension'] == dim]
            
            ax = axes[i]
            ax.plot(dim_data['episode'], dim_data['mean'], label='Mean', alpha=0.7)
            ax.fill_between(dim_data['episode'], dim_data['min'], dim_data['max'], 
                           alpha=0.3, label='Min-Max Range')
            ax.set_xlabel('Episode Index')
            ax.set_ylabel(f'{dim} Value')
            ax.set_title(f'Action: {dim}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        if len(dimensions) < len(axes):
            for i in range(len(dimensions), len(axes)):
                axes[i].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存图表: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_trajectory_samples(self, episodes_data: List[pd.DataFrame], 
                                    output_path: str = None):
        """可视化轨迹样本"""
        fig = plt.figure(figsize=(16, 12))
        
        # 3D轨迹图
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        start_plotted = False
        end_plotted = False
        for i, df in enumerate(episodes_data[:10]):  # 最多显示10条轨迹
            if 'state' in df.columns:
                states = np.stack(df['state'].values)
                # 绘制轨迹
                ax1.plot(states[:, 0], states[:, 1], states[:, 2], 
                        label=f'Ep {i}', alpha=0.7, linewidth=1)
                # 标记起始点（绿色）
                ax1.scatter(states[0, 0], states[0, 1], states[0, 2], 
                           c='green', s=80, marker='o', edgecolors='darkgreen', 
                           linewidths=1.5, zorder=10,
                           label='Start' if not start_plotted else '')
                # 标记结束点（红色）
                ax1.scatter(states[-1, 0], states[-1, 1], states[-1, 2], 
                           c='red', s=80, marker='s', edgecolors='darkred', 
                           linewidths=1.5, zorder=10,
                           label='End' if not end_plotted else '')
                start_plotted = True
                end_plotted = True
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_zlabel('Z Position')
        ax1.set_title('3D End-Effector Trajectories')
        ax1.legend(fontsize=8, loc='upper right')
        
        # X-Y平面投影
        ax2 = fig.add_subplot(2, 2, 2)
        start_plotted = False
        end_plotted = False
        for i, df in enumerate(episodes_data[:10]):
            if 'state' in df.columns:
                states = np.stack(df['state'].values)
                # 绘制轨迹
                ax2.plot(states[:, 0], states[:, 1], label=f'Ep {i}', alpha=0.7, linewidth=1)
                # 标记起始点（绿色圆点）
                ax2.scatter(states[0, 0], states[0, 1], 
                           c='green', s=100, marker='o', edgecolors='darkgreen', 
                           linewidths=2, zorder=10,
                           label='Start' if not start_plotted else '')
                # 标记结束点（红色方块）
                ax2.scatter(states[-1, 0], states[-1, 1], 
                           c='red', s=100, marker='s', edgecolors='darkred', 
                           linewidths=2, zorder=10,
                           label='End' if not end_plotted else '')
                start_plotted = True
                end_plotted = True
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('XY Plane Projection')
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Gripper宽度随时间变化
        ax3 = fig.add_subplot(2, 2, 3)
        for i, df in enumerate(episodes_data[:5]):
            if 'state' in df.columns:
                states = np.stack(df['state'].values)
                time = np.arange(len(states)) / 30.0  # 假设30 FPS
                ax3.plot(time, states[:, 6], label=f'Ep {i}', alpha=0.7, linewidth=1)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Gripper Width')
        ax3.set_title('Gripper Width Over Time')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Action幅值统计
        ax4 = fig.add_subplot(2, 2, 4)
        action_norms = []
        for df in episodes_data:
            if 'actions' in df.columns:
                actions = np.stack(df['actions'].values)
                norms = np.linalg.norm(actions[:, :6], axis=1)  # 不包括gripper
                action_norms.extend(norms)
        ax4.hist(action_norms, bins=50, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Action Norm (excluding gripper)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Action Magnitude Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存图表: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def check_data_quality(self):
        """检查数据质量"""
        print("\n" + "="*60)
        print("数据质量检查")
        print("="*60)
        
        issues = []
        
        # 检查episode长度异常
        lengths = [ep['stats']['actions']['count'][0] for ep in self.episodes_stats]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        for i, length in enumerate(lengths):
            if abs(length - mean_length) > 3 * std_length:
                issues.append(f"Episode {i}: 长度异常 ({length} 帧, "
                            f"偏离均值 {abs(length - mean_length):.1f} 帧)")
        
        # 检查gripper范围
        for ep in self.episodes_stats:
            ep_idx = ep['episode_index']
            gripper_min = ep['stats']['state']['min'][6]
            gripper_max = ep['stats']['state']['max'][6]
            
            if gripper_min < 0 or gripper_max > 0.1:
                issues.append(f"Episode {ep_idx}: Gripper值异常 "
                            f"(范围: [{gripper_min:.4f}, {gripper_max:.4f}])")
        
        if issues:
            print(f"\n发现 {len(issues)} 个潜在问题:")
            for issue in issues[:20]:  # 只显示前20个
                print(f"  - {issue}")
            if len(issues) > 20:
                print(f"  ... 还有 {len(issues) - 20} 个问题")
        else:
            print("\n✓ 未发现明显的数据质量问题")
    
    def generate_report(self, output_dir: str = None):
        """生成完整分析报告"""
        if output_dir is None:
            output_dir = self.dataset_path / "analysis_report"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"\n正在生成分析报告到: {output_dir}")
        
        # 基本信息
        self.print_basic_info()
        self.print_features_info()
        
        # Episode统计
        episode_df = self.analyze_episode_statistics()
        
        # 归一化统计
        self.analyze_normalization_stats()
        
        # 状态和动作分布
        state_df, action_df = self.analyze_state_action_distribution()
        
        # 加载样本数据
        episodes_data = self.load_sample_episodes(num_episodes=self.num_samples)
        
        # 数据质量检查
        self.check_data_quality()
        
        # 生成可视化
        print("\n正在生成可视化图表...")
        self.visualize_episode_lengths(episode_df, 
                                      output_dir / "episode_lengths.png")
        self.visualize_state_distribution(state_df, 
                                         output_dir / "state_distribution.png")
        self.visualize_action_distribution(action_df, 
                                          output_dir / "action_distribution.png")
        
        if episodes_data:
            self.visualize_trajectory_samples(episodes_data, 
                                            output_dir / "trajectory_samples.png")
        
        # 保存统计数据
        episode_df.to_csv(output_dir / "episode_stats.csv", index=False)
        state_df.to_csv(output_dir / "state_stats.csv", index=False)
        action_df.to_csv(output_dir / "action_stats.csv", index=False)
        
        print(f"\n✓ 分析报告生成完成!")
        print(f"  报告目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="分析 UMI LeRobot 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析数据集并生成报告
  python analyze_dataset.py /data/umi_lerobot_dataset_v3
  
  # 指定输出目录
  python analyze_dataset.py /data/umi_lerobot_dataset_v3 --output ./my_report
  
  # 只显示基本信息
  python analyze_dataset.py /data/umi_lerobot_dataset_v3 --info-only
        """
    )
    
    parser.add_argument(
        'dataset_path',
        type=str,
        help='数据集根目录路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='报告输出目录 (默认: <dataset_path>/analysis_report)'
    )
    
    parser.add_argument(
        '--info-only',
        action='store_true',
        help='只显示基本信息，不生成完整报告'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='加载用于可视化的样本episode数量 (默认: 10)'
    )
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = DatasetAnalyzer(args.dataset_path, args.num_samples)
    
    if args.info_only:
        # 只显示基本信息
        analyzer.print_basic_info()
        analyzer.print_features_info()
    else:
        # 生成完整报告
        analyzer.generate_report(output_dir=args.output)


if __name__ == "__main__":
    main()

