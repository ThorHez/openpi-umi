#!/usr/bin/env python3
"""
3D轨迹可视化工具 - 生成静态图片（PNG格式）

功能：
- 加载zarr格式数据集
- 生成3D轨迹的静态图片
- 支持多角度视图
- 使用matplotlib生成高质量图片
"""

import argparse
from pathlib import Path
import numpy as np
import zarr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ZarrDatasetLoader:
    """加载和解析zarr数据集"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.zarr_root = None
        self.episode_starts = []
        self.episode_ends = []
        self._load_dataset()
    
    def _load_dataset(self):
        """加载zarr数据集"""
        logging.info(f"加载数据集: {self.dataset_path}")
        self.zarr_root = zarr.open(str("/root/openpi-umi/temp_test"), mode='r')
        logging.info(f"数据集加载成功！包含的键: {list(self.zarr_root.keys())}")
        self._load_episode_boundaries()
    
    def _load_episode_boundaries(self):
        """从meta中加载每个episode的起始和结束索引"""
        if 'meta' not in self.zarr_root:
            if 'episode_ends' in self.zarr_root:
                episode_ends = np.array(self.zarr_root['episode_ends'])
                self.episode_ends = episode_ends.tolist()
                self.episode_starts = [0] + self.episode_ends[:-1]
                logging.info(f"从episode_ends加载了 {len(self.episode_ends)} 个episodes")
            else:
                raise ValueError("无法找到episode边界信息！")
            return
        
        meta = self.zarr_root['meta']
        if 'episode_ends' in meta:
            episode_ends = np.array(meta['episode_ends'])
            self.episode_ends = episode_ends.tolist()
            self.episode_starts = [0] + self.episode_ends[:-1]
            logging.info(f"从 meta/episode_ends 加载了 {len(self.episode_ends)} 个episodes")
        elif 'episode_data_index' in meta:
            index = meta['episode_data_index']
            self.episode_starts = np.array(index['from']).tolist()
            self.episode_ends = np.array(index['to']).tolist()
            logging.info(f"从 meta/episode_data_index 加载了 {len(self.episode_starts)} 个episodes")
        else:
            raise ValueError("无法找到episode边界信息！")
    
    def get_episode_count(self) -> int:
        """获取episode数量"""
        return len(self.episode_ends)
    
    def get_episode_data(self, episode_idx: int) -> dict:
        """获取单个episode的数据"""
        if 'data' not in self.zarr_root:
            raise ValueError("数据集中没有'data'键")
        
        start_idx = self.episode_starts[episode_idx]
        end_idx = self.episode_ends[episode_idx]
        
        logging.info(f"Episode {episode_idx}: 起始={start_idx}, 结束={end_idx}, 长度={end_idx - start_idx}")
        
        data_group = self.zarr_root['data']
        result = {}
        
        possible_pos_keys = ['robot0_eef_pos', 'eef_pos', 'state']
        
        for key in possible_pos_keys:
            if key in data_group:
                data = np.array(data_group[key][start_idx:end_idx])
                if data.shape[-1] >= 3:
                    result['eef_pos'] = data[..., :3]
                else:
                    result['eef_pos'] = data
                logging.info(f"找到位置数据: {key}, shape: {result['eef_pos'].shape}")
                break
        
        if 'eef_pos' not in result:
            raise ValueError("无法找到位置数据！")
        
        return result


class Trajectory3DImageGenerator:
    """3D轨迹静态图片生成器"""
    
    def generate_single_episode(self, episode_data: dict, episode_idx: int, 
                               output_path: str = None, multi_view: bool = False):
        """生成单个episode的3D轨迹图片"""
        
        eef_pos = episode_data['eef_pos']
        T = len(eef_pos)
        
        logging.info(f"轨迹数据 shape: {eef_pos.shape}")
        logging.info(f"X范围: [{eef_pos[:, 0].min():.4f}, {eef_pos[:, 0].max():.4f}]")
        logging.info(f"Y范围: [{eef_pos[:, 1].min():.4f}, {eef_pos[:, 1].max():.4f}]")
        logging.info(f"Z范围: [{eef_pos[:, 2].min():.4f}, {eef_pos[:, 2].max():.4f}]")
        
        if output_path is None:
            output_path = f'trajectory_3d_episode_{episode_idx}.png'
        
        if multi_view:
            self._generate_multi_view(eef_pos, episode_idx, output_path, T)
        else:
            self._generate_single_view(eef_pos, episode_idx, output_path, T)
        
        logging.info(f"✓ 已保存3D轨迹图片到: {output_path}")
    
    def _generate_single_view(self, eef_pos, episode_idx, output_path, T):
        """生成单视角图片"""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建颜色映射（时间渐变）
        colors = plt.cm.viridis(np.linspace(0, 1, T))
        
        # 绘制轨迹线 - 标准坐标系，Z轴向上
        for i in range(T - 1):
            ax.plot(eef_pos[i:i+2, 0],   # X
                   eef_pos[i:i+2, 1],    # Y
                   eef_pos[i:i+2, 2],    # Z作为高度轴
                   color=colors[i], linewidth=2, alpha=0.8)
        
        # 标记原点（坐标系原点）
        ax.scatter(0, 0, 0, color='black', s=300, marker='x', 
                  linewidths=3, label='Origin (0,0,0)', zorder=10)
        
        # 标记起点和终点 - 标准坐标系
        ax.scatter(eef_pos[0, 0], eef_pos[0, 1], eef_pos[0, 2], 
                  color='green', s=200, marker='o', 
                  edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax.scatter(eef_pos[-1, 0], eef_pos[-1, 1], eef_pos[-1, 2], 
                  color='red', s=200, marker='s', 
                  edgecolors='black', linewidths=2, label='End', zorder=5)
        
        # 添加Z=0参考平面（底面/地面） - XY平面
        # 扩大范围使底面更明显
        x_range = [eef_pos[:, 0].min() - 0.1, eef_pos[:, 0].max() + 0.1]
        y_range = [eef_pos[:, 1].min() - 0.1, eef_pos[:, 1].max() + 0.1]
        xx, yy = np.meshgrid(
            np.linspace(x_range[0], x_range[1], 20),
            np.linspace(y_range[0], y_range[1], 20)
        )
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.25, color='lightgray', 
                       edgecolor='gray', linewidth=0.5, label='Z=0 Ground')
        
        # 设置标签和标题 - Z轴是高度
        ax.set_xlabel('X Position (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y Position (m)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z Position (m) [Height]', fontsize=12, labelpad=10)
        ax.set_title(f'3D Trajectory - Episode {episode_idx}\n'
                    f'Total Steps: {T} | Duration: {T/30:.2f}s', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 设置视角 - 从上往下俯视，Z轴朝上作为高度
        ax.view_init(elev=25, azim=45)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=0, vmax=T))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Time Step', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_multi_view(self, eef_pos, episode_idx, output_path, T):
        """生成多视角图片（2x2布局）"""
        fig = plt.figure(figsize=(16, 12))
        
        # 4个不同的视角 - 以Z面为底面
        views = [
            (25, 45, 'View 1: Perspective (NE)'),
            (25, 135, 'View 2: Perspective (SE)'),
            (70, 45, 'View 3: Top View'),
            (10, 90, 'View 4: Side View')
        ]
        
        colors = plt.cm.viridis(np.linspace(0, 1, T))
        
        for idx, (elev, azim, title) in enumerate(views, 1):
            ax = fig.add_subplot(2, 2, idx, projection='3d')
            
            # 绘制轨迹 - 标准坐标系，Z轴向上
            for i in range(T - 1):
                ax.plot(eef_pos[i:i+2, 0],   # X
                       eef_pos[i:i+2, 1],    # Y
                       eef_pos[i:i+2, 2],    # Z作为高度
                       color=colors[i], linewidth=1.5, alpha=0.7)
            
            # 原点（只在第一个子图显示label）
            if idx == 1:
                ax.scatter(0, 0, 0, color='black', s=150, marker='x', 
                          linewidths=2, label='Origin', zorder=10)
            else:
                ax.scatter(0, 0, 0, color='black', s=150, marker='x', 
                          linewidths=2, zorder=10)
            
            # 起点和终点 - 标准坐标系
            ax.scatter(eef_pos[0, 0], eef_pos[0, 1], eef_pos[0, 2], 
                      color='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter(eef_pos[-1, 0], eef_pos[-1, 1], eef_pos[-1, 2], 
                      color='red', s=100, marker='s', label='End', zorder=5)
            
            # 添加Z=0参考平面（XY平面作为地面）
            x_range = [eef_pos[:, 0].min() - 0.1, eef_pos[:, 0].max() + 0.1]
            y_range = [eef_pos[:, 1].min() - 0.1, eef_pos[:, 1].max() + 0.1]
            xx, yy = np.meshgrid(
                np.linspace(x_range[0], x_range[1], 20),
                np.linspace(y_range[0], y_range[1], 20)
            )
            zz = np.zeros_like(xx)
            ax.plot_surface(xx, yy, zz, alpha=0.2, color='lightgray', 
                           edgecolor='gray', linewidth=0.3)
            
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_zlabel('Z (m) [Height]', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.view_init(elev=elev, azim=azim)
            ax.grid(True, alpha=0.3)
            if idx == 1:
                ax.legend(loc='upper right', fontsize=9)
        
        fig.suptitle(f'3D Trajectory - Episode {episode_idx} (Multiple Views)\n'
                    f'Total Steps: {T} | Duration: {T/30:.2f}s', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_multiple_episodes(self, episodes_data: list, episode_indices: list,
                                  output_path: str = None):
        """生成多个episodes对比图片"""
        
        if output_path is None:
            output_path = f'trajectory_3d_multiple_episodes.png'
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 标记原点
        ax.scatter(0, 0, 0, color='black', s=300, marker='x', 
                  linewidths=3, label='Origin (0,0,0)', zorder=10)
        
        # 颜色列表
        colors_list = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown']
        
        for i, (episode_data, ep_idx) in enumerate(zip(episodes_data, episode_indices)):
            eef_pos = episode_data['eef_pos']
            color = colors_list[i % len(colors_list)]
            
            # 绘制轨迹 - 标准坐标系，Z轴向上
            ax.plot(eef_pos[:, 0], eef_pos[:, 1], eef_pos[:, 2],
                   color=color, linewidth=2, alpha=0.7, label=f'Episode {ep_idx}')
            
            # 起点 - 标准坐标系
            ax.scatter(eef_pos[0, 0], eef_pos[0, 1], eef_pos[0, 2],
                      color=color, s=150, marker='o', edgecolors='black', 
                      linewidths=2, zorder=5)
        
        # 添加Z=0参考平面（XY平面作为地面）
        all_eef = np.vstack([ep['eef_pos'] for ep in episodes_data])
        x_range = [all_eef[:, 0].min() - 0.1, all_eef[:, 0].max() + 0.1]
        y_range = [all_eef[:, 1].min() - 0.1, all_eef[:, 1].max() + 0.1]
        xx, yy = np.meshgrid(
            np.linspace(x_range[0], x_range[1], 20),
            np.linspace(y_range[0], y_range[1], 20)
        )
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.25, color='lightgray', 
                       edgecolor='gray', linewidth=0.5)
        
        ax.set_xlabel('X Position (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y Position (m)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z Position (m) [Height]', fontsize=12, labelpad=10)
        ax.set_title(f'3D Trajectory Comparison - {len(episodes_data)} Episodes', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✓ 已保存多episode对比图片到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='生成zarr数据集的3D轨迹静态图片'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='/root/openpi-umi/data/dataset_ray_cyrus_Mason.zarr.zip',
        help='数据集路径'
    )
    parser.add_argument(
        '--episodes',
        type=str,
        default='0',
        help='要可视化的episode索引（例如: "0" 或 "0,1,2" 或 "0-5"）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出图片路径'
    )
    parser.add_argument(
        '--multi-view',
        action='store_true',
        help='生成多视角图片（仅对单个episode有效）'
    )
    
    args = parser.parse_args()
    
    # 解析episode索引
    episode_indices = []
    if '-' in args.episodes:
        start, end = map(int, args.episodes.split('-'))
        episode_indices = list(range(start, end + 1))
    elif ',' in args.episodes:
        episode_indices = [int(x.strip()) for x in args.episodes.split(',')]
    else:
        episode_indices = [int(args.episodes)]
    
    logging.info("="*70)
    logging.info("3D轨迹图片生成工具")
    logging.info("="*70)
    logging.info(f"Episodes: {episode_indices}")
    logging.info(f"多视角: {args.multi_view}")
    logging.info("="*70)
    
    # 加载数据集
    loader = ZarrDatasetLoader(args.dataset)
    total_episodes = loader.get_episode_count()
    logging.info(f"数据集包含 {total_episodes} 个episodes")
    
    # 验证索引
    for ep_idx in episode_indices:
        if ep_idx >= total_episodes:
            logging.error(f"Episode {ep_idx} 超出范围！")
            return
    
    # 创建生成器
    generator = Trajectory3DImageGenerator()
    
    # 生成图片
    if len(episode_indices) == 1:
        ep_idx = episode_indices[0]
        logging.info(f"\n加载 Episode {ep_idx}...")
        episode_data = loader.get_episode_data(ep_idx)
        
        output_path = args.output or (
            f'trajectory_3d_episode_{ep_idx}_multiview.png' if args.multi_view 
            else f'trajectory_3d_episode_{ep_idx}.png'
        )
        generator.generate_single_episode(
            episode_data, 
            ep_idx,
            output_path=output_path,
            multi_view=args.multi_view
        )
    else:
        logging.info(f"\n加载 {len(episode_indices)} 个episodes...")
        episodes_data = []
        for ep_idx in episode_indices:
            logging.info(f"  加载 Episode {ep_idx}...")
            episodes_data.append(loader.get_episode_data(ep_idx))
        
        output_path = args.output or 'trajectory_3d_multiple_episodes.png'
        generator.generate_multiple_episodes(episodes_data, episode_indices, output_path)
    
    logging.info("\n" + "="*70)
    logging.info("✓ 完成！")
    logging.info("="*70)


if __name__ == "__main__":
    main()

