#!/usr/bin/env python3
"""
3D轨迹可视化工具 - 支持交互式查看zarr数据集的机器人末端执行器轨迹

功能：
- 加载zarr格式数据集
- 提取并可视化3D轨迹（end-effector position）
- 支持交互式拖动、旋转、缩放
- 可选择单个或多个episodes进行展示
- 使用颜色渐变显示时间序列
"""

import argparse
from pathlib import Path
import numpy as np
import zarr
import plotly.graph_objects as go
import logging
import cv2

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
        
        # 直接加载zarr目录
        self.zarr_root = zarr.open(str("/root/openpi-umi/temp_test"), mode='r')
        
        logging.info(f"数据集加载成功！包含的键: {list(self.zarr_root.keys())}")
        
        # 加载meta信息以获取episode边界
        self._load_episode_boundaries()
    
    def _load_episode_boundaries(self):
        """从meta中加载每个episode的起始和结束索引"""
        if 'meta' not in self.zarr_root:
            logging.warning("未找到meta信息，尝试从episode_ends_index推断...")
            # 尝试其他可能的键
            if 'episode_ends' in self.zarr_root:
                episode_ends = np.array(self.zarr_root['episode_ends'])
                self.episode_ends = episode_ends.tolist()
                self.episode_starts = [0] + self.episode_ends[:-1]
                logging.info(f"从episode_ends加载了 {len(self.episode_ends)} 个episodes")
            else:
                raise ValueError("无法找到episode边界信息！请检查数据集结构。")
            return
        
        meta = self.zarr_root['meta']
        logging.info(f"Meta 包含的键: {list(meta.keys())}")
        
        # 尝试不同的可能键名
        possible_keys = ['episode_ends', 'episode_data_index', 'episodes']
        
        for key in possible_keys:
            if key in meta:
                if key == 'episode_ends':
                    episode_ends = np.array(meta[key])
                    self.episode_ends = episode_ends.tolist()
                    # 起始索引是前一个episode的结束索引
                    self.episode_starts = [0] + self.episode_ends[:-1]
                    logging.info(f"从 meta/{key} 加载了 {len(self.episode_ends)} 个episodes")
                    return
                elif key == 'episode_data_index':
                    # LeRobot格式: episode_data_index['from'] 和 episode_data_index['to']
                    index = meta[key]
                    if 'from' in index and 'to' in index:
                        self.episode_starts = np.array(index['from']).tolist()
                        self.episode_ends = np.array(index['to']).tolist()
                        logging.info(f"从 meta/{key} 加载了 {len(self.episode_starts)} 个episodes")
                        return
        
        raise ValueError("无法找到episode边界信息！请检查meta结构。")
    
    def get_episode_count(self) -> int:
        """获取episode数量"""
        return len(self.episode_ends)
    
    def get_episode_data(self, episode_idx: int, load_images: bool = False) -> dict:
        """获取单个episode的数据
        
        使用meta中的起始和结束索引从连续数据中切片获取
        
        Args:
            episode_idx: Episode索引
            load_images: 是否加载相机图像
        
        Returns:
            dict: {
                'eef_pos': (T, 3) - end-effector position
                'eef_rot': (T, 3 or 4) - end-effector rotation
                'gripper': (T, 1) - gripper state
                'images': (T, H, W, 3) - camera images (if load_images=True)
            }
        """
        if 'data' not in self.zarr_root:
            raise ValueError("数据集中没有'data'键")
        
        # 获取该episode的起始和结束索引
        start_idx = self.episode_starts[episode_idx]
        end_idx = self.episode_ends[episode_idx]
        
        logging.info(f"Episode {episode_idx}: 起始索引={start_idx}, 结束索引={end_idx}, 长度={end_idx - start_idx}")
        
        data_group = self.zarr_root['data']
        result = {}
        
        # 尝试不同的键名来找到位置数据
        possible_pos_keys = ['robot0_eef_pos', 'eef_pos', 'state', 'observation.robot0_eef_pos']
        possible_rot_keys = ['robot0_eef_rot_axis_angle', 'eef_rot', 'rotation', 'observation.robot0_eef_rot_axis_angle']
        possible_gripper_keys = ['robot0_gripper_width', 'gripper_width', 'gripper', 'observation.robot0_gripper_width']
        possible_image_keys = ['observation.images.top', 'camera0_rgb', 'images', 'observation.camera0_rgb']
        
        # 查找位置数据（使用切片）
        for key in possible_pos_keys:
            if key in data_group:
                # 使用起始和结束索引切片数据
                data = np.array(data_group[key][start_idx:end_idx])
                logging.info(f"找到位置数据 {key}, shape: {data.shape}")
                
                if data.shape[-1] >= 3:
                    result['eef_pos'] = data[..., :3]  # 只取前3维 (x, y, z)
                else:
                    result['eef_pos'] = data
                logging.info(f"处理后位置数据 shape: {result['eef_pos'].shape}")
                break
        
        # 查找旋转数据
        for key in possible_rot_keys:
            if key in data_group:
                result['eef_rot'] = np.array(data_group[key][start_idx:end_idx])
                logging.info(f"找到旋转数据: {key}, shape: {result['eef_rot'].shape}")
                break
        
        # 查找夹爪数据
        for key in possible_gripper_keys:
            if key in data_group:
                result['gripper'] = np.array(data_group[key][start_idx:end_idx])
                logging.info(f"找到夹爪数据: {key}, shape: {result['gripper'].shape}")
                break
        
        # 查找相机图像
        if load_images:
            for key in possible_image_keys:
                if key in data_group:
                    images = np.array(data_group[key][start_idx:end_idx])
                    result['images'] = images
                    logging.info(f"找到相机图像: {key}, shape: {images.shape}")
                    break
            
            if 'images' not in result:
                logging.warning(f"未找到相机图像数据。可用的键: {list(data_group.keys())}")
        
        if 'eef_pos' not in result:
            logging.warning(f"数据集中可用的键: {list(data_group.keys())}")
            raise ValueError("无法找到位置数据！请检查数据集格式。")
        
        return result


class Trajectory3DVisualizer:
    """3D轨迹可视化器"""
    
    def __init__(self):
        self.fig = None
    
    def visualize_single_episode(self, episode_data: dict, episode_idx: int, 
                                save_html: bool = True, output_path: str = None):
        """可视化单个episode的3D轨迹"""
        
        eef_pos = episode_data['eef_pos']
        T = len(eef_pos)
        
        # 数据验证和调试
        logging.info(f"轨迹数据 shape: {eef_pos.shape}")
        logging.info(f"X范围: [{eef_pos[:, 0].min():.4f}, {eef_pos[:, 0].max():.4f}]")
        logging.info(f"Y范围: [{eef_pos[:, 1].min():.4f}, {eef_pos[:, 1].max():.4f}]")
        logging.info(f"Z范围: [{eef_pos[:, 2].min():.4f}, {eef_pos[:, 2].max():.4f}]")
        logging.info(f"前5个点:\n{eef_pos[:5]}")
        
        # 检查是否有NaN或Inf
        if np.any(np.isnan(eef_pos)):
            logging.warning("警告: 数据中包含NaN值！")
        if np.any(np.isinf(eef_pos)):
            logging.warning("警告: 数据中包含Inf值！")
        
        # 创建颜色渐变（从蓝到红，表示时间流逝）
        colors = np.linspace(0, 1, T)
        
        # 创建3D轨迹图
        self.fig = go.Figure()
        
        # 添加Z=0参考平面（XY平面作为地面）- 增强显示
        x_range = [eef_pos[:, 0].min() - 0.1, eef_pos[:, 0].max() + 0.1]
        y_range = [eef_pos[:, 1].min() - 0.1, eef_pos[:, 1].max() + 0.1]
        xx, yy = np.meshgrid(
            np.linspace(x_range[0], x_range[1], 20),
            np.linspace(y_range[0], y_range[1], 20)
        )
        zz = np.zeros_like(xx)
        
        self.fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, 'rgb(220,220,220)'], [1, 'rgb(200,200,200)']],
            showscale=False,
            opacity=0.4,
            name='Z=0 Ground Plane',
            hovertemplate='Ground Plane (Z=0)<extra></extra>',
            contours=dict(
                x=dict(show=True, color='gray', width=1),
                y=dict(show=True, color='gray', width=1)
            )
        ))
        
        # 标记原点
        self.fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=15, color='black', symbol='x', line=dict(width=2)),
            name='Origin (0,0,0)',
            hovertemplate='<b>Origin</b><br>X: 0.0000<br>Y: 0.0000<br>Z: 0.0000 [Height]<extra></extra>'
        ))
        
        # 添加轨迹线 - 标准坐标系，Z轴向上
        self.fig.add_trace(go.Scatter3d(
            x=eef_pos[:, 0],  # X
            y=eef_pos[:, 1],  # Y
            z=eef_pos[:, 2],  # Z作为高度轴
            mode='lines+markers',
            line=dict(
                color=colors,
                colorscale='Viridis',
                width=4,
                colorbar=dict(title="Time Progress", x=1.1)
            ),
            marker=dict(
                size=3,
                color=colors,
                colorscale='Viridis',
                showscale=False
            ),
            name=f'Episode {episode_idx}',
            hovertemplate=(
                '<b>Step %{text}</b><br>' +
                'X: %{x:.4f}<br>' +
                'Y: %{y:.4f}<br>' +
                'Z: %{z:.4f} [Height]<br>' +
                '<extra></extra>'
            ),
            text=list(range(T))
        ))
        
        # 标记起点和终点 - 标准坐标系
        self.fig.add_trace(go.Scatter3d(
            x=[eef_pos[0, 0]],
            y=[eef_pos[0, 1]],
            z=[eef_pos[0, 2]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='diamond'),
            name='Start',
            hovertemplate='<b>Start Point</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f} [Height]<extra></extra>'
        ))
        
        self.fig.add_trace(go.Scatter3d(
            x=[eef_pos[-1, 0]],
            y=[eef_pos[-1, 1]],
            z=[eef_pos[-1, 2]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            name='End',
            hovertemplate='<b>End Point</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f} [Height]<extra></extra>'
        ))
        
        # 设置布局
        self.fig.update_layout(
            title=dict(
                text=f'3D Trajectory Visualization - Episode {episode_idx}<br>' +
                     f'<sub>Total Steps: {T} | Duration: {T/30:.2f}s (assuming 30Hz)</sub>',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='X Position (m)',
                yaxis_title='Y Position (m)',
                zaxis_title='Z Position (m) [Height]',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1200,
            height=800,
            showlegend=True,
            hovermode='closest'
        )
        
        # 保存为HTML文件（可交互）
        if save_html:
            if output_path is None:
                output_path = f'trajectory_3d_episode_{episode_idx}.html'
            
            # 使用完整的HTML选项，确保可以离线查看
            # 'cdn' - 需要网络连接，文件小
            # True - 嵌入plotly.js，文件大但可离线查看
            self.fig.write_html(
                output_path,
                include_plotlyjs=True,  # 改为True可离线查看（文件会变大~3MB）
                full_html=True,
                include_mathjax=False
            )
            logging.info(f"✓ 已保存交互式3D图表到: {output_path}")
            logging.info(f"  用浏览器打开该文件即可交互查看（拖动、旋转、缩放）")
            logging.info(f"  文件大小: {Path(output_path).stat().st_size / 1024:.2f} KB")
        
        return self.fig
    
    def visualize_multiple_episodes(self, episodes_data: list, episode_indices: list,
                                   save_html: bool = True, output_path: str = None):
        """可视化多个episodes的3D轨迹"""
        
        self.fig = go.Figure()
        
        # 计算所有episodes的范围
        all_eef = np.vstack([ep['eef_pos'] for ep in episodes_data])
        x_range = [all_eef[:, 0].min() - 0.05, all_eef[:, 0].max() + 0.05]
        y_range = [all_eef[:, 1].min() - 0.05, all_eef[:, 1].max() + 0.05]
        
        # 添加Z=0参考平面（XY平面作为地面）- 增强显示
        xx, yy = np.meshgrid(
            np.linspace(x_range[0], x_range[1], 20),
            np.linspace(y_range[0], y_range[1], 20)
        )
        zz = np.zeros_like(xx)
        
        self.fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, 'rgb(220,220,220)'], [1, 'rgb(200,200,200)']],
            showscale=False,
            opacity=0.4,
            name='Z=0 Ground Plane',
            hovertemplate='Ground Plane (Z=0)<extra></extra>',
            contours=dict(
                x=dict(show=True, color='gray', width=1),
                y=dict(show=True, color='gray', width=1)
            )
        ))
        
        # 标记原点
        self.fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=15, color='black', symbol='x', line=dict(width=2)),
            name='Origin (0,0,0)',
            hovertemplate='<b>Origin</b><br>X: 0.0000<br>Y: 0.0000<br>Z: 0.0000 [Height]<extra></extra>'
        ))
        
        # 为每个episode使用不同颜色
        colors_palette = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        for i, (episode_data, ep_idx) in enumerate(zip(episodes_data, episode_indices)):
            eef_pos = episode_data['eef_pos']
            T = len(eef_pos)
            color = colors_palette[i % len(colors_palette)]
            
            # 添加轨迹 - 标准坐标系，Z轴向上
            self.fig.add_trace(go.Scatter3d(
                x=eef_pos[:, 0],  # X
                y=eef_pos[:, 1],  # Y
                z=eef_pos[:, 2],  # Z作为高度
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=2, color=color),
                name=f'Episode {ep_idx}',
                hovertemplate=(
                    f'<b>Episode {ep_idx} - Step %{{text}}</b><br>' +
                    'X: %{x:.4f}<br>' +
                    'Y: %{y:.4f}<br>' +
                    'Z: %{z:.4f} [Height]<br>' +
                    '<extra></extra>'
                ),
                text=list(range(T))
            ))
            
            # 标记起点 - 标准坐标系
            self.fig.add_trace(go.Scatter3d(
                x=[eef_pos[0, 0]],
                y=[eef_pos[0, 1]],
                z=[eef_pos[0, 2]],
                mode='markers',
                marker=dict(size=10, color=color, symbol='diamond'),
                name=f'Ep{ep_idx} Start',
                showlegend=False,
                hovertemplate=f'<b>Episode {ep_idx} Start</b><br>X: %{{x:.4f}}<br>Y: %{{y:.4f}}<br>Z: %{{z:.4f}} [Height]<extra></extra>'
            ))
        
        # 设置布局
        self.fig.update_layout(
            title=dict(
                text=f'3D Trajectory Comparison - {len(episodes_data)} Episodes',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='X Position (m)',
                yaxis_title='Y Position (m)',
                zaxis_title='Z Position (m) [Height]',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1400,
            height=900,
            showlegend=True,
            hovermode='closest'
        )
        
        # 保存为HTML文件
        if save_html:
            if output_path is None:
                output_path = f'trajectory_3d_multiple_episodes.html'
            
            self.fig.write_html(
                output_path,
                include_plotlyjs=True,  # 嵌入plotly.js，可离线查看
                full_html=True,
                include_mathjax=False
            )
            logging.info(f"✓ 已保存交互式3D图表到: {output_path}")
            logging.info(f"  用浏览器打开该文件即可交互查看")
            logging.info(f"  文件大小: {Path(output_path).stat().st_size / 1024:.2f} KB")
        
        return self.fig
    
    def show(self):
        """在浏览器中显示图表"""
        if self.fig:
            self.fig.show()
    
    def save_camera_images(self, episode_data: dict, episode_idx: int, output_dir: str = None):
        """保存episode对应的相机图像
        
        Args:
            episode_data: Episode数据（需包含'images'键）
            episode_idx: Episode索引
            output_dir: 输出目录，默认为'camera_images_episode_{idx}'
        """
        if 'images' not in episode_data:
            logging.warning(f"Episode {episode_idx} 没有图像数据，跳过保存")
            return
        
        images = episode_data['images']
        T = len(images)
        
        if output_dir is None:
            output_dir = f'camera_images_episode_{episode_idx}'
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"保存 {T} 张相机图像到: {output_path}")
        
        for step, img in enumerate(images):
            # 处理不同的图像格式
            if img.dtype == np.uint8:
                img_to_save = img
            else:
                # 归一化到0-255
                img_to_save = (img * 255).astype(np.uint8)
            
            # 如果是RGB，转换为BGR用于OpenCV保存
            if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3:
                img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
            
            filename = output_path / f"step_{step:04d}.png"
            cv2.imwrite(str(filename), img_to_save)
        
        logging.info(f"✓ 成功保存 {T} 张图像")
        
        # 创建索引文件
        index_file = output_path / "index.txt"
        with open(index_file, 'w') as f:
            f.write(f"Episode {episode_idx} - Camera Images\n")
            f.write(f"Total Steps: {T}\n")
            f.write(f"Image Format: PNG\n\n")
            f.write("Step to Filename Mapping:\n")
            for step in range(T):
                f.write(f"Step {step:4d} -> step_{step:04d}.png\n")
        
        logging.info(f"✓ 创建索引文件: {index_file}")


def main():
    parser = argparse.ArgumentParser(
        description='可视化zarr数据集的3D轨迹（交互式）'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='/root/openpi-umi/data/dataset_ray_cyrus_Mason.zarr.zip',
        help='数据集路径（.zarr 或 .zarr.zip）'
    )
    parser.add_argument(
        '--episodes',
        type=str,
        default='0',
        help='要可视化的episode索引，可以是单个数字、逗号分隔的列表或范围（例如: "0" 或 "0,1,2" 或 "0-5"）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出HTML文件路径'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='不保存HTML文件，只在浏览器中显示'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='在浏览器中自动打开可视化'
    )
    parser.add_argument(
        '--save-images',
        action='store_true',
        help='保存每个step对应的相机图像'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default=None,
        help='相机图像保存目录（默认为camera_images_episode_{idx}）'
    )
    
    args = parser.parse_args()
    
    # 解析episode索引
    episode_indices = []
    if '-' in args.episodes:
        # 范围格式: "0-5"
        start, end = map(int, args.episodes.split('-'))
        episode_indices = list(range(start, end + 1))
    elif ',' in args.episodes:
        # 列表格式: "0,1,2"
        episode_indices = [int(x.strip()) for x in args.episodes.split(',')]
    else:
        # 单个数字: "0"
        episode_indices = [int(args.episodes)]
    
    logging.info("="*70)
    logging.info("3D轨迹可视化工具")
    logging.info("="*70)
    logging.info(f"数据集: {args.dataset}")
    logging.info(f"Episodes: {episode_indices}")
    logging.info("="*70)
    
    # 加载数据集
    loader = ZarrDatasetLoader(args.dataset)
    total_episodes = loader.get_episode_count()
    logging.info(f"数据集包含 {total_episodes} 个episodes")
    
    # 验证episode索引
    for ep_idx in episode_indices:
        if ep_idx >= total_episodes:
            logging.error(f"Episode {ep_idx} 超出范围！数据集只有 {total_episodes} 个episodes（0-{total_episodes-1}）")
            return
    
    # 创建可视化器
    visualizer = Trajectory3DVisualizer()
    
    # 可视化
    if len(episode_indices) == 1:
        # 单个episode
        ep_idx = episode_indices[0]
        logging.info(f"\n加载 Episode {ep_idx}...")
        episode_data = loader.get_episode_data(ep_idx, load_images=args.save_images)
        
        output_path = args.output or f'trajectory_3d_episode_{ep_idx}.html'
        visualizer.visualize_single_episode(
            episode_data, 
            ep_idx,
            save_html=not args.no_save,
            output_path=output_path
        )
        
        # 保存相机图像
        if args.save_images:
            logging.info(f"\n保存相机图像...")
            images_output_dir = args.images_dir or f'camera_images_episode_{ep_idx}'
            visualizer.save_camera_images(episode_data, ep_idx, images_output_dir)
    else:
        # 多个episodes
        logging.info(f"\n加载 {len(episode_indices)} 个episodes...")
        episodes_data = []
        for ep_idx in episode_indices:
            logging.info(f"  加载 Episode {ep_idx}...")
            episodes_data.append(loader.get_episode_data(ep_idx, load_images=args.save_images))
        
        output_path = args.output or 'trajectory_3d_multiple_episodes.html'
        visualizer.visualize_multiple_episodes(
            episodes_data,
            episode_indices,
            save_html=not args.no_save,
            output_path=output_path
        )
        
        # 保存相机图像
        if args.save_images:
            logging.info(f"\n保存相机图像...")
            for episode_data, ep_idx in zip(episodes_data, episode_indices):
                images_output_dir = args.images_dir or f'camera_images_episode_{ep_idx}'
                if args.images_dir:
                    images_output_dir = f"{args.images_dir}_ep{ep_idx}"
                visualizer.save_camera_images(episode_data, ep_idx, images_output_dir)
    
    # 在浏览器中显示
    if args.show:
        logging.info("\n在浏览器中打开可视化...")
        visualizer.show()
    
    logging.info("\n" + "="*70)
    logging.info("✓ 完成！")
    logging.info("="*70)
    logging.info("\n使用提示:")
    logging.info("  - 拖动鼠标: 旋转视角")
    logging.info("  - 滚轮: 缩放")
    logging.info("  - 右键拖动: 平移")
    logging.info("  - 悬停在轨迹上可查看详细信息")
    logging.info("  - 点击图例可显示/隐藏特定轨迹")
    if args.save_images:
        logging.info(f"\n相机图像已保存:")
        for ep_idx in episode_indices:
            img_dir = args.images_dir or f'camera_images_episode_{ep_idx}'
            if len(episode_indices) > 1 and args.images_dir:
                img_dir = f"{args.images_dir}_ep{ep_idx}"
            logging.info(f"  Episode {ep_idx}: {img_dir}/")
    logging.info("="*70)


if __name__ == "__main__":
    main()

