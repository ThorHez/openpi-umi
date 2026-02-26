#!/usr/bin/env python3
"""
本地推理测试脚本 - 使用本地数据集测试模型推理性能
基于 serve_policy.py 的结构，用于评估训练好的模型在本地数据集上的表现
"""

import dataclasses
import enum
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import tyro
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

import cv2
import numpy as np

sns.set_style("whitegrid")


class EnvMode(enum.Enum):
    """支持的环境类型"""
    UMI = "umi"
    ALOHA = "aloha"
    DROID = "droid"


@dataclasses.dataclass
class Checkpoint:
    """从训练好的checkpoint加载策略"""
    # 训练配置名称 (例如: "pi0_umi")
    config: str
    # Checkpoint目录 (例如: "checkpoints/pi0_umi/exp/10000")
    dir: str


@dataclasses.dataclass
class Args:
    """本地推理测试参数"""
    
    # 数据集路径
    dataset_path: str = "data/umi_lerobot_dataset_v3"
    
    # Checkpoint配置
    checkpoint_config: str = "pi0_umi"
    checkpoint_dir: str = "checkpoints/pi0_umi/exp/10000"
    
    # 测试参数
    num_episodes: int = 10  # 测试的episode数量，-1表示测试全部
    start_episode: int = 0  # 起始episode索引
    
    # 环境类型
    env: EnvMode = EnvMode.UMI
    
    # 默认prompt（如果数据中没有prompt）
    default_prompt: str = "pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box"
    
    # 输出目录
    output_dir: str = "inference_results"
    
    # 是否保存详细结果
    save_detailed: bool = True
    
    # 是否生成可视化
    visualize: bool = True
    
    # 是否保存观测图像
    save_images: bool = True


class LocalInferenceEvaluator:
    """本地推理评估器"""
    
    def __init__(self, policy: _policy.Policy, dataset_path: Path, output_dir: Path, save_images: bool = True):
        self.policy = policy
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        
        # 创建图像保存目录
        if self.save_images:
            self.images_dir = self.output_dir / "observation_images"
            self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'episode_errors': [],
            'episode_indices': [],
            'action_predictions': [],
            'action_ground_truth': [],
            'state_sequences': [],
        }
        
        logging.info(f"初始化评估器 - 数据集: {dataset_path}, 输出: {output_dir}")
    
    def load_episode(self, episode_idx: int) -> dict | None:
        """加载一个episode的数据"""
        episode_path = self.dataset_path / "data" / f"chunk-000" / f"episode_{episode_idx:06d}.parquet"
        
        if not episode_path.exists():
            logging.warning(f"Episode {episode_idx} 不存在: {episode_path}")
            return None
        
        try:
            df = pq.read_table(episode_path).to_pandas()
            return df
        except Exception as e:
            logging.error(f"加载 Episode {episode_idx} 失败: {e}")
            return None
    
    def prepare_observation(self, row: dict) -> dict:
        """准备观测数据用于推理"""
        # 根据环境类型准备数据
        img_bytes = row["observation.camera0_rgb"]["bytes"]       # 你的 DataFrame 中的数据
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # (H, W, C)，BGR 格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        obs = {
            "state": np.array(row["state"]),
            "camera0_rgb": img,
        }
        return obs
    
    def evaluate_episode(self, episode_idx: int) -> dict | None:
        """评估单个episode"""
        df = self.load_episode(episode_idx)
        if df is None:
            return None
        
        logging.info(f"评估 Episode {episode_idx} ({len(df)} 帧)")
        
        # 为该episode创建图像子目录
        if self.save_images:
            episode_img_dir = self.images_dir / f"episode_{episode_idx:06d}"
            episode_img_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储预测和真实值
        predictions = []
        ground_truth = []
        states = []
        
        # 遍历episode中的每一帧
        for idx, row in df.iterrows():
            print(f"inference episode {episode_idx} frame {idx}")
            # 准备观测
            obs = self.prepare_observation(row)
            
            # 保存观测图像
            if self.save_images:
                img_rgb = obs["camera0_rgb"]
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                img_path = episode_img_dir / f"frame_{idx:06d}.png"
                cv2.imwrite(str(img_path), img_bgr)
            
            # 模型推理
            action = self.policy.infer(obs)
            predicted_action = action["actions"]
            
            # 如果模型返回多步预测（action chunking），只取第一步
            if predicted_action.ndim == 2:
                predicted_action = predicted_action[0]  # 取第一个时间步 (10, 7) -> (7,)
            
            # 获取真实动作
            true_action = np.array(row["actions"])
            
            predictions.append(predicted_action)
            ground_truth.append(true_action)
            states.append(np.array(row["state"]))
        
        if not predictions:
            logging.warning(f"Episode {episode_idx} 没有成功的推理")
            return None
        
        # 转换为numpy数组
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        states = np.array(states)
        
        # 计算误差
        mse = np.mean((predictions - ground_truth) ** 2, axis=0)  # 每个维度的MSE
        mae = np.mean(np.abs(predictions - ground_truth), axis=0)  # 每个维度的MAE
        overall_mse = np.mean((predictions - ground_truth) ** 2)
        overall_mae = np.mean(np.abs(predictions - ground_truth))
        
        # 计算后100帧的误差（如果episode长度足够）
        last_n_frames = 100
        if len(predictions) >= last_n_frames:
            last_100_predictions = predictions[-last_n_frames:]
            last_100_ground_truth = ground_truth[-last_n_frames:]
            
            last_100_mse = np.mean((last_100_predictions - last_100_ground_truth) ** 2, axis=0)
            last_100_mae = np.mean(np.abs(last_100_predictions - last_100_ground_truth), axis=0)
            last_100_overall_mse = np.mean((last_100_predictions - last_100_ground_truth) ** 2)
            last_100_overall_mae = np.mean(np.abs(last_100_predictions - last_100_ground_truth))
        else:
            # 如果episode长度不足100帧，使用全部帧
            last_100_mse = mse
            last_100_mae = mae
            last_100_overall_mse = overall_mse
            last_100_overall_mae = overall_mae
        
        results = {
            'episode_idx': episode_idx,
            'num_frames': len(predictions),
            'mse_per_dim': mse,
            'mae_per_dim': mae,
            'overall_mse': overall_mse,
            'overall_mae': overall_mae,
            'last_100_mse_per_dim': last_100_mse,
            'last_100_mae_per_dim': last_100_mae,
            'last_100_overall_mse': last_100_overall_mse,
            'last_100_overall_mae': last_100_overall_mae,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'states': states,
        }
        
        logging.info(f"Episode {episode_idx}: MSE={overall_mse:.6f}, MAE={overall_mae:.6f} | Last100: MSE={last_100_overall_mse:.6f}, MAE={last_100_overall_mae:.6f}")
        
        if self.save_images and 'episode_img_dir' in locals():
            logging.info(f"Episode {episode_idx}: 已保存 {len(predictions)} 帧图像到 {episode_img_dir}")
        
        return results
    
    def evaluate_episodes(self, start_idx: int = 0, num_episodes: int = -1):
        """评估多个episodes"""
        # 获取可用的episodes数量
        data_dir = self.dataset_path / "data" / "chunk-000"
        available_episodes = sorted([
            int(f.stem.split('_')[1]) 
            for f in data_dir.glob("episode_*.parquet")
        ])
        
        if num_episodes == -1:
            num_episodes = len(available_episodes)
        
        episodes_to_test = available_episodes[start_idx:start_idx + num_episodes]
        
        logging.info(f"开始评估 {len(episodes_to_test)} 个episodes (索引 {start_idx} 到 {start_idx + len(episodes_to_test) - 1})")
        
        all_results = []
        
        for ep_idx in tqdm(episodes_to_test, desc="评估进度"):
            result = self.evaluate_episode(ep_idx)
            if result:
                all_results.append(result)
                self.results['episode_errors'].append(result['overall_mse'])
                self.results['episode_indices'].append(ep_idx)
        
        return all_results
    
    def compute_statistics(self, all_results: list[dict]):
        """计算统计信息"""
        if not all_results:
            logging.warning("没有可用的评估结果")
            return
        
        print("\n" + "="*70)
        print("评估统计摘要")
        print("="*70)
        
        # 总体统计
        all_mse = [r['overall_mse'] for r in all_results]
        all_mae = [r['overall_mae'] for r in all_results]
        
        print(f"\n总体性能 (测试了 {len(all_results)} 个episodes):")
        print(f"  平均 MSE: {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
        print(f"  平均 MAE: {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
        print(f"  中位数 MSE: {np.median(all_mse):.6f}")
        print(f"  中位数 MAE: {np.median(all_mae):.6f}")
        
        # 后100帧的统计
        all_last100_mse = [r['last_100_overall_mse'] for r in all_results]
        all_last100_mae = [r['last_100_overall_mae'] for r in all_results]
        
        print(f"\n后100帧性能:")
        print(f"  平均 MSE: {np.mean(all_last100_mse):.6f} ± {np.std(all_last100_mse):.6f}")
        print(f"  平均 MAE: {np.mean(all_last100_mae):.6f} ± {np.std(all_last100_mae):.6f}")
        print(f"  中位数 MSE: {np.median(all_last100_mse):.6f}")
        print(f"  中位数 MAE: {np.median(all_last100_mae):.6f}")
        
        # 计算后100帧相对于全局的变化
        mse_ratio = np.mean(all_last100_mse) / np.mean(all_mse)
        mae_ratio = np.mean(all_last100_mae) / np.mean(all_mae)
        print(f"  后100帧 vs 全局: MSE比率={mse_ratio:.3f}x, MAE比率={mae_ratio:.3f}x")
        
        # 每个维度的统计
        print(f"\n各维度性能 (全局):")
        action_labels = ['eef_x', 'eef_y', 'eef_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        
        mse_per_dim = np.array([r['mse_per_dim'] for r in all_results])
        mae_per_dim = np.array([r['mae_per_dim'] for r in all_results])
        
        for i, label in enumerate(action_labels):
            print(f"  {label:10s}: MSE={np.mean(mse_per_dim[:, i]):.6f}, MAE={np.mean(mae_per_dim[:, i]):.6f}")
        
        # 后100帧每个维度的统计
        print(f"\n各维度性能 (后100帧):")
        last100_mse_per_dim = np.array([r['last_100_mse_per_dim'] for r in all_results])
        last100_mae_per_dim = np.array([r['last_100_mae_per_dim'] for r in all_results])
        
        for i, label in enumerate(action_labels):
            print(f"  {label:10s}: MSE={np.mean(last100_mse_per_dim[:, i]):.6f}, MAE={np.mean(last100_mae_per_dim[:, i]):.6f}")
        
        # 保存统计结果
        stats_file = self.output_dir / "statistics.txt"
        with open(stats_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("评估统计摘要\n")
            f.write("="*70 + "\n\n")
            f.write(f"测试episodes数: {len(all_results)}\n\n")
            
            f.write("全局性能:\n")
            f.write(f"  平均 MSE: {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}\n")
            f.write(f"  平均 MAE: {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}\n")
            f.write(f"  中位数 MSE: {np.median(all_mse):.6f}\n")
            f.write(f"  中位数 MAE: {np.median(all_mae):.6f}\n\n")
            
            f.write("后100帧性能:\n")
            f.write(f"  平均 MSE: {np.mean(all_last100_mse):.6f} ± {np.std(all_last100_mse):.6f}\n")
            f.write(f"  平均 MAE: {np.mean(all_last100_mae):.6f} ± {np.std(all_last100_mae):.6f}\n")
            f.write(f"  中位数 MSE: {np.median(all_last100_mse):.6f}\n")
            f.write(f"  中位数 MAE: {np.median(all_last100_mae):.6f}\n")
            f.write(f"  后100帧 vs 全局: MSE比率={mse_ratio:.3f}x, MAE比率={mae_ratio:.3f}x\n\n")
            
            f.write("各维度性能 (全局):\n")
            for i, label in enumerate(action_labels):
                f.write(f"  {label:10s}: MSE={np.mean(mse_per_dim[:, i]):.6f}, MAE={np.mean(mae_per_dim[:, i]):.6f}\n")
            
            f.write("\n各维度性能 (后100帧):\n")
            for i, label in enumerate(action_labels):
                f.write(f"  {label:10s}: MSE={np.mean(last100_mse_per_dim[:, i]):.6f}, MAE={np.mean(last100_mae_per_dim[:, i]):.6f}\n")
        
        logging.info(f"统计结果已保存到: {stats_file}")
    
    def visualize_results(self, all_results: list[dict]):
        """可视化评估结果"""
        if not all_results:
            return
        
        logging.info("生成可视化图表...")
        
        # 1. MSE分布图 - 扩展为3x2布局以包含后100帧的对比
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        
        # 1.1 Episode MSE分布
        ax1 = axes[0, 0]
        episode_mse = [r['overall_mse'] for r in all_results]
        episode_indices = [r['episode_idx'] for r in all_results]
        ax1.plot(episode_indices, episode_mse, marker='o', markersize=4, linewidth=1)
        ax1.set_xlabel('Episode Index')
        ax1.set_ylabel('MSE')
        ax1.set_title('MSE per Episode')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(np.mean(episode_mse), color='r', linestyle='--', alpha=0.5, label='Mean')
        ax1.legend()
        
        # 1.2 MSE直方图
        ax2 = axes[0, 1]
        ax2.hist(episode_mse, bins=30, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('MSE')
        ax2.set_ylabel('Frequency')
        ax2.set_title('MSE Distribution')
        ax2.axvline(np.mean(episode_mse), color='r', linestyle='--', label=f'Mean: {np.mean(episode_mse):.6f}')
        ax2.axvline(np.median(episode_mse), color='g', linestyle='--', label=f'Median: {np.median(episode_mse):.6f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 1.3 各维度MSE对比（全局）
        ax3 = axes[1, 0]
        action_labels = ['eef_x', 'eef_y', 'eef_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        mse_per_dim = np.array([r['mse_per_dim'] for r in all_results])
        mean_mse_per_dim = np.mean(mse_per_dim, axis=0)
        ax3.bar(action_labels, mean_mse_per_dim, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Action Dimension')
        ax3.set_ylabel('Mean MSE')
        ax3.set_title('MSE by Action Dimension (Overall)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 1.4 MAE vs MSE
        ax4 = axes[1, 1]
        episode_mae = [r['overall_mae'] for r in all_results]
        ax4.scatter(episode_mse, episode_mae, alpha=0.6)
        ax4.set_xlabel('MSE')
        ax4.set_ylabel('MAE')
        ax4.set_title('MSE vs MAE (Overall)')
        ax4.grid(True, alpha=0.3)
        
        # 1.5 后100帧MSE对比
        ax5 = axes[2, 0]
        last100_mse = [r['last_100_overall_mse'] for r in all_results]
        ax5.plot(episode_indices, last100_mse, marker='o', markersize=4, linewidth=1, color='orange')
        ax5.set_xlabel('Episode Index')
        ax5.set_ylabel('MSE (Last 100 Frames)')
        ax5.set_title('MSE per Episode (Last 100 Frames)')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(np.mean(last100_mse), color='r', linestyle='--', alpha=0.5, label='Mean')
        ax5.legend()
        
        # 1.6 全局 vs 后100帧对比
        ax6 = axes[2, 1]
        x_pos = np.arange(len(action_labels))
        width = 0.35
        last100_mse_per_dim = np.array([r['last_100_mse_per_dim'] for r in all_results])
        mean_last100_mse_per_dim = np.mean(last100_mse_per_dim, axis=0)
        
        ax6.bar(x_pos - width/2, mean_mse_per_dim, width, label='Overall', alpha=0.7)
        ax6.bar(x_pos + width/2, mean_last100_mse_per_dim, width, label='Last 100', alpha=0.7)
        ax6.set_xlabel('Action Dimension')
        ax6.set_ylabel('Mean MSE')
        ax6.set_title('MSE Comparison: Overall vs Last 100 Frames')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(action_labels, rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "evaluation_summary.png", dpi=150, bbox_inches='tight')
        logging.info(f"保存图表: {self.output_dir / 'evaluation_summary.png'}")
        plt.close()
        
        # 2. 样本episode的预测vs真实值对比
        if len(all_results) > 0:
            self._plot_sample_episode(all_results[0])
    
    def _plot_sample_episode(self, result: dict):
        """绘制单个episode的预测vs真实值对比"""
        predictions = result['predictions']
        ground_truth = result['ground_truth']
        episode_idx = result['episode_idx']
        
        action_labels = ['eef_x', 'eef_y', 'eef_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
        
        # 图1: 每个维度单独的子图
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        axes = axes.flatten()
        
        time_steps = np.arange(len(predictions))
        
        for i, label in enumerate(action_labels):
            ax = axes[i]
            ax.plot(time_steps, ground_truth[:, i], label='Ground Truth', linewidth=1.5, alpha=0.7)
            ax.plot(time_steps, predictions[:, i], label='Prediction', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'{label} Value')
            ax.set_title(f'{label} - Episode {episode_idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"episode_{episode_idx}_predictions.png", dpi=150, bbox_inches='tight')
        logging.info(f"保存图表: {self.output_dir / f'episode_{episode_idx}_predictions.png'}")
        plt.close()
        
        # 图2: 所有维度在一张图上
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # 定义颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(action_labels)))
        
        for i, (label, color) in enumerate(zip(action_labels, colors)):
            # 绘制真实值（实线）
            ax.plot(time_steps, ground_truth[:, i], 
                   label=f'{label} (GT)', 
                   linewidth=2, 
                   alpha=0.8, 
                   color=color, 
                   linestyle='-')
            # 绘制预测值（虚线）
            ax.plot(time_steps, predictions[:, i], 
                   label=f'{label} (Pred)', 
                   linewidth=2, 
                   alpha=0.6, 
                   color=color, 
                   linestyle='--')
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Action Value', fontsize=12)
        ax.set_title(f'All Action Dimensions - Episode {episode_idx}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"episode_{episode_idx}_predictions_combined.png", dpi=150, bbox_inches='tight')
        logging.info(f"保存图表: {self.output_dir / f'episode_{episode_idx}_predictions_combined.png'}")
        plt.close()
    
    def save_results(self, all_results: list[dict]):
        """保存详细结果"""
        import json

        # 保存摘要（包含后100帧统计）
        summary = {
            'num_episodes': len(all_results),
            'episodes': [
                {
                    'episode_idx': r['episode_idx'],
                    'num_frames': r['num_frames'],
                    'overall_mse': float(r['overall_mse']),
                    'overall_mae': float(r['overall_mae']),
                    'mse_per_dim': r['mse_per_dim'].tolist(),
                    'mae_per_dim': r['mae_per_dim'].tolist(),
                    'last_100_overall_mse': float(r['last_100_overall_mse']),
                    'last_100_overall_mae': float(r['last_100_overall_mae']),
                    'last_100_mse_per_dim': r['last_100_mse_per_dim'].tolist(),
                    'last_100_mae_per_dim': r['last_100_mae_per_dim'].tolist(),
                }
                for r in all_results
            ]
        }

        summary_file = self.output_dir / "results_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logging.info(f"结果摘要已保存到: {summary_file}")

        # 保存详细数据 (numpy格式)
        if all_results:
            detailed_file = self.output_dir / "detailed_results.npz"

            episode_indices = np.array([r['episode_idx'] for r in all_results], dtype=np.int32)
            episode_mse     = np.array([r['overall_mse'] for r in all_results], dtype=np.float32)
            episode_mae     = np.array([r['overall_mae'] for r in all_results], dtype=np.float32)
            
            # 后100帧的统计
            episode_last100_mse = np.array([r['last_100_overall_mse'] for r in all_results], dtype=np.float32)
            episode_last100_mae = np.array([r['last_100_overall_mae'] for r in all_results], dtype=np.float32)

            # 每个 episode 一条 (T_i, 7) / (T_i, 32) 的 ndarray
            preds_list = [r['predictions']   for r in all_results]
            gts_list   = [r['ground_truth']  for r in all_results]
            states_list= [r['states']        for r in all_results]

            # 用 object 数组包装，兼容不同长度的 episode
            preds_arr  = np.array(preds_list,  dtype=object)
            gts_arr    = np.array(gts_list,    dtype=object)
            states_arr = np.array(states_list, dtype=object)

            np.savez(
                detailed_file,
                episode_indices=episode_indices,
                episode_mse=episode_mse,
                episode_mae=episode_mae,
                episode_last100_mse=episode_last100_mse,
                episode_last100_mae=episode_last100_mae,
                action_predictions=preds_arr,
                action_ground_truth=gts_arr,
                states=states_arr,
            )

            logging.info(f"详细结果已保存到: {detailed_file}")


def create_policy(checkpoint_config: str, checkpoint_dir: str, default_prompt: str) -> _policy.Policy:
    """创建策略"""
    logging.info(f"加载策略: config={checkpoint_config}, dir={checkpoint_dir}")
    
    policy = _policy_config.create_trained_policy(
        _config.get_config(checkpoint_config),
        checkpoint_dir,
        default_prompt=default_prompt
    )
    
    return policy


def main() -> None:
    dataset_path = "/root/openpi-umi/data/umi_lerobot_dataset_v3_val"
    checkpoint_config = "pi05_umi_32d_80k_95"
    checkpoint_dir = "/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95/my_experiment_80k_95/79999"
    output_dir = "/root/openpi-umi/inference_results_1"
    num_episodes = 1
    start_episode = 0
    default_prompt = "pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box"
    visualize = True
    save_detailed = True
    save_images = True

    """主函数"""
    logging.info("="*70)
    logging.info("本地推理测试")
    logging.info("="*70)
    logging.info(f"数据集路径: {dataset_path}")
    logging.info(f"Checkpoint: {checkpoint_dir}")
    logging.info(f"测试Episodes: {num_episodes if num_episodes > 0 else '全部'}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"保存观测图像: {save_images}")
    logging.info("="*70)
    
    # 创建策略
    policy = create_policy(checkpoint_config, checkpoint_dir, default_prompt)
    
    # 创建评估器
    evaluator = LocalInferenceEvaluator(
        policy=policy,
        dataset_path=Path(dataset_path),
        output_dir=Path(output_dir),
        save_images=save_images
    )
    
    # 运行评估
    all_results = evaluator.evaluate_episodes(
        start_idx=start_episode,
        num_episodes=num_episodes
    )
    
    # 计算统计信息
    evaluator.compute_statistics(all_results)
    
    # 可视化结果
    if visualize:
        evaluator.visualize_results(all_results)
    
    # 保存结果
    if save_detailed:
        evaluator.save_results(all_results)
    
    logging.info("\n" + "="*70)
    logging.info("评估完成！")
    logging.info(f"结果保存在: {output_dir}")
    logging.info("="*70)


if __name__ == "__main__":
    main()

