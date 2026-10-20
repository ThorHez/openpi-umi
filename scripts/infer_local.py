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

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

import cv2
import numpy as np
from openpi import transforms as _transforms


IMAGE_KEYS = (
    "left_wrist_0_rgb_0",
    "left_wrist_1_rgb_0",
    "right_wrist_0_rgb_0",
    "right_wrist_1_rgb_0",
)

STATE_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_pos_wrt_start",
    "robot0_eef_rot_axis_angle",
    "robot0_eef_rot_axis_angle_wrt_start",
    "robot0_gripper_width",
    "robot1_eef_pos",
    "robot1_eef_pos_wrt_start",
    "robot1_eef_rot_axis_angle",
    "robot1_eef_rot_axis_angle_wrt_start",
    "robot1_gripper_width",
)

ROBUSTNESS_ACTION_LABELS = (
    "robot0_pos_x", "robot0_pos_y", "robot0_pos_z",
    "robot0_rot_0", "robot0_rot_1", "robot0_rot_2",
    "robot0_rot_3", "robot0_rot_4", "robot0_rot_5", "robot0_gripper",
    "robot1_pos_x", "robot1_pos_y", "robot1_pos_z",
    "robot1_rot_0", "robot1_rot_1", "robot1_rot_2",
    "robot1_rot_3", "robot1_rot_4", "robot1_rot_5", "robot1_gripper",
)


def action_labels(num_dims: int) -> list[str]:
    labels = list(ROBUSTNESS_ACTION_LABELS[:num_dims])
    labels.extend(f"action_{i}" for i in range(len(labels), num_dims))
    return labels


@dataclasses.dataclass(frozen=True)
class TrimRobustnessPolicyOutputs:
    """Remove padded outputs before applying the checkpoint's 20-D norm stats."""

    action_dim: int = 20

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        data.pop("state", None)
        data["actions"] = data["actions"][..., :self.action_dim]
        return data


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

    # LeRobot数据集根目录
    dataset_path: str

    # Checkpoint目录（例如 checkpoints/pi05_umi_robustness/1017/21000）
    checkpoint_dir: str

    # 训练配置名称
    checkpoint_config: str = "pi05_umi_robustness"

    # 要评估的episode编号
    episode_index: int = 0
    
    # 默认prompt（如果数据中没有prompt）
    default_prompt: str = "pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box"
    
    # 输出目录
    output_dir: str = "inference_results"
    
    # 是否保存详细结果
    save_detailed: bool = True
    
    # 是否生成可视化
    visualize: bool = True
    
class LocalInferenceEvaluator:
    """本地推理评估器"""
    
    def __init__(self, policy: _policy.Policy, dataset_path: Path, output_dir: Path):
        self.policy = policy
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'episode_errors': [],
            'episode_indices': [],
            'action_predictions': [],
            'action_ground_truth': [],
            'state_sequences': [],
        }
        
        logging.info(f"初始化评估器 - 数据集: {dataset_path}, 输出: {output_dir}")
    
    def load_episode(self, episode_idx: int) -> list[dict] | None:
        """加载一个episode的数据"""
        matches = sorted((self.dataset_path / "data").glob(f"chunk-*/episode_{episode_idx:06d}.parquet"))
        if not matches:
            logging.warning(f"Episode {episode_idx} 不存在于: {self.dataset_path / 'data'}")
            return None
        episode_path = matches[0]
        
        try:
            return pq.read_table(episode_path).to_pylist()
        except Exception as e:
            logging.error(f"加载 Episode {episode_idx} 失败: {e}")
            return None
    
    def prepare_observation(self, row: dict) -> dict:
        """准备pi05_umi_robustness所需的四视角双臂观测。"""
        obs = {key: np.asarray(row[key], dtype=np.float32) for key in STATE_KEYS}
        for key in IMAGE_KEYS:
            encoded = row[key]
            img_bytes = encoded["bytes"] if isinstance(encoded, dict) else encoded
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法解码图像列 {key}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            obs[key] = np.moveaxis(img, -1, 0)
        obs["actions"] = np.asarray(row["actions"], dtype=np.float32)
        return obs
    
    def evaluate_episode(self, episode_idx: int) -> dict | None:
        """评估单个episode"""
        rows = self.load_episode(episode_idx)
        if rows is None:
            return None
        
        logging.info(f"评估 Episode {episode_idx} ({len(rows)} 帧)")
        
        # 存储预测和真实值
        predictions = []
        ground_truth = []
        states = []
        
        # 遍历episode中的每一帧
        for idx, row in enumerate(rows):
            print(f"inference episode {episode_idx} frame {idx}")
            # 准备观测
            obs = self.prepare_observation(row)
            
            # 模型推理
            action = self.policy.infer(obs)
            predicted_action = np.asarray(action["actions"])
            
            # 如果模型返回多步预测（action chunking），只取第一步
            if predicted_action.ndim == 2:
                predicted_action = predicted_action[1]
            
            # 获取真实动作
            true_action = np.asarray(row["actions"], dtype=np.float32)
            if true_action.ndim == 2:
                true_action = true_action[1]
            if predicted_action.shape[-1] < true_action.shape[-1]:
                raise ValueError(
                    f"预测动作维度 {predicted_action.shape[-1]} 小于真实动作维度 {true_action.shape[-1]}"
                )
            # pi05模型输出32维；该配置中前20维为双臂有效动作，其余为padding。
            predicted_action = predicted_action[:true_action.shape[-1]]
            
            predictions.append(predicted_action)
            ground_truth.append(true_action)
            states.append(np.concatenate([obs[key] for key in STATE_KEYS], axis=-1))
        
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
        mse_per_dim = np.array([r['mse_per_dim'] for r in all_results])
        mae_per_dim = np.array([r['mae_per_dim'] for r in all_results])
        labels = action_labels(mse_per_dim.shape[-1])
        
        for i, label in enumerate(labels):
            print(f"  {label:10s}: MSE={np.mean(mse_per_dim[:, i]):.6f}, MAE={np.mean(mae_per_dim[:, i]):.6f}")
        
        # 后100帧每个维度的统计
        print(f"\n各维度性能 (后100帧):")
        last100_mse_per_dim = np.array([r['last_100_mse_per_dim'] for r in all_results])
        last100_mae_per_dim = np.array([r['last_100_mae_per_dim'] for r in all_results])
        
        for i, label in enumerate(labels):
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
            for i, label in enumerate(labels):
                f.write(f"  {label:10s}: MSE={np.mean(mse_per_dim[:, i]):.6f}, MAE={np.mean(mae_per_dim[:, i]):.6f}\n")
            
            f.write("\n各维度性能 (后100帧):\n")
            for i, label in enumerate(labels):
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
        mse_per_dim = np.array([r['mse_per_dim'] for r in all_results])
        mean_mse_per_dim = np.mean(mse_per_dim, axis=0)
        labels = action_labels(mean_mse_per_dim.shape[-1])
        ax3.bar(labels, mean_mse_per_dim, edgecolor='black', alpha=0.7)
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
        x_pos = np.arange(len(labels))
        width = 0.35
        last100_mse_per_dim = np.array([r['last_100_mse_per_dim'] for r in all_results])
        mean_last100_mse_per_dim = np.mean(last100_mse_per_dim, axis=0)
        
        ax6.bar(x_pos - width/2, mean_mse_per_dim, width, label='Overall', alpha=0.7)
        ax6.bar(x_pos + width/2, mean_last100_mse_per_dim, width, label='Last 100', alpha=0.7)
        ax6.set_xlabel('Action Dimension')
        ax6.set_ylabel('Mean MSE')
        ax6.set_title('MSE Comparison: Overall vs Last 100 Frames')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(labels, rotation=45)
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
        
        labels = action_labels(predictions.shape[-1])
        
        # 图1: 每个维度单独的子图
        num_cols = 4
        num_rows = (len(labels) + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 4 * num_rows))
        axes = np.asarray(axes).reshape(-1)
        
        time_steps = np.arange(len(predictions))
        
        for i, label in enumerate(labels):
            ax = axes[i]
            ax.plot(time_steps, ground_truth[:, i], label='Ground Truth', linewidth=1.5, alpha=0.7)
            ax.plot(time_steps, predictions[:, i], label='Prediction', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'{label} Value')
            ax.set_title(f'{label} - Episode {episode_idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for ax in axes[len(labels):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"episode_{episode_idx}_predictions.png", dpi=150, bbox_inches='tight')
        logging.info(f"保存图表: {self.output_dir / f'episode_{episode_idx}_predictions.png'}")
        plt.close()
        
        # 图2: 所有维度在一张图上
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # 定义颜色映射
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        
        for i, (label, color) in enumerate(zip(labels, colors)):
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
    policy._output_transform = _transforms.compose(
        [TrimRobustnessPolicyOutputs(), policy._output_transform]
    )
    return policy


def main() -> None:
    args = tyro.cli(Args)

    """主函数"""
    logging.info("="*70)
    logging.info("本地推理测试")
    logging.info("="*70)
    logging.info(f"数据集路径: {args.dataset_path}")
    logging.info(f"配置: {args.checkpoint_config}")
    logging.info(f"Checkpoint: {args.checkpoint_dir}")
    logging.info(f"测试Episode: {args.episode_index}")
    logging.info(f"输出目录: {args.output_dir}")
    logging.info("="*70)
    
    # 创建策略
    policy = create_policy(args.checkpoint_config, args.checkpoint_dir, args.default_prompt)
    
    # 创建评估器
    evaluator = LocalInferenceEvaluator(
        policy=policy,
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
    )
    
    # 运行指定episode的评估
    result = evaluator.evaluate_episode(args.episode_index)
    all_results = [] if result is None else [result]
    
    # 计算统计信息
    evaluator.compute_statistics(all_results)
    
    # 可视化结果
    if args.visualize:
        evaluator.visualize_results(all_results)
    
    # 保存结果
    if args.save_detailed:
        evaluator.save_results(all_results)
    
    logging.info("\n" + "="*70)
    logging.info("评估完成！")
    logging.info(f"结果保存在: {args.output_dir}")
    logging.info("="*70)


if __name__ == "__main__":
    main()

