#!/usr/bin/env python3
"""
数据质量检查脚本 - 深度检查数据集的异常和问题
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
from collections import defaultdict

sns.set_style("whitegrid")


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.meta_path = self.dataset_path / "meta"
        self.data_path = self.dataset_path / "data"
        
        with open(self.meta_path / "info.json", 'r') as f:
            self.info = json.load(f)
        
        self.issues = defaultdict(list)
        self.warnings = defaultdict(list)
        
        print(f"开始检查数据集: {dataset_path}")
    
    def check_file_integrity(self):
        """检查文件完整性"""
        print("\n" + "="*60)
        print("1. 文件完整性检查")
        print("="*60)
        
        total_episodes = self.info['total_episodes']
        missing_files = []
        corrupted_files = []
        
        for i in range(total_episodes):
            episode_path = self.data_path / f"chunk-000/episode_{i:06d}.parquet"
            
            if not episode_path.exists():
                missing_files.append(i)
                self.issues['missing_files'].append(i)
            else:
                try:
                    # 尝试读取文件
                    table = pq.read_table(episode_path)
                    if len(table) == 0:
                        corrupted_files.append(i)
                        self.issues['empty_files'].append(i)
                except Exception as e:
                    corrupted_files.append(i)
                    self.issues['corrupted_files'].append((i, str(e)))
        
        print(f"总Episodes数: {total_episodes}")
        print(f"缺失文件: {len(missing_files)}")
        print(f"损坏/空文件: {len(corrupted_files)}")
        
        if missing_files:
            print(f"\n缺失的Episodes: {missing_files[:10]}")
            if len(missing_files) > 10:
                print(f"... 还有 {len(missing_files) - 10} 个")
        
        if corrupted_files:
            print(f"\n损坏的Episodes: {corrupted_files[:10]}")
            if len(corrupted_files) > 10:
                print(f"... 还有 {len(corrupted_files) - 10} 个")
        
        if not missing_files and not corrupted_files:
            print("✓ 所有文件完整")
    
    def check_timestamp_continuity(self, sample_size: int = 50):
        """检查时间戳连续性"""
        print("\n" + "="*60)
        print("2. 时间戳连续性检查")
        print("="*60)
        
        fps = self.info['fps']
        expected_dt = 1.0 / fps
        
        timestamp_issues = []
        
        episodes_to_check = min(sample_size, self.info['total_episodes'])
        
        for i in range(episodes_to_check):
            episode_path = self.data_path / f"chunk-000/episode_{i:06d}.parquet"
            
            if not episode_path.exists():
                continue
            
            try:
                df = pq.read_table(episode_path).to_pandas()
                
                if 'timestamp' not in df.columns:
                    continue
                
                timestamps = df['timestamp'].values
                if len(timestamps) < 2:
                    continue
                
                # 检查时间间隔
                dts = np.diff(timestamps)
                
                # 检查异常间隔 (偏离超过50%)
                abnormal_indices = np.where(np.abs(dts - expected_dt) > expected_dt * 0.5)[0]
                
                if len(abnormal_indices) > 0:
                    timestamp_issues.append({
                        'episode': i,
                        'num_abnormal': len(abnormal_indices),
                        'max_deviation': np.max(np.abs(dts - expected_dt)),
                        'indices': abnormal_indices[:5].tolist()  # 只记录前5个
                    })
                    self.warnings['timestamp_gaps'].append(i)
            
            except Exception as e:
                print(f"  Episode {i}: 读取错误 - {e}")
        
        print(f"检查了 {episodes_to_check} 个episodes")
        print(f"发现时间戳异常: {len(timestamp_issues)} 个episodes")
        
        if timestamp_issues:
            print("\n时间戳异常详情:")
            for issue in timestamp_issues[:5]:
                print(f"  Episode {issue['episode']}: "
                      f"{issue['num_abnormal']} 个异常间隔, "
                      f"最大偏差 {issue['max_deviation']:.4f}s")
            if len(timestamp_issues) > 5:
                print(f"  ... 还有 {len(timestamp_issues) - 5} 个episodes")
    
    def check_action_magnitude(self, sample_size: int = 50):
        """检查动作幅值异常"""
        print("\n" + "="*60)
        print("3. 动作幅值检查")
        print("="*60)
        
        all_action_norms = []
        extreme_actions = []
        
        episodes_to_check = min(sample_size, self.info['total_episodes'])
        
        for i in range(episodes_to_check):
            episode_path = self.data_path / f"chunk-000/episode_{i:06d}.parquet"
            
            if not episode_path.exists():
                continue
            
            try:
                df = pq.read_table(episode_path).to_pandas()
                
                if 'actions' not in df.columns:
                    continue
                
                actions = np.stack(df['actions'].values)
                
                # 计算动作范数（不包括gripper维度）
                action_norms = np.linalg.norm(actions[:, :6], axis=1)
                all_action_norms.extend(action_norms)
                
                # 检查极端动作
                max_norm = np.max(action_norms)
                if max_norm > 0.1:  # 阈值可调整
                    extreme_indices = np.where(action_norms > 0.1)[0]
                    extreme_actions.append({
                        'episode': i,
                        'max_norm': max_norm,
                        'num_extreme': len(extreme_indices),
                        'percentage': len(extreme_indices) / len(actions) * 100
                    })
                    self.warnings['large_actions'].append(i)
            
            except Exception as e:
                print(f"  Episode {i}: 读取错误 - {e}")
        
        if all_action_norms:
            all_action_norms = np.array(all_action_norms)
            print(f"\n动作幅值统计 (前6维范数):")
            print(f"  平均: {np.mean(all_action_norms):.6f}")
            print(f"  中位数: {np.median(all_action_norms):.6f}")
            print(f"  标准差: {np.std(all_action_norms):.6f}")
            print(f"  最大值: {np.max(all_action_norms):.6f}")
            print(f"  第99百分位: {np.percentile(all_action_norms, 99):.6f}")
            
            if extreme_actions:
                print(f"\n发现 {len(extreme_actions)} 个episodes包含极端动作:")
                for item in extreme_actions[:5]:
                    print(f"  Episode {item['episode']}: "
                          f"最大范数={item['max_norm']:.4f}, "
                          f"{item['num_extreme']} 个异常帧 ({item['percentage']:.1f}%)")
                if len(extreme_actions) > 5:
                    print(f"  ... 还有 {len(extreme_actions) - 5} 个episodes")
    
    def check_state_consistency(self, sample_size: int = 50):
        """检查状态一致性"""
        print("\n" + "="*60)
        print("4. 状态一致性检查")
        print("="*60)
        
        state_jump_issues = []
        
        episodes_to_check = min(sample_size, self.info['total_episodes'])
        
        for i in range(episodes_to_check):
            episode_path = self.data_path / f"chunk-000/episode_{i:06d}.parquet"
            
            if not episode_path.exists():
                continue
            
            try:
                df = pq.read_table(episode_path).to_pandas()
                
                if 'state' not in df.columns:
                    continue
                
                states = np.stack(df['state'].values)
                
                # 检查状态突变
                state_diffs = np.diff(states, axis=0)
                state_diff_norms = np.linalg.norm(state_diffs[:, :6], axis=1)
                
                # 异常阈值：超过均值的5倍标准差
                mean_diff = np.mean(state_diff_norms)
                std_diff = np.std(state_diff_norms)
                threshold = mean_diff + 5 * std_diff
                
                abnormal_jumps = np.where(state_diff_norms > threshold)[0]
                
                if len(abnormal_jumps) > 0:
                    state_jump_issues.append({
                        'episode': i,
                        'num_jumps': len(abnormal_jumps),
                        'max_jump': np.max(state_diff_norms),
                        'threshold': threshold,
                        'indices': abnormal_jumps[:5].tolist()
                    })
                    self.warnings['state_jumps'].append(i)
            
            except Exception as e:
                print(f"  Episode {i}: 读取错误 - {e}")
        
        print(f"检查了 {episodes_to_check} 个episodes")
        print(f"发现状态突变: {len(state_jump_issues)} 个episodes")
        
        if state_jump_issues:
            print("\n状态突变详情:")
            for issue in state_jump_issues[:5]:
                print(f"  Episode {issue['episode']}: "
                      f"{issue['num_jumps']} 个突变, "
                      f"最大跳变={issue['max_jump']:.4f} "
                      f"(阈值={issue['threshold']:.4f})")
            if len(state_jump_issues) > 5:
                print(f"  ... 还有 {len(state_jump_issues) - 5} 个episodes")
    
    def check_gripper_consistency(self, sample_size: int = 50):
        """检查夹爪一致性"""
        print("\n" + "="*60)
        print("5. 夹爪一致性检查")
        print("="*60)
        
        gripper_issues = []
        
        episodes_to_check = min(sample_size, self.info['total_episodes'])
        
        for i in range(episodes_to_check):
            episode_path = self.data_path / f"chunk-000/episode_{i:06d}.parquet"
            
            if not episode_path.exists():
                continue
            
            try:
                df = pq.read_table(episode_path).to_pandas()
                
                if 'state' not in df.columns:
                    continue
                
                states = np.stack(df['state'].values)
                gripper_values = states[:, 6]
                
                # 检查范围
                if np.any(gripper_values < 0) or np.any(gripper_values > 0.1):
                    gripper_issues.append({
                        'episode': i,
                        'type': 'out_of_range',
                        'min': np.min(gripper_values),
                        'max': np.max(gripper_values)
                    })
                    self.warnings['gripper_range'].append(i)
                
                # 检查突变
                gripper_diffs = np.abs(np.diff(gripper_values))
                max_diff = np.max(gripper_diffs)
                
                if max_diff > 0.03:  # 单帧变化超过3cm
                    gripper_issues.append({
                        'episode': i,
                        'type': 'sudden_change',
                        'max_change': max_diff
                    })
                    self.warnings['gripper_jumps'].append(i)
            
            except Exception as e:
                print(f"  Episode {i}: 读取错误 - {e}")
        
        print(f"检查了 {episodes_to_check} 个episodes")
        print(f"发现夹爪异常: {len(gripper_issues)} 个问题")
        
        if gripper_issues:
            print("\n夹爪异常详情:")
            for issue in gripper_issues[:10]:
                if issue['type'] == 'out_of_range':
                    print(f"  Episode {issue['episode']}: "
                          f"范围异常 [{issue['min']:.4f}, {issue['max']:.4f}]")
                elif issue['type'] == 'sudden_change':
                    print(f"  Episode {issue['episode']}: "
                          f"突变 {issue['max_change']:.4f}")
            if len(gripper_issues) > 10:
                print(f"  ... 还有 {len(gripper_issues) - 10} 个问题")
    
    def check_nan_inf_values(self, sample_size: int = 100):
        """检查NaN和Inf值"""
        print("\n" + "="*60)
        print("6. NaN/Inf值检查")
        print("="*60)
        
        nan_inf_issues = []
        
        episodes_to_check = min(sample_size, self.info['total_episodes'])
        
        for i in range(episodes_to_check):
            episode_path = self.data_path / f"chunk-000/episode_{i:06d}.parquet"
            
            if not episode_path.exists():
                continue
            
            try:
                df = pq.read_table(episode_path).to_pandas()
                
                # 检查数值列
                for col in ['state', 'actions', 'timestamp']:
                    if col not in df.columns:
                        continue
                    
                    if col in ['state', 'actions']:
                        values = np.stack(df[col].values)
                    else:
                        values = df[col].values
                    
                    has_nan = np.any(np.isnan(values))
                    has_inf = np.any(np.isinf(values))
                    
                    if has_nan or has_inf:
                        nan_inf_issues.append({
                            'episode': i,
                            'column': col,
                            'has_nan': has_nan,
                            'has_inf': has_inf
                        })
                        self.issues['nan_inf_values'].append((i, col))
            
            except Exception as e:
                print(f"  Episode {i}: 读取错误 - {e}")
        
        print(f"检查了 {episodes_to_check} 个episodes")
        
        if nan_inf_issues:
            print(f"发现 {len(nan_inf_issues)} 个NaN/Inf问题:")
            for issue in nan_inf_issues[:10]:
                status = []
                if issue['has_nan']:
                    status.append("NaN")
                if issue['has_inf']:
                    status.append("Inf")
                print(f"  Episode {issue['episode']}, 列 '{issue['column']}': {', '.join(status)}")
            if len(nan_inf_issues) > 10:
                print(f"  ... 还有 {len(nan_inf_issues) - 10} 个问题")
        else:
            print("✓ 未发现NaN或Inf值")
    
    def generate_summary_report(self, output_path: str = None):
        """生成总结报告"""
        print("\n" + "="*60)
        print("数据质量检查总结")
        print("="*60)
        
        total_issues = sum(len(v) for v in self.issues.values())
        total_warnings = sum(len(v) for v in self.warnings.values())
        
        print(f"\n严重问题: {total_issues}")
        for issue_type, items in self.issues.items():
            print(f"  - {issue_type}: {len(items)}")
        
        print(f"\n警告: {total_warnings}")
        for warning_type, items in self.warnings.items():
            print(f"  - {warning_type}: {len(items)}")
        
        if total_issues == 0 and total_warnings == 0:
            print("\n✓✓✓ 数据集质量良好，未发现严重问题！")
        elif total_issues == 0:
            print(f"\n✓ 数据集基本正常，有 {total_warnings} 个轻微警告")
        else:
            print(f"\n⚠ 数据集存在 {total_issues} 个严重问题，需要处理")
        
        # 保存报告
        if output_path:
            report_data = {
                'summary': {
                    'total_issues': total_issues,
                    'total_warnings': total_warnings
                },
                'issues': dict(self.issues),
                'warnings': dict(self.warnings)
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"\n报告已保存到: {output_path}")
    
    def run_all_checks(self, sample_size: int = 50):
        """运行所有检查"""
        self.check_file_integrity()
        self.check_nan_inf_values(sample_size=min(sample_size * 2, 100))
        self.check_timestamp_continuity(sample_size=sample_size)
        self.check_action_magnitude(sample_size=sample_size)
        self.check_state_consistency(sample_size=sample_size)
        self.check_gripper_consistency(sample_size=sample_size)
        
        # 生成总结
        output_path = self.dataset_path / "data_quality_report.json"
        self.generate_summary_report(output_path=output_path)


def main():
    parser = argparse.ArgumentParser(
        description="深度检查数据集质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 检查数据集（采样50个episodes）
  python check_data_quality.py /data/umi_lerobot_dataset_v3
  
  # 检查更多episodes
  python check_data_quality.py /data/umi_lerobot_dataset_v3 --sample-size 100
  
  # 只检查文件完整性
  python check_data_quality.py /data/umi_lerobot_dataset_v3 --check-files-only
        """
    )
    
    parser.add_argument(
        'dataset_path',
        type=str,
        help='数据集根目录路径'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50,
        help='检查的样本episode数量 (默认: 50)'
    )
    
    parser.add_argument(
        '--check-files-only',
        action='store_true',
        help='只检查文件完整性'
    )
    
    args = parser.parse_args()
    
    checker = DataQualityChecker(args.dataset_path)
    
    if args.check_files_only:
        checker.check_file_integrity()
    else:
        checker.run_all_checks(sample_size=args.sample_size)


if __name__ == "__main__":
    main()

