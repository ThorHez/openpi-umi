#!/usr/bin/env python3
"""
分析并可视化 UMI LeRobot 数据集的 eef_pos（末端执行器位置）轨迹。

功能：
- 从 LeRobot 格式数据集（parquet + meta）加载 robot0_eef_pos / robot1_eef_pos
- 统计：轨迹长度、范围、起止距离、简单统计量
- 可视化：3D 轨迹图（单/多 episode）、2D 投影、位置-时间曲线

用法:
  python analyze_eef_pos_trajectory.py \\
    --dataset /root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_two_stage_28_02_clean \\
    --episodes 0 1 2 \\
    --output-dir ./eef_trajectory_plots
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Optional deps: try import and fallback
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def load_episodes_meta(meta_path: Path) -> Tuple[Dict, List[Dict]]:
    """加载 meta/info.json 和 meta/episodes.jsonl。"""
    with open(meta_path / "info.json", "r") as f:
        info = json.load(f)
    episodes = []
    with open(meta_path / "episodes.jsonl", "r") as f:
        for line in f:
            episodes.append(json.loads(line))
    return info, episodes


def load_episode_eef_trajectories(
    data_path: Path,
    info: Dict,
    episode_indices: List[int],
    eef_keys: Optional[List[str]] = None,
    position_index: int = 0,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    从 parquet 加载指定 episodes 的 eef 位置轨迹。

    data_path: 数据集 data 目录（其下有 chunk-000/ 等）
    info: meta/info.json 内容
    episode_indices: 要加载的 episode 下标
    eef_keys: 要读取的列，如 ['robot0_eef_pos', 'robot1_eef_pos']
    position_index: 每帧 shape (2,3) 时取哪一维，0=当前，1=下一帧

    返回: { episode_idx: { 'robot0_eef_pos': (T,3), 'robot1_eef_pos': (T,3), ... } }
    """
    if pq is None:
        raise RuntimeError("需要安装 pyarrow: pip install pyarrow")
    if eef_keys is None:
        eef_keys = ["robot0_eef_pos", "robot1_eef_pos"]
    chunks_size = info.get("chunks_size", 1000)
    result = {}
    for ep_idx in episode_indices:
        chunk_id = ep_idx // chunks_size
        chunk_dir = data_path / f"chunk-{chunk_id:03d}"
        parquet_file = chunk_dir / f"episode_{ep_idx:06d}.parquet"
        if not parquet_file.exists():
            raise FileNotFoundError(f"Episode parquet 不存在: {parquet_file}")
        table = pq.read_table(parquet_file)
        traj = {}
        for key in eef_keys:
            if key not in table.column_names:
                continue
            col = table.column(key)
            # 每行可能是 list/array shape (2, 3)
            positions = []
            for i in range(len(col)):
                row = col[i]
                if hasattr(row, "as_py"):
                    row = row.as_py()
                arr = np.asarray(row, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] >= 1:
                    positions.append(arr[position_index])
                elif arr.ndim == 1 and len(arr) >= 3:
                    positions.append(arr[:3])
                else:
                    positions.append(np.zeros(3, dtype=np.float32))
            traj[key] = np.stack(positions, axis=0)
        result[ep_idx] = traj
    return result


def analyze_trajectories(
    trajectories: Dict[int, Dict[str, np.ndarray]],
    fps: float = 60.0,
) -> Dict:
    """对轨迹做简单分析，返回统计字典。"""
    stats = {
        "episodes": {},
        "global": {"robot0_eef_pos": {"min": [], "max": [], "length": []}, "robot1_eef_pos": {"min": [], "max": [], "length": []}},
    }
    for ep_idx, traj_dict in trajectories.items():
        ep_stat = {}
        for key, pos in traj_dict.items():
            T = len(pos)
            ep_stat[key] = {
                "length": T,
                "duration_sec": T / fps,
                "min": pos.min(axis=0).tolist(),
                "max": pos.max(axis=0).tolist(),
                "start": pos[0].tolist(),
                "end": pos[-1].tolist(),
                "start_end_distance": float(np.linalg.norm(pos[-1] - pos[0])),
            }
            if key in stats["global"]:
                stats["global"][key]["min"].append(pos.min(axis=0))
                stats["global"][key]["max"].append(pos.max(axis=0))
                stats["global"][key]["length"].append(T)
        stats["episodes"][ep_idx] = ep_stat
    for key in list(stats["global"].keys()):
        g = stats["global"][key]
        if g["min"]:
            g["min"] = np.min(g["min"], axis=0).tolist()
            g["max"] = np.max(g["max"], axis=0).tolist()
            g["length_mean"] = float(np.mean(g["length"]))
            g["length_min"] = int(min(g["length"]))
            g["length_max"] = int(max(g["length"]))
        else:
            g["min"] = g["max"] = g["length_mean"] = g["length_min"] = g["length_max"] = None
    return stats


def print_analysis(stats: Dict, info: Dict):
    """打印分析结果到终端。"""
    print("\n" + "=" * 60)
    print("EEF 轨迹分析结果")
    print("=" * 60)
    print(f"FPS: {info.get('fps', 'N/A')}")
    print(f"Episodes 数量: {len(stats['episodes'])}")
    for key in ["robot0_eef_pos", "robot1_eef_pos"]:
        g = stats["global"].get(key)
        if not g or g["min"] is None:
            continue
        print(f"\n{key}:")
        print(f"  全局范围: min={g['min']}, max={g['max']}")
        print(f"  轨迹长度: mean={g['length_mean']:.1f}, min={g['length_min']}, max={g['length_max']}")
    print("\n各 Episode 摘要:")
    for ep_idx, ep_stat in list(stats["episodes"].items())[:10]:
        for k, v in ep_stat.items():
            print(f"  Episode {ep_idx} {k}: len={v['length']}, start_end_dist={v['start_end_distance']:.4f}")
    if len(stats["episodes"]) > 10:
        print("  ...")
    print("=" * 60)


def plot_2d_projections(
    trajectories: Dict[int, Dict[str, np.ndarray]],
    output_dir: Path,
    max_episodes_plot: int = 5,
):
    """绘制 2D 投影（XY, XZ, YZ）与位置-时间曲线。"""
    if plt is None:
        print("跳过 2D 图: 未安装 matplotlib")
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max_episodes_plot))
    for robot_key in ["robot0_eef_pos", "robot1_eef_pos"]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # XY
        ax = axes[0, 0]
        for i, (ep_idx, traj_dict) in enumerate(list(trajectories.items())[:max_episodes_plot]):
            if robot_key not in traj_dict:
                continue
            pos = traj_dict[robot_key]
            ax.plot(pos[:, 0], pos[:, 1], color=colors[i], label=f"Ep {ep_idx}", alpha=0.8)
            ax.scatter(pos[0, 0], pos[0, 1], color=colors[i], s=50, marker="o", zorder=5)
            ax.scatter(pos[-1, 0], pos[-1, 1], color=colors[i], s=50, marker="s", zorder=5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{robot_key} - XY")
        ax.legend(loc="best", fontsize=8)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        # XZ
        ax = axes[0, 1]
        for i, (ep_idx, traj_dict) in enumerate(list(trajectories.items())[:max_episodes_plot]):
            if robot_key not in traj_dict:
                continue
            pos = traj_dict[robot_key]
            ax.plot(pos[:, 0], pos[:, 2], color=colors[i], label=f"Ep {ep_idx}", alpha=0.8)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{robot_key} - XZ")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        # YZ
        ax = axes[1, 0]
        for i, (ep_idx, traj_dict) in enumerate(list(trajectories.items())[:max_episodes_plot]):
            if robot_key not in traj_dict:
                continue
            pos = traj_dict[robot_key]
            ax.plot(pos[:, 1], pos[:, 2], color=colors[i], label=f"Ep {ep_idx}", alpha=0.8)
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{robot_key} - YZ")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        # 第一个 episode 的 X,Y,Z vs time
        ax = axes[1, 1]
        ep_idx, traj_dict = next(iter(trajectories.items()), (None, None))
        if traj_dict and robot_key in traj_dict:
            pos = traj_dict[robot_key]
            t = np.arange(len(pos)) / 60.0
            ax.plot(t, pos[:, 0], label="X", alpha=0.8)
            ax.plot(t, pos[:, 1], label="Y", alpha=0.8)
            ax.plot(t, pos[:, 2], label="Z", alpha=0.8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Position (m)")
            ax.set_title(f"{robot_key} - Episode {ep_idx} position vs time")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle(f"EEF trajectory analysis - {robot_key}", fontsize=12)
        plt.tight_layout()
        out_file = output_dir / f"eef_2d_{robot_key}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_file}")


def plot_3d_trajectories(
    trajectories: Dict[int, Dict[str, np.ndarray]],
    output_dir: Path,
    max_episodes_plot: int = 8,
):
    """用 Plotly 绘制 3D 轨迹（可交互）。"""
    if go is None:
        print("跳过 3D 图: 未安装 plotly")
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = [
        "rgb(31,119,180)", "rgb(255,127,14)", "rgb(44,160,44)", "rgb(214,39,40)",
        "rgb(148,103,189)", "rgb(140,86,75)", "rgb(227,119,194)", "rgb(127,127,127)",
    ]
    for robot_key in ["robot0_eef_pos", "robot1_eef_pos"]:
        fig = go.Figure()
        for i, (ep_idx, traj_dict) in enumerate(list(trajectories.items())[:max_episodes_plot]):
            if robot_key not in traj_dict:
                continue
            pos = traj_dict[robot_key]
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=2, color=color),
                name=f"Episode {ep_idx}",
            ))
            fig.add_trace(go.Scatter3d(
                x=[pos[0, 0]],
                y=[pos[0, 1]],
                z=[pos[0, 2]],
                mode="markers",
                marker=dict(size=10, color=color, symbol="diamond"),
                name=f"Ep{ep_idx} Start",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter3d(
                x=[pos[-1, 0]],
                y=[pos[-1, 1]],
                z=[pos[-1, 2]],
                mode="markers",
                marker=dict(size=10, color=color, symbol="square"),
                name=f"Ep{ep_idx} End",
                showlegend=False,
            ))
        fig.update_layout(
            title=f"3D EEF trajectory - {robot_key}",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
            ),
            width=900,
            height=700,
            showlegend=True,
        )
        out_file = output_dir / f"eef_3d_{robot_key}.html"
        fig.write_html(str(out_file), include_plotlyjs=True, full_html=True)
        print(f"  已保存: {out_file}")
    # 双臂一起显示（robot0 + robot1）
    fig = go.Figure()
    for i, (ep_idx, traj_dict) in enumerate(list(trajectories.items())[:max_episodes_plot]):
        for robot_key, label in [("robot0_eef_pos", "R0"), ("robot1_eef_pos", "R1")]:
            if robot_key not in traj_dict:
                continue
            pos = traj_dict[robot_key]
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(size=2, color=color),
                name=f"Ep{ep_idx} {label}",
            ))
    fig.update_layout(
        title="3D EEF trajectories - both arms",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        width=900,
        height=700,
        showlegend=True,
    )
    out_file = output_dir / "eef_3d_both_arms.html"
    fig.write_html(str(out_file), include_plotlyjs=True, full_html=True)
    print(f"  已保存: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="分析并可视化 UMI LeRobot 数据集的 eef_pos 轨迹",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_two_stage_28_02_clean",
        help="LeRobot 数据集根目录（含 meta/ 与 data/）",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="要分析的 episode 下标；默认取前 20 个",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=20,
        help="未指定 --episodes 时加载的 episode 数量",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="图表输出目录；默认为 dataset 下的 eef_trajectory_plots",
    )
    parser.add_argument(
        "--no-2d",
        action="store_true",
        help="不生成 2D 投影图",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="不生成 3D 交互图",
    )
    parser.add_argument(
        "--position-index",
        type=int,
        default=0,
        help="每帧 eef_pos (2,3) 中使用的索引，0=当前帧",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"数据集路径不存在: {dataset_path}")
    meta_path = dataset_path / "meta"
    data_path = dataset_path / "data"
    if not meta_path.exists() or not data_path.exists():
        raise SystemExit(f"数据集需包含 meta/ 与 data/ 目录: {dataset_path}")

    info, episodes = load_episodes_meta(meta_path)
    total = info.get("total_episodes", len(episodes))
    if args.episodes is not None:
        episode_indices = list(args.episodes)
    else:
        episode_indices = list(range(min(args.max_episodes, total)))
    fps = float(info.get("fps", 60))

    print(f"加载 {len(episode_indices)} 个 episodes 的 eef 轨迹: {episode_indices[:10]}{'...' if len(episode_indices) > 10 else ''}")
    trajectories = load_episode_eef_trajectories(
        data_path,
        info,
        episode_indices,
        position_index=args.position_index,
    )
    stats = analyze_trajectories(trajectories, fps=fps)
    print_analysis(stats, info)

    output_dir = Path(args.output_dir) if args.output_dir else dataset_path / "eef_trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eef_trajectory_stats.json", "w") as f:
        # 转为可序列化
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_serializable(x) for x in obj]
            return obj

        json.dump(to_serializable(stats), f, indent=2, ensure_ascii=False)
    print(f"统计已保存: {output_dir / 'eef_trajectory_stats.json'}")

    if not args.no_2d:
        plot_2d_projections(trajectories, output_dir)
    if not args.no_3d:
        plot_3d_trajectories(trajectories, output_dir)

    print("\n完成。")


if __name__ == "__main__":
    main()
