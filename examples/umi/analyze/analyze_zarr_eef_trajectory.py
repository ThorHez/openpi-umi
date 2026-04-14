#!/usr/bin/env python3
"""
分析并可视化 UMI zarr 数据集的机械臂末端执行器（EEF）轨迹。

从 zarr 中读取 robot0_eef_pos、robot1_eef_pos（若存在），按 episode 切分，
输出 CSV/NPZ、控制台摘要，并生成 2D/3D 轨迹图。

用法:
    python analyze_zarr_eef_trajectory.py --zarr /path/to/dataset.zarr.zip
    python analyze_zarr_eef_trajectory.py --zarr /path/to/dataset.zarr.zip --output-dir ./eef_trajectories
    python analyze_zarr_eef_trajectory.py --zarr /path/to/dataset.zarr.zip --max-episodes 5 --no-3d
"""

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import zarr

# 可选：可视化
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


# EEF 位置在 data 中的 key，支持单臂/双臂
EEF_POS_KEYS = ["robot0_eef_pos", "robot1_eef_pos"]


def load_zarr(zarr_path: str, tmp_dir: str | None = None):
    """
    加载 zarr：.zarr.zip 解压到临时目录后打开，目录则直接打开。
    返回 (root, tmp_dir_or_none)，若创建了临时目录则需调用方在结束后删除。
    """
    zarr_path = str(zarr_path)
    path = Path(zarr_path)
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {zarr_path}")

    if zarr_path.endswith(".zip"):
        tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="zarr_eef_")
        print(f"解压 zarr 到: {tmp_dir}")
        with zipfile.ZipFile(zarr_path, "r") as zf:
            zf.extractall(tmp_dir)
        root = zarr.open(tmp_dir, mode="r")
        return root, tmp_dir
    else:
        root = zarr.open(zarr_path, mode="r")
        return root, None


def get_episode_ranges(meta: zarr.Group):
    """从 meta['episode_ends'] 得到每个 episode 的 (start, end) 索引。"""
    episode_ends = np.array(meta["episode_ends"][:])
    ranges = []
    start = 0
    for end in episode_ends:
        ranges.append((int(start), int(end)))
        start = end
    return ranges


def extract_eef_trajectory_slice(arr: np.ndarray, start: int, end: int):
    """
    从 data[key] 中取 [start:end]，得到 (T,) 或 (T, 3) 或 (T, H, 3) 等。
    若最后一维是 3，视为 (x,y,z)；若有 H 维，取最后一帧（当前步）作为位置。
    返回 (T, 3) 的 numpy 数组。
    """
    slab = np.array(arr[start:end])
    if slab.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    # 展平为 (T, ...)
    if slab.ndim == 1:
        # 可能是 (T*3,) 或单维，尽量变成 (T,3)
        n = len(slab)
        if n % 3 == 0:
            return slab.reshape(-1, 3).astype(np.float64)
        return np.zeros((0, 3), dtype=np.float64)
    if slab.ndim == 2:
        if slab.shape[-1] == 3:
            return slab.astype(np.float64)
        if slab.shape[-1] == 6:
            return slab[..., :3].astype(np.float64)
        return np.zeros((0, 3), dtype=np.float64)
    if slab.ndim >= 3:
        # (T, H, 3) 或 (T, ..., 3)：取最后一维为 3 的，再取 last step
        if slab.shape[-1] == 3:
            last_dim = slab.shape[-2]  # e.g. horizon
            return slab[:, -1, :].astype(np.float64)  # (T, 3)
        return np.zeros((0, 3), dtype=np.float64)
    return np.zeros((0, 3), dtype=np.float64)


def plot_2d_trajectories(
    trajectories: Dict[int, Dict[str, np.ndarray]],
    output_dir: Path,
    max_episodes: int = 8,
):
    """绘制 2D 投影（XY/XZ/YZ）与位置-时间曲线。"""
    if not _HAS_MPL:
        print("跳过 2D 图: 未安装 matplotlib")
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max_episodes))
    items = list(trajectories.items())[:max_episodes]
    for robot_key in EEF_POS_KEYS:
        if not any(robot_key in t for t in trajectories.values()):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        for i, (ep_idx, traj_dict) in enumerate(items):
            if robot_key not in traj_dict:
                continue
            pos = traj_dict[robot_key]
            if len(pos) == 0:
                continue
            ax.plot(pos[:, 0], pos[:, 1], color=colors[i], label=f"Ep {ep_idx}", alpha=0.8)
            ax.scatter(pos[0, 0], pos[0, 1], color=colors[i], s=50, marker="o", zorder=5)
            ax.scatter(pos[-1, 0], pos[-1, 1], color=colors[i], s=50, marker="s", zorder=5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{robot_key} - XY")
        ax.legend(loc="best", fontsize=8)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for i, (ep_idx, traj_dict) in enumerate(items):
            if robot_key not in traj_dict:
                continue
            pos = traj_dict[robot_key]
            if len(pos) == 0:
                continue
            ax.plot(pos[:, 0], pos[:, 2], color=colors[i], label=f"Ep {ep_idx}", alpha=0.8)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{robot_key} - XZ")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for i, (ep_idx, traj_dict) in enumerate(items):
            if robot_key not in traj_dict:
                continue
            pos = traj_dict[robot_key]
            if len(pos) == 0:
                continue
            ax.plot(pos[:, 1], pos[:, 2], color=colors[i], label=f"Ep {ep_idx}", alpha=0.8)
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{robot_key} - YZ")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if items:
            ep_idx, traj_dict = items[0]
            if robot_key in traj_dict:
                pos = traj_dict[robot_key]
                if len(pos) > 0:
                    t = np.arange(len(pos))
                    ax.plot(t, pos[:, 0], label="X", alpha=0.8)
                    ax.plot(t, pos[:, 1], label="Y", alpha=0.8)
                    ax.plot(t, pos[:, 2], label="Z", alpha=0.8)
                    ax.set_xlabel("Frame")
                    ax.set_ylabel("Position (m)")
                    ax.set_title(f"{robot_key} - Ep {ep_idx} vs time")
                    ax.legend(loc="best", fontsize=8)
                    ax.grid(True, alpha=0.3)
        plt.suptitle(f"EEF trajectory - {robot_key}", fontsize=12)
        plt.tight_layout()
        out_file = output_dir / f"eef_2d_{robot_key}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_file}")


def _plot_3d_matplotlib(
    trajectories: Dict[int, Dict[str, np.ndarray]],
    output_dir: Path,
    max_episodes: int,
    eef_keys: List[str],
):
    """用 matplotlib mplot3d 绘制 3D 轨迹（无需 plotly）。"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    output_dir = Path(output_dir)
    colors = plt.cm.tab10(np.linspace(0, 1, max_episodes))
    items = list(trajectories.items())[:max_episodes]
    for robot_key in eef_keys:
        if not any(robot_key in t for t in trajectories.values()):
            continue
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for i, (ep_idx, traj_dict) in enumerate(items):
            if robot_key not in traj_dict or len(traj_dict[robot_key]) == 0:
                continue
            pos = traj_dict[robot_key]
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=colors[i], label=f"Ep {ep_idx}", alpha=0.8)
            ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color=colors[i], s=40, marker="o")
            ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color=colors[i], s=40, marker="s")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"3D EEF trajectory - {robot_key}")
        out_file = output_dir / f"eef_3d_{robot_key}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_file}")
    if len(eef_keys) >= 2:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for i, (ep_idx, traj_dict) in enumerate(items):
            for robot_key, label in [("robot0_eef_pos", "R0"), ("robot1_eef_pos", "R1")]:
                if robot_key not in traj_dict or len(traj_dict[robot_key]) == 0:
                    continue
                pos = traj_dict[robot_key]
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=colors[i], label=f"Ep{ep_idx} {label}", alpha=0.8)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(loc="best", fontsize=8)
        ax.set_title("3D EEF trajectories - both arms")
        out_file = output_dir / "eef_3d_both_arms.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_file}")


def plot_3d_trajectories(
    trajectories: Dict[int, Dict[str, np.ndarray]],
    output_dir: Path,
    max_episodes: int = 8,
):
    """绘制 3D 轨迹：优先 Plotly（交互 HTML），否则用 matplotlib。"""
    output_dir = Path(output_dir)
    eef_keys = [k for k in EEF_POS_KEYS if any(k in t for t in trajectories.values())]
    if not eef_keys:
        return
    if _HAS_PLOTLY:
        _plot_3d_plotly(trajectories, output_dir, max_episodes, eef_keys)
    elif _HAS_MPL:
        print("  使用 matplotlib 生成 3D 图（安装 plotly 可得到交互式 HTML）")
        _plot_3d_matplotlib(trajectories, output_dir, max_episodes, eef_keys)
    else:
        print("跳过 3D 图: 未安装 matplotlib / plotly")
        return


def _plot_3d_plotly(
    trajectories: Dict[int, Dict[str, np.ndarray]],
    output_dir: Path,
    max_episodes: int,
    eef_keys: List[str],
):
    """用 Plotly 绘制 3D 轨迹（可交互 HTML）。"""
    if not _HAS_PLOTLY:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = [
        "rgb(31,119,180)", "rgb(255,127,14)", "rgb(44,160,44)", "rgb(214,39,40)",
        "rgb(148,103,189)", "rgb(140,86,75)", "rgb(227,119,194)", "rgb(127,127,127)",
    ]
    items = list(trajectories.items())[:max_episodes]
    for robot_key in eef_keys:
        fig = go.Figure()
        for i, (ep_idx, traj_dict) in enumerate(items):
            if robot_key not in traj_dict:
                continue
            pos = traj_dict[robot_key]
            if len(pos) == 0:
                continue
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=2, color=color),
                name=f"Episode {ep_idx}",
            ))
            fig.add_trace(go.Scatter3d(
                x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]],
                mode="markers",
                marker=dict(size=10, color=color, symbol="diamond"),
                name=f"Ep{ep_idx} Start",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter3d(
                x=[pos[-1, 0]], y=[pos[-1, 1]], z=[pos[-1, 2]],
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

    # 双臂一起（仅当存在两个臂时）
    if len(eef_keys) < 2:
        return
    fig = go.Figure()
    for i, (ep_idx, traj_dict) in enumerate(items):
        for robot_key, label in [("robot0_eef_pos", "R0"), ("robot1_eef_pos", "R1")]:
            if robot_key not in traj_dict:
                continue
            pos = traj_dict[robot_key]
            if len(pos) == 0:
                continue
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
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


def analyze_and_output_trajectories(
    zarr_path: str,
    output_dir: str | Path | None = None,
    max_episodes: int | None = None,
    output_format: str = "both",
    no_2d: bool = False,
    no_3d: bool = False,
    max_episodes_plot: int = 8,
):
    """
    分析 zarr，按 episode 输出 EEF 轨迹，并可选生成 2D/3D 可视化。
    output_format: "csv" | "npz" | "both"
    """
    root, tmp_dir = load_zarr(zarr_path)
    try:
        data = root["data"]
        meta = root["meta"]
    except KeyError as e:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise KeyError(f"zarr 缺少必要组: {e}") from e

    # 存在的 EEF key
    eef_keys = [k for k in EEF_POS_KEYS if k in data]
    if not eef_keys:
        print("未找到 robot0_eef_pos / robot1_eef_pos，data 中的 key:", list(data.keys()))
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    episode_ranges = get_episode_ranges(meta)
    num_episodes = len(episode_ranges)
    if max_episodes is not None:
        episode_ranges = episode_ranges[: max_episodes]
        num_episodes = len(episode_ranges)

    out_path = Path(output_dir) if output_dir else Path(zarr_path).parent / "eef_trajectories"
    out_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "zarr_path": str(zarr_path),
        "num_episodes": num_episodes,
        "eef_keys": eef_keys,
        "episodes": [],
    }
    all_trajectories: Dict[int, Dict[str, np.ndarray]] = {}

    for ep_idx, (start, end) in enumerate(episode_ranges):
        T = end - start
        ep_info = {"episode": ep_idx, "start": start, "end": end, "length": T}
        trajectories = {}

        for key in eef_keys:
            arr = data[key]
            pos = extract_eef_trajectory_slice(arr, start, end)  # (T, 3)
            trajectories[key] = pos
            if pos.size > 0:
                ep_info[f"{key}_shape"] = list(pos.shape)
                ep_info[f"{key}_min"] = np.min(pos, axis=0).tolist()
                ep_info[f"{key}_max"] = np.max(pos, axis=0).tolist()

        all_trajectories[ep_idx] = trajectories
        summary["episodes"].append(ep_info)

        # 输出该 episode 的轨迹文件
        ep_prefix = out_path / f"episode_{ep_idx:06d}"
        if output_format in ("csv", "both"):
            for key in eef_keys:
                pos = trajectories[key]
                if pos.size == 0:
                    continue
                csv_path = f"{ep_prefix}_{key}.csv"
                header = "t,x,y,z"
                np.savetxt(
                    csv_path,
                    np.column_stack([np.arange(len(pos)), pos]),
                    delimiter=",",
                    header=header,
                    comments="",
                    fmt="%.6f",
                )
        if output_format in ("npz", "both"):
            npz_data = {k: trajectories[k] for k in eef_keys}
            np.savez(f"{ep_prefix}_eef.npz", **npz_data)

    # 汇总 JSON
    summary_path = out_path / "eef_trajectory_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 控制台：简要输出轨迹信息
    print("\n" + "=" * 60)
    print("EEF 轨迹分析结果")
    print("=" * 60)
    print(f"Zarr: {zarr_path}")
    print(f"Episode 数: {num_episodes}")
    print(f"EEF keys: {eef_keys}")
    print(f"输出目录: {out_path.absolute()}")
    print(f"摘要: {summary_path}")
    print()
    for ep_idx, (start, end) in enumerate(episode_ranges[:10]):
        T = end - start
        line = f"  Episode {ep_idx}: frames [{start}:{end}] 长度={T}"
        for key in eef_keys:
            pos = np.array(data[key][start:end])
            if pos.size > 0 and pos.ndim >= 2 and pos.shape[-1] >= 3:
                if pos.ndim == 3:
                    pos = pos[:, -1, :3]
                else:
                    pos = pos[..., :3]
                line += f"  {key} shape={pos.shape}"
        print(line)
    if num_episodes > 10:
        print(f"  ... 共 {num_episodes} 个 episodes（仅显示前 10）")
    print("=" * 60)

    # 2D/3D 可视化（与 CSV/NPZ 同目录，便于查看）
    if all_trajectories:
        if not no_2d:
            print("\n生成 2D 轨迹图...")
            plot_2d_trajectories(all_trajectories, out_path, max_episodes_plot=max_episodes_plot)
        if not no_3d:
            print("\n生成 3D 轨迹图...")
            plot_3d_trajectories(all_trajectories, out_path, max_episodes_plot=max_episodes_plot)
        # 列出生成的可视化文件
        for ext in ("*.png", "*.html"):
            for f in out_path.glob(ext):
                print(f"  可视化: {f.name}")

    if tmp_dir and Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="分析 zarr 数据集并输出机械臂 EEF 轨迹",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--zarr",
        type=str,
        default="/root/openpi-umi/data/fold_clothes_with_depth/datasets_no_filter_20260312.zarr.zip",
        help="zarr 文件路径（.zarr 或 .zarr.zip）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="轨迹输出目录；默认在 zarr 同目录下 eef_trajectories",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最多处理的 episode 数，不设则处理全部",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "npz", "both"],
        default="both",
        help="输出格式",
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
        "--max-episodes-plot",
        type=int,
        default=8,
        help="参与绘图的 episode 数量（前 N 个）",
    )
    args = parser.parse_args()

    analyze_and_output_trajectories(
        zarr_path=args.zarr,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        output_format=args.format,
        no_2d=args.no_2d,
        no_3d=args.no_3d,
        max_episodes_plot=args.max_episodes_plot,
    )


if __name__ == "__main__":
    main()
