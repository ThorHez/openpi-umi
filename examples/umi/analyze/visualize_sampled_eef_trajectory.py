#!/usr/bin/env python3
"""
可视化经 sampler_v2 处理后的 EEF 轨迹。

说明：sampler 对 EEF 做的是「时间索引 + 插值」，不是低通滤波/平滑。
因此「当前步」观测 ≈ 原始轨迹在该帧的值，毛刺会原样保留。
详见同目录 SAMPLER_EEF_FILTERING_EXPLAINED.md。

用法:
    python visualize_sampled_eef_trajectory.py --zarr /path/to/dataset.zarr.zip
    python visualize_sampled_eef_trajectory.py --zarr ... --max-episodes 3 --compare-raw
    python visualize_sampled_eef_trajectory.py --zarr ... --smooth-window 15  # 对轨迹做滑动平均对比
    python visualize_sampled_eef_trajectory.py --zarr ... --eef-index 0   # 取 [0]（3 帧前，仅滞后）
    python visualize_sampled_eef_trajectory.py --zarr ... --eef-index mean  # 取 horizon 均值，轻微平滑
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import zarr

# 将项目根加入 path 以便导入 examples.umi.sampler_v2
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from examples.umi.sampler_v2 import (
    SequenceSampler,
    DEFAULT_KEY_HORIZON,
    DEFAULT_KEY_LATENCY_STEPS,
    DEFAULT_KEY_DOWN_SAMPLE_STEPS,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


EEF_POS_KEYS = ["robot0_eef_pos", "robot1_eef_pos"]


def smooth_trajectory(pos: np.ndarray, window: int) -> np.ndarray:
    """对 (T, 3) 轨迹做滑动平均，边界用 valid 方式（不 padding）。"""
    if window < 2 or len(pos) < window:
        return pos.copy()
    kernel = np.ones(window) / window
    out = np.zeros_like(pos)
    for d in range(pos.shape[1]):
        out[:, d] = np.convolve(pos[:, d], kernel, mode="same")
    return out.astype(np.float64)


def load_zarr(zarr_path: str, tmp_dir: str | None = None):
    """加载 zarr：.zarr.zip 解压到临时目录后打开。返回 (root, tmp_dir_or_none)。"""
    path = Path(zarr_path)
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {zarr_path}")
    if str(zarr_path).endswith(".zip"):
        tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="zarr_sampled_eef_")
        print(f"解压 zarr 到: {tmp_dir}")
        with zipfile.ZipFile(zarr_path, "r") as zf:
            zf.extractall(tmp_dir)
        root = zarr.open(tmp_dir, mode="r")
        return root, tmp_dir
    root = zarr.open(zarr_path, mode="r")
    return root, None


def _to_2d_array(arr: np.ndarray) -> np.ndarray:
    """将 zarr 读出的数组变为 (T, D)：若为 (T, H, D) 则取 [:, -1, :]。"""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if len(arr) and arr.size % 3 == 0:
            return arr.reshape(-1, 3).astype(np.float32)
        return arr.reshape(-1, 1).astype(np.float32)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim >= 3:
        return arr[:, -1, :].astype(np.float32)
    return arr.astype(np.float32)


def build_replay_buffer_from_zarr(data: zarr.Group, lowdim_keys: List[str]) -> Dict[str, np.ndarray]:
    """从 zarr data group 构建 sampler 所需的 replay_buffer（全长度、2D）。"""
    replay_buffer = {}
    for key in lowdim_keys:
        if key not in data:
            continue
        arr = np.array(data[key][:])
        replay_buffer[key] = _to_2d_array(arr)
    return replay_buffer


def get_episode_ranges(meta: zarr.Group):
    """从 meta['episode_ends'] 得到每个 episode 的 (start, end)。"""
    episode_ends = np.array(meta["episode_ends"][:])
    ranges = []
    start = 0
    for end in episode_ends:
        ranges.append((int(start), int(end)))
        start = end
    return ranges


def collect_sampled_eef_trajectories(
    sampler: SequenceSampler,
    episode_ranges: List[tuple],
    max_episodes: int | None = None,
    max_frames_per_episode: int | None = None,
    step: int = 1,
    eef_index: int | str = -1,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    对每个 episode，遍历其对应的 sample 下标，调用 sample_sequence 取观测中的 EEF，
    得到该 episode 的轨迹。
    eef_index: -1=当前帧 [current_idx]，0=前一时刻 [current_idx-3]（仅滞后，不消毛刺），"mean"=horizon 内均值（轻微平滑）。
    step: 每隔 step 个 frame 采样一次（减少计算量）。
    """
    # 将 sampler 的 indices 按 (start_idx, end_idx) 分组到 episode
    ep_to_indices: Dict[tuple, List[int]] = {}
    for idx in range(len(sampler)):
        _, start_idx, end_idx, _ = sampler.indices[idx]
        key = (start_idx, end_idx)
        if key not in ep_to_indices:
            ep_to_indices[key] = []
        ep_to_indices[key].append(idx)

    trajectories = {}
    ranges_used = episode_ranges[: max_episodes] if max_episodes else episode_ranges
    for ep_idx, (start, end) in enumerate(ranges_used):
        key = (start, end)
        if key not in ep_to_indices:
            continue
        indices_in_ep = sorted(ep_to_indices[key])
        if step > 1:
            indices_in_ep = indices_in_ep[::step]
        if max_frames_per_episode is not None:
            indices_in_ep = indices_in_ep[: max_frames_per_episode]

        pos_lists: Dict[str, List[np.ndarray]] = {k: [] for k in EEF_POS_KEYS}
        for sample_idx in indices_in_ep:
            result = sampler.sample_sequence(sample_idx)
            for k in EEF_POS_KEYS:
                if k not in result:
                    continue
                obs = np.asarray(result[k], dtype=np.float64)  # (horizon, 3)
                if eef_index == "mean":
                    pos_lists[k].append(np.mean(obs, axis=0))
                elif eef_index == 0:
                    pos_lists[k].append(obs[0])  # 3 帧前，仅滞后
                else:
                    pos_lists[k].append(obs[-1])  # 当前帧，默认
        trajectories[ep_idx] = {
            k: np.stack(pos_lists[k], axis=0) if pos_lists[k] else np.zeros((0, 3), dtype=np.float64)
            for k in EEF_POS_KEYS
        }
    return trajectories


def collect_raw_trajectories(
    data: zarr.Group,
    episode_ranges: List[tuple],
    eef_keys: List[str],
    max_episodes: int | None = None,
    max_frames_per_episode: int | None = None,
    step: int = 1,
) -> Dict[int, Dict[str, np.ndarray]]:
    """从 zarr 直接取原始 EEF 轨迹（与滤波对比用）。"""
    trajectories = {}
    ranges_used = episode_ranges[: max_episodes] if max_episodes else episode_ranges
    for ep_idx, (start, end) in enumerate(ranges_used):
        out = {}
        for k in eef_keys:
            if k not in data:
                out[k] = np.zeros((0, 3), dtype=np.float64)
                continue
            arr = np.array(data[k][start:end])
            if arr.ndim == 3:
                arr = arr[:, -1, :3]
            elif arr.ndim == 2 and arr.shape[-1] >= 3:
                arr = arr[..., :3]
            else:
                arr = np.zeros((0, 3), dtype=np.float64)
            if step > 1:
                arr = arr[::step]
            if max_frames_per_episode is not None:
                arr = arr[: max_frames_per_episode]
            out[k] = arr.astype(np.float64)
        trajectories[ep_idx] = out
    return trajectories


def plot_2d_sampled(
    sampled: Dict[int, Dict[str, np.ndarray]],
    raw: Dict[int, Dict[str, np.ndarray]] | None,
    output_dir: Path,
    max_episodes: int = 8,
    smoothed: Dict[int, Dict[str, np.ndarray]] | None = None,
):
    """绘制 2D 投影：sampler 输出轨迹；若提供 raw 则叠加原始；若提供 smoothed 则叠加平滑后（对比用）。"""
    if not _HAS_MPL:
        print("跳过 2D 图: 未安装 matplotlib")
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max_episodes))
    items = list(sampled.items())[:max_episodes]
    for robot_key in EEF_POS_KEYS:
        if not any(robot_key in t for t in sampled.values()):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # XY
        ax = axes[0, 0]
        for i, (ep_idx, traj_dict) in enumerate(items):
            if robot_key not in traj_dict or len(traj_dict[robot_key]) == 0:
                continue
            pos = traj_dict[robot_key]
            ax.plot(pos[:, 0], pos[:, 1], color=colors[i], label=f"Ep {ep_idx} (sampled)", alpha=0.9, lw=1.5)
            ax.scatter(pos[0, 0], pos[0, 1], color=colors[i], s=40, marker="o", zorder=5)
            ax.scatter(pos[-1, 0], pos[-1, 1], color=colors[i], s=40, marker="s", zorder=5)
            if raw and ep_idx in raw and robot_key in raw[ep_idx] and len(raw[ep_idx][robot_key]) > 0:
                r = raw[ep_idx][robot_key]
                ax.plot(r[:, 0], r[:, 1], color=colors[i], alpha=0.25, linestyle="--", label=f"Ep {ep_idx} (raw)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{robot_key} - XY (sampled)")
        ax.legend(loc="best", fontsize=7)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for i, (ep_idx, traj_dict) in enumerate(items):
            if robot_key not in traj_dict or len(traj_dict[robot_key]) == 0:
                continue
            pos = traj_dict[robot_key]
            ax.plot(pos[:, 0], pos[:, 2], color=colors[i], label=f"Ep {ep_idx}", alpha=0.9)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{robot_key} - XZ")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for i, (ep_idx, traj_dict) in enumerate(items):
            if robot_key not in traj_dict or len(traj_dict[robot_key]) == 0:
                continue
            pos = traj_dict[robot_key]
            ax.plot(pos[:, 1], pos[:, 2], color=colors[i], label=f"Ep {ep_idx}", alpha=0.9)
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{robot_key} - YZ")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if items:
            ep_idx, traj_dict = items[0]
            if robot_key in traj_dict and len(traj_dict[robot_key]) > 0:
                pos = traj_dict[robot_key]
                t = np.arange(len(pos))
                ax.plot(t, pos[:, 0], label="X (sampled)", alpha=0.8)
                ax.plot(t, pos[:, 1], label="Y (sampled)", alpha=0.8)
                ax.plot(t, pos[:, 2], label="Z (sampled)", alpha=0.8)
                if smoothed and ep_idx in smoothed and robot_key in smoothed[ep_idx] and len(smoothed[ep_idx][robot_key]) > 0:
                    sm = smoothed[ep_idx][robot_key]
                    ax.plot(t[: len(sm)], sm[:, 0], label="X (smooth)", alpha=0.9, linestyle="--")
                    ax.plot(t[: len(sm)], sm[:, 1], label="Y (smooth)", alpha=0.9, linestyle="--")
                    ax.plot(t[: len(sm)], sm[:, 2], label="Z (smooth)", alpha=0.9, linestyle="--")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Position (m)")
                ax.set_title(f"{robot_key} - Ep {ep_idx} vs time (sampled=插值, 非平滑)")
                ax.legend(loc="best", fontsize=7)
                ax.grid(True, alpha=0.3)
        title = "EEF trajectory (sampler 输出=插值取值，无低通滤波)"
        if smoothed:
            title += " + 滑动平均对比"
        plt.suptitle(f"{title} - {robot_key}", fontsize=11)
        plt.tight_layout()
        out_file = output_dir / f"eef_2d_sampled_{robot_key}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_file}")


def plot_3d_sampled(
    sampled: Dict[int, Dict[str, np.ndarray]],
    output_dir: Path,
    max_episodes: int = 8,
):
    """绘制 3D 轨迹（matplotlib）。"""
    if not _HAS_MPL:
        return
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    output_dir = Path(output_dir)
    colors = plt.cm.tab10(np.linspace(0, 1, max_episodes))
    items = list(sampled.items())[:max_episodes]
    eef_keys = [k for k in EEF_POS_KEYS if any(k in t for t in sampled.values())]
    for robot_key in eef_keys:
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
        ax.set_title(f"3D EEF (after sampler) - {robot_key}")
        out_file = output_dir / f"eef_3d_sampled_{robot_key}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_file}")
    if len(eef_keys) >= 2:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for i, (ep_idx, traj_dict) in enumerate(items):
            for robot_key in ("robot0_eef_pos", "robot1_eef_pos"):
                if robot_key not in traj_dict or len(traj_dict[robot_key]) == 0:
                    continue
                pos = traj_dict[robot_key]
                label = "R0" if "robot0" in robot_key else "R1"
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=colors[i], label=f"Ep{ep_idx} {label}", alpha=0.8)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(loc="best", fontsize=8)
        ax.set_title("3D EEF (after sampler) - both arms")
        out_file = output_dir / "eef_3d_sampled_both_arms.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="可视化经 sampler_v2 滤波后的 EEF 轨迹",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--zarr",
        type=str,
        default="/root/openpi-umi/data/fold_clothes_with_depth/datasets_no_filter_20260312.zarr.zip",
        help="zarr 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录；默认 zarr 同目录下 eef_trajectories_sampled",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=5,
        help="最多处理的 episode 数",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="每个 episode 最多取多少帧（用于长 episode 加快）",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="每隔 step 帧采样一次（减小计算量）",
    )
    parser.add_argument(
        "--compare-raw",
        action="store_true",
        help="在 2D 图中叠加原始（未滤波）轨迹对比",
    )
    parser.add_argument(
        "--no-2d",
        action="store_true",
        help="不生成 2D 图",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="不生成 3D 图",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        metavar="N",
        help="对 sampler 输出轨迹做 N 帧滑动平均并画在 position-vs-time 子图中对比（说明：真正去毛刺需此类平滑，sampler 本身不做）",
    )
    parser.add_argument(
        "--eef-index",
        type=str,
        default="-1",
        choices=("-1", "0", "mean"),
        help="取观测的哪一维：-1=当前帧(默认)，0=3帧前(仅滞后不消毛刺)，mean=horizon 内均值(轻微平滑)",
    )
    args = parser.parse_args()

    root, tmp_dir = load_zarr(args.zarr)
    try:
        data = root["data"]
        meta = root["meta"]
    except KeyError as e:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise SystemExit(f"zarr 缺少必要组: {e}") from e

    # 构建 replay_buffer：只包含 zarr 中存在的、sampler 需要的 key
    required_for_sampler = [
        "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width",
        "robot1_eef_pos", "robot1_eef_rot_axis_angle", "robot1_gripper_width",
        "robot0_demo_start_pose", "robot1_demo_start_pose",
    ]
    lowdim_keys = [k for k in required_for_sampler if k in data]
    if "robot0_eef_pos" not in lowdim_keys or "robot0_gripper_width" not in lowdim_keys:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise SystemExit("zarr 中缺少 robot0_eef_pos 或 robot0_gripper_width，无法运行 sampler")

    replay_buffer = build_replay_buffer_from_zarr(data, lowdim_keys)
    episode_ends = np.array(meta["episode_ends"][:]).tolist()
    episode_ranges = get_episode_ranges(meta)

    sampler = SequenceSampler(
        replay_buffer=replay_buffer,
        episode_ends=episode_ends,
        key_horizon=DEFAULT_KEY_HORIZON,
        key_latency_steps=DEFAULT_KEY_LATENCY_STEPS,
        key_down_sample_steps=DEFAULT_KEY_DOWN_SAMPLE_STEPS,
        lowdim_keys=lowdim_keys,
        rgb_keys=[],
    )
    print(f"Sampler 总样本数: {len(sampler)}，episode 数: {len(episode_ranges)}")

    eef_index_arg = args.eef_index
    eef_index: int | str = int(eef_index_arg) if eef_index_arg in ("-1", "0") else "mean"
    if eef_index_arg == "0":
        print("说明: --eef-index 0 取的是 3 帧前的值，仅时间滞后，不能去毛刺。")
    elif eef_index_arg == "mean":
        print("说明: --eef-index mean 取 horizon 内均值，有轻微平滑效果。")
    print("正在通过 sampler 收集 EEF 轨迹（可能较慢）...")
    sampled_trajectories = collect_sampled_eef_trajectories(
        sampler,
        episode_ranges,
        max_episodes=args.max_episodes,
        max_frames_per_episode=args.max_frames_per_episode,
        step=args.step,
        eef_index=eef_index,
    )
    raw_trajectories = None
    if args.compare_raw:
        eef_keys = [k for k in EEF_POS_KEYS if k in data]
        raw_trajectories = collect_raw_trajectories(
            data,
            episode_ranges,
            eef_keys,
            max_episodes=args.max_episodes,
            max_frames_per_episode=args.max_frames_per_episode,
            step=args.step,
        )

    smoothed_trajectories = None
    if args.smooth_window and args.smooth_window >= 2:
        smoothed_trajectories = {}
        for ep_idx, traj_dict in sampled_trajectories.items():
            smoothed_trajectories[ep_idx] = {
                k: smooth_trajectory(traj_dict[k], args.smooth_window)
                for k in EEF_POS_KEYS
                if k in traj_dict and len(traj_dict[k]) > 0
            }
        print(f"已对轨迹做 {args.smooth_window} 帧滑动平均，用于对比（说明：sampler 本身不做平滑，故毛刺仍在）")

    out_path = Path(args.output_dir) if args.output_dir else Path(args.zarr).parent / "eef_trajectories_sampled"
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_path.absolute()}")

    if not args.no_2d:
        print("生成 2D 轨迹图...")
        plot_2d_sampled(
            sampled_trajectories,
            raw_trajectories,
            out_path,
            max_episodes=args.max_episodes,
            smoothed=smoothed_trajectories,
        )
    if not args.no_3d:
        print("生成 3D 轨迹图...")
        plot_3d_sampled(sampled_trajectories, out_path, max_episodes=args.max_episodes)

    for ext in ("*.png", "*.html"):
        for f in out_path.glob(ext):
            print(f"  可视化: {f.name}")

    if tmp_dir and Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("完成。")


if __name__ == "__main__":
    main()
