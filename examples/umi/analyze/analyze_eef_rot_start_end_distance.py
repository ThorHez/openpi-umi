#!/usr/bin/env python3
"""
分析双臂 robot{id}_eef_pos_wrt_start 的起始与结束欧几里得距离（首帧与末帧的差值）；
并检测双臂是否同步：当一臂首次开始移动时，在随后 1 秒（可配）时间窗内若另一臂无明显位移则判为不同步。
当检测到双臂不同步时，可将多路摄像头画面合成为一屏视频并保存；当首末帧位姿差异较大时，可将轨迹绘制为图片保存，便于调试。

用法:
    python analyze_eef_rot_start_end_distance.py --dataset /path/to/lerobot_dataset [--threshold 1.0] [--max-episodes N]
    python ... --sync-follow-window-sec 1.0 --sync-min-movement 0.03 --sync-max-still 0.01  # 双臂同步检测参数
    python ... --save-out-of-sync-videos --out-of-sync-videos-dir ./out_of_sync_videos   # 将不同步 episode 导出为可播放视频
    python ... --save-abnormal-trajectory-images --abnormal-trajectory-dir ./abnormal_trajectories  # 将首末帧差异大的 episode 轨迹导出为图片
"""

import argparse
import io
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def _to_pos(row_value):
    """将 parquet 一行的 eef_pos 转为单帧 3D 向量 (支持 (2,3) 取当前帧)。"""
    arr = np.asarray(row_value, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[-1]
    return arr.ravel()[:3]


def _get_episode_sync_displacements(
    table,
    pos_wrt_start_keys: list,
    sync_end_frame: int,
) -> Optional[tuple]:
    """
    从已打开的 table 中，用 robot{id}_eef_pos_wrt_start 计算前 sync_end_frame 帧内两臂位移（米）。
    返回 (disp_robot0, disp_robot1)，任一列缺失则返回 None。
    """
    if pos_wrt_start_keys[0] not in table.column_names or pos_wrt_start_keys[1] not in table.column_names:
        return None
    n = table.num_rows
    if n == 0:
        return None
    end_idx = min(sync_end_frame, n - 1)
    if end_idx <= 0:
        return None
    col0 = table.column(pos_wrt_start_keys[0]).to_pylist()
    col1 = table.column(pos_wrt_start_keys[1]).to_pylist()
    p0_start = _to_pos(col0[0])
    p0_end = _to_pos(col0[end_idx])
    p1_start = _to_pos(col1[0])
    p1_end = _to_pos(col1[end_idx])
    disp0 = float(np.linalg.norm(np.asarray(p0_end) - np.asarray(p0_start)))
    disp1 = float(np.linalg.norm(np.asarray(p1_end) - np.asarray(p1_start)))
    return disp0, disp1


def _check_bimanual_sync_by_first_move(
    table,
    pos_wrt_start_keys: list,
    fps: float,
    sync_min_movement: float,
    sync_max_still: float,
    follow_window_sec: float = 1.0,
) -> Optional[tuple]:
    """
    判断双臂是否同步：当一臂「首次开始移动」时，在随后 follow_window_sec 秒的时间窗内，
    若另一臂位移小于 sync_max_still，则视为不同步（一动一静）。
    返回 (out_of_sync, start_frame_0, start_frame_1, disp1_when_0_moved, disp0_when_1_moved) 或 None。
    """
    if pos_wrt_start_keys[0] not in table.column_names or pos_wrt_start_keys[1] not in table.column_names:
        return None
    n = table.num_rows
    if n == 0:
        return None
    col0 = table.column(pos_wrt_start_keys[0]).to_pylist()
    col1 = table.column(pos_wrt_start_keys[1]).to_pylist()
    p0 = np.array([_to_pos(col0[i]) for i in range(n)])
    p1 = np.array([_to_pos(col1[i]) for i in range(n)])
    disp0_from_start = np.linalg.norm(p0 - p0[0], axis=1)
    disp1_from_start = np.linalg.norm(p1 - p1[0], axis=1)

    start_frame_0 = None
    start_frame_1 = None
    for i in range(1, n):
        if start_frame_0 is None and disp0_from_start[i] > sync_min_movement:
            start_frame_0 = i
        if start_frame_1 is None and disp1_from_start[i] > sync_min_movement:
            start_frame_1 = i

    follow_frames = max(1, int(fps * follow_window_sec))
    out_of_sync = False
    disp1_when_0_moved = np.nan
    disp0_when_1_moved = np.nan

    if start_frame_0 is not None:
        end_idx = min(start_frame_0 + follow_frames, n - 1)
        # 1 秒窗结束后另一臂相对窗起点的位移作为判断
        disp1_when_0_moved = float(np.linalg.norm(p1[end_idx] - p1[start_frame_0]))
        if disp1_when_0_moved < sync_max_still:
            out_of_sync = True

    if start_frame_1 is not None:
        end_idx = min(start_frame_1 + follow_frames, n - 1)
        disp0_when_1_moved = float(np.linalg.norm(p0[end_idx] - p0[start_frame_1]))
        if disp0_when_1_moved < sync_max_still:
            out_of_sync = True

    return (
        out_of_sync,
        start_frame_0,
        start_frame_1,
        disp1_when_0_moved,
        disp0_when_1_moved,
    )


def _get_image_columns_for_video(table, info: Optional[dict] = None) -> List[str]:
    """
    获取用于合成视频的图像列名。优先从 info["features"] 中取 dtype==image 的列；
    若存在多路同一相机（如 left_wrist_0_rgb_0, left_wrist_0_rgb_1），只保留后缀 _0 的列以得到「每相机一路」。
    """
    if info and "features" in info:
        image_cols = [
            k for k, v in info["features"].items()
            if v.get("dtype") == "image" and k in table.column_names
        ]
    else:
        image_cols = [
            c for c in table.column_names
            if any(p in c.lower() for p in ("rgb", "wrist", "image", "camera", "base"))
        ]
    if not image_cols:
        return []
    # 只保留「每相机一路」：同名前缀下只保留以 _0 结尾的列（当前帧）
    with_0 = [c for c in image_cols if c.endswith("_0")]
    use = with_0 if with_0 else image_cols
    return sorted(use)


def _cell_to_image(cell) -> Optional[np.ndarray]:
    """将 parquet 单元格解码为 (H, W, 3) RGB uint8，支持 ndarray 或 dict with 'bytes'。"""
    if cell is None:
        return None
    if isinstance(cell, dict) and "bytes" in cell:
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(cell["bytes"]))
            return np.asarray(img.convert("RGB"))
        except Exception:
            return None
    arr = np.asarray(cell, dtype=np.uint8)
    if arr.ndim != 3:
        return None
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _plot_abnormal_trajectory(
    table,
    episode_index: int,
    robot_pos_wrt_start_keys: List[str],
    threshold: float,
    robot0_dist: float,
    robot1_dist: float,
    output_path: Path,
) -> bool:
    """
    将双臂 eef_pos_wrt_start 的轨迹绘制为图片并保存：平面投影 + 相对起点的距离随时间变化。
    返回是否成功。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if robot_pos_wrt_start_keys[0] not in table.column_names or robot_pos_wrt_start_keys[1] not in table.column_names:
        return False
    n = table.num_rows
    if n == 0:
        return False
    col0 = table.column(robot_pos_wrt_start_keys[0]).to_pylist()
    col1 = table.column(robot_pos_wrt_start_keys[1]).to_pylist()
    p0 = np.array([_to_pos(col0[i]) for i in range(n)])
    p1 = np.array([_to_pos(col1[i]) for i in range(n)])
    frames = np.arange(n)
    dist0_from_start = np.linalg.norm(p0 - p0[0], axis=1)
    dist1_from_start = np.linalg.norm(p1 - p1[0], axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # 第一行：XY / XZ / YZ 平面投影，双臂轨迹
    for ax, (xi, yi), (xlabel, ylabel) in zip(
        axes[0],
        [(0, 1), (0, 2), (1, 2)],
        [("X (m)", "Y (m)"), ("X (m)", "Z (m)"), ("Y (m)", "Z (m)")],
    ):
        ax.plot(p0[:, xi], p0[:, yi], "-", color="C0", label="robot0", linewidth=1.5, alpha=0.9)
        ax.plot(p1[:, xi], p1[:, yi], "-", color="C1", label="robot1", linewidth=1.5, alpha=0.9)
        ax.scatter(p0[0, xi], p0[0, yi], c="C0", s=80, marker="o", zorder=5, edgecolors="black")
        ax.scatter(p0[-1, xi], p0[-1, yi], c="C0", s=80, marker="s", zorder=5, edgecolors="black")
        ax.scatter(p1[0, xi], p1[0, yi], c="C1", s=80, marker="o", zorder=5, edgecolors="black")
        ax.scatter(p1[-1, xi], p1[-1, yi], c="C1", s=80, marker="s", zorder=5, edgecolors="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_title("XY")
    axes[0, 1].set_title("XZ")
    axes[0, 2].set_title("YZ")
    # 第二行：相对起点的距离随时间变化
    ax_d = axes[1, 0]
    ax_d.plot(frames, dist0_from_start, "-", color="C0", label=f"robot0 (首末距={robot0_dist:.4f}m)")
    ax_d.plot(frames, dist1_from_start, "-", color="C1", label=f"robot1 (首末距={robot1_dist:.4f}m)")
    ax_d.axhline(y=threshold, color="gray", linestyle="--", alpha=0.7, label=f"阈值={threshold}m")
    ax_d.set_xlabel("帧")
    ax_d.set_ylabel("相对首帧距离 (m)")
    ax_d.legend(loc="upper right", fontsize=8)
    ax_d.grid(True, alpha=0.3)
    # 第二行中：X、Z 分量随时间
    for i, (ax, label) in enumerate([(axes[1, 1], "X"), (axes[1, 2], "Z")]):
        ax.plot(frames, p0[:, i], "-", color="C0", alpha=0.8, label="robot0")
        ax.plot(frames, p1[:, i], "-", color="C1", alpha=0.8, label="robot1")
        ax.set_xlabel("帧")
        ax.set_ylabel(f"{label} (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[1, 1].set_title("X 分量")
    axes[1, 2].set_title("Z 分量")

    fig.suptitle(
        f"Episode {episode_index} 轨迹（首末帧位姿差异较大，阈值={threshold}m）\n"
        f"robot0 首末距={robot0_dist:.4f}m, robot1 首末距={robot1_dist:.4f}m",
        fontsize=11,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def _write_episode_composite_video(
    table,
    episode_index: int,
    output_path: Path,
    fps: float,
    image_columns: List[str],
    tile_height: int = 224,
    tile_width: int = 224,
) -> bool:
    """
    将多路摄像头画面按帧合成为一屏（横向拼接），写入可播放视频文件。
    返回是否成功。
    """
    if not image_columns:
        return False
    try:
        import imageio
    except ImportError:
        return False
    try:
        from PIL import Image as PILImage
    except ImportError:
        return False

    n_frames = table.num_rows
    if n_frames == 0:
        return False

    # 每路缩放到统一尺寸，横向拼接
    n_cams = len(image_columns)
    out_width = n_cams * tile_width
    out_height = tile_height
    cols_pylists = {col: table.column(col).to_pylist() for col in image_columns}

    try:
        writer = imageio.v2.get_writer(str(output_path), fps=float(fps))
    except AttributeError:
        writer = imageio.get_writer(str(output_path), fps=float(fps))
    for i in range(n_frames):
        row_images = []
        for col in image_columns:
            cell = cols_pylists[col][i]
            img = _cell_to_image(cell)
            if img is None:
                img = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            else:
                img = np.asarray(
                    PILImage.fromarray(img).resize((tile_width, tile_height), PILImage.BILINEAR)
                )
            row_images.append(img)
        frame = np.concatenate(row_images, axis=1)
        writer.append_data(frame)
    writer.close()
    return True


def analyze_eef_rot_start_end_distance(
    dataset_path: str,
    threshold: float = 1.0,
    max_episodes: Optional[int] = None,
    chunk_size: Optional[int] = None,
    check_sync: bool = True,
    sync_window_sec: float = 1.5,
    sync_follow_window_sec: float = 1.0,
    sync_min_movement: float = 0.03,
    sync_max_still: float = 0.01,
    save_out_of_sync_videos: bool = False,
    out_of_sync_videos_dir: Optional[str] = None,
    save_abnormal_trajectory_images: bool = False,
    abnormal_trajectory_dir: Optional[str] = None,
):
    """
    分析所有 episode 中 robot0/robot1 的 eef_pos_wrt_start 首帧与末帧的欧氏距离；
    若 check_sync 为 True，则检测双臂是否同步：当一臂首次开始移动时，在随后 sync_follow_window_sec 秒内
    若另一臂无明显位移（< sync_max_still）则视为不同步。
    若 save_out_of_sync_videos 为 True，会将不同步 episode 的多路摄像头画面合成为视频并保存到 out_of_sync_videos_dir。
    若 save_abnormal_trajectory_images 为 True，会将首末帧位姿差异较大的 episode 的轨迹绘制为图片并保存。
    """
    dataset_path = Path(dataset_path)
    meta_path = dataset_path / "meta" / "info.json"
    data_dir = dataset_path / "data"

    if not meta_path.exists():
        raise FileNotFoundError(f"info.json not found: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        info = json.load(f)

    videos_dir = Path(out_of_sync_videos_dir) if out_of_sync_videos_dir else (dataset_path / "out_of_sync_videos")
    if save_out_of_sync_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    trajectory_dir = None
    if save_abnormal_trajectory_images:
        trajectory_dir = Path(abnormal_trajectory_dir) if abnormal_trajectory_dir else (dataset_path / "abnormal_trajectories")
        trajectory_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = info["total_episodes"]
    fps = info.get("fps", 60)
    if chunk_size is None:
        chunk_size = info.get("chunks_size", 1000)
    robot_pos_wrt_start_keys = ["robot0_eef_pos_wrt_start", "robot1_eef_pos_wrt_start"]
    sync_end_frame = max(1, int(fps * sync_window_sec))

    # 确定要扫描的 episode 列表（按 chunk 目录）
    episodes_to_scan = list(range(total_episodes))
    if max_episodes is not None and max_episodes < total_episodes:
        episodes_to_scan = episodes_to_scan[:max_episodes]

    results = []
    abnormal_episodes = []
    out_of_sync_episodes = []

    for ep_idx in tqdm(episodes_to_scan, desc="Episodes"):
        chunk_idx = ep_idx // chunk_size
        episode_path = data_dir / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not episode_path.exists():
            results.append(
                {
                    "episode_index": ep_idx,
                    "robot0_dist": np.nan,
                    "robot1_dist": np.nan,
                    "abnormal": False,
                    "out_of_sync": False,
                    "robot0_sync_disp": np.nan,
                    "robot1_sync_disp": np.nan,
                    "error": "file_not_found",
                }
            )
            continue

        try:
            table = pq.read_table(episode_path)
        except Exception:
            results.append(
                {
                    "episode_index": ep_idx,
                    "robot0_dist": np.nan,
                    "robot1_dist": np.nan,
                    "abnormal": False,
                    "out_of_sync": False,
                    "robot0_sync_disp": np.nan,
                    "robot1_sync_disp": np.nan,
                    "error": "read_error",
                }
            )
            continue

        row = {
            "episode_index": ep_idx,
            "robot0_dist": np.nan,
            "robot1_dist": np.nan,
            "abnormal": False,
            "out_of_sync": False,
            "robot0_sync_disp": np.nan,
            "robot1_sync_disp": np.nan,
        }
        any_abnormal = False

        # 起止距离：eef_pos_wrt_start 首帧与末帧的欧氏距离
        for i, key in enumerate(robot_pos_wrt_start_keys):
            if key not in table.column_names:
                continue
            col = table.column(key).to_pylist()
            if not col:
                continue
            start_vec = _to_pos(col[0])
            end_vec = _to_pos(col[-1])
            dist = float(np.linalg.norm(np.asarray(end_vec) - np.asarray(start_vec)))
            row[f"robot{i}_dist"] = dist
            if dist > threshold:
                any_abnormal = True

        # 双臂同步：当一臂首次开始移动时，在随后 follow_window_sec 秒内另一臂若无明显位移则判为不同步
        if check_sync:
            sync_result = _check_bimanual_sync_by_first_move(
                table,
                robot_pos_wrt_start_keys,
                fps=fps,
                sync_min_movement=sync_min_movement,
                sync_max_still=sync_max_still,
                follow_window_sec=sync_follow_window_sec,
            )
            if sync_result is not None:
                out_of_sync_flag, start_f0, start_f1, disp1_when_0, disp0_when_1 = sync_result
                row["out_of_sync"] = out_of_sync_flag
                if out_of_sync_flag:
                    out_of_sync_episodes.append(ep_idx)
                    if save_out_of_sync_videos:
                        image_cols = _get_image_columns_for_video(table, info)
                        if image_cols:
                            out_path = videos_dir / f"episode_{ep_idx:06d}_out_of_sync.mp4"
                            if _write_episode_composite_video(
                                table, ep_idx, out_path, fps, image_cols
                            ):
                                row["out_of_sync_video"] = str(out_path)
                # 用于统计：取 episode 开头时间窗内位移（便于报告）
                sync_disp = _get_episode_sync_displacements(
                    table, robot_pos_wrt_start_keys, sync_end_frame
                )
                if sync_disp is not None:
                    row["robot0_sync_disp"], row["robot1_sync_disp"] = sync_disp

        row["abnormal"] = any_abnormal
        if any_abnormal:
            abnormal_episodes.append(ep_idx)
            if save_abnormal_trajectory_images and trajectory_dir is not None:
                r0_d = row["robot0_dist"] if not np.isnan(row.get("robot0_dist", np.nan)) else 0.0
                r1_d = row["robot1_dist"] if not np.isnan(row.get("robot1_dist", np.nan)) else 0.0
                out_path = trajectory_dir / f"episode_{ep_idx:06d}_abnormal_trajectory.png"
                if _plot_abnormal_trajectory(
                    table, ep_idx, robot_pos_wrt_start_keys, threshold, r0_d, r1_d, out_path
                ):
                    row["abnormal_trajectory_image"] = str(out_path)
        results.append(row)

    return {
        "results": results,
        "abnormal_episodes": abnormal_episodes,
        "out_of_sync_episodes": out_of_sync_episodes,
        "threshold": threshold,
        "total_episodes": len(episodes_to_scan),
        "robot_pos_wrt_start_keys": robot_pos_wrt_start_keys,
        "check_sync": check_sync,
        "sync_window_sec": sync_window_sec,
        "sync_follow_window_sec": sync_follow_window_sec,
        "sync_min_movement": sync_min_movement,
        "sync_max_still": sync_max_still,
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="分析双臂 eef_pos_wrt_start 首帧与末帧欧氏距离，检出未回起始位置的 episode；可选检测开头双臂同步。"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="LeRobot 数据集根目录（含 meta/info.json 和 data/chunk-*/episode_*.parquet）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="起止欧氏距离超过此值视为异常（默认 1.0）",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最多分析的 episode 数（默认全部）",
    )
    parser.add_argument(
        "--no-sync-check",
        action="store_true",
        help="不进行双臂同步检测",
    )
    parser.add_argument(
        "--sync-window-sec",
        type=float,
        default=5,
        help="开头时间窗（秒），用于报告「开头时间窗内末端位移」统计（默认 5）",
    )
    parser.add_argument(
        "--sync-follow-window-sec",
        type=float,
        default=1.0,
        help="一臂开始移动后，判断另一臂是否跟上的时间窗（秒），窗内另一臂位移小于 sync-max-still 则判为不同步（默认 1.0）",
    )
    parser.add_argument(
        "--sync-min-movement",
        type=float,
        default=0.02,
        help="判定为「在动」的最小位移（米），默认 0.03",
    )
    parser.add_argument(
        "--sync-max-still",
        type=float,
        default=0.02,
        help="判定为「不动」的最大位移（米），默认 0.01；一动一静则判为不同步",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="将每 episode 的起止距离与同步结果写入 JSON 文件（可选）",
    )
    parser.add_argument(
        "--save-out-of-sync-videos",
        action="store_true",
        help="将双臂不同步的 episode 的多路摄像头画面合成为一屏视频并保存，便于调试",
    )
    parser.add_argument(
        "--out-of-sync-videos-dir",
        type=str,
        default=None,
        help="不同步 episode 视频保存目录（默认：<dataset>/out_of_sync_videos）",
    )
    parser.add_argument(
        "--save-abnormal-trajectory-images",
        action="store_true",
        help="将首末帧位姿差异较大的 episode 的轨迹绘制为图片并保存",
    )
    parser.add_argument(
        "--abnormal-trajectory-dir",
        type=str,
        default=None,
        help="异常轨迹图片保存目录（默认：<dataset>/abnormal_trajectories）",
    )
    args = parser.parse_args()

    out = analyze_eef_rot_start_end_distance(
        dataset_path=args.dataset,
        threshold=args.threshold,
        max_episodes=args.max_episodes,
        check_sync=not args.no_sync_check,
        sync_window_sec=args.sync_window_sec,
        sync_follow_window_sec=args.sync_follow_window_sec,
        sync_min_movement=args.sync_min_movement,
        sync_max_still=args.sync_max_still,
        save_out_of_sync_videos=args.save_out_of_sync_videos,
        out_of_sync_videos_dir=args.out_of_sync_videos_dir,
        save_abnormal_trajectory_images=args.save_abnormal_trajectory_images,
        abnormal_trajectory_dir=args.abnormal_trajectory_dir,
    )

    results = out["results"]
    abnormal = out["abnormal_episodes"]
    out_of_sync = out["out_of_sync_episodes"]
    threshold = out["threshold"]

    r0_dists = [r["robot0_dist"] for r in results if not np.isnan(r["robot0_dist"])]
    r1_dists = [r["robot1_dist"] for r in results if not np.isnan(r["robot1_dist"])]

    print("\n" + "=" * 60)
    print("双臂 eef_pos_wrt_start 首帧与末帧欧氏距离 + 开头同步检测")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"起止距离异常阈值: {threshold}")
    print(f"已分析 episode 数: {out['total_episodes']}")
    if out["check_sync"]:
        print(
            f"同步检测: 一臂开始动后跟随窗={out['sync_follow_window_sec']}s, "
            f"动>={out['sync_min_movement']}m, 静<={out['sync_max_still']}m"
        )
    print()

    if r0_dists:
        print("robot0_eef_pos_wrt_start 首帧与末帧距离:")
        print(f"  最小: {np.min(r0_dists):.6f}")
        print(f"  最大: {np.max(r0_dists):.6f}")
        print(f"  平均: {np.mean(r0_dists):.6f}")
        print(f"  中位数: {np.median(r0_dists):.6f}")
    if r1_dists:
        print("robot1_eef_pos_wrt_start 首帧与末帧距离:")
        print(f"  最小: {np.min(r1_dists):.6f}")
        print(f"  最大: {np.max(r1_dists):.6f}")
        print(f"  平均: {np.mean(r1_dists):.6f}")
        print(f"  中位数: {np.median(r1_dists):.6f}")

    print()
    print(f"起止距离 > {threshold} 的异常 episode 数: {len(abnormal)}")
    if abnormal:
        print("  索引:", abnormal[:50])
        if len(abnormal) > 50:
            print(f"  ... 共 {len(abnormal)} 个")
        if args.save_abnormal_trajectory_images:
            traj_dir = Path(args.abnormal_trajectory_dir) if args.abnormal_trajectory_dir else Path(args.dataset) / "abnormal_trajectories"
            print(f"  异常轨迹图已保存至: {traj_dir} (episode_XXXXXX_abnormal_trajectory.png)")

    if out["check_sync"]:
        print()
        print("双臂不同步（一臂先动时 1 秒窗内另一臂未动）的 episode 数:", len(out_of_sync))
        if out_of_sync:
            print("  索引:", out_of_sync[:50])
            if len(out_of_sync) > 50:
                print(f"  ... 共 {len(out_of_sync)} 个")
            if args.save_out_of_sync_videos:
                videos_dir = Path(args.out_of_sync_videos_dir) if args.out_of_sync_videos_dir else Path(args.dataset) / "out_of_sync_videos"
                print(f"  三路摄像头合成视频已保存至: {videos_dir} (episode_XXXXXX_out_of_sync.mp4)")
        sync_disp0 = [r["robot0_sync_disp"] for r in results if not np.isnan(r.get("robot0_sync_disp", np.nan))]
        sync_disp1 = [r["robot1_sync_disp"] for r in results if not np.isnan(r.get("robot1_sync_disp", np.nan))]
        if sync_disp0 and sync_disp1:
            print()
            print("开头时间窗内末端位移统计（米）:")
            print(f"  robot0: 最小={np.min(sync_disp0):.4f}, 最大={np.max(sync_disp0):.4f}, 平均={np.mean(sync_disp0):.4f}")
            print(f"  robot1: 最小={np.min(sync_disp1):.4f}, 最大={np.max(sync_disp1):.4f}, 平均={np.mean(sync_disp1):.4f}")

    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "threshold": threshold,
            "abnormal_episodes": abnormal,
            "total_analyzed": out["total_episodes"],
            "results": results,
        }
        if out["check_sync"]:
            payload["out_of_sync_episodes"] = out_of_sync
            payload["sync_window_sec"] = out["sync_window_sec"]
            payload["sync_follow_window_sec"] = out["sync_follow_window_sec"]
            payload["sync_min_movement"] = out["sync_min_movement"]
            payload["sync_max_still"] = out["sync_max_still"]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"详细结果已写入: {out_path}")


if __name__ == "__main__":
    main()
