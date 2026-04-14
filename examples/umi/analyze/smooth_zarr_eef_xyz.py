#!/usr/bin/env python3
"""
对 zarr 数据集中的 EEF 位置（xyz）按 episode 做平滑，输出新 zarr 供训练使用。

只平滑 robot0_eef_pos、robot1_eef_pos 的 xyz；按 episode 边界分别平滑，不跨 episode。
支持形状 (T, 3) 或 (T, H, 3)；平滑沿时间轴（第 0 维）。

用法:
    python smooth_zarr_eef_xyz.py --input /path/to/datasets_no_filter_20260312.zarr.zip --output /path/to/datasets_smoothed.zarr.zip
    python smooth_zarr_eef_xyz.py --input /path/to/dataset.zarr.zip --output ./out.zarr.zip --window 15 --method savgol
    python smooth_zarr_eef_xyz.py -i in.zarr.zip -o out.zarr.zip --plot-episode 0 --plot-output-dir ./plots
"""

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import zarr
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except ImportError:
    _HAS_SAVGOL = False

POS_KEYS = ["robot0_eef_pos", "robot1_eef_pos"]


def get_episode_ranges(meta: zarr.Group):
    """从 meta['episode_ends'] 得到 (start, end) 列表。"""
    ends = np.array(meta["episode_ends"][:])
    ranges = []
    start = 0
    for end in ends:
        ranges.append((int(start), int(end)))
        start = end
    return ranges


def smooth_series_1d(x: np.ndarray, window: int, method: str, polyorder: int) -> np.ndarray:
    """对一维序列做平滑，返回新数组。"""
    if window < 2 or len(x) < window:
        return x.copy()
    if method == "savgol":
        if not _HAS_SAVGOL:
            method = "mean"
        else:
            w = min(window, len(x) - 1 if len(x) % 2 == 0 else len(x))
            w = w if w % 2 == 1 else w - 1
            if w < 3:
                return x.copy()
            return savgol_filter(x.astype(np.float64), window_length=w, polyorder=min(polyorder, w - 1), mode="nearest")
    # moving average
    kernel = np.ones(min(window, len(x))) / min(window, len(x))
    return np.convolve(x.astype(np.float64), kernel, mode="same").astype(x.dtype)


def smooth_pos_episode(block: np.ndarray, window: int, method: str, polyorder: int) -> np.ndarray:
    """
    对一个 episode 的位置块做平滑。block 形状 (T, 3) 或 (T, H, 3)。
    沿第 0 维（时间）平滑。
    """
    out = block.astype(np.float64).copy()
    if block.ndim == 2 and block.shape[-1] == 3:
        for d in range(3):
            out[:, d] = smooth_series_1d(block[:, d], window, method, polyorder)
    elif block.ndim == 3 and block.shape[-1] == 3:
        for h in range(block.shape[1]):
            for d in range(3):
                out[:, h, d] = smooth_series_1d(block[:, h, d], window, method, polyorder)
    else:
        return block.copy()
    return out.astype(block.dtype)


def _max_abs_diff_episode(orig: np.ndarray, smooth: np.ndarray) -> float:
    """计算一个 episode 块上 |原始 - 平滑| 的最大值（统一为 T,3 后算）。"""
    o = _to_t_3(orig)
    s = _to_t_3(smooth)
    if o.shape != s.shape or o.size == 0:
        return 0.0
    return float(np.max(np.abs(o.astype(np.float64) - s.astype(np.float64))))


def _to_t_3(pos: np.ndarray) -> np.ndarray:
    """统一为 (T, 3) 便于绘图。"""
    if pos.ndim == 2 and pos.shape[-1] == 3:
        return pos
    if pos.ndim == 3 and pos.shape[-1] == 3:
        return pos[:, -1, :]
    return pos.reshape(-1, 3)


def plot_before_after(
    orig: Dict[str, np.ndarray],
    smooth: Dict[str, np.ndarray],
    episode_idx: int,
    output_dir: Path,
    window: int,
    method: str,
    extra_title: str = "",
):
    """绘制一个 episode 的平滑前后对比（XY/XZ/YZ + 位置-时间）。"""
    if not _HAS_MPL:
        print("未安装 matplotlib，跳过对比图")
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in POS_KEYS:
        if key not in orig or key not in smooth:
            continue
        o = _to_t_3(orig[key])
        s = _to_t_3(smooth[key])
        if len(o) != len(s) or len(o) == 0:
            continue
        T = len(o)
        t = np.arange(T)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # XY
        ax = axes[0, 0]
        ax.plot(o[:, 0], o[:, 1], "b-", alpha=0.7, label="平滑前", lw=1)
        ax.plot(s[:, 0], s[:, 1], "r-", alpha=0.8, label="平滑后", lw=1.2)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{key} - XY")
        ax.legend()
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

        # XZ
        ax = axes[0, 1]
        ax.plot(o[:, 0], o[:, 2], "b-", alpha=0.7, label="平滑前", lw=1)
        ax.plot(s[:, 0], s[:, 2], "r-", alpha=0.8, label="平滑后", lw=1.2)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{key} - XZ")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # YZ
        ax = axes[1, 0]
        ax.plot(o[:, 1], o[:, 2], "b-", alpha=0.7, label="平滑前", lw=1)
        ax.plot(s[:, 1], s[:, 2], "r-", alpha=0.8, label="平滑后", lw=1.2)
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{key} - YZ")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 位置 vs 时间
        ax = axes[1, 1]
        ax.plot(t, o[:, 0], "b-", alpha=0.7, label="X 平滑前", lw=1)
        ax.plot(t, o[:, 1], "b--", alpha=0.7, label="Y 平滑前", lw=1)
        ax.plot(t, o[:, 2], "b:", alpha=0.7, label="Z 平滑前", lw=1)
        ax.plot(t, s[:, 0], "r-", alpha=0.8, label="X 平滑后", lw=1.2)
        ax.plot(t, s[:, 1], "r--", alpha=0.8, label="Y 平滑后", lw=1.2)
        ax.plot(t, s[:, 2], "r:", alpha=0.8, label="Z 平滑后", lw=1.2)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Position (m)")
        ax.set_title(f"{key} - 位置 vs 时间")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Episode {episode_idx} 平滑前后对比 (window={window}, method={method}){extra_title}", fontsize=12)
        plt.tight_layout()
        out_file = output_dir / f"episode_{episode_idx:04d}_{key}_before_after.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  对比图: {out_file}")


def _save_filter_stats(
    output_path: Path,
    threshold: float,
    total_episodes: int,
    kept_count: int,
    deleted_indices: List[int],
    episode_ranges: List[Tuple[int, int]],
    bad_ep_max_diff: Dict[int, float],
) -> None:
    """将删除的 episode 统计写入 JSON，与输出 zarr 同目录。"""
    out_dir = output_path.parent
    stem = output_path.stem
    if stem.endswith(".zarr"):
        stem = stem[:-5]
    json_path = out_dir / f"{stem}_deleted_episodes.json"
    detail = []
    for i in deleted_indices:
        start, end = episode_ranges[i]
        detail.append({
            "episode_index": i,
            "frame_start": start,
            "frame_end": end,
            "frame_count": end - start,
            "max_abs_diff_m": round(bad_ep_max_diff.get(i, 0.0), 6),
        })
    stats = {
        "threshold_m": threshold,
        "total_episodes_before": total_episodes,
        "deleted_count": len(deleted_indices),
        "kept_count": kept_count,
        "deleted_episode_indices": deleted_indices,
        "deleted_episodes_detail": detail,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  删除统计已保存: {json_path}")


def _write_filtered_zarr(
    root: zarr.Group,
    data: zarr.Group,
    meta: zarr.Group,
    episode_ranges: List[Tuple[int, int]],
    good_episodes: List[int],
    output_path: Path,
) -> None:
    """只保留 good_episodes，写入新 zarr 并打包为 output_path；保留的 episode 存原数据（不存平滑后）。"""
    out_parent = Path(output_path).parent
    out_parent.mkdir(parents=True, exist_ok=True)
    tmp_out = tempfile.mkdtemp(prefix="zarr_filtered_", dir=str(out_parent))
    try:
        root_out = zarr.open(tmp_out, mode="w")
        data_out = root_out.create_group("data")
        meta_out = root_out.create_group("meta")

        new_episode_ends = []
        total_frames = 0
        for ep_idx in good_episodes:
            start, end = episode_ranges[ep_idx]
            total_frames += end - start
            new_episode_ends.append(total_frames)
        episode_ends_arr = np.array(new_episode_ends, dtype=np.int64)
        meta_out["episode_ends"] = episode_ends_arr

        all_keys = list(data.keys())
        print(f"  正在写入 {len(all_keys)} 个 key（含图像时可能较慢）…")
        for key in tqdm(all_keys, desc="写入过滤后 data", unit="key"):
            tqdm.write(f"    正在处理: {key}")
            arr = np.array(data[key][:])
            if arr.nbytes > 500_000_000:
                tqdm.write(f"    （约 {arr.nbytes / 1e9:.1f} GB，可能较慢）")
            parts = []
            for ep_idx in good_episodes:
                start, end = episode_ranges[ep_idx]
                parts.append(arr[start:end])
            if not parts:
                continue
            concat = np.concatenate(parts, axis=0)
            data_out[key] = concat

        # 复制 meta 下其他数组/属性（若有）
        for k in meta.keys():
            if k == "episode_ends":
                continue
            try:
                a = np.array(meta[k][:])
                meta_out[k] = a
            except Exception:
                pass

        file_list = []
        for root_dir, _dirs, files in os.walk(tmp_out):
            for f in files:
                full = os.path.join(root_dir, f)
                arcname = os.path.relpath(full, tmp_out)
                file_list.append((full, arcname))
        out_zip = output_path
        if out_zip.exists():
            out_zip.unlink()
        print(f"写入 {out_zip}（共 {len(file_list)} 个文件）")
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for full, arcname in tqdm(file_list, desc="打包 zip", unit="file"):
                zf.write(full, arcname)
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)


def run(
    input_path: str,
    output_path: str,
    window: int = 11,
    method: str = "savgol",
    polyorder: int = 3,
    plot_episode: Optional[int] = None,
    plot_output_dir: Optional[str] = None,
    filter_abnormal: bool = False,
    threshold: float = 0.02,
):
    """
    读取 input zarr，对 data 中的 robot0_eef_pos、robot1_eef_pos 按 episode 平滑 xyz，
    写出到 output_path。若 filter_abnormal=True，则丢弃「原始与平滑差值过大」的 episode。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入不存在: {input_path}")

    is_zip = str(input_path).endswith(".zip")
    out_parent = output_path.parent
    out_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="zarr_smooth_", dir=str(out_parent))
    try:
        if is_zip:
            print(f"解压 {input_path} -> {tmp_dir}")
            with zipfile.ZipFile(input_path, "r") as zf:
                zf.extractall(tmp_dir)
        else:
            print(f"复制 {input_path} -> {tmp_dir}")
            dest = Path(tmp_dir)
            for f in Path(input_path).rglob("*"):
                if f.is_file():
                    rel = f.relative_to(input_path)
                    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest / rel)

        root = zarr.open(tmp_dir, mode="a")
        data = root["data"]
        meta = root["meta"]
        episode_ranges = get_episode_ranges(meta)
        n_ep = len(episode_ranges)
        print(f"Episode 数: {n_ep}")

        if method == "savgol" and not _HAS_SAVGOL:
            print("未安装 scipy，改用滑动平均 (method=mean)")
            method = "mean"

        plot_ep = plot_episode if plot_episode is not None and 0 <= plot_episode < n_ep else None
        orig_ep: Dict[str, np.ndarray] = {}
        smooth_ep: Dict[str, np.ndarray] = {}
        smoothed_full: Dict[str, np.ndarray] = {}
        bad_episodes: Set[int] = set()
        bad_ep_max_diff: Dict[int, float] = {}
        # 过滤模式下暂不收集 plot 数据，后面按「保留的 episode」再取
        collect_plot_in_loop = plot_ep is not None and not filter_abnormal

        for key in POS_KEYS:
            if key not in data:
                print(f"  跳过（不存在）: {key}")
                continue
            arr = data[key]
            full = np.array(arr[:])
            if collect_plot_in_loop:
                start, end = episode_ranges[plot_ep]
                orig_ep[key] = full[start:end].copy()
            for ep_idx, (start, end) in enumerate(tqdm(episode_ranges, desc=f"smooth {key}", unit="ep")):
                block = full[start:end]
                if block.size == 0:
                    continue
                smoothed = smooth_pos_episode(block, window, method, polyorder)
                if filter_abnormal:
                    # 必须在 full[start:end]=smoothed 之前算差值，否则 block 是视图会被覆盖成平滑值
                    max_d = _max_abs_diff_episode(block, smoothed)
                    if max_d > threshold:
                        bad_episodes.add(ep_idx)
                        bad_ep_max_diff[ep_idx] = max(bad_ep_max_diff.get(ep_idx, 0.0), float(max_d))
                full[start:end] = smoothed
            if collect_plot_in_loop:
                start, end = episode_ranges[plot_ep]
                smooth_ep[key] = full[start:end].copy()
            smoothed_full[key] = full
            if not filter_abnormal:
                arr[:] = full
            print(f"  已平滑: {key} (window={window}, method={method})")

        if filter_abnormal and bad_episodes:
            good_episodes = [i for i in range(n_ep) if i not in bad_episodes]
            deleted_list = sorted(bad_episodes)
            print(f"  删除的 episode（|原始-平滑| > {threshold}m）: 共 {len(deleted_list)} 个")
            print(f"    删除的索引: {deleted_list}")
            print(f"    保留: {len(good_episodes)} 个 episode")
            _save_filter_stats(
                output_path,
                threshold,
                n_ep,
                len(good_episodes),
                deleted_list,
                episode_ranges,
                bad_ep_max_diff,
            )
            _write_filtered_zarr(
                root, data, meta, episode_ranges,
                good_episodes, output_path,
            )
        elif filter_abnormal:
            print("  未发现异常 episode，无需写入新文件，请直接使用源文件。")
        else:
            # 非过滤模式：打包平滑后的 zarr
            if output_path.suffix == ".zip" or str(output_path).endswith(".zarr.zip"):
                out_zip = output_path
                if out_zip.exists():
                    out_zip.unlink()
                file_list = []
                for root_dir, _dirs, files in os.walk(tmp_dir):
                    for f in files:
                        full = os.path.join(root_dir, f)
                        arcname = os.path.relpath(full, tmp_dir)
                        file_list.append((full, arcname))
                print(f"写入 {out_zip}（共 {len(file_list)} 个文件，压缩较慢请稍候）")
                with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    for full, arcname in tqdm(file_list, desc="打包 zip", unit="file"):
                        zf.write(full, arcname)
            else:
                out_dir = output_path
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
                shutil.copytree(tmp_dir, out_dir)
                print(f"写入目录 {out_dir}")

        # 平滑前后对比图（过滤模式下展示「保留的」某个 episode 的原始与平滑）
        if plot_output_dir and (plot_ep is not None or filter_abnormal):
            plot_ep_actual = plot_ep if plot_ep is not None else 0
            if filter_abnormal and bad_episodes:
                good_episodes = [i for i in range(n_ep) if i not in bad_episodes]
                if good_episodes:
                    plot_ep_actual = plot_ep if (plot_ep is not None and plot_ep in good_episodes) else good_episodes[0]
                    orig_ep = {}
                    smooth_ep = {}
                    for key in POS_KEYS:
                        if key not in data:
                            continue
                        start, end = episode_ranges[plot_ep_actual]
                        orig_ep[key] = np.array(data[key][start:end])
                        smooth_ep[key] = smooth_pos_episode(orig_ep[key].copy(), window, method, polyorder)
                    print(f"  过滤后展示保留的 Episode {plot_ep_actual} 的平滑前后对比")
            elif filter_abnormal and not bad_episodes and plot_ep is not None:
                plot_ep_actual = plot_ep
                orig_ep = {}
                smooth_ep = {}
                for key in POS_KEYS:
                    if key not in data or key not in smoothed_full:
                        continue
                    start, end = episode_ranges[plot_ep_actual]
                    orig_ep[key] = np.array(data[key][start:end])
                    smooth_ep[key] = smoothed_full[key][start:end].copy()
            if orig_ep and smooth_ep:
                print(f"生成 Episode {plot_ep_actual} 平滑前后对比图 -> {plot_output_dir}")
                plot_before_after(orig_ep, smooth_ep, plot_ep_actual, Path(plot_output_dir), window, method)

        # 绘制被删除 episode 的轨迹图
        if plot_output_dir and filter_abnormal and bad_episodes:
            deleted_plot_dir = Path(plot_output_dir) / "deleted_episodes"
            deleted_list = sorted(bad_episodes)
            print(f"  正在生成 {len(deleted_list)} 个被删除 episode 的轨迹图…")
            for ep_idx in deleted_list:
                start, end = episode_ranges[ep_idx]
                max_diff = bad_ep_max_diff.get(ep_idx, 0.0)
                del_orig: Dict[str, np.ndarray] = {}
                del_smooth: Dict[str, np.ndarray] = {}
                for key in POS_KEYS:
                    if key not in data or key not in smoothed_full:
                        continue
                    del_orig[key] = np.array(data[key][start:end])
                    del_smooth[key] = smoothed_full[key][start:end].copy()
                if del_orig and del_smooth:
                    plot_before_after(
                        del_orig, del_smooth, ep_idx, deleted_plot_dir,
                        window, method,
                        extra_title=f"\n[已删除] max|diff|={max_diff:.4f}m > threshold={threshold}m",
                    )
            print(f"  被删除 episode 轨迹图已保存至: {deleted_plot_dir}")

        print("完成。请使用输出路径作为训练数据。")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="对 zarr 中 EEF 的 xyz 按 episode 平滑并输出新 zarr",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/root/openpi-umi/data/fold_clothes_with_depth/datasets_no_filter_20260312.zarr.zip",
        help="输入 zarr 路径（.zarr 或 .zarr.zip）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出路径（.zarr.zip 或 .zarr 目录）；默认在输入同目录下，文件名加 _smoothed",
    )
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        default=11,
        help="平滑窗口长度（奇数；过短 episode 会自动缩小）",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("savgol", "mean"),
        default="savgol",
        help="savgol=Savitzky-Golay，mean=滑动平均",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay 多项式阶数（仅 method=savgol 时有效）",
    )
    parser.add_argument(
        "--plot-episode",
        type=int,
        default=0,
        metavar="N",
        help="完成后抽取第 N 个 episode 绘制平滑前后对比图（默认 0）",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default=None,
        help="对比图输出目录；默认在输出 zarr 同目录下的 smooth_before_after_plots",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不生成对比图",
    )
    parser.add_argument(
        "--filter-abnormal",
        action="store_true",
        help="丢弃「原始与平滑差值过大」的 episode，只输出正常 episode 的新 zarr",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        metavar="M",
        help="异常判定阈值(m)：episode 内 max|原始-平滑| > 此值则丢弃，需配合 --filter-abnormal",
    )
    args = parser.parse_args()

    if args.window % 2 == 0:
        args.window += 1
        print(f"窗口改为奇数: {args.window}")

    out = args.output
    if out is None:
        p = Path(args.input)
        stem = p.stem
        if stem.endswith(".zarr"):
            stem = stem[:-5]
        out = p.parent / f"{stem}_smoothed.zarr.zip"
    plot_dir = args.plot_output_dir
    if plot_dir is None and not args.no_plot:
        plot_dir = str(Path(out).parent / "smooth_before_after_plots")
    run(
        input_path=args.input,
        output_path=out,
        window=args.window,
        method=args.method,
        polyorder=args.polyorder,
        plot_episode=None if args.no_plot else args.plot_episode,
        plot_output_dir=plot_dir,
        filter_abnormal=args.filter_abnormal,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
