#!/usr/bin/env python3
"""
可视化 zarr 中机械臂三路相机并导出拼接视频。

默认会优先使用以下三路相机:
  - camera0_rgb
  - camera1_rgb
  - head_rgb

用法示例:
  python visualize_zarr_three_views.py \
      --zarr /root/openpi-umi/data/fold_clothes_with_depth/260320_fold_black_1815_left_horizon_dataset.zarr.zip \
      --output /root/openpi-umi/data/fold_clothes_with_depth/three_views.mp4
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List

import numpy as np
import zarr
from PIL import Image


DEFAULT_CAMERA_KEYS = ["camera0_rgb", "camera1_rgb", "head_rgb"]


def load_zarr(path: str):
    """
    打开 zarr 或 zarr.zip。
    优先尝试 ZipStore（避免整包解压），失败时再回退到临时目录解压。
    返回:
      root, cleanup_tmp_dir, cleanup_store
    """
    zarr_path = Path(path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"zarr 路径不存在: {zarr_path}")

    if zarr_path.suffix == ".zip":
        try:
            from zarr.storage import ZipStore

            store = ZipStore(str(zarr_path), mode="r")
            root = zarr.open(store, mode="r")
            return root, None, store
        except (ImportError, AttributeError, TypeError, ValueError, OSError):
            tmp_dir = tempfile.mkdtemp(prefix="zarr_three_views_")
            with zipfile.ZipFile(zarr_path, "r") as zf:
                zf.extractall(tmp_dir)
            root = zarr.open(tmp_dir, mode="r")
            return root, tmp_dir, None

    root = zarr.open(str(zarr_path), mode="r")
    return root, None, None


def _is_rgb_array(arr: zarr.Array) -> bool:
    shape = getattr(arr, "shape", ())
    return len(shape) >= 4 and shape[-1] == 3


def infer_camera_keys(data_group: zarr.Group, requested: str | None) -> List[str]:
    if requested:
        keys = [k.strip() for k in requested.split(",") if k.strip()]
        for k in keys:
            if k not in data_group:
                raise KeyError(f"指定相机键不存在: {k}")
        return keys

    if all(k in data_group for k in DEFAULT_CAMERA_KEYS):
        return DEFAULT_CAMERA_KEYS

    fallback = []
    for key in data_group.keys():
        lower = key.lower()
        if any(token in lower for token in ("camera", "cam", "rgb", "image")):
            arr = data_group[key]
            if _is_rgb_array(arr):
                fallback.append(key)

    if len(fallback) < 3:
        raise RuntimeError(
            f"自动识别三路相机失败。可用候选: {fallback}。"
            "请使用 --camera-keys 手动指定，例如: --camera-keys camera0_rgb,camera1_rgb,head_rgb"
        )
    return sorted(fallback)[:3]


def get_indices(total: int, start: int, end: int | None, step: int, max_frames: int | None) -> np.ndarray:
    if total <= 0:
        return np.array([], dtype=np.int64)
    if start < 0:
        start = 0
    if end is None or end > total:
        end = total
    if end <= start:
        return np.array([], dtype=np.int64)
    indices = np.arange(start, end, step, dtype=np.int64)
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def get_episode_ranges(meta_group: zarr.Group) -> List[tuple[int, int]]:
    if "episode_ends" not in meta_group:
        raise KeyError("meta 中缺少 episode_ends，无法按 episode 导出")
    episode_ends = np.asarray(meta_group["episode_ends"][:], dtype=np.int64)
    if episode_ends.size == 0:
        raise RuntimeError("episode_ends 为空，无法按 episode 导出")
    ranges: List[tuple[int, int]] = []
    start = 0
    for end in episode_ends:
        ranges.append((int(start), int(end)))
        start = int(end)
    return ranges


def _to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"单帧图像维度应为 3，实际是: {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"单帧图像通道数应为 3，实际是: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize(frame: np.ndarray, tile_size: int) -> np.ndarray:
    if frame.shape[0] == tile_size and frame.shape[1] == tile_size:
        return frame
    img = Image.fromarray(frame)
    return np.asarray(img.resize((tile_size, tile_size), Image.Resampling.BILINEAR))


def compose_frame(data_group: zarr.Group, camera_keys: Iterable[str], frame_idx: int, tile_size: int) -> np.ndarray:
    row = []
    for key in camera_keys:
        frame = _to_hwc_uint8(np.array(data_group[key][frame_idx]))
        row.append(_resize(frame, tile_size))
    return np.concatenate(row, axis=1)


def write_three_view_video(
    zarr_path: str,
    output_path: str,
    camera_keys: str | None = None,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
    max_frames: int | None = None,
    fps: float = 30.0,
    tile_size: int = 224,
    random_episode: bool = True,
    episode_idx: int | None = None,
    seed: int | None = None,
):
    if step <= 0:
        raise ValueError("--step 必须为正整数")
    if fps <= 0:
        raise ValueError("--fps 必须为正数")
    if tile_size <= 0:
        raise ValueError("--tile-size 必须为正整数")

    try:
        import imageio
    except ImportError as e:
        raise RuntimeError(
            "缺少 imageio。请先安装: pip install imageio imageio-ffmpeg"
        ) from e

    root, tmp_dir, store = load_zarr(zarr_path)
    try:
        if "data" not in root:
            raise KeyError("zarr 缺少 data group")
        data = root["data"]
        selected_keys = infer_camera_keys(data, camera_keys)
        first_arr = data[selected_keys[0]]
        total = int(first_arr.shape[0])

        if episode_idx is not None and random_episode:
            raise ValueError("--episode-idx 与 --random-episode 不能同时使用")

        if episode_idx is not None or random_episode:
            if "meta" not in root:
                raise KeyError("zarr 缺少 meta group，无法按 episode 导出")
            episode_ranges = get_episode_ranges(root["meta"])
            if episode_idx is not None:
                if episode_idx < 0 or episode_idx >= len(episode_ranges):
                    raise ValueError(
                        f"--episode-idx 超出范围: {episode_idx}，有效范围 [0, {len(episode_ranges) - 1}]"
                    )
                chosen_episode = episode_idx
            else:
                rng = np.random.default_rng(seed)
                chosen_episode = int(rng.integers(0, len(episode_ranges)))
            ep_start, ep_end = episode_ranges[chosen_episode]
            start = ep_start
            end = ep_end
            print(f"选择 episode: {chosen_episode} (frames: [{ep_start}, {ep_end}))")

        indices = get_indices(total, start, end, step, max_frames)
        if len(indices) == 0:
            raise RuntimeError("没有可写入的视频帧，请检查 --start/--end/--step/--max-frames")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        preview_path = output.with_name(output.stem + "_preview.png")

        print("将使用以下三路相机:", selected_keys)
        print(f"总帧数: {total}, 导出帧数: {len(indices)}, fps: {fps}")
        print(f"输出视频: {output}")

        preview = compose_frame(data, selected_keys, int(indices[0]), tile_size)
        imageio.imwrite(preview_path, preview)
        print(f"预览图已保存: {preview_path}")

        writer = imageio.get_writer(str(output), fps=float(fps))
        try:
            for i, frame_idx in enumerate(indices):
                frame = compose_frame(data, selected_keys, int(frame_idx), tile_size)
                writer.append_data(frame)
                if (i + 1) % 200 == 0 or i == len(indices) - 1:
                    print(f"写入进度: {i + 1}/{len(indices)}")
        finally:
            writer.close()

        print("视频导出完成。")
    finally:
        if store is not None:
            try:
                store.close()
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="可视化 zarr 三路机械臂相机并导出拼接视频",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--zarr",
        type=str,
        default="/root/openpi-umi/data/fold_clothes_with_depth/260320_fold_black_1815_left_horizon_dataset.zarr.zip",
        help="zarr 或 zarr.zip 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/openpi-umi/data/fold_clothes_with_depth/three_views.mp4",
        help="输出视频路径 (mp4)",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=None,
        help="手动指定三路相机键，逗号分隔（例如 camera0_rgb,camera1_rgb,head_rgb）",
    )
    parser.add_argument("--start", type=int, default=0, help="起始帧（含）")
    parser.add_argument("--end", type=int, default=None, help="结束帧（不含）")
    parser.add_argument("--step", type=int, default=1, help="每隔几帧取一帧")
    parser.add_argument("--max-frames", type=int, default=None, help="最多导出帧数")
    parser.add_argument("--fps", type=float, default=30.0, help="输出视频帧率")
    parser.add_argument("--tile-size", type=int, default=224, help="每路相机缩放边长")
    parser.add_argument(
        "--random-episode",
        action="store_true",
        help="随机挑选一个 episode 导出（默认行为）",
    )
    parser.add_argument(
        "--no-random-episode",
        action="store_true",
        help="关闭随机 episode，按 --start/--end 导出全局帧区间",
    )
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=None,
        help="手动指定要导出的 episode 索引（优先级高于随机）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机挑选 episode 的随机种子")
    args = parser.parse_args()

    if args.no_random_episode and args.random_episode:
        raise SystemExit("--random-episode 与 --no-random-episode 不能同时指定")
    use_random_episode = not args.no_random_episode

    write_three_view_video(
        zarr_path=args.zarr,
        output_path=args.output,
        camera_keys=args.camera_keys,
        start=args.start,
        end=args.end,
        step=args.step,
        max_frames=args.max_frames,
        fps=args.fps,
        tile_size=args.tile_size,
        random_episode=use_random_episode,
        episode_idx=args.episode_idx,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
