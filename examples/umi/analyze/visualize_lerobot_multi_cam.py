#!/usr/bin/env python3
"""
将 LeRobot 数据集中所有相机维度画面拼接为调试视频 (MP4)。

从 meta/info.json 自动识别 dtype=image 的 feature，按 episode 读取 parquet，
将多路相机拼成网格（4 路默认 2x2）并导出 MP4。

用法示例:
  /root/openpi-umi/.venv/bin/python visualize_lerobot_multi_cam.py \\
      --dataset-root /root/openpi-umi/data/wbcd/wbcd_4_views_260516 \\
      --output-dir /root/openpi-umi/data/wbcd/wbcd_4_views_260516/debug_videos \\
      --episodes 1 3 5 \\
      --fps 20 \\
      --step 2
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    **({"force": True} if sys.version_info >= (3, 8) else {}),
)
logger = logging.getLogger(__name__)


def load_image_keys(dataset_root: Path, camera_keys: str | None) -> list[str]:
    if camera_keys:
        keys = [k.strip() for k in camera_keys.split(",") if k.strip()]
        return keys

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"缺少 meta/info.json: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    keys = [name for name, spec in features.items() if spec.get("dtype") == "image"]
    if not keys:
        raise RuntimeError("info.json 中未找到 dtype=image 的相机 feature")
    return sorted(keys)


def decode_image(raw) -> np.ndarray:
    """将 parquet 中的图像 cell 解码为 uint8 HWC RGB。"""
    if isinstance(raw, dict):
        data = raw.get("bytes") or raw.get("data")
    else:
        data = raw
    if data is None:
        raise ValueError("图像 cell 为空")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def grid_shape(n_views: int, layout: str) -> tuple[int, int]:
    if layout == "row":
        return 1, n_views
    if layout == "col":
        return n_views, 1
    if layout == "auto":
        if n_views <= 3:
            return 1, n_views
        cols = 2 if n_views == 4 else math.ceil(math.sqrt(n_views))
        rows = math.ceil(n_views / cols)
        return rows, cols
    match = re.fullmatch(r"(\d+)x(\d+)", layout.strip())
    if not match:
        raise ValueError(f"无效 layout: {layout!r}，请使用 auto/row/col 或如 2x2")
    rows, cols = int(match.group(1)), int(match.group(2))
    if rows * cols < n_views:
        raise ValueError(f"layout {layout} 格子数 ({rows * cols}) 少于相机数 ({n_views})")
    return rows, cols


def compose_frame(
    tiles: list[np.ndarray],
    labels: list[str],
    rows: int,
    cols: int,
    tile_size: int,
    gap: int,
    frame_idx: int | None,
    episode_idx: int | None,
) -> np.ndarray:
    canvas_w = cols * tile_size + (cols - 1) * gap
    canvas_h = rows * tile_size + (rows - 1) * gap
    header_h = 28 if frame_idx is not None or episode_idx is not None else 0
    canvas = np.zeros((canvas_h + header_h, canvas_w, 3), dtype=np.uint8)

    for i, (img, label) in enumerate(zip(tiles, labels, strict=True)):
        r, c = divmod(i, cols)
        y0 = header_h + r * (tile_size + gap)
        x0 = c * (tile_size + gap)
        tile = img
        if tile.shape[:2] != (tile_size, tile_size):
            tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile
        cv2.putText(
            canvas,
            label,
            (x0 + 4, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if header_h > 0:
        parts = []
        if episode_idx is not None:
            parts.append(f"episode {episode_idx}")
        if frame_idx is not None:
            parts.append(f"frame {frame_idx}")
        cv2.putText(
            canvas,
            " | ".join(parts),
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    return canvas


def episode_index_from_path(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)", path.name)
    return int(match.group(1)) if match else None


def iter_parquet_files(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 parquet: {data_dir}/chunk-*/episode_*.parquet")
    return files


def render_episode(
    parquet_path: Path,
    output_path: Path,
    image_keys: list[str],
    fps: float,
    tile_size: int,
    gap: int,
    step: int,
    max_frames: int | None,
    layout: str,
    show_header: bool,
) -> int:
    table = pq.read_table(parquet_path)
    missing = [k for k in image_keys if k not in table.column_names]
    if missing:
        raise KeyError(f"{parquet_path.name} 缺少列: {missing}")

    n_rows = table.num_rows
    if n_rows == 0:
        logger.warning("跳过空 episode: %s", parquet_path.name)
        return 0

    frame_indices = np.arange(0, n_rows, step, dtype=np.int64)
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    rows, cols = grid_shape(len(image_keys), layout)
    episode_idx = episode_index_from_path(parquet_path)
    if "episode_index" in table.column_names and table.num_rows > 0:
        ep_val = table.column("episode_index")[0].as_py()
        if isinstance(ep_val, (list, np.ndarray)):
            episode_idx = int(ep_val[0]) if len(ep_val) else episode_idx
        elif ep_val is not None:
            episode_idx = int(ep_val)

    img_cols = {k: table.column(k) for k in image_keys}
    fi_col = table.column("frame_index") if "frame_index" in table.column_names else None

    first_raw = img_cols[image_keys[0]][int(frame_indices[0])].as_py()
    sample = decode_image(first_raw)
    if tile_size <= 0:
        tile_size = max(sample.shape[0], sample.shape[1])

    canvas_sample = compose_frame(
        [sample] * len(image_keys),
        image_keys,
        rows,
        cols,
        tile_size,
        gap,
        0 if show_header else None,
        episode_idx if show_header else None,
    )
    canvas_h, canvas_w = canvas_sample.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {output_path}")

    try:
        for row_idx in frame_indices:
            row_idx = int(row_idx)
            tiles = [decode_image(img_cols[k][row_idx].as_py()) for k in image_keys]
            frame_idx = None
            if show_header and fi_col is not None:
                fi_val = fi_col[row_idx].as_py()
                if isinstance(fi_val, (list, np.ndarray)):
                    frame_idx = int(fi_val[0]) if len(fi_val) else row_idx
                else:
                    frame_idx = int(fi_val)
            canvas = compose_frame(
                tiles,
                image_keys,
                rows,
                cols,
                tile_size,
                gap,
                frame_idx if show_header else None,
                episode_idx if show_header else None,
            )
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return len(frame_indices)


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else dataset_root / "debug_multi_cam"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_keys = load_image_keys(dataset_root, args.camera_keys)
    logger.info("相机 keys (%d): %s", len(image_keys), image_keys)
    logger.info("布局: %s -> %s", args.layout, grid_shape(len(image_keys), args.layout))

    parquet_files = iter_parquet_files(dataset_root)
    if args.episodes is not None:
        wanted = set(args.episodes)
        parquet_files = [p for p in parquet_files if episode_index_from_path(p) in wanted]

    if not parquet_files:
        raise RuntimeError("没有匹配的 episode parquet 文件")

    logger.info("将导出 %d 个 episode 到 %s", len(parquet_files), output_dir)

    total_frames = 0
    for parquet_path in tqdm(parquet_files, desc="episodes"):
        ep_idx = episode_index_from_path(parquet_path)
        out_name = f"episode_{ep_idx:06d}_multi_cam.mp4" if ep_idx is not None else f"{parquet_path.stem}_multi_cam.mp4"
        out_path = output_dir / out_name
        n = render_episode(
            parquet_path=parquet_path,
            output_path=out_path,
            image_keys=image_keys,
            fps=args.fps,
            tile_size=args.tile_size,
            gap=args.gap,
            step=args.step,
            max_frames=args.max_frames,
            layout=args.layout,
            show_header=not args.no_header,
        )
        total_frames += n
        logger.info("已保存 %s (%d 帧)", out_path, n)

    logger.info("完成。共写入 %d 帧，输出目录: %s", total_frames, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LeRobot 多相机拼接调试视频导出",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/root/openpi-umi/data/wbcd/wbcd_4_views_260516",
        help="LeRobot 数据集根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出 MP4 目录（默认: <dataset-root>/debug_multi_cam）",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=None,
        help="逗号分隔的相机列名；默认从 meta/info.json 读取全部 image feature",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="仅导出指定 episode 索引（默认全部）",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="输出视频帧率")
    parser.add_argument("--tile-size", type=int, default=224, help="每路相机缩放边长")
    parser.add_argument("--gap", type=int, default=6, help="相机格子间距（像素）")
    parser.add_argument("--step", type=int, default=1, help="每隔几帧取一帧")
    parser.add_argument("--max-frames", type=int, default=None, help="每个 episode 最多导出帧数")
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        help="拼接布局: auto(4路2x2)/row/col/或如2x2",
    )
    parser.add_argument("--no-header", action="store_true", help="不显示 episode/frame 顶部文字")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
