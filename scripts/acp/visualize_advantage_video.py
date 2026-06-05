"""Visualize LeRobot dataset as video: 3 camera views + advantage / is_positive overlay.

Reads image columns (left_wrist_0_rgb_0, right_wrist_0_rgb_0, base_0_rgb_0) from parquet,
annotates each frame with advantage, is_positive, predicted_value, and value_target,
then writes one MP4 per episode.

Usage:
    python scripts/visualize_advantage_video.py \
        --dataset-root ./data/my_lerobot_dataset \
        --output-dir ./advantage_videos \
        --fps 20
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    **({"force": True} if sys.version_info >= (3, 8) else {}),
)
logger = logging.getLogger(__name__)

IMAGE_COLS = ["left_wrist_0_rgb_0", "right_wrist_0_rgb_0", "base_0_rgb_0"]
VIEW_LABELS = ["left_wrist", "right_wrist", "base"]

SCALAR_COLS = ["advantage", "is_positive", "predicted_value", "value_target"]


def _decode_image(raw) -> np.ndarray:
    """Decode parquet image cell (dict with 'bytes' or raw bytes) to uint8 HWC RGB array."""
    if isinstance(raw, dict):
        data = raw.get("bytes") or raw.get("data")
    else:
        data = raw
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _scalar_val(col_data, idx: int) -> float | int | None:
    v = col_data[idx]
    if hasattr(v, "as_py"):
        v = v.as_py()
    if v is None:
        return None
    if isinstance(v, (list, np.ndarray)):
        return float(v[0]) if len(v) else None
    return v


def _build_overlay_text(scalars: dict[str, float | int | None], frame_idx: int) -> list[str]:
    lines = [f"frame {frame_idx}"]
    adv = scalars.get("advantage")
    isp = scalars.get("is_positive")
    pv = scalars.get("predicted_value")
    vt = scalars.get("value_target")
    if adv is not None:
        lines.append(f"advantage: {float(adv):+.4f}")
    if isp is not None:
        label = "POSITIVE" if int(isp) else "NEGATIVE"
        lines.append(f"is_positive: {int(isp)} ({label})")
    if pv is not None:
        lines.append(f"pred_value: {float(pv):.4f}")
    if vt is not None:
        lines.append(f"target_value: {float(vt):.4f}")
    return lines


def _put_text_block(
    canvas: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    line_spacing: int = 4,
) -> np.ndarray:
    """Draw multi-line text with background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x0, y0 = origin
    max_w = 0
    total_h = 0
    sizes = []
    for line in lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        sizes.append((w, h, baseline))
        max_w = max(max_w, w)
        total_h += h + baseline + line_spacing

    pad = 6
    cv2.rectangle(
        canvas,
        (x0 - pad, y0 - pad),
        (x0 + max_w + pad, y0 + total_h + pad),
        bg_color,
        cv2.FILLED,
    )

    cy = y0
    for i, line in enumerate(lines):
        w, h, baseline = sizes[i]
        cy += h
        cv2.putText(canvas, line, (x0, cy), font, font_scale, color, thickness, cv2.LINE_AA)
        cy += baseline + line_spacing

    return canvas


def _colorize_border(img: np.ndarray, is_positive: int | None, border: int = 4) -> np.ndarray:
    """Add a thin colored border: green for positive, red for negative."""
    if is_positive is None:
        return img
    color = (0, 200, 0) if int(is_positive) else (200, 0, 0)
    h, w = img.shape[:2]
    img[:border, :] = color
    img[-border:, :] = color
    img[:, :border] = color
    img[:, -border:] = color
    return img


def process_episode(
    table,
    episode_idx: int,
    output_dir: Path,
    fps: int,
    available_img_cols: list[str],
    available_scalar_cols: list[str],
    img_size: int = 224,
    gap: int = 8,
) -> Path:
    ep_col = table.column("episode_index")
    ep_np = np.array(ep_col.to_pylist(), dtype=np.int64).reshape(-1)
    mask = ep_np == episode_idx
    row_indices = np.where(mask)[0]
    if len(row_indices) == 0:
        raise ValueError(f"No rows for episode_index={episode_idx}")

    fi_col = table.column("frame_index")
    fi_np = np.array(fi_col.to_pylist(), dtype=np.int64).reshape(-1)
    order = np.argsort(fi_np[row_indices], kind="stable")
    row_indices = row_indices[order]

    n_views = len(available_img_cols)
    canvas_w = n_views * img_size + (n_views - 1) * gap
    canvas_h = img_size + 100

    out_path = output_dir / f"episode_{episode_idx}_advantage.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_w, canvas_h))

    img_columns = {col: table.column(col) for col in available_img_cols}
    scalar_columns = {col: table.column(col) for col in available_scalar_cols}

    for ri in tqdm(row_indices, desc=f"ep {episode_idx}", leave=False):
        ri = int(ri)
        frame_idx = int(fi_np[ri])

        scalars = {col: _scalar_val(scalar_columns[col], ri) for col in available_scalar_cols}
        is_pos = scalars.get("is_positive")

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for vi, col in enumerate(available_img_cols):
            raw = img_columns[col][ri].as_py()
            img = _decode_image(raw)
            if img.shape[:2] != (img_size, img_size):
                img = cv2.resize(img, (img_size, img_size))
            img = _colorize_border(img, is_pos, border=3)

            label_idx = IMAGE_COLS.index(col) if col in IMAGE_COLS else vi
            label = VIEW_LABELS[label_idx] if label_idx < len(VIEW_LABELS) else col
            cv2.putText(
                img, label, (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

            x_off = vi * (img_size + gap)
            canvas[:img_size, x_off: x_off + img_size] = img

        lines = _build_overlay_text(scalars, frame_idx)
        _put_text_block(canvas, lines, origin=(8, img_size + 6), font_scale=0.50, thickness=1)

        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        writer.write(canvas_bgr)

    writer.release()
    return out_path


def _extract_episode_arrays(table, episode_idx: int) -> dict[str, np.ndarray]:
    """Extract sorted scalar arrays for one episode. Returns dict of column -> 1-D ndarray."""
    ep_np = np.array(table.column("episode_index").to_pylist(), dtype=np.int64).reshape(-1)
    mask = ep_np == episode_idx
    row_indices = np.where(mask)[0]

    fi_np = np.array(table.column("frame_index").to_pylist(), dtype=np.int64).reshape(-1)
    order = np.argsort(fi_np[row_indices], kind="stable")
    row_indices = row_indices[order]

    out: dict[str, np.ndarray] = {"frame_index": fi_np[row_indices]}
    for col_name in ["advantage", "is_positive", "predicted_value", "value_target", "action_source"]:
        if col_name not in table.column_names:
            continue
        raw = [table.column(col_name)[int(ri)].as_py() for ri in row_indices]
        out[col_name] = np.array(raw, dtype=np.float64)
    return out


def _find_intervention_spans(action_source: np.ndarray, frame_index: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start_frame, end_frame) for contiguous action_source==1 regions."""
    spans: list[tuple[int, int]] = []
    in_span = False
    start = 0
    for i, v in enumerate(action_source):
        if v == 1 and not in_span:
            in_span = True
            start = int(frame_index[i])
        elif v != 1 and in_span:
            in_span = False
            spans.append((start, int(frame_index[i - 1])))
    if in_span:
        spans.append((start, int(frame_index[-1])))
    return spans


def save_advantage_timeline(
    table,
    episode_idx: int,
    output_dir: Path,
    dpi: int = 150,
) -> Path:
    """Plot advantage timeline with human intervention shading for one episode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    arrays = _extract_episode_arrays(table, episode_idx)
    frames = arrays["frame_index"]
    advantage = arrays.get("advantage")
    if advantage is None:
        raise ValueError(f"No 'advantage' column for episode {episode_idx}")

    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=dpi)

    # Human intervention shading
    has_intervention = False
    if "action_source" in arrays:
        spans = _find_intervention_spans(arrays["action_source"], frames)
        for s, e in spans:
            ax.axvspan(s, e, alpha=0.22, color="#FF6B6B", zorder=0)
            has_intervention = True

    # Advantage curve
    ax.plot(frames, advantage, linewidth=0.7, color="#2196F3", zorder=2)
    ax.fill_between(
        frames, advantage, 0,
        where=(advantage >= 0), interpolate=True, alpha=0.15, color="#4CAF50", zorder=1,
    )
    ax.fill_between(
        frames, advantage, 0,
        where=(advantage < 0), interpolate=True, alpha=0.15, color="#F44336", zorder=1,
    )

    # Zero line
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)

    # is_positive: shade negative regions
    if "is_positive" in arrays:
        ip = arrays["is_positive"]
        neg_mask = ip == 0
        if neg_mask.any():
            ax.fill_between(
                frames, ax.get_ylim()[0], ax.get_ylim()[1],
                where=neg_mask, alpha=0.08, color="#9E9E9E", zorder=0,
                label="is_positive=0",
            )

    # predicted_value & value_target on secondary axis
    has_secondary = False
    if "predicted_value" in arrays or "value_target" in arrays:
        ax2 = ax.twinx()
        if "value_target" in arrays:
            ax2.plot(frames, arrays["value_target"], linewidth=0.7, color="#FF9800",
                     alpha=0.7, linestyle="--", label="value_target", zorder=2)
            has_secondary = True
        if "predicted_value" in arrays:
            ax2.plot(frames, arrays["predicted_value"], linewidth=0.7, color="#9C27B0",
                     alpha=0.7, linestyle="-.", label="predicted_value", zorder=2)
            has_secondary = True
        ax2.set_ylabel("value", fontsize=9, color="#666")
        ax2.tick_params(labelsize=7, colors="#666")
        if has_secondary:
            ax2.legend(loc="upper right", fontsize=7, framealpha=0.8)

    ax.set_xlabel("frame index", fontsize=10)
    ax.set_ylabel("advantage", fontsize=10)
    ax.set_title(f"Episode {episode_idx} — Advantage Timeline", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.set_xlim(frames[0], frames[-1])

    # Vertical marker: start of the last 50 frames (by sorted frame_index)
    if len(frames) >= 50:
        last50_start = int(frames[-50])
        ax.axvline(
            last50_start,
            color="#FFC107",
            linewidth=1.4,
            linestyle="-",
            zorder=4,
            alpha=0.95,
        )
        y0, y1 = ax.get_ylim()
        y_ann = y1 - 0.06 * (y1 - y0)
        ax.text(
            last50_start,
            y_ann,
            " last 50 frames →",
            fontsize=8,
            color="#E65100",
            ha="left",
            va="top",
            zorder=5,
            clip_on=True,
        )

    # Legend
    handles = [
        mpatches.Patch(color="#2196F3", alpha=0.6, label="advantage"),
    ]
    if has_intervention:
        handles.append(mpatches.Patch(color="#FF6B6B", alpha=0.22, label="human intervention"))
    if "is_positive" in arrays and (arrays["is_positive"] == 0).any():
        handles.append(mpatches.Patch(color="#9E9E9E", alpha=0.15, label="is_positive=0"))
    if len(frames) >= 50:
        handles.append(
            mlines.Line2D(
                [], [], color="#FFC107", linewidth=1.4, linestyle="-",
                label=f"last 50 frames start (frame {last50_start})",
            )
        )
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.8)

    fig.tight_layout()
    out_path = output_dir / f"episode_{episode_idx}_advantage_timeline.png"
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = args.fps

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")

    import pyarrow as pa
    tables = [pq.read_table(f) for f in parquet_files]
    table = pa.concat_tables(tables)
    logger.info("Loaded %d frames from %d parquet files.", table.num_rows, len(parquet_files))

    available_img_cols = [c for c in IMAGE_COLS if c in table.column_names]
    if not available_img_cols:
        raise ValueError(f"No image columns found. Expected any of {IMAGE_COLS}, got {table.column_names}")
    logger.info("Image columns: %s", available_img_cols)

    available_scalar_cols = [c for c in SCALAR_COLS if c in table.column_names]
    logger.info("Scalar columns: %s", available_scalar_cols)

    ep_col = np.array(table.column("episode_index").to_pylist(), dtype=np.int64).reshape(-1)
    unique_eps = sorted(set(ep_col.tolist()))
    if args.episodes is not None:
        unique_eps = [e for e in unique_eps if e in args.episodes]
    logger.info("Episodes to render: %s", unique_eps)

    for ep in unique_eps:
        out_path = process_episode(
            table, ep, output_dir, fps,
            available_img_cols, available_scalar_cols,
        )
        logger.info("Saved video: %s", out_path)

        timeline_path = save_advantage_timeline(table, ep, output_dir, dpi=args.dpi)
        logger.info("Saved timeline: %s", timeline_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render 3-view video with advantage/is_positive overlay per episode.")
    p.add_argument("--dataset-root", type=str, required=True, help="LeRobot dataset root")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory for MP4 files (default: dataset root)")
    p.add_argument("--fps", type=int, default=20, help="Video FPS (default 20, match dataset fps if needed)")
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="Episode indices to render (default: all)")
    p.add_argument("--dpi", type=int, default=150, help="DPI for timeline PNG (default 150)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
