#!/usr/bin/env python3
"""Export selected episodes from the real-cup Zarr-v2 replay buffer.

The dedicated Qwen environment intentionally does not contain Zarr/numcodecs.
This utility decodes the source buffer once and writes compact uint8 NPZ/MP4
assets that can be consumed by the existing Qwen3-VL inference stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from numcodecs import get_codec
import numpy as np
from PIL import Image
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUFFER = ROOT / "data/cup_replay_buffer/replay_buffer.zarr"
DEFAULT_OUTPUT = ROOT / "artifacts/cup_replay_real_qwen_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", type=Path, default=DEFAULT_BUFFER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode-ids", default="0,1,2")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--contact-frames", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class ZarrV2Array:
    """Minimal local reader for arrays whose trailing dimensions fit one chunk."""

    def __init__(self, path: Path):
        self.path = path
        self.meta = json.loads((path / ".zarray").read_text(encoding="utf-8"))
        self.shape = tuple(int(value) for value in self.meta["shape"])
        self.chunks = tuple(int(value) for value in self.meta["chunks"])
        self.dtype = np.dtype(self.meta["dtype"])
        compressor = self.meta.get("compressor")
        self.codec = None if compressor is None else get_codec(compressor)
        if self.chunks[1:] != self.shape[1:]:
            raise ValueError(
                f"Only arrays with full trailing-dimension chunks are supported: "
                f"shape={self.shape}, chunks={self.chunks}"
            )

    def _chunk(self, index: int) -> np.ndarray:
        chunk_coordinates = [index, *([0] * (len(self.shape) - 1))]
        name = ".".join(str(value) for value in chunk_coordinates)
        payload = (self.path / name).read_bytes()
        if self.codec is not None:
            payload = self.codec.decode(payload)
        return np.frombuffer(payload, dtype=self.dtype).reshape(self.chunks)

    def slice_first_axis(self, start: int, end: int) -> np.ndarray:
        if not 0 <= start <= end <= self.shape[0]:
            raise ValueError(f"Invalid slice [{start},{end}) for shape {self.shape}")
        first = start // self.chunks[0]
        last = (end - 1) // self.chunks[0] if end > start else first
        values = [self._chunk(index) for index in range(first, last + 1)]
        joined = np.concatenate(values, axis=0)
        local_start = start - first * self.chunks[0]
        return np.asarray(joined[local_start : local_start + end - start])

    def take_first_axis(self, indices: np.ndarray) -> np.ndarray:
        """Read arbitrary first-axis indices without decoding the gaps between them."""

        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"Expected a non-empty 1-D index array, got {indices.shape}")
        if int(indices.min()) < 0 or int(indices.max()) >= self.shape[0]:
            raise IndexError(f"Indices outside [0,{self.shape[0]}): {indices.tolist()}")
        result = np.empty((len(indices), *self.shape[1:]), dtype=self.dtype)
        chunk_ids = indices // self.chunks[0]
        for chunk_id in np.unique(chunk_ids):
            selected = np.flatnonzero(chunk_ids == chunk_id)
            local = indices[selected] - chunk_id * self.chunks[0]
            result[selected] = self._chunk(int(chunk_id))[local]
        return result


def episode_ends(buffer: Path) -> np.ndarray:
    array = ZarrV2Array(buffer / "meta/episode_ends")
    return array.slice_first_axis(0, array.shape[0]).astype(np.int64)


def to_uint8(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    return np.clip(np.rint(frames.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)


def contact_sheet(frames: np.ndarray, indices: np.ndarray, episode: int) -> Image.Image:
    selected = frames[indices]
    height, width = selected.shape[1:3]
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    label_height = 22
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for position, (index, frame) in enumerate(zip(indices, selected, strict=True)):
        x = position % columns * width
        y = position // columns * (height + label_height)
        sheet.paste(Image.fromarray(frame), (x, y + label_height))
        draw.text((x + 4, y + 4), f"ep {episode:03d}  frame {int(index):04d}", fill="black")
    return sheet


def main() -> None:
    args = parse_args()
    buffer = args.buffer.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes = [int(value.strip()) for value in args.episode_ids.split(",") if value.strip()]
    ends = episode_ends(buffer)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    camera = ZarrV2Array(buffer / "data/camera0_rgb")
    timestamp = ZarrV2Array(buffer / "data/timestamp")

    summary = []
    for episode in episodes:
        if not 0 <= episode < len(ends):
            raise ValueError(f"Episode {episode} outside [0,{len(ends)})")
        start, end = int(starts[episode]), int(ends[episode])
        frames = to_uint8(camera.slice_first_axis(start, end))
        timestamps = timestamp.slice_first_axis(start, end).astype(np.float64)
        npz_path = output / f"episode_{episode:03d}.npz"
        mp4_path = output / f"episode_{episode:03d}.mp4"
        sheet_path = output / f"episode_{episode:03d}_contact.png"
        for path in (npz_path, mp4_path, sheet_path):
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
        np.savez_compressed(npz_path, frames=frames, timestamps=timestamps)
        imageio.mimwrite(mp4_path, frames, fps=args.fps, codec="libx264", quality=7)
        indices = np.linspace(0, len(frames) - 1, num=min(args.contact_frames, len(frames)), dtype=np.int64)
        contact_sheet(frames, indices, episode).save(sheet_path)
        median_dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else float("nan")
        summary.append(
            {
                "episode_index": episode,
                "global_start": start,
                "global_end": end,
                "frames": len(frames),
                "duration_from_timestamp_seconds": float(timestamps[-1] - timestamps[0]),
                "median_dt_seconds": median_dt,
                "median_hz": float(1.0 / median_dt),
                "npz": str(npz_path),
                "mp4": str(mp4_path),
                "contact_sheet": str(sheet_path),
            }
        )
        print(json.dumps(summary[-1], sort_keys=True), flush=True)
    (output / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
