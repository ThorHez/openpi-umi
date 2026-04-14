#!/usr/bin/env python3
"""
Downsample a LeRobot v2.1 dataset by keeping every N-th frame (integer ratio).

Typical use: 60 Hz -> 20 Hz with step=3. Rewrites per-episode parquet files,
updates fps / total_frames / episode lengths, and recomputes meta/episodes_stats.jsonl
from the new tables (image columns skipped for stats, same as remove_abnormal_data.py).

Does not copy norm_stats.json or eval artifacts; recompute norm stats for training if needed.

Usage (project venv at ``openpi-umi/.venv``):

  cd /root/openpi-umi
  .venv/bin/python examples/umi/downsample_lerobot_dataset.py \\
    --input data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260320 \\
    --output data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260320_20hz \\
    --target-fps 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Reuse episode stats helper (skips heavy / image columns).
_UMI_DIR = Path(__file__).resolve().parent
if str(_UMI_DIR) not in sys.path:
    sys.path.insert(0, str(_UMI_DIR))
from remove_abnormal_data import compute_episode_stats, get_heavy_columns


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_episodes_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_tasks_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def episode_parquet_path(dataset_root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def downsample_table(
    table: pa.Table,
    *,
    step: int,
    start_offset: int,
    episode_index: int,
    global_index_start: int,
    target_fps: float,
) -> pa.Table:
    n = table.num_rows
    if n == 0:
        raise ValueError(f"Empty episode parquet for episode_index={episode_index}")
    indices = np.arange(start_offset, n, step, dtype=np.int64)
    if indices.size == 0:
        # At least one frame if original had any
        indices = np.array([0], dtype=np.int64)

    out = table.take(pa.array(indices))

    new_len = out.num_rows
    timestamps = np.arange(new_len, dtype=np.float32) / float(target_fps)
    frame_indices = np.arange(new_len, dtype=np.int64)
    global_indices = np.arange(global_index_start, global_index_start + new_len, dtype=np.int64)
    ep_indices = np.full(new_len, episode_index, dtype=np.int64)

    col_names = out.column_names
    arrays = []
    for name in col_names:
        if name == "timestamp":
            arrays.append(pa.array(timestamps))
        elif name == "frame_index":
            arrays.append(pa.array(frame_indices))
        elif name == "index":
            arrays.append(pa.array(global_indices))
        elif name == "episode_index":
            arrays.append(pa.array(ep_indices))
        else:
            arrays.append(out.column(name))

    return pa.Table.from_arrays(arrays, names=col_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample LeRobot dataset (integer FPS ratio).")
    parser.add_argument("--input", type=Path, required=True, help="Source dataset root (contains meta/, data/).")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset root (created).")
    parser.add_argument("--source-fps", type=float, default=None, help="Override meta fps (default: read from info.json).")
    parser.add_argument("--target-fps", type=float, default=20.0, help="Target fps after downsampling.")
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="First kept frame offset in [0, step); default 0 keeps frames 0, step, 2*step, ...",
    )
    parser.add_argument("--skip-stats", action="store_true", help="Do not write meta/episodes_stats.jsonl.")
    args = parser.parse_args()

    src = args.input.expanduser().resolve()
    dst = args.output.expanduser().resolve()

    if not (src / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing meta/info.json under {src}")

    info = load_json(src / "meta" / "info.json")
    source_fps = float(args.source_fps) if args.source_fps is not None else float(info["fps"])
    target_fps = float(args.target_fps)
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS values must be positive.")
    ratio = source_fps / target_fps
    step = int(round(ratio))
    if not np.isclose(ratio, step):
        raise ValueError(
            f"source_fps/target_fps must be an integer ratio; got {source_fps}/{target_fps} = {ratio}. "
            "Adjust --source-fps or --target-fps."
        )
    if step < 1:
        raise ValueError("step must be >= 1 (target fps cannot exceed source fps).")
    if args.start_offset < 0 or args.start_offset >= step:
        raise ValueError("--start-offset must satisfy 0 <= start_offset < step.")

    episodes = load_episodes_jsonl(src / "meta" / "episodes.jsonl")
    tasks = load_tasks_jsonl(src / "meta" / "tasks.jsonl") if (src / "meta" / "tasks.jsonl").is_file() else []
    chunks_size = int(info.get("chunks_size", 1000))

    sorted_indices = sorted(int(ep["episode_index"]) for ep in episodes)
    index_set = set(sorted_indices)
    if len(index_set) != len(episodes):
        raise ValueError("Duplicate episode_index in meta/episodes.jsonl.")

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    # Copy tasks unchanged
    if tasks:
        with open(dst / "meta" / "tasks.jsonl", "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")

    heavy = get_heavy_columns(info)
    new_episode_rows: Dict[int, int] = {}
    global_next = 0

    for ep_idx in tqdm(sorted_indices, desc="Downsample episodes"):
        src_pq = episode_parquet_path(src, ep_idx, chunks_size)
        if not src_pq.is_file():
            raise FileNotFoundError(f"Missing parquet: {src_pq}")

        table = pq.read_table(src_pq)
        new_table = downsample_table(
            table,
            step=step,
            start_offset=args.start_offset,
            episode_index=ep_idx,
            global_index_start=global_next,
            target_fps=target_fps,
        )

        chunk = ep_idx // chunks_size
        out_dir = dst / "data" / f"chunk-{chunk:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pq = out_dir / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(new_table, out_pq)

        new_episode_rows[ep_idx] = new_table.num_rows
        global_next += new_table.num_rows

    new_episodes: List[Dict[str, Any]] = []
    for ep in episodes:
        eid = int(ep["episode_index"])
        new_episodes.append(
            {
                **ep,
                "length": new_episode_rows[eid],
            }
        )

    new_info = dict(info)
    new_info["fps"] = int(target_fps) if float(target_fps).is_integer() else target_fps
    new_info["total_frames"] = global_next
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_chunks"] = max(1, (max(sorted_indices) // chunks_size) + 1) if sorted_indices else 1

    with open(dst / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(new_info, f, indent=4)

    with open(dst / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    if not args.skip_stats:
        with open(dst / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
            for ep_idx in tqdm(sorted_indices, desc="Episode stats"):
                out_pq = episode_parquet_path(dst, ep_idx, chunks_size)
                t = pq.read_table(out_pq)
                st = compute_episode_stats(t, heavy)
                f.write(json.dumps({"episode_index": ep_idx, "stats": st}) + "\n")

    print(
        f"Done: {len(sorted_indices)} episodes, {global_next} frames (was {info.get('total_frames')}), "
        f"fps {source_fps} -> {target_fps} (step={step}, start_offset={args.start_offset})."
    )
    print(f"Output: {dst}")
    print("Note: value_target / advantage columns are subsampled as-is; re-run value-target scripts if you need consistency at 20 Hz.")


if __name__ == "__main__":
    main()
