#!/usr/bin/env python3
"""Cache online fixed-chunk raw-RGB grid statistics for RoboMME."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mem.cache_robomme_event_rgb_grid_features import _grid_features_batch  # noqa: E402

DEFAULT_SEQUENCE = ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_OUTPUT = ROOT / "artifacts/robomme_fixed_chunk_rgb_grid8_v1_260829"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _process_row(payload):
    row_index, row, indices, grid_size = payload
    unique = sorted(set(indices.reshape(-1).tolist()))
    with h5py.File(row["h5_path"], "r") as source:
        episode = source[row["episode_name"]]
        images = np.stack(
            [episode[f"timestep_{frame}/obs/front_rgb"][()] for frame in unique]
        )
    encoded = _grid_features_batch(images, grid_size)
    features = dict(zip(unique, encoded, strict=True))
    chunks = np.stack(
        [np.stack([features[int(frame)] for frame in chunk]) for chunk in indices]
    )
    return row_index, row, chunks.astype(np.float16)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        rows = _read_jsonl(args.sequence_dir / f"{split}.jsonl")
        with np.load(args.sequence_dir / f"{split}.npz", allow_pickle=False) as payload:
            frame_indices = np.asarray(payload["frame_indices"])
            step_mask = np.asarray(payload["step_mask"])
            task_ids = np.asarray(payload["task_ids"])
        output = args.output_dir / f"{split}.h5"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite")
        with h5py.File(output, "w") as target:
            target.attrs.update(
                schema_version=1,
                sequence=str((args.sequence_dir / f"{split}.npz").resolve()),
                feature_type="raw_rgb_cell_mean_std_gradient_xy",
                grid_size=args.grid_size,
                spatial_tokens=args.grid_size**2,
                patch_width=12,
                chunk_frames=12,
                stride_frames=12,
                overlapping_windows=False,
            )
            tasks = []
            for row_index, row in enumerate(rows):
                if int(task_ids[row_index]) < 3:
                    count = int(step_mask[row_index].sum())
                    tasks.append(
                        (
                            row_index,
                            row,
                            frame_indices[row_index, :count],
                            args.grid_size,
                        )
                    )
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                processed = executor.map(_process_row, tasks, chunksize=1)
                for completed, (row_index, row, chunks) in enumerate(processed, start=1):
                    count = len(chunks)
                    group = target.create_group(f"episode_{row_index:06d}")
                    group.create_dataset(
                        "patch_tokens",
                        data=chunks,
                        compression="lzf",
                    )
                    group.attrs.update(
                        complete=True,
                        source=row["source"],
                        episode_index=int(row["episode_index"]),
                        episode_name=row["episode_name"],
                        num_chunks=count,
                    )
                    if completed % 20 == 0:
                        target.flush()
                        print(
                            f"{split}: {completed}/{len(tasks)} episodes",
                            flush=True,
                        )
        print(f"Completed {split}: {output}", flush=True)


if __name__ == "__main__":
    main()
