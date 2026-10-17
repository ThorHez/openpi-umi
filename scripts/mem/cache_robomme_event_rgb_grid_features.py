#!/usr/bin/env python3
"""Cache controlled 4x4 or 8x8 raw-RGB grid statistics for event windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "artifacts/robomme_four_task_visual_student_sequences_v1_260826"
DEFAULT_OUTPUT = ROOT / "artifacts/robomme_event_rgb_grid8_v1_260829"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-size", type=int, choices=(4, 8), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _grid_features_batch(images: np.ndarray, grid_size: int) -> np.ndarray:
    """Vectorized cell statistics for images shaped [N, H, W, 3]."""

    height, width = images.shape[1:3]
    usable_height = height - height % grid_size
    usable_width = width - width % grid_size
    images = images[:, :usable_height, :usable_width].astype(np.float32) / 255.0
    cell_height = usable_height // grid_size
    cell_width = usable_width // grid_size
    cells = images.reshape(
        len(images), grid_size, cell_height, grid_size, cell_width, 3
    ).transpose(0, 1, 3, 2, 4, 5)
    mean = cells.mean(axis=(3, 4))
    std = cells.std(axis=(3, 4))
    gradient_x = np.abs(np.diff(cells, axis=4)).mean(axis=(3, 4))
    gradient_y = np.abs(np.diff(cells, axis=3)).mean(axis=(3, 4))
    return np.concatenate((mean, std, gradient_x, gradient_y), axis=-1).reshape(
        len(images), grid_size**2, 12
    )


def _grid_features(image: np.ndarray, grid_size: int) -> np.ndarray:
    return _grid_features_batch(image[None], grid_size)[0]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        rows = _read_jsonl(args.manifest_dir / f"{split}.jsonl")
        output = args.output_dir / f"{split}.h5"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite")
        with h5py.File(output, "w") as target:
            target.attrs.update(
                schema_version=1,
                manifest=str((args.manifest_dir / f"{split}.jsonl").resolve()),
                feature_type="raw_rgb_cell_mean_std_gradient_xy",
                grid_size=args.grid_size,
                spatial_tokens=args.grid_size**2,
                patch_width=12,
                window_frames=12,
            )
            for row_index, row in enumerate(rows):
                with h5py.File(row["h5_path"], "r") as source:
                    episode = source[row["episode_name"]]
                    event_features = [
                        np.stack(
                            [
                                _grid_features(
                                    episode[f"timestep_{frame}/obs/front_rgb"][()],
                                    args.grid_size,
                                )
                                for frame in event["frame_indices"]
                            ]
                        )
                        for event in row["events"]
                    ]
                group = target.create_group(f"episode_{row_index:06d}")
                group.create_dataset(
                    "patch_tokens",
                    data=np.asarray(event_features, dtype=np.float16),
                    compression="lzf",
                )
                group.attrs.update(
                    complete=True,
                    source=row["source"],
                    episode_index=int(row["episode_index"]),
                    episode_name=row["episode_name"],
                    num_events=len(row["events"]),
                )
        print(f"Completed {split}: {output}", flush=True)


if __name__ == "__main__":
    main()
