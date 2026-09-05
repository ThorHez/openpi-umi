#!/usr/bin/env python3
"""Export a synchronized original-vs-augmentation preview for cup_0904."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.mem.train_shellgame_real_memory_robust_cup0904 import Cup0904Reader  # noqa: E402
from scripts.mem.train_shellgame_real_memory_robust_cup0904 import augment_history  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "evaluation/shellgame_real/cup0904_fullframe_augconsistency_seed42_v1/augmentation_preview",
    )
    return parser.parse_args()


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (223, 25), (0, 0, 0), thickness=-1)
    cv2.putText(
        result,
        title,
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return result


def make_grid(views: list[np.ndarray], frame_index: int) -> np.ndarray:
    tiles = []
    for index, view in enumerate(views):
        title = "original" if index == 0 else f"aug {index - 1}"
        tile = cv2.cvtColor(view[frame_index], cv2.COLOR_RGB2BGR)
        tiles.append(add_title(tile, title))
    rows = [np.concatenate(tiles[row * 3 : (row + 1) * 3], axis=1) for row in range(3)]
    return np.concatenate(rows, axis=0)


def main() -> None:
    args = parse_args()
    if not 0 <= args.episode < 21:
        raise ValueError("--episode must be in [0, 20]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = Cup0904Reader().read(args.episode)
    views = [original]
    for variant in range(8):
        augmentation_seed = args.seed * 1_000_003 + args.episode * 101 + variant
        views.append(augment_history(original, seed=augmentation_seed))

    stem = f"cup0904_ep{args.episode:03d}_original_plus_aug8"
    video_path = args.output_dir / f"{stem}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (224 * 3, 224 * 3),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the mp4v video writer")
    for frame_index in range(original.shape[0]):
        writer.write(make_grid(views, frame_index))
    writer.release()

    keyframes = (0, 80, 160, 240)
    for frame_index in keyframes:
        image_path = args.output_dir / f"{stem}_frame{frame_index:03d}.jpg"
        if not cv2.imwrite(str(image_path), make_grid(views, frame_index)):
            raise RuntimeError(f"Could not write {image_path}")
    print(video_path.resolve())
    for frame_index in keyframes:
        print((args.output_dir / f"{stem}_frame{frame_index:03d}.jpg").resolve())


if __name__ == "__main__":
    main()
