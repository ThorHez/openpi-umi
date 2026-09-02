#!/usr/bin/env python3
"""Cache deployable target-conditioned RGB motion features for PickXTimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE = ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_OUTPUT = ROOT / "artifacts/pickxtimes_target_motion_features_v1_260831"
SPLITS = ("train", "dev", "test")
FEATURE_NAMES = (
    "front_target_y",
    "front_target_x",
    "front_target_sqrt_area",
    "front_target_visible",
    "wrist_target_y",
    "wrist_target_x",
    "wrist_target_sqrt_area",
    "wrist_target_visible",
    "eef_x",
    "eef_y",
    "eef_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def target_mask(image: np.ndarray, color_id: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    red, green, blue = (image[..., index] for index in range(3))
    masks = (
        (red > 180) & (green < 70) & (blue < 70),
        (green > 180) & (red < 70) & (blue < 70),
        (blue > 180) & (red < 70) & (green < 70),
    )
    if color_id not in (1, 2, 3):
        raise ValueError(f"Unsupported target color id {color_id}")
    return masks[color_id - 1]


def image_features(image: np.ndarray, color_id: int) -> np.ndarray:
    mask = target_mask(image, color_id)
    y, x = np.nonzero(mask)
    if len(y) < 6:
        return np.zeros(4, dtype=np.float32)
    height, width = mask.shape
    return np.asarray(
        (
            2.0 * float(np.median(y)) / max(height - 1, 1) - 1.0,
            2.0 * float(np.median(x)) / max(width - 1, 1) - 1.0,
            np.sqrt(float(len(y)) / float(height * width)),
            1.0,
        ),
        dtype=np.float32,
    )


def frame_features(episode: h5py.Group, frame: int, color_id: int) -> np.ndarray:
    obs = episode[f"timestep_{frame}/obs"]
    front = image_features(obs["front_rgb"][()], color_id)
    wrist = image_features(obs["wrist_rgb"][()], color_id)
    eef = np.asarray(obs["eef_state_raw/pose"][()], dtype=np.float32)
    return np.concatenate((front, wrist, eef), axis=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / f"{split}.h5" for split in SPLITS]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Outputs already exist in {args.output_dir}")

    train_sum = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    train_sq = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    train_count = 0
    split_summary = {}
    for split in SPLITS:
        rows = read_jsonl(args.sequence_dir / f"{split}.jsonl")
        with np.load(args.sequence_dir / f"{split}.npz", allow_pickle=False) as payload:
            task_ids = np.asarray(payload["task_ids"])
            colors = np.asarray(payload["goal_color_ids"])
            frame_indices = np.asarray(payload["frame_indices"])
            step_mask = np.asarray(payload["step_mask"])
        output = args.output_dir / f"{split}.h5"
        if output.exists():
            output.unlink()
        episode_count = 0
        chunk_count = 0
        visible_front = 0
        visible_wrist = 0
        with h5py.File(output, "w") as target:
            target.attrs.update(
                schema_version=1,
                task="PickXtimes",
                feature_names=json.dumps(FEATURE_NAMES),
                chunk_frames=12,
                source="front/wrist RGB color masks plus observed EEF pose",
                privileged_inputs=False,
            )
            for row_index, row in enumerate(rows):
                if int(task_ids[row_index]) != 3:
                    continue
                length = int(step_mask[row_index].sum())
                indices = frame_indices[row_index, :length]
                color_id = int(np.asarray(colors[row_index]).reshape(-1)[0])
                unique = sorted(set(indices.reshape(-1).tolist()))
                with h5py.File(row["h5_path"], "r") as source:
                    episode = source[row["episode_name"]]
                    encoded = {
                        frame: frame_features(episode, frame, color_id) for frame in unique
                    }
                chunks = np.stack(
                    [np.stack([encoded[int(frame)] for frame in chunk]) for chunk in indices]
                ).astype(np.float32)
                group = target.create_group(f"episode_{row_index:06d}")
                group.create_dataset("target_motion", data=chunks, compression="lzf")
                group.attrs.update(
                    source=row["source"],
                    episode_index=int(row["episode_index"]),
                    target_color_id=color_id,
                    num_chunks=length,
                )
                episode_count += 1
                chunk_count += length
                visible_front += int(chunks[..., 3].sum())
                visible_wrist += int(chunks[..., 7].sum())
                if split == "train":
                    flat = chunks.reshape(-1, chunks.shape[-1]).astype(np.float64)
                    train_sum += flat.sum(axis=0)
                    train_sq += np.square(flat).sum(axis=0)
                    train_count += len(flat)
        split_summary[split] = {
            "episodes": episode_count,
            "chunks": chunk_count,
            "front_visible_fraction": visible_front / max(chunk_count * 12, 1),
            "wrist_visible_fraction": visible_wrist / max(chunk_count * 12, 1),
        }
        print(json.dumps({"split": split, **split_summary[split]}), flush=True)

    mean = train_sum / train_count
    variance = np.maximum(train_sq / train_count - np.square(mean), 1e-8)
    summary = {
        "schema_version": 1,
        "feature_names": FEATURE_NAMES,
        "normalization": {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()},
        "splits": split_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
