#!/usr/bin/env python3
"""Cache frozen 4x4 visual tokens for fixed, non-overlapping RoboMME chunks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import h5py
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.mem.cache_robomme_four_task_visual_features import encode_images  # noqa: E402
from scripts.mem.cache_robomme_four_task_visual_features import load_backbone  # noqa: E402

DEFAULT_SEQUENCE = _ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_CHECKPOINT = _ROOT / (
    "checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/"
    "absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params"
)
DEFAULT_OUTPUT = _ROOT / "artifacts/robomme_four_task_fixed_chunk_features_4x4_v1_260826"
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera-view", choices=("front", "wrist"), default="front")
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Optionally cache only rows whose fixed-sequence task id matches.",
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite-incomplete", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    os.environ.setdefault("OPENPI_DATA_HOME", "/data2/hzl_workspace_for_pi_mem/.cache/openpi")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_backbone(args.checkpoint)
    started = time.monotonic()

    for split in args.splits:
        rows = _read_jsonl(args.sequence_dir / f"{split}.jsonl")
        with np.load(args.sequence_dir / f"{split}.npz", allow_pickle=False) as payload:
            frame_indices = np.asarray(payload["frame_indices"])
            step_mask = np.asarray(payload["step_mask"])
            task_ids = np.asarray(payload["task_ids"])
        if args.max_episodes is not None:
            rows = rows[: args.max_episodes]
            frame_indices = frame_indices[: args.max_episodes]
            step_mask = step_mask[: args.max_episodes]
            task_ids = task_ids[: args.max_episodes]
        output = args.output_dir / f"{split}.h5"
        with h5py.File(output, "a") as target:
            target.attrs.update(
                schema_version=1,
                checkpoint=str(args.checkpoint.resolve()),
                sequence=str((args.sequence_dir / f"{split}.npz").resolve()),
                chunk_frames=12,
                stride_frames=12,
                overlapping_windows=False,
                explicit_event_trigger=False,
                spatial_tokens=16,
                patch_width=1152,
                camera_view=args.camera_view,
                task_id=-1 if args.task_id is None else args.task_id,
            )
            for ordinal, row in enumerate(rows):
                if args.task_id is not None and int(task_ids[ordinal]) != args.task_id:
                    continue
                name = f"episode_{ordinal:06d}"
                if name in target and bool(target[name].attrs.get("complete", False)):
                    continue
                if name in target:
                    if not args.overwrite_incomplete:
                        raise RuntimeError(f"Incomplete cache group {output}:{name}")
                    del target[name]
                num_chunks = int(step_mask[ordinal].sum())
                indices = frame_indices[ordinal, :num_chunks]
                unique_indices = sorted(set(indices.reshape(-1).tolist()))
                with h5py.File(row["h5_path"], "r") as source:
                    episode = source[row["episode_name"]]
                    images = np.stack(
                        [
                            episode[
                                f"timestep_{index}/obs/{args.camera_view}_rgb"
                            ][()]
                            for index in unique_indices
                        ]
                    )
                encoded_chunks = []
                for start in range(0, len(images), args.batch_size):
                    real_count = min(args.batch_size, len(images) - start)
                    image_batch = images[start : start + real_count]
                    if real_count < args.batch_size:
                        image_batch = np.concatenate(
                            (
                                image_batch,
                                np.repeat(image_batch[-1:], args.batch_size - real_count, axis=0),
                            )
                        )
                    encoded_chunks.append(
                        np.asarray(encode_images(model, jnp.asarray(image_batch)))[:real_count]
                    )
                encoded = np.concatenate(encoded_chunks).astype(np.float16)
                lookup = {frame: index for index, frame in enumerate(unique_indices)}
                chunk_tokens = np.stack(
                    [encoded[[lookup[int(frame)] for frame in chunk]] for chunk in indices]
                )
                group = target.create_group(name)
                group.create_dataset("patch_tokens", data=chunk_tokens, compression="lzf")
                group.attrs.update(
                    complete=True,
                    source=row["source"],
                    episode_index=int(row["episode_index"]),
                    episode_name=row["episode_name"],
                    num_chunks=num_chunks,
                )
                target.flush()
                print(
                    f"{split} [{ordinal + 1}/{len(rows)}] {row['source']}:{row['episode_name']} "
                    f"chunks={num_chunks} unique_frames={len(unique_indices)} "
                    f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                    flush=True,
                )
        print(f"Completed {split}: {output}", flush=True)


if __name__ == "__main__":
    main()
