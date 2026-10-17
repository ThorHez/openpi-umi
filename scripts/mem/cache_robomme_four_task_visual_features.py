#!/usr/bin/env python3
"""Cache frozen 4x4 PaliGemma/SigLIP tokens for four-task student windows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import flax.nnx as nnx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import siglip_mem_semantic as memory_core
from openpi.training.mem.recipes import shellgame_semantic_action

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = _ROOT / "artifacts/robomme_four_task_visual_student_sequences_v1_260826"
DEFAULT_CHECKPOINT = _ROOT / (
    "checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/"
    "absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params"
)
DEFAULT_OUTPUT = _ROOT / "artifacts/robomme_four_task_visual_features_4x4_v1_260826"
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pool-factor", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite-incomplete", action="store_true")
    return parser.parse_args()


def make_image_encoder(pool_factor: int):
    """Create a compiled frozen-image encoder for one static pooling factor."""

    if pool_factor not in (1, 2, 4):
        raise ValueError(f"Unsupported pool factor: {pool_factor}")

    @nnx.jit
    def encode(model, uint8_images):
        images = uint8_images.astype(jnp.float32) / 127.5 - 1.0
        images = jax.image.resize(
            images,
            (images.shape[0], 224, 224, 3),
            method="linear",
            antialias=True,
        )
        _, encoder_out = model.PaliGemma.img(images[:, None], train=False)
        patches = encoder_out["with_posemb"][:, 0]
        return memory_core.pool_fixed_grid(
            patches[:, None], pool_factor=pool_factor
        )[:, 0]

    return encode


encode_images = make_image_encoder(pool_factor=4)


def load_backbone(checkpoint: Path):
    print(f"Restoring frozen visual backbone from {checkpoint.resolve()}", flush=True)
    params = _model.restore_params(checkpoint.resolve(), dtype=jnp.bfloat16)
    model = shellgame_semantic_action.MODEL_CONFIG.load(params)
    model.eval()
    return model


def _read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_unique_images(row: dict) -> tuple[np.ndarray, list[int]]:
    indices = sorted({index for event in row["events"] for index in event["frame_indices"]})
    with h5py.File(row["h5_path"], "r") as source:
        episode = source[row["episode_name"]]
        images = np.stack(
            [episode[f"timestep_{index}/obs/front_rgb"][()] for index in indices]
        )
    return images, indices


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    os.environ.setdefault("OPENPI_DATA_HOME", "/data2/hzl_workspace_for_pi_mem/.cache/openpi")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_backbone(args.checkpoint)
    image_encoder = make_image_encoder(args.pool_factor)
    grid_size = 16 // args.pool_factor
    started = time.monotonic()

    for split in args.splits:
        rows = _read_manifest(args.manifest_dir / f"{split}.jsonl")
        if args.max_episodes is not None:
            rows = rows[: args.max_episodes]
        output = args.output_dir / f"{split}.h5"
        with h5py.File(output, "a") as target:
            target.attrs.update(
                schema_version=1,
                checkpoint=str(args.checkpoint.resolve()),
                manifest=str((args.manifest_dir / f"{split}.jsonl").resolve()),
                pool_factor=args.pool_factor,
                spatial_tokens=grid_size**2,
                patch_width=1152,
                window_frames=12,
            )
            for ordinal, row in enumerate(rows):
                name = f"episode_{ordinal:06d}"
                if name in target and bool(target[name].attrs.get("complete", False)):
                    continue
                if name in target:
                    if not args.overwrite_incomplete:
                        raise RuntimeError(f"Incomplete cache group {output}:{name}")
                    del target[name]

                images, unique_indices = _read_unique_images(row)
                encoded_chunks = []
                for start in range(0, len(images), args.batch_size):
                    real = images[start : start + args.batch_size]
                    if len(real) < args.batch_size:
                        real = np.concatenate(
                            (real, np.repeat(real[-1:], args.batch_size - len(real), axis=0))
                        )
                    encoded = np.asarray(image_encoder(model, jnp.asarray(real)))
                    encoded_chunks.append(encoded[: min(args.batch_size, len(images) - start)])
                encoded = np.concatenate(encoded_chunks).astype(np.float16)
                lookup = {frame: index for index, frame in enumerate(unique_indices)}
                event_tokens = np.stack(
                    [encoded[[lookup[index] for index in event["frame_indices"]]] for event in row["events"]]
                )
                group = target.create_group(name)
                group.create_dataset("patch_tokens", data=event_tokens, compression="lzf")
                group.attrs.update(
                    complete=True,
                    source=row["source"],
                    episode_index=int(row["episode_index"]),
                    episode_name=row["episode_name"],
                    num_events=len(row["events"]),
                )
                target.flush()
                print(
                    f"{split} [{ordinal + 1}/{len(rows)}] {row['source']}:{row['episode_name']} "
                    f"events={len(row['events'])} unique_frames={len(images)} "
                    f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                    flush=True,
                )
        print(f"Completed {split}: {output}", flush=True)


if __name__ == "__main__":
    main()
