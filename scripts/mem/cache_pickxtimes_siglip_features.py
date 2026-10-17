#!/usr/bin/env python3
"""Cache frozen PaliGemma vision and language features for PickXtimes."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import flax.nnx as nnx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.training.mem.recipes import shellgame_semantic_action

DEFAULT_CHECKPOINT = pathlib.Path(
    "checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/"
    "absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite-incomplete", action="store_true")
    return parser.parse_args()


@nnx.jit
def encode_images(model, uint8_images):
    images = uint8_images.astype(jnp.float32) / 127.5 - 1.0
    images = jax.image.resize(
        images,
        (images.shape[0], 224, 224, 3),
        method="linear",
        antialias=True,
    )
    _, encoder_out = model.PaliGemma.img(images[:, None], train=False)
    return encoder_out["with_posemb"][:, 0]


@nnx.jit
def embed_prompts(model, token_ids):
    return model.PaliGemma.llm(token_ids, method="embed")


def load_backbone(checkpoint: pathlib.Path):
    checkpoint = checkpoint.expanduser().resolve()
    print(f"Restoring frozen backbone from {checkpoint}", flush=True)
    params = _model.restore_params(checkpoint, dtype=jnp.bfloat16)
    model = shellgame_semantic_action.MODEL_CONFIG.load(params)
    model.eval()
    return model


def read_frame_batch(episode: h5py.Group, start: int, stop: int, batch_size: int) -> np.ndarray:
    frames = [episode[f"timestep_{index}/obs/front_rgb"][()] for index in range(start, stop)]
    real_count = len(frames)
    if real_count < batch_size:
        frames.extend([frames[-1]] * (batch_size - real_count))
    return np.stack(frames), real_count


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    os.environ.setdefault("OPENPI_DATA_HOME", "/data2/hzl_workspace_for_pi_mem/.cache/openpi")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.labels.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    model = load_backbone(args.checkpoint)
    tokenizer = _tokenizer.PaligemmaTokenizer(shellgame_semantic_action.MODEL_CONFIG.max_token_len)
    started = time.monotonic()
    with h5py.File(args.h5, "r") as source, h5py.File(args.output, "a") as target:
        target.attrs["schema_version"] = 1
        target.attrs["source_h5"] = str(args.h5.expanduser().resolve())
        target.attrs["checkpoint"] = str(args.checkpoint.expanduser().resolve())
        target.attrs["image_size"] = 224
        target.attrs["patch_tokens"] = 256
        target.attrs["patch_width"] = 1152
        for ordinal, metadata in enumerate(episodes, start=1):
            episode_name = metadata["episode_name"]
            if episode_name in target and bool(target[episode_name].attrs.get("complete", False)):
                print(f"[{ordinal}/{len(episodes)}] {episode_name}: cached", flush=True)
                continue
            if episode_name in target:
                if not args.overwrite_incomplete:
                    raise RuntimeError(
                        f"Incomplete group {episode_name} exists; pass --overwrite-incomplete to rebuild it"
                    )
                del target[episode_name]

            source_episode = source[episode_name]
            num_steps = int(metadata["num_steps"])
            group = target.create_group(episode_name)
            patches = group.create_dataset(
                "patch_tokens",
                shape=(num_steps, 256, 1152),
                dtype=np.float16,
                chunks=(1, 256, 1152),
            )
            episode_started = time.monotonic()
            for start in range(0, num_steps, args.batch_size):
                stop = min(start + args.batch_size, num_steps)
                frame_batch, real_count = read_frame_batch(source_episode, start, stop, args.batch_size)
                encoded = np.asarray(encode_images(model, jnp.asarray(frame_batch)))[:real_count]
                patches[start:stop] = encoded.astype(np.float16)

            prompt_ids, prompt_masks = zip(*(tokenizer.tokenize(prompt) for prompt in metadata["prompts"]), strict=True)
            prompt_ids = np.stack(prompt_ids).astype(np.int32)
            prompt_masks = np.stack(prompt_masks).astype(np.bool_)
            prompt_tokens = np.asarray(embed_prompts(model, jnp.asarray(prompt_ids)))
            group.create_dataset("prompt_tokens", data=prompt_tokens.astype(np.float16))
            group.create_dataset("prompt_mask", data=prompt_masks)
            group.attrs["num_steps"] = num_steps
            group.attrs["complete"] = True
            target.flush()
            elapsed = time.monotonic() - episode_started
            total_elapsed = time.monotonic() - started
            print(
                f"[{ordinal}/{len(episodes)}] {episode_name}: {num_steps} frames in {elapsed:.1f}s "
                f"(total {total_elapsed / 60:.1f} min)",
                flush=True,
            )


if __name__ == "__main__":
    main()
