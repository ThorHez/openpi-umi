#!/usr/bin/env python3
"""Cache frozen PaliGemma image/language features for VideoUnmask demo frames."""

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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite-incomplete", action="store_true")
    return parser.parse_args()


@nnx.jit
def encode_images(model, uint8_images):
    images = uint8_images.astype(jnp.float32) / 127.5 - 1.0
    images = jax.image.resize(images, (images.shape[0], 224, 224, 3), method="linear", antialias=True)
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


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    os.environ.setdefault("OPENPI_DATA_HOME", "/data2/hzl_workspace_for_pi_mem/.cache/openpi")
    payload = json.loads(args.labels.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = load_backbone(args.checkpoint)
    tokenizer = _tokenizer.PaligemmaTokenizer(shellgame_semantic_action.MODEL_CONFIG.max_token_len)
    started = time.monotonic()
    with h5py.File(args.h5, "r") as source, h5py.File(args.output, "a") as target:
        target.attrs["schema_version"] = 1
        target.attrs["source_h5"] = str(args.h5.expanduser().resolve())
        target.attrs["checkpoint"] = str(args.checkpoint.expanduser().resolve())
        target.attrs["num_demo_frames"] = int(payload["num_demo_frames"])
        for ordinal, metadata in enumerate(episodes, start=1):
            episode_name = metadata["episode_name"]
            if episode_name in target and bool(target[episode_name].attrs.get("complete", False)):
                print(f"[{ordinal}/{len(episodes)}] {episode_name}: cached", flush=True)
                continue
            if episode_name in target:
                if not args.overwrite_incomplete:
                    raise RuntimeError(f"Incomplete group {episode_name}; pass --overwrite-incomplete")
                del target[episode_name]

            episode = source[episode_name]
            indices = [int(index) for index in metadata["demo_indices"]]
            images = np.stack([episode[f"timestep_{index}/obs/front_rgb"][()] for index in indices])
            real_count = len(images)
            if real_count < args.batch_size:
                images = np.concatenate((images, np.repeat(images[-1:], args.batch_size - real_count, axis=0)))
            patch_tokens = np.asarray(encode_images(model, jnp.asarray(images)))[:real_count]

            prompt_ids, prompt_masks = zip(
                *(tokenizer.tokenize(prompt) for prompt in metadata["prompts"]), strict=True
            )
            prompt_ids = np.stack(prompt_ids).astype(np.int32)
            prompt_masks = np.stack(prompt_masks).astype(np.bool_)
            prompt_tokens = np.asarray(embed_prompts(model, jnp.asarray(prompt_ids)))

            group = target.create_group(episode_name)
            group.create_dataset("demo_patch_tokens", data=patch_tokens.astype(np.float16), compression="lzf")
            group.create_dataset("prompt_tokens", data=prompt_tokens.astype(np.float16), compression="lzf")
            group.create_dataset("prompt_mask", data=prompt_masks)
            group.create_dataset("demo_indices", data=np.asarray(indices, dtype=np.int32))
            group.attrs["complete"] = True
            target.flush()
            print(
                f"[{ordinal}/{len(episodes)}] {episode_name}: {real_count} frames "
                f"(total {(time.monotonic() - started) / 60:.1f} min)",
                flush=True,
            )


if __name__ == "__main__":
    main()
