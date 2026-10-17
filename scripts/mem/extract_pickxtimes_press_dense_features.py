#!/usr/bin/env python3
"""Extract frozen post-final-PLACE features for dense PRESS training."""

from __future__ import annotations

import argparse
import json
import pathlib

import eval_pickxtimes_causal_event_memory as causal_eval
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip_mem_semantic as memory_core
from openpi.tasks.robomme.pickxtimes import semantic_memory_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--split", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--window-batch-size", type=int, default=16)
    return parser.parse_args()


def scan_segment(
    patch_dataset: h5py.Dataset,
    gripper_closed: np.ndarray,
    *,
    first_start: int,
    batch_size: int,
    classifier_apply,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_windows = patch_dataset.shape[0] - semantic_memory_event.WINDOW_SIZE + 1
    starts_out = []
    logits_out = []
    semantic_out = []
    for start in range(first_start, num_windows, batch_size):
        stop = min(start + batch_size, num_windows)
        timeline = np.asarray(patch_dataset[start : stop + semantic_memory_event.WINDOW_SIZE - 1])
        windows = np.stack(
            [timeline[offset : offset + semantic_memory_event.WINDOW_SIZE] for offset in range(stop - start)]
        )
        gripper_windows = np.stack(
            [gripper_closed[index : index + semantic_memory_event.WINDOW_SIZE] for index in range(start, stop)]
        )
        real_count = windows.shape[0]
        if real_count < batch_size:
            padding = batch_size - real_count
            windows = np.concatenate((windows, np.repeat(windows[-1:], padding, axis=0)), axis=0)
            gripper_windows = np.concatenate(
                (gripper_windows, np.repeat(gripper_windows[-1:], padding, axis=0)), axis=0
            )
        event_logits, semantic_features = classifier_apply(
            jnp.asarray(windows),
            jnp.asarray(gripper_windows),
        )
        starts_out.append(np.arange(start, stop, dtype=np.int32))
        logits_out.append(np.asarray(event_logits)[:real_count])
        semantic_out.append(np.asarray(semantic_features)[:real_count])
    return np.concatenate(starts_out), np.concatenate(logits_out), np.concatenate(semantic_out)


def main() -> None:
    args = parse_args()
    if args.window_batch_size < 1:
        raise ValueError("--window-batch-size must be positive")
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["episodes"]
    metadata_by_index = {int(episode["episode_index"]): episode for episode in labels}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_indices = [int(value) for value in split["train_episode_indices"]]
    dev_indices = [int(value) for value in split["val_episode_indices"]]
    selected_indices = train_indices + dev_indices
    if set(train_indices) & set(dev_indices):
        raise ValueError("Train/dev overlap")

    tracker = causal_eval.make_tracker(config)
    params = causal_eval.initialize_and_restore(tracker, args.checkpoint)
    classifier = semantic_memory_event.PickXtimesSlidingWindowEventClassifier(
        input_width=tracker.input_width,
        width=tracker.encoder_width,
        depth=tracker.encoder_depth,
        num_heads=tracker.encoder_heads,
        dtype_mm=tracker.dtype_mm,
    )
    gate_fusion = semantic_memory_event.PickXtimesGripperGateFusion()

    @jax.jit
    def classifier_apply(windows, gripper_windows):
        pooled = memory_core.pool_fixed_grid(windows, pool_factor=2)
        visual_event_logits, _, semantic_features = classifier.apply(
            {"params": params["window_classifier"]},
            pooled,
            train=False,
            return_features=True,
        )
        event_logits = gate_fusion.apply(
            {"params": params["gripper_gate_fusion"]},
            visual_event_logits,
            gripper_windows,
        )
        return event_logits, semantic_features

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.features, "r") as source, h5py.File(args.output, "w") as output:
        output.attrs["source_checkpoint"] = str(args.checkpoint.resolve())
        output.attrs["train_episode_indices"] = json.dumps(train_indices)
        output.attrs["dev_episode_indices"] = json.dumps(dev_indices)
        for ordinal, episode_index in enumerate(selected_indices, start=1):
            metadata = metadata_by_index[episode_index]
            final_place = metadata["events"][-2]
            press_event = metadata["events"][-1]
            first_start = int(final_place["anchor"]) + 1
            starts, base_logits, semantic_features = scan_segment(
                source[f"{metadata['episode_name']}/patch_tokens"],
                np.asarray(metadata["gripper_closed"], dtype=np.bool_),
                first_start=first_start,
                batch_size=args.window_batch_size,
                classifier_apply=classifier_apply,
            )
            group = output.create_group(str(episode_index))
            group.create_dataset("starts", data=starts, compression="gzip")
            group.create_dataset("base_event_logits", data=base_logits.astype(np.float32), compression="gzip")
            group.create_dataset("semantic_features", data=semantic_features.astype(np.float16), compression="gzip")
            group.create_dataset(
                "positive_mask",
                data=np.isin(starts, np.asarray(press_event["positive_starts"], dtype=np.int32)),
                compression="gzip",
            )
            group.attrs["final_place_anchor"] = int(final_place["anchor"])
            group.attrs["press_anchor"] = int(press_event["anchor"])
            print(
                f"extract {ordinal}/{len(selected_indices)} episode={episode_index} windows={len(starts)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
