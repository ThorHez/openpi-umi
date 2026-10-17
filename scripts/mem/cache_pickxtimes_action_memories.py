#!/usr/bin/env python3
"""Cache frozen Round-9 and oracle memory tokens for PickXtimes action BC."""

from __future__ import annotations

import argparse
import json
import pathlib

import eval_pickxtimes_causal_event_memory as causal_eval
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from openpi.tasks.robomme.pickxtimes import semantic_memory_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--split", type=pathlib.Path, required=True)
    parser.add_argument("--detections", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _padded_sequence(starts: list[int], types: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(starts) != len(types):
        raise ValueError(f"Mismatched event starts/types: {len(starts)} vs {len(types)}")
    starts = starts[: semantic_memory_event.MAX_EVENTS]
    types = types[: semantic_memory_event.MAX_EVENTS]
    count = len(starts)
    padding_start = starts[0] if starts else 0
    padding_type = types[0] if types else semantic_memory_event.PICK_COMPLETE
    starts = starts + [padding_start] * (semantic_memory_event.MAX_EVENTS - count)
    types = types + [padding_type] * (semantic_memory_event.MAX_EVENTS - count)
    mask = np.arange(semantic_memory_event.MAX_EVENTS) < count
    return np.asarray(starts, dtype=np.int32), np.asarray(types, dtype=np.int32), mask


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["episodes"]
    metadata = {int(item["episode_index"]): item for item in labels}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_indices = [int(value) for value in split["train_episode_indices"]]
    dev_indices = [int(value) for value in split["val_episode_indices"]]
    selected = train_indices + dev_indices
    frozen_test = {int(value) for value in split.get("test_episode_indices", [])}
    if set(selected) & frozen_test:
        raise ValueError("Frozen-test leakage in selected action-memory episodes")

    detection_payload = json.loads(args.detections.read_text(encoding="utf-8"))
    calibrated = detection_payload["calibrated_threshold"]
    records = calibrated.get("calibration_episodes", []) + calibrated["episodes"]
    detections = {int(record["episode_index"]): record for record in records}
    missing = sorted(set(selected) - set(detections))
    if missing:
        raise ValueError(f"Missing causal detection records: {missing}")

    tracker = causal_eval.make_tracker(config)
    params = causal_eval.initialize_and_restore(tracker, args.checkpoint)

    @jax.jit
    def apply_tracker(windows, gripper, prompt_tokens, prompt_mask, event_types, sequence_mask):
        return tracker.apply(
            {"params": params},
            windows,
            gripper,
            prompt_tokens,
            prompt_mask,
            jnp.arange(semantic_memory_event.MAX_EVENTS, dtype=jnp.int32)[None],
            sequence_mask,
            causal_selection=False,
            candidate_valid_mask=sequence_mask,
            sequence_event_types=event_types,
            train=False,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.features, "r") as features, h5py.File(args.output, "w") as output:
        output.attrs.update(
            schema_version=1,
            checkpoint=str(args.checkpoint.resolve()),
            detections=str(args.detections.resolve()),
            source_features=str(args.features.resolve()),
            train_episodes=len(train_indices),
            dev_episodes=len(dev_indices),
            frozen_test_accessed=False,
            window_size=semantic_memory_event.WINDOW_SIZE,
        )
        for ordinal, episode_index in enumerate(selected, start=1):
            item = metadata[episode_index]
            feature_episode = features[item["episode_name"]]
            gripper_timeline = np.asarray(item["gripper_closed"], dtype=np.bool_)
            modes = {
                "oracle": (
                    [int(np.median(event["positive_starts"])) for event in item["events"]],
                    [int(event["event_type_id"]) for event in item["events"]],
                ),
                "predicted": (
                    [int(value) for value in detections[episode_index]["trigger_starts"]],
                    [int(value) for value in detections[episode_index]["predicted_types"]],
                ),
            }
            episode_group = output.create_group(item["episode_name"])
            episode_group.attrs["episode_index"] = episode_index
            initial_memory = None
            for mode, (raw_starts, raw_types) in modes.items():
                starts, event_types, mask = _padded_sequence(raw_starts, raw_types)
                windows = np.stack(
                    [
                        np.asarray(
                            feature_episode["patch_tokens"][
                                start : start + semantic_memory_event.WINDOW_SIZE
                            ]
                        )
                        for start in starts
                    ]
                )[None]
                gripper_windows = np.stack(
                    [
                        gripper_timeline[
                            start : start + semantic_memory_event.WINDOW_SIZE
                        ]
                        for start in starts
                    ]
                )[None]
                outputs = jax.device_get(
                    apply_tracker(
                        jnp.asarray(windows),
                        jnp.asarray(gripper_windows),
                        jnp.asarray(feature_episode["prompt_tokens"][0])[None],
                        jnp.asarray(feature_episode["prompt_mask"][0])[None],
                        jnp.asarray(event_types)[None],
                        jnp.asarray(mask)[None],
                    )
                )
                current_initial = np.asarray(outputs["initial_memory"][0], dtype=np.float16)
                if initial_memory is None:
                    initial_memory = current_initial
                    episode_group.create_dataset(
                        "initial_memory", data=initial_memory, compression="gzip", compression_opts=1
                    )
                elif not np.array_equal(initial_memory, current_initial):
                    raise AssertionError(f"Initial memory differs between modes for episode {episode_index}")
                count = int(mask.sum())
                mode_group = episode_group.create_group(mode)
                mode_group.create_dataset(
                    "stage_memories",
                    data=np.asarray(outputs["stage_memories"][0, :count], dtype=np.float16),
                    compression="gzip",
                    compression_opts=1,
                )
                visible_timesteps = starts[:count] + semantic_memory_event.WINDOW_SIZE - 1
                mode_group.create_dataset("visible_timesteps", data=visible_timesteps.astype(np.int32))
                mode_group.create_dataset("event_types", data=event_types[:count].astype(np.int8))
            print(f"[{ordinal}/{len(selected)}] cached episode={episode_index}", flush=True)

    print(f"Wrote frozen action memories to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
