#!/usr/bin/env python3
"""Build deterministic VideoUnmask demo-to-target supervision from RoboMME HDF5."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

import h5py
import numpy as np

TARGET_RE = re.compile(r"hiding the (red|green|blue) cube")
NUM_DEMO_FRAMES = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=260823)
    return parser.parse_args()


def _decode(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes | np.bytes_) else str(value)


def _timestep_names(episode: h5py.Group) -> list[str]:
    return sorted(
        (key for key in episode if key.startswith("timestep_")),
        key=lambda key: int(key.split("_")[-1]),
    )


def _first_execution_target(episode: h5py.Group, timesteps: list[str]) -> tuple[int, list[int]]:
    for index, name in enumerate(timesteps):
        timestep = episode[name]
        if bool(timestep["info/is_video_demo"][()]):
            continue
        payload = json.loads(_decode(timestep["action/choice_action"][()]))
        point = payload.get("point", [])
        if str(payload.get("choice", "")).upper() == "A" and len(point) == 2:
            return index, [int(point[0]), int(point[1])]
    raise ValueError(f"No first execution pick target in {episode.name}")


def _stratified_split(episodes: list[dict], val_count: int, seed: int) -> tuple[list[int], list[int]]:
    if not 0 < val_count < len(episodes):
        raise ValueError(f"val_count must be in (0, {len(episodes)}), got {val_count}")
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, str, int], list[int]] = collections.defaultdict(list)
    for episode in episodes:
        key = (episode["target_color"], episode["difficulty"], episode["num_targets"])
        strata[key].append(int(episode["episode_index"]))
    for indices in strata.values():
        rng.shuffle(indices)

    exact = {key: len(indices) * val_count / len(episodes) for key, indices in strata.items()}
    allocation = {key: int(np.floor(value)) for key, value in exact.items()}
    remaining = val_count - sum(allocation.values())
    order = sorted(strata, key=lambda key: (exact[key] - allocation[key], len(strata[key])), reverse=True)
    for key in order[:remaining]:
        allocation[key] += 1

    val = sorted(index for key, indices in strata.items() for index in indices[: allocation[key]])
    val_set = set(val)
    train = sorted(int(episode["episode_index"]) for episode in episodes if episode["episode_index"] not in val_set)
    return train, val


def main() -> None:
    args = parse_args()
    episodes = []
    with h5py.File(args.h5, "r") as source:
        episode_names = sorted(
            (name for name in source if name.startswith("episode_")),
            key=lambda name: int(name.split("_")[-1]),
        )
        for episode_name in episode_names:
            episode = source[episode_name]
            episode_index = int(episode_name.split("_")[-1])
            timesteps = _timestep_names(episode)
            demo_mask = [bool(episode[f"{name}/info/is_video_demo"][()]) for name in timesteps]
            demo_count = sum(demo_mask)
            if demo_count < NUM_DEMO_FRAMES or not all(demo_mask[:demo_count]) or any(demo_mask[demo_count:]):
                raise ValueError(f"Expected one contiguous demo prefix in {episode_name}: {demo_count=}")
            demo_indices = np.linspace(0, demo_count - 1, NUM_DEMO_FRAMES).round().astype(int).tolist()
            if len(set(demo_indices)) != NUM_DEMO_FRAMES:
                raise ValueError(f"Duplicate sampled demo indices in {episode_name}: {demo_indices}")

            prompts = [_decode(value) for value in episode["setup/task_goal"][()]]
            match = TARGET_RE.search(prompts[0])
            if match is None:
                raise ValueError(f"Cannot parse first target color from {prompts[0]!r}")
            execution_index, target_point = _first_execution_target(episode, timesteps)
            episodes.append(
                {
                    "episode_name": episode_name,
                    "episode_index": episode_index,
                    "num_steps": len(timesteps),
                    "demo_count": demo_count,
                    "demo_indices": demo_indices,
                    "execution_start": demo_count,
                    "first_target_action_index": execution_index,
                    "target_point_yx": target_point,
                    "target_point_normalized_yx": [target_point[0] / 255.0, target_point[1] / 255.0],
                    "target_color": match.group(1),
                    "difficulty": _decode(episode["setup/difficulty"][()]),
                    "num_targets": len(TARGET_RE.findall(prompts[0])),
                    "prompts": prompts,
                }
            )

    train, val = _stratified_split(episodes, args.val_episodes, args.seed)
    payload = {
        "schema_version": 1,
        "source_h5": str(args.h5.expanduser().resolve()),
        "image_size": 256,
        "num_demo_frames": NUM_DEMO_FRAMES,
        "seed": args.seed,
        "train_episode_indices": train,
        "val_episode_indices": val,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(episodes)} episodes: train={len(train)} val={len(val)} to {args.output}")
    for split_name, indices in (("train", train), ("val", val)):
        selected = [episodes[index] for index in indices]
        counts = collections.Counter((item["target_color"], item["difficulty"], item["num_targets"]) for item in selected)
        print(f"{split_name}: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
