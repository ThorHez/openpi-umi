#!/usr/bin/env python3
"""Create a deterministic episode-level PickXtimes train/dev/test split."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

FIELDS = ("required_count", "target_color", "difficulty")


def _histograms(episodes):
    return {field: dict(collections.Counter(str(episode[field]) for episode in episodes)) for field in FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260822)
    parser.add_argument("--trials", type=int, default=100_000)
    args = parser.parse_args()

    episodes = json.loads(args.labels.read_text(encoding="utf-8"))["episodes"]
    if not 1 <= args.val_episodes < len(episodes):
        raise ValueError(f"val_episodes must be in [1,{len(episodes) - 1}]")
    if args.test_episodes < 0 or args.val_episodes + args.test_episodes >= len(episodes):
        raise ValueError("test_episodes must be non-negative and leave at least one training episode")
    rng = np.random.default_rng(args.seed)
    full_hist = _histograms(episodes)
    val_ratio = args.val_episodes / len(episodes)
    test_ratio = args.test_episodes / len(episodes)
    best_score = float("inf")
    best_indices = None
    for _ in range(args.trials):
        candidate = rng.choice(len(episodes), size=args.val_episodes + args.test_episodes, replace=False)
        val_candidate = np.sort(candidate[: args.val_episodes])
        test_candidate = np.sort(candidate[args.val_episodes :])
        val_hist = _histograms([episodes[index] for index in val_candidate])
        test_hist = _histograms([episodes[index] for index in test_candidate])
        if any(len(val_hist[field]) != len(full_hist[field]) for field in FIELDS):
            continue
        if args.test_episodes and any(len(test_hist[field]) != len(full_hist[field]) for field in FIELDS):
            continue
        score = 0.0
        for histogram, ratio in ((val_hist, val_ratio), (test_hist, test_ratio)):
            if ratio == 0:
                continue
            for field in FIELDS:
                for value, total in full_hist[field].items():
                    expected = total * ratio
                    observed = histogram[field].get(value, 0)
                    score += (observed - expected) ** 2 / max(expected, 1.0)
        if score < best_score:
            best_score = score
            best_indices = (val_candidate, test_candidate)
    if best_indices is None:
        raise RuntimeError("No split covered every value of required_count, target_color, and difficulty")

    val_set = set(best_indices[0].tolist())
    test_set = set(best_indices[1].tolist())
    train = [episode for index, episode in enumerate(episodes) if index not in val_set and index not in test_set]
    val = [episode for index, episode in enumerate(episodes) if index in val_set]
    test = [episode for index, episode in enumerate(episodes) if index in test_set]
    output = {
        "seed": args.seed,
        "score": best_score,
        "train_episode_indices": [int(episode["episode_index"]) for episode in train],
        "val_episode_indices": [int(episode["episode_index"]) for episode in val],
        "test_episode_indices": [int(episode["episode_index"]) for episode in test],
        "full_distribution": full_hist,
        "train_distribution": _histograms(train),
        "val_distribution": _histograms(val),
        "test_distribution": _histograms(test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
