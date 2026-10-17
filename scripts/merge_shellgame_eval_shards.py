"""Merge non-overlapping ShellGame evaluation shard results."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


COUNT_KEYS = (
    "num_trials",
    "strict_success",
    "cup_selection_correct",
    "cup_selection_decisions",
    "correct_selection_and_contact",
    "target_cup_contact",
    "any_cup_contact",
    "target_cup_lift_success",
    "any_cup_lift_success",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.shards]
    episodes = sorted(
        (episode for payload in payloads for episode in payload["episodes"]),
        key=lambda episode: episode["trial"],
    )
    trials = [int(episode["trial"]) for episode in episodes]
    seeds = [int(episode["episode_seed"]) for episode in episodes]
    expected_trials = list(range(len(episodes)))
    if trials != expected_trials:
        raise RuntimeError(f"Expected trials {expected_trials}, got {trials}")
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Episode seeds are not unique")

    protocol = dict(payloads[0]["protocol"])
    protocol["num_trials"] = len(episodes)
    protocol["trial_start"] = 0
    merged = {
        "protocol": protocol,
        **{key: sum(int(payload[key]) for payload in payloads) for key in COUNT_KEYS},
        "episodes": episodes,
    }
    if merged["num_trials"] != len(episodes):
        raise RuntimeError(
            f"Shard trial total {merged['num_trials']} != episode count {len(episodes)}"
        )

    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    summary = {key: merged[key] for key in COUNT_KEYS}
    summary.update(
        episode_count=len(episodes),
        unique_trials=len(set(trials)),
        trial_min=min(trials),
        trial_max=max(trials),
        unique_seeds=len(set(seeds)),
        target_distribution=dict(Counter(episode["target_cup"] for episode in episodes)),
        selected_distribution=dict(Counter(episode["selected_cup"] for episode in episodes)),
        contact_trials=[episode["trial"] for episode in episodes if episode["any_cup_contact"]],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
