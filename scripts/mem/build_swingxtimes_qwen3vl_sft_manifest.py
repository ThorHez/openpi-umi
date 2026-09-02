#!/usr/bin/env python3
"""Build causal sparse-prefix Qwen3-VL SFT manifests for RoboMME SwingXtimes."""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme.swingxtimes.qwen3vl_sft_contract import compact_response  # noqa: E402

DEFAULT_H5 = _ROOT / "data/robomme_extracted/record_dataset_SwingXtimes.h5"
DEFAULT_OUTPUT = _ROOT / "artifacts/swingxtimes_qwen3vl_sft_seed260825"
_ARRIVAL_RE = re.compile(
    r"move to the top of the (right|left)-side target for the (first|second|third) time"
)
_COLOR_RE = re.compile(r"pick up the (red|green|blue) cube")
_ORDINAL = {"first": 1, "second": 2, "third": 3}
FRAME_COUNT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--prefix-variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260825)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _decode(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return value.decode() if isinstance(value, bytes | np.bytes_) else str(value)


def _timesteps(episode: h5py.Group) -> list[str]:
    return sorted(
        (name for name in episode if name.startswith("timestep_")),
        key=lambda name: int(name.split("_")[-1]),
    )


def _episode_metadata(episode: h5py.Group, episode_index: int) -> dict[str, Any]:
    timesteps = _timesteps(episode)
    transitions = []
    previous = None
    for frame_index, name in enumerate(timesteps):
        subgoal = _decode(episode[f"{name}/info/simple_subgoal"][()])
        if subgoal != previous:
            transitions.append((frame_index, subgoal))
            previous = subgoal
    starts = []
    for transition_index, (frame_index, subgoal) in enumerate(transitions):
        match = _ARRIVAL_RE.fullmatch(subgoal)
        if match is not None:
            starts.append((transition_index, frame_index, match.group(1), _ORDINAL[match.group(2)]))
    events = []
    for transition_index, start, side, ordinal in starts:
        completion = transitions[transition_index + 1][0]
        events.append({"start": start, "completion": completion, "side": side, "ordinal": ordinal})
    target_round_trips = sum(event["side"] == "right" for event in events)
    expected = [
        (side, ordinal)
        for ordinal in range(1, target_round_trips + 1)
        for side in ("right", "left")
    ]
    observed = [(str(event["side"]), int(event["ordinal"])) for event in events]
    if observed != expected:
        raise ValueError(f"episode {episode_index}: invalid arrival sequence {observed}")
    goal = _decode(episode["setup/task_goal"][()][0])
    color_match = _COLOR_RE.search(goal)
    if color_match is None:
        raise ValueError(f"episode {episode_index}: cannot parse target color from {goal!r}")
    if any(bool(episode[f"{name}/info/is_video_demo"][()]) for name in timesteps):
        raise ValueError(f"episode {episode_index}: unexpectedly contains demonstration frames")
    return {
        "episode_index": episode_index,
        "episode_name": f"episode_{episode_index}",
        "num_frames": len(timesteps),
        "difficulty": _decode(episode["setup/difficulty"][()]),
        "target_color": color_match.group(1),
        "target_round_trips": target_round_trips,
        "goal": goal,
        "events": events,
    }


def _stratified_split(metadata: list[dict[str, Any]], val_count: int, seed: int) -> tuple[set[int], set[int]]:
    rng = random.Random(seed)
    strata: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for item in metadata:
        strata[(item["target_round_trips"], item["difficulty"], item["target_color"])].append(
            item["episode_index"]
        )
    for values in strata.values():
        rng.shuffle(values)
    exact = {key: len(values) * val_count / len(metadata) for key, values in strata.items()}
    allocation = {key: int(np.floor(value)) for key, value in exact.items()}
    remaining = val_count - sum(allocation.values())
    order = sorted(strata, key=lambda key: exact[key] - allocation[key], reverse=True)
    for key in order[:remaining]:
        allocation[key] += 1
    val = {episode for key, values in strata.items() for episode in values[: allocation[key]]}
    train = {item["episode_index"] for item in metadata} - val
    return train, val


def _prefix_frames(item: dict[str, Any], event_count: int, variant: int) -> list[int]:
    events = item["events"][:event_count]
    offsets = ((-8, -2), (-6, -1), (-4, 0))[variant % 3]
    evidence = [
        max(int(event["start"]), int(event["completion"]) + offset)
        for event in events
        for offset in offsets
    ]
    filler_count = FRAME_COUNT - len(evidence)
    first_event_start = int(item["events"][0]["start"])
    filler = (
        np.linspace(0, max(0, first_event_start - 1), filler_count).round().astype(int).tolist()
        if filler_count
        else []
    )
    frames = sorted(filler + evidence)
    if len(frames) != FRAME_COUNT or max(frames) > int(events[-1]["completion"]):
        raise AssertionError(f"Invalid causal prefix frames: {frames}")
    return frames


def _rows(item: dict[str, Any], h5_path: Path, variants: int) -> list[dict[str, Any]]:
    common = {
        "schema_version": 1,
        "source": "swingxtimes",
        "episode_index": item["episode_index"],
        "episode_name": item["episode_name"],
        "h5_path": str(h5_path.resolve()),
        "difficulty": item["difficulty"],
        "target_color": item["target_color"],
        "target_round_trips": item["target_round_trips"],
    }
    rows = []
    for event_index, event in enumerate(item["events"], start=1):
        right_count = (event_index + 1) // 2
        left_count = event_index // 2
        event_name = f'{event["side"]}_arrival'
        target = compact_response(
            "causal_prefix",
            event=event_name,
            right_count=right_count,
            left_count=left_count,
            target_round_trips=item["target_round_trips"],
        )
        rows.extend(

                {
                    **common,
                    "sample_type": "causal_prefix",
                    "event_index": event_index,
                    "variant": variant,
                    "frame_indices": _prefix_frames(item, event_index, variant),
                    "target": target,
                }
                for variant in range(variants)

        )
        if event_index >= 2:
            start, completion = int(event["start"]), int(event["completion"])
            local_frames = np.linspace(start, completion, FRAME_COUNT).round().astype(int).tolist()
            rows.append(
                {
                    **common,
                    "sample_type": "local_only",
                    "event_index": event_index,
                    "variant": 0,
                    "frame_indices": local_frames,
                    "target": compact_response("local_only"),
                }
            )
    first_start = int(item["events"][0]["start"])
    for variant in range(2):
        stop = max(11, first_start - 1 - variant * 8)
        frames = np.linspace(0, stop, FRAME_COUNT).round().astype(int).tolist()
        rows.append(
            {
                **common,
                "sample_type": "no_event",
                "event_index": 0,
                "variant": variant,
                "frame_indices": frames,
                "target": compact_response("no_event"),
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "episodes": len({row["episode_index"] for row in rows}),
        "sample_types": dict(sorted(Counter(row["sample_type"] for row in rows).items())),
        "target_round_trips": dict(
            sorted(Counter(row["target_round_trips"] for row in rows if row["variant"] == 0).items())
        ),
    }


def main() -> None:
    args = parse_args()
    with h5py.File(args.h5, "r") as source:
        metadata = [
            _episode_metadata(source[f"episode_{episode_index}"], episode_index)
            for episode_index in range(100)
        ]
    train_ids, val_ids = _stratified_split(metadata, args.val_episodes, args.seed)
    train_rows, val_rows = [], []
    for item in metadata:
        destination = train_rows if item["episode_index"] in train_ids else val_rows
        destination.extend(_rows(item, args.h5, args.prefix_variants))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = args.output_dir / "train.jsonl", args.output_dir / "val.jsonl"
    if (train_path.exists() or val_path.exists()) and not args.overwrite:
        raise FileExistsError("Manifest exists; pass --overwrite")
    _write(train_path, train_rows)
    _write(val_path, val_rows)
    summary = {
        "schema_version": 1,
        "h5": str(args.h5.resolve()),
        "seed": args.seed,
        "frame_count": FRAME_COUNT,
        "execution_prefix_only": True,
        "future_frame_leakage": False,
        "train_episode_ids": sorted(train_ids),
        "val_episode_ids": sorted(val_ids),
        "train": _counts(train_rows),
        "val": _counts(val_rows),
        "copied_video_bytes": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
