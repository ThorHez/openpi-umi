#!/usr/bin/env python3
"""Audit variable-length temporal structure for the four-task RoboMME pilot."""

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

from openpi.tasks.robomme import four_task_temporal_contract as temporal_contract  # noqa: E402

_TASK_FILES = {
    "VideoUnmask": _ROOT / "data/robomme_extracted/record_dataset_VideoUnmask.h5",
    "VideoUnmaskSwap": _ROOT / "data/robomme_extracted/record_dataset_VideoUnmaskSwap.h5",
    "VideoPlaceOrder": _ROOT / "data/robomme_extracted/record_dataset_VideoPlaceOrder.h5",
    "PickXtimes": _ROOT / "data/robomme_extracted/record_dataset_PickXtimes.h5",
}
_COLORS = ("red", "green", "blue")
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "artifacts/robomme_four_task_pilot_seed260826",
    )
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _decode(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return value.decode() if isinstance(value, bytes | np.bytes_) else str(value)


def _step_names(episode: h5py.Group) -> list[str]:
    return sorted(
        (name for name in episode if name.startswith("timestep_")),
        key=lambda name: int(name.split("_")[-1]),
    )


def _goal_features(task: str, goal: str) -> dict[str, Any]:
    colors = [color for color in _COLORS if re.search(rf"\b{color}\b", goal)]
    result: dict[str, Any] = {"target_colors": colors}
    if task == "VideoPlaceOrder":
        result["ordinal"] = next(
            (value for word, value in _ORDINALS.items() if re.search(rf"\b{word}\b", goal)),
            None,
        )
    if task == "PickXtimes":
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        result["required_count"] = next(
            (value for word, value in words.items() if re.search(rf"\b{word}\b", goal)),
            1,
        )
    return result


def _episode_record(task: str, episode_index: int, episode: h5py.Group) -> dict[str, Any]:
    names = _step_names(episode)
    if not names:
        raise ValueError(f"{task}/episode_{episode_index}: empty episode")
    goal = _decode(episode["setup/task_goal"][()])
    difficulty = _decode(episode["setup/difficulty"][()])
    boundaries = []
    demo_flags = []
    for step_index, name in enumerate(names):
        info = episode[name]["info"]
        is_demo = bool(info["is_video_demo"][()])
        demo_flags.append(is_demo)
        if bool(info["is_subgoal_boundary"][()]):
            boundaries.append(
                {
                    "step": step_index,
                    "is_video_demo": is_demo,
                    "simple_subgoal": _decode(info["simple_subgoal"][()]),
                    "grounded_subgoal": _decode(info["grounded_subgoal"][()]),
                }
            )
    true_indices = [index for index, value in enumerate(demo_flags) if value]
    demo_frames = len(true_indices)
    demo_contiguous = not true_indices or true_indices == list(range(true_indices[-1] + 1))
    mode = "observe_then_act" if demo_frames else "observe_act_update"
    student_horizon = demo_frames if demo_frames else len(names)
    student_clips = temporal_contract.causal_student_clips(student_horizon)
    subgoal_counts = Counter(item["simple_subgoal"] for item in boundaries)
    record = {
        "episode_index": episode_index,
        "difficulty": difficulty,
        "goal": goal,
        **_goal_features(task, goal),
        "num_steps": len(names),
        "mode": mode,
        "demo_frames": demo_frames,
        "demo_contiguous_prefix": demo_contiguous,
        "demo_end": true_indices[-1] + 1 if true_indices else None,
        "student_clip_frames": temporal_contract.STUDENT_CLIP_FRAME_COUNT,
        "student_clip_stride": temporal_contract.STUDENT_CLIP_STRIDE,
        "student_clip_count": len(student_clips),
        "student_tail_valid_frames": sum(student_clips[-1].frame_mask) if student_clips else 0,
        "boundary_count": len(boundaries),
        "subgoal_counts": dict(sorted(subgoal_counts.items())),
        "boundaries": boundaries,
    }
    if task == "VideoUnmaskSwap":
        # The environment schedules a 64-frame reveal/cover prefix followed by
        # 50 frames per swap and a short settle tail.  This is an audit-derived
        # label only; models must not receive it as an input.
        estimate = int(round(max(0, demo_frames - 64) / 50))
        record["schedule_swap_count_estimate"] = estimate
    if task == "VideoPlaceOrder":
        record["num_demonstrated_targets"] = int(
            subgoal_counts.get("drop the cube onto target", 0)
        )
        record["has_demo_static_or_swap"] = bool(subgoal_counts.get("static", 0))
    return record


def _stratum(task: str, record: dict[str, Any]) -> tuple[Any, ...]:
    common: tuple[Any, ...] = (
        record["difficulty"],
        tuple(record["target_colors"]),
    )
    if task == "VideoUnmaskSwap":
        return common + (record["schedule_swap_count_estimate"],)
    if task == "VideoPlaceOrder":
        return common + (record["ordinal"], record["num_demonstrated_targets"])
    if task == "PickXtimes":
        return common + (record["required_count"],)
    return common


def _interleaved_stratified_order(
    task: str, records: list[dict[str, Any]], seed: int
) -> list[int]:
    rng = random.Random(seed)
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for record in records:
        groups[_stratum(task, record)].append(int(record["episode_index"]))
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups, key=repr)
    rng.shuffle(keys)
    result = []
    while keys:
        next_keys = []
        for key in keys:
            values = groups[key]
            if values:
                result.append(values.pop())
            if values:
                next_keys.append(key)
        rng.shuffle(next_keys)
        keys = next_keys
    return result


def _range_stats(values: list[int]) -> dict[str, float | int]:
    return {
        "min": int(min(values)),
        "mean": float(np.mean(values)),
        "max": int(max(values)),
    }


def _task_summary(task: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    modes = Counter(record["mode"] for record in records)
    summary: dict[str, Any] = {
        "episodes": len(records),
        "mode": dict(sorted(modes.items())),
        "num_steps": _range_stats([int(record["num_steps"]) for record in records]),
        "demo_frames": _range_stats([int(record["demo_frames"]) for record in records]),
        "all_demo_contiguous_prefix": all(record["demo_contiguous_prefix"] for record in records),
        "difficulty": dict(sorted(Counter(record["difficulty"] for record in records).items())),
        "target_colors": dict(
            sorted(Counter("+".join(record["target_colors"]) for record in records).items())
        ),
    }
    if task == "VideoUnmaskSwap":
        summary["schedule_swap_count_estimate"] = dict(
            sorted(Counter(str(record["schedule_swap_count_estimate"]) for record in records).items())
        )
    if task == "VideoPlaceOrder":
        summary["ordinal"] = dict(
            sorted(Counter(str(record["ordinal"]) for record in records).items())
        )
        summary["num_demonstrated_targets"] = dict(
            sorted(Counter(str(record["num_demonstrated_targets"]) for record in records).items())
        )
    if task == "PickXtimes":
        summary["required_count"] = dict(
            sorted(Counter(str(record["required_count"]) for record in records).items())
        )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "sequence_audit.json"
    split_output = args.output_dir / "episode_splits.json"
    if (output.exists() or split_output.exists()) and not args.overwrite:
        raise FileExistsError(f"Pilot outputs already exist in {args.output_dir}; pass --overwrite")

    all_records: dict[str, list[dict[str, Any]]] = {}
    summaries = {}
    splits = {"schema_version": 1, "seed": args.seed, "tasks": {}}
    for task, path in _TASK_FILES.items():
        with h5py.File(path, "r") as source:
            episode_ids = sorted(
                (int(name.split("_")[-1]) for name in source if name.startswith("episode_"))
            )
            records = [
                _episode_record(task, episode_index, source[f"episode_{episode_index}"])
                for episode_index in episode_ids
            ]
        if len(records) != 100:
            raise ValueError(f"{task}: expected 100 episodes, found {len(records)}")
        all_records[task] = records
        summaries[task] = _task_summary(task, records)
        order = _interleaved_stratified_order(task, records, args.seed)
        splits["tasks"][task] = {
            "train_episode_indices": sorted(order[:70]),
            "dev_episode_indices": sorted(order[70:85]),
            "test_episode_indices": sorted(order[85:]),
        }

    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "fixed_episode_length_assumption": False,
        "temporal_contract": {
            "teacher_frame_count": temporal_contract.TEACHER_FRAME_COUNT,
            "student_clip_frame_count": temporal_contract.STUDENT_CLIP_FRAME_COUNT,
            "student_clip_stride": temporal_contract.STUDENT_CLIP_STRIDE,
        },
        "tasks": summaries,
        "episodes": all_records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    split_output.write_text(json.dumps(splits, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
