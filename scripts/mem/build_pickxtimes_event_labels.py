#!/usr/bin/env python3
"""Build causal sliding-window labels for RoboMME PickXtimes.

The script reads only scalar metadata and robot state from the raw HDF5 file;
RGB/depth arrays are never materialized.  Subgoal boundaries provide coarse
macro-event intervals, while gripper open/close edges refine PICK/PLACE event
anchors.  The resulting JSON is the task-owned supervision contract for the
generic event-memory modules.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import itertools
import json
import pathlib
import re
from typing import Any

import h5py
import numpy as np

EVENT_TYPES = ("pick_complete", "place_complete", "press_complete")
CHOICE_TO_EVENT = {"A": EVENT_TYPES[0], "B": EVENT_TYPES[1], "C": EVENT_TYPES[2]}
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
COLORS = ("red", "green", "blue", "yellow", "orange", "purple")


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return str(value)


def _numeric_suffix(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def _parse_choice(value: Any) -> str:
    text = _decode(value).strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text.upper()
    return str(parsed.get("choice", "")).upper()


def _parse_goal(prompt: str) -> tuple[str, int]:
    lowered = prompt.lower()
    color_matches = [color for color in COLORS if re.search(rf"\b{color}\b", lowered)]
    count_matches = [value for word, value in NUMBER_WORDS.items() if re.search(rf"\b{word}\b", lowered)]
    count_matches.extend(int(value) for value in re.findall(r"\b(\d+)\s+times?\b", lowered))
    if not count_matches and "pick up" in lowered and "place it" in lowered and "then press" in lowered:
        # RoboMME omits an explicit "one time" phrase for the single-repeat
        # variant: "pick ... place ..., then press the button to stop".
        count_matches.append(1)
    if len(color_matches) != 1 or len(set(count_matches)) != 1:
        raise ValueError(
            f"Expected exactly one color and repetition count in prompt, got "
            f"colors={color_matches}, counts={count_matches}: {prompt!r}"
        )
    return color_matches[0], count_matches[0]


def _mode_choice(choices: list[str], *, episode: str, start: int, end: int) -> str:
    counts = collections.Counter(choice for choice in choices[start:end] if choice)
    if not counts:
        raise ValueError(f"No choice_action in {episode} interval [{start},{end})")
    [(choice, count), *rest] = counts.most_common()
    if rest and rest[0][1] == count:
        raise ValueError(f"Ambiguous choice_action in {episode} interval [{start},{end}): {counts}")
    if choice not in CHOICE_TO_EVENT:
        raise ValueError(f"Unknown choice_action={choice!r} in {episode} interval [{start},{end})")
    return choice


def _transition_edges(closed: np.ndarray, start: int, end: int, *, rising: bool) -> list[int]:
    edges = []
    for index in range(max(start, 1), end):
        before, after = bool(closed[index - 1]), bool(closed[index])
        if (not before and after) if rising else (before and not after):
            edges.append(index)
    return edges


def _positive_starts(
    *,
    anchor: int,
    segment_start: int,
    segment_end: int,
    num_steps: int,
    window_size: int,
    post_frames: int,
    jitter_radius: int,
    allow_after_segment: bool = False,
) -> list[int]:
    max_start = num_steps - window_size
    desired_anchor_position = window_size - post_frames - 1
    canonical = anchor - desired_anchor_position
    starts = []
    for offset in range(-jitter_radius, jitter_radius + 1):
        candidate = canonical + offset
        # PICK/PLACE windows stay inside their macro-action segment so that
        # frames from the next action cannot leak into the event classifier.
        # PRESS is the final event and its anchor is the segment's last frame;
        # forcing the whole window inside the segment collapses all temporal
        # jitter to one start.  It is safe (and useful) for PRESS to include
        # the episode's post-completion frames.
        upper_bound = max_start if allow_after_segment else segment_end - window_size
        candidate = min(max(candidate, segment_start), upper_bound)
        candidate = min(max(candidate, 0), max_start)
        if candidate <= anchor < candidate + window_size:
            starts.append(candidate)
    return sorted(set(starts))


def _window_pools(
    *,
    num_steps: int,
    window_size: int,
    events: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    positive = {start for event in events for start in event["positive_starts"]}
    anchors = [int(event["anchor"]) for event in events]
    boundaries = [int(event["end"]) for event in events]
    hard, ordinary = [], []
    for start in range(num_steps - window_size + 1):
        if start in positive:
            continue
        end = start + window_size
        near_transition = any(start <= anchor < end for anchor in anchors) or any(
            abs(start - positive_start) <= window_size // 2 for positive_start in positive
        )
        near_boundary = any(abs(end - boundary) <= window_size // 2 for boundary in boundaries)
        if near_transition or near_boundary:
            hard.append(start)
        elif start % 3 == 0:
            ordinary.append(start)
    return hard, ordinary


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    window_size: int = 10
    post_frames: int = 3
    jitter_radius: int = 2


def build_labels(h5_path: pathlib.Path, config: BuildConfig) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as handle:
        episode_names = sorted(
            (name for name in handle if name.startswith("episode_")),
            key=_numeric_suffix,
        )
        for episode_name in episode_names:
            episode = handle[episode_name]
            step_names = sorted(
                (name for name in episode if name.startswith("timestep_")),
                key=_numeric_suffix,
            )
            step_ids = [_numeric_suffix(name) for name in step_names]
            if step_ids != list(range(len(step_names))):
                raise ValueError(f"{episode_name} timesteps are not dense from zero")

            prompts = [_decode(value).strip() for value in episode["setup/task_goal"][()].reshape(-1)]
            parsed_goals = {_parse_goal(prompt) for prompt in prompts}
            if len(parsed_goals) != 1:
                raise ValueError(f"Prompt paraphrases disagree in {episode_name}: {parsed_goals}")
            target_color, required_count = parsed_goals.pop()

            choices = []
            closed = []
            boundary_flags = []
            completed = []
            for step_name in step_names:
                timestep = episode[step_name]
                choices.append(_parse_choice(timestep["action/choice_action"][()]))
                closed.append(bool(timestep["obs/is_gripper_close"][()]))
                boundary_flags.append(bool(timestep["info/is_subgoal_boundary"][()]))
                completed.append(bool(timestep["info/is_completed"][()]))
            closed_array = np.asarray(closed, dtype=np.bool_)
            boundaries = np.flatnonzero(boundary_flags).astype(int).tolist()
            completion_rises = [
                index for index, value in enumerate(completed) if value and (index == 0 or not completed[index - 1])
            ]
            completion_rise = completion_rises[0] if completion_rises else len(step_names) - 1
            if not boundaries or boundaries[0] != 0:
                boundaries.insert(0, 0)
            if boundaries[-1] < completion_rise:
                boundaries.append(completion_rise)
            boundaries = sorted(set(boundaries))

            events = []
            count = 0
            holding = False
            done = False
            for start, end in itertools.pairwise(boundaries):
                if end - start < config.window_size:
                    continue
                choice = _mode_choice(choices, episode=episode_name, start=start, end=end)
                event_type = CHOICE_TO_EVENT[choice]
                if event_type == "pick_complete":
                    edges = _transition_edges(closed_array, start, end, rising=True)
                    holding = True
                elif event_type == "place_complete":
                    edges = _transition_edges(closed_array, start, end, rising=False)
                    count += 1
                    holding = False
                else:
                    edges = []
                    done = True

                if edges:
                    anchor = edges[-1]
                    anchor_source = "gripper_edge"
                    anchor_quality = "strong"
                else:
                    anchor = end - 1
                    anchor_source = "segment_end"
                    anchor_quality = "weak" if event_type != "press_complete" else "boundary"
                positive_starts = _positive_starts(
                    anchor=anchor,
                    segment_start=start,
                    segment_end=end,
                    num_steps=len(step_names),
                    window_size=config.window_size,
                    post_frames=config.post_frames,
                    jitter_radius=config.jitter_radius,
                    allow_after_segment=event_type == "press_complete",
                )
                events.append(
                    {
                        "start": start,
                        "end": end,
                        "choice": choice,
                        "event_type": event_type,
                        "event_type_id": EVENT_TYPES.index(event_type),
                        "anchor": anchor,
                        "anchor_source": anchor_source,
                        "anchor_quality": anchor_quality,
                        "positive_starts": positive_starts,
                        "state_after": {
                            "completed_count": count,
                            "holding": holding,
                            "remaining_count": max(required_count - count, 0),
                            "should_press": count == required_count and not holding and not done,
                            "done": done,
                        },
                    }
                )

            hard_negative_starts, ordinary_negative_starts = _window_pools(
                num_steps=len(step_names),
                window_size=config.window_size,
                events=events,
            )
            episodes.append(
                {
                    "episode_index": _numeric_suffix(episode_name),
                    "episode_name": episode_name,
                    "seed": int(episode["setup/seed"][()]),
                    "difficulty": _decode(episode["setup/difficulty"][()]),
                    "prompts": prompts,
                    "target_color": target_color,
                    "required_count": required_count,
                    "num_steps": len(step_names),
                    # Online-observable proprioception cached beside labels
                    # to avoid repeatedly walking scalar HDF5 groups.
                    "gripper_closed": closed,
                    "completion_rise": completion_rise,
                    "boundaries": boundaries,
                    "events": events,
                    "hard_negative_starts": hard_negative_starts,
                    "ordinary_negative_starts": ordinary_negative_starts,
                }
            )

    event_counts = collections.Counter(event["event_type"] for episode in episodes for event in episode["events"])
    anchor_counts = collections.Counter(
        (event["event_type"], event["anchor_quality"]) for episode in episodes for event in episode["events"]
    )
    count_distribution = collections.Counter(episode["required_count"] for episode in episodes)
    color_distribution = collections.Counter(episode["target_color"] for episode in episodes)
    event_lengths = [event["end"] - event["start"] for episode in episodes for event in episode["events"]]

    def distribution(values: list[int]) -> dict[str, float | int]:
        return {
            "min": min(values),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "max": max(values),
        }

    return {
        "schema_version": 3,
        "source_h5": str(h5_path.resolve()),
        "window_size": config.window_size,
        "post_frames": config.post_frames,
        "jitter_radius": config.jitter_radius,
        "event_types": list(EVENT_TYPES),
        "summary": {
            "num_episodes": len(episodes),
            "num_events": sum(event_counts.values()),
            "event_counts": dict(event_counts),
            "anchor_quality_counts": {f"{key[0]}:{key[1]}": value for key, value in anchor_counts.items()},
            "required_count_distribution": {str(key): value for key, value in sorted(count_distribution.items())},
            "target_color_distribution": dict(color_distribution),
            "max_events_per_episode": max(len(episode["events"]) for episode in episodes),
            "event_length_frames": distribution(event_lengths),
            "anchor_to_segment_end_frames": {
                event_type: distribution(
                    [
                        event["end"] - 1 - event["anchor"]
                        for episode in episodes
                        for event in episode["events"]
                        if event["event_type"] == event_type
                    ]
                )
                for event_type in EVENT_TYPES
            },
        },
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--post-frames", type=int, default=3)
    parser.add_argument("--jitter-radius", type=int, default=2)
    args = parser.parse_args()
    config = BuildConfig(
        window_size=args.window_size,
        post_frames=args.post_frames,
        jitter_radius=args.jitter_radius,
    )
    labels = build_labels(args.h5_path, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(labels["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
