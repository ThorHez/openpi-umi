"""Value target computation for Pi0 value training.

Implements the normalized value target formula used by the reference Pi0.6
value function: g_norm = (-remaining_steps - c_fail*I(fail)) / (task_max + c_fail),
clipped to [clip_min, clip_max].
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np


@dataclasses.dataclass(frozen=True)
class EpisodeTargetInfo:
    """Per-episode metadata required to compute value targets."""

    task_index: int
    length: int
    success: bool


def load_episode_target_info_from_lerobot_meta(
    dataset_root: str | pathlib.Path,
    *,
    success_field: str = "success",
    default_success_when_missing: bool = True,
) -> tuple[dict[int, EpisodeTargetInfo], dict[int, int]]:
    """Build episode_info and task_max_lengths from LeRobot ``meta/episodes.jsonl`` (and optional ``tasks.jsonl``).

    Matches the convention used in ``scripts/lerobot_value_infer.py`` so local UMI / HITL datasets work without a
    separate ``episode_metadata.json`` when each episode row has ``episode_index`` and ``length``.
    """
    root = pathlib.Path(dataset_root).expanduser().resolve()
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"LeRobot episodes metadata not found: {episodes_path}")

    episodes: list[dict] = []
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    task_name_to_index: dict[str, int] = {}
    tasks_path = root / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        with open(tasks_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    task_name_to_index[entry["task"]] = int(entry["task_index"])

    episode_info: dict[int, EpisodeTargetInfo] = {}
    task_max_length: dict[int, int] = {}

    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        ep_length = int(ep["length"])

        if "task_index" in ep:
            task_index = int(ep["task_index"])
        else:
            tasks = ep.get("tasks", [])
            task_name = tasks[0] if isinstance(tasks, list) and tasks else "unknown"
            task_index = task_name_to_index.get(task_name, 0)

        explicit_success = ep.get(success_field)
        if explicit_success is not None:
            ep_success = bool(explicit_success)
        else:
            ep_success = default_success_when_missing

        episode_info[ep_idx] = EpisodeTargetInfo(
            task_index=task_index,
            length=ep_length,
            success=ep_success,
        )
        task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)

    return episode_info, task_max_length


def compute_normalized_value_targets(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    episode_info: dict[int, EpisodeTargetInfo],
    task_max_lengths: dict[int, int],
    c_fail_coef: float,
    *,
    clip_min: float = -1.0,
    clip_max: float = 0.0,
) -> np.ndarray:
    """Compute normalized scalar value targets for each (episode_index, frame_index) pair.

    Formula:
      remaining_steps = ep.length - frame_index - 1
      g = -remaining_steps
      if not ep.success:
          g -= task_max * c_fail_coef
      g_norm = g / (task_max + c_fail)
      target = clip(g_norm, clip_min, clip_max)

    Args:
        episode_indices: Shape (n,). Episode index per sample.
        frame_indices: Shape (n,). Frame index within episode per sample.
        episode_info: Mapping episode_index -> EpisodeTargetInfo (task_index, length, success).
        task_max_lengths: Mapping task_index -> max episode length for normalization.
        c_fail_coef: Coefficient for failure penalty; c_fail = task_max * c_fail_coef.
        clip_min: Lower bound for normalized target (default -1.0).
        clip_max: Upper bound for normalized target (default 0.0).

    Returns:
        Float32 array of shape (n,) with normalized value targets in [clip_min, clip_max].
    """
    if episode_indices.shape != frame_indices.shape:
        raise ValueError("episode_indices and frame_indices must have the same shape.")
    if c_fail_coef < 0:
        raise ValueError("'c_fail_coef' must be non-negative.")

    targets = np.zeros(episode_indices.shape[0], dtype=np.float32)
    for i in range(episode_indices.shape[0]):
        ep_idx = int(episode_indices[i])
        if ep_idx not in episode_info:
            raise KeyError(f"Missing episode metadata for episode_index={ep_idx}.")
        ep = episode_info[ep_idx]
        task_max = task_max_lengths.get(ep.task_index)
        if task_max is None:
            raise KeyError(f"Missing task max length for task_index={ep.task_index}.")
        if task_max <= 0:
            raise ValueError(
                f"Invalid task max length {task_max} for task_index={ep.task_index}."
            )

        remaining_steps = ep.length - int(frame_indices[i]) - 1
        c_fail = float(task_max) * c_fail_coef
        g = -float(remaining_steps)
        if not ep.success:
            g -= c_fail

        denom = float(task_max) + c_fail
        g_norm = g / denom
        targets[i] = np.clip(g_norm, clip_min, clip_max)

    return targets
