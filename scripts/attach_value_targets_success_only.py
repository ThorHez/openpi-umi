"""Attach Pi0-style normalized value targets to an existing LeRobot dataset (success-only).

This script is for datasets where every episode is considered SUCCESS, and you want to
populate the `value_target` column exactly like in OpenPI training utilities:

  remaining_steps = ep_length - frame_index - 1
  g = -remaining_steps
  g_norm = g / (task_max + task_max * c_fail_coef)
  value_target = clip(g_norm, [clip_min, clip_max])

It writes `value_target` back into each episode parquet file in-place, and updates
`meta/info.json` to declare the feature.

Usage:
  python scripts/attach_value_targets_success_only.py \
    --dataset-root /path/to/lerobot_dataset \
    --c-fail-coef 1.0 --clip-min -1.0 --clip-max 0.0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from openpi.training import value_targets

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _load_tasks(dataset_root: Path) -> dict[str, int]:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return {}
    task_to_index: dict[str, int] = {}
    with open(tasks_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            task_to_index[str(entry["task"])] = int(entry["task_index"])
    return task_to_index


def _load_episodes(dataset_root: Path) -> list[dict]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes metadata: {episodes_path}")
    episodes: list[dict] = []
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
    return episodes


def _update_info_json(dataset_root: Path) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        logger.warning("info.json not found at %s; skipping feature metadata update.", info_path)
        return
    with open(info_path) as f:
        info = json.load(f)
    info.setdefault("features", {})
    info["features"]["value_target"] = {"dtype": "float32", "shape": [1], "names": None}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Updated %s with value_target feature.", info_path)


def attach_value_targets(
    dataset_root: Path,
    *,
    c_fail_coef: float,
    clip_min: float,
    clip_max: float,
) -> None:
    dataset_root = dataset_root.resolve()
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {data_dir}")

    tasks = _load_tasks(dataset_root)
    episodes = _load_episodes(dataset_root)

    # Build episode_info and task_max_lengths from meta (success-only).
    episode_info: dict[int, value_targets.EpisodeTargetInfo] = {}
    task_max_lengths: dict[int, int] = {}
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        ep_len = int(ep["length"])
        task_names = ep.get("tasks", [])
        task_name = task_names[0] if isinstance(task_names, list) and task_names else "unknown"
        task_index = int(tasks.get(task_name, 0))
        episode_info[ep_idx] = value_targets.EpisodeTargetInfo(
            task_index=task_index,
            length=ep_len,
            success=True,
        )
        task_max_lengths[task_index] = max(task_max_lengths.get(task_index, 0), ep_len)

    logger.info(
        "Loaded meta: %d episodes, %d tasks. Will write %d parquet files.",
        len(episode_info),
        len(task_max_lengths),
        len(parquet_files),
    )

    # Write value_target per episode parquet.
    for pq_path in tqdm(parquet_files, desc="Writing value_target", unit="episode"):
        table = pq.read_table(pq_path)
        if "episode_index" not in table.column_names or "frame_index" not in table.column_names:
            raise ValueError(f"Missing episode_index/frame_index columns in {pq_path}")

        # Episode parquet should contain exactly one episode_index.
        ep_ids = np.asarray(table["episode_index"].to_numpy()).astype(np.int64, copy=False).reshape(-1)
        unique_eps = np.unique(ep_ids)
        if unique_eps.size != 1:
            raise ValueError(f"Expected one episode per parquet, got {unique_eps.tolist()} in {pq_path}")
        ep_idx = int(unique_eps[0])
        if ep_idx not in episode_info:
            raise KeyError(f"Episode {ep_idx} not present in meta/episodes.jsonl (file {pq_path})")

        frame_idx = np.asarray(table["frame_index"].to_numpy()).astype(np.int64, copy=False).reshape(-1)
        ep_len_meta = int(episode_info[ep_idx].length)
        ep_len_infer = int(frame_idx.max()) + 1 if frame_idx.size else 0
        if ep_len_infer and ep_len_meta != ep_len_infer:
            # Not fatal; prefer parquet reality so indices align.
            logger.warning(
                "Episode %d length mismatch: meta=%d parquet=%d (using parquet length for targets). File=%s",
                ep_idx,
                ep_len_meta,
                ep_len_infer,
                pq_path.name,
            )
            episode_info_ep = value_targets.EpisodeTargetInfo(
                task_index=episode_info[ep_idx].task_index,
                length=ep_len_infer,
                success=True,
            )
        else:
            episode_info_ep = episode_info[ep_idx]

        # Compute targets aligned to frame_index (which should be 0..len-1).
        episode_indices = np.full((frame_idx.shape[0],), ep_idx, dtype=np.int64)
        targets = value_targets.compute_normalized_value_targets(
            episode_indices=episode_indices,
            frame_indices=frame_idx,
            episode_info={ep_idx: episode_info_ep},
            task_max_lengths=task_max_lengths,
            c_fail_coef=c_fail_coef,
            clip_min=clip_min,
            clip_max=clip_max,
        ).astype(np.float32)

        # Store as float32 with LeRobot convention shape (1,) in meta; parquet stores scalar float32.
        new_col = pa.array(targets, type=pa.float32())
        if "value_target" in table.column_names:
            col_idx = table.schema.names.index("value_target")
            table = table.set_column(col_idx, "value_target", new_col)
        else:
            table = table.append_column("value_target", new_col)
        pq.write_table(table, pq_path, compression="snappy")

    _update_info_json(dataset_root)
    logger.info("Done. Attached value_target to %s", dataset_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach value_target to success-only LeRobot dataset")
    p.add_argument("--dataset-root", type=str, required=True, help="LeRobot dataset root")
    p.add_argument("--c-fail-coef", type=float, default=1.0, help="Failure penalty coef (unused when success-only, kept for formula parity)")
    p.add_argument("--clip-min", type=float, default=-1.0, help="Lower clip bound")
    p.add_argument("--clip-max", type=float, default=0.0, help="Upper clip bound")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    attach_value_targets(
        Path(args.dataset_root),
        c_fail_coef=args.c_fail_coef,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
