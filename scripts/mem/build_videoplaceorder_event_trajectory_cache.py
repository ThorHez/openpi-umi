#!/usr/bin/env python3
"""Build train/dev VideoPlaceOrder event trajectories and proprio chunks.

The event/payload sequence is a training-only privileged teacher derived from
the replay metadata.  The student inputs contain only deployable RGB features,
joint state, gripper state, goal ordinal, and episode-local visual anchors.
The test split is deliberately not materialized by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.mem import train_robomme_anchor_conditioned_decomposition as anchor_base  # noqa: E402
from scripts.mem import train_robomme_decomposed_region_distillation as base  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts/videoplaceorder_observable_event_trajectories_v1_260831"
DEFAULT_H5 = ROOT / "data/robomme_extracted/record_dataset_VideoPlaceOrder.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, default=base.DEFAULT_FIXED)
    parser.add_argument("--teacher-dir", type=Path, default=base.DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=base.DEFAULT_FEATURES)
    parser.add_argument("--anchor-dir", type=Path, default=anchor_base.DEFAULT_ANCHORS)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_split(split: str, args: argparse.Namespace) -> dict[str, object]:
    data = anchor_base.AnchorRegionDataset(split, args)
    try:
        rows = data.rows[data.fixed["task_ids"][data.rows] == 2]
        count = len(rows)
        max_steps = data.max_steps
        proprio = np.zeros((count, max_steps, 12, 8), dtype=np.float32)
        with h5py.File(args.h5, "r") as source:
            for output_row, row_value in enumerate(rows):
                row = int(row_value)
                episode_index = int(data.fixed["episode_index"][row])
                episode = source[f"episode_{episode_index}"]
                length = int(data.fixed["step_mask"][row].sum())
                frame_indices = data.fixed["frame_indices"][row, :length]
                for chunk, indices in enumerate(frame_indices):
                    for frame, frame_index in enumerate(indices):
                        obs = episode[f"timestep_{int(frame_index)}/obs"]
                        joint = np.asarray(obs["joint_state"][()], dtype=np.float32)
                        gripper = float(np.asarray(obs["gripper_state"][()]).reshape(-1)[0])
                        proprio[output_row, chunk, frame, :7] = joint
                        proprio[output_row, chunk, frame, 7] = gripper

        payload = {
            "source_rows": rows.astype(np.int32),
            "episode_index": data.fixed["episode_index"][rows].astype(np.int32),
            "patch_group_index": rows.astype(np.int32),
            "sequence_mask": data.fixed["step_mask"][rows],
            "task_ids": data.fixed["task_ids"][rows].astype(np.int32),
            "goal_color_ids": data.fixed["goal_color_ids"][rows].astype(np.int32),
            "queried_ordinals": data.fixed["queried_ordinals"][rows].astype(np.int32),
            "num_regions": data.fixed["num_regions"][rows].astype(np.int32),
            "anchor_yx": data.anchor_yx[rows].astype(np.float32),
            "anchor_mask": data.anchor_mask[rows],
            "proprio_tokens": proprio,
            "event_type": data.event_type[rows].astype(np.int32),
            "write_entity": data.write_entity[rows].astype(np.int32),
            "write_region": data.write_region[rows].astype(np.int32),
            "swap_pair": data.swap_pair[rows].astype(np.int32),
            "micro_mask": data.micro_mask[rows],
            "write_mask": data.write_mask[rows],
            "swap_mask": data.swap_mask[rows],
            "table_targets": data.table_targets[rows].astype(np.int32),
            "table_mask": data.table_mask[rows],
            "state_change_mask": data.fixed["state_change_mask"][rows],
        }
        np.savez_compressed(args.output_dir / f"{split}.npz", **payload)
        valid = payload["micro_mask"]
        event_type = payload["event_type"]
        counts = np.bincount(event_type[valid], minlength=3)
        return {
            "episodes": count,
            "chunks": int(payload["sequence_mask"].sum()),
            "event_counts": {"hold": int(counts[0]), "write": int(counts[1]), "swap": int(counts[2])},
            "ordinal_histogram": np.bincount(payload["queried_ordinals"], minlength=5).tolist(),
            "proprio": {
                "joint_dimensions": 7,
                "gripper_dimension": 1,
                "gripper_min": float(proprio[..., 7][payload["sequence_mask"]].min()),
                "gripper_max": float(proprio[..., 7][payload["sequence_mask"]].max()),
            },
        }
    finally:
        data.close()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {split: build_split(split, args) for split in ("train", "dev")}
    audit["contract"] = {
        "teacher": "training-only GT event type/payload trajectory",
        "student_inputs": ["RGB grid features", "joint_state", "gripper_state", "goal ordinal", "visual anchors"],
        "test_split_materialized": False,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
