#!/usr/bin/env python3
"""Cache PickXTimes proprio aligned to the existing fixed 12-frame chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE = _ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_OUTPUT = _ROOT / "artifacts/pickxtimes_fixed_chunk_proprio_v1_260828"
SPLITS = ("train", "dev", "test")
PICK_TASK = teacher_lib.TASKS.index("pickxtimes_local_event")
FIELDS = (
    "obs/gripper_state[0]",
    "obs/gripper_state[1]",
    "obs/is_gripper_close",
    "action/eef_action[6]",
    "obs/eef_state_raw/pose[2]",
    "action/eef_action_raw/pose[2]",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _frame_proprio(timestep: h5py.Group) -> np.ndarray:
    gripper = np.asarray(timestep["obs/gripper_state"][()], dtype=np.float32)
    return np.asarray(
        (
            gripper[0],
            gripper[1],
            timestep["obs/is_gripper_close"][()],
            np.asarray(timestep["action/eef_action"][()])[6],
            np.asarray(timestep["obs/eef_state_raw/pose"][()])[2],
            np.asarray(timestep["action/eef_action_raw/pose"][()])[2],
        ),
        dtype=np.float32,
    )


def build_split(args: argparse.Namespace, split: str) -> tuple[dict[str, int], list[np.ndarray]]:
    sequence = _load_npz(args.sequence_dir / f"{split}.npz")
    rows = _read_jsonl(args.sequence_dir / f"{split}.jsonl")
    pick_rows = np.flatnonzero(sequence["task_ids"] == PICK_TASK)
    output = args.output_dir / f"{split}.h5"
    all_values: list[np.ndarray] = []
    handles: dict[str, h5py.File] = {}
    with h5py.File(output, "w") as target:
        target.attrs.update(
            schema_version=1,
            chunk_frames=12,
            stride_frames=12,
            proprio_dim=len(FIELDS),
        )
        try:
            for row_index in pick_rows:
                row = rows[int(row_index)]
                path = str(row["h5_path"])
                if path not in handles:
                    handles[path] = h5py.File(path, "r")
                episode = handles[path][row["episode_name"]]
                count = int(sequence["step_mask"][row_index].sum())
                indices = sequence["frame_indices"][row_index, :count]
                values = np.stack(
                    [
                        np.stack(
                            [
                                _frame_proprio(episode[f"timestep_{int(frame)}"])
                                for frame in chunk
                            ]
                        )
                        for chunk in indices
                    ]
                ).astype(np.float32)
                group = target.create_group(f"episode_{int(row_index):06d}")
                group.create_dataset("proprio", data=values, compression="lzf")
                group.attrs.update(
                    complete=True,
                    episode_index=int(sequence["episode_index"][row_index]),
                    num_chunks=count,
                )
                all_values.append(values)
        finally:
            for handle in handles.values():
                handle.close()
    return {
        "episodes": int(len(pick_rows)),
        "chunks": int(sum(len(values) for values in all_values)),
    }, all_values


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / f"{split}.h5" for split in SPLITS]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Output exists in {args.output_dir}; pass --overwrite")
    summary: dict[str, object] = {
        "schema_version": 1,
        "fields": FIELDS,
        "chunk_frames": 12,
        "stride_frames": 12,
        "splits": {},
    }
    train_values = None
    for split in SPLITS:
        split_summary, values = build_split(args, split)
        summary["splits"][split] = split_summary
        if split == "train":
            train_values = np.concatenate(values, axis=0).astype(np.float64)
    assert train_values is not None
    mean = train_values.mean(axis=(0, 1))
    std = np.maximum(train_values.std(axis=(0, 1)), 1e-6)
    summary["normalization"] = {"mean": mean.tolist(), "std": std.tolist()}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
