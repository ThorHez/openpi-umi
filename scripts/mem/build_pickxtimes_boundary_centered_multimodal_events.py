#!/usr/bin/env python3
"""Build native-boundary-centered PickXTimes RGB/proprio event examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VISUAL = _ROOT / "artifacts/robomme_four_task_visual_student_sequences_v1_260826"
DEFAULT_FEATURES = _ROOT / "artifacts/robomme_four_task_fixed_chunk_features_4x4_v1_260826"
DEFAULT_TEACHER = _ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
DEFAULT_OUTPUT = _ROOT / "artifacts/pickxtimes_boundary_centered_multimodal_events_v1_260827"
SPLITS = ("train", "dev", "test")
PICK_TASK = teacher_lib.TASKS.index("pickxtimes_local_event")
PICK_EVENT_IDS = np.asarray(
    [
        teacher_lib.EVENTS.index("pick_complete"),
        teacher_lib.EVENTS.index("place_complete"),
        teacher_lib.EVENTS.index("press_complete"),
    ],
    dtype=np.int32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pre-frames", type=int, default=6)
    parser.add_argument("--post-frames", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _native_boundaries(episode: h5py.Group) -> tuple[list[int], int]:
    frame_ids = sorted(
        int(key.rsplit("_", 1)[1]) for key in episode if key.startswith("timestep_")
    )
    boundaries = [
        index
        for index in frame_ids
        if bool(episode[f"timestep_{index}/info/is_subgoal_boundary"][()])
    ]
    return boundaries[1:], frame_ids[-1] + 1


def build_split(args: argparse.Namespace, split: str) -> dict[str, object]:
    rows = _read_jsonl(args.visual_dir / f"{split}.jsonl")
    teacher = _load_npz(args.teacher_dir / f"{split}.npz")
    pick_rows = np.flatnonzero(teacher["task_id"] == PICK_TASK)
    frame_offsets = np.arange(-args.pre_frames, args.post_frames, dtype=np.int32)
    if len(frame_offsets) != 12:
        raise ValueError("The frozen RGB encoder requires exactly 12 centered frames")

    episode_offsets = [0]
    episode_indices: list[int] = []
    required_counts: list[int] = []
    event_targets: list[int] = []
    event_global_ids: list[int] = []
    boundary_frames: list[int] = []
    centered_frames: list[np.ndarray] = []
    pre_margins: list[int] = []
    post_margins: list[int] = []
    rgb: list[np.ndarray] = []
    gripper: list[np.ndarray] = []
    eef_z: list[np.ndarray] = []

    h5_handles: dict[str, h5py.File] = {}
    with h5py.File(args.feature_dir / f"{split}.h5", "r") as feature_file:
        try:
            for source_row in pick_rows:
                row = rows[int(source_row)]
                if not row["source"].startswith("pick"):
                    raise ValueError(f"Source mismatch on {split}:{source_row}")
                path = str(row["h5_path"])
                if path not in h5_handles:
                    h5_handles[path] = h5py.File(path, "r")
                episode = h5_handles[path][row["episode_name"]]
                native_events, num_frames = _native_boundaries(episode)
                num_events = int(teacher["step_mask"][source_row].sum())
                if len(native_events) != num_events:
                    raise ValueError(
                        f"Boundary mismatch on {split}:{source_row}: "
                        f"{len(native_events)} != {num_events}"
                    )
                pre_margins.append(min(native_events))
                post_margins.append(num_frames - 1 - max(native_events))
                feature_group = feature_file[f"episode_{int(source_row):06d}"]
                flat_rgb = np.asarray(feature_group["patch_tokens"]).reshape(
                    -1, 16, 1152
                )
                events = teacher["event_ids"][source_row, :num_events]
                local_events = np.searchsorted(PICK_EVENT_IDS, events)
                if np.any(PICK_EVENT_IDS[local_events] != events):
                    raise ValueError(f"Unexpected event on {split}:{source_row}")
                for boundary, global_event, local_event in zip(
                    native_events, events, local_events, strict=True
                ):
                    frames = boundary + frame_offsets
                    if frames[0] < 0 or frames[-1] >= num_frames:
                        raise ValueError(f"Raw centered window out of bounds: {split}:{source_row}")
                    if frames[-1] >= len(flat_rgb):
                        raise ValueError(
                            f"RGB cache does not cover centered window: {split}:{source_row}, "
                            f"{frames[-1]} >= {len(flat_rgb)}"
                        )
                    grip_rows = []
                    z_rows = []
                    for frame in frames:
                        timestep = episode[f"timestep_{int(frame)}"]
                        grip_state = np.asarray(
                            timestep["obs/gripper_state"][()], dtype=np.float32
                        )
                        grip_rows.append(
                            [
                                float(grip_state[0]),
                                float(grip_state[1]),
                                float(timestep["obs/is_gripper_close"][()]),
                                float(timestep["action/eef_action"][()][6]),
                            ]
                        )
                        z_rows.append(
                            [
                                float(timestep["obs/eef_state_raw/pose"][()][2]),
                                float(timestep["action/eef_action_raw/pose"][()][2]),
                            ]
                        )
                    rgb.append(np.asarray(flat_rgb[frames], dtype=np.float16))
                    gripper.append(np.asarray(grip_rows, dtype=np.float32))
                    eef_z.append(np.asarray(z_rows, dtype=np.float32))
                    event_targets.append(int(local_event))
                    event_global_ids.append(int(global_event))
                    boundary_frames.append(int(boundary))
                    centered_frames.append(frames)
                episode_offsets.append(len(event_targets))
                episode_indices.append(int(teacher["episode_index"][source_row]))
                required_counts.append(int(teacher["required_count"][source_row]))
        finally:
            for handle in h5_handles.values():
                handle.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_dir / f"{split}.h5", "w") as output:
        output.create_dataset(
            "patch_tokens", data=np.asarray(rgb), compression="lzf", chunks=(1, 12, 16, 1152)
        )
        output.create_dataset("gripper", data=np.asarray(gripper), compression="lzf")
        output.create_dataset("eef_z", data=np.asarray(eef_z), compression="lzf")
    arrays = {
        "event_targets": np.asarray(event_targets, dtype=np.int32),
        "event_global_ids": np.asarray(event_global_ids, dtype=np.int32),
        "boundary_frames": np.asarray(boundary_frames, dtype=np.int32),
        "centered_frames": np.asarray(centered_frames, dtype=np.int32),
        "episode_offsets": np.asarray(episode_offsets, dtype=np.int32),
        "episode_index": np.asarray(episode_indices, dtype=np.int32),
        "required_counts": np.asarray(required_counts, dtype=np.int32),
    }
    np.savez_compressed(args.output_dir / f"{split}.npz", **arrays)
    counts = np.bincount(arrays["event_targets"], minlength=3)
    return {
        "episodes": len(episode_indices),
        "events": len(event_targets),
        "event_counts": {
            teacher_lib.EVENTS[int(event_id)]: int(count)
            for event_id, count in zip(PICK_EVENT_IDS, counts, strict=True)
        },
        "min_boundary_frame": int(np.min(boundary_frames)),
        "min_pre_margin": int(min(pre_margins)),
        "min_post_margin": int(min(post_margins)),
    }


def main() -> None:
    args = parse_args()
    outputs = [args.output_dir / f"{split}.{suffix}" for split in SPLITS for suffix in ("h5", "npz")]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Output exists in {args.output_dir}; pass --overwrite")
    summary = {
        "schema_version": 1,
        "window": {
            "pre_frames": args.pre_frames,
            "post_frames": args.post_frames,
            "relative_indices": list(range(-args.pre_frames, args.post_frames)),
            "causal_at_boundary": False,
        },
        "rgb_source": "frozen_siglip_4x4_patch_tokens",
        "gripper_fields": [
            "obs/gripper_state[0]",
            "obs/gripper_state[1]",
            "obs/is_gripper_close",
            "action/eef_action[6]",
        ],
        "eef_z_fields": ["obs/eef_state_raw/pose[2]", "action/eef_action_raw/pose[2]"],
        "splits": {},
    }
    for split in SPLITS:
        summary["splits"][split] = build_split(args, split)
    # Training-only normalization avoids leaking dev/test statistics.
    with h5py.File(args.output_dir / "train.h5", "r") as train:
        for key in ("gripper", "eef_z"):
            values = np.asarray(train[key], dtype=np.float64)
            summary[f"{key}_normalization"] = {
                "mean": values.mean(axis=(0, 1)).tolist(),
                "std": np.maximum(values.std(axis=(0, 1)), 1e-6).tolist(),
            }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
