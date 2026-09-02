#!/usr/bin/env python3
"""Build 6-pre/6-post PickXTimes event windows with RGB features and proprioception."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE = (
    _ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_pick_native_v1_260827"
)
DEFAULT_FEATURES = _ROOT / "artifacts/robomme_four_task_fixed_chunk_features_4x4_v1_260826"
DEFAULT_TEACHER = _ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
DEFAULT_OUTPUT = (
    _ROOT / "artifacts/pickxtimes_boundary_centered_multimodal_6pre6post_v1_260827"
)
PICK_TASK = teacher_lib.TASKS.index("pickxtimes_local_event")
EVENT_IDS = np.asarray(
    [
        teacher_lib.EVENTS.index("pick_complete"),
        teacher_lib.EVENTS.index("place_complete"),
        teacher_lib.EVENTS.index("press_complete"),
    ],
    dtype=np.int32,
)
OFFSETS = np.arange(-6, 6, dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def build_split(args: argparse.Namespace, split: str) -> dict[str, int | dict[str, int]]:
    rows = _read_jsonl(args.sequence_dir / f"{split}.jsonl")
    fixed = _load_npz(args.sequence_dir / f"{split}.npz")
    teacher = _load_npz(args.teacher_dir / f"{split}.npz")
    pick_rows = np.flatnonzero(fixed["task_ids"] == PICK_TASK)
    total_events = int(teacher["step_mask"][pick_rows].sum())
    output_path = args.output_dir / f"{split}.h5"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; pass --overwrite")

    with h5py.File(args.feature_dir / f"{split}.h5", "r") as visual, h5py.File(
        output_path, "w"
    ) as output:
        patch_tokens = output.create_dataset(
            "patch_tokens",
            shape=(total_events, 12, 16, 1152),
            dtype=np.float16,
            chunks=(1, 12, 16, 1152),
            compression="lzf",
        )
        # Layout: left finger, right finger, gripper command, observed EEF Z.
        proprio = output.create_dataset(
            "proprio", shape=(total_events, 12, 4), dtype=np.float32
        )
        event_target = output.create_dataset(
            "event_target", shape=(total_events,), dtype=np.int32
        )
        episode_index = output.create_dataset(
            "episode_index", shape=(total_events,), dtype=np.int32
        )
        episode_row = output.create_dataset(
            "episode_row", shape=(total_events,), dtype=np.int32
        )
        event_ordinal = output.create_dataset(
            "event_ordinal", shape=(total_events,), dtype=np.int32
        )
        boundary_frame = output.create_dataset(
            "boundary_frame", shape=(total_events,), dtype=np.int32
        )
        frame_indices_out = output.create_dataset(
            "frame_indices", shape=(total_events, 12), dtype=np.int32
        )

        cursor = 0
        for source_row in pick_rows:
            row = rows[int(source_row)]
            num_events = int(teacher["step_mask"][source_row].sum())
            boundaries = np.asarray(row["event_completion_frames"], dtype=np.int32)
            if len(boundaries) != num_events:
                raise ValueError(f"Boundary count mismatch on {split}:{source_row}")
            global_events = teacher["event_ids"][source_row, :num_events]
            local_events = np.searchsorted(EVENT_IDS, global_events)
            if np.any(EVENT_IDS[local_events] != global_events):
                raise ValueError(f"Unexpected event type on {split}:{source_row}")

            valid_chunks = int(fixed["step_mask"][source_row].sum())
            cached_indices = fixed["frame_indices"][source_row, :valid_chunks]
            cached_tokens = visual[f"episode_{source_row:06d}/patch_tokens"][:]
            if len(cached_tokens) != valid_chunks:
                raise ValueError(f"Feature/chunk mismatch on {split}:{source_row}")
            feature_lookup: dict[int, np.ndarray] = {}
            for chunk_indices, chunk_tokens in zip(
                cached_indices, cached_tokens, strict=True
            ):
                for frame, tokens in zip(chunk_indices, chunk_tokens, strict=True):
                    feature_lookup.setdefault(int(frame), tokens)

            with h5py.File(row["h5_path"], "r") as source:
                episode = source[row["episode_name"]]
                for ordinal, (boundary, target) in enumerate(
                    zip(boundaries, local_events, strict=True)
                ):
                    indices = boundary + OFFSETS
                    missing = [int(frame) for frame in indices if int(frame) not in feature_lookup]
                    if missing:
                        raise ValueError(
                            f"Missing centered frames on {split}:{source_row}: {missing}"
                        )
                    patch_tokens[cursor] = np.stack(
                        [feature_lookup[int(frame)] for frame in indices]
                    )
                    values = []
                    for frame in indices:
                        timestep = episode[f"timestep_{int(frame)}"]
                        gripper = np.asarray(
                            timestep["obs/gripper_state"][()], dtype=np.float32
                        ).reshape(2)
                        command = float(
                            np.asarray(timestep["action/eef_action"][()])[-1]
                        )
                        eef_z = float(
                            np.asarray(timestep["obs/eef_state"][()])[2]
                        )
                        values.append([gripper[0], gripper[1], command, eef_z])
                    proprio[cursor] = np.asarray(values, dtype=np.float32)
                    event_target[cursor] = int(target)
                    episode_index[cursor] = int(row["episode_index"])
                    episode_row[cursor] = int(source_row)
                    event_ordinal[cursor] = ordinal
                    boundary_frame[cursor] = int(boundary)
                    frame_indices_out[cursor] = indices
                    cursor += 1

        if cursor != total_events:
            raise RuntimeError(f"Wrote {cursor} events, expected {total_events}")
        output.attrs.update(
            schema_version=1,
            split=split,
            window_contract="6_pre_plus_6_post",
            pre_frames=6,
            post_frames=6,
            boundary_in_window_index=6,
            rgb_source="frozen_siglip_4x4_patch_tokens",
            proprio_layout=json.dumps(
                ["gripper_left", "gripper_right", "gripper_command", "eef_z"]
            ),
            oracle_boundary_source="info/is_subgoal_boundary",
        )

    counts = np.bincount(
        teacher["event_ids"][pick_rows][teacher["step_mask"][pick_rows]],
        minlength=len(teacher_lib.EVENTS),
    )
    return {
        "episodes": int(len(pick_rows)),
        "events": total_events,
        "event_counts": {
            teacher_lib.EVENTS[int(event_id)]: int(counts[int(event_id)])
            for event_id in EVENT_IDS
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "window_contract": "6_pre_plus_6_post",
        "frame_offsets": OFFSETS.tolist(),
        "proprio_layout": [
            "gripper_left",
            "gripper_right",
            "gripper_command",
            "eef_z",
        ],
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        summary["splits"][split] = build_split(args, split)
        print(json.dumps({split: summary["splits"][split]}, sort_keys=True), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
