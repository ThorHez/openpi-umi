#!/usr/bin/env python3
"""Build fixed, non-overlapping 12-frame sequences with chunk-end teacher states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VISUAL = _ROOT / "artifacts/robomme_four_task_visual_student_sequences_v1_260826"
DEFAULT_TEACHER = _ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
DEFAULT_OUTPUT = _ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-frames", type=int, default=12)
    parser.add_argument("--max-chunks", type=int, default=96)
    parser.add_argument(
        "--pick-native-simulator-events",
        action="store_true",
        help=(
            "For PickXTimes, align teacher state changes to simulator "
            "is_subgoal_boundary timestamps instead of sampled visual-event windows."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_teacher(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    for singular, plural in (
        ("task_id", "task_ids"),
        ("required_count", "required_counts"),
        ("queried_ordinal", "queried_ordinals"),
    ):
        arrays[plural] = arrays.pop(singular)
    return arrays


def _pick_native_event_ends(row: dict) -> list[int]:
    """Return PickXTimes event completions from privileged simulator metadata."""
    with h5py.File(row["h5_path"], "r") as payload:
        episode = payload[row["episode_name"]]
        timesteps = sorted(
            (key for key in episode if key.startswith("timestep_")),
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )
        boundaries = [
            int(key.rsplit("_", 1)[1])
            for key in timesteps
            if bool(episode[key]["info/is_subgoal_boundary"][()])
        ]
    # The first boundary initializes the first subgoal. Every later boundary
    # marks completion of exactly one canonical Pick/Place/Press event.
    event_ends = boundaries[1:]
    if len(event_ends) != len(row["events"]):
        raise ValueError(
            f"Native Pick boundary/event mismatch for {row['episode_name']}: "
            f"{len(event_ends)} != {len(row['events'])}"
        )
    return event_ends


def build_split(args: argparse.Namespace, split: str) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = _read_jsonl(args.visual_dir / f"{split}.jsonl")
    teacher = _load_teacher(args.teacher_dir / f"{split}.npz")
    if len(rows) != len(teacher["task_ids"]):
        raise ValueError(f"Visual/teacher episode mismatch on {split}")

    count = len(rows)
    frame_indices = np.zeros(
        (count, args.max_chunks, args.chunk_frames), dtype=np.int32
    )
    step_mask = np.zeros((count, args.max_chunks), dtype=np.bool_)
    state_change_mask = np.zeros((count, args.max_chunks), dtype=np.bool_)
    # Index zero is the initial teacher state. Each valid chunk maps to the
    # latest canonical state after all events completed by its causal end.
    teacher_state_index = np.zeros((count, args.max_chunks + 1), dtype=np.int32)
    metadata = []
    no_change_chunks = 0
    state_change_chunks = 0

    for index, row in enumerate(rows):
        if int(row["episode_index"]) != int(teacher["episode_index"][index]):
            raise ValueError(f"Episode order mismatch on {split}:{index}")
        sampled_event_ends = [max(event["frame_indices"]) for event in row["events"]]
        event_ends = (
            _pick_native_event_ends(row)
            if args.pick_native_simulator_events and row["source"].startswith("pick")
            else sampled_event_ends
        )
        if event_ends != sorted(event_ends):
            raise ValueError(f"Events are not temporal on {split}:{index}: {event_ends}")
        # Preserve the existing fixed visual horizon so cached non-overlapping
        # patches remain reusable. Native Pick completions are always causal
        # and no later than this sampled visual horizon in the audited corpus.
        relevant_end = max(sampled_event_ends)
        if max(event_ends) > relevant_end:
            raise ValueError(
                f"Native event exceeds cached visual horizon on {split}:{index}: "
                f"{max(event_ends)} > {relevant_end}"
            )
        num_chunks = (relevant_end + args.chunk_frames) // args.chunk_frames
        if num_chunks > args.max_chunks:
            raise ValueError(
                f"{split}:{index} needs {num_chunks} chunks, max is {args.max_chunks}"
            )
        previous_state = 0
        for chunk_index in range(num_chunks):
            start = chunk_index * args.chunk_frames
            raw = np.arange(start, start + args.chunk_frames, dtype=np.int32)
            # Only the final partial chunk is padded, by repeating its final
            # available frame. There is no overlap with an earlier chunk.
            raw = np.minimum(raw, relevant_end)
            frame_indices[index, chunk_index] = raw
            step_mask[index, chunk_index] = True
            causal_end = int(raw[-1])
            state_index = sum(event_end <= causal_end for event_end in event_ends)
            teacher_state_index[index, chunk_index + 1] = state_index
            if state_index == previous_state:
                no_change_chunks += 1
            else:
                state_change_chunks += 1
                state_change_mask[index, chunk_index] = True
            previous_state = state_index
        teacher_state_index[index, num_chunks + 1 :] = previous_state
        if previous_state != len(event_ends):
            raise ValueError(
                f"Final chunk does not cover all events on {split}:{index}: "
                f"{previous_state} != {len(event_ends)}"
            )
        metadata.append(
            {
                "source": row["source"],
                "episode_index": int(row["episode_index"]),
                "h5_path": row["h5_path"],
                "episode_name": row["episode_name"],
                "num_chunks": num_chunks,
                "relevant_start": 0,
                "relevant_end": relevant_end,
                "event_completion_frames": event_ends,
                "sampled_event_completion_frames": sampled_event_ends,
                "event_time_source": (
                    "native_simulator_is_subgoal_boundary"
                    if args.pick_native_simulator_events and row["source"].startswith("pick")
                    else "sampled_visual_event_window"
                ),
            }
        )

    arrays = {
        "task_ids": teacher["task_ids"],
        "goal_color_ids": teacher["goal_color_ids"],
        "required_counts": teacher["required_counts"],
        "queried_ordinals": teacher["queried_ordinals"],
        "num_regions": teacher["num_regions"],
        "episode_index": teacher["episode_index"],
        "frame_indices": frame_indices,
        "step_mask": step_mask,
        "state_change_mask": state_change_mask,
        "teacher_state_index": teacher_state_index,
        "state_change_chunks": np.asarray(state_change_chunks),
        "no_change_chunks": np.asarray(no_change_chunks),
    }
    return metadata, arrays


def main() -> None:
    args = parse_args()
    if min(args.chunk_frames, args.max_chunks) < 1:
        raise ValueError("chunk-frames and max-chunks must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / f"{split}.{suffix}" for split in SPLITS for suffix in ("npz", "jsonl")]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Outputs exist in {args.output_dir}; pass --overwrite")
    summary = {
        "schema_version": 1,
        "chunk_frames": args.chunk_frames,
        "stride_frames": args.chunk_frames,
        "overlapping_windows": False,
        "explicit_event_trigger": False,
        "max_chunks": args.max_chunks,
        "splits": {},
    }
    for split in SPLITS:
        metadata, arrays = build_split(args, split)
        np.savez_compressed(args.output_dir / f"{split}.npz", **arrays)
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in metadata),
            encoding="utf-8",
        )
        valid_chunks = int(arrays["step_mask"].sum())
        changes = int(arrays["state_change_chunks"])
        no_changes = int(arrays["no_change_chunks"])
        summary["splits"][split] = {
            "episodes": len(metadata),
            "valid_chunks": valid_chunks,
            "max_valid_chunks": int(arrays["step_mask"].sum(axis=1).max()),
            "state_change_chunks": changes,
            "no_change_chunks": no_changes,
            "no_change_fraction": no_changes / valid_chunks,
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
