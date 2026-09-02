#!/usr/bin/env python3
"""Attach causal 12-frame visual windows to the four-task GT-teacher episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openpi.tasks.robomme import unified_gt_teacher as teacher_contract  # noqa: E402
from scripts.mem.build_robomme_four_task_gt_teacher_sequences import _canonical_rows  # noqa: E402

DEFAULT_EVENTS = _ROOT / "artifacts/robomme_qwen_unified_events_optimized_v2_seed260826"
DEFAULT_TEACHER = _ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
DEFAULT_OUTPUT = _ROOT / "artifacts/robomme_four_task_visual_student_sequences_v1_260826"
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _event_signature(event: dict) -> tuple[str, str | None, str | None, str | None]:
    return event["event"], event.get("entity"), event.get("region_a"), event.get("region_b")


def _row_signature(row: dict) -> tuple[str, str | None, str | None, str | None]:
    target = row["target"]
    if isinstance(target, str):
        target = json.loads(target)
    return target["event"], target.get("entity"), target.get("region_a"), target.get("region_b")


def build_split(events_dir: Path, teacher_dir: Path, split: str) -> list[dict]:
    teacher_episodes = _read_jsonl(teacher_dir / f"{split}.jsonl")
    rows_by_episode: dict[tuple[str, int], list[dict]] = {}
    sources = sorted({episode["source"] for episode in teacher_episodes})
    for source in sources:
        source_rows = _read_jsonl(events_dir / source / f"{split}.jsonl")
        episode_ids = sorted({int(row["episode_index"]) for row in source_rows})
        for episode_index in episode_ids:
            rows = [row for row in source_rows if int(row["episode_index"]) == episode_index]
            rows_by_episode[(source, episode_index)] = _canonical_rows(source, rows)

    output = []
    for teacher in teacher_episodes:
        key = teacher["source"], int(teacher["episode_index"])
        visual_rows = rows_by_episode[key]
        expected = [_event_signature(event) for event in teacher["events"]]
        actual = [_row_signature(row) for row in visual_rows]
        if actual != expected:
            raise ValueError(f"Visual/teacher event mismatch for {key}: {actual} != {expected}")
        if any(len(row["frame_indices"]) != 12 for row in visual_rows):
            raise ValueError(f"Every visual window must contain 12 frames: {key}")
        output.append(
            {
                "source": teacher["source"],
                "episode_index": int(teacher["episode_index"]),
                "h5_path": visual_rows[0]["h5_path"],
                "episode_name": visual_rows[0]["episode_name"],
                "task_id": teacher_contract.TASKS.index(teacher["source"]),
                "goal_color_ids": teacher["goal_color_ids"],
                "required_count": teacher["required_count"],
                "queried_ordinal": teacher["queried_ordinal"],
                "num_regions": teacher["num_regions"],
                "events": [
                    {
                        "frame_indices": [int(index) for index in row["frame_indices"]],
                        "sample_type": row["sample_type"],
                        "event": teacher["events"][index],
                    }
                    for index, row in enumerate(visual_rows)
                ],
            }
        )
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / f"{split}.jsonl" for split in SPLITS]
    if not args.overwrite and any(path.exists() for path in paths):
        raise FileExistsError(f"Outputs exist in {args.output_dir}; pass --overwrite")
    summary = {"schema_version": 1, "window_frames": 12, "splits": {}}
    for split, path in zip(SPLITS, paths, strict=True):
        episodes = build_split(args.events_dir, args.teacher_dir, split)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in episodes),
            encoding="utf-8",
        )
        summary["splits"][split] = {
            "episodes": len(episodes),
            "events": sum(len(row["events"]) for row in episodes),
            "task_counts": {
                source: sum(row["source"] == source for row in episodes)
                for source in sorted({row["source"] for row in episodes})
            },
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
