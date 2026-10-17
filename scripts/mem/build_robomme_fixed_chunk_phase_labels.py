#!/usr/bin/env python3
"""Build privileged causal phase labels for fixed RoboMME chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXED = ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_VISUAL = ROOT / "artifacts/robomme_four_task_visual_student_sequences_v1_260826"
DEFAULT_OUTPUT = ROOT / "artifacts/robomme_fixed_chunk_phase_labels_v1_260829"
PHASES = ("idle", "moving", "settling", "complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--settling-fraction", type=float, default=0.7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    args = parse_args()
    if not 0.0 < args.settling_fraction < 1.0:
        raise ValueError("settling fraction must be inside (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        output = args.output_dir / f"{split}.npz"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite")
        fixed_rows = _read_jsonl(args.fixed_dir / f"{split}.jsonl")
        visual_rows = _read_jsonl(args.visual_dir / f"{split}.jsonl")
        if len(fixed_rows) != len(visual_rows):
            raise ValueError(f"Row mismatch on {split}")
        with np.load(args.fixed_dir / f"{split}.npz", allow_pickle=False) as payload:
            step_mask = np.asarray(payload["step_mask"])
            frame_indices = np.asarray(payload["frame_indices"])
            state_index = np.asarray(payload["teacher_state_index"])
        phase = np.zeros_like(step_mask, dtype=np.int32)
        completion_count = np.zeros_like(step_mask, dtype=np.int32)
        for row_index, (fixed_row, visual_row) in enumerate(
            zip(fixed_rows, visual_rows, strict=True)
        ):
            if fixed_row["episode_index"] != visual_row["episode_index"]:
                raise ValueError(f"Episode mismatch {split}:{row_index}")
            starts = [min(event["frame_indices"]) for event in visual_row["events"]]
            ends = fixed_row["event_completion_frames"]
            if len(starts) != len(ends):
                raise ValueError(f"Event mismatch {split}:{row_index}")
            length = int(step_mask[row_index].sum())
            for chunk in range(length):
                before = int(state_index[row_index, chunk])
                after = int(state_index[row_index, chunk + 1])
                completion_count[row_index, chunk] = after - before
                if after > before:
                    phase[row_index, chunk] = PHASES.index("complete")
                    continue
                if before >= len(starts):
                    phase[row_index, chunk] = PHASES.index("idle")
                    continue
                causal_end = int(frame_indices[row_index, chunk, -1])
                start, end = starts[before], ends[before]
                if causal_end < start:
                    phase[row_index, chunk] = PHASES.index("idle")
                    continue
                progress = (causal_end - start) / max(end - start, 1)
                phase[row_index, chunk] = PHASES.index(
                    "moving" if progress < args.settling_fraction else "settling"
                )
        np.savez_compressed(
            output,
            phase=phase,
            completion_count=completion_count,
            step_mask=step_mask,
        )
        valid = step_mask
        counts = np.bincount(phase[valid], minlength=len(PHASES))
        print(
            json.dumps(
                {
                    "split": split,
                    "phase_counts": dict(zip(PHASES, counts.tolist(), strict=True)),
                    "completion_count": np.bincount(
                        completion_count[valid], minlength=3
                    ).tolist(),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
