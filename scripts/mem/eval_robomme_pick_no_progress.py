#!/usr/bin/env python3
"""Evaluate whether PickXTimes MEM stays at its initial state on no-progress rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib
from scripts.mem import robomme_fixed_chunk_inference as inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional params file; defaults to <training-dir>/best/params.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = {int(value) for value in args.episodes.split(",") if value.strip()}
    sequence = np.load(
        ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826/train.npz"
    )
    teacher = np.load(
        ROOT / "artifacts/robomme_four_task_gt_teacher_memory_v2_260826/train.npz"
    )
    lookup = {int(value): index for index, value in enumerate(sequence["episode_index"])}
    predictor = inference.FixedChunkMemoryPredictor(args.training_dir, checkpoint=args.checkpoint)
    rows = []
    for episode in sorted(requested):
        index = lookup[episode]
        path = args.cache_dir / f"episode_{episode:03d}.npz"
        with np.load(path, allow_pickle=False) as payload:
            tokens = np.asarray(payload["patch_tokens"], dtype=np.float16)
        chunks = inference.tokens_to_chunks(tokens[: (len(tokens) // 12) * 12])
        output = predictor.predict_encoded(
            chunks,
            task_id=int(sequence["task_ids"][index]),
            goal_color_ids=sequence["goal_color_ids"][index].tolist(),
            required_count=int(sequence["required_counts"][index]),
            queried_ordinal=int(sequence["queried_ordinals"][index]),
            num_regions=int(sequence["num_regions"][index]),
        )
        target = teacher["state_targets"][index, 0]
        mask = teacher["state_field_mask"][index, 0]
        predictions = output["all_predictions"]
        exact = np.all((predictions == target[None]) | ~mask[None], axis=1)
        initial = output["all_memories"][0]
        final = output["all_memories"][-1]
        cosine = float(
            np.sum(initial * final)
            / max(np.linalg.norm(initial) * np.linalg.norm(final), 1e-8)
        )
        final_prediction = predictions[-1]
        decoded = {
            field: int(final_prediction[teacher_lib.STATE_FIELDS.index(field)])
            for field in ("completed_count", "holding", "ready_to_press", "done")
        }
        rows.append(
            {
                "episode": episode,
                "chunks": len(chunks),
                "all_state_exact": float(np.mean(exact)),
                "final_state_exact": bool(exact[-1]),
                "initial_final_memory_cosine": cosine,
                **decoded,
            }
        )
    result = {
        "training_dir": str(args.training_dir.resolve()),
        "checkpoint": str(predictor.checkpoint),
        "episodes": rows,
        "summary": {
            "episodes": len(rows),
            "final_state_exact": float(np.mean([row["final_state_exact"] for row in rows])),
            "all_state_exact": float(np.mean([row["all_state_exact"] for row in rows])),
            "mean_initial_final_memory_cosine": float(
                np.mean([row["initial_final_memory_cosine"] for row in rows])
            ),
            "false_done_rate": float(np.mean([row["done"] != 0 for row in rows])),
            "false_completed_count_rate": float(
                np.mean([row["completed_count"] != 0 for row in rows])
            ),
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
