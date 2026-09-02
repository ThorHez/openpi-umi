#!/usr/bin/env python3
"""Build an action-training bank from frozen direct-visual MEM predictions.

Episodes whose visual tracker predicts the wrong final cup are replaced by the
symbolic teacher memory only in this action-training artifact.  Otherwise the
action objective would pair an incorrect target memory with the oracle action
for another cup and explicitly reward the action expert for ignoring memory.
The pure visual bank remains unchanged and must be used for final evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from openpi.training.mem.recipes import shellgame_qwen_event_memory_action as action_recipe
from scripts.mem import cache_shellgame_qwen_distilled_visual_memory as visual_cache


DEFAULT_OUTPUT = Path(
    "artifacts/shellgame_direct_visual_memory_step999_action_train_filtered_260825.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-memory", type=Path, default=visual_cache.DEFAULT_ALL_OUTPUT)
    parser.add_argument("--teacher-memory", type=Path, default=action_recipe.DEFAULT_MEMORY_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    with np.load(args.visual_memory.expanduser().resolve(), allow_pickle=False) as source:
        episodes = np.asarray(source["episode_index"], dtype=np.int32)
        visual = np.asarray(source["final_memory"], dtype=np.float32)
        labels = np.asarray(source["final_label"], dtype=np.int32)
        predictions = np.asarray(source["final_prediction"], dtype=np.int32)
        source_metadata = json.loads(str(np.asarray(source["metadata_json"]).reshape(())))
    with np.load(args.teacher_memory.expanduser().resolve(), allow_pickle=False) as source:
        teacher_templates = np.asarray(source["memory_templates"], dtype=np.float32)
        teacher_indices = np.asarray(source["episode_template_index"], dtype=np.int32)
    if not np.array_equal(episodes, np.arange(len(episodes), dtype=np.int32)):
        raise ValueError("Expected a dense all-episode visual memory bank")
    if visual.shape != (len(episodes), 128, 64):
        raise ValueError(f"Invalid visual memory shape: {visual.shape}")
    if len(teacher_indices) < len(episodes):
        raise ValueError("Teacher memory bank does not cover every episode")

    incorrect = predictions != labels
    train_memory = visual.copy()
    train_memory[incorrect] = teacher_templates[teacher_indices[episodes[incorrect]]]
    episode_template_index = np.arange(len(episodes), dtype=np.int32)
    metadata = {
        "source_visual_memory": str(args.visual_memory.expanduser().resolve()),
        "source_teacher_memory": str(args.teacher_memory.expanduser().resolve()),
        "episodes": int(len(episodes)),
        "visual_correct_episodes": int(np.sum(~incorrect)),
        "teacher_fallback_episodes": int(np.sum(incorrect)),
        "teacher_fallback_fraction": float(np.mean(incorrect)),
        "purpose": "action training only; use the pure visual bank for evaluation",
        "reason": "avoid contradictory wrong-memory plus oracle-action supervision",
        "source_metadata": source_metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        memory_templates=train_memory,
        episode_template_index=episode_template_index,
        visual_memory_correct=(~incorrect),
        final_label=labels,
        final_prediction=predictions,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
