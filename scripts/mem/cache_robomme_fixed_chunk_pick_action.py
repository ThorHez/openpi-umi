#!/usr/bin/env python3
"""Replace PickXTimes action-cache memories with fixed-chunk student trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib
from scripts.mem import robomme_fixed_chunk_inference as inference


DEFAULT_BASE_CACHE = ROOT / "data/robomme_extracted/pickxtimes_memory_action_train70_dev15_stride2.h5"
DEFAULT_SEQUENCE = ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_FEATURES = ROOT / "artifacts/robomme_four_task_fixed_chunk_features_4x4_v1_260826"
DEFAULT_TRAINING = ROOT / "checkpoints/robomme_single_task_pick_equal_exposure_seed260827_260827"
DEFAULT_OUTPUT = ROOT / (
    "data/robomme_extracted/pickxtimes_fixed_chunk_single_mem_action_train70_dev15_stride2_260827.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite")
    predictor = inference.FixedChunkMemoryPredictor(args.training_dir)
    task_id = teacher_lib.TASKS.index("pickxtimes_local_event")
    episode_banks: dict[int, np.ndarray] = {}
    episode_visible: dict[int, np.ndarray] = {}

    for split in ("train", "dev"):
        rows = _rows(args.sequence_dir / f"{split}.jsonl")
        with (
            np.load(args.sequence_dir / f"{split}.npz", allow_pickle=False) as sequence,
            h5py.File(args.feature_dir / f"{split}.h5", "r") as features,
        ):
            for ordinal, row in enumerate(rows):
                if row["source"] != "pickxtimes_local_event":
                    continue
                count = int(np.asarray(sequence["step_mask"])[ordinal].sum())
                chunks = np.asarray(
                    features[f"episode_{ordinal:06d}/patch_tokens"][:count], dtype=np.float16
                )
                output = predictor.predict_encoded(
                    chunks,
                    task_id=task_id,
                    goal_color_ids=np.asarray(sequence["goal_color_ids"])[ordinal].tolist(),
                    required_count=int(np.asarray(sequence["required_counts"])[ordinal]),
                    queried_ordinal=int(np.asarray(sequence["queried_ordinals"])[ordinal]),
                    num_regions=int(np.asarray(sequence["num_regions"])[ordinal]),
                )
                episode_index = int(row["episode_index"])
                episode_banks[episode_index] = output["all_memories"].astype(np.float16)
                frame_indices = np.asarray(sequence["frame_indices"])[ordinal, :count]
                episode_visible[episode_index] = frame_indices[:, -1].astype(np.int32)
                print(
                    f"{split} episode={episode_index} chunks={count} "
                    f"gate={float(output['write_gates'].mean()):.4f}",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.base_cache, "r") as source, h5py.File(args.output, "w") as target:
        for key, value in source.attrs.items():
            target.attrs[key] = value
        target.attrs.update(
            memory_source="fixed_chunk_single_task_student",
            memory_training_dir=str(args.training_dir.resolve()),
            memory_checkpoint=str((args.training_dir / "best/params").resolve()),
            frozen_test_accessed=False,
        )
        for key in source.keys():
            if key not in {"predicted", "oracle"}:
                source.copy(key, target)
        if "oracle" in source:
            source.copy("oracle", target)

        source_episodes = np.asarray(source["episode_indices"], dtype=np.int32)
        timesteps = np.asarray(source["timesteps"], dtype=np.int32)
        banks = []
        indices = np.empty(len(source_episodes), dtype=np.int32)
        offset = 0
        for episode_index in dict.fromkeys(source_episodes.tolist()):
            if episode_index not in episode_banks:
                raise ValueError(f"Missing fixed-chunk trajectory for episode {episode_index}")
            bank = episode_banks[episode_index]
            visible = episode_visible[episode_index]
            mask = source_episodes == episode_index
            indices[mask] = offset + np.searchsorted(visible, timesteps[mask], side="right")
            banks.append(bank)
            offset += len(bank)
        predicted = target.create_group("predicted")
        predicted.create_dataset(
            "memory_bank", data=np.concatenate(banks), compression="gzip", compression_opts=1
        )
        predicted.create_dataset("memory_indices", data=indices, compression="gzip", compression_opts=1)
    print(f"Wrote {args.output.resolve()} with {len(episode_banks)} episode trajectories", flush=True)


if __name__ == "__main__":
    main()
