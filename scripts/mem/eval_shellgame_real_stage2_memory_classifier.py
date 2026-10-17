#!/usr/bin/env python3
"""Evaluate the model-native ShellGame memory classifier on episode histories."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from openpi.policies import policy_config
from openpi.training import config as training_config

DATASET = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10"
)
LABELS = Path("/data2/hzl_workspace_for_pi_mem/labels_merged_306_degap.jsonl")
CONFIG = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_stage2"
CHECKPOINT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_stage2/"
    "real306_currentrel_full80_interface_pi05_seed42_v1/20999"
)
PROMPT = "The shell game has ended. Grasp and lift the cup containing the ball."
IMAGE_KEY = "observation.left_wrist_0_rgb_0"
VIDEO_FRAME_KEY_PREFIX = "left_wrist_0_rgb_0_"
HISTORY_FRAMES = 241
CUP_NAMES = ("left", "middle", "right")
SWAP_PAIRS = ((0, 1), (0, 2), (1, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--config-kind", choices=("registered", "fresh_memory"), default="registered")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--split", choices=("validation", "all"), default="validation")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split-domain", choices=("old306", "cup0903"))
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def episode_path(dataset: Path, episode_id: int) -> Path:
    candidates = sorted(dataset.glob(f"data/chunk-*/episode_{episode_id:06d}.parquet"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one parquet for episode {episode_id}, got {candidates}")
    return candidates[0]


def decode_image(cell: dict) -> np.ndarray:
    image = np.asarray(Image.open(BytesIO(cell["bytes"])).convert("RGB"), dtype=np.uint8)
    if image.shape != (224, 224, 3):
        raise ValueError(f"Unexpected image shape {image.shape}")
    return np.ascontiguousarray(image)


def load_history(dataset: Path, episode_id: int) -> list[np.ndarray]:
    table = pq.read_table(episode_path(dataset, episode_id), columns=[IMAGE_KEY]).slice(0, HISTORY_FRAMES)
    cells = table.column(IMAGE_KEY).to_pylist()
    if len(cells) != HISTORY_FRAMES:
        raise ValueError(f"Episode {episode_id} has only {len(cells)} history frames")
    return [decode_image(cell) for cell in cells]


def load_labels(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [int(row["episode_id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Labels are not ordered by contiguous episode_id")
    return rows


def apply_swap(cup: int, move: list[int]) -> int:
    left, right = (int(value) for value in move)
    if cup == left:
        return right
    if cup == right:
        return left
    return cup


def label_targets(label: dict) -> tuple[int, list[int], list[int]]:
    initial = int(label["initial_cup"])
    cup = initial
    relation_ids = []
    stage_cups = []
    for move in label["moves"]:
        pair = tuple(sorted(int(value) for value in move))
        relation_ids.append(SWAP_PAIRS.index(pair))
        cup = apply_swap(cup, move)
        stage_cups.append(cup)
    if cup != int(label["final_cup"]):
        raise ValueError(f"Label final cup mismatch for episode {label['episode_id']}")
    return initial, relation_ids, stage_cups


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= np.max(values, axis=-1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def classifier_observation(history: list[np.ndarray]) -> dict:
    observation = {f"{VIDEO_FRAME_KEY_PREFIX}{index}": frame for index, frame in enumerate(history)}
    observation[f"{VIDEO_FRAME_KEY_PREFIX}{HISTORY_FRAMES}"] = history[-1]
    observation.update(
        {
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_rot_axis_angle": np.asarray(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                dtype=np.float32,
            ),
            "robot0_gripper_width": np.zeros(1, dtype=np.float32),
            "prompt": PROMPT,
        }
    )
    return observation


def safe_accuracy(correct: int, total: int) -> float | None:
    return float(correct / total) if total else None


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    labels = load_labels(args.labels.resolve())
    if (args.split_manifest is None) != (args.split_domain is None):
        raise ValueError("--split-manifest and --split-domain must be provided together")
    if args.split_manifest is not None:
        split_payload = json.loads(args.split_manifest.resolve().read_text(encoding="utf-8"))
        split = split_payload["episode_split"]
        split_name = "validation" if args.split == "validation" else "training"
        episodes = [
            int(value) - args.episode_offset for value in split[split_name][args.split_domain]["global_episode_ids"]
        ]
    else:
        audit = json.loads((dataset / "conversion_audit.json").read_text(encoding="utf-8"))
        episodes = (
            [int(value) for value in audit["validation_episode_ids"]]
            if args.split == "validation"
            else list(range(len(labels)))
        )

    if args.config_kind == "fresh_memory":
        from openpi.training.mem.recipes import shellgame_real_memory_mild_all

        config = shellgame_real_memory_mild_all.make_train_config()
    else:
        config = training_config.get_config(args.config)
    policy = policy_config.create_trained_policy(config, args.checkpoint.resolve())
    confusion = np.zeros((3, 3), dtype=np.int64)
    initial_correct = 0
    relation_correct = 0
    relation_total = 0
    stage_correct = 0
    stage_total = 0
    rows = []

    for progress, episode_id in enumerate(episodes, start=1):
        initial_gt, relations_gt, stages_gt = label_targets(labels[episode_id])
        outputs = policy.infer_memory(classifier_observation(load_history(dataset, episode_id)))
        initial_probabilities = softmax(outputs["initial_logits"])
        relation_probabilities = softmax(outputs["relation_logits"])
        stage_probabilities = softmax(outputs["stage_logits"])
        initial_pred = int(np.argmax(initial_probabilities))
        relations_pred = np.argmax(relation_probabilities, axis=-1).astype(int).tolist()
        stages_pred = np.argmax(stage_probabilities, axis=-1).astype(int).tolist()
        final_gt = int(labels[episode_id]["final_cup"])
        final_pred = int(stages_pred[-1])

        confusion[final_gt, final_pred] += 1
        initial_correct += int(initial_pred == initial_gt)
        relation_correct += sum(pred == gt for pred, gt in zip(relations_pred, relations_gt, strict=True))
        relation_total += len(relations_gt)
        stage_correct += sum(pred == gt for pred, gt in zip(stages_pred, stages_gt, strict=True))
        stage_total += len(stages_gt)
        row = {
            "episode_id": episode_id,
            "final_gt": final_gt,
            "final_gt_name": CUP_NAMES[final_gt],
            "final_pred": final_pred,
            "final_pred_name": CUP_NAMES[final_pred],
            "final_probabilities": stage_probabilities[-1].tolist(),
            "initial_gt": initial_gt,
            "initial_pred": initial_pred,
            "initial_probabilities": initial_probabilities.tolist(),
            "relations_gt": relations_gt,
            "relations_pred": relations_pred,
            "relation_probabilities": relation_probabilities.tolist(),
            "stages_gt": stages_gt,
            "stages_pred": stages_pred,
            "stage_probabilities": stage_probabilities.tolist(),
            "memory_infer_ms": float(outputs["policy_timing"]["memory_infer_ms"]),
        }
        rows.append(row)
        print(
            f"[{progress:03d}/{len(episodes):03d}] ep={episode_id:03d} "
            f"gt={CUP_NAMES[final_gt]} pred={CUP_NAMES[final_pred]} "
            f"p={np.round(stage_probabilities[-1], 3).tolist()}",
            flush=True,
        )

    per_class = {}
    for cup in range(3):
        total = int(confusion[cup].sum())
        correct = int(confusion[cup, cup])
        per_class[CUP_NAMES[cup]] = {
            "correct": correct,
            "total": total,
            "accuracy": safe_accuracy(correct, total),
        }
    total_correct = int(np.trace(confusion))
    summary = {
        "split": args.split,
        "episodes": len(episodes),
        "final_correct": total_correct,
        "final_accuracy": safe_accuracy(total_correct, len(episodes)),
        "confusion_matrix_gt_rows_pred_cols": confusion.tolist(),
        "per_class": per_class,
        "initial_accuracy": safe_accuracy(initial_correct, len(episodes)),
        "relation_accuracy": safe_accuracy(relation_correct, relation_total),
        "stage_accuracy": safe_accuracy(stage_correct, stage_total),
        "mean_memory_infer_ms_excluding_compile": float(
            np.mean([row["memory_infer_ms"] for row in rows[1:]]) if len(rows) > 1 else rows[0]["memory_infer_ms"]
        ),
    }
    payload = {
        "config": args.config,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(dataset),
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
