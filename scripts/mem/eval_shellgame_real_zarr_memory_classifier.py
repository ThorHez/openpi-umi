#!/usr/bin/env python3
"""Evaluate the frozen real-ShellGame MEM classifier directly on a raw Zarr replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import zarr


OPENPI_ROOT = Path(__file__).resolve().parents[2]
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as training_config  # noqa: E402
from scripts.mem import eval_shellgame_real_stage2_memory_classifier as _base  # noqa: E402


DEFAULT_ZARR = Path("/data2/hzl_workspace_for_pi_mem/cup_0903/replay_buffer.zarr")
DEFAULT_LABELS = Path("/data2/hzl_workspace_for_pi_mem/cup_0903/labels.jsonl")
DEFAULT_OUTPUT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame_real/"
    "cup_0903_old_mem4999_all100/memory_classifier.json"
)
OLD_MEMORY_CHECKPOINT = Path("/data2/hzl_workspace_for_pi_mem/4999/params")
BASELINE = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame_real/"
    "real306_currentrel_full80_interface_pi05_seed42_v1_step20999/"
    "memory_classifier_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--config", default=_base.CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=_base.CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    return args


def _to_uint8(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.shape[1:] != (224, 224, 3):
        raise ValueError(f"Unexpected camera shape {images.shape}")
    if np.issubdtype(images.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(images)) <= 1.5 else 1.0
        images = np.clip(np.rint(images * scale), 0, 255).astype(np.uint8)
    elif images.dtype != np.uint8:
        images = np.clip(images, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(images)


def _load_labels(path: Path) -> list[dict]:
    labels = _base.load_labels(path)
    for label in labels:
        if int(label.get("n_observe_frames", -1)) != _base.HISTORY_FRAMES:
            raise ValueError(
                f"Episode {label['episode_id']} has n_observe_frames="
                f"{label.get('n_observe_frames')}, expected {_base.HISTORY_FRAMES}"
            )
        _base.label_targets(label)
    return labels


def _metric_summary(rows: list[dict]) -> dict:
    confusion = np.zeros((3, 3), dtype=np.int64)
    initial_correct = 0
    relation_correct = 0
    relation_total = 0
    stage_correct = 0
    stage_total = 0
    for row in rows:
        confusion[row["final_gt"], row["final_pred"]] += 1
        initial_correct += int(row["initial_gt"] == row["initial_pred"])
        relation_correct += sum(
            pred == target
            for pred, target in zip(row["relations_pred"], row["relations_gt"], strict=True)
        )
        relation_total += len(row["relations_gt"])
        stage_correct += sum(
            pred == target
            for pred, target in zip(row["stages_pred"], row["stages_gt"], strict=True)
        )
        stage_total += len(row["stages_gt"])

    per_class = {}
    for cup, name in enumerate(_base.CUP_NAMES):
        total = int(confusion[cup].sum())
        correct = int(confusion[cup, cup])
        per_class[name] = {
            "correct": correct,
            "total": total,
            "accuracy": _base.safe_accuracy(correct, total),
        }
    predicted_counts = {
        name: int(confusion[:, cup].sum()) for cup, name in enumerate(_base.CUP_NAMES)
    }
    correct_confidences = [row["final_confidence"] for row in rows if row["final_correct"]]
    incorrect_confidences = [row["final_confidence"] for row in rows if not row["final_correct"]]
    return {
        "episodes": len(rows),
        "final_correct": int(np.trace(confusion)),
        "final_accuracy": _base.safe_accuracy(int(np.trace(confusion)), len(rows)),
        "confusion_matrix_gt_rows_pred_cols": confusion.tolist(),
        "per_class": per_class,
        "predicted_counts": predicted_counts,
        "initial_accuracy": _base.safe_accuracy(initial_correct, len(rows)),
        "relation_accuracy": _base.safe_accuracy(relation_correct, relation_total),
        "stage_accuracy": _base.safe_accuracy(stage_correct, stage_total),
        "mean_final_confidence": float(np.mean([row["final_confidence"] for row in rows])),
        "mean_correct_final_confidence": (
            float(np.mean(correct_confidences)) if correct_confidences else None
        ),
        "mean_incorrect_final_confidence": (
            float(np.mean(incorrect_confidences)) if incorrect_confidences else None
        ),
        "mean_memory_infer_ms_excluding_compile": float(
            np.mean([row["memory_infer_ms"] for row in rows[1:]])
            if len(rows) > 1
            else rows[0]["memory_infer_ms"]
        ),
    }


def main() -> None:
    args = parse_args()
    zarr_path = args.zarr.resolve()
    labels_path = args.labels.resolve()
    labels = _load_labels(labels_path)
    replay = zarr.open(zarr_path, mode="r")
    camera = replay["data"]["camera0_rgb"]
    episode_ends = np.asarray(replay["meta"]["episode_ends"][:], dtype=np.int64)
    episode_starts = np.concatenate((np.zeros(1, dtype=np.int64), episode_ends[:-1]))
    if len(episode_ends) != len(labels):
        raise ValueError(f"Zarr has {len(episode_ends)} episodes, labels have {len(labels)}")
    if camera.shape != (int(episode_ends[-1]), 224, 224, 3):
        raise ValueError(f"Unexpected camera array shape {camera.shape}")
    actual_lengths = episode_ends - episode_starts
    if np.any(actual_lengths < _base.HISTORY_FRAMES):
        bad = np.flatnonzero(actual_lengths < _base.HISTORY_FRAMES).tolist()
        raise ValueError(f"Episodes shorter than {_base.HISTORY_FRAMES} frames: {bad}")

    episode_ids = list(range(len(labels)))
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]

    config = training_config.get_config(args.config)
    policy = policy_config.create_trained_policy(config, args.checkpoint.resolve())
    rows = []
    for progress, episode_id in enumerate(episode_ids, start=1):
        start = int(episode_starts[episode_id])
        history_array = _to_uint8(camera[start : start + _base.HISTORY_FRAMES])
        history = [history_array[index] for index in range(_base.HISTORY_FRAMES)]
        initial_gt, relations_gt, stages_gt = _base.label_targets(labels[episode_id])
        outputs = policy.infer_memory(_base.classifier_observation(history))
        initial_probabilities = _base.softmax(outputs["initial_logits"])
        relation_probabilities = _base.softmax(outputs["relation_logits"])
        stage_probabilities = _base.softmax(outputs["stage_logits"])
        initial_pred = int(np.argmax(initial_probabilities))
        relations_pred = np.argmax(relation_probabilities, axis=-1).astype(int).tolist()
        stages_pred = np.argmax(stage_probabilities, axis=-1).astype(int).tolist()
        final_gt = int(labels[episode_id]["final_cup"])
        final_pred = int(stages_pred[-1])
        row = {
            "episode_id": episode_id,
            "actual_episode_frames": int(actual_lengths[episode_id]),
            "label_n_frames": int(labels[episode_id].get("n_frames", -1)),
            "final_gt": final_gt,
            "final_gt_name": _base.CUP_NAMES[final_gt],
            "final_pred": final_pred,
            "final_pred_name": _base.CUP_NAMES[final_pred],
            "final_correct": final_pred == final_gt,
            "final_confidence": float(stage_probabilities[-1, final_pred]),
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
            f"[{progress:03d}/{len(episode_ids):03d}] ep={episode_id:03d} "
            f"gt={row['final_gt_name']} pred={row['final_pred_name']} "
            f"p={np.round(stage_probabilities[-1], 3).tolist()}",
            flush=True,
        )

    summary = _metric_summary(rows)
    old_baseline = None
    if BASELINE.is_file():
        old_baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["summary"]
    payload = {
        "config": args.config,
        "model_container_checkpoint": str(args.checkpoint.resolve()),
        "frozen_memory_origin_checkpoint": str(OLD_MEMORY_CHECKPOINT.resolve()),
        "memory_weight_contract": (
            "The Stage2 container copies /4999 MEM weights and freezes the complete visual tracker; "
            "action-policy training does not update these classifier weights."
        ),
        "zarr": str(zarr_path),
        "labels": str(labels_path),
        "history_frames": _base.HISTORY_FRAMES,
        "image_conversion": "float [0,1] -> round(x*255) uint8, matching the training-data converter",
        "label_vs_zarr_length_mismatch_episodes": int(
            np.sum(actual_lengths != np.asarray([row.get("n_frames", -1) for row in labels]))
        ),
        "summary": summary,
        "old_environment_validation_baseline": old_baseline,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
