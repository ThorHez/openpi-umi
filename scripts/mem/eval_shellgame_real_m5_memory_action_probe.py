#!/usr/bin/env python3
"""Evaluate the frozen-MEM deterministic M5 action probe on held-out episodes."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from openpi.policies import policy_config
import pyarrow.parquet as pq
from PIL import Image


OPENPI_ROOT = Path(__file__).resolve().parents[2]
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from openpi.training.mem.recipes import shellgame_real_wrist_m5 as _m5  # noqa: E402
from scripts.mem import eval_shellgame_real_m5_oracle_action_probe as _oracle_eval  # noqa: E402
from scripts.mem import eval_shellgame_real_stage2_checkpoint as _action_eval  # noqa: E402


CHECKPOINT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5/"
    "real306_m5_memory_seed42_v1/999"
)
MEMORY_RESULTS = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame_real/"
    "real306_currentrel_full80_interface_pi05_seed42_v1_step20999/"
    "memory_classifier_validation.json"
)
OUTPUT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame_real/"
    "real306_m5_memory_seed42_v1_step999/m5_memory_action_validation.json"
)
CUP_NAMES = ("left", "middle", "right")
FRAME_INDEX = 241
ACTION_HORIZON = 16
ACTION_DIM = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_action_eval.DATASET)
    parser.add_argument("--labels", type=Path, default=_action_eval.LABELS)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--memory-results", type=Path, default=MEMORY_RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def load_eval_row(dataset: Path, episode_id: int) -> dict:
    columns = [
        "actions",
        "frame_index",
        "final_cup",
        "observation.robot0_eef_pos",
        "observation.robot0_eef_rot_axis_angle",
        "observation.robot0_gripper_width",
        "episode_length",
        _action_eval.IMAGE_KEY,
    ]
    table = pq.read_table(_action_eval.episode_path(dataset, episode_id), columns=columns)
    frame_indices = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)
    matches = np.flatnonzero(frame_indices == FRAME_INDEX)
    if matches.size != 1:
        raise ValueError(f"Episode {episode_id} has {matches.size} rows at frame {FRAME_INDEX}")
    return table.slice(int(matches[0]), 1).to_pylist()[0]


def build_observation(history: list[np.ndarray], row: dict, episode_id: int) -> dict:
    observation = {
        "prompt": _action_eval.PROMPT,
        "robot0_eef_pos": np.asarray(row["observation.robot0_eef_pos"], dtype=np.float32),
        "robot0_eef_rot_axis_angle": np.asarray(
            row["observation.robot0_eef_rot_axis_angle"], dtype=np.float32
        ),
        "robot0_gripper_width": np.asarray(
            [row["observation.robot0_gripper_width"]], dtype=np.float32
        ),
        "episode_index": np.asarray(episode_id, dtype=np.int32),
        "frame_index": np.asarray(FRAME_INDEX, dtype=np.int32),
        "episode_length": np.asarray(row["episode_length"], dtype=np.int32),
    }
    observation.update(
        {
            f"{_action_eval.VIDEO_FRAME_KEY_PREFIX}{index}": frame
            for index, frame in enumerate(history)
        }
    )
    current = Image.open(BytesIO(row[_action_eval.IMAGE_KEY]["bytes"])).convert("RGB")
    observation[f"{_action_eval.VIDEO_FRAME_KEY_PREFIX}{_action_eval.HISTORY_FRAMES}"] = (
        np.ascontiguousarray(np.asarray(current, dtype=np.uint8))
    )
    return observation


def confusion(rows: list[dict], target_key: str, prediction_key: str) -> list[list[int]]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for row in rows:
        matrix[int(row[target_key]), int(row[prediction_key])] += 1
    return matrix.tolist()


def fraction(rows: list[dict], predicate) -> float | None:
    return float(np.mean([predicate(row) for row in rows])) if rows else None


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    checkpoint = args.checkpoint.resolve()
    training_episodes, validation_episodes = _oracle_eval.load_split(dataset)
    labels = _oracle_eval.load_labels(args.labels.resolve())
    centroids = _oracle_eval.build_training_centroids(dataset, training_episodes)

    memory_payload = json.loads(args.memory_results.read_text(encoding="utf-8"))
    memory_by_episode = {int(row["episode_id"]): row for row in memory_payload["rows"]}
    if sorted(memory_by_episode) != validation_episodes:
        raise ValueError("Memory result episodes do not match the seed-42 validation split")

    config = _m5.make_train_config(
        semantic_source="memory",
        exp_name="evaluation_only",
        checkpoint=str(checkpoint),
        steps=1,
        batch_size=1,
        fsdp_devices=1,
        num_workers=0,
        eval_interval=1,
        eval_batches=1,
        save_interval=1,
    )
    policy = policy_config.create_trained_policy(config, checkpoint)
    result_rows = []

    for progress, episode_id in enumerate(validation_episodes, start=1):
        row = load_eval_row(dataset, episode_id)
        history = _action_eval.load_history(dataset, episode_id)
        prediction = policy.infer(build_observation(history, row, episode_id))
        actions = np.asarray(prediction["actions"], dtype=np.float64)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"Policy returned {actions.shape}")
        gt_actions = np.asarray(row["actions"], dtype=np.float64)
        action_class, distances = _oracle_eval.nearest_class(actions, centroids)
        memory_row = memory_by_episode[episode_id]
        memory_class = int(memory_row["final_pred"])
        gt_class = labels[episode_id]
        result = {
            "episode_id": episode_id,
            "ground_truth_cup": gt_class,
            "memory_predicted_cup": memory_class,
            "memory_probabilities": memory_row["final_probabilities"],
            "predicted_action_class": action_class,
            "predicted_action_class_name": CUP_NAMES[action_class],
            "memory_correct": memory_class == gt_class,
            "action_matches_ground_truth": action_class == gt_class,
            "action_matches_memory": action_class == memory_class,
            "distance_to_training_centroids": distances,
            "xyz_rmse_mm": float(
                np.sqrt(np.mean(np.square(actions[..., :3] - gt_actions[..., :3]))) * 1000
            ),
            "pred_actions": actions.tolist(),
            "gt_actions": gt_actions.tolist(),
        }
        result_rows.append(result)
        print(
            f"[{progress:02d}/31] ep={episode_id:03d} gt={CUP_NAMES[gt_class]} "
            f"mem={CUP_NAMES[memory_class]} action={CUP_NAMES[action_class]} "
            f"mem_ok={result['memory_correct']} follows_mem={result['action_matches_memory']} "
            f"xyz_rmse={result['xyz_rmse_mm']:.2f}mm",
            flush=True,
        )

    memory_correct_rows = [row for row in result_rows if row["memory_correct"]]
    memory_wrong_rows = [row for row in result_rows if not row["memory_correct"]]
    predicted_left_rows = [row for row in result_rows if row["memory_predicted_cup"] == 0]
    true_left_rows = [row for row in result_rows if row["ground_truth_cup"] == 0]
    summary = {
        "validation_episodes": len(result_rows),
        "memory_accuracy": fraction(result_rows, lambda row: row["memory_correct"]),
        "action_ground_truth_accuracy": fraction(
            result_rows, lambda row: row["action_matches_ground_truth"]
        ),
        "action_follows_memory_accuracy": fraction(
            result_rows, lambda row: row["action_matches_memory"]
        ),
        "memory_confusion_gt_rows_pred_cols": confusion(
            result_rows, "ground_truth_cup", "memory_predicted_cup"
        ),
        "action_confusion_gt_rows_pred_cols": confusion(
            result_rows, "ground_truth_cup", "predicted_action_class"
        ),
        "memory_to_action_confusion_memory_rows_action_cols": confusion(
            result_rows, "memory_predicted_cup", "predicted_action_class"
        ),
        "when_memory_correct": {
            "rows": len(memory_correct_rows),
            "action_ground_truth_accuracy": fraction(
                memory_correct_rows, lambda row: row["action_matches_ground_truth"]
            ),
            "action_follows_memory_accuracy": fraction(
                memory_correct_rows, lambda row: row["action_matches_memory"]
            ),
        },
        "when_memory_wrong": {
            "rows": len(memory_wrong_rows),
            "action_ground_truth_accuracy": fraction(
                memory_wrong_rows, lambda row: row["action_matches_ground_truth"]
            ),
            "action_follows_memory_accuracy": fraction(
                memory_wrong_rows, lambda row: row["action_matches_memory"]
            ),
        },
        "when_memory_predicts_left": {
            "rows": len(predicted_left_rows),
            "action_goes_left_accuracy": fraction(
                predicted_left_rows, lambda row: row["predicted_action_class"] == 0
            ),
            "ground_truth_left_rows": sum(
                int(row["ground_truth_cup"] == 0) for row in predicted_left_rows
            ),
        },
        "true_left": {
            "rows": len(true_left_rows),
            "memory_predicts_left_accuracy": fraction(
                true_left_rows, lambda row: row["memory_predicted_cup"] == 0
            ),
            "action_goes_left_accuracy": fraction(
                true_left_rows, lambda row: row["predicted_action_class"] == 0
            ),
        },
        "mean_xyz_rmse_mm": float(np.mean([row["xyz_rmse_mm"] for row in result_rows])),
    }
    payload = {
        "checkpoint": str(checkpoint),
        "memory_results": str(args.memory_results.resolve()),
        "dataset": str(dataset),
        "frame_index": FRAME_INDEX,
        "action_contract": "commands 242..257 relative to measured frame 241",
        "centroid_source": "275 seed-42 training episodes, frame 241, flattened XYZ",
        "summary": summary,
        "rows": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
