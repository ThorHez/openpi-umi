#!/usr/bin/env python3
"""Evaluate whether the real ShellGame M5 oracle probe routes actions by cup.

The evaluation uses the same seed-42 held-out episodes as training.  Class
centroids are built only from the 275 training episodes at frame 241.  For each
of the 31 validation states, it evaluates both the normal ground-truth cup and
three counterfactual oracle inputs (left/middle/right) while holding the robot
state fixed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

from openpi.policies import policy_config

OPENPI_ROOT = Path(__file__).resolve().parents[2]
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from openpi.training.mem.recipes import shellgame_real_wrist_m5 as _m5  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_m5_mixed as _m5_mixed  # noqa: E402
from scripts.mem import eval_shellgame_real_stage2_checkpoint as _action_eval  # noqa: E402

CHECKPOINT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5/"
    "real306_m5_oracle_seed42_v1/999"
)
OUTPUT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame_real/"
    "real306_m5_oracle_seed42_v1_step999/m5_oracle_action_validation.json"
)
CUP_NAMES = ("left", "middle", "right")
FRAME_INDEX = 241
ACTION_HORIZON = 16
ACTION_DIM = 10
MODEL_ACTION_DIM = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_action_eval.DATASET)
    parser.add_argument("--labels", type=Path, default=_action_eval.LABELS)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config-kind", choices=("old", "mixed"), default="old")
    parser.add_argument("--episode-offset", type=int, default=0)
    return parser.parse_args()


def load_split(dataset: Path) -> tuple[list[int], list[int]]:
    audit = json.loads((dataset / "conversion_audit.json").read_text(encoding="utf-8"))
    validation = sorted(int(value) for value in audit["validation_episode_ids"])
    if "training_episode_ids" in audit:
        training = sorted(int(value) for value in audit["training_episode_ids"])
    else:
        validation_set = set(validation)
        training = [episode for episode in range(int(audit["episodes"])) if episode not in validation_set]
    if not training or not validation:
        raise ValueError("Training and validation episode splits must both be nonempty")
    return training, validation


def load_labels(path: Path) -> list[int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [int(row["episode_id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("labels must be ordered by contiguous episode_id")
    return [int(row["final_cup"]) for row in rows]


def load_action_row(dataset: Path, episode_id: int, *, include_state: bool) -> dict:
    columns = ["actions", "frame_index", "final_cup"]
    if include_state:
        columns.extend(
            (
                "observation.robot0_eef_pos",
                "observation.robot0_eef_rot_axis_angle",
                "observation.robot0_gripper_width",
                "episode_length",
            )
        )
    table = pq.read_table(_action_eval.episode_path(dataset, episode_id), columns=columns)
    frame_indices = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)
    matches = np.flatnonzero(frame_indices == FRAME_INDEX)
    if matches.size != 1:
        raise ValueError(f"Episode {episode_id} has {matches.size} rows at frame {FRAME_INDEX}")
    return table.slice(int(matches[0]), 1).to_pylist()[0]


def build_training_centroids(
    dataset: Path,
    training_episodes: list[int],
) -> dict[int, np.ndarray]:
    by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    for episode_id in training_episodes:
        row = load_action_row(dataset, episode_id, include_state=False)
        cup = int(row["final_cup"])
        by_class[cup].append(np.asarray(row["actions"], dtype=np.float64)[..., :3].reshape(-1))
    if sorted(by_class) != [0, 1, 2]:
        raise ValueError(f"Training centroids do not cover all classes: {sorted(by_class)}")
    return {cup: np.mean(np.stack(features), axis=0) for cup, features in by_class.items()}


def nearest_class(actions: np.ndarray, centroids: dict[int, np.ndarray]) -> tuple[int, list[float]]:
    feature = np.asarray(actions, dtype=np.float64)[..., :3].reshape(-1)
    distances = [float(np.linalg.norm(feature - centroids[cup])) for cup in range(3)]
    return int(np.argmin(distances)), distances


def build_observation(
    row: dict,
    *,
    oracle_episode_id: int,
    actual_episode_id: int,
    zero_frame: np.ndarray,
) -> dict:
    observation = {
        "prompt": _action_eval.PROMPT,
        "robot0_eef_pos": np.asarray(row["observation.robot0_eef_pos"], dtype=np.float32),
        "robot0_eef_rot_axis_angle": np.asarray(
            row["observation.robot0_eef_rot_axis_angle"], dtype=np.float32
        ),
        "robot0_gripper_width": np.asarray(
            [row["observation.robot0_gripper_width"]], dtype=np.float32
        ),
        # The M5 oracle model uses episode_index only to look up the semantic
        # class. State and target remain from actual_episode_id.
        "episode_index": np.asarray(oracle_episode_id, dtype=np.int32),
        "frame_index": np.asarray(FRAME_INDEX, dtype=np.int32),
        "episode_length": np.asarray(row["episode_length"], dtype=np.int32),
    }
    observation.update(
        {
            f"{_action_eval.VIDEO_FRAME_KEY_PREFIX}{index}": zero_frame
            for index in range(_action_eval.HISTORY_FRAMES + 1)
        }
    )
    del actual_episode_id
    return observation


def confusion(rows: list[dict], target_key: str, prediction_key: str) -> list[list[int]]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for row in rows:
        matrix[int(row[target_key]), int(row[prediction_key])] += 1
    return matrix.tolist()


def class_metrics(rows: list[dict], target_key: str, prediction_key: str) -> dict:
    result = {}
    for cup in range(3):
        selected = [row for row in rows if int(row[target_key]) == cup]
        result[CUP_NAMES[cup]] = {
            "rows": len(selected),
            "accuracy": float(np.mean([int(row[prediction_key]) == cup for row in selected])),
        }
    return result


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    checkpoint = args.checkpoint.resolve()
    training_episodes, validation_episodes = load_split(dataset)
    labels = load_labels(args.labels.resolve())
    centroids = build_training_centroids(dataset, training_episodes)
    donor_by_class = {
        cup: next(episode for episode in training_episodes if labels[episode] == cup)
        for cup in range(3)
    }

    config_factory = _m5_mixed.make_train_config if args.config_kind == "mixed" else _m5.make_train_config
    config = config_factory(
        semantic_source="oracle",
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
    zero_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    normal_rows = []
    counterfactual_rows = []

    for progress, episode_id in enumerate(validation_episodes, start=1):
        row = load_action_row(dataset, episode_id, include_state=True)
        gt_cup = labels[episode_id]
        gt_actions = np.asarray(row["actions"], dtype=np.float64)

        normal_prediction = policy.infer(
            build_observation(
                row,
                oracle_episode_id=episode_id + args.episode_offset,
                actual_episode_id=episode_id,
                zero_frame=zero_frame,
            )
        )
        normal_actions = np.asarray(normal_prediction["actions"], dtype=np.float64)
        if normal_actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"Policy returned {normal_actions.shape}")
        normal_class, normal_distances = nearest_class(normal_actions, centroids)
        normal_rows.append(
            {
                "episode_id": episode_id,
                "ground_truth_cup": gt_cup,
                "predicted_action_class": normal_class,
                "predicted_action_class_name": CUP_NAMES[normal_class],
                "distance_to_training_centroids": normal_distances,
                "xyz_rmse_mm": float(
                    np.sqrt(np.mean(np.square(normal_actions[..., :3] - gt_actions[..., :3]))) * 1000
                ),
                "pred_actions": normal_actions.tolist(),
                "gt_actions": gt_actions.tolist(),
            }
        )

        forced_classes = []
        forced_actions = []
        for forced_cup in range(3):
            prediction = policy.infer(
                build_observation(
                    row,
                    oracle_episode_id=donor_by_class[forced_cup] + args.episode_offset,
                    actual_episode_id=episode_id,
                    zero_frame=zero_frame,
                )
            )
            actions = np.asarray(prediction["actions"], dtype=np.float64)
            predicted_class, distances = nearest_class(actions, centroids)
            forced_classes.append(predicted_class)
            forced_actions.append(actions[..., :3].reshape(-1))
            counterfactual_rows.append(
                {
                    "episode_id": episode_id,
                    "actual_ground_truth_cup": gt_cup,
                    "forced_oracle_cup": forced_cup,
                    "predicted_action_class": predicted_class,
                    "predicted_action_class_name": CUP_NAMES[predicted_class],
                    "distance_to_training_centroids": distances,
                }
            )
        pairwise_separation = [
            float(np.linalg.norm(forced_actions[left] - forced_actions[right]))
            for left in range(3)
            for right in range(left + 1, 3)
        ]
        print(
            f"[{progress:02d}/{len(validation_episodes):02d}] ep={episode_id:03d} gt={CUP_NAMES[gt_cup]} "
            f"normal={CUP_NAMES[normal_class]} forced={[CUP_NAMES[x] for x in forced_classes]} "
            f"xyz_rmse={normal_rows[-1]['xyz_rmse_mm']:.2f}mm "
            f"forced_sep={np.mean(pairwise_separation):.4f}",
            flush=True,
        )

    normal_correct = sum(
        int(row["predicted_action_class"] == row["ground_truth_cup"]) for row in normal_rows
    )
    counterfactual_correct = sum(
        int(row["predicted_action_class"] == row["forced_oracle_cup"])
        for row in counterfactual_rows
    )
    summary = {
        "training_episodes_for_centroids": len(training_episodes),
        "validation_episodes": len(validation_episodes),
        "normal_oracle_correct": normal_correct,
        "normal_oracle_accuracy": float(normal_correct / len(normal_rows)),
        "normal_confusion_gt_rows_pred_cols": confusion(
            normal_rows, "ground_truth_cup", "predicted_action_class"
        ),
        "normal_by_class": class_metrics(
            normal_rows, "ground_truth_cup", "predicted_action_class"
        ),
        "normal_xyz_rmse_mm": float(np.mean([row["xyz_rmse_mm"] for row in normal_rows])),
        "counterfactual_correct": counterfactual_correct,
        "counterfactual_rows": len(counterfactual_rows),
        "counterfactual_forced_class_accuracy": float(
            counterfactual_correct / len(counterfactual_rows)
        ),
        "counterfactual_confusion_forced_rows_pred_cols": confusion(
            counterfactual_rows, "forced_oracle_cup", "predicted_action_class"
        ),
        "counterfactual_by_forced_class": class_metrics(
            counterfactual_rows, "forced_oracle_cup", "predicted_action_class"
        ),
    }
    payload = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "frame_index": FRAME_INDEX,
        "config_kind": args.config_kind,
        "episode_offset": args.episode_offset,
        "action_contract": "commands 242..257 relative to measured frame 241",
        "centroid_source": "275 seed-42 training episodes, frame 241, flattened XYZ",
        "summary": summary,
        "normal_rows": normal_rows,
        "counterfactual_rows": counterfactual_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
