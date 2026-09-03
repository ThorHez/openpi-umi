#!/usr/bin/env python3
"""Test whether MEM-left histories produce left-cup action trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import eval_shellgame_real_stage2_checkpoint as action_eval
import numpy as np

from openpi.policies import policy_config
from openpi.training import config as training_config

CONFIG = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_stage2"
CHECKPOINT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_stage2/"
    "real306_currentrel_full80_interface_pi05_seed42_v1/20999"
)
MEMORY_RESULTS = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame_real/"
    "real306_currentrel_full80_interface_pi05_seed42_v1_step20999/"
    "memory_classifier_validation.json"
)
OUTPUT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame_real/"
    "real306_currentrel_full80_interface_pi05_seed42_v1_step20999/"
    "memory_left_action_alignment.json"
)
CUP_NAMES = ("left", "middle", "right")
MODEL_ACTION_DIM = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=action_eval.DATASET)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--memory-results", type=Path, default=MEMORY_RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--memory-class", type=int, choices=(0, 1, 2), default=0)
    return parser.parse_args()


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def build_gt_centroids(dataset: Path, episode_ids: list[int]) -> dict[int, dict[int, np.ndarray]]:
    bank: dict[tuple[int, int], list[np.ndarray]] = {}
    for episode_id in episode_ids:
        rows = action_eval.load_eval_rows(dataset, episode_id)
        for frame_index in action_eval.eval_frame_indices(len(rows)):
            row = rows[frame_index]
            cup = int(row["final_cup"])
            feature = np.asarray(row["actions"], dtype=np.float64)[..., :3].reshape(-1)
            bank.setdefault((frame_index, cup), []).append(feature)
    centroids: dict[int, dict[int, np.ndarray]] = {}
    for (frame_index, cup), features in bank.items():
        centroids.setdefault(frame_index, {})[cup] = np.mean(np.stack(features), axis=0)
    incomplete = {
        frame_index: sorted(by_cup)
        for frame_index, by_cup in centroids.items()
        if sorted(by_cup) != [0, 1, 2]
    }
    if incomplete:
        raise ValueError(f"Ground-truth centroid frames do not cover all cups: {incomplete}")
    return centroids


def build_observation(history: list[np.ndarray], row: dict) -> dict:
    observation = {
        "prompt": action_eval.PROMPT,
        "robot0_eef_pos": np.asarray(row["observation.robot0_eef_pos"], dtype=np.float32),
        "robot0_eef_rot_axis_angle": np.asarray(
            row["observation.robot0_eef_rot_axis_angle"],
            dtype=np.float32,
        ),
        "robot0_gripper_width": np.asarray(
            [row["observation.robot0_gripper_width"]],
            dtype=np.float32,
        ),
    }
    observation.update(
        {
            f"{action_eval.VIDEO_FRAME_KEY_PREFIX}{index}": frame
            for index, frame in enumerate(history)
        }
    )
    observation[f"{action_eval.VIDEO_FRAME_KEY_PREFIX}{action_eval.HISTORY_FRAMES}"] = (
        action_eval.decode_image(row[action_eval.IMAGE_KEY])
    )
    return observation


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    memory_payload = json.loads(args.memory_results.read_text(encoding="utf-8"))
    all_validation_episodes = [int(row["episode_id"]) for row in memory_payload["rows"]]
    selected_memory_rows = [
        row for row in memory_payload["rows"] if int(row["final_pred"]) == args.memory_class
    ]
    selected_episodes = [int(row["episode_id"]) for row in selected_memory_rows]
    if not selected_episodes:
        raise ValueError(f"No validation episodes predicted as memory class {args.memory_class}")

    centroids = build_gt_centroids(dataset, all_validation_episodes)
    config = training_config.get_config(args.config)
    policy = policy_config.create_trained_policy(config, args.checkpoint.resolve())
    result_rows = []

    for episode_progress, memory_row in enumerate(selected_memory_rows, start=1):
        episode_id = int(memory_row["episode_id"])
        history = action_eval.load_history(dataset, episode_id)
        rows = action_eval.load_eval_rows(dataset, episode_id)
        for frame_index in action_eval.eval_frame_indices(len(rows)):
            row = rows[frame_index]
            noise_seed = episode_id * 1_000_003 + frame_index * 101
            noise = np.random.default_rng(noise_seed).standard_normal(
                (action_eval.ACTION_HORIZON, MODEL_ACTION_DIM),
                dtype=np.float32,
            )
            prediction = policy.infer(build_observation(history, row), noise=noise)
            actions = np.asarray(prediction["actions"], dtype=np.float64)
            if actions.shape != (action_eval.ACTION_HORIZON, action_eval.ACTION_DIM):
                raise ValueError(
                    f"Policy returned {actions.shape}, expected "
                    f"{(action_eval.ACTION_HORIZON, action_eval.ACTION_DIM)}"
                )
            feature = actions[..., :3].reshape(-1)
            frame_centroids = centroids[frame_index]
            distances = np.asarray(
                [np.linalg.norm(feature - frame_centroids[cup]) for cup in range(3)]
            )
            nearest_class = int(np.argmin(distances))
            gt_actions = np.asarray(row["actions"], dtype=np.float64)
            result = {
                "episode_id": episode_id,
                "frame_index": frame_index,
                "memory_pred": int(memory_row["final_pred"]),
                "memory_probabilities": memory_row["final_probabilities"],
                "ground_truth_cup": int(row["final_cup"]),
                "action_nearest_class": nearest_class,
                "action_nearest_class_name": CUP_NAMES[nearest_class],
                "distance_to_gt_class_centroids": distances.tolist(),
                "cosine_to_memory_class_centroid": cosine(
                    feature,
                    frame_centroids[args.memory_class],
                ),
                "xyz_rmse_to_ground_truth_mm": float(
                    np.sqrt(np.mean(np.square(actions[..., :3] - gt_actions[..., :3]))) * 1000
                ),
                "pred_actions": actions.tolist(),
                "gt_actions": gt_actions.tolist(),
            }
            result_rows.append(result)
            print(
                f"[{episode_progress:02d}/{len(selected_episodes):02d}] ep={episode_id:03d} "
                f"frame={frame_index:03d} mem={CUP_NAMES[args.memory_class]} "
                f"gt={CUP_NAMES[result['ground_truth_cup']]} "
                f"action={CUP_NAMES[nearest_class]} "
                f"dist={np.round(distances, 3).tolist()}",
                flush=True,
            )

    nearest_counts = np.bincount(
        [row["action_nearest_class"] for row in result_rows],
        minlength=3,
    )
    by_episode = {}
    for episode_id in selected_episodes:
        episode_rows = [row for row in result_rows if row["episode_id"] == episode_id]
        counts = np.bincount(
            [row["action_nearest_class"] for row in episode_rows],
            minlength=3,
        )
        by_episode[str(episode_id)] = {
            "ground_truth_cup": int(episode_rows[0]["ground_truth_cup"]),
            "action_nearest_counts": counts.tolist(),
            "action_majority_class": int(np.argmax(counts)),
            "action_majority_class_name": CUP_NAMES[int(np.argmax(counts))],
        }
    matching_rows = sum(
        int(row["action_nearest_class"] == args.memory_class) for row in result_rows
    )
    matching_episodes = sum(
        int(value["action_majority_class"] == args.memory_class) for value in by_episode.values()
    )
    true_memory_class_rows = [
        row for row in result_rows if row["ground_truth_cup"] == args.memory_class
    ]
    summary = {
        "memory_class": args.memory_class,
        "memory_class_name": CUP_NAMES[args.memory_class],
        "selected_episodes": selected_episodes,
        "n_episodes": len(selected_episodes),
        "n_action_rows": len(result_rows),
        "action_nearest_class_counts": nearest_counts.tolist(),
        "action_matches_memory_class_rows": matching_rows,
        "action_matches_memory_class_row_fraction": float(matching_rows / len(result_rows)),
        "action_matches_memory_class_majority_episodes": matching_episodes,
        "action_matches_memory_class_episode_fraction": float(matching_episodes / len(selected_episodes)),
        "true_memory_class_rows": len(true_memory_class_rows),
        "true_memory_class_action_match_fraction": float(
            np.mean(
                [row["action_nearest_class"] == args.memory_class for row in true_memory_class_rows]
            )
        ),
        "mean_cosine_to_memory_class_centroid": float(
            np.mean(
                [
                    row["cosine_to_memory_class_centroid"]
                    for row in result_rows
                    if row["cosine_to_memory_class_centroid"] is not None
                ]
            )
        ),
        "by_episode": by_episode,
    }
    payload = {
        "config": args.config,
        "checkpoint": str(args.checkpoint.resolve()),
        "memory_results": str(args.memory_results.resolve()),
        "summary": summary,
        "rows": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
