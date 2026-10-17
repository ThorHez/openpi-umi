#!/usr/bin/env python3
"""Exact held-out M6 test for MEM-derived and counterfactual direction prompts.

The 275 training episodes define frame-241 XYZ action centroids.  On the exact
31 seed-42 validation episodes, the evaluator holds history/state/noise fixed
and forces LEFT, MIDDLE, and RIGHT prompts.  It separately reports deployment
behavior, where the prompt is selected from the frozen MEM prediction.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from openpi.policies import policy_config

OPENPI_ROOT = Path(__file__).resolve().parents[2]
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from openpi.training.mem.recipes import shellgame_real_wrist_m6 as _m6  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_m6_direction_stage1 as _stage1  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_m6_direction_stage1_h32 as _stage1_h32  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_m6_direction_stage1_mixed as _stage1_mixed  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_m6_mixed as _m6_mixed  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_m6_prompt_ablation as _prompt_ablation  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_stage2_h32 as _stage2_h32  # noqa: E402
from scripts.mem import eval_shellgame_real_m5_memory_action_probe as _m5_memory_eval  # noqa: E402
from scripts.mem import eval_shellgame_real_m5_oracle_action_probe as _m5_oracle_eval  # noqa: E402
from scripts.mem import eval_shellgame_real_stage2_checkpoint as _action_eval  # noqa: E402

CHECKPOINT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    f"{_m6.M6_CONFIG_NAME}/real306_m6_direction_prompt_seed42_v1/20999"
)
MEMORY_RESULTS = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame_real/"
    "real306_currentrel_full80_interface_pi05_seed42_v1_step20999/"
    "memory_classifier_validation.json"
)
OUTPUT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame_real/"
    "real306_m6_direction_prompt_seed42_v1_step20999/"
    "m6_direction_prompt_validation.json"
)
FRAME_INDEX = 241
ACTION_HORIZON = 16
ACTION_DIM = 10
MODEL_ACTION_DIM = 32
MOTION_THRESHOLD_MM = 2.0
DEPLOYMENT_FOLLOW_THRESHOLD = 0.80
COUNTERFACTUAL_FOLLOW_THRESHOLD = 0.80
PER_PROMPT_FOLLOW_THRESHOLD = 0.70
THREE_WAY_RESPONSE_THRESHOLD = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--labels", type=Path, default=_action_eval.LABELS)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--memory-results", type=Path, default=MEMORY_RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument(
        "--config-kind",
        choices=(
            "m6",
            "stage1",
            "stage1_h32",
            "mixed_stage1",
            "mixed_full",
            "prompt_only_ablation",
            "prompt_memory_ablation",
        ),
        default="m6",
    )
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split-domain", choices=("old306", "cup0903"))
    parser.add_argument(
        "--episodes-per-class",
        type=int,
        default=0,
        help="Use a deterministic balanced subset; 0 evaluates all 31 held-out episodes.",
    )
    return parser.parse_args()


def build_observation(history: list[np.ndarray], row: dict, episode_id: int, prompt: str) -> dict:
    observation = {
        "prompt": prompt,
        "robot0_eef_pos": np.asarray(row["observation.robot0_eef_pos"], dtype=np.float32),
        "robot0_eef_rot_axis_angle": np.asarray(row["observation.robot0_eef_rot_axis_angle"], dtype=np.float32),
        "robot0_gripper_width": np.asarray([row["observation.robot0_gripper_width"]], dtype=np.float32),
        "episode_index": np.asarray(episode_id, dtype=np.int32),
        "frame_index": np.asarray(FRAME_INDEX, dtype=np.int32),
        "episode_length": np.asarray(row["episode_length"], dtype=np.int32),
    }
    observation.update({f"{_action_eval.VIDEO_FRAME_KEY_PREFIX}{index}": frame for index, frame in enumerate(history)})
    current = Image.open(BytesIO(row[_action_eval.IMAGE_KEY]["bytes"])).convert("RGB")
    observation[f"{_action_eval.VIDEO_FRAME_KEY_PREFIX}{_action_eval.HISTORY_FRAMES}"] = np.ascontiguousarray(
        np.asarray(current, dtype=np.uint8)
    )
    return observation


def confusion(rows: list[dict], target_key: str, prediction_key: str) -> list[list[int]]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for row in rows:
        matrix[int(row[target_key]), int(row[prediction_key])] += 1
    return matrix.tolist()


def fraction(rows: list[dict], predicate) -> float | None:
    return float(np.mean([predicate(row) for row in rows])) if rows else None


def xyz_rms_mm(actions: np.ndarray) -> float:
    """RMS magnitude of the commanded current-relative XYZ trajectory."""
    xyz = np.asarray(actions, dtype=np.float64)[..., :3]
    return float(np.sqrt(np.mean(np.square(xyz))) * 1000)


def xyz_delta_rms_mm(left: np.ndarray, right: np.ndarray) -> float:
    """RMS XYZ response change caused only by changing the direction prompt."""
    delta = np.asarray(left, dtype=np.float64)[..., :3] - np.asarray(right, dtype=np.float64)[..., :3]
    return float(np.sqrt(np.mean(np.square(delta))) * 1000)


def endpoint_xy_error_mm(predicted: np.ndarray, target: np.ndarray) -> float:
    """Euclidean XY error at the last action in the predicted chunk."""
    delta = np.asarray(predicted, dtype=np.float64)[-1, :2] - np.asarray(target, dtype=np.float64)[-1, :2]
    return float(np.linalg.norm(delta) * 1000)


def endpoint_xy_delta_mm(left: np.ndarray, right: np.ndarray) -> float:
    """Euclidean separation between two prompted trajectory endpoints."""
    delta = np.asarray(left, dtype=np.float64)[-1, :2] - np.asarray(right, dtype=np.float64)[-1, :2]
    return float(np.linalg.norm(delta) * 1000)


def class_breakdown(rows: list[dict], selector_key: str, match_key: str) -> dict:
    result = {}
    for cup, name in enumerate(_m6.CUP_NAMES):
        selected = [row for row in rows if int(row[selector_key]) == cup]
        result[name] = {
            "rows": len(selected),
            "correct": sum(bool(row[match_key]) for row in selected),
            "accuracy": fraction(selected, lambda row: bool(row[match_key])),
        }
    return result


def minimum_accuracy(breakdown: dict) -> float | None:
    values = [float(metrics["accuracy"]) for metrics in breakdown.values() if metrics["accuracy"] is not None]
    return min(values) if values else None


def write_report(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    verdict = summary["mem_direction_action_verdict"]
    checks = verdict["checks"]
    per_prompt = summary["counterfactual_by_forced_prompt"]
    lines = [
        "# M6 MEM 方位到机械臂动作专项测试",
        "",
        f"- 离线结论: **{verdict['status'].upper()}**",
        f"- 验证 episode: {summary['validation_episodes']}",
        f"- MEM 分类准确率: {summary['memory_accuracy']:.2%}",
        (f"- 部署路径中动作跟随 MEM prompt: {summary['deployment_action_follows_memory_prompt_accuracy']:.2%}"),
        (
            "- 强制 left/middle/right prompt 后动作跟随 prompt: "
            f"{summary['counterfactual_prompt_following_accuracy']:.2%}"
        ),
        (
            "- 单个 episode 中三个 prompt 均得到对应方向动作: "
            f"{summary['counterfactual_all_three_prompts_follow_episode_accuracy']:.2%}"
        ),
        (
            "- 三个 prompt 产生三个不同动作类别: "
            f"{summary['counterfactual_three_distinct_action_classes_episode_accuracy']:.2%}"
        ),
        (
            "- 部署动作达到非零位移阈值的比例: "
            f"{summary['deployment_nontrivial_motion_fraction']:.2%} "
            f"(阈值 {MOTION_THRESHOLD_MM:.1f} mm RMS)"
        ),
        (f"- 三种 prompt 轨迹的平均两两 XYZ 差异: {summary['mean_prompt_pairwise_xyz_delta_rms_mm']:.2f} mm RMS"),
        "",
        "## 各 prompt 跟随率",
        "",
    ]
    for name in _m6.CUP_NAMES:
        metrics = per_prompt[name]
        lines.append(f"- {name}: {metrics['correct']}/{metrics['rows']} ({metrics['accuracy']:.2%})")
    lines.extend(["", "## 自动门槛", ""])
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    lines.extend(
        [
            "",
            "该结论验证的是模型输出的 EEF10 动作是否响应 MEM 方位, "
            "不等价于已经验证真机碰撞安全、工作空间限制或实际抓取成功率。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.samples_per_prompt <= 0:
        raise ValueError("samples-per-prompt must be positive")
    action_horizon = _stage2_h32.ACTION_HORIZON if args.config_kind == "stage1_h32" else ACTION_HORIZON
    default_dataset = Path(_stage2_h32.DATASET_ROOT) if args.config_kind == "stage1_h32" else _action_eval.DATASET
    dataset = (args.dataset or default_dataset).resolve()
    checkpoint = args.checkpoint.resolve()
    if (args.split_manifest is None) != (args.split_domain is None):
        raise ValueError("--split-manifest and --split-domain must be provided together")
    if args.split_manifest is not None:
        split_payload = json.loads(args.split_manifest.resolve().read_text(encoding="utf-8"))
        split = split_payload["episode_split"]
        training_episodes = [
            int(value) - args.episode_offset for value in split["training"][args.split_domain]["global_episode_ids"]
        ]
        validation_episodes = [
            int(value) - args.episode_offset for value in split["validation"][args.split_domain]["global_episode_ids"]
        ]
    else:
        training_episodes, validation_episodes = _m5_oracle_eval.load_split(dataset)
    labels = _m5_oracle_eval.load_labels(args.labels.resolve())
    if args.episodes_per_class < 0:
        raise ValueError("episodes-per-class must be nonnegative")
    if args.episodes_per_class:
        selected = []
        for cup in range(3):
            candidates = [episode for episode in validation_episodes if labels[episode] == cup]
            if len(candidates) < args.episodes_per_class:
                raise ValueError(f"held-out split has only {len(candidates)} {_m6.CUP_NAMES[cup]} episodes")
            selected.extend(candidates[: args.episodes_per_class])
        validation_episodes = sorted(selected)
    centroids = _m5_oracle_eval.build_training_centroids(dataset, training_episodes)
    memory_payload = json.loads(args.memory_results.read_text(encoding="utf-8"))
    memory_by_episode = {int(row["episode_id"]): row for row in memory_payload["rows"]}
    if not set(validation_episodes).issubset(memory_by_episode):
        raise ValueError("Memory results do not cover the selected seed-42 validation episodes")

    common_config_args = {
        "exp_name": "evaluation_only",
        "steps": 1,
        "batch_size": 1,
        "fsdp_devices": 1,
        "num_workers": 0,
        "eval_interval": 1,
        "eval_batches": 1,
        "save_interval": 1,
    }
    if args.config_kind in ("prompt_only_ablation", "prompt_memory_ablation"):
        condition_mode = "prompt_only" if args.config_kind == "prompt_only_ablation" else "prompt_memory"
        config = _prompt_ablation.make_train_config(
            condition_mode=condition_mode,
            checkpoint=str(checkpoint),
            eval_batch_size=1,
            **common_config_args,
        )
    elif args.config_kind == "mixed_stage1":
        config = _stage1_mixed.make_train_config(
            action_checkpoint=str(checkpoint),
            eval_batch_size=1,
            **common_config_args,
        )
    elif args.config_kind == "mixed_full":
        config = _m6_mixed.make_train_config(
            checkpoint=str(checkpoint),
            eval_batch_size=1,
            **common_config_args,
        )
    else:
        config_factory = {
            "m6": _m6.make_train_config,
            "stage1": _stage1.make_train_config,
            "stage1_h32": _stage1_h32.make_train_config,
        }[args.config_kind]
        config = config_factory(checkpoint=str(checkpoint), **common_config_args)
    policy = policy_config.create_trained_policy(config, checkpoint)
    deployment_rows: list[dict] = []
    counterfactual_rows: list[dict] = []
    prompt_response_rows: list[dict] = []

    for progress, episode_id in enumerate(validation_episodes, start=1):
        row = _m5_memory_eval.load_eval_row(dataset, episode_id)
        history = _action_eval.load_history(dataset, episode_id)
        gt_cup = labels[episode_id]
        memory_cup = int(memory_by_episode[episode_id]["final_pred"])
        gt_actions = np.asarray(row["actions"], dtype=np.float64)
        predictions: dict[int, np.ndarray] = {}
        classes: dict[int, int] = {}

        for forced_cup in range(3):
            samples = []
            for sample_index in range(args.samples_per_prompt):
                noise_seed = episode_id * 1_000_003 + FRAME_INDEX * 101 + sample_index
                noise = np.random.default_rng(noise_seed).standard_normal(
                    (action_horizon, MODEL_ACTION_DIM), dtype=np.float32
                )
                result = policy.infer(
                    build_observation(
                        history,
                        row,
                        episode_id + args.episode_offset,
                        _m6.direction_prompt(forced_cup),
                    ),
                    noise=noise,
                )
                action = np.asarray(result["actions"], dtype=np.float64)
                if action.shape != (action_horizon, ACTION_DIM):
                    raise ValueError(f"Policy returned {action.shape}")
                samples.append(action)
            actions = np.mean(np.stack(samples), axis=0)
            action_class, distances = _m5_oracle_eval.nearest_class(actions, centroids)
            predictions[forced_cup] = actions
            classes[forced_cup] = action_class
            counterfactual_rows.append(
                {
                    "episode_id": episode_id,
                    "ground_truth_cup": gt_cup,
                    "memory_predicted_cup": memory_cup,
                    "forced_prompt_cup": forced_cup,
                    "forced_prompt_cup_name": _m6.CUP_NAMES[forced_cup],
                    "predicted_action_class": action_class,
                    "predicted_action_class_name": _m6.CUP_NAMES[action_class],
                    "action_follows_prompt": action_class == forced_cup,
                    "commanded_xyz_rms_mm": xyz_rms_mm(actions),
                    "nontrivial_motion": xyz_rms_mm(actions) >= MOTION_THRESHOLD_MM,
                    "distance_to_training_centroids": distances,
                }
            )

        pairwise_prompt_deltas = [
            {
                "left_prompt": left,
                "left_prompt_name": _m6.CUP_NAMES[left],
                "right_prompt": right,
                "right_prompt_name": _m6.CUP_NAMES[right],
                "xyz_delta_rms_mm": xyz_delta_rms_mm(predictions[left], predictions[right]),
                "endpoint_xy_delta_mm": endpoint_xy_delta_mm(predictions[left], predictions[right]),
            }
            for left in range(3)
            for right in range(left + 1, 3)
        ]
        prompt_response_rows.append(
            {
                "episode_id": episode_id,
                "predicted_action_classes_by_prompt": [classes[cup] for cup in range(3)],
                "predicted_action_class_names_by_prompt": [_m6.CUP_NAMES[classes[cup]] for cup in range(3)],
                "all_three_prompts_follow": all(classes[cup] == cup for cup in range(3)),
                "three_distinct_action_classes": len(set(classes.values())) == 3,
                "pairwise_prompt_xyz_deltas": pairwise_prompt_deltas,
                "mean_pairwise_prompt_xyz_delta_rms_mm": float(
                    np.mean([row["xyz_delta_rms_mm"] for row in pairwise_prompt_deltas])
                ),
                "min_pairwise_prompt_xyz_delta_rms_mm": float(
                    np.min([row["xyz_delta_rms_mm"] for row in pairwise_prompt_deltas])
                ),
            }
        )

        deployment_actions = predictions[memory_cup]
        deployment_class = classes[memory_cup]
        oracle_actions = predictions[gt_cup]
        deployment_rows.append(
            {
                "episode_id": episode_id,
                "ground_truth_cup": gt_cup,
                "memory_predicted_cup": memory_cup,
                "memory_predicted_cup_name": _m6.CUP_NAMES[memory_cup],
                "memory_probabilities": memory_by_episode[episode_id]["final_probabilities"],
                "deployment_prompt": _m6.direction_prompt(memory_cup),
                "deployment_action_class": deployment_class,
                "deployment_action_class_name": _m6.CUP_NAMES[deployment_class],
                "deployment_follows_memory_prompt": deployment_class == memory_cup,
                "deployment_action_matches_ground_truth": deployment_class == gt_cup,
                "deployment_commanded_xyz_rms_mm": xyz_rms_mm(deployment_actions),
                "deployment_nontrivial_motion": (xyz_rms_mm(deployment_actions) >= MOTION_THRESHOLD_MM),
                "oracle_prompt_action_class": classes[gt_cup],
                "oracle_prompt_action_class_name": _m6.CUP_NAMES[classes[gt_cup]],
                "oracle_prompt_action_matches_ground_truth": classes[gt_cup] == gt_cup,
                "deployment_xyz_rmse_mm": float(
                    np.sqrt(np.mean(np.square(deployment_actions[..., :3] - gt_actions[..., :3]))) * 1000
                ),
                "oracle_prompt_xyz_rmse_mm": float(
                    np.sqrt(np.mean(np.square(oracle_actions[..., :3] - gt_actions[..., :3]))) * 1000
                ),
                "deployment_endpoint_xy_error_mm": endpoint_xy_error_mm(deployment_actions, gt_actions),
                "oracle_prompt_endpoint_xy_error_mm": endpoint_xy_error_mm(oracle_actions, gt_actions),
                "deployment_actions": deployment_actions.tolist(),
                "ground_truth_actions": gt_actions.tolist(),
            }
        )
        print(
            f"[{progress:02d}/{len(validation_episodes):02d}] ep={episode_id:03d} gt={_m6.CUP_NAMES[gt_cup]} "
            f"mem={_m6.CUP_NAMES[memory_cup]} "
            f"forced={[_m6.CUP_NAMES[classes[cup]] for cup in range(3)]} "
            f"deploy={_m6.CUP_NAMES[deployment_class]}",
            flush=True,
        )

    deployment_by_mem_prompt = class_breakdown(
        deployment_rows,
        "memory_predicted_cup",
        "deployment_follows_memory_prompt",
    )
    counterfactual_by_prompt = class_breakdown(
        counterfactual_rows,
        "forced_prompt_cup",
        "action_follows_prompt",
    )
    all_pairwise_deltas = [
        pair["xyz_delta_rms_mm"] for row in prompt_response_rows for pair in row["pairwise_prompt_xyz_deltas"]
    ]
    all_endpoint_pairwise_deltas = [
        pair["endpoint_xy_delta_mm"] for row in prompt_response_rows for pair in row["pairwise_prompt_xyz_deltas"]
    ]
    summary = {
        "validation_episodes": len(deployment_rows),
        "memory_accuracy": fraction(
            deployment_rows,
            lambda row: row["memory_predicted_cup"] == row["ground_truth_cup"],
        ),
        "deployment_action_follows_memory_prompt_accuracy": fraction(
            deployment_rows, lambda row: row["deployment_follows_memory_prompt"]
        ),
        "deployment_action_ground_truth_accuracy": fraction(
            deployment_rows, lambda row: row["deployment_action_matches_ground_truth"]
        ),
        "oracle_prompt_action_ground_truth_accuracy": fraction(
            deployment_rows, lambda row: row["oracle_prompt_action_matches_ground_truth"]
        ),
        "counterfactual_prompt_following_accuracy": fraction(
            counterfactual_rows, lambda row: row["action_follows_prompt"]
        ),
        "counterfactual_all_three_prompts_follow_episode_accuracy": fraction(
            prompt_response_rows, lambda row: row["all_three_prompts_follow"]
        ),
        "counterfactual_three_distinct_action_classes_episode_accuracy": fraction(
            prompt_response_rows, lambda row: row["three_distinct_action_classes"]
        ),
        "deployment_nontrivial_motion_fraction": fraction(
            deployment_rows, lambda row: row["deployment_nontrivial_motion"]
        ),
        "mean_deployment_commanded_xyz_rms_mm": float(
            np.mean([row["deployment_commanded_xyz_rms_mm"] for row in deployment_rows])
        ),
        "mean_prompt_pairwise_xyz_delta_rms_mm": float(np.mean(all_pairwise_deltas)),
        "min_prompt_pairwise_xyz_delta_rms_mm": float(np.min(all_pairwise_deltas)),
        "mean_prompt_pairwise_endpoint_xy_delta_mm": float(np.mean(all_endpoint_pairwise_deltas)),
        "min_prompt_pairwise_endpoint_xy_delta_mm": float(np.min(all_endpoint_pairwise_deltas)),
        "deployment_by_memory_prompt": deployment_by_mem_prompt,
        "counterfactual_by_forced_prompt": counterfactual_by_prompt,
        "counterfactual_confusion_prompt_rows_action_cols": confusion(
            counterfactual_rows, "forced_prompt_cup", "predicted_action_class"
        ),
        "deployment_confusion_gt_rows_action_cols": confusion(
            deployment_rows, "ground_truth_cup", "deployment_action_class"
        ),
        "mean_deployment_xyz_rmse_mm": float(np.mean([row["deployment_xyz_rmse_mm"] for row in deployment_rows])),
        "mean_oracle_prompt_xyz_rmse_mm": float(np.mean([row["oracle_prompt_xyz_rmse_mm"] for row in deployment_rows])),
        "mean_deployment_endpoint_xy_error_mm": float(
            np.mean([row["deployment_endpoint_xy_error_mm"] for row in deployment_rows])
        ),
        "mean_oracle_prompt_endpoint_xy_error_mm": float(
            np.mean([row["oracle_prompt_endpoint_xy_error_mm"] for row in deployment_rows])
        ),
    }
    per_prompt_minimum = minimum_accuracy(counterfactual_by_prompt)
    verdict_checks = {
        "deployment action follows MEM prompt >= 80%": (
            summary["deployment_action_follows_memory_prompt_accuracy"] >= DEPLOYMENT_FOLLOW_THRESHOLD
        ),
        "counterfactual action follows forced prompt >= 80%": (
            summary["counterfactual_prompt_following_accuracy"] >= COUNTERFACTUAL_FOLLOW_THRESHOLD
        ),
        "every prompt class follow rate >= 70%": (
            per_prompt_minimum is not None and per_prompt_minimum >= PER_PROMPT_FOLLOW_THRESHOLD
        ),
        "three-way prompt response episodes >= 70%": (
            summary["counterfactual_all_three_prompts_follow_episode_accuracy"] >= THREE_WAY_RESPONSE_THRESHOLD
        ),
        "deployment produces nontrivial XYZ motion >= 90%": (summary["deployment_nontrivial_motion_fraction"] >= 0.90),
    }
    passed_checks = sum(verdict_checks.values())
    summary["mem_direction_action_verdict"] = {
        "status": ("pass" if passed_checks == len(verdict_checks) else "partial" if passed_checks >= 3 else "fail"),
        "passed_checks": passed_checks,
        "total_checks": len(verdict_checks),
        "checks": verdict_checks,
        "motion_threshold_mm_rms": MOTION_THRESHOLD_MM,
        "scope": "offline predicted EEF10 actions; no physical robot motion was executed",
    }
    payload = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "frame_index": FRAME_INDEX,
        "action_horizon": action_horizon,
        "samples_per_prompt": args.samples_per_prompt,
        "config_kind": args.config_kind,
        "episodes_per_class": args.episodes_per_class,
        "episode_offset": args.episode_offset,
        "prompt_template": _m6.PROMPT_TEMPLATE,
        "deployment_prompt_source": "frozen MEM final-cup prediction",
        "centroid_source": "275 seed-42 training episodes, frame 241, flattened XYZ",
        "summary": summary,
        "deployment_rows": deployment_rows,
        "counterfactual_rows": counterfactual_rows,
        "prompt_response_rows": prompt_response_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = args.output.with_name("m6_mem_direction_action_report.md")
    write_report(report, payload)
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"saved: {args.output.resolve()}", flush=True)
    print(f"report: {report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
