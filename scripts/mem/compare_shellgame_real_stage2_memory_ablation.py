#!/usr/bin/env python3
"""Compare normal, zero-history, and wrong-history ShellGame evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--wrong-episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def row_key(row: dict) -> tuple[int, int]:
    return int(row["episode_id"]), int(row["frame_index"])


def compare(normal: dict, ablated: dict) -> dict:
    normal_rows = {row_key(row): row for row in normal["rows"]}
    ablated_rows = {row_key(row): row for row in ablated["rows"]}
    if normal_rows.keys() != ablated_rows.keys():
        raise ValueError("Evaluation row sets differ between normal and ablated runs")
    normal_actions = np.asarray([normal_rows[key]["pred_actions"] for key in sorted(normal_rows)], dtype=np.float64)
    ablated_actions = np.asarray([ablated_rows[key]["pred_actions"] for key in sorted(normal_rows)], dtype=np.float64)
    delta = ablated_actions - normal_actions
    normal_summary = normal["summary"]
    ablated_summary = ablated["summary"]
    return {
        "history_mode": ablated["history_mode"],
        "action_delta_xyz_rms_mm": float(np.sqrt(np.mean(np.square(delta[..., :3]))) * 1000),
        "action_delta_xyz_mean_abs_mm": float(np.mean(np.abs(delta[..., :3])) * 1000),
        "xyz_rmse_mm": ablated_summary["xyz_rmse_mm"],
        "xyz_rmse_change_mm": ablated_summary["xyz_rmse_mm"] - normal_summary["xyz_rmse_mm"],
        "xyz_rmse_ratio_to_normal": (ablated_summary["xyz_rmse_mm"] / max(normal_summary["xyz_rmse_mm"], 1e-12)),
        "direction_cosine_change": (
            ablated_summary["xyz_direction_cosine_mean"] - normal_summary["xyz_direction_cosine_mean"]
        ),
        "class_nearest_gt_centroid_accuracy": ablated_summary["class_nearest_gt_centroid_accuracy"],
        "class_accuracy_change": (
            ablated_summary["class_nearest_gt_centroid_accuracy"] - normal_summary["class_nearest_gt_centroid_accuracy"]
        ),
        "class_predicted_counts": ablated_summary["class_predicted_counts"],
        "class_max_predicted_fraction": ablated_summary["class_max_predicted_fraction"],
    }


def main() -> None:
    args = parse_args()
    normal = json.loads(args.normal.read_text(encoding="utf-8"))
    zero = json.loads(args.zero.read_text(encoding="utf-8"))
    wrong = json.loads(args.wrong_episode.read_text(encoding="utf-8"))
    if normal["history_mode"] != "normal":
        raise ValueError("--normal payload is not a normal-history run")
    zero_comparison = compare(normal, zero)
    wrong_comparison = compare(normal, wrong)
    normal_summary = normal["summary"]
    payload = {
        "normal": {
            "xyz_rmse_mm": normal_summary["xyz_rmse_mm"],
            "xyz_zero_baseline_rmse_mm": normal_summary["xyz_zero_baseline_rmse_mm"],
            "xyz_rmse_vs_zero_ratio": normal_summary["xyz_rmse_vs_zero_ratio"],
            "xyz_direction_cosine_mean": normal_summary["xyz_direction_cosine_mean"],
            "early_xyz_direction_cosine_mean": normal_summary["early_xyz_direction_cosine_mean"],
            "class_nearest_gt_centroid_accuracy": normal_summary["class_nearest_gt_centroid_accuracy"],
            "class_predicted_counts": normal_summary["class_predicted_counts"],
            "class_max_predicted_fraction": normal_summary["class_max_predicted_fraction"],
            "class_centroid_separation_ratio": normal_summary["class_centroid_separation_ratio"],
        },
        "zero_history": zero_comparison,
        "wrong_episode_history": wrong_comparison,
        "diagnostic_flags": {
            "beats_zero_action_baseline": normal_summary["xyz_rmse_vs_zero_ratio"] < 1.0,
            "positive_mean_direction": normal_summary["xyz_direction_cosine_mean"] > 0.0,
            "positive_early_direction": normal_summary["early_xyz_direction_cosine_mean"] > 0.0,
            "not_single_class_collapsed": normal_summary["class_max_predicted_fraction"] < 0.8,
            "zero_history_changes_actions": zero_comparison["action_delta_xyz_rms_mm"] > 1.0,
            "wrong_history_changes_actions": wrong_comparison["action_delta_xyz_rms_mm"] > 1.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
