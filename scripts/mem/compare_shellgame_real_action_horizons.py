#!/usr/bin/env python3
"""Compare completed H16 and H32 guarded direction experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUMMARY_METRICS = (
    "deployment_action_follows_memory_prompt_accuracy",
    "deployment_action_ground_truth_accuracy",
    "counterfactual_prompt_following_accuracy",
    "counterfactual_all_three_prompts_follow_episode_accuracy",
    "counterfactual_three_distinct_action_classes_episode_accuracy",
    "mean_prompt_pairwise_xyz_delta_rms_mm",
    "mean_deployment_xyz_rmse_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h16-guard", type=Path, required=True)
    parser.add_argument("--h32-guard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_run(path: Path, expected_horizon: int) -> dict:
    guard = json.loads(path.read_text(encoding="utf-8"))
    evaluation = Path(guard["full_evaluation"])
    full = json.loads(evaluation.read_text(encoding="utf-8"))
    horizon = int(full.get("action_horizon", 16))
    if horizon != expected_horizon:
        raise ValueError(f"Expected H{expected_horizon} at {path}, got H{horizon}")
    return {
        "guard_path": str(path.resolve()),
        "action_horizon": horizon,
        "stop_reason": guard["stop_reason"],
        "best_step": int(guard["best_step"]),
        "best_checkpoint": guard["best_checkpoint"],
        "best_screen_score": float(guard["best_screen_score"]),
        "full_evaluation": str(evaluation.resolve()),
        "full_summary": full["summary"],
    }


def main() -> None:
    args = parse_args()
    h16 = load_run(args.h16_guard.resolve(), 16)
    h32 = load_run(args.h32_guard.resolve(), 32)
    delta = {
        metric: float(h32["full_summary"][metric]) - float(h16["full_summary"][metric]) for metric in SUMMARY_METRICS
    }
    payload = {
        "comparison": "H32 minus H16; same seed-42 split, frame241 sampler, direction loss, and global batch 32",
        "h16": h16,
        "h32": h32,
        "delta_h32_minus_h16": delta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = args.output.with_suffix(".md")
    lines = [
        "# ShellGame real action horizon: H16 vs H32",
        "",
        "Only action horizon differs. Both runs use the seed42 episode split, frame241, "
        "balanced left/middle/right sampling, global batch 32, and flow loss + 0.1 x direction loss.",
        "",
        "| metric | H16 | H32 | H32-H16 |",
        "|---|---:|---:|---:|",
        f"| best screen direction accuracy | {h16['best_screen_score']:.4f} | {h32['best_screen_score']:.4f} | {h32['best_screen_score'] - h16['best_screen_score']:+.4f} |",
    ]
    lines.extend(
        (
            f"| {metric} | {float(h16['full_summary'][metric]):.4f} | "
            f"{float(h32['full_summary'][metric]):.4f} | {delta[metric]:+.4f} |"
        )
        for metric in SUMMARY_METRICS
    )
    lines.extend(
        [
            "",
            f"- H16 best: `{h16['best_checkpoint']}` (step {h16['best_step']})",
            f"- H32 best: `{h32['best_checkpoint']}` (step {h32['best_step']})",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"report: {report.resolve()}")


if __name__ == "__main__":
    main()
