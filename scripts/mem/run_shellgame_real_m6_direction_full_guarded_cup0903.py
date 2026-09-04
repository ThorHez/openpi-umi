#!/usr/bin/env python3
"""Train direction-preserving full-suffix M6 with full dual-domain validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mem import run_shellgame_real_m5_m6_mixed_cup0903 as _pipeline
from openpi.training.mem.recipes import shellgame_real_wrist_m6_direction_full_mixed as _recipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--interval", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--direction-loss-weight", type=float, default=0.1)
    parser.add_argument("--anchor-fraction", type=float, default=0.5)
    parser.add_argument("--new-direction-floor", type=float, default=0.80)
    parser.add_argument("--old-direction-floor", type=float, default=0.70)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--port", type=int, default=18037)
    return parser.parse_args()


def source_metrics(summary: dict) -> dict[str, float]:
    per_prompt = [
        float(row["accuracy"])
        for row in summary["counterfactual_by_forced_prompt"].values()
    ]
    return {
        "counterfactual": float(summary["counterfactual_prompt_following_accuracy"]),
        "deployment": float(summary["deployment_action_follows_memory_prompt_accuracy"]),
        "min_prompt": min(per_prompt),
        "all_three": float(summary["counterfactual_all_three_prompts_follow_episode_accuracy"]),
    }


def passes_gate(metrics: dict[str, dict[str, float]], args: argparse.Namespace) -> bool:
    return (
        metrics["cup0903"]["counterfactual"] >= args.new_direction_floor
        and metrics["old306"]["counterfactual"] >= args.old_direction_floor
        and metrics["cup0903"]["deployment"] >= args.new_direction_floor
        and metrics["old306"]["deployment"] >= args.old_direction_floor
        and metrics["cup0903"]["min_prompt"] >= 0.70
        and metrics["old306"]["min_prompt"] >= 0.60
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.interval <= 0 or args.steps % args.interval:
        raise ValueError("steps must be positive and divisible by interval")
    if args.batch_size % 8:
        raise ValueError("batch-size must be divisible by 8")
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params not found: {args.checkpoint}")

    output_root = ROOT / "evaluation/shellgame_real" / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "pipeline_state.json"
    if state_path.exists():
        raise FileExistsError(f"State already exists: {state_path}")
    exp_name = f"{args.run_name}_m6_direction_full"
    checkpoint_root = ROOT / "checkpoints" / _recipe.CONFIG_NAME / exp_name
    best_root = ROOT / "checkpoints" / _recipe.CONFIG_NAME / f"{exp_name}_best_direction"
    if checkpoint_root.exists() or best_root.exists():
        raise FileExistsError(f"Checkpoint experiment already exists: {checkpoint_root}")

    _pipeline.update_state(
        state_path,
        status="running",
        stage="m6_direction_full",
        started_at=time.time(),
        initialization_checkpoint=str(args.checkpoint),
        direction_frames=[_recipe.DIRECTION_FRAME_START, _recipe.DIRECTION_FRAME_END],
        direction_loss_weight=args.direction_loss_weight,
        anchor_fraction=args.anchor_fraction,
        validation_interval=args.interval,
        validation_scope="all held-out episodes in old306 and cup0903; two samples per forced prompt",
    )

    history: list[dict] = []
    best_score = -1.0
    best_step = None
    best_passed = False
    no_improve = 0
    for target_step in range(args.interval, args.steps + 1, args.interval):
        command = [
            str(ROOT / ".venv/bin/python"),
            "scripts/mem/train_shellgame_real_m6_direction_full_mixed_cup0903.py",
            "--exp-name",
            exp_name,
            "--checkpoint",
            str(args.checkpoint),
            "--steps",
            str(target_step),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.batch_size),
            "--fsdp-devices",
            "8",
            "--direction-loss-weight",
            str(args.direction_loss_weight),
            "--anchor-fraction",
            str(args.anchor_fraction),
            "--save-interval",
            "5000",
        ]
        command.append("--overwrite" if target_step == args.interval else "--resume")
        _pipeline.run(command, output_root / "train.log")
        saved_step, checkpoint = _pipeline.latest_checkpoint(_recipe.CONFIG_NAME, exp_name)

        summaries = {
            domain: _pipeline.direction_eval(
                checkpoint,
                output_root / "direction_eval" / f"step{saved_step}_{domain}.json",
                domain=domain,
                config_kind="mixed_full",
                episodes_per_class=0,
                samples_per_prompt=2,
            )
            for domain in ("old306", "cup0903")
        }
        metrics = {domain: source_metrics(summary) for domain, summary in summaries.items()}
        robust_score = min(
            metrics[domain][metric]
            for domain in metrics
            for metric in ("counterfactual", "deployment", "min_prompt")
        )
        passed = passes_gate(metrics, args)
        should_select = (passed and not best_passed) or (
            passed == best_passed and robust_score >= best_score + 0.02
        )
        if should_select:
            best_score = robust_score
            best_step = saved_step
            best_passed = passed
            no_improve = 0
            _pipeline.snapshot(checkpoint, best_root / str(saved_step))
            _pipeline.prune_numeric_checkpoints(best_root, {saved_step})
        else:
            no_improve += 1
        history.append(
            {
                "step": saved_step,
                "metrics": metrics,
                "robust_score": robust_score,
                "passed_gate": passed,
                "selected_best_step": best_step,
            }
        )
        _pipeline.update_state(
            state_path,
            history=history,
            best_step=best_step,
            best_score=best_score,
            best_passed_gate=best_passed,
        )
        _pipeline.prune_numeric_checkpoints(checkpoint_root, {saved_step})

        if target_step >= 3 * args.interval and no_improve >= args.patience:
            break

    if best_step is None:
        raise RuntimeError("No direction-full checkpoint was selected")
    best_checkpoint = best_root / str(best_step)
    if not best_passed:
        _pipeline.update_state(
            state_path,
            status="stopped",
            stop_reason="full_validation_direction_gate_not_reached",
            best_checkpoint=str(best_checkpoint),
            completed_at=time.time(),
        )
        return

    _pipeline.update_state(state_path, stage="gradual_suffix_eval")
    _pipeline.gradual_eval(best_checkpoint, output_root / "gradual_suffix_eval", args.port)
    _pipeline.update_state(
        state_path,
        status="complete",
        stage="complete",
        best_checkpoint=str(best_checkpoint),
        completed_at=time.time(),
    )


if __name__ == "__main__":
    main()
