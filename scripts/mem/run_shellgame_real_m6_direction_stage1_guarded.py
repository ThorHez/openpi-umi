#!/usr/bin/env python3
"""Guarded 8-GPU stage-1 training with fixed-step forced-direction evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CONFIG = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_stage1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="real306_m6_direction_stage1_frame241_flowonly_seed42_v1")
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--episodes-per-class", type=int, default=2)
    parser.add_argument(
        "--min-early-stop-step",
        type=int,
        default=500,
        help="Do not declare direction ineffective/stagnant before this many updates.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--direction-temperature", type=float, default=5e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def checkpoint_step(run_dir: Path) -> tuple[int, Path]:
    candidates = sorted((int(path.name), path) for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit())
    if not candidates:
        raise FileNotFoundError(f"No numeric checkpoint under {run_dir}")
    return candidates[-1]


def hardlink_snapshot(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=os.link, symlinks=True)


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.interval <= 0 or args.batch_size <= 0:
        raise ValueError("max-steps, interval, and batch-size must be positive")
    if args.max_steps % args.interval:
        raise ValueError("max-steps must be divisible by interval")
    if args.batch_size % 8:
        raise ValueError("batch-size must be divisible by the eight FSDP devices")
    if args.eval_batch_size % 8:
        raise ValueError("eval-batch-size must be divisible by the eight FSDP devices")
    if args.min_early_stop_step < args.interval:
        raise ValueError("min-early-stop-step must be at least one evaluation interval")

    run_dir = ROOT / "checkpoints" / CONFIG / args.exp_name
    best_root = ROOT / "checkpoints" / CONFIG / f"{args.exp_name}_best_direction"
    output_root = ROOT / "evaluation" / "shellgame_real" / args.exp_name
    train_log = ROOT / f"train_{args.exp_name}.log"
    guard_log = ROOT / f"guard_{args.exp_name}.json"
    if args.overwrite:
        for path in (run_dir, best_root, output_root):
            if path.exists():
                shutil.rmtree(path)
        for path in (train_log, guard_log):
            if path.exists():
                path.unlink()
    elif run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}; use --overwrite or choose another exp-name")

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "PYTHONUNBUFFERED": "1",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
            "HF_HOME": str(ROOT / ".cache/shellgame_real_stage2/huggingface"),
            "HF_DATASETS_CACHE": str(ROOT / ".cache/shellgame_real_stage2/huggingface/datasets"),
        }
    )
    (ROOT / ".cache/shellgame_real_stage2/huggingface/datasets").mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_score = float("-inf")
    best_step = None
    no_meaningful_improvement = 0
    stop_reason = None
    started = time.time()

    for target_step in range(args.interval, args.max_steps + 1, args.interval):
        # Trainer step paths are zero based: target 100 updates ends at checkpoint 99.
        num_train_steps = target_step
        command = [
            str(ROOT / ".venv/bin/python"),
            "scripts/mem/train_shellgame_real_m6_direction_stage1.py",
            "--exp-name", args.exp_name,
            "--steps", str(num_train_steps),
            "--schedule-steps", str(args.max_steps),
            "--batch-size", str(args.batch_size),
            "--eval-batch-size", str(args.eval_batch_size),
            "--fsdp-devices", "8",
            "--num-workers", "16",
            "--eval-interval", str(args.interval),
            "--eval-batches", "3",
            "--direction-loss-weight", str(args.direction_loss_weight),
            "--direction-temperature", str(args.direction_temperature),
            "--save-interval", str(args.interval),
            "--disable-direction-early-stop",
        ]
        if target_step == args.interval:
            command.append("--overwrite")
        else:
            command.append("--resume")
        run(command, env=env, log_path=train_log)

        saved_step, checkpoint = checkpoint_step(run_dir)
        eval_output = output_root / f"screen_step{saved_step}" / "direction_screen.json"
        eval_env = dict(env)
        eval_env["CUDA_VISIBLE_DEVICES"] = "0"
        eval_command = [
            str(ROOT / ".venv/bin/python"),
            "scripts/mem/eval_shellgame_real_m6_direction_prompt.py",
            "--config-kind", "stage1",
            "--checkpoint", str(checkpoint),
            "--output", str(eval_output),
            "--samples-per-prompt", "1",
            "--episodes-per-class", str(args.episodes_per_class),
        ]
        run(eval_command, env=eval_env, log_path=output_root / f"screen_step{saved_step}" / "eval.log")
        summary = json.loads(eval_output.read_text(encoding="utf-8"))["summary"]
        score = float(summary["counterfactual_prompt_following_accuracy"])
        min_class = min(
            float(values["accuracy"])
            for values in summary["counterfactual_by_forced_prompt"].values()
        )
        three_way = float(summary["counterfactual_all_three_prompts_follow_episode_accuracy"])
        distinct = float(summary["counterfactual_three_distinct_action_classes_episode_accuracy"])

        meaningful = score >= best_score + 0.03
        if meaningful:
            best_score = score
            best_step = saved_step
            no_meaningful_improvement = 0
            if best_root.exists():
                shutil.rmtree(best_root)
            hardlink_snapshot(checkpoint, best_root / str(saved_step))
        else:
            no_meaningful_improvement += 1
        record = {
            "checkpoint_step": saved_step,
            "forced_prompt_accuracy": score,
            "min_prompt_class_accuracy": min_class,
            "all_three_accuracy": three_way,
            "three_distinct_accuracy": distinct,
            "meaningful_improvement": meaningful,
            "best_score": best_score,
            "best_step": best_step,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        if score >= 0.80 and min_class >= 0.70 and three_way >= 0.70:
            stop_reason = "success_gate"
        elif (
            saved_step + 1 >= args.min_early_stop_step
            and no_meaningful_improvement >= args.early_stop_patience
            and best_score < 0.55
        ):
            stop_reason = "direction_ineffective_early_stop"
        elif (
            saved_step + 1 >= args.min_early_stop_step
            and no_meaningful_improvement >= args.early_stop_patience
        ):
            stop_reason = "direction_stagnation_early_stop"
        if stop_reason:
            break

    if best_step is None:
        raise RuntimeError("No direction checkpoint was selected")
    best_checkpoint = best_root / str(best_step)
    full_output = output_root / f"best_step{best_step}_full31" / "direction_full31.json"
    full_env = dict(env)
    full_env["CUDA_VISIBLE_DEVICES"] = "0"
    run(
        [
            str(ROOT / ".venv/bin/python"),
            "scripts/mem/eval_shellgame_real_m6_direction_prompt.py",
            "--config-kind", "stage1",
            "--checkpoint", str(best_checkpoint),
            "--output", str(full_output),
            "--samples-per-prompt", "2",
        ],
        env=full_env,
        log_path=full_output.parent / "eval.log",
    )
    payload = {
        "status": "complete",
        "stop_reason": stop_reason or "max_steps",
        "best_step": best_step,
        "best_checkpoint": str(best_checkpoint),
        "best_screen_score": best_score,
        "full_evaluation": str(full_output),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    guard_log.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"guarded stage1 failed: {error}", file=sys.stderr, flush=True)
        raise
