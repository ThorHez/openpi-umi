#!/usr/bin/env python3
"""Guarded three-fold cup_0904 MEM adaptation followed by a final all-data fit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "scripts/mem/train_shellgame_real_memory_robust_cup0904.py"
CONFIG_NAME = "pi0_mem_shellgame_real_relation_robust_cup0904"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="cup0904_fullframe_augconsistency_seed42_v1")
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cup0904-final-floor", type=float, default=0.80)
    parser.add_argument("--cup0904-relation-floor", type=float, default=0.85)
    parser.add_argument("--cup0904-aug-final-floor", type=float, default=0.70)
    parser.add_argument("--cup0903-final-floor", type=float, default=0.95)
    parser.add_argument("--old-final-floor", type=float, default=0.70)
    return parser.parse_args()


def write_state(path: Path, payload: dict) -> None:
    payload = {**payload, "updated_at_unix": time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def summary_path(exp_name: str) -> Path:
    return REPO / "checkpoints" / CONFIG_NAME / exp_name / "run_summary.json"


def run_one(exp_name: str, extra: list[str], log_path: Path) -> dict:
    summary = summary_path(exp_name)
    if summary.is_file():
        return json.loads(summary.read_text(encoding="utf-8"))
    command = [
        str(REPO / ".venv/bin/python"), "-u", str(TRAINER),
        "--exp-name", exp_name,
        *extra,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=REPO,
            env=os.environ.copy(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Training failed with exit code {result.returncode}; see {log_path}")
    if not summary.is_file():
        raise FileNotFoundError(f"Training returned successfully but did not write {summary}")
    return json.loads(summary.read_text(encoding="utf-8"))


def fold_passed(summary: dict, args: argparse.Namespace) -> tuple[bool, dict]:
    measured = {
        "cup0904_final": summary["cup0904_validation"]["final_accuracy"],
        "cup0904_relation": summary["cup0904_validation"]["relation_accuracy"],
        "cup0904_aug_final": summary["cup0904_augmented_validation"]["final_accuracy"],
        "cup0903_final": summary["cup0903_validation"]["final_accuracy"],
        "old_final": summary["old306_validation"]["final_accuracy"],
    }
    passed = (
        measured["cup0904_final"] >= args.cup0904_final_floor
        and measured["cup0904_relation"] >= args.cup0904_relation_floor
        and measured["cup0904_aug_final"] >= args.cup0904_aug_final_floor
        and measured["cup0903_final"] >= args.cup0903_final_floor
        and measured["old_final"] >= args.old_final_floor
    )
    return passed, measured


def main() -> None:
    args = parse_args()
    run_dir = REPO / "evaluation/shellgame_real" / args.run_name
    state_path = run_dir / "pipeline_state.json"
    fold_summaries = []
    write_state(state_path, {"status": "running", "phase": "fold0", "folds": []})
    for fold in range(3):
        exp_name = f"{args.run_name}_fold{fold}"
        write_state(state_path, {
            "status": "running", "phase": f"fold{fold}",
            "completed_folds": [item["fold"] for item in fold_summaries],
        })
        summary = run_one(
            exp_name,
            [
                "--fold", str(fold), "--num-train-steps", str(args.steps),
                "--eval-interval", str(args.eval_interval), "--seed", str(args.seed),
            ],
            run_dir / f"train_fold{fold}.log",
        )
        passed, measured = fold_passed(summary, args)
        fold_summaries.append({"fold": fold, "passed": passed, "measured": measured, "summary": str(summary_path(exp_name))})
        write_state(state_path, {
            "status": "running" if passed else "stopped_by_gate",
            "phase": f"fold{fold}_complete", "folds": fold_summaries,
        })
        if not passed:
            print(f"PIPELINE_STOP fold={fold} metrics={json.dumps(measured, sort_keys=True)}", flush=True)
            return

    final_steps = max(100, int(statistics.median(
        json.loads(Path(item["summary"]).read_text(encoding="utf-8"))["selected_step"]
        for item in fold_summaries
    )))
    final_exp = f"{args.run_name}_all21_final"
    write_state(state_path, {
        "status": "running", "phase": "all21_final_fit", "folds": fold_summaries,
        "final_fit_steps": final_steps,
    })
    final_summary = run_one(
        final_exp,
        [
            "--fold", "all", "--select-last", "--num-train-steps", str(final_steps),
            "--eval-interval", str(final_steps), "--early-stop-min-step", str(final_steps + 1),
            "--seed", str(args.seed),
        ],
        run_dir / "train_all21_final.log",
    )
    write_state(state_path, {
        "status": "complete", "phase": "complete", "folds": fold_summaries,
        "final_fit_steps": final_steps,
        "final_checkpoint": final_summary["checkpoint"],
        "final_summary": str(summary_path(final_exp)),
    })
    print(f"PIPELINE_COMPLETE checkpoint={final_summary['checkpoint']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PIPELINE_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
