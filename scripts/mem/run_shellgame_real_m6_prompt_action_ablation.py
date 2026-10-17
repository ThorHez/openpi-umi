#!/usr/bin/env python3
"""Run paired prompt-only and prompt+MEM M6 training, evaluation, and comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
CONFIG_NAME = "pi0_mem_shellgame_real_m6_prompt_action_ablation_mixed"
MEMORY_ROOT = ROOT / (
    "checkpoints/pi0_mem_shellgame_real_fresh_memory_mild_all/"
    "freshmem_train383_val44_mildaug_officialbase_b32_seed42_split20260904_v1"
)
MEMORY_CHECKPOINT = MEMORY_ROOT / "1800"
SPLIT_MANIFEST = MEMORY_ROOT / "training_manifest.json"
OLD_DATASET = ROOT / "data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10"
NEW_DATASET = ROOT / "data/shellgame_real_cup0903_state_epfirst_action_currentrel_eef10"
OLD_LABELS = WORKSPACE / "labels_merged_306_degap.jsonl"
NEW_LABELS = WORKSPACE / "cup_0903/labels.jsonl"
MODES = ("prompt_only", "prompt_memory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="freshmem1800_prompt_only_vs_memory_seed42_v1")
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.batch_size % 4:
        parser.error("steps and batch-size must be positive; batch-size must be divisible by 4")
    if args.samples_per_prompt <= 0:
        parser.error("samples-per-prompt must be positive")
    return args


def environment(gpus: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpus,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": str(ROOT / ".cache/shellgame_prompt_ablation/huggingface"),
            "HF_DATASETS_CACHE": str(ROOT / ".cache/shellgame_prompt_ablation/huggingface/datasets"),
        }
    )
    return env


def update_state(path: Path, **values) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(values)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], log_path: Path, *, gpus: str) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(gpus),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def latest_checkpoint(exp_name: str) -> Path:
    root = ROOT / "checkpoints" / CONFIG_NAME / exp_name
    candidates = sorted((int(path.name), path) for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    if not candidates:
        raise FileNotFoundError(f"No checkpoint under {root}")
    return candidates[-1][1]


def weighted(summaries: dict[str, dict], metric: str) -> float:
    total = sum(int(summary["validation_episodes"]) for summary in summaries.values())
    return sum(int(summary["validation_episodes"]) * float(summary[metric]) for summary in summaries.values()) / total


def main() -> None:
    args = parse_args()
    required = (MEMORY_CHECKPOINT / "params", SPLIT_MANIFEST, OLD_DATASET, NEW_DATASET, OLD_LABELS, NEW_LABELS)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prerequisites: {missing}")

    output_root = ROOT / "evaluation/shellgame_real" / args.run_name
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "pipeline_state.json"
    experiment_names = {mode: f"{args.run_name}_{mode}" for mode in MODES}
    update_state(
        state_path,
        status="running",
        stage="paired_training",
        started_at=time.time(),
        shared_memory_checkpoint=str(MEMORY_CHECKPOINT),
        split_manifest=str(SPLIT_MANIFEST),
        experiment_names=experiment_names,
    )

    processes = {}
    logs = {}
    for mode, gpus in zip(MODES, ("0,1,2,3", "4,5,6,7"), strict=True):
        log_path = output_root / f"train_{mode}.log"
        logs[mode] = log_path.open("a", encoding="utf-8")
        command = [
            str(ROOT / ".venv/bin/python"),
            "scripts/mem/train_shellgame_real_m6_prompt_ablation.py",
            "--condition-mode",
            mode,
            "--exp-name",
            experiment_names[mode],
            "--checkpoint",
            str(MEMORY_CHECKPOINT),
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--fsdp-devices",
            "4",
            "--eval-interval",
            "100",
            "--eval-batches",
            "4",
            "--save-interval",
            "5000",
        ]
        if args.overwrite:
            command.append("--overwrite")
        logs[mode].write("$ " + " ".join(command) + "\n")
        logs[mode].flush()
        processes[mode] = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment(gpus),
            stdout=logs[mode],
            stderr=subprocess.STDOUT,
        )
    update_state(state_path, training_pids={mode: process.pid for mode, process in processes.items()})

    failures = {}
    for mode, process in processes.items():
        return_code = process.wait()
        logs[mode].close()
        if return_code:
            failures[mode] = return_code
    if failures:
        update_state(state_path, status="failed", stage="paired_training", failures=failures)
        raise RuntimeError(f"Paired training failed: {failures}")

    checkpoints = {mode: latest_checkpoint(experiment_names[mode]) for mode in MODES}
    update_state(
        state_path,
        stage="memory_validation",
        selected_checkpoints={mode: str(path) for mode, path in checkpoints.items()},
    )
    memory_results = {}
    for domain, dataset, labels, offset in (
        ("old306", OLD_DATASET, OLD_LABELS, 0),
        ("cup0903", NEW_DATASET, NEW_LABELS, 306),
    ):
        output = output_root / f"memory_{domain}.json"
        run(
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/mem/eval_shellgame_real_stage2_memory_classifier.py",
                "--dataset",
                str(dataset),
                "--labels",
                str(labels),
                "--config-kind",
                "fresh_memory",
                "--checkpoint",
                str(MEMORY_CHECKPOINT),
                "--split",
                "validation",
                "--split-manifest",
                str(SPLIT_MANIFEST),
                "--split-domain",
                domain,
                "--episode-offset",
                str(offset),
                "--output",
                str(output),
            ],
            output.with_suffix(".log"),
            gpus="0",
        )
        memory_results[domain] = output

    update_state(state_path, stage="action_ablation_evaluation")
    summaries: dict[str, dict[str, dict]] = {mode: {} for mode in MODES}
    for mode in MODES:
        for domain, dataset, labels, offset in (
            ("old306", OLD_DATASET, OLD_LABELS, 0),
            ("cup0903", NEW_DATASET, NEW_LABELS, 306),
        ):
            output = output_root / f"eval_{mode}_{domain}.json"
            run(
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/mem/eval_shellgame_real_m6_direction_prompt.py",
                    "--config-kind",
                    f"{mode}_ablation",
                    "--dataset",
                    str(dataset),
                    "--labels",
                    str(labels),
                    "--episode-offset",
                    str(offset),
                    "--split-manifest",
                    str(SPLIT_MANIFEST),
                    "--split-domain",
                    domain,
                    "--memory-results",
                    str(memory_results[domain]),
                    "--checkpoint",
                    str(checkpoints[mode]),
                    "--samples-per-prompt",
                    str(args.samples_per_prompt),
                    "--output",
                    str(output),
                ],
                output.with_suffix(".log"),
                gpus="0",
            )
            summaries[mode][domain] = json.loads(output.read_text(encoding="utf-8"))["summary"]

    aggregate = {}
    metrics = (
        "memory_accuracy",
        "deployment_action_follows_memory_prompt_accuracy",
        "deployment_action_ground_truth_accuracy",
        "oracle_prompt_action_ground_truth_accuracy",
        "counterfactual_prompt_following_accuracy",
        "counterfactual_all_three_prompts_follow_episode_accuracy",
        "mean_deployment_xyz_rmse_mm",
        "mean_oracle_prompt_xyz_rmse_mm",
        "mean_deployment_endpoint_xy_error_mm",
        "mean_oracle_prompt_endpoint_xy_error_mm",
        "mean_prompt_pairwise_endpoint_xy_delta_mm",
    )
    for mode in MODES:
        aggregate[mode] = {metric: weighted(summaries[mode], metric) for metric in metrics}
    prompt = aggregate["prompt_only"]
    baseline = aggregate["prompt_memory"]
    direction_tolerance = 0.02
    prompt_only_preferred = (
        prompt["oracle_prompt_action_ground_truth_accuracy"]
        >= baseline["oracle_prompt_action_ground_truth_accuracy"] - direction_tolerance
        and prompt["mean_oracle_prompt_endpoint_xy_error_mm"] < baseline["mean_oracle_prompt_endpoint_xy_error_mm"]
    )
    comparison = {
        "contract": ("paired seed/data/split/prompt/flow-loss/MEM checkpoint; only action_memory_injection differs"),
        "selected_checkpoints": {mode: str(path) for mode, path in checkpoints.items()},
        "per_domain": summaries,
        "aggregate": aggregate,
        "decision_rule": (
            "prefer prompt_only when oracle direction accuracy is within 2 percentage points of "
            "prompt_memory and oracle endpoint XY error is lower"
        ),
        "preferred_mode": "prompt_only" if prompt_only_preferred else "prompt_memory",
    }
    comparison_path = output_root / "ablation_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    update_state(
        state_path,
        status="complete",
        stage="complete",
        completed_at=time.time(),
        comparison=str(comparison_path),
        preferred_mode=comparison["preferred_mode"],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ABORTED: {error}", file=sys.stderr, flush=True)
        raise
