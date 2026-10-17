#!/usr/bin/env python3
"""Guarded prompt-only continuation on every action-time frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.training.mem.recipes import shellgame_real_wrist_m6_prompt_ablation as _ablation  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_m6_prompt_only_full_suffix as _recipe  # noqa: E402
from scripts.mem import run_shellgame_real_m5_m6_mixed_cup0903 as _pipeline  # noqa: E402

OLD_DATASET = ROOT / "data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10"
NEW_DATASET = ROOT / "data/shellgame_real_cup0903_state_epfirst_action_currentrel_eef10"
OLD_LABELS = ROOT.parent / "labels_merged_306_degap.jsonl"
NEW_LABELS = ROOT.parent / "cup_0903/labels.jsonl"
MEMORY_RESULTS_ROOT = ROOT / "evaluation/shellgame_real/freshmem1800_prompt_only_vs_memory_seed42_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="freshmem1800_prompt_only800_fullsuffix_anchor30_seed42_v1")
    parser.add_argument("--checkpoint", type=Path, default=_recipe.DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--interval", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--anchor-fraction", type=float, default=0.30)
    parser.add_argument("--direction-drop-tolerance", type=float, default=0.10)
    parser.add_argument("--failure-patience", type=int, default=2)
    parser.add_argument("--port", type=int, default=18047)
    return parser.parse_args()


def direction_eval(checkpoint: Path, output: Path, *, domain: str, episodes_per_class: int, samples: int) -> dict:
    if domain == "old306":
        dataset, labels, memory, offset = OLD_DATASET, OLD_LABELS, MEMORY_RESULTS_ROOT / "memory_old306.json", 0
    else:
        dataset, labels, memory, offset = NEW_DATASET, NEW_LABELS, MEMORY_RESULTS_ROOT / "memory_cup0903.json", 306
    command = [
        str(ROOT / ".venv/bin/python"),
        "scripts/mem/eval_shellgame_real_m6_direction_prompt.py",
        "--config-kind",
        "prompt_only_ablation",
        "--dataset",
        str(dataset),
        "--labels",
        str(labels),
        "--memory-results",
        str(memory),
        "--checkpoint",
        str(checkpoint),
        "--episode-offset",
        str(offset),
        "--split-manifest",
        str(_ablation.MEMORY_SPLIT_MANIFEST),
        "--split-domain",
        domain,
        "--episodes-per-class",
        str(episodes_per_class),
        "--samples-per-prompt",
        str(samples),
        "--output",
        str(output),
    ]
    _pipeline.run(command, output.with_suffix(".log"), eval_gpu=True)
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def aggregate(summaries: dict[str, dict], metric: str) -> float:
    total = sum(int(summary["validation_episodes"]) for summary in summaries.values())
    return sum(int(summary["validation_episodes"]) * float(summary[metric]) for summary in summaries.values()) / total


def direction_metrics(summaries: dict[str, dict]) -> dict[str, float]:
    return {
        "counterfactual": aggregate(summaries, "counterfactual_prompt_following_accuracy"),
        "oracle": aggregate(summaries, "oracle_prompt_action_ground_truth_accuracy"),
        "deployment": aggregate(summaries, "deployment_action_follows_memory_prompt_accuracy"),
    }


def wait_for_port(port: int, process: subprocess.Popen, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Prompt-only server exited with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"Prompt-only server did not listen on {port}")


def gradual_eval(checkpoint: Path, output_root: Path, port: int) -> dict[str, dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    with (output_root / "cached_server.log").open("a", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/mem/serve_shellgame_real_m6_cached.py",
                "--checkpoint",
                str(checkpoint),
                "--condition-mode",
                "prompt_only",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=_pipeline.environment(eval_gpu=True),
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_port(port, server)
            for domain, dataset, labels, offset in (
                ("old306", OLD_DATASET, OLD_LABELS, 0),
                ("cup0903", NEW_DATASET, NEW_LABELS, 306),
            ):
                output = output_root / f"gradual_suffix_{domain}.json"
                _pipeline.run(
                    [
                        str(ROOT / ".venv/bin/python"),
                        "scripts/mem/eval_shellgame_real_stage2_checkpoint.py",
                        "--dataset",
                        str(dataset),
                        "--labels",
                        str(labels),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--episodes-per-class",
                        "2",
                        "--samples-per-frame",
                        "1",
                        "--split-manifest",
                        str(_ablation.MEMORY_SPLIT_MANIFEST),
                        "--split-domain",
                        domain,
                        "--episode-offset",
                        str(offset),
                        "--output",
                        str(output),
                    ],
                    output.with_suffix(".log"),
                    eval_gpu=True,
                )
                summaries[domain] = json.loads(output.read_text(encoding="utf-8"))["summary"]
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    return summaries


def gradual_score(summaries: dict[str, dict]) -> float:
    """Validation XYZ RMSE weighted by the number of evaluated suffix rows."""
    total = sum(int(summary["n_eval_rows"]) for summary in summaries.values())
    return sum(int(summary["n_eval_rows"]) * float(summary["xyz_rmse_mm"]) for summary in summaries.values()) / total


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.interval <= 0 or args.steps % args.interval:
        raise ValueError("steps must be positive and divisible by interval")
    if args.batch_size % 8:
        raise ValueError("batch-size must be divisible by 8")
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params not found: {args.checkpoint}")

    output_root = ROOT / "evaluation/shellgame_real" / args.run_name
    state_path = output_root / "pipeline_state.json"
    exp_name = f"{args.run_name}_fullsuffix"
    checkpoint_root = ROOT / "checkpoints" / _recipe.CONFIG_NAME / exp_name
    best_root = ROOT / "checkpoints" / _recipe.CONFIG_NAME / f"{exp_name}_best_guarded"
    if output_root.exists() or checkpoint_root.exists() or best_root.exists():
        raise FileExistsError("Run output/checkpoint already exists; choose a new run-name")
    output_root.mkdir(parents=True)
    _pipeline.update_state(
        state_path,
        status="running",
        stage="baseline_direction_eval",
        initialization_checkpoint=str(args.checkpoint),
        action_frames="all frame_index >= 241",
        direction_loss_weight=0.0,
        anchor_fraction=args.anchor_fraction,
        validation_interval=args.interval,
        started_at=time.time(),
    )

    baseline_summaries = {
        domain: direction_eval(
            args.checkpoint,
            output_root / "direction_eval" / f"baseline_{domain}.json",
            domain=domain,
            episodes_per_class=2,
            samples=1,
        )
        for domain in ("old306", "cup0903")
    }
    baseline = direction_metrics(baseline_summaries)
    _pipeline.update_state(state_path, baseline_direction_metrics=baseline, stage="full_suffix_training")

    history = []
    best_checkpoint = args.checkpoint
    best_step = 0
    best_gradual_score = float("inf")
    consecutive_failures = 0
    stopped_early = False
    for target_step in range(args.interval, args.steps + 1, args.interval):
        command = [
            str(ROOT / ".venv/bin/python"),
            "scripts/mem/train_shellgame_real_m6_prompt_only_full_suffix.py",
            "--exp-name",
            exp_name,
            "--checkpoint",
            str(args.checkpoint),
            "--steps",
            str(target_step),
            "--batch-size",
            str(args.batch_size),
            "--fsdp-devices",
            "8",
            "--anchor-fraction",
            str(args.anchor_fraction),
            "--save-interval",
            "5000",
            "--overwrite" if target_step == args.interval else "--resume",
        ]
        _pipeline.run(command, output_root / "train.log")
        saved_step, checkpoint = _pipeline.latest_checkpoint(_recipe.CONFIG_NAME, exp_name)
        summaries = {
            domain: direction_eval(
                checkpoint,
                output_root / "direction_eval" / f"step{saved_step}_{domain}.json",
                domain=domain,
                episodes_per_class=2,
                samples=1,
            )
            for domain in ("old306", "cup0903")
        }
        metrics = direction_metrics(summaries)
        passed = (
            metrics["counterfactual"] >= baseline["counterfactual"] - args.direction_drop_tolerance
            and metrics["oracle"] >= baseline["oracle"] - args.direction_drop_tolerance
        )
        suffix_summaries = None
        suffix_score = None
        if passed:
            suffix_summaries = gradual_eval(
                checkpoint,
                output_root / "gradual_eval" / f"step{saved_step}",
                args.port,
            )
            suffix_score = gradual_score(suffix_summaries)
            if suffix_score < best_gradual_score:
                best_checkpoint = best_root / str(saved_step)
                _pipeline.snapshot(checkpoint, best_checkpoint)
                _pipeline.prune_numeric_checkpoints(best_root, {saved_step})
                best_step = saved_step
                best_gradual_score = suffix_score
            consecutive_failures = 0
        else:
            consecutive_failures += 1
        history.append(
            {
                "step": saved_step,
                "direction_metrics": metrics,
                "passed_guard": passed,
                "gradual_validation_score_xyz_rmse_mm": suffix_score,
                "selected_best_step": best_step,
            }
        )
        _pipeline.update_state(
            state_path,
            history=history,
            best_step=best_step,
            best_checkpoint=str(best_checkpoint),
            best_gradual_validation_score_xyz_rmse_mm=(
                None if best_gradual_score == float("inf") else best_gradual_score
            ),
            consecutive_direction_failures=consecutive_failures,
        )
        _pipeline.prune_numeric_checkpoints(checkpoint_root, {saved_step})
        if consecutive_failures >= args.failure_patience:
            stopped_early = True
            break

    _pipeline.update_state(state_path, stage="full_direction_eval", stopped_early=stopped_early)
    final_direction = {
        domain: direction_eval(
            best_checkpoint,
            output_root / "direction_eval" / f"final_{domain}.json",
            domain=domain,
            episodes_per_class=0,
            samples=2,
        )
        for domain in ("old306", "cup0903")
    }
    _pipeline.update_state(state_path, final_direction_metrics=direction_metrics(final_direction), stage="gradual_eval")
    gradual_eval(best_checkpoint, output_root / "gradual_eval" / "final_selected", args.port)
    _pipeline.update_state(
        state_path,
        status="complete",
        stage="complete",
        completed_at=time.time(),
        best_step=best_step,
        best_checkpoint=str(best_checkpoint),
        stopped_early=stopped_early,
    )


if __name__ == "__main__":
    main()
