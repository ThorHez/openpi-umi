#!/usr/bin/env python3
"""Unattended M5 -> guarded M6 -> full-suffix training and evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
M5_CONFIG = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5_mixed_cup0903"
M6_STAGE1_CONFIG = (
    "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_stage1_mixed_cup0903"
)
M6_FULL_CONFIG = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_mixed_cup0903"
OLD_DATASET = ROOT / "data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10"
NEW_DATASET = ROOT / "data/shellgame_real_cup0903_state_epfirst_action_currentrel_eef10"
OLD_LABELS = ROOT.parent / "labels_merged_306_degap.jsonl"
NEW_LABELS = ROOT.parent / "cup_0903/labels.jsonl"
OLD_MEMORY = ROOT / (
    "evaluation/shellgame_real/cup0903_mem_adapt_step500_old306_validation/memory_classifier.json"
)
NEW_MEMORY = ROOT / "evaluation/shellgame_real/cup0903_mem_adapt_step500_all100/memory_classifier.json"
OLD_H16_ACTION_CHECKPOINT = ROOT / (
    "checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_stage1/"
    "real306_m6_direction_stage1_frame241_dirloss010_b32_seed42_v1_best_direction/1199"
)
ADAPTED_MEMORY_CHECKPOINT = ROOT / (
    "checkpoints/pi0_mem_shellgame_real_relation_adapt_new75_old25/"
    "cup0903_new75_old25_relation_only_lr1e5_b32_seed42_v1/500"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="cup0903_mixed25_75_h16_seed42_v1")
    parser.add_argument("--m5-steps", type=int, default=1_000)
    parser.add_argument("--m6-stage1-max-steps", type=int, default=2_000)
    parser.add_argument("--m6-stage1-interval", type=int, default=250)
    parser.add_argument("--m6-full-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--port", type=int, default=18017)
    parser.add_argument(
        "--m6-only",
        action="store_true",
        help="Skip M5 training/gates and start directly from the configured H16 action + adapted MEM checkpoints.",
    )
    parser.add_argument("--new-direction-floor", type=float, default=0.80)
    parser.add_argument("--old-direction-floor", type=float, default=0.70)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def environment(*, eval_gpu: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0" if eval_gpu else "0,1,2,3,4,5,6,7",
            "PYTHONUNBUFFERED": "1",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
            "HF_HOME": str(ROOT / ".cache/shellgame_real_mixed/huggingface"),
            "HF_DATASETS_CACHE": str(ROOT / ".cache/shellgame_real_mixed/huggingface/datasets"),
            # Local validation talks to a temporary websocket server on this
            # machine.  Do not route that connection through the user's HTTP
            # or SOCKS proxy.
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return env


def run(command: list[str], log_path: Path, *, eval_gpu: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(eval_gpu=eval_gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def latest_checkpoint(config_name: str, exp_name: str) -> tuple[int, Path]:
    root = ROOT / "checkpoints" / config_name / exp_name
    candidates = sorted((int(path.name), path) for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    if not candidates:
        raise FileNotFoundError(f"No numeric checkpoint under {root}")
    return candidates[-1]


def snapshot(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=os.link, symlinks=True)


def prune_numeric_checkpoints(root: Path, keep_steps: set[int]) -> None:
    """Keep only explicitly needed checkpoints inside this run's experiment root."""
    if not root.exists():
        return
    for path in root.iterdir():
        if path.is_dir() and path.name.isdigit() and int(path.name) not in keep_steps:
            shutil.rmtree(path)


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def update_state(path: Path, **values) -> None:
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(values)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def m5_eval_commands(kind: str, checkpoint: Path, output_root: Path) -> list[tuple[list[str], Path]]:
    result = []
    for domain, dataset, labels, memory, offset in (
        ("old306", OLD_DATASET, OLD_LABELS, OLD_MEMORY, 0),
        ("cup0903", NEW_DATASET, NEW_LABELS, NEW_MEMORY, 306),
    ):
        output = output_root / f"{kind}_{domain}.json"
        script = (
            "scripts/mem/eval_shellgame_real_m5_oracle_action_probe.py"
            if kind == "oracle"
            else "scripts/mem/eval_shellgame_real_m5_memory_action_probe.py"
        )
        command = [
            str(ROOT / ".venv/bin/python"),
            script,
            "--config-kind",
            "mixed",
            "--episode-offset",
            str(offset),
            "--dataset",
            str(dataset),
            "--labels",
            str(labels),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ]
        if kind == "memory":
            command.extend(("--memory-results", str(memory)))
        result.append((command, output))
    return result


def direction_eval(
    checkpoint: Path,
    output: Path,
    *,
    domain: str,
    config_kind: str,
    episodes_per_class: int,
    samples_per_prompt: int,
) -> dict:
    if domain == "cup0903":
        dataset, labels, memory, offset = NEW_DATASET, NEW_LABELS, NEW_MEMORY, 306
    else:
        dataset, labels, memory, offset = OLD_DATASET, OLD_LABELS, OLD_MEMORY, 0
    command = [
        str(ROOT / ".venv/bin/python"),
        "scripts/mem/eval_shellgame_real_m6_direction_prompt.py",
        "--config-kind",
        config_kind,
        "--episode-offset",
        str(offset),
        "--dataset",
        str(dataset),
        "--labels",
        str(labels),
        "--memory-results",
        str(memory),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--samples-per-prompt",
        str(samples_per_prompt),
    ]
    if episodes_per_class:
        command.extend(("--episodes-per-class", str(episodes_per_class)))
    run(command, output.with_suffix(".log"), eval_gpu=True)
    return load_summary(output)


def wait_for_port(port: int, process: subprocess.Popen, timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Evaluation server exited with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"Evaluation server did not listen on port {port}")


def gradual_eval(checkpoint: Path, output_root: Path, port: int) -> None:
    server_log_path = output_root / "cached_server.log"
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    with server_log_path.open("a", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/mem/serve_shellgame_real_stage2_cached.py",
                "--checkpoint",
                str(checkpoint),
                "--port",
                str(port),
                "--prompt-from-memory",
            ],
            cwd=ROOT,
            env=environment(eval_gpu=True),
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_port(port, server)
            for domain, dataset, labels in (
                ("old306", OLD_DATASET, OLD_LABELS),
                ("cup0903", NEW_DATASET, NEW_LABELS),
            ):
                output = output_root / f"gradual_suffix_{domain}.json"
                run(
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
                        "--output",
                        str(output),
                    ],
                    output.with_suffix(".log"),
                    eval_gpu=True,
                )
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main() -> None:
    args = parse_args()
    if args.batch_size % 8:
        raise ValueError("batch-size must be divisible by 8")
    if args.m6_stage1_max_steps % args.m6_stage1_interval:
        raise ValueError("m6-stage1-max-steps must be divisible by m6-stage1-interval")
    required = (OLD_DATASET, NEW_DATASET, OLD_LABELS, NEW_LABELS, OLD_MEMORY, NEW_MEMORY)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prerequisites: {missing}")

    output_root = ROOT / "evaluation/shellgame_real" / args.run_name
    state_path = output_root / "pipeline_state.json"
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_root}; pass --overwrite or choose another run-name")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    first_stage = "m6_stage1" if args.m6_only else "m5_oracle"
    update_state(
        state_path,
        status="running",
        stage=first_stage,
        started_at=time.time(),
        m6_only=args.m6_only,
        action_checkpoint=str(OLD_H16_ACTION_CHECKPOINT),
        memory_checkpoint=str(ADAPTED_MEMORY_CHECKPOINT),
        new_direction_floor=args.new_direction_floor,
        old_direction_floor=args.old_direction_floor,
    )

    m5_checkpoints = {}
    for semantic_source in (() if args.m6_only else ("oracle", "memory")):
        exp_name = f"{args.run_name}_m5_{semantic_source}"
        update_state(state_path, stage=f"m5_{semantic_source}")
        run(
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/mem/train_shellgame_real_m5_action_probe_mixed_cup0903.py",
                "--semantic-source",
                semantic_source,
                "--exp-name",
                exp_name,
                "--steps",
                str(args.m5_steps),
                "--batch-size",
                str(args.batch_size),
                "--eval-batch-size",
                str(args.batch_size),
                "--fsdp-devices",
                "8",
                "--save-interval",
                "5000",
                "--overwrite",
            ],
            output_root / f"train_m5_{semantic_source}.log",
        )
        _, checkpoint = latest_checkpoint(M5_CONFIG, exp_name)
        m5_checkpoints[semantic_source] = str(checkpoint)
        eval_summaries = {}
        for command, output in m5_eval_commands(semantic_source, checkpoint, output_root / "m5_eval"):
            run(command, output.with_suffix(".log"), eval_gpu=True)
            eval_summaries[output.stem] = load_summary(output)
        update_state(state_path, m5_checkpoints=m5_checkpoints, **{f"m5_{semantic_source}_eval": eval_summaries})
        metric = (
            "counterfactual_forced_class_accuracy" if semantic_source == "oracle" else "action_follows_memory_accuracy"
        )
        if min(float(summary[metric]) for summary in eval_summaries.values()) < 0.65:
            update_state(state_path, status="stopped", stop_reason=f"m5_{semantic_source}_gate_failed")
            return

    update_state(state_path, stage="m6_stage1")
    stage1_exp = f"{args.run_name}_m6_stage1"
    stage1_best_root = ROOT / "checkpoints" / M6_STAGE1_CONFIG / f"{stage1_exp}_best_direction"
    best_score = -1.0
    best_step = None
    no_improve = 0
    stage1_history = []
    for target_step in range(args.m6_stage1_interval, args.m6_stage1_max_steps + 1, args.m6_stage1_interval):
        command = [
            str(ROOT / ".venv/bin/python"),
            "scripts/mem/train_shellgame_real_m6_direction_stage1_mixed_cup0903.py",
            "--exp-name",
            stage1_exp,
            "--steps",
            str(target_step),
            "--schedule-steps",
            str(args.m6_stage1_max_steps),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.batch_size),
            "--fsdp-devices",
            "8",
            "--eval-interval",
            str(args.m6_stage1_interval),
            "--save-interval",
            "5000",
            "--disable-direction-early-stop",
        ]
        command.append("--overwrite" if target_step == args.m6_stage1_interval else "--resume")
        run(command, output_root / "train_m6_stage1.log")
        saved_step, checkpoint = latest_checkpoint(M6_STAGE1_CONFIG, stage1_exp)
        summaries = {
            domain: direction_eval(
                checkpoint,
                output_root / "m6_stage1" / f"screen_step{saved_step}_{domain}.json",
                domain=domain,
                config_kind="mixed_stage1",
                episodes_per_class=2,
                samples_per_prompt=1,
            )
            for domain in ("old306", "cup0903")
        }
        source_scores = {
            domain: float(summary["counterfactual_prompt_following_accuracy"])
            for domain, summary in summaries.items()
        }
        score = min(source_scores.values())
        minimum = min(
            float(row["accuracy"])
            for summary in summaries.values()
            for row in summary["counterfactual_by_forced_prompt"].values()
        )
        all_three = min(
            float(summary["counterfactual_all_three_prompts_follow_episode_accuracy"])
            for summary in summaries.values()
        )
        meaningful = score >= best_score + 0.03
        if meaningful:
            best_score, best_step, no_improve = score, saved_step, 0
            snapshot(checkpoint, stage1_best_root / str(saved_step))
            prune_numeric_checkpoints(stage1_best_root, {saved_step})
        else:
            no_improve += 1
        stage1_history.append(
            {
                "step": saved_step,
                "score": score,
                "source_scores": source_scores,
                "min_class": minimum,
                "all_three": all_three,
                "best": best_score,
            }
        )
        update_state(state_path, m6_stage1_history=stage1_history, m6_stage1_best_step=best_step)
        prune_numeric_checkpoints(checkpoint.parent, {saved_step})
        if (
            saved_step + 1 >= 500
            and source_scores["cup0903"] >= args.new_direction_floor
            and source_scores["old306"] >= args.old_direction_floor
            and minimum >= 2 / 3
            and all_three >= 2 / 3
        ):
            break
        if saved_step + 1 >= 750 and no_improve >= 2 and best_score < 0.55:
            update_state(state_path, status="stopped", stop_reason="m6_stage1_direction_ineffective")
            return
    if best_step is None or best_score < 0.65:
        update_state(state_path, status="stopped", stop_reason="m6_stage1_direction_gate_failed")
        return
    stage1_best = stage1_best_root / str(best_step)
    for domain in ("old306", "cup0903"):
        summary = direction_eval(
            stage1_best,
            output_root / "m6_stage1" / f"best_full_{domain}.json",
            domain=domain,
            config_kind="mixed_stage1",
            episodes_per_class=0,
            samples_per_prompt=2,
        )
        update_state(state_path, **{f"m6_stage1_best_{domain}": summary})

    update_state(state_path, stage="m6_full")
    full_exp = f"{args.run_name}_m6_full"
    full_best_root = ROOT / "checkpoints" / M6_FULL_CONFIG / f"{full_exp}_best_direction"
    full_best_score = -1.0
    full_best_step = None
    full_history = []
    for target_step in range(5_000, args.m6_full_steps + 1, 5_000):
        command = [
            str(ROOT / ".venv/bin/python"),
            "scripts/mem/train_shellgame_real_m6_mixed_cup0903.py",
            "--exp-name",
            full_exp,
            "--checkpoint",
            str(stage1_best),
            "--steps",
            str(target_step),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.batch_size),
            "--fsdp-devices",
            "8",
            "--save-interval",
            "5000",
        ]
        command.append("--overwrite" if target_step == 5_000 else "--resume")
        run(command, output_root / "train_m6_full.log")
        saved_step, checkpoint = latest_checkpoint(M6_FULL_CONFIG, full_exp)
        summaries = {
            domain: direction_eval(
                checkpoint,
                output_root / "m6_full" / f"screen_step{saved_step}_{domain}.json",
                domain=domain,
                config_kind="mixed_full",
                episodes_per_class=2,
                samples_per_prompt=1,
            )
            for domain in ("old306", "cup0903")
        }
        source_scores = {
            domain: float(summary["counterfactual_prompt_following_accuracy"])
            for domain, summary in summaries.items()
        }
        score = min(source_scores.values())
        if score >= full_best_score:
            full_best_score, full_best_step = score, saved_step
            snapshot(checkpoint, full_best_root / str(saved_step))
            prune_numeric_checkpoints(full_best_root, {saved_step})
        full_history.append({"step": saved_step, "score": score, "source_scores": source_scores})
        update_state(
            state_path,
            m6_full_history=full_history,
            m6_full_best_step=full_best_step,
            m6_full_best_score=full_best_score,
        )
        prune_numeric_checkpoints(checkpoint.parent, {saved_step})
        if score < best_score - 0.15:
            break
    if full_best_step is None:
        raise RuntimeError("No full-suffix checkpoint was selected")
    full_best = full_best_root / str(full_best_step)
    for domain in ("old306", "cup0903"):
        summary = direction_eval(
            full_best,
            output_root / "m6_full" / f"best_full_{domain}.json",
            domain=domain,
            config_kind="mixed_full",
            episodes_per_class=0,
            samples_per_prompt=2,
        )
        update_state(state_path, **{f"m6_full_best_{domain}": summary})
    final_new = load_summary(output_root / "m6_full" / "best_full_cup0903.json")
    final_old = load_summary(output_root / "m6_full" / "best_full_old306.json")
    final_new_score = float(final_new["counterfactual_prompt_following_accuracy"])
    final_old_score = float(final_old["counterfactual_prompt_following_accuracy"])
    if final_new_score < args.new_direction_floor or final_old_score < args.old_direction_floor:
        update_state(
            state_path,
            status="stopped",
            stop_reason="m6_full_dual_domain_direction_gate_failed",
            best_checkpoint=str(full_best),
        )
        return
    gradual_eval(full_best, output_root / "m6_full", args.port)
    update_state(
        state_path,
        status="complete",
        stage="complete",
        completed_at=time.time(),
        best_checkpoint=str(full_best),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"mixed M5/M6 pipeline failed: {error}", file=sys.stderr, flush=True)
        raise
