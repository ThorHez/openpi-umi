"""Aggregate the matched three-seed teacher-memory ablation."""

from __future__ import annotations

import ast
import csv
import json
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "evaluation/shellgame/teacher_memory_necessity_12f_3seed_260831"
SEED42_RESULT = ROOT / "evaluation/shellgame/teacher_memory_necessity_12f_260826/result.json"
SEEDS = (42, 43, 44)
VARIANTS = ("state_only", "state_plus_distill")
METRICS = ("update_1", "update_2", "final", "mean")


def _canonical_metrics(metrics: dict[str, float]) -> dict[str, float]:
    updates = [float(metrics[f"val/slot_{index}_accuracy"]) for index in range(3)]
    return {
        "update_1": updates[0],
        "update_2": updates[1],
        "final": updates[2],
        "mean": float(metrics["val/stage_memory_accuracy"]),
        "stage_cross_entropy": float(metrics["val/stage_memory_loss"]),
        "memory_cosine_loss": float(metrics["val/memory_cosine_loss"]),
    }


def _load_seed42() -> dict[str, dict[str, float]]:
    source = json.loads(SEED42_RESULT.read_text(encoding="utf-8"))["runs"]
    result = {}
    for variant, source_key in (
        ("state_only", "A_state_only"),
        ("state_plus_distill", "B_state_plus_teacher"),
    ):
        metrics = source[source_key]["final_validation"]
        updates = [float(value) for value in metrics["slot_accuracy"]]
        result[variant] = {
            "update_1": updates[0],
            "update_2": updates[1],
            "final": updates[2],
            "mean": float(metrics["stage_accuracy"]),
            "stage_cross_entropy": float(metrics["stage_cross_entropy"]),
            "memory_cosine_loss": float(
                metrics[f"memory_cosine_loss{'_diagnostic_only' if variant == 'state_only' else ''}"]
            ),
        }
    return result


def _load_log(seed: int, variant: str) -> dict[str, float]:
    exp_name = f"teacher_necessity_12f_{variant}_seed{seed}_260831"
    checkpoint = (
        ROOT
        / "checkpoints/shellgame_qwen_distilled_direct_visual_recurrent_memory_probe"
        / exp_name
        / "999/_CHECKPOINT_METADATA"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Completed checkpoint not found: {checkpoint}")
    log_path = RESULT_ROOT / "logs" / f"{exp_name}.log"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Final semantic-memory validation: (\{.*?\}); best val/loss=", text)
    if not matches:
        raise ValueError(f"Final validation metrics not found in {log_path}")
    return _canonical_metrics(ast.literal_eval(matches[-1]))


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
    }


def main() -> None:
    seed42 = _load_seed42()
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for variant in VARIANTS:
            per_seed[str(seed)][variant] = seed42[variant] if seed == 42 else _load_log(seed, variant)

    aggregate = {}
    for variant in VARIANTS:
        aggregate[variant] = {
            metric: _summary([per_seed[str(seed)][variant][metric] for seed in SEEDS])
            for metric in (*METRICS, "stage_cross_entropy", "memory_cosine_loss")
        }
    aggregate["paired_teacher_gain"] = {
        metric: _summary(
            [
                per_seed[str(seed)]["state_plus_distill"][metric] - per_seed[str(seed)]["state_only"][metric]
                for seed in SEEDS
            ]
        )
        for metric in METRICS
    }

    result = {
        "schema_version": 1,
        "experiment": "teacher_memory_necessity_12f_3seed_260831",
        "seeds": list(SEEDS),
        "split_seed": 42,
        "validation_samples_per_seed_and_variant": 240,
        "training_steps": 1000,
        "batch_size": 12,
        "only_changed_factor_within_each_seed": "teacher latent-memory loss weight: state_only=0, state_plus_distill=1",
        "dispersion": "sample standard deviation across training seeds",
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    with (RESULT_ROOT / "per_seed.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("seed", "variant", *METRICS, "stage_cross_entropy", "memory_cosine_loss")
        )
        writer.writeheader()
        for seed in SEEDS:
            for variant in VARIANTS:
                writer.writerow({"seed": seed, "variant": variant, **per_seed[str(seed)][variant]})

    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
