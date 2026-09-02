#!/usr/bin/env python3
"""Summarize the fixed-step recurrent-hidden-state RoboMME ablation."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (260951, 260952, 260953)
STEP = 2000
VARIANTS = {
    "causal_hidden_soft": "robomme_unified_framework_no_carry_seed{seed}_260901",
    "reset_hidden_soft": "robomme_unified_framework_no_causal_no_carry_seed{seed}_260901",
    "causal_hidden_unconditional": (
        "robomme_unified_framework_unconditional_no_carry_seed{seed}_260901"
    ),
    "causal_hidden_soft_no_trajectory_teacher": (
        "robomme_unified_framework_no_trajectory_teacher_seed{seed}_"
        "260901_correct_hidden_state"
    ),
}
EXPECTED = {
    "causal_hidden_soft": (True, False, True, True),
    "reset_hidden_soft": (False, False, True, True),
    "causal_hidden_unconditional": (True, False, False, True),
    "causal_hidden_soft_no_trajectory_teacher": (True, False, True, False),
}
METRICS = (
    "field_accuracy",
    "state_exact_accuracy",
    "transition_state_exact_accuracy",
    "no_change_state_exact_accuracy",
    "final_state_exact_accuracy",
    "terminal_answer_exact_accuracy",
    "sequence_exact_accuracy",
)
MODES = ("normal", "zero_video", "reverse_chunks", "shuffle_episode_video")


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def main() -> None:
    raw: dict[str, dict[int, dict[str, Any]]] = {}
    audit: dict[str, dict[int, dict[str, Any]]] = {}
    for variant, pattern in VARIANTS.items():
        raw[variant] = {}
        audit[variant] = {}
        for seed in SEEDS:
            run = ROOT / "checkpoints" / pattern.format(seed=seed)
            config = json.loads((run / "training_config.json").read_text(encoding="utf-8"))
            result_path = run / f"test_visual_dependence_step{STEP}.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            actual = (
                bool(config.get("causal_evidence_state", False)),
                bool(config.get("recurrent_memory_carry", True)),
                bool(config.get("write_gate", False)),
                bool(config.get("privileged_trajectory_teacher_used", True)),
            )
            if actual != EXPECTED[variant]:
                raise ValueError(
                    f"contract mismatch for {variant} seed {seed}: "
                    f"expected {EXPECTED[variant]}, got {actual}"
                )
            checkpoint = str((run / str(STEP) / "params").resolve())
            if result["checkpoint"] != checkpoint:
                raise ValueError(
                    f"fixed-step checkpoint mismatch for {variant} seed {seed}: "
                    f"{result['checkpoint']} != {checkpoint}"
                )
            raw[variant][seed] = result
            audit[variant][seed] = {
                "training_dir": str(run),
                "checkpoint": checkpoint,
                "contract": {
                    "causal_hidden_state": actual[0],
                    "output_latent_carry": actual[1],
                    "soft_write_gate": actual[2],
                    "privileged_trajectory_teacher": actual[3],
                },
            }

    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        modes: dict[str, Any] = {}
        for mode in MODES:
            modes[mode] = {
                metric: stats(
                    [
                        raw[variant][seed]["modes"][mode]["overall"][metric]
                        for seed in SEEDS
                    ]
                )
                for metric in METRICS
            }
            modes[mode]["write_gate"] = {
                subset: stats(
                    [
                        raw[variant][seed]["modes"][mode]["write_gate"][subset]
                        for seed in SEEDS
                    ]
                )
                for subset in ("all", "change", "hold")
            }
        variants[variant] = {"modes": modes}

    primary = "causal_hidden_soft"
    paired = {
        variant: {
            metric: stats(
                [
                    raw[primary][seed]["modes"]["normal"]["overall"][metric]
                    - raw[variant][seed]["modes"]["normal"]["overall"][metric]
                    for seed in SEEDS
                ]
            )
            for metric in METRICS
        }
        for variant in VARIANTS
        if variant != primary
    }
    summary = {
        "schema_version": 1,
        "experiment": "correct_recurrent_hidden_state_fixed_step_ablation",
        "protocol": {
            "training_seeds": list(SEEDS),
            "fixed_checkpoint_step": STEP,
            "persistent_state": "learned causal hidden state",
            "action_memory_readout": "128x64 output tokens; not recurrently carried",
            "checkpoint_selection_used_for_table": False,
        },
        "variants": variants,
        "paired_primary_minus_comparison": paired,
        "audit": audit,
    }
    output = ROOT / "checkpoints/robomme_correct_hidden_state_ablation_3seed_step2000.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Variant | Field | State | Change | Hold | Final | Answer | Sequence")
    print("--- | --- | --- | --- | --- | --- | --- | ---")
    for variant in VARIANTS:
        normal = variants[variant]["modes"]["normal"]
        cells = [variant]
        for metric in METRICS:
            value = normal[metric]
            cells.append(f"{100 * value['mean']:.1f} ± {100 * value['sample_sd']:.1f}")
        print(" | ".join(cells))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
