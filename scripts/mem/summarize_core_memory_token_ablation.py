#!/usr/bin/env python3
"""Audit and summarize the matched Fig. 2 memory-token ablation."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (260951, 260952, 260953)
STEP = 2000
TAG = "260901_core_memory_tokens"
VARIANTS = {
    "token_carry_soft": (
        "robomme_core_memory_token_soft_seed{seed}_" + TAG
    ),
    "reset_token_soft": (
        "robomme_core_memory_token_reset_soft_seed{seed}_" + TAG
    ),
    "token_carry_unconditional": (
        "robomme_core_memory_token_unconditional_seed{seed}_" + TAG
    ),
    "token_carry_soft_no_trajectory_teacher": (
        "robomme_core_memory_token_no_trajectory_teacher_seed{seed}_" + TAG
    ),
}
# (128x64 token carry, soft gate, trajectory teacher)
EXPECTED = {
    "token_carry_soft": (True, True, True),
    "reset_token_soft": (False, True, True),
    "token_carry_unconditional": (True, False, True),
    "token_carry_soft_no_trajectory_teacher": (True, True, False),
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
            # This assertion is the key safeguard against repeating the earlier
            # causal-hidden-state experiment under a misleading name.
            if bool(config.get("causal_evidence_state", False)):
                raise ValueError(f"64D causal state is enabled in core run: {run}")
            actual = (
                bool(config.get("recurrent_memory_carry", True)),
                bool(config.get("write_gate", False)),
                bool(config.get("privileged_trajectory_teacher_used", True)),
            )
            if actual != EXPECTED[variant]:
                raise ValueError(
                    f"contract mismatch for {variant} seed {seed}: "
                    f"expected {EXPECTED[variant]}, got {actual}"
                )
            if int(config["steps"]) != STEP or int(config["seed"]) != seed:
                raise ValueError(f"step/seed mismatch in {run}")
            result_path = run / f"test_visual_dependence_step{STEP}.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint = str((run / str(STEP) / "params").resolve())
            if result["checkpoint"] != checkpoint:
                raise ValueError(
                    f"fixed-step checkpoint mismatch for {variant} seed {seed}: "
                    f"{result['checkpoint']} != {checkpoint}"
                )
            raw[variant][seed] = result
            audit[variant][seed] = {
                "training_dir": str(run.resolve()),
                "checkpoint": checkpoint,
                "contract": {
                    "memory_shape": [128, 64],
                    "memory_token_carry": actual[0],
                    "soft_write_gate": actual[1],
                    "trajectory_teacher": actual[2],
                    "causal_evidence_state": False,
                    "explicit_event_trigger": False,
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

    primary = "token_carry_soft"
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
        "experiment": "fig2_core_128x64_memory_token_ablation",
        "protocol": {
            "training_seeds": list(SEEDS),
            "fixed_checkpoint_step": STEP,
            "persistent_state": "128x64 memory tokens",
            "update_core": "siglip_mem_semantic.MemoryUpdateBlock",
            "causal_evidence_state": False,
            "explicit_event_trigger": False,
            "checkpoint_selection_used_for_table": False,
        },
        "variants": variants,
        "paired_primary_minus_comparison": paired,
        "audit": audit,
    }
    output = ROOT / "checkpoints/robomme_core_memory_token_ablation_3seed_step2000.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Variant | Field | State | Change | Hold | Final | Answer | Sequence")
    print("--- | --- | --- | --- | --- | --- | --- | ---")
    for variant in VARIANTS:
        normal = variants[variant]["modes"]["normal"]
        cells = [variant]
        for metric in METRICS:
            value = normal[metric]
            cells.append(f"{100 * value['mean']:.1f} +/- {100 * value['sample_sd']:.1f}")
        print(" | ".join(cells))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
