#!/usr/bin/env python3
"""Audit and summarize the matched single-latent RoboMME ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

SEEDS = (260971, 260972, 260973)
VARIANTS = (
    "latent_soft",
    "reset_soft",
    "latent_unconditional",
    "latent_soft_no_trajectory_teacher",
)
METRICS = (
    "field_accuracy",
    "state_exact_accuracy",
    "transition_state_exact_accuracy",
    "no_change_state_exact_accuracy",
    "final_state_exact_accuracy",
    "terminal_answer_exact_accuracy",
)
TASKS = (
    "videounmask_variable_demo",
    "videounmaskswap_local_event",
    "videoplaceorder_local_event",
    "pickxtimes_local_event",
)
INTERVENTION_MODES = (
    "normal",
    "zero_video",
    "reverse_chunks",
    "shuffle_episode_video",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--tag", default="260901_single_latent_confirm")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def expected_contract(variant: str) -> dict[str, Any]:
    return {
        "causal_evidence_state": False,
        "recurrent_memory_carry": variant != "reset_soft",
        "write_gate": variant != "latent_unconditional",
        "supervision_mode": (
            "terminal_answer_only"
            if variant == "latent_soft_no_trajectory_teacher"
            else "full"
        ),
        "checkpoint_selection_objective": "terminal_answer_then_final_state",
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    raw: dict[str, dict[int, dict[str, Any]]] = {}
    audit: dict[str, dict[int, dict[str, Any]]] = {}
    for variant in VARIANTS:
        raw[variant] = {}
        audit[variant] = {}
        contract = expected_contract(variant)
        for seed in SEEDS:
            run = root / "checkpoints" / f"robomme_single_latent_{variant}_seed{seed}_{args.tag}"
            result_path = run / "result.json"
            config_path = run / "training_config.json"
            if not result_path.exists() or not config_path.exists():
                raise FileNotFoundError(f"incomplete run: {run}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            mismatches = {
                key: {"expected": expected, "actual": config.get(key)}
                for key, expected in contract.items()
                if config.get(key) != expected
            }
            if mismatches:
                raise ValueError(f"contract mismatch for {variant} seed {seed}: {mismatches}")
            raw[variant][seed] = result
            audit[variant][seed] = {
                "run": str(run),
                "best_step": result["best_step"],
                "contract": contract,
            }

    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        subsets: dict[str, Any] = {}
        for subset in ("overall", *TASKS):
            subsets[subset] = {
                metric: stats(
                    [raw[variant][seed]["test"][subset][metric] for seed in SEEDS]
                )
                for metric in METRICS
            }
        variants[variant] = {
            "best_steps": [raw[variant][seed]["best_step"] for seed in SEEDS],
            "test": subsets,
        }

    paired: dict[str, Any] = {}
    for comparison in VARIANTS[1:]:
        paired[comparison] = {
            metric: stats(
                [
                    raw["latent_soft"][seed]["test"]["overall"][metric]
                    - raw[comparison][seed]["test"]["overall"][metric]
                    for seed in SEEDS
                ]
            )
            for metric in METRICS
        }

    intervention_raw: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        path = (
            root
            / "checkpoints"
            / f"robomme_single_latent_latent_soft_seed{seed}_{args.tag}"
            / "test_visual_dependence.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"missing visual intervention result: {path}")
        intervention_raw[seed] = json.loads(path.read_text(encoding="utf-8"))

    interventions: dict[str, Any] = {}
    for mode in INTERVENTION_MODES:
        interventions[mode] = {
            "overall": {
                metric: stats(
                    [
                        intervention_raw[seed]["modes"][mode]["overall"][metric]
                        for seed in SEEDS
                    ]
                )
                for metric in METRICS
            },
            "write_gate": {
                subset: stats(
                    [
                        intervention_raw[seed]["modes"][mode]["write_gate"][subset]
                        for seed in SEEDS
                    ]
                )
                for subset in ("all", "change", "hold")
            },
        }

    summary = {
        "schema_version": 2,
        "experiment": "single_persistent_128x64_latent_matched_ablation",
        "tag": args.tag,
        "protocol": {
            "training_seeds": list(SEEDS),
            "shared_checkpoint_per_seed": True,
            "tasks": list(TASKS),
            "external_causal_evidence_scan": False,
            "persistent_state": "128x64 output latent memory only",
            "checkpoint_selection": "dev terminal Answer, then final-state, then all-state",
            "variant_contracts": {variant: expected_contract(variant) for variant in VARIANTS},
        },
        "variants": variants,
        "paired_latent_soft_minus_comparison": paired,
        "visual_interventions_on_latent_soft": interventions,
        "action_promotion_decision": {
            "promote": False,
            "reason": (
                "zeroing, reversing, or replacing video does not reduce State or "
                "terminal Answer; current gains do not establish visual dependence"
            ),
        },
        "audit": audit,
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "checkpoints" / f"robomme_single_latent_ablation_3seed_{args.tag}.json"
    )
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    header = ("Variant", "Field", "State", "Change", "Hold", "Final", "Answer")
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for variant in VARIANTS:
        overall = variants[variant]["test"]["overall"]
        cells = [variant]
        for metric in METRICS:
            item = overall[metric]
            cells.append(f"{100 * item['mean']:.1f} ± {100 * item['sample_sd']:.1f}")
        print(" | ".join(cells))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
