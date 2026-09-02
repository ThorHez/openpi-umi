#!/usr/bin/env python3
"""Summarize the paired three-seed unified RoboMME memory ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = (
    "no_carry",
    "full",
    "no_causal_no_carry",
    "unconditional_no_carry",
)
PRIMARY_VARIANT = "no_carry"
SEEDS = (260951, 260952, 260953)
TASKS = (
    "pickxtimes_local_event",
    "videounmask_variable_demo",
    "videounmaskswap_local_event",
    "videoplaceorder_local_event",
)
METRICS = (
    "field_accuracy",
    "state_exact_accuracy",
    "transition_state_exact_accuracy",
    "no_change_state_exact_accuracy",
    "final_state_exact_accuracy",
    "terminal_answer_exact_accuracy",
    "sequence_exact_accuracy",
)


def _result_path(variant: str, seed: int, tag: str) -> Path:
    return (
        ROOT
        / "checkpoints"
        / f"robomme_unified_framework_{variant}_seed{seed}_{tag}"
        / "result.json"
    )


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": mean(values),
        "sample_sd": stdev(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="260901")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "checkpoints/robomme_unified_framework_ablation_3seed_260901.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw: dict[str, dict[int, dict[str, Any]]] = {}
    for variant in VARIANTS:
        raw[variant] = {}
        for seed in SEEDS:
            path = _result_path(variant, seed, args.tag)
            if not path.exists():
                raise FileNotFoundError(path)
            raw[variant][seed] = json.loads(path.read_text(encoding="utf-8"))
            answer_path = path.parent / "test_visual_dependence.json"
            if answer_path.exists():
                answer = json.loads(answer_path.read_text(encoding="utf-8"))["modes"]["normal"]
                for subset in ("overall", *TASKS):
                    raw[variant][seed]["test"][subset]["terminal_answer_exact_accuracy"] = (
                        answer[subset]["terminal_answer_exact_accuracy"]
                    )
            else:
                raise FileNotFoundError(answer_path)

    variants = {}
    for variant in VARIANTS:
        summaries = {}
        for subset in ("overall", *TASKS):
            summaries[subset] = {
                metric: _stats(
                    [
                        raw[variant][seed]["test"][subset][metric]
                        for seed in SEEDS
                    ]
                )
                for metric in METRICS
            }
        variants[variant] = {
            "best_steps": [raw[variant][seed]["best_step"] for seed in SEEDS],
            "test": summaries,
        }

    paired_deltas = {}
    for variant in VARIANTS:
        if variant == PRIMARY_VARIANT:
            continue
        paired_deltas[variant] = {
            metric: _stats(
                [
                    raw[PRIMARY_VARIANT][seed]["test"]["overall"][metric]
                    - raw[variant][seed]["test"]["overall"][metric]
                    for seed in SEEDS
                ]
            )
            for metric in METRICS
        }

    result = {
        "schema_version": 1,
        "training_seeds": list(SEEDS),
        "protocol": {
            "shared_checkpoint_per_seed": True,
            "shared_tasks": list(TASKS),
            "test_episodes": 60,
            "variant_definitions": {
                "no_carry": "primary: shared soft write + causal evidence; no separate latent-memory carry",
                "full": "primary plus recurrent latent-memory carry",
                "no_causal_no_carry": "primary without shared causal evidence state",
                "unconditional_no_carry": "primary with write gate fixed to one",
            },
            "terminal_answer_definition": (
                "exact final action query: Pick dynamic control tuple; queried color-to-region "
                "bindings for Unmask/Swap; queried ordinal-to-region for PlaceOrder"
            ),
        },
        "variants": variants,
        "primary_variant": PRIMARY_VARIANT,
        "paired_primary_minus_comparison": paired_deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for variant in VARIANTS:
        overall = variants[variant]["test"]["overall"]
        cells = []
        for metric in METRICS:
            item = overall[metric]
            cells.append(
                f"{metric}={100 * item['mean']:.1f}+-{100 * item['sample_sd']:.1f}"
            )
        print(f"{variant}: " + ", ".join(cells))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
