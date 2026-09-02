#!/usr/bin/env python3
"""Merge FrameSamp->V10 ShellGame closed-loop shards with statistical checks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


SLOTS = ("left", "middle", "right")
DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi0_shellgame_framesamp_v10_action_adapter_eef7_v1/"
    "framesamp_modul_step9999_v10_adapter_nominal5000_b8_s500_260828/499"
)


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [center - radius, center + radius]


def _count(records: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row[key]) for row in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = (args.output or root / "result.json").expanduser().resolve()
    shard_paths = sorted(root.glob("shard_*/result.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard results below {root}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    records = [row for payload in payloads for row in payload["records"]]
    episodes = [int(row["episode"]) for row in records]
    if len(records) != 100 or len(set(episodes)) != len(episodes):
        raise ValueError(f"Expected 100 unique episodes, got {len(records)}/{len(set(episodes))}")
    if any(row.get("condition") != "direct_visual" for row in records):
        raise ValueError("Formal FrameSamp evaluation expects only direct_visual records")

    records.sort(key=lambda row: int(row["episode"]))
    targets = [str(row["target_cup_identity_scoring_only"]) for row in records]
    selected = [str(row["selected_cup_identity"]) for row in records]
    target_distribution = dict(Counter(targets))
    if target_distribution != {"left": 34, "middle": 33, "right": 33}:
        raise ValueError(f"Unexpected target balance: {target_distribution}")

    confusion = {
        target: {
            prediction: sum(
                row["target_cup_identity_scoring_only"] == target
                and row["selected_cup_identity"] == prediction
                for row in records
            )
            for prediction in SLOTS
        }
        for target in SLOTS
    }
    per_target = {}
    for target in SLOTS:
        subset = [row for row in records if row["target_cup_identity_scoring_only"] == target]
        per_target[target] = {
            "episodes": len(subset),
            "cup_selection_correct": _count(subset, "cup_selection_correct"),
            "correct_selection_and_contact": _count(subset, "correct_selection_and_contact"),
            "target_cup_contact": _count(subset, "target_cup_contact"),
            "target_lift_success": _count(subset, "success"),
        }

    selection = _count(records, "cup_selection_correct")
    correct_contact = _count(records, "correct_selection_and_contact")
    target_contact = _count(records, "target_cup_contact")
    any_contact = _count(records, "any_cup_contact")
    lift = _count(records, "success")
    min_target_xy = np.asarray([row["min_target_xy_m"] for row in records], dtype=np.float64)
    max_target_lift = np.asarray([row["max_target_lift_m"] for row in records], dtype=np.float64)
    inference_ms = np.asarray([row["mean_inference_ms"] for row in records], dtype=np.float64)
    binomial = stats.binomtest(selection, len(records), p=1.0 / 3.0, alternative="two-sided")
    versus_no_mem = stats.fisher_exact([[selection, 100 - selection], [35, 65]])

    summary = {
        "episodes": len(records),
        "cup_selection_correct": selection,
        "cup_selection_accuracy": selection / len(records),
        "cup_selection_wilson_95": _wilson(selection, len(records)),
        "correct_selection_and_contact": correct_contact,
        "correct_selection_and_contact_rate": correct_contact / len(records),
        "target_cup_contact": target_contact,
        "target_cup_contact_rate": target_contact / len(records),
        "any_cup_contact": any_contact,
        "any_cup_contact_rate": any_contact / len(records),
        "target_lift_success": lift,
        "target_lift_success_rate": lift / len(records),
        "target_lift_wilson_95": _wilson(lift, len(records)),
        "target_approach_within_60mm": int(np.sum(min_target_xy <= 0.06)),
        "target_precision_within_30mm": int(np.sum(min_target_xy <= 0.03)),
        "mean_min_target_xy_m": float(np.mean(min_target_xy)),
        "median_min_target_xy_m": float(np.median(min_target_xy)),
        "mean_max_target_lift_m": float(np.mean(max_target_lift)),
        "mean_policy_inference_ms": float(np.mean(inference_ms)),
        "target_distribution": target_distribution,
        "selected_distribution": dict(Counter(selected)),
        "confusion_target_by_selected": confusion,
        "per_target": per_target,
        "random_one_third_two_sided_binomial_p": float(binomial.pvalue),
        "descriptive_fisher_p_vs_prior_no_mem_35_of_100": float(versus_no_mem.pvalue),
    }
    merged = {
        "schema_version": 1,
        "experiment": "MME FrameSamp memory -> trained interface -> frozen V10 action closed loop",
        # Early shards inherited a legacy evaluator's descriptive checkpoint
        # default. The server command and this canonical merged record use the
        # actual adapter checkpoint below.
        "checkpoint": str(DEFAULT_CHECKPOINT.resolve()),
        "framesamp_bank": payloads[0]["framesamp_bank"],
        "shards": [str(path) for path in shard_paths],
        "protocol": {
            **payloads[0]["control"],
            "episode_split": "100 class-balanced episodes from seed42 held-out adapter validation set",
            "deterministic_diffusion_noise": True,
            "noise_salt": payloads[0]["noise_salt"],
            "videos_saved": True,
        },
        "model_contract": {
            "framesamp_memory_shape": [512, 1024],
            "v10_action_parameters_updated": False,
            "parallel_action_interface_updated": True,
            "old_v10_memory_condition_strength": 0.0,
        },
        "summary": summary,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
