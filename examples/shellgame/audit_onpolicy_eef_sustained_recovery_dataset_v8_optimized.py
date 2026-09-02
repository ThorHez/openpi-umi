"""Strict audit for optimized V8 sustained-recovery episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_onpolicy_eef_sustained_recovery_dataset_v8 as _audit
import generate_onpolicy_eef_sustained_recovery_dataset_v8_optimized as optimized


def audit(root: Path, *, expected_episodes: int | None = None) -> dict:
    _audit.v8 = optimized
    result = _audit.audit(root, expected_episodes=expected_episodes)
    result["generation_optimizations"] = {
        "final_slot_seed_prefilter": True,
        "combined_anchor_and_perturb_xy": True,
        "feedback_perturbation": True,
        "minimum_open_recovery_steps": optimized.MIN_UNIQUE_RECOVERY_ROWS,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.root.expanduser().resolve(), expected_episodes=args.expected_episodes),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
