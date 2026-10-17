"""Parallel fixed-slot driver for optimized V8 generation."""

# ruff: noqa: SLF001

from __future__ import annotations

import json

import generate_onpolicy_eef_low_stage_gated_dataset_v6_parallel as _parallel
import generate_onpolicy_eef_sustained_recovery_dataset_v8_optimized as optimized

_original_write_summary = _parallel._write_summary
_original_init_worker = _parallel._init_worker
_original_generate_slot = _parallel._generate_slot


def _init_worker(args_dict: dict) -> None:
    _parallel.v6 = optimized
    _original_init_worker(args_dict)


def _generate_slot(slot: int, retries: int, retry_start: int) -> dict:
    _parallel.v6 = optimized
    return _original_generate_slot(slot, retries, retry_start)


def _write_summary(args, elapsed_s: float) -> dict:
    summary = _original_write_summary(args, elapsed_s)
    summary["dataset_kind"] = optimized.DATASET_KIND
    summary["generator"] = "parallel_fixed_slot_v8_optimized"
    summary["generation_optimizations"] = {
        "final_slot_seed_prefilter": True,
        "combined_anchor_and_perturb_xy": True,
        "feedback_perturbation": True,
        "minimum_unique_recovery_rows": optimized.MIN_UNIQUE_RECOVERY_ROWS,
    }
    summary["training_contract"].update(
        {
            "minimum_unique_recovery_rows_per_episode": optimized.MIN_UNIQUE_RECOVERY_ROWS,
            "row_replication_allowed": False,
            "offset_directions": optimized.NUM_OFFSET_SECTORS,
            "low_height_only": True,
        }
    )
    (args.output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    _parallel.v6 = optimized
    _parallel._init_worker = _init_worker
    _parallel._generate_slot = _generate_slot
    _parallel._write_summary = _write_summary
    _parallel.main()


if __name__ == "__main__":
    main()
