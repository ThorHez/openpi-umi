"""Parallel fixed-slot driver for the V8 sustained-recovery dataset."""

# Private driver hooks are dependency-injected to preserve the proven V6
# multiprocessing writer while keeping V6 source files unchanged.
# ruff: noqa: I001, SLF001

from __future__ import annotations

import json

import generate_onpolicy_eef_low_stage_gated_dataset_v6_parallel as _parallel
import generate_onpolicy_eef_sustained_recovery_dataset_v8 as v8


_original_write_summary = _parallel._write_summary
_original_init_worker = _parallel._init_worker
_original_generate_slot = _parallel._generate_slot


def _init_worker(args_dict: dict) -> None:
    # Spawn imports the V6 driver afresh, so dependency injection must also be
    # performed inside every worker process.
    v8._configure_v6_backend()
    _parallel.v6 = v8
    _original_init_worker(args_dict)


def _generate_slot(slot: int, retries: int, retry_start: int) -> dict:
    _parallel.v6 = v8
    return _original_generate_slot(slot, retries, retry_start)


def _write_summary(args, elapsed_s: float) -> dict:
    summary = _original_write_summary(args, elapsed_s)
    summary["dataset_kind"] = v8.DATASET_KIND
    summary["generator"] = "parallel_fixed_slot_v8"
    summary["training_contract"].update(
        {
            "minimum_unique_recovery_rows_per_episode": v8.MIN_UNIQUE_RECOVERY_ROWS,
            "row_replication_allowed": False,
            "offset_directions": v8.NUM_OFFSET_SECTORS,
            "low_height_only": True,
        }
    )
    (args.output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    # The V6 driver is intentionally dependency-injected with the V8 contract;
    # worker processes import this module through the spawn initializer.
    v8._configure_v6_backend()
    _parallel.v6 = v8
    _parallel._init_worker = _init_worker
    _parallel._generate_slot = _generate_slot
    _parallel._write_summary = _write_summary
    _parallel.main()


if __name__ == "__main__":
    main()
