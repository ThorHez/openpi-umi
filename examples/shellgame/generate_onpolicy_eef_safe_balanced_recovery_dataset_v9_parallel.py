"""Parallel fixed-design-slot driver for the V9 safe recovery dataset."""

# ruff: noqa: SLF001

from __future__ import annotations

from collections import Counter
from collections import deque
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
import json
import logging
import math
import multiprocessing as mp
import shutil
import time

import generate_onpolicy_eef_low_stage_gated_dataset_v6_parallel as _parallel
import generate_onpolicy_eef_safe_balanced_recovery_dataset_v9 as v9

_original_write_summary = _parallel._write_summary
_original_init_worker = _parallel._init_worker
_original_generate_slot = _parallel._generate_slot


def _init_worker(args_dict: dict) -> None:
    _parallel.v6 = v9
    _original_init_worker(args_dict)


def _generate_slot(slot: int, retries: int, retry_start: int) -> dict:
    _parallel.v6 = v9
    return _original_generate_slot(slot, retries, retry_start)


def _write_summary(args, elapsed_s: float) -> dict:
    summary = _original_write_summary(args, elapsed_s)
    rows = []
    manifest = args.output / "generation_manifest.jsonl"
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    physics_rows = [row for row in rows if row.get("physics_started", True)]
    summary["dataset_kind"] = v9.DATASET_KIND
    summary["generator"] = "parallel_fixed_slot_v9_safe_balanced"
    summary["generation_optimizations"] = {
        "final_slot_seed_prefilter": True,
        "fixed_design_slot_retry_until_filled": True,
        "safe_switch_path": "raise_then_lateral_at_clearance_then_vertical_descent",
        "minimum_open_recovery_steps": 0,
        "measured_sector_acceptance": True,
        "maximum_hidden_cup_displacement_m": v9.MAX_HIDDEN_CUP_DISPLACEMENT_M,
    }
    summary["physics_attempts"] = len(physics_rows)
    summary["physics_reasons"] = dict(
        Counter(row.get("reason", "unknown") for row in physics_rows)
    )
    summary["training_contract"].update(
        {
            "quota_complete_before_conversion": True,
            "row_replication_required": False,
            "offset_directions": v9.NUM_OFFSET_SECTORS,
            "descent_recovery_only": True,
            "precontact_switch_height_design": True,
            "low_stage_rows_provided_by_oracle_suffix": True,
            "hidden_safe_path_actions_stored": False,
            "minimum_initial_xy_error_m": v9.MIN_INITIAL_XY_ERROR_M,
        }
    )
    (args.output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    _parallel.v6 = v9
    _parallel._init_worker = _init_worker
    _parallel._generate_slot = _generate_slot
    _parallel._write_summary = _write_summary

    args = v9.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    v9._validate_args(args)
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.output.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if existing and not args.resume:
        raise FileExistsError(f"{args.output} already contains {len(existing)} episodes")
    existing_slots = {int(path.name.split("_")[-1]) for path in existing}
    invalid = sorted(slot for slot in existing_slots if not 0 <= slot < args.num_episodes)
    if invalid:
        raise ValueError(f"Output contains slots outside requested range: {invalid[:10]}")
    missing_slots = [slot for slot in range(args.num_episodes) if slot not in existing_slots]

    if not missing_slots:
        summary = _write_summary(args, 0.0)
        logging.info("Dataset already complete: %s", json.dumps(summary, sort_keys=True))
        return

    # A raw episode is about 13 MiB at 224x224.  Preserve enough room for the
    # subsequent LeRobot conversion plus an 8 GiB operational margin.  Refuse
    # to start a full run that is likely to fill the filesystem midway.
    if args.num_episodes == 1200:
        estimated_raw_bytes = len(missing_slots) * 14 * 1024**2
        conversion_and_margin_bytes = 18 * 1024**3
        free_bytes = shutil.disk_usage(args.output.parent).free
        required_bytes = estimated_raw_bytes + conversion_and_margin_bytes
        if free_bytes < required_bytes:
            raise RuntimeError(
                "Insufficient disk for V9 raw generation plus safe conversion reserve: "
                f"free={free_bytes / 1024**3:.1f} GiB, "
                f"required={required_bytes / 1024**3:.1f} GiB"
            )

    # Stream every attempt to the manifest immediately.  The older driver ran
    # all retries for a hard slot inside one worker and only flushed after that
    # slot finished, hiding slow or impossible designs for many minutes.
    retries_per_run = max(1, math.ceil(args.max_attempts / args.num_episodes))
    manifest_path = args.output / "generation_manifest.jsonl"
    retry_starts: dict[int, int] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "accepted_index" in row and "retry" in row:
                slot = int(row["accepted_index"])
                retry_starts[slot] = max(retry_starts.get(slot, 0), int(row["retry"]) + 1)

    work = deque((slot, retry_starts.get(slot, 0), 0) for slot in missing_slots)
    completed = len(existing_slots)
    failed_slots: list[int] = []
    start_time = time.time()
    args_dict = {**vars(args), "output": str(args.output)}
    logging.info(
        "Starting %d V9 workers for %d missing slots (%d streamed retries/slot, %d existing)",
        args.workers,
        len(missing_slots),
        retries_per_run,
        len(existing_slots),
    )

    context = mp.get_context("spawn")
    with (
        manifest_path.open("a", encoding="utf-8") as manifest,
        ProcessPoolExecutor(
            max_workers=min(args.workers, len(missing_slots)),
            mp_context=context,
            initializer=_init_worker,
            initargs=(args_dict,),
        ) as executor,
    ):
        active = {}
        while work or active:
            while work and len(active) < min(args.workers, len(missing_slots)):
                slot, retry, used = work.popleft()
                future = executor.submit(_generate_slot, slot, 1, retry)
                active[future] = (slot, retry, used)

            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                slot, retry, used = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "slot": slot,
                        "accepted": False,
                        "audits": [
                            {
                                "accepted_index": slot,
                                "retry": retry,
                                "reason": "future_exception",
                                "detail": f"{type(exc).__name__}: {exc}",
                            }
                        ],
                    }
                for audit in result["audits"]:
                    manifest.write(json.dumps(audit, sort_keys=True) + "\n")
                manifest.flush()

                if result["accepted"]:
                    completed += 1
                    logging.info(
                        "slot=%d accepted retry=%d completed=%d/%d elapsed=%.1fmin",
                        slot,
                        retry,
                        completed,
                        args.num_episodes,
                        (time.time() - start_time) / 60.0,
                    )
                elif used + 1 < retries_per_run:
                    work.append((slot, retry + 1, used + 1))
                else:
                    failed_slots.append(slot)
                    logging.error(
                        "slot=%d exhausted %d streamed retries; completed=%d/%d",
                        slot,
                        retries_per_run,
                        completed,
                        args.num_episodes,
                    )

    summary = _write_summary(args, time.time() - start_time)
    if failed_slots:
        raise RuntimeError(
            f"Failed to fill {len(failed_slots)} slots after {retries_per_run} retries in this run; "
            "rerun with --resume to add another retry budget. "
            f"First slots: {failed_slots[:20]}"
        )
    logging.info("Generation complete: %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
