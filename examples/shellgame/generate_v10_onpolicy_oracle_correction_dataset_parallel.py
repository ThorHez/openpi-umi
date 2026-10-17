"""Parallel fixed-slot driver for real V10 on-policy correction data."""

# ruff: noqa: SLF001

from __future__ import annotations

from collections import Counter
import json
import logging
import multiprocessing as mp
from pathlib import Path

import generate_onpolicy_eef_low_stage_gated_dataset_v6_parallel as parallel
import generate_onpolicy_eef_safe_balanced_recovery_dataset_v9_parallel as stream
import generate_v10_onpolicy_oracle_correction_dataset as v10_data


def _init_worker(args_dict: dict) -> None:
    """Spawn-safe backend selection (parent-process monkeypatches do not copy)."""
    args_dict = dict(args_dict)
    ports = [int(item.strip()) for item in str(args_dict.get("ports", "")).split(",") if item.strip()]
    if ports:
        identity = mp.current_process()._identity  # noqa: SLF001
        worker_index = (identity[0] - 1) if identity else 0
        args_dict["port"] = ports[worker_index % len(ports)]
    parallel.v6 = v10_data
    stream._original_init_worker(args_dict)


def _generate_slot(slot: int, retries: int, retry_start: int) -> dict:
    parallel.v6 = v10_data
    return stream._original_generate_slot(slot, retries, retry_start)


def _write_summary(args, elapsed_s: float) -> dict:
    episodes = sorted(args.output.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    metadata = [
        json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        for episode in episodes
    ]
    manifest_path = args.output / "generation_manifest.jsonl"
    rows = []
    if manifest_path.exists():
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = {
        "dataset_kind": v10_data.DATASET_KIND,
        "generator": "parallel_fixed_slot_real_v10_onpolicy",
        "output": str(args.output),
        "requested_episodes": args.num_episodes,
        "accepted_episodes": len(episodes),
        "attempts": len(rows),
        "physics_attempts": sum(row.get("physics_started", True) for row in rows),
        "reasons": dict(Counter(row.get("reason", "unknown") for row in rows)),
        "final_spatial_slots": dict(
            Counter(item["final_spatial_slot"] for item in metadata)
        ),
        "switch_offset_mm": _stats(
            [item["switch"]["offset_m"] * 1_000.0 for item in metadata]
        ),
        "switch_safe_height_mm": _stats(
            [item["switch"]["safe_height_m"] * 1_000.0 for item in metadata]
        ),
        "policy_prefix_steps": dict(
            Counter(str(item["model_prefix"]["executed_steps"]) for item in metadata)
        ),
        "workers": args.workers,
        "elapsed_s": elapsed_s,
        "training_contract": {
            "real_v10_closed_loop_switch_state": True,
            "hidden_perturbation_used": False,
            "model_generated_actions_stored": False,
            "model_generated_actions_supervised": False,
            "first_supervised_observation_frame": 60,
            "first_training_pair": "observation[60] -> oracle_action[61]",
            "supervised_action_source": "oracle_only",
            "balanced_final_spatial_slot": True,
        },
        "settings": {**vars(args), "output": str(args.output)},
    }
    (args.output / "generation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    values = sorted(values)
    return {
        "min": values[0],
        "mean": sum(values) / len(values),
        "median": values[len(values) // 2],
        "max": values[-1],
    }


def main() -> None:
    # Reuse the streamed retry scheduler, while replacing all dataset-specific
    # hooks.  This keeps a slow or rejected slot from hiding progress.
    stream.v9 = v10_data
    stream._init_worker = _init_worker
    stream._generate_slot = _generate_slot
    stream._write_summary = _write_summary
    parallel.v6 = v10_data
    logging.basicConfig(level=logging.INFO, force=True)
    stream.main()


if __name__ == "__main__":
    main()
