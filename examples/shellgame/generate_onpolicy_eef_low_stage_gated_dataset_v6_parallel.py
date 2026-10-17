"""Parallel fixed-slot driver for the V6 low-stage gated EEF dataset."""

# Private helpers are intentionally shared with the validated serial writer.
# ruff: noqa: SLF001

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import generate_onpolicy_eef_correction_dataset as legacy
import generate_onpolicy_eef_low_stage_gated_dataset_v6 as v6

_WORKER = {}


def _init_worker(args_dict: dict) -> None:
    args_dict = dict(args_dict)
    args_dict["output"] = Path(args_dict["output"])
    args = v6.argparse.Namespace(**args_dict)

    from openpi_client import websocket_client_policy

    original_append = legacy.base._append_observation
    visual_guard = v6._VisualGuard(original_append)
    legacy.base._append_observation = visual_guard
    legacy.base._policy_input = legacy.fixed_eef._fixed_history_policy_input
    shell = legacy.base._import_shellgame_tools(args.robosuite_root)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    _WORKER.update(
        args=args,
        shell=shell,
        client=client,
        policy_args=legacy._policy_args(args),
        visual_guard=visual_guard,
    )


def _generate_slot(slot: int, retries: int, retry_start: int) -> dict:
    args = _WORKER["args"]
    episode_dir = args.output / f"episode_{slot:06d}"
    if episode_dir.exists():
        return {"slot": slot, "accepted": True, "resumed": True, "audits": []}

    audits = []
    for retry in range(retry_start, retry_start + retries):
        attempt_index = slot + retry * args.num_episodes
        started = time.time()
        try:
            audit, payload = v6._attempt(
                _WORKER["shell"],
                _WORKER["client"],
                args,
                _WORKER["policy_args"],
                _WORKER["visual_guard"],
                attempt_index=attempt_index,
                accepted_index=slot,
            )
        except Exception as exc:
            audit = {
                "attempt_index": attempt_index,
                "accepted_index": slot,
                "reason": "worker_exception",
                "detail": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
            payload = None
        audit["accepted_index"] = slot
        audit["retry"] = retry
        audit["worker_elapsed_s"] = time.time() - started
        audits.append(audit)

        if payload is not None:
            if episode_dir.exists():
                raise FileExistsError(episode_dir)
            legacy._save_episode(
                episode_dir,
                observations=payload[0],
                actions=payload[1],
                action_mask=payload[2],
                phase_ids=payload[3],
                supervision_source=payload[4],
                metadata=payload[5],
                initial_ball_cup=payload[6],
                final_ball_cup=payload[7],
                fps=args.fps,
            )
            return {
                "slot": slot,
                "accepted": True,
                "resumed": False,
                "audits": audits,
            }

    return {
        "slot": slot,
        "accepted": False,
        "resumed": False,
        "audits": audits,
    }


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _write_summary(args, elapsed_s: float) -> dict:
    output = args.output
    episodes = sorted(output.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    metadata = [json.loads((episode / "metadata.json").read_text(encoding="utf-8")) for episode in episodes]
    manifest = output / "generation_manifest.jsonl"
    rows = []
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    accepted_rows = [row for row in rows if row.get("reason") == "accepted"]
    summary = {
        "dataset_kind": v6.DATASET_KIND,
        "generator": "parallel_fixed_slot_v6",
        "output": str(output),
        "requested_episodes": args.num_episodes,
        "accepted_episodes": len(episodes),
        "attempts": len(rows),
        "reasons": dict(Counter(row.get("reason", "unknown") for row in rows)),
        "manifest_accepted_records": len(accepted_rows),
        "resumed_preexisting_episodes": len(episodes) - len(accepted_rows),
        "anchor_stages": dict(Counter(item["anchor_stage"] for item in metadata)),
        "offset_bins": dict(Counter(item["perturbation"]["offset_bin"] for item in metadata)),
        "final_spatial_slots": dict(Counter(item["final_spatial_slot"] for item in metadata)),
        "offset_sectors": dict(Counter(str(item["offset_sector"]) for item in metadata)),
        "prefix_steps": dict(Counter(str(item["switch"]["prefix_steps"]) for item in metadata)),
        "measured_anchor_height_m": {
            stage: _stats(
                [
                    item["anchor_positioning"]["measured_height_above_grasp_m"]
                    for item in metadata
                    if item["anchor_stage"] == stage
                ]
            )
            for stage in v6.ANCHOR_BANDS_MM
        },
        "gated_open_steps": _stats([item["oracle"]["open_steps"] for item in metadata]),
        "workers": args.workers,
        "elapsed_s": elapsed_s,
        "training_contract": {
            "model_generated_actions_stored": False,
            "anchor_actions_stored": False,
            "perturbation_actions_stored": False,
            "supervised_action_source": "oracle_only",
            "z_and_gripper_gated_by_live_xy_error": True,
            "first_supervised_observation_frame": 60,
            "episode_frames": v6.EXPECTED_EPISODE_FRAMES,
        },
        "settings": {**vars(args), "output": str(output)},
    }
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    args = v6.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    v6._validate_args(args)
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

    retries = max(1, math.ceil(args.max_attempts / args.num_episodes))
    manifest_path = args.output / "generation_manifest.jsonl"
    retry_starts = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "accepted_index" in row and "retry" in row:
                slot = int(row["accepted_index"])
                retry_starts[slot] = max(retry_starts.get(slot, 0), int(row["retry"]) + 1)

    start_time = time.time()
    completed = len(existing_slots)
    failed_slots = []
    args_dict = {**vars(args), "output": str(args.output)}
    logging.info(
        "Starting %d V6 workers for %d missing slots (%d retries/slot, %d existing)",
        args.workers,
        len(missing_slots),
        retries,
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
        futures = {
            executor.submit(
                _generate_slot,
                slot,
                retries,
                retry_starts.get(slot, 0),
            ): slot
            for slot in missing_slots
        }
        for future in as_completed(futures):
            slot = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "slot": slot,
                    "accepted": False,
                    "audits": [
                        {
                            "accepted_index": slot,
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
            else:
                failed_slots.append(slot)
            logging.info(
                "slot=%d accepted=%s completed=%d/%d attempts=%d elapsed=%.1fmin",
                slot,
                result["accepted"],
                completed,
                args.num_episodes,
                len(result["audits"]),
                (time.time() - start_time) / 60.0,
            )

    summary = _write_summary(args, time.time() - start_time)
    if failed_slots:
        raise RuntimeError(
            f"Failed to fill {len(failed_slots)} slots after {retries} retries each; "
            "rerun with --resume and a larger --max-attempts. "
            f"First slots: {failed_slots[:20]}"
        )
    logging.info("Generation complete: %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
