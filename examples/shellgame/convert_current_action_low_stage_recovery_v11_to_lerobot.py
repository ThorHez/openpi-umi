"""Strictly audit and convert current-action V11 coherent recovery data."""

# ruff: noqa: E402, SLF001

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

ROBOSUITE_SCRIPTS = Path(__file__).resolve().parents[3] / "robosuite/robosuite/scripts"
sys.path.insert(0, str(ROBOSUITE_SCRIPTS))

import convert_shellgame_to_lerobot_raw_action as raw
import generate_current_action_low_stage_recovery_v11_parallel as generator

EXPECTED_EPISODES = 306
EXPECTED_FRAMES = 156
FIRST_ELIGIBLE = 60
LAST_ELIGIBLE = 154
ACTION_HORIZON = 16
OBSERVE_TASK = "Observe the ball moving under a cup and remember which cup contains it."
GRASP_TASK = "The shell game has ended. Grasp and lift the cup containing the ball."
MEMORY_BANK_NAME = "frozen_direct_visual_memory_v11.npz"


def _audit_raw(paths: list[Path]) -> dict:
    bins = Counter()
    sources = set()
    offsets = []
    heights = []
    open_steps = []
    max_grasp_lift_xy = []
    excluded = set(generator.DEFAULT_EXCLUDED_EPISODES)
    for path in paths:
        episode_dir = path.parent
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("dataset_kind") != generator.DATASET_KIND:
            raise RuntimeError(f"{episode_dir}: wrong dataset_kind")
        source_episode = int(metadata["source_episode"])
        if source_episode in sources or source_episode in excluded:
            raise RuntimeError(f"{episode_dir}: duplicate or evaluation source episode {source_episode}")
        sources.add(source_episode)
        prefix = int(metadata["model_prefix"]["executed_steps"])
        bins[(metadata["final_ball_cup"], prefix)] += 1
        contract = metadata["supervision_contract"]
        if contract.get("model_generated_actions_supervised") is not False:
            raise RuntimeError(f"{episode_dir}: model action entered supervision")
        if contract.get("model_generated_frames_supervised") is not False:
            raise RuntimeError(f"{episode_dir}: model frame entered supervision")
        if contract.get("first_training_pair") != "observation[60] -> oracle_action[61]":
            raise RuntimeError(f"{episode_dir}: shifted first training pair")
        oracle = metadata["oracle"]
        if not all(
            oracle.get(key) is True
            for key in (
                "sustained_live_cup_xy",
                "gated_z_descent",
                "closure_requires_measured_xy_z",
                "continuous_lift",
            )
        ):
            raise RuntimeError(f"{episode_dir}: incomplete coherent Oracle contract")
        offsets.append(float(metadata["switch"]["offset_m"]))
        heights.append(float(metadata["switch"]["safe_height_m"]))
        open_steps.append(int(oracle["open_steps"]))
        max_grasp_lift_xy.append(float(oracle["max_grasp_lift_xy_error_m"]))
        with np.load(path, allow_pickle=False) as source:
            if len(source["actions"]) != EXPECTED_FRAMES:
                raise RuntimeError(f"{path}: expected {EXPECTED_FRAMES} rows")
            mask = np.asarray(source["action_mask"], dtype=bool)
            supervision = np.asarray(source["supervision_source"], dtype=np.uint8)
            expected = np.arange(EXPECTED_FRAMES) >= 61
            if not np.array_equal(mask, expected):
                raise RuntimeError(f"{path}: action mask mismatch")
            if not np.array_equal(supervision == generator.legacy.SUPERVISION_ORACLE, expected):
                raise RuntimeError(f"{path}: non-Oracle supervised row")
            actions = np.asarray(source["actions"], dtype=np.float32)
            if actions.shape != (EXPECTED_FRAMES, 7) or not np.all(np.isfinite(actions)):
                raise RuntimeError(f"{path}: invalid absolute EEF7 actions")
    expected_bins = {
        (slot, prefix): EXPECTED_EPISODES // (len(generator.SLOTS) * len(generator.PREFIX_STEPS))
        for slot in generator.SLOTS
        for prefix in generator.PREFIX_STEPS
    }
    if dict(bins) != expected_bins:
        raise RuntimeError(f"Unbalanced slot/prefix bins: {dict(bins)}")
    return {
        "episodes": len(paths),
        "unique_source_episodes": len(sources),
        "evaluation_episode_overlap": 0,
        "balanced_slot_prefix_bins": {f"{slot}:{prefix}": count for (slot, prefix), count in bins.items()},
        "model_generated_actions_supervised": False,
        "oracle_only_rows": True,
        "switch_offset_m": _stats(offsets),
        "switch_safe_height_m": _stats(heights),
        "oracle_open_steps": _stats(open_steps),
        "max_grasp_lift_xy_error_m": _stats(max_grasp_lift_xy),
    }


def _stats(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _write_remapped_memory_bank(output: Path, source_paths: list[Path]) -> dict:
    """Remap the dense 5000-episode bank to the converted local episode IDs."""
    source_bank_path = Path(generator.DEFAULT_MEMORY).expanduser().resolve()
    with np.load(source_bank_path, allow_pickle=False) as source:
        source_episode_ids = np.asarray(source["episode_index"], dtype=np.int64)
        source_memory = np.asarray(source["final_memory"], dtype=np.float32)
        source_labels = np.asarray(source["final_label"], dtype=np.int32)
        source_predictions = np.asarray(source["final_prediction"], dtype=np.int32)
    if not np.array_equal(source_episode_ids, np.arange(len(source_episode_ids))):
        raise RuntimeError(f"Expected dense source memory bank: {source_bank_path}")

    original_episode_ids = []
    local_labels = []
    for local_episode, path in enumerate(source_paths):
        if path.parent.name != f"episode_{local_episode:06d}":
            raise RuntimeError(
                f"Raw episode order is not dense: local={local_episode} path={path}"
            )
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        source_episode = int(metadata["source_episode"])
        original_episode_ids.append(source_episode)
        local_labels.append(generator.SLOTS.index(str(metadata["final_ball_cup"])))

    original_episode_ids = np.asarray(original_episode_ids, dtype=np.int64)
    local_labels = np.asarray(local_labels, dtype=np.int32)
    if np.any(source_predictions[original_episode_ids] != source_labels[original_episode_ids]):
        raise RuntimeError("V11 contains an episode whose direct visual memory is incorrect")
    if not np.array_equal(source_labels[original_episode_ids], local_labels):
        raise RuntimeError("Raw metadata label disagrees with the frozen memory bank")

    local_memory = source_memory[original_episode_ids]
    local_episode_ids = np.arange(len(source_paths), dtype=np.int32)
    metadata = {
        "dataset_kind": f"{generator.DATASET_KIND}_remapped_memory",
        "source_memory_bank": str(source_bank_path),
        "episodes": len(source_paths),
        "all_memory_predictions_correct": True,
        "mapping": "converted episode_index -> original source_episode",
    }
    bank_path = output / MEMORY_BANK_NAME
    np.savez_compressed(
        bank_path,
        episode_index=local_episode_ids,
        source_episode=original_episode_ids.astype(np.int32),
        final_label=local_labels,
        final_prediction=local_labels.copy(),
        final_memory=local_memory,
        memory_templates=local_memory,
        episode_template_index=local_episode_ids,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "path": str(bank_path),
        "episodes": len(source_paths),
        "unique_source_episodes": len(np.unique(original_episode_ids)),
        "all_predictions_correct": True,
        "memory_shape": list(local_memory.shape),
    }


def _audit_converted(output: Path, source_paths: list[Path]) -> dict:
    parquet_paths = sorted(output.rglob("episode_*.parquet"))
    if len(parquet_paths) != EXPECTED_EPISODES:
        raise RuntimeError(f"Expected {EXPECTED_EPISODES} parquet files, got {len(parquet_paths)}")
    expected_eligible = np.arange(FIRST_ELIGIBLE, LAST_ELIGIBLE + 1)
    checked = 0
    for parquet_path, source_path in zip(parquet_paths, source_paths, strict=True):
        table = pq.read_table(parquet_path, columns=["frame_index", "action_mask", "actions"])
        frames = table["frame_index"].to_numpy()
        mask = table["action_mask"].to_numpy()
        if not np.array_equal(frames[mask], expected_eligible):
            raise RuntimeError(f"{parquet_path}: eligible observation rows shifted")
        converted = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        with np.load(source_path, allow_pickle=False) as source:
            canonical = raw.canonicalize_absolute_rotation_vectors(source["actions"])
        hold = raw.terminal_hold_action(canonical, osc_input_type="absolute")
        expected = np.broadcast_to(hold, converted.shape).copy()
        source_indices = frames[:, None] + 1 + np.arange(ACTION_HORIZON)[None, :]
        valid = source_indices < len(canonical)
        expected[valid] = canonical[source_indices[valid]]
        if not np.allclose(converted, expected, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"{parquet_path}: action horizon mismatch")
        checked += int(mask.sum())
    tasks = [
        json.loads(line)
        for line in (output / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if tasks != [
        {"task_index": 0, "task": OBSERVE_TASK},
        {"task_index": 1, "task": GRASP_TASK},
    ]:
        raise RuntimeError(f"Prompt contract mismatch: {tasks}")
    return {
        "episodes": len(parquet_paths),
        "eligible_rows_per_episode": len(expected_eligible),
        "exact_action_windows_checked": checked,
        "first_aligned_pair": "observation[60] -> oracle_action[61]",
        "action_horizon": ACTION_HORIZON,
    }


def main() -> None:
    args = raw.parse_args()
    if not args.phase_instructions or args.observe_task != OBSERVE_TASK or args.grasp_task != GRASP_TASK:
        raise ValueError("Use the validated phase prompts")
    if {int(value) for value in args.grasp_phase_ids.split(",")} != {8, 9, 10, 11}:
        raise ValueError("Use --grasp-phase-ids 8,9,10,11")
    if args.action_horizon != ACTION_HORIZON:
        raise ValueError(f"V11 conversion requires action_horizon={ACTION_HORIZON}")
    input_root = Path(args.input).expanduser().resolve()
    stale_candidates = sorted(input_root.glob(".candidate_*"))
    if stale_candidates:
        raise RuntimeError(
            f"Refusing to convert {len(stale_candidates)} unfinished candidate directories: "
            f"{stale_candidates[:3]}"
        )
    paths = sorted(input_root.glob("episode_*/vla_trajectory.npz"))
    if args.max_episodes is not None:
        paths = paths[: args.max_episodes]
    if len(paths) != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} raw episodes, got {len(paths)}")
    raw_audit = _audit_raw(paths)
    raw.main()
    output = Path(args.output).expanduser().resolve()
    converted_audit = _audit_converted(output, paths)
    memory_bank_audit = _write_remapped_memory_bank(output, paths)
    payload = {
        "ok": True,
        "dataset_kind": f"{generator.DATASET_KIND}_lerobot",
        "raw_audit": raw_audit,
        "converted_audit": converted_audit,
        "memory_bank_audit": memory_bank_audit,
    }
    audit_path = output / "current_action_low_stage_recovery_v11_audit.json"
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
