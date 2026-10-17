"""Strictly audit and convert the 150-episode V10 on-policy dataset."""

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
import convert_shellgame_to_openpi_umi_v2_openpi_action as common
import generate_v10_onpolicy_oracle_correction_dataset as generator


EXPECTED_EPISODES = 150
EXPECTED_FRAMES = 156
FIRST_ELIGIBLE = 60
LAST_ELIGIBLE = 154
OBSERVE_TASK = "Observe the ball moving under a cup and remember which cup contains it."
GRASP_TASK = "The shell game has ended. Grasp and lift the cup containing the ball."


def _audit_raw(paths: list[Path]) -> dict:
    slots = Counter()
    offsets = []
    heights = []
    for path in paths:
        episode_dir = path.parent
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("dataset_kind") != generator.DATASET_KIND:
            raise RuntimeError(f"{episode_dir}: wrong dataset_kind")
        if metadata["model_prefix"].get("state_source") != "real_v10_closed_loop_no_hidden_perturbation":
            raise RuntimeError(f"{episode_dir}: not a real V10 on-policy switch state")
        contract = metadata["supervision_contract"]
        if contract.get("model_generated_actions_supervised") is not False:
            raise RuntimeError(f"{episode_dir}: model-generated action entered supervision")
        if metadata["switch"]["selected_cup"] != metadata["switch"]["target_cup"]:
            raise RuntimeError(f"{episode_dir}: wrong cup selected at switch")
        slots[metadata["final_spatial_slot"]] += 1
        offsets.append(float(metadata["switch"]["offset_m"]))
        heights.append(float(metadata["switch"]["safe_height_m"]))
        with np.load(path, allow_pickle=False) as episode:
            if len(episode["actions"]) != EXPECTED_FRAMES:
                raise RuntimeError(f"{path}: expected {EXPECTED_FRAMES} frames")
            mask = np.asarray(episode["action_mask"], dtype=bool)
            source = np.asarray(episode["supervision_source"], dtype=np.uint8)
            expected = np.arange(EXPECTED_FRAMES) >= 61
            if not np.array_equal(mask, expected) or not np.array_equal(source == 1, expected):
                raise RuntimeError(f"{path}: Oracle-only mask contract failed")
    expected_slots = {"left": 50, "middle": 50, "right": 50}
    if dict(slots) != expected_slots:
        raise RuntimeError(f"Final spatial slots are not exactly balanced: {dict(slots)}")
    return {
        "episodes": len(paths),
        "frames_per_episode": EXPECTED_FRAMES,
        "final_spatial_slots": dict(slots),
        "offset_m": {"min": min(offsets), "mean": float(np.mean(offsets)), "max": max(offsets)},
        "safe_height_m": {"min": min(heights), "mean": float(np.mean(heights)), "max": max(heights)},
        "model_generated_actions_supervised": False,
        "oracle_only_rows": True,
    }


def _audit_converted(output: Path, source_paths: list[Path], action_horizon: int) -> dict:
    parquet_paths = sorted(output.rglob("episode_*.parquet"))
    if len(parquet_paths) != EXPECTED_EPISODES:
        raise RuntimeError(f"Expected {EXPECTED_EPISODES} parquet files, found {len(parquet_paths)}")
    expected_eligible = np.arange(FIRST_ELIGIBLE, LAST_ELIGIBLE + 1)
    windows_checked = 0
    for parquet_path, source_path in zip(parquet_paths, source_paths, strict=True):
        table = pq.read_table(parquet_path, columns=["frame_index", "action_mask", "actions"])
        frames = table["frame_index"].to_numpy()
        mask = table["action_mask"].to_numpy()
        if not np.array_equal(frames[mask], expected_eligible):
            raise RuntimeError(f"{parquet_path}: shifted mask mismatch")
        converted = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        with np.load(source_path, allow_pickle=False) as source:
            canonical = raw.canonicalize_absolute_rotation_vectors(source["actions"])
        hold = raw.terminal_hold_action(canonical, osc_input_type="absolute")
        expected = np.broadcast_to(hold, converted.shape).copy()
        source_indices = frames[:, None] + 1 + np.arange(action_horizon)[None, :]
        valid = source_indices < len(canonical)
        expected[valid] = canonical[source_indices[valid]]
        if not np.allclose(converted, expected, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"{parquet_path}: exact action horizon mismatch")
        windows_checked += int(mask.sum())
    tasks = [json.loads(line) for line in (output / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
    expected_tasks = [
        {"task_index": 0, "task": OBSERVE_TASK},
        {"task_index": 1, "task": GRASP_TASK},
    ]
    if tasks != expected_tasks:
        raise RuntimeError(f"Converted prompt contract mismatch: {tasks}")
    return {
        "episodes": len(parquet_paths),
        "eligible_observation_frames": [FIRST_ELIGIBLE, LAST_ELIGIBLE],
        "eligible_rows_per_episode": len(expected_eligible),
        "exact_action_windows_checked": windows_checked,
        "first_aligned_pair": "observation[60] -> oracle_action[61]",
        "action_horizon": action_horizon,
    }


def main() -> None:
    args = raw.parse_args()
    if not args.phase_instructions or args.observe_task != OBSERVE_TASK or args.grasp_task != GRASP_TASK:
        raise ValueError("Use the validated phase prompts for V10 on-policy conversion")
    if {int(x) for x in args.grasp_phase_ids.split(",")} != {8, 9, 10, 11}:
        raise ValueError("Use --grasp-phase-ids 8,9,10,11")
    if args.action_horizon != 16:
        raise ValueError("V10 on-policy conversion requires action_horizon=16")
    paths = common.find_npz_paths(args.input, args.max_episodes)
    if len(paths) != EXPECTED_EPISODES:
        raise ValueError(f"Conversion requires exactly {EXPECTED_EPISODES} episodes, found {len(paths)}")
    raw_audit = _audit_raw(paths)
    raw.main()
    output = Path(args.output).expanduser().resolve()
    converted_audit = _audit_converted(output, paths, args.action_horizon)
    payload = {
        "ok": True,
        "dataset_kind": f"{generator.DATASET_KIND}_lerobot",
        "raw_audit": raw_audit,
        "converted_audit": converted_audit,
    }
    (output / "v10_real_onpolicy_oracle_supervision_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
