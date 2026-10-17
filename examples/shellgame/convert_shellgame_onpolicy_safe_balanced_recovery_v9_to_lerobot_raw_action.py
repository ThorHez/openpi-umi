"""Convert only a quota-complete, strictly audited V9 raw dataset."""

# ruff: noqa: E402, I001, SLF001

from __future__ import annotations

from collections import Counter
import concurrent.futures
import json
from pathlib import Path
import shutil
import sys

ROBOSUITE_SCRIPTS = Path(__file__).resolve().parents[3] / "robosuite/robosuite/scripts"
sys.path.insert(0, str(ROBOSUITE_SCRIPTS))

import audit_onpolicy_eef_safe_balanced_recovery_dataset_v9 as audit_v9
import convert_shellgame_onpolicy_continuous_descent_v4_to_lerobot_raw_action as v4_convert
import convert_shellgame_to_lerobot_raw_action as raw
import convert_shellgame_to_openpi_umi_v2_openpi_action as common
import generate_onpolicy_eef_safe_balanced_recovery_dataset_v9 as v9
import numpy as np
import pyarrow.parquet as pq


EXPECTED_OBSERVE_TASK = (
    "Observe the ball moving under a cup and remember which cup contains it."
)
EXPECTED_GRASP_TASK = (
    "The shell game has ended. Grasp and lift the cup containing the ball."
)


def _audit_prompt_metadata(output_dir: Path) -> dict:
    tasks = [
        json.loads(line)
        for line in (output_dir / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_tasks = [
        {"task_index": 0, "task": EXPECTED_OBSERVE_TASK},
        {"task_index": 1, "task": EXPECTED_GRASP_TASK},
    ]
    if tasks != expected_tasks:
        raise RuntimeError(
            f"V9 converted prompt metadata mismatch: actual={tasks}, expected={expected_tasks}"
        )

    episodes = [
        json.loads(line)
        for line in (output_dir / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_episode_tasks = [EXPECTED_OBSERVE_TASK, EXPECTED_GRASP_TASK]
    mismatched = [
        int(row["episode_index"])
        for row in episodes
        if row.get("tasks") != expected_episode_tasks
    ]
    if mismatched:
        raise RuntimeError(
            "V9 episode prompt metadata mismatch for episodes "
            f"{mismatched[:10]} (total={len(mismatched)})"
        )
    return {
        "observe_task": EXPECTED_OBSERVE_TASK,
        "grasp_task": EXPECTED_GRASP_TASK,
        "episodes_checked": len(episodes),
    }


def _audit_converted_episode(job: tuple[Path, Path, int]) -> tuple[int, int]:
    parquet_path, npz_path, action_horizon = job
    table = pq.read_table(parquet_path, columns=["frame_index", "action_mask", "actions"])
    frames = table["frame_index"].to_numpy()
    aligned_mask = table["action_mask"].to_numpy()
    eligible = frames[aligned_mask]
    expected_eligible = np.arange(
        v4_convert.SWITCH_OBSERVATION_FRAME,
        v4_convert.LAST_ELIGIBLE_OBSERVATION_FRAME + 1,
        dtype=np.int64,
    )
    if not np.array_equal(eligible, expected_eligible):
        raise RuntimeError(f"{parquet_path}: aligned action-mask contract failed")

    converted = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
    expected_shape = (v4_convert.EXPECTED_FRAMES, action_horizon, raw.RAW_ACTION_DIM)
    if converted.shape != expected_shape:
        raise RuntimeError(f"{parquet_path}: unexpected converted action shape")
    with np.load(npz_path, allow_pickle=False) as source:
        canonical = raw.canonicalize_absolute_rotation_vectors(source["actions"])

    hold = raw.terminal_hold_action(canonical, osc_input_type="absolute")
    expected = np.broadcast_to(
        hold,
        (len(eligible), action_horizon, raw.RAW_ACTION_DIM),
    ).copy()
    source_indices = eligible[:, None] + 1 + np.arange(action_horizon)[None, :]
    valid = source_indices < len(canonical)
    expected[valid] = canonical[source_indices[valid]]
    if not np.allclose(converted[eligible], expected, atol=1e-6, rtol=0.0):
        delta = float(np.max(np.abs(converted[eligible] - expected)))
        raise RuntimeError(f"{parquet_path}: horizon-window mismatch, max_abs={delta}")
    return int(aligned_mask.sum()), len(eligible)


def _audit_converted_parallel(
    output_dir: Path,
    npz_paths: list[Path],
    action_horizon: int,
    workers: int,
) -> dict:
    parquet_paths = sorted(output_dir.rglob("episode_*.parquet"))
    if len(parquet_paths) != len(npz_paths):
        raise RuntimeError(
            f"Converted {len(parquet_paths)} Parquet episodes for {len(npz_paths)} raw episodes"
        )
    jobs = [
        (parquet_path, npz_path, action_horizon)
        for parquet_path, npz_path in zip(parquet_paths, npz_paths, strict=True)
    ]
    if workers == 1:
        results = map(_audit_converted_episode, jobs)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_audit_converted_episode, jobs, chunksize=4)
    eligible_counts = Counter()
    checked_windows = 0
    try:
        for eligible_count, windows in results:
            eligible_counts[eligible_count] += 1
            checked_windows += windows
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    return {
        "converted_episodes": len(parquet_paths),
        "eligible_observation_frames": [
            v4_convert.SWITCH_OBSERVATION_FRAME,
            v4_convert.LAST_ELIGIBLE_OBSERVATION_FRAME,
        ],
        "eligible_rows_per_episode": dict(sorted(eligible_counts.items())),
        "first_aligned_pair": "observation[60] -> raw Oracle action[61]",
        "action_horizon": action_horizon,
        "exact_consecutive_windows_checked": checked_windows,
        "terminal_padding": "repeat_final_absolute_controller_command",
        "context_rows_trainable": False,
        "audit_workers": workers,
    }


def main() -> None:
    args = raw.parse_args()
    if not args.phase_instructions:
        raise ValueError("V9 conversion requires --phase-instructions")
    if args.observe_task != EXPECTED_OBSERVE_TASK:
        raise ValueError(
            "V9 conversion requires the validated observe prompt; got "
            f"{args.observe_task!r}"
        )
    if args.grasp_task != EXPECTED_GRASP_TASK:
        raise ValueError(
            "V9 conversion requires --grasp-task "
            f"{EXPECTED_GRASP_TASK!r}; got {args.grasp_task!r}"
        )
    grasp_phases = {int(item.strip()) for item in args.grasp_phase_ids.split(",") if item.strip()}
    if grasp_phases != {8, 9, 10, 11}:
        raise ValueError("V9 conversion requires --grasp-phase-ids 8,9,10,11")
    if args.action_horizon != 16:
        raise ValueError("V9 conversion requires --action-horizon 16")

    input_paths = common.find_npz_paths(args.input, args.max_episodes)
    if len(input_paths) != 1200:
        raise ValueError(
            "V9 conversion is blocked until all 1200 fixed design slots exist; "
            f"found {len(input_paths)}"
        )
    raw_audit = audit_v9.audit(
        Path(args.input),
        expected_episodes=1200,
        require_complete_quota=True,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output.parent).free
    required_bytes = 20 * 1024**3
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Insufficient disk for V9 LeRobot conversion plus 8 GiB margin: "
            f"free={free_bytes / 1024**3:.1f} GiB, "
            f"required={required_bytes / 1024**3:.1f} GiB"
        )
    raw.main()
    converted_audit = _audit_converted_parallel(
        output,
        input_paths,
        args.action_horizon,
        args.workers,
    )
    prompt_audit = _audit_prompt_metadata(output)
    payload = {
        "ok": True,
        "dataset_kind": f"{v9.DATASET_KIND}_lerobot",
        "raw_audit": raw_audit,
        "converted_audit": converted_audit,
        "prompt_audit": prompt_audit,
        "training_contract": {
            "fixed_design_quota_complete": True,
            "hidden_actions_supervised": False,
            "first_aligned_pair": "observation[60] -> raw Oracle action[61]",
            "full_consecutive_horizon": 16,
            "sampler_must_use_measured_xy_error": True,
            "sampler_must_retain_v6_replay": True,
        },
    }
    audit_path = output / "safe_balanced_recovery_v9_oracle_supervision_audit.json"
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
