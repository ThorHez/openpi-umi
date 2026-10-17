"""Build the audited per-row sampling sidecar for V9 EEF recovery data.

The V9 training recipe samples by the error visible to the policy, not by a
fixed frame range.  For every converted observation row this sidecar stores:

* current EEF-to-first-target XY error and correction direction;
* current height above the episode's Oracle grasp height;
* the aligned target phase (observation i -> command i + 1);
* final spatial cup, physical target identity, and V9 design strata.

The sidecar contains no images or action chunks and is therefore cheap to
load in every training process.  Construction is blocked unless the V9 raw
and converted audits, fixed prompt contract, and all 1200 design metadata
records are present.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import pyarrow.parquet as pq

DEFAULT_CONVERTED_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819"
)
DEFAULT_RAW_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819"
)
AUDIT_NAME = "safe_balanced_recovery_v9_oracle_supervision_audit.json"
EXPECTED_EPISODES = 1_200
EXPECTED_FRAMES = 155
NUM_DIRECTION_SECTORS = 16

SLOT_ID = {"left": 0, "middle": 1, "right": 2}
STAGE_ID = {"high": 0, "mid": 1, "late": 2}
OFFSET_BIN_ID = {"small": 0, "medium": 1, "large": 2}
EXPECTED_OBSERVE_TASK = "Observe the ball moving under a cup and remember which cup contains it."
EXPECTED_GRASP_TASK = "The shell game has ended. Grasp and lift the cup containing the ball."


def _jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_prompt_metadata(root: pathlib.Path) -> None:
    expected_tasks = [
        {"task_index": 0, "task": EXPECTED_OBSERVE_TASK},
        {"task_index": 1, "task": EXPECTED_GRASP_TASK},
    ]
    tasks = _jsonl(root / "meta/tasks.jsonl")
    if tasks != expected_tasks:
        raise ValueError(f"V9 task metadata mismatch: actual={tasks}, expected={expected_tasks}")
    episode_tasks = [EXPECTED_OBSERVE_TASK, EXPECTED_GRASP_TASK]
    episodes = _jsonl(root / "meta/episodes.jsonl")
    mismatched = [int(row["episode_index"]) for row in episodes if row.get("tasks") != episode_tasks]
    if mismatched:
        raise ValueError(f"V9 episode prompt mismatch: {mismatched[:10]} total={len(mismatched)}")


def _load_design(raw_root: pathlib.Path) -> dict[int, dict]:
    output: dict[int, dict] = {}
    for episode_dir in sorted(raw_root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")):
        episode = int(episode_dir.name.split("_")[-1])
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        if int(metadata["accepted_index"]) != episode:
            raise ValueError(f"{episode_dir}: accepted_index mismatch")
        if metadata.get("success") is not True:
            raise ValueError(f"{episode_dir}: unsuccessful episode in V9 training data")
        output[episode] = {
            "final_slot": SLOT_ID[str(metadata["final_spatial_slot"])],
            "target_identity": SLOT_ID[str(metadata["target_cup_identity"])],
            "offset_sector": int(metadata["v9_distribution"]["measured_offset_sector"]),
            "anchor_stage": STAGE_ID[str(metadata["anchor_stage"])],
            "offset_bin": OFFSET_BIN_ID[str(metadata["perturbation"]["offset_bin"])],
            "initial_xy_error_m": float(metadata["oracle"]["gating_trace"][0]["pre_command_xy_error_m"]),
        }
    if sorted(output) != list(range(EXPECTED_EPISODES)):
        raise ValueError(f"V9 raw design must contain episode indices 0..{EXPECTED_EPISODES - 1}; found {len(output)}")
    return output


def _correction_sector(delta_x: np.ndarray, delta_y: np.ndarray) -> np.ndarray:
    angle = np.mod(np.arctan2(delta_y, delta_x), 2.0 * math.pi)
    return (np.rint(angle * NUM_DIRECTION_SECTORS / (2.0 * math.pi)).astype(np.int16) % NUM_DIRECTION_SECTORS).astype(
        np.int8
    )


def build(
    converted_root: pathlib.Path,
    raw_root: pathlib.Path,
    output: pathlib.Path,
    *,
    overwrite: bool,
) -> dict:
    converted_root = converted_root.expanduser().resolve()
    raw_root = raw_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")

    audit_path = converted_root / AUDIT_NAME
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("ok") is not True or audit.get("raw_audit", {}).get("quota_complete") is not True:
        raise ValueError(f"V9 audit is not complete: {audit_path}")
    _validate_prompt_metadata(converted_root)
    design = _load_design(raw_root)

    info = json.loads((converted_root / "meta/info.json").read_text(encoding="utf-8"))
    if int(info["total_episodes"]) != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} episodes, got {info['total_episodes']}")
    total_rows = int(info["total_frames"])
    expected_rows = EXPECTED_EPISODES * EXPECTED_FRAMES
    if total_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {total_rows}")

    episode_index = np.full(total_rows, -1, dtype=np.int32)
    frame_index = np.full(total_rows, -1, dtype=np.int16)
    target_phase = np.full(total_rows, -1, dtype=np.int8)
    target_lift_step = np.full(total_rows, -1, dtype=np.int16)
    action_mask = np.zeros(total_rows, dtype=np.bool_)
    xy_error_m = np.full(total_rows, np.nan, dtype=np.float32)
    delta_x_m = np.full(total_rows, np.nan, dtype=np.float32)
    delta_y_m = np.full(total_rows, np.nan, dtype=np.float32)
    correction_sector = np.full(total_rows, -1, dtype=np.int8)
    height_above_grasp_m = np.full(total_rows, np.nan, dtype=np.float32)
    final_slot = np.full(total_rows, -1, dtype=np.int8)
    target_identity = np.full(total_rows, -1, dtype=np.int8)
    design_offset_sector = np.full(total_rows, -1, dtype=np.int8)
    anchor_stage = np.full(total_rows, -1, dtype=np.int8)
    offset_bin = np.full(total_rows, -1, dtype=np.int8)
    initial_xy_error_m = np.full(total_rows, np.nan, dtype=np.float32)
    seen = np.zeros(total_rows, dtype=np.bool_)

    columns = [
        "index",
        "episode_index",
        "frame_index",
        "phase_id",
        "action_mask",
        "observation.robot0_eef_pos",
        "actions",
    ]
    files = sorted((converted_root / "data").glob("chunk-*/*.parquet"))
    if len(files) != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} Parquet files, found {len(files)}")

    for path in files:
        rows = pq.read_table(path, columns=columns).to_pylist()
        if len(rows) != EXPECTED_FRAMES:
            raise ValueError(f"{path}: expected {EXPECTED_FRAMES} rows, got {len(rows)}")
        episode_values = {int(row["episode_index"]) for row in rows}
        if len(episode_values) != 1:
            raise ValueError(f"{path}: mixed episode indices {episode_values}")
        episode = episode_values.pop()
        if episode not in design:
            raise KeyError(f"{path}: missing raw design metadata for episode {episode}")

        phases = np.asarray([row["phase_id"] for row in rows], dtype=np.int16)
        frames = np.asarray([row["frame_index"] for row in rows], dtype=np.int16)
        indices = np.asarray([row["index"] for row in rows], dtype=np.int64)
        if not np.array_equal(frames, np.arange(EXPECTED_FRAMES)):
            raise ValueError(f"{path}: non-consecutive frame indices")
        expected_indices = episode * EXPECTED_FRAMES + frames.astype(np.int64)
        if not np.array_equal(indices, expected_indices):
            raise ValueError(f"{path}: global index contract mismatch")
        if np.any(seen[indices]):
            raise ValueError(f"{path}: duplicate global indices")

        next_phase = np.full(EXPECTED_FRAMES, -1, dtype=np.int16)
        next_phase[:-1] = phases[1:]
        action0 = np.asarray([row["actions"][0] for row in rows], dtype=np.float32)
        eef = np.asarray([row["observation.robot0_eef_pos"][0] for row in rows], dtype=np.float32)
        delta = action0[:, :2] - eef[:, :2]
        error = np.linalg.norm(delta, axis=1)
        grasp_rows = np.flatnonzero(next_phase == 10)
        if grasp_rows.size == 0:
            raise ValueError(f"{path}: no aligned grasp target rows")
        grasp_z_values = action0[grasp_rows, 2]
        if float(np.ptp(grasp_z_values)) > 1e-5:
            raise ValueError(f"{path}: inconsistent Oracle grasp Z")
        grasp_z = float(np.median(grasp_z_values))
        lift_frames = frames[phases == 11]
        if lift_frames.size == 0:
            raise ValueError(f"{path}: no lift phase")
        first_lift = int(lift_frames.min())
        lift_step = np.where(next_phase == 11, frames + 1 - first_lift, -1)

        d = design[episode]
        episode_index[indices] = episode
        frame_index[indices] = frames
        target_phase[indices] = next_phase.astype(np.int8)
        target_lift_step[indices] = lift_step.astype(np.int16)
        action_mask[indices] = np.asarray([row["action_mask"] for row in rows], dtype=np.bool_)
        xy_error_m[indices] = error
        delta_x_m[indices] = delta[:, 0]
        delta_y_m[indices] = delta[:, 1]
        correction_sector[indices] = _correction_sector(delta[:, 0], delta[:, 1])
        height_above_grasp_m[indices] = eef[:, 2] - grasp_z
        final_slot[indices] = d["final_slot"]
        target_identity[indices] = d["target_identity"]
        design_offset_sector[indices] = d["offset_sector"]
        anchor_stage[indices] = d["anchor_stage"]
        offset_bin[indices] = d["offset_bin"]
        initial_xy_error_m[indices] = d["initial_xy_error_m"]
        seen[indices] = True

    if not np.all(seen):
        missing = np.flatnonzero(~seen)
        raise RuntimeError(f"V9 sidecar missed {len(missing)} rows; first={missing[:10]}")
    eligible = action_mask & (frame_index >= 60) & (frame_index <= 153)
    if int(eligible.sum()) != EXPECTED_EPISODES * 94:
        raise RuntimeError(f"Unexpected V9 eligible-row count: {int(eligible.sum())}")
    if not np.all(np.isfinite(xy_error_m[eligible])) or not np.all(np.isfinite(height_above_grasp_m[eligible])):
        raise RuntimeError("V9 eligible metrics contain NaN or Inf")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        schema_version=np.asarray(2, dtype=np.int32),
        converted_root=np.asarray(str(converted_root)),
        raw_root=np.asarray(str(raw_root)),
        episode_index=episode_index,
        frame_index=frame_index,
        target_phase=target_phase,
        target_lift_step=target_lift_step,
        action_mask=action_mask,
        xy_error_m=xy_error_m,
        delta_x_m=delta_x_m,
        delta_y_m=delta_y_m,
        correction_sector=correction_sector,
        height_above_grasp_m=height_above_grasp_m,
        final_slot=final_slot,
        target_identity=target_identity,
        design_offset_sector=design_offset_sector,
        anchor_stage=anchor_stage,
        offset_bin=offset_bin,
        initial_xy_error_m=initial_xy_error_m,
    )

    recovery = eligible & np.isin(target_phase, (8, 9))
    summary = {
        "output": str(output),
        "schema_version": 2,
        "rows": total_rows,
        "episodes": EXPECTED_EPISODES,
        "eligible_rows": int(eligible.sum()),
        # Every V9 episode starts >5 mm from the live cup centre.  The action
        # target includes <=2 mm grasp jitter, so a 4 mm target-error floor is
        # the conservative per-row margin that still covers all 1200 starts.
        "hard_initial_gt5mm_target_gt4mm": int(
            np.count_nonzero(recovery & (initial_xy_error_m > 0.005) & (xy_error_m > 0.004))
        ),
        "low_target_1_4mm_le40mm": int(
            np.count_nonzero(recovery & (xy_error_m >= 0.001) & (xy_error_m <= 0.004) & (height_above_grasp_m <= 0.040))
        ),
        "aligned_lt1mm_le60mm": int(
            np.count_nonzero(recovery & (xy_error_m < 0.001) & (height_above_grasp_m <= 0.060))
        ),
        "grasp_rows": int(np.count_nonzero(eligible & (target_phase == 10))),
        "early_lift_rows": int(
            np.count_nonzero(eligible & (target_phase == 11) & (target_lift_step >= 0) & (target_lift_step < 10))
        ),
        "prompt_contract": "validated_long_phase_prompt",
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-root", type=pathlib.Path, default=DEFAULT_CONVERTED_ROOT)
    parser.add_argument("--raw-root", type=pathlib.Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or args.converted_root / "xy_sampling_metrics_v9.npz"
    build(args.converted_root, args.raw_root, output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
