"""Build compact per-row XY difficulty metrics for the V7 EEF sampler.

The raw7 converter stores observation ``i`` with the first target action from
raw controller command ``i + 1``.  Since the correction oracle commands an
absolute world-frame cup-centre XY, the difference between the current EEF XY
and ``actions[0, :2]`` is exactly the recovery error visible to the policy.

Keeping these scalar metrics in a sidecar avoids loading every 16x7 action
chunk in every training process merely to classify sampling difficulty.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

DEFAULT_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_onpolicy_eef_low_stage_gated_v6_balanced1200_260816"
)
SLOT_ID = {"left": 0, "middle": 1, "right": 2}


def _episode_labels(root: pathlib.Path) -> dict[int, tuple[int, int]]:
    labels: dict[int, tuple[int, int]] = {}
    with (root / "meta/episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            episode = int(payload["episode_index"])
            # initial_ball_cup is the physical cup identity containing the ball;
            # final_ball_cup is that cup's final spatial slot after swaps.
            labels[episode] = (
                SLOT_ID[str(payload["final_ball_cup"])],
                SLOT_ID[str(payload["initial_ball_cup"])],
            )
    return labels


def build(root: pathlib.Path, output: pathlib.Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    labels = _episode_labels(root)
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files below {root / 'data'}")

    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    total_rows = int(info["total_frames"])
    episode_index = np.full(total_rows, -1, dtype=np.int32)
    frame_index = np.full(total_rows, -1, dtype=np.int16)
    target_phase = np.full(total_rows, -1, dtype=np.int8)
    target_frame = np.full(total_rows, -1, dtype=np.int16)
    target_lift_step = np.full(total_rows, -1, dtype=np.int16)
    action_mask = np.zeros(total_rows, dtype=np.bool_)
    xy_error_m = np.full(total_rows, np.nan, dtype=np.float32)
    delta_x_m = np.full(total_rows, np.nan, dtype=np.float32)
    delta_y_m = np.full(total_rows, np.nan, dtype=np.float32)
    target_x_m = np.full(total_rows, np.nan, dtype=np.float32)
    target_y_m = np.full(total_rows, np.nan, dtype=np.float32)
    final_slot = np.full(total_rows, -1, dtype=np.int8)
    target_identity = np.full(total_rows, -1, dtype=np.int8)
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
    for path in files:
        rows = pq.read_table(path, columns=columns).to_pylist()
        phases = np.asarray([row["phase_id"] for row in rows], dtype=np.int16)
        frames = np.asarray([row["frame_index"] for row in rows], dtype=np.int16)
        lift_frames = frames[phases == 11]
        first_lift = int(lift_frames.min()) if lift_frames.size else 32_767
        for local, row in enumerate(rows):
            index = int(row["index"])
            if index < 0 or index >= total_rows or seen[index]:
                raise ValueError(f"Invalid or duplicate global index {index} in {path}")
            seen[index] = True
            episode = int(row["episode_index"])
            if episode not in labels:
                raise KeyError(f"Episode {episode} is missing from episodes.jsonl")
            episode_index[index] = episode
            frame_index[index] = int(row["frame_index"])
            action_mask[index] = bool(row["action_mask"])
            final_slot[index], target_identity[index] = labels[episode]

            eef_xy = np.asarray(row["observation.robot0_eef_pos"][0][:2], dtype=np.float32)
            action_xy = np.asarray(row["actions"][0][:2], dtype=np.float32)
            delta = action_xy - eef_xy
            xy_error_m[index] = np.linalg.norm(delta)
            delta_x_m[index], delta_y_m[index] = delta
            target_x_m[index], target_y_m[index] = action_xy
            if local + 1 < len(rows):
                target_phase[index] = int(rows[local + 1]["phase_id"])
                target_frame[index] = int(rows[local + 1]["frame_index"])
                if target_phase[index] == 11:
                    target_lift_step[index] = target_frame[index] - first_lift

    if not np.all(seen):
        missing = np.flatnonzero(~seen)
        raise RuntimeError(f"Sidecar construction missed {len(missing)} rows; first={missing[:10]}")
    if np.any(episode_index < 0) or np.any(frame_index < 0):
        raise RuntimeError("Sidecar contains uninitialized episode/frame identifiers")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        schema_version=np.asarray(1, dtype=np.int32),
        source_root=np.asarray(str(root)),
        episode_index=episode_index,
        frame_index=frame_index,
        target_phase=target_phase,
        target_frame=target_frame,
        target_lift_step=target_lift_step,
        action_mask=action_mask,
        xy_error_m=xy_error_m,
        delta_x_m=delta_x_m,
        delta_y_m=delta_y_m,
        target_x_m=target_x_m,
        target_y_m=target_y_m,
        final_slot=final_slot,
        target_identity=target_identity,
    )
    valid = action_mask & (frame_index >= 60) & (frame_index <= 153)
    hard = valid & (xy_error_m > 0.005) & (target_y_m >= 0.10) & (delta_y_m >= 0.005)
    recovery = valid & (xy_error_m > 0.005) & ~hard
    aligned = valid & (xy_error_m >= 0.002) & (xy_error_m <= 0.005) & np.isin(target_phase, (8, 9))
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": total_rows,
                "episodes": len(labels),
                "hard_positive_y_inward": int(hard.sum()),
                "general_recovery": int(recovery.sum()),
                "aligned_2_5mm": int(aligned.sum()),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output or root / "xy_sampling_metrics_v7.npz"
    build(root, output.expanduser().resolve(), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
