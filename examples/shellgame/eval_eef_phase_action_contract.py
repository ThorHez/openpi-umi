"""Measure checkpoint action quality at each recorded V9 recovery phase.

This is an offline, paired diagnostic.  It rebuilds the deployed fixed-history
observation (reveal frames 0..59 plus a later live frame) and compares the
first five predicted commands with the untouched Oracle suffix.  The four
phase groups isolate recenter, descent, grasp, and lift behavior without
running MuJoCo closed loop.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import gc
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import numpy as np

import eval_eef_correction_action_contract as initial_contract
import joint_fk_selection_eval as infer_utils
import main_absolute_eef_fixed_history as fixed_eef
from serve_old_tracker_full_absolute_eef import _build_config

from examples.shellgame.eval_old_tracker_query_action_closed_loop_gate import PROMPT
from openpi.policies import policy_config


PHASES = {"recenter": 8, "descent": 9, "grasp": 10, "lift": 11}
ACTION_HORIZON = 16
EXECUTED_PREFIX = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states-per-phase", type=int, default=12)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--state-seed", type=int, default=260820)
    parser.add_argument("--sample-seed", type=int, default=260814)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    return parser.parse_args()


def _target_chunk(commands: np.ndarray, current: int) -> np.ndarray:
    start = current + 1
    chunk = commands[start : start + ACTION_HORIZON]
    if len(chunk) == 0:
        raise ValueError(f"No future command after frame {current}")
    if len(chunk) < ACTION_HORIZON:
        chunk = np.concatenate(
            [chunk, np.repeat(chunk[-1:], ACTION_HORIZON - len(chunk), axis=0)],
            axis=0,
        )
    return np.asarray(chunk, dtype=np.float32)


def _load_episode_states(
    episode_dir: Path,
    wanted: set[str],
    rng: np.random.Generator,
    policy_args,
) -> list[tuple[str, dict, dict]]:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
        wrist = np.asarray(source["wrist_images"])
        base_images = np.asarray(source["third_person_images"])
        eef_pos = np.asarray(source["eef_pos"], dtype=np.float32)
        eef_quat = np.asarray(source["eef_quat"], dtype=np.float32)
        gripper = np.asarray(source["gripper_state"], dtype=np.float32).reshape(-1)
        commands = np.asarray(source["controller_actions"], dtype=np.float32)
        phase_ids = np.asarray(source["phase_ids"], dtype=np.int16)
        action_mask = np.asarray(source["action_mask"], dtype=bool)

    history_prefix = [
        {"wrist": wrist[index], "base": base_images[index]}
        for index in range(fixed_eef.HISTORY_FRAMES)
    ]
    rows: list[tuple[str, dict, dict]] = []
    for name in sorted(wanted):
        phase_id = PHASES[name]
        candidates = np.flatnonzero(
            action_mask[1:]
            & (phase_ids[1:] == phase_id)
            & (np.arange(len(phase_ids) - 1) >= fixed_eef.HISTORY_FRAMES)
        )
        if len(candidates) == 0:
            continue
        current = int(rng.choice(candidates))
        history = list(history_prefix)
        history.append(
            {
                "wrist": wrist[current],
                "base": base_images[current],
                "eef_pos": eef_pos[current],
                "eef_quat": eef_quat[current],
                "gripper_width": float(gripper[current]),
            }
        )
        observation = fixed_eef._fixed_history_policy_input(
            history,
            eef_pos[current],
            args=policy_args,
            prompt=PROMPT,
        )
        record = {
            "episode": episode_dir.name,
            "frame": current,
            "phase": name,
            "target_cup": str(metadata["target_cup_identity"]),
            "target_slot": str(metadata["final_ball_cup"]),
            "oracle": _target_chunk(commands, current),
        }
        rows.append((name, observation, record))
    return rows


def _evaluate(record: dict, prediction: np.ndarray) -> dict:
    predicted = np.asarray(prediction, dtype=np.float64)
    oracle = np.asarray(record.pop("oracle"), dtype=np.float64)
    prefix_pred = predicted[:EXECUTED_PREFIX]
    prefix_oracle = oracle[:EXECUTED_PREFIX]
    xyz_delta = prefix_pred[:, :3] - prefix_oracle[:, :3]
    xy_delta = xyz_delta[:, :2]
    pred_closed = prefix_pred[:, 6] > 0.0
    oracle_closed = prefix_oracle[:, 6] > 0.0
    return {
        **record,
        "xyz_l2_error_mm": float(np.mean(np.linalg.norm(xyz_delta, axis=1)) * 1_000.0),
        "xy_l2_error_mm": float(np.mean(np.linalg.norm(xy_delta, axis=1)) * 1_000.0),
        "z_abs_error_mm": float(np.mean(np.abs(xyz_delta[:, 2])) * 1_000.0),
        "rotation_vector_l2_error": float(
            np.mean(np.linalg.norm(prefix_pred[:, 3:6] - prefix_oracle[:, 3:6], axis=1))
        ),
        "gripper_sign_accuracy": float(np.mean(pred_closed == oracle_closed)),
        "predicted_close_by_5": bool(np.any(pred_closed)),
        "oracle_close_by_5": bool(np.any(oracle_closed)),
        "predicted_closed_steps": int(np.count_nonzero(pred_closed)),
        "oracle_closed_steps": int(np.count_nonzero(oracle_closed)),
    }


def _aggregate(rows: list[dict]) -> dict:
    result = {}
    for phase in PHASES:
        group = [row for row in rows if row["phase"] == phase]
        result[phase] = {
            "states": len(group),
            "target_cup_distribution": dict(collections.Counter(row["target_cup"] for row in group)),
            "mean_xyz_l2_error_mm": float(np.mean([row["xyz_l2_error_mm"] for row in group])),
            "mean_xy_l2_error_mm": float(np.mean([row["xy_l2_error_mm"] for row in group])),
            "mean_z_abs_error_mm": float(np.mean([row["z_abs_error_mm"] for row in group])),
            "mean_rotation_vector_l2_error": float(
                np.mean([row["rotation_vector_l2_error"] for row in group])
            ),
            "mean_gripper_sign_accuracy": float(
                np.mean([row["gripper_sign_accuracy"] for row in group])
            ),
            "predicted_close_by_5_rate": float(np.mean([row["predicted_close_by_5"] for row in group])),
            "oracle_close_by_5_rate": float(np.mean([row["oracle_close_by_5"] for row in group])),
            "mean_predicted_closed_steps": float(np.mean([row["predicted_closed_steps"] for row in group])),
            "mean_oracle_closed_steps": float(np.mean([row["oracle_closed_steps"] for row in group])),
        }
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    root = args.dataset_root.expanduser().resolve()
    episode_dirs = sorted(path for path in root.glob("episode_*") if path.is_dir())
    val_ids = initial_contract._validation_episode_ids(len(episode_dirs), args.val_ratio, args.split_seed)
    rng = np.random.default_rng(args.state_seed)
    val_ids = rng.permutation(val_ids)
    counts = collections.Counter()
    observations = []
    records = []
    policy_args = initial_contract._policy_args()
    for episode_id in val_ids:
        wanted = {phase for phase in PHASES if counts[phase] < args.states_per_phase}
        if not wanted:
            break
        for phase, observation, record in _load_episode_states(
            episode_dirs[int(episode_id)], wanted, rng, policy_args
        ):
            observations.append(observation)
            records.append(record)
            counts[phase] += 1
    if any(counts[phase] != args.states_per_phase for phase in PHASES):
        raise RuntimeError(f"Could not fill phase quotas: {dict(counts)}")
    logging.info("Loaded paired held-out states: %s", dict(counts))

    config = _build_config(sampling_steps=args.num_sampling_steps)
    policy = policy_config.create_trained_policy(
        dataclasses.replace(config, fsdp_devices=1),
        args.checkpoint_dir,
        default_prompt=PROMPT,
        sample_kwargs={"num_steps": args.num_sampling_steps},
    )
    try:
        predictions = infer_utils._batched_infer(
            policy, observations, batch_size=args.batch_size, seed=args.sample_seed
        )
    finally:
        del policy
        jax.clear_caches()
        gc.collect()

    rows = [_evaluate(record, prediction) for record, prediction in zip(records, predictions, strict=True)]
    payload = {
        "experiment": "held-out V9 per-phase first-five action contract",
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "dataset_root": str(root),
        "sampling": {
            "split_seed": args.split_seed,
            "state_seed": args.state_seed,
            "sample_seed": args.sample_seed,
            "states_per_phase": args.states_per_phase,
            "executed_prefix": EXECUTED_PREFIX,
        },
        "summary": _aggregate(rows),
        "states": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
