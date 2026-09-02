"""Compare absolute-EEF checkpoints on held-out on-policy offset states.

Each raw correction episode stores the exact post-model offset observation at
frame 60, followed by Oracle recenter / descend commands.  This evaluator
rebuilds the deployed fixed-history input (frames 0..59 plus frame 60), samples
one action chunk, and measures whether the first three commands implement the
training contract: recenter in XY without descending for two commands, then
start descending on command three.

The split matches the action trainer: episode-level NumPy permutation with
``seed=42`` and a 10% validation ratio.  Running multiple checkpoints with the
same CLI seeds therefore gives a paired comparison on identical observations
and identical diffusion keys.
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

import joint_fk_selection_eval as infer_utils
import main as base
import main_absolute_eef_fixed_history as fixed_eef
from serve_old_tracker_full_absolute_eef import _build_config

from examples.shellgame.eval_old_tracker_query_action_closed_loop_gate import PROMPT
from examples.shellgame import train_old_tracker_full_joint_grasp as full_joint
from openpi.policies import policy_config


CURRENT_FRAME = 60
FIRST_ORACLE_FRAME = 61
EXPECTED_EPISODE_FRAMES = 155


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "../robosuite/outputs/"
            "shellgame_onpolicy_eef_correction_raw7_replan_v2_500ep_260813"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=260814)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--early-z-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--min-third-descent-mm", type=float, default=3.0)
    return parser.parse_args()


def _validation_episode_ids(total: int, val_ratio: float, seed: int) -> np.ndarray:
    episodes = np.arange(total, dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(episodes)
    count = min(max(1, round(total * val_ratio)), total - 1)
    return np.sort(shuffled[:count])


def _policy_args() -> base.Args:
    args = base.Args()
    args.num_frames = fixed_eef.TOTAL_FRAMES
    args.frame_stride = 1
    args.policy_input_mode = "history"
    args.action_horizon = 16
    args.action_dim = 7
    args.action_mode = "raw7"
    args.observation_position_frame = "absolute"
    args.osc_input_type = "absolute"
    args.task = PROMPT
    args.phase_instructions = True
    args.grasp_task = PROMPT
    return args


def _load_observation(episode_dir: Path, policy_args: base.Args) -> tuple[dict, dict]:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
        wrist = np.asarray(source["wrist_images"])
        base_images = np.asarray(source["third_person_images"])
        eef_pos = np.asarray(source["eef_pos"], dtype=np.float32)
        eef_quat = np.asarray(source["eef_quat"], dtype=np.float32)
        gripper = np.asarray(source["gripper_state"], dtype=np.float32).reshape(-1)
        commands = np.asarray(source["controller_actions"], dtype=np.float32)

    if wrist.shape[0] != EXPECTED_EPISODE_FRAMES or commands.shape != (EXPECTED_EPISODE_FRAMES, 7):
        raise ValueError(f"Unexpected episode shape in {episode_dir}")

    history = [
        {"wrist": wrist[index], "base": base_images[index]}
        for index in range(fixed_eef.HISTORY_FRAMES)
    ]
    history.append(
        {
            "wrist": wrist[CURRENT_FRAME],
            "base": base_images[CURRENT_FRAME],
            "eef_pos": eef_pos[CURRENT_FRAME],
            "eef_quat": eef_quat[CURRENT_FRAME],
            "gripper_width": float(gripper[CURRENT_FRAME]),
        }
    )
    observation = fixed_eef._fixed_history_policy_input(
        history,
        eef_pos[CURRENT_FRAME],
        args=policy_args,
        prompt=PROMPT,
    )
    target_xy = commands[FIRST_ORACLE_FRAME, :2]
    initial_xy_error_m = float(np.linalg.norm(eef_pos[CURRENT_FRAME, :2] - target_xy))
    record = {
        "episode": episode_dir.name,
        "target_cup": str(metadata["target_cup_identity"]),
        "target_slot": str(metadata["final_ball_cup"]),
        "current_eef": eef_pos[CURRENT_FRAME].astype(np.float64),
        "target_xy": target_xy.astype(np.float64),
        "initial_xy_error_m": initial_xy_error_m,
        "oracle_first_three": commands[FIRST_ORACLE_FRAME : FIRST_ORACLE_FRAME + 3].astype(np.float64),
    }
    return observation, record


def _evaluate_prediction(record: dict, predicted: np.ndarray, args: argparse.Namespace) -> dict:
    predicted = np.asarray(predicted, dtype=np.float64)
    if predicted.shape != (16, 7) or not np.isfinite(predicted).all():
        raise ValueError(f"Invalid predicted action shape/value for {record['episode']}: {predicted.shape}")

    current = record["current_eef"]
    target_xy = record["target_xy"]
    initial_error = record["initial_xy_error_m"]
    first_three = predicted[:3]
    xy_error = np.linalg.norm(first_three[:, :2] - target_xy[None, :], axis=1)
    xy_progress = initial_error - xy_error
    z_delta = first_three[:, 2] - current[2]
    early_safe = z_delta[:2] >= -(args.early_z_tolerance_mm / 1000.0)
    early_xy_improves = xy_progress[:2] > 0.0
    third_descends = z_delta[2] <= -(args.min_third_descent_mm / 1000.0)
    contract_pass = bool(np.all(early_safe) and np.all(early_xy_improves) and third_descends)

    oracle = record["oracle_first_three"]
    oracle_xy_error = np.linalg.norm(oracle[:, :2] - target_xy[None, :], axis=1)
    oracle_z_delta = oracle[:, 2] - current[2]
    return {
        "episode": record["episode"],
        "target_cup": record["target_cup"],
        "target_slot": record["target_slot"],
        "initial_xy_error_mm": initial_error * 1000.0,
        "predicted_xy_error_mm": (xy_error * 1000.0).tolist(),
        "predicted_xy_progress_mm": (xy_progress * 1000.0).tolist(),
        "predicted_z_delta_mm": (z_delta * 1000.0).tolist(),
        "first_two_xy_both_improve": bool(np.all(early_xy_improves)),
        "first_two_z_both_safe": bool(np.all(early_safe)),
        "third_descends": bool(third_descends),
        "contract_pass": contract_pass,
        "oracle_xy_error_mm": (oracle_xy_error * 1000.0).tolist(),
        "oracle_z_delta_mm": (oracle_z_delta * 1000.0).tolist(),
        "predicted_first_three": first_three.tolist(),
    }


def _mean_matrix(rows: list[dict], key: str) -> list[float]:
    return np.mean(np.asarray([row[key] for row in rows], dtype=np.float64), axis=0).tolist()


def _aggregate(rows: list[dict]) -> dict:
    return {
        "num_episodes": len(rows),
        "target_cup_distribution": dict(collections.Counter(row["target_cup"] for row in rows)),
        "mean_initial_xy_error_mm": float(np.mean([row["initial_xy_error_mm"] for row in rows])),
        "mean_predicted_xy_error_mm_first_three": _mean_matrix(rows, "predicted_xy_error_mm"),
        "mean_predicted_xy_progress_mm_first_three": _mean_matrix(rows, "predicted_xy_progress_mm"),
        "mean_predicted_z_delta_mm_first_three": _mean_matrix(rows, "predicted_z_delta_mm"),
        "first_two_xy_both_improve_rate": float(np.mean([row["first_two_xy_both_improve"] for row in rows])),
        "first_two_z_both_safe_rate": float(np.mean([row["first_two_z_both_safe"] for row in rows])),
        "third_descent_rate": float(np.mean([row["third_descends"] for row in rows])),
        "full_contract_pass_rate": float(np.mean([row["contract_pass"] for row in rows])),
        "mean_oracle_xy_error_mm_first_three": _mean_matrix(rows, "oracle_xy_error_mm"),
        "mean_oracle_z_delta_mm_first_three": _mean_matrix(rows, "oracle_z_delta_mm"),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    root = args.dataset_root.expanduser().resolve()
    episode_dirs = sorted(path for path in root.glob("episode_*") if path.is_dir())
    if len(episode_dirs) < 2:
        raise ValueError(f"No correction episodes found under {root}")
    validation_ids = _validation_episode_ids(len(episode_dirs), args.val_ratio, args.split_seed)
    if args.num_episodes > len(validation_ids):
        raise ValueError(f"Requested {args.num_episodes} episodes but validation split has {len(validation_ids)}")
    selected_ids = validation_ids[: args.num_episodes]
    policy_args = _policy_args()
    observations = []
    records = []
    for episode_id in selected_ids:
        observation, record = _load_observation(episode_dirs[int(episode_id)], policy_args)
        observations.append(observation)
        records.append(record)
    logging.info("Loaded %d held-out correction states: %s", len(records), selected_ids.tolist())

    config = _build_config(sampling_steps=args.num_sampling_steps)
    policy = policy_config.create_trained_policy(
        dataclasses.replace(config, fsdp_devices=1),
        args.checkpoint_dir,
        default_prompt=PROMPT,
        sample_kwargs={"num_steps": args.num_sampling_steps},
    )
    if not isinstance(policy._model, full_joint.OldTrackerFullJointGraspModel):  # noqa: SLF001
        raise TypeError(f"Unexpected restored model: {type(policy._model).__name__}")  # noqa: SLF001
    try:
        predictions = infer_utils._batched_infer(
            policy,
            observations,
            batch_size=args.batch_size,
            seed=args.sample_seed,
        )
    finally:
        del policy
        jax.clear_caches()
        gc.collect()

    rows = [_evaluate_prediction(record, prediction, args) for record, prediction in zip(records, predictions, strict=True)]
    payload = {
        "experiment": "held-out absolute-EEF correction action contract",
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "dataset_root": str(root),
        "same_as_training_validation_split": True,
        "validation_split": {
            "val_ratio": args.val_ratio,
            "split_seed": args.split_seed,
            "selected_episode_ids": selected_ids.tolist(),
        },
        "sampling": {
            "sample_seed": args.sample_seed,
            "num_sampling_steps": args.num_sampling_steps,
            "batch_size": args.batch_size,
        },
        "contract_thresholds": {
            "early_z_tolerance_mm": args.early_z_tolerance_mm,
            "min_third_descent_mm": args.min_third_descent_mm,
        },
        "aggregate": _aggregate(rows),
        "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
