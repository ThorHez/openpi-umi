"""Compare full-action checkpoints at the grasp boundary using joint FK.

Every observation contains the fixed tracker prefix (raw frames 0..59) plus
the dynamic current frame 109.  The predicted 16-step chunk therefore covers
raw frames 110..125: all ten grasp frames and the first six lift frames.
No controller action is executed, so the metrics isolate action prediction
from JOINT_POSITION tracking and contact dynamics.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path

import jax
import numpy as np

import joint_fk_selection_eval as fk_eval
import main as base
import main_v2_absolute_joint as joint_eval
from serve_old_tracker_full_joint_grasp import _build_config
import training_cup_eval

from openpi.policies import policy_config


CURRENT_FRAME = 109
FIRST_ACTION_FRAME = CURRENT_FRAME + 1
HISTORY_FRAMES = 60
TOTAL_FRAMES = 61
GRASP_STEPS = 10
PROMPT = "The shell game has ended. Grasp and lift the cup containing the ball."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../robosuite/outputs/shellgame_absolute_joint_dataset"),
    )
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--num-episodes", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=260810)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_record(episode_dir: Path) -> dict:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
        wrist = np.asarray(source["wrist_images"])
        third_person = np.asarray(source["third_person_images"])
        joint = np.asarray(source["joint_pos"], dtype=np.float32)
        gripper = np.asarray(source["gripper_state"], dtype=np.float32)
        eef = np.asarray(source["eef_pos"], dtype=np.float32)

    frame_indices = [*range(HISTORY_FRAMES), CURRENT_FRAME]
    observation = {
        "robot0_joint_pos": joint[CURRENT_FRAME].reshape(1, joint_eval.JOINT_DIM),
        "robot0_gripper_width": np.asarray(
            [[base._gripper_width(gripper[CURRENT_FRAME])]], dtype=np.float32
        ),
        "actions": np.zeros((16, joint_eval.ACTION_DIM), dtype=np.float32),
        "prompt": PROMPT,
        "video_frame_valid_mask": {
            "left_wrist_0_rgb_0": np.ones(TOTAL_FRAMES, dtype=bool),
            "left_wrist_0_rgb_1": np.ones(TOTAL_FRAMES, dtype=bool),
        },
    }
    for output_index, source_index in enumerate(frame_indices):
        observation[f"left_wrist_0_rgb_0_{output_index}"] = wrist[source_index].copy()
        observation[f"left_wrist_0_rgb_1_{output_index}"] = third_person[source_index].copy()

    reference_gripper = np.asarray(
        [
            base._gripper_width(value)
            for value in gripper[FIRST_ACTION_FRAME : FIRST_ACTION_FRAME + 16]
        ],
        dtype=np.float32,
    )
    return {
        "episode": episode_dir.name,
        "metadata": metadata,
        "observation": observation,
        "target_cup": str(metadata["target_cup_identity"]),
        "target_slot": str(metadata["final_ball_cup"]),
        "cup_positions": fk_eval._cup_positions(metadata),
        "reference_joint": joint[FIRST_ACTION_FRAME : FIRST_ACTION_FRAME + 16].copy(),
        "reference_gripper": reference_gripper,
        "reference_eef": eef[FIRST_ACTION_FRAME : FIRST_ACTION_FRAME + 16].copy(),
    }


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(left) - np.asarray(right)))))


def _mean_metrics(samples: list[dict]) -> dict[str, float]:
    metric_names = (
        "joint_rmse_rad",
        "eef_rmse_m",
        "grasp_eef_rmse_m",
        "early_lift_eef_rmse_m",
        "end_grasp_eef_error_m",
        "endpoint_eef_error_m",
        "end_grasp_target_cup_xy_error_m",
        "gripper_width_rmse_m",
    )
    return {name: float(np.mean([sample[name] for sample in samples])) for name in metric_names}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    episode_dirs = sorted(
        path for path in args.dataset_root.expanduser().resolve().glob("episode_*") if path.is_dir()
    )
    selected_ids = training_cup_eval._balanced_validation_ids(
        episode_dirs,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        sample_seed=args.sample_seed,
        num_episodes=args.num_episodes,
    )
    logging.info("Loading %d held-out episodes balanced by final slot", len(selected_ids))
    records = [_load_record(episode_dirs[int(index)]) for index in selected_ids]

    config = _build_config(sampling_steps=args.num_sampling_steps)
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=PROMPT,
        sample_kwargs={"num_steps": args.num_sampling_steps},
    )
    predicted_chunks = fk_eval._batched_infer(
        policy,
        [record["observation"] for record in records],
        args.batch_size,
        args.sample_seed,
    )

    shell = base._import_shellgame_tools(args.robosuite_root)
    fk_env = fk_eval._make_fk_env(shell, records[0])
    samples = []
    try:
        for record, predicted in zip(records, predicted_chunks, strict=True):
            predicted_eef, clipped_values = fk_eval._fk_chunk(shell, fk_env, predicted)
            reference_fk, _ = fk_eval._fk_chunk(shell, fk_env, record["reference_joint"])
            target_xy = record["cup_positions"][record["target_cup"]]
            end_grasp_xy_error = float(
                np.linalg.norm(predicted_eef[GRASP_STEPS - 1, :2] - target_xy)
            )
            samples.append(
                {
                    "episode": record["episode"],
                    "target_cup": record["target_cup"],
                    "target_slot": record["target_slot"],
                    "joint_rmse_rad": _rmse(
                        predicted[:, : joint_eval.JOINT_DIM], record["reference_joint"]
                    ),
                    "eef_rmse_m": _rmse(predicted_eef, reference_fk),
                    "grasp_eef_rmse_m": _rmse(
                        predicted_eef[:GRASP_STEPS], reference_fk[:GRASP_STEPS]
                    ),
                    "early_lift_eef_rmse_m": _rmse(
                        predicted_eef[GRASP_STEPS:], reference_fk[GRASP_STEPS:]
                    ),
                    "end_grasp_eef_error_m": float(
                        np.linalg.norm(
                            predicted_eef[GRASP_STEPS - 1]
                            - reference_fk[GRASP_STEPS - 1]
                        )
                    ),
                    "endpoint_eef_error_m": float(
                        np.linalg.norm(predicted_eef[-1] - reference_fk[-1])
                    ),
                    "end_grasp_target_cup_xy_error_m": end_grasp_xy_error,
                    "gripper_width_rmse_m": _rmse(
                        predicted[:, 7], record["reference_gripper"]
                    ),
                    "reference_fk_vs_saved_eef_rmse_m": _rmse(
                        reference_fk, record["reference_eef"]
                    ),
                    "clipped_joint_values": int(clipped_values),
                }
            )
    finally:
        fk_env.close()

    by_slot = {}
    for slot in ("left", "middle", "right"):
        slot_samples = [sample for sample in samples if sample["target_slot"] == slot]
        by_slot[slot] = {"num_episodes": len(slot_samples), **_mean_metrics(slot_samples)}
    output = {
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "current_frame": CURRENT_FRAME,
        "predicted_raw_frames": [FIRST_ACTION_FRAME, FIRST_ACTION_FRAME + 15],
        "fixed_diffusion_noise": True,
        "settings": {
            "num_episodes": args.num_episodes,
            "batch_size": args.batch_size,
            "num_sampling_steps": args.num_sampling_steps,
            "split_seed": args.split_seed,
            "sample_seed": args.sample_seed,
            "selected_episode_ids": selected_ids.tolist(),
        },
        "target_slot_distribution": dict(
            collections.Counter(record["target_slot"] for record in records)
        ),
        "overall": _mean_metrics(samples),
        "by_target_slot": by_slot,
        "mean_reference_fk_vs_saved_eef_rmse_m": float(
            np.mean([sample["reference_fk_vs_saved_eef_rmse_m"] for sample in samples])
        ),
        "total_clipped_joint_values": int(
            sum(sample["clipped_joint_values"] for sample in samples)
        ),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"overall": output["overall"], "by_target_slot": by_slot}, indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
