"""Evaluate ShellGame cup intent by applying FK to predicted joint chunks.

The policy is queried once at the first grasp-prompt observation (raw frame 59).
No controller step is executed.  Predicted absolute Panda joints are clipped to
the evaluator's limits, converted to EEF positions with MuJoCo forward
kinematics, and classified by their proximity to the three settled cups.
"""

# This diagnostic intentionally reuses the evaluator and Policy internals so
# its preprocessing and action sampling exactly match online inference.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import main as base
import main_v2_absolute_joint as joint_eval
import numpy as np
import oracle_joint_replay as oracle_replay

from openpi.models import model as model_api
from openpi.policies import policy_config
from openpi.training import config as training_config

CURRENT_FRAME = 59
FIRST_ACTION_FRAME = 60
GRASP_PROMPT = "The shell game has ended. Grasp and lift the cup containing the ball."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="pi0_mem_compress_evan_shellgame_openpi_joint_260727",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/pi0_mem_compress_evan_shellgame_openpi_joint_260727/my_experiment/23000"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../robosuite/outputs/shellgame_absolute_joint_dataset"),
    )
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=260806)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--selection-radius", type=float, default=0.06)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/shellgame/joint_fk_selection/23000_val50_s3.json"),
    )
    return parser.parse_args()


def _validation_episode_ids(num_episodes: int, val_ratio: float, seed: int) -> np.ndarray:
    episodes = np.arange(num_episodes, dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(episodes)
    num_val = min(max(1, round(num_episodes * val_ratio)), num_episodes - 1)
    return np.sort(shuffled[:num_val])


def _history_indices(current: int, num_frames: int, stride: int) -> list[int]:
    # Match examples/shellgame/main.py::_window exactly, including leading
    # frame-0 repetition used during online evaluation.
    return [max(0, current - (num_frames - 1 - i) * stride) for i in range(num_frames)]


def _cup_positions(metadata: dict) -> dict[str, np.ndarray]:
    frame = metadata["frames"][CURRENT_FRAME]
    cup_slots = frame["cup_slots"]
    offsets = metadata["cup_slot_offsets"]
    command = metadata["command_args"]
    spacing = float(command["cup_spacing"])
    slot_value = {"left": -spacing, "middle": 0.0, "right": spacing}
    positions = {}
    for cup, slot in cup_slots.items():
        axis_offset, cross_offset = (float(v) for v in offsets[slot])
        if command["layout_axis"] == "x":
            positions[cup] = np.asarray([slot_value[slot] + axis_offset, cross_offset], dtype=np.float32)
        else:
            positions[cup] = np.asarray([cross_offset, slot_value[slot] + axis_offset], dtype=np.float32)
    return positions


def _load_episode(episode_dir: Path, args: argparse.Namespace) -> dict:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
        wrist = np.asarray(source["wrist_images"])
        third_person = np.asarray(source["third_person_images"])
        joint_pos = np.asarray(source["joint_pos"], dtype=np.float32)
        gripper_state = np.asarray(source["gripper_state"], dtype=np.float32)
        eef_pos = np.asarray(source["eef_pos"], dtype=np.float32)

    indices = _history_indices(CURRENT_FRAME, args.num_frames, args.frame_stride)
    obs = {
        "robot0_joint_pos": joint_pos[CURRENT_FRAME].reshape(1, joint_eval.JOINT_DIM),
        "robot0_gripper_width": np.asarray([[base._gripper_width(gripper_state[CURRENT_FRAME])]], dtype=np.float32),
        "actions": np.zeros((16, joint_eval.ACTION_DIM), dtype=np.float32),
        "prompt": GRASP_PROMPT,
        # The online evaluator does not mark its frame-0 repetitions invalid.
        "video_frame_valid_mask": {
            "left_wrist_0_rgb_0": np.ones(args.num_frames, dtype=bool),
            "left_wrist_0_rgb_1": np.ones(args.num_frames, dtype=bool),
        },
    }
    for output_index, source_index in enumerate(indices):
        # Copy selected frames so the returned record does not keep both full
        # 155-frame episode arrays alive through NumPy views.  This matters for
        # in-training task eval, where many fixed episodes are cached.
        obs[f"left_wrist_0_rgb_0_{output_index}"] = wrist[source_index].copy()
        obs[f"left_wrist_0_rgb_1_{output_index}"] = third_person[source_index].copy()

    target_cup = str(metadata["target_cup_identity"])
    return {
        "episode": episode_dir.name,
        "metadata": metadata,
        "obs": obs,
        "target_cup": target_cup,
        "final_ball_slot": str(metadata["final_ball_cup"]),
        "cup_slots": dict(metadata["frames"][CURRENT_FRAME]["cup_slots"]),
        "cup_positions": _cup_positions(metadata),
        "reveal_wrist": wrist[:10].copy(),
        "reveal_third_person": third_person[:10].copy(),
        "reference_joint": joint_pos[FIRST_ACTION_FRAME : FIRST_ACTION_FRAME + 16].copy(),
        "reference_eef": eef_pos[FIRST_ACTION_FRAME : FIRST_ACTION_FRAME + 16].copy(),
    }


def _stack_dicts(items: list[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack(xs, axis=0), *items)


def _batched_infer(policy, observations: list[dict], batch_size: int, seed: int) -> list[np.ndarray]:
    outputs = []
    for start in range(0, len(observations), batch_size):
        raw_batch = observations[start : start + batch_size]
        transformed = [policy._input_transform(jax.tree.map(lambda x: x, obs)) for obs in raw_batch]
        valid_size = len(transformed)
        while len(transformed) < batch_size:
            transformed.append(transformed[-1])
        stacked = jax.tree.map(jnp.asarray, _stack_dicts(transformed))
        observation = model_api.Observation.from_dict(stacked)
        sample_rng = jax.random.key(seed + start // batch_size)
        actions = np.asarray(policy._sample_actions(sample_rng, observation, **policy._sample_kwargs))
        for index in range(valid_size):
            result = policy._output_transform(
                {
                    "state": np.asarray(transformed[index]["state"]),
                    "actions": actions[index],
                }
            )
            outputs.append(np.asarray(result["actions"], dtype=np.float32))
        logging.info("inference %d/%d", min(start + valid_size, len(observations)), len(observations))
    return outputs


def _make_fk_env(shell, first_record: dict):
    command = first_record["metadata"]["command_args"]
    eval_args = joint_eval.Args()
    for key, value in command.items():
        if hasattr(eval_args, key):
            setattr(eval_args, key, value)
    eval_args.gpu_id = -1
    eval_args.width = 64
    eval_args.height = 64
    eval_args.resize_size = 64
    eval_args.observe_eef_frames = 0
    ep_args = joint_eval._episode_namespace(
        eval_args,
        seed=int(command["seed"]),
        initial_ball_cup=str(command["initial_ball_cup"]),
        num_swaps=int(command["num_swaps"]),
    )
    env = shell.make_env(ep_args)
    oracle_replay._disable_image_observables(env)
    env.reset()
    return env


def _fk_chunk(shell, env, joint_chunk: np.ndarray) -> tuple[np.ndarray, int]:
    q_indices = np.asarray(env.robots[0]._ref_joint_pos_indexes, dtype=np.int64)
    action_low, action_high = (np.asarray(value, dtype=np.float32) for value in env.action_spec)
    q_low = action_low[: joint_eval.JOINT_DIM]
    q_high = action_high[: joint_eval.JOINT_DIM]
    clipped = np.clip(np.asarray(joint_chunk[:, : joint_eval.JOINT_DIM]), q_low, q_high)
    clipped_values = int(np.count_nonzero(np.abs(clipped - joint_chunk[:, : joint_eval.JOINT_DIM]) > 1e-6))
    eef = []
    for q_target in clipped:
        env.sim.data.qpos[q_indices] = q_target
        env.sim.forward()
        eef.append(shell.get_eef_pos(env))
    return np.asarray(eef, dtype=np.float32), clipped_values


def _classify(eef_xy: np.ndarray, cup_positions: dict[str, np.ndarray], radius: float) -> dict:
    distances = {cup: np.linalg.norm(eef_xy - position[None, :], axis=1) for cup, position in cup_positions.items()}
    endpoint_distances = {cup: float(values[-1]) for cup, values in distances.items()}
    endpoint_cup = min(endpoint_distances, key=endpoint_distances.get)
    min_path_distances = {cup: float(np.min(values)) for cup, values in distances.items()}
    min_path_cup = min(min_path_distances, key=min_path_distances.get)

    votes = dict.fromkeys(cup_positions, 0)
    vote_distance_sums = dict.fromkeys(cup_positions, 0.0)
    for step in range(min(10, len(eef_xy)), len(eef_xy)):
        step_distances = {cup: float(values[step]) for cup, values in distances.items()}
        nearest = min(step_distances, key=step_distances.get)
        if step_distances[nearest] <= radius:
            votes[nearest] += 1
            vote_distance_sums[nearest] += step_distances[nearest]
    max_votes = max(votes.values(), default=0)
    vote_cup = None
    if max_votes > 0:
        candidates = [cup for cup, count in votes.items() if count == max_votes]
        vote_cup = min(candidates, key=lambda cup: vote_distance_sums[cup] / votes[cup])
    return {
        "endpoint_cup": endpoint_cup,
        "endpoint_distances_m": endpoint_distances,
        "min_path_cup": min_path_cup,
        "min_path_distances_m": min_path_distances,
        "vote_cup": vote_cup,
        "votes": votes,
    }


def _majority(values: list[str | None]) -> str | None:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return None
    return collections.Counter(non_null).most_common(1)[0][0]


def _summarize(records: list[dict], samples: list[dict], args: argparse.Namespace) -> dict:
    episode_results = []
    for record in records:
        current = [sample for sample in samples if sample["episode"] == record["episode"]]
        endpoint_majority = _majority([sample["endpoint_cup"] for sample in current])
        vote_majority = _majority([sample["vote_cup"] for sample in current])
        episode_results.append(
            {
                "episode": record["episode"],
                "target_cup": record["target_cup"],
                "endpoint_majority_cup": endpoint_majority,
                "endpoint_majority_correct": endpoint_majority == record["target_cup"],
                "vote_majority_cup": vote_majority,
                "vote_majority_correct": vote_majority == record["target_cup"],
                "num_unique_endpoint_cups": len({sample["endpoint_cup"] for sample in current}),
            }
        )

    endpoint_correct = sum(sample["endpoint_correct"] for sample in samples)
    vote_decisions = sum(sample["vote_cup"] is not None for sample in samples)
    vote_correct = sum(sample["vote_correct"] for sample in samples)
    confusion = collections.defaultdict(collections.Counter)
    for sample in samples:
        confusion[sample["target_cup"]][sample["endpoint_cup"]] += 1
    return {
        "num_episodes": len(records),
        "samples_per_episode": args.samples_per_episode,
        "num_predictions": len(samples),
        "target_distribution": dict(collections.Counter(record["target_cup"] for record in records)),
        "reference_endpoint_accuracy": float(np.mean([record["reference_endpoint_correct"] for record in records])),
        "reference_vote_accuracy": float(np.mean([record["reference_vote_correct"] for record in records])),
        "reference_fk_endpoint_rmse_m": float(
            np.sqrt(np.mean([record["reference_fk_endpoint_sq_error"] for record in records]))
        ),
        "sample_endpoint_accuracy": endpoint_correct / len(samples),
        "sample_vote_decisions": vote_decisions,
        "sample_vote_decision_rate": vote_decisions / len(samples),
        "sample_vote_accuracy_all": vote_correct / len(samples),
        "sample_vote_accuracy_decided": vote_correct / vote_decisions if vote_decisions else 0.0,
        "episode_endpoint_majority_accuracy": float(
            np.mean([item["endpoint_majority_correct"] for item in episode_results])
        ),
        "episode_vote_majority_accuracy": float(np.mean([item["vote_majority_correct"] for item in episode_results])),
        "episode_endpoint_consistency_rate": float(
            np.mean([item["num_unique_endpoint_cups"] == 1 for item in episode_results])
        ),
        "mean_predicted_joint_mse": float(np.mean([sample["joint_mse"] for sample in samples])),
        "mean_endpoint_nearest_distance_m": float(
            np.mean([min(sample["endpoint_distances_m"].values()) for sample in samples])
        ),
        "clipped_joint_values": int(sum(sample["clipped_joint_values"] for sample in samples)),
        "endpoint_confusion": {target: dict(counts) for target, counts in confusion.items()},
        "episode_results": episode_results,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    episode_dirs = sorted(path for path in args.dataset_root.expanduser().resolve().glob("episode_*") if path.is_dir())
    val_ids = _validation_episode_ids(len(episode_dirs), args.val_ratio, args.split_seed)
    if not 0 < args.num_episodes <= len(val_ids):
        raise ValueError(f"--num-episodes must be in [1, {len(val_ids)}]")
    selected_ids = np.sort(
        np.random.default_rng(args.sample_seed).choice(val_ids, size=args.num_episodes, replace=False)
    )
    logging.info("Loading %d held-out validation episodes", len(selected_ids))
    records = [_load_episode(episode_dirs[int(index)], args) for index in selected_ids]

    shell = base._import_shellgame_tools(args.robosuite_root)
    fk_env = _make_fk_env(shell, records[0])
    try:
        for record in records:
            reference_fk, _ = _fk_chunk(shell, fk_env, record["reference_joint"])
            reference_class = _classify(reference_fk[:, :2], record["cup_positions"], args.selection_radius)
            record["reference_endpoint_correct"] = reference_class["endpoint_cup"] == record["target_cup"]
            record["reference_vote_correct"] = reference_class["vote_cup"] == record["target_cup"]
            record["reference_fk_endpoint_sq_error"] = float(
                np.mean(np.square(reference_fk[-1] - record["reference_eef"][-1]))
            )

        logging.info("Loading policy %s from %s", args.config, args.checkpoint_dir)
        config = training_config.get_config(args.config)
        policy = policy_config.create_trained_policy(
            config,
            args.checkpoint_dir,
            default_prompt=GRASP_PROMPT,
            sample_kwargs={"num_steps": args.num_sampling_steps},
        )
        requests = [(record, sample_index) for record in records for sample_index in range(args.samples_per_episode)]
        predicted_chunks = _batched_infer(
            policy,
            [record["obs"] for record, _ in requests],
            args.batch_size,
            args.sample_seed,
        )

        samples = []
        for (record, sample_index), predicted in zip(requests, predicted_chunks, strict=True):
            predicted_eef, clipped_values = _fk_chunk(shell, fk_env, predicted)
            classification = _classify(predicted_eef[:, :2], record["cup_positions"], args.selection_radius)
            samples.append(
                {
                    "episode": record["episode"],
                    "sample_index": sample_index,
                    "target_cup": record["target_cup"],
                    **classification,
                    "endpoint_correct": classification["endpoint_cup"] == record["target_cup"],
                    "vote_correct": classification["vote_cup"] == record["target_cup"],
                    "endpoint_eef_xyz_m": predicted_eef[-1].tolist(),
                    "joint_mse": float(
                        np.mean(np.square(predicted[:, : joint_eval.JOINT_DIM] - record["reference_joint"]))
                    ),
                    "clipped_joint_values": clipped_values,
                }
            )

        summary = _summarize(records, samples, args)
        output = {
            "config": args.config,
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "split": {
                "split_seed": args.split_seed,
                "val_ratio": args.val_ratio,
                "selected_episode_ids": selected_ids.tolist(),
            },
            "summary": summary,
            "samples": samples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key != "episode_results"}, indent=2))
        print(f"Wrote {args.output.resolve()}")
    finally:
        fk_env.close()


if __name__ == "__main__":
    main()
