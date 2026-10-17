"""Paired history ablations for ShellGame joint-to-FK cup selection."""

# This diagnostic intentionally uses Policy / evaluator internals so every
# non-ablated preprocessing and sampling detail matches online inference.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
from pathlib import Path

import joint_fk_selection_eval as fk_eval
import main as base
import numpy as np

from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as training_config

MODES = ("normal", "memory_off", "shuffle_history", "wrong_history", "reveal_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi0_mem_compress_evan_shellgame_openpi_joint_260727")
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
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/shellgame/joint_fk_memory_ablation/23000_val50_s3.json"),
    )
    return parser.parse_args()


def _history_key(stream: str, index: int) -> str:
    return f"{stream}_{index}"


def _clone_obs(record: dict) -> dict:
    return dict(record["obs"])


def _mode_observation(
    record: dict,
    mode: str,
    *,
    donor: dict | None,
    seed: int,
    num_frames: int,
    frame_stride: int,
) -> dict:
    obs = _clone_obs(record)
    history_count = num_frames - 1
    streams = ("left_wrist_0_rgb_0", "left_wrist_0_rgb_1")

    if mode in ("normal", "memory_off"):
        return obs

    if mode == "shuffle_history":
        episode_id = int(record["episode"].split("_")[-1])
        permutation = np.random.default_rng(seed + episode_id).permutation(history_count)
        for stream in streams:
            original = [record["obs"][_history_key(stream, index)] for index in range(history_count)]
            for output_index, source_index in enumerate(permutation):
                obs[_history_key(stream, output_index)] = original[int(source_index)]
        return obs

    if mode == "wrong_history":
        if donor is None:
            raise ValueError("wrong_history requires a donor")
        for stream in streams:
            for index in range(history_count):
                obs[_history_key(stream, index)] = donor["obs"][_history_key(stream, index)]
        return obs

    if mode == "reveal_only":
        source_indices = fk_eval._history_indices(fk_eval.CURRENT_FRAME, num_frames, frame_stride)
        # Keep the online padding pattern. Once a requested source frame moves
        # beyond reveal, hold the last reveal frame (frame 9), erasing all
        # cover / swap / settle evidence while preserving the revealed ball.
        for output_index, source_index in enumerate(source_indices[:-1]):
            reveal_index = min(int(source_index), 9)
            obs[_history_key("left_wrist_0_rgb_0", output_index)] = record["reveal_wrist"][reveal_index]
            obs[_history_key("left_wrist_0_rgb_1", output_index)] = record["reveal_third_person"][reveal_index]
        return obs

    raise ValueError(f"Unknown mode {mode}")


def _choose_donors(records: list[dict]) -> dict[str, dict]:
    donors = {}
    for index, record in enumerate(records):
        for offset in range(1, len(records) + 1):
            candidate = records[(index + offset) % len(records)]
            if candidate["final_ball_slot"] != record["final_ball_slot"]:
                donors[record["episode"]] = candidate
                break
        if record["episode"] not in donors:
            raise RuntimeError("Could not find a wrong-history donor with a different final slot")
    return donors


def _cup_in_slot(record: dict, slot: str) -> str:
    matches = [cup for cup, current_slot in record["cup_slots"].items() if current_slot == slot]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one cup in slot {slot!r}, got {matches}")
    return matches[0]


def _set_fixed_memory_gate(policy, value: float) -> None:
    module = policy._model.PaliGemma.img.module
    policy._model.PaliGemma.img.module = dataclasses.replace(module, history_gate_fixed=float(value))
    policy._sample_actions = nnx_utils.module_jit(policy._model.sample_actions)


def _majority(values: list[str | None]) -> str | None:
    values = [value for value in values if value is not None]
    return collections.Counter(values).most_common(1)[0][0] if values else None


def _summarize_mode(records: list[dict], samples: list[dict], mode: str) -> dict:
    episode_results = []
    for record in records:
        current = [sample for sample in samples if sample["episode"] == record["episode"]]
        endpoint_majority = _majority([sample["endpoint_cup"] for sample in current])
        vote_majority = _majority([sample["vote_cup"] for sample in current])
        item = {
            "episode": record["episode"],
            "target_cup": record["target_cup"],
            "endpoint_majority_cup": endpoint_majority,
            "endpoint_majority_correct": endpoint_majority == record["target_cup"],
            "vote_majority_cup": vote_majority,
            "vote_majority_correct": vote_majority == record["target_cup"],
            "num_unique_endpoint_cups": len({sample["endpoint_cup"] for sample in current}),
        }
        if mode == "wrong_history":
            expected = current[0]["donor_expected_cup"]
            item["donor_expected_cup"] = expected
            item["endpoint_majority_follows_donor"] = endpoint_majority == expected
        if mode == "reveal_only":
            expected = current[0]["initial_slot_expected_cup"]
            item["initial_slot_expected_cup"] = expected
            item["endpoint_majority_follows_initial_slot"] = endpoint_majority == expected
        episode_results.append(item)

    vote_decisions = sum(sample["vote_cup"] is not None for sample in samples)
    summary = {
        "num_predictions": len(samples),
        "sample_endpoint_accuracy": float(np.mean([sample["endpoint_correct"] for sample in samples])),
        "sample_vote_decision_rate": vote_decisions / len(samples),
        "sample_vote_accuracy_all": float(np.mean([sample["vote_correct"] for sample in samples])),
        "episode_endpoint_majority_accuracy": float(
            np.mean([item["endpoint_majority_correct"] for item in episode_results])
        ),
        "episode_vote_majority_accuracy": float(np.mean([item["vote_majority_correct"] for item in episode_results])),
        "episode_endpoint_consistency_rate": float(
            np.mean([item["num_unique_endpoint_cups"] == 1 for item in episode_results])
        ),
        "predicted_distribution": dict(collections.Counter(sample["endpoint_cup"] for sample in samples)),
        "mean_predicted_joint_mse": float(np.mean([sample["joint_mse"] for sample in samples])),
        "mean_endpoint_nearest_distance_m": float(
            np.mean([min(sample["endpoint_distances_m"].values()) for sample in samples])
        ),
        "clipped_joint_values": int(sum(sample["clipped_joint_values"] for sample in samples)),
        "episode_results": episode_results,
    }
    if mode == "wrong_history":
        summary["sample_follows_donor_slot_rate"] = float(
            np.mean([sample["endpoint_cup"] == sample["donor_expected_cup"] for sample in samples])
        )
        summary["episode_majority_follows_donor_slot_rate"] = float(
            np.mean([item["endpoint_majority_follows_donor"] for item in episode_results])
        )
    if mode == "reveal_only":
        summary["sample_follows_initial_slot_rate"] = float(
            np.mean([sample["endpoint_cup"] == sample["initial_slot_expected_cup"] for sample in samples])
        )
        summary["episode_majority_follows_initial_slot_rate"] = float(
            np.mean([item["endpoint_majority_follows_initial_slot"] for item in episode_results])
        )
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    episode_dirs = sorted(path for path in args.dataset_root.expanduser().resolve().glob("episode_*") if path.is_dir())
    val_ids = fk_eval._validation_episode_ids(len(episode_dirs), args.val_ratio, args.split_seed)
    if not 0 < args.num_episodes <= len(val_ids):
        raise ValueError(f"--num-episodes must be in [1, {len(val_ids)}]")
    selected_ids = np.sort(
        np.random.default_rng(args.sample_seed).choice(val_ids, size=args.num_episodes, replace=False)
    )
    records = [fk_eval._load_episode(episode_dirs[int(index)], args) for index in selected_ids]
    donors = _choose_donors(records)
    requests = [(record, sample_index) for record in records for sample_index in range(args.samples_per_episode)]

    shell = base._import_shellgame_tools(args.robosuite_root)
    fk_env = fk_eval._make_fk_env(shell, records[0])
    try:
        config = training_config.get_config(args.config)
        policy = policy_config.create_trained_policy(
            config,
            args.checkpoint_dir,
            default_prompt=fk_eval.GRASP_PROMPT,
            sample_kwargs={"num_steps": args.num_sampling_steps},
        )
        original_gate = float(policy._model.PaliGemma.img.module.history_gate_fixed)
        all_samples: dict[str, list[dict]] = {}
        summaries = {}

        for mode in args.modes:
            logging.info("=== mode=%s ===", mode)
            _set_fixed_memory_gate(policy, 0.0 if mode == "memory_off" else original_gate)
            observations = [
                _mode_observation(
                    record,
                    mode,
                    donor=donors[record["episode"]],
                    seed=args.sample_seed,
                    num_frames=args.num_frames,
                    frame_stride=args.frame_stride,
                )
                for record, _ in requests
            ]
            predicted_chunks = fk_eval._batched_infer(
                policy,
                observations,
                args.batch_size,
                args.sample_seed,
            )
            samples = []
            for (record, sample_index), predicted in zip(requests, predicted_chunks, strict=True):
                predicted_eef, clipped_values = fk_eval._fk_chunk(shell, fk_env, predicted)
                classification = fk_eval._classify(predicted_eef[:, :2], record["cup_positions"], args.selection_radius)
                donor = donors[record["episode"]]
                sample = {
                    "episode": record["episode"],
                    "sample_index": sample_index,
                    "target_cup": record["target_cup"],
                    **classification,
                    "endpoint_correct": classification["endpoint_cup"] == record["target_cup"],
                    "vote_correct": classification["vote_cup"] == record["target_cup"],
                    "joint_mse": float(
                        np.mean(np.square(predicted[:, : fk_eval.joint_eval.JOINT_DIM] - record["reference_joint"]))
                    ),
                    "clipped_joint_values": clipped_values,
                }
                if mode == "wrong_history":
                    sample.update(
                        {
                            "donor_episode": donor["episode"],
                            "donor_final_ball_slot": donor["final_ball_slot"],
                            "donor_expected_cup": _cup_in_slot(record, donor["final_ball_slot"]),
                        }
                    )
                if mode == "reveal_only":
                    sample["initial_slot_expected_cup"] = _cup_in_slot(record, record["target_cup"])
                samples.append(sample)
            all_samples[mode] = samples
            summaries[mode] = _summarize_mode(records, samples, mode)

        _set_fixed_memory_gate(policy, original_gate)
        normal_lookup = {
            (sample["episode"], sample["sample_index"]): sample["endpoint_cup"]
            for sample in all_samples.get("normal", [])
        }
        if normal_lookup:
            for mode, samples in all_samples.items():
                summaries[mode]["paired_endpoint_change_vs_normal_rate"] = float(
                    np.mean(
                        [
                            sample["endpoint_cup"] != normal_lookup[(sample["episode"], sample["sample_index"])]
                            for sample in samples
                        ]
                    )
                )

        compact = {
            mode: {key: value for key, value in summary.items() if key != "episode_results"}
            for mode, summary in summaries.items()
        }
        output = {
            "config": args.config,
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "modes": args.modes,
            "paired_diffusion_noise": True,
            "split": {
                "split_seed": args.split_seed,
                "val_ratio": args.val_ratio,
                "selected_episode_ids": selected_ids.tolist(),
            },
            "summaries": summaries,
            "samples": all_samples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(compact, indent=2, sort_keys=True))
        print(f"Wrote {args.output.resolve()}")
    finally:
        fk_env.close()


if __name__ == "__main__":
    main()
