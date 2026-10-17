#!/usr/bin/env python3
"""Stratified fixed-dev action sampling for PickXtimes Pi0.5 checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from openpi_client import image_tools

from openpi.policies import policy_config
from openpi.training import config as training_config
from openpi.training.mem.recipes import robomme_pickxtimes_pi_action as recipe

COLOR_TO_ID = {"red": 0, "green": 1, "blue": 2}
DEFAULT_DEV_ROOT = Path("data/robomme_pickxtimes_lerobot_pi_action_dev15_stride2")
DEFAULT_H5 = Path("data/robomme_extracted/record_dataset_PickXtimes.h5")
DEFAULT_MEMORY = Path("data/robomme_extracted/pickxtimes_action_memory_tokens_round9_train70_dev15.h5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trained-mode", choices=("predicted", "action_only"), required=True)
    parser.add_argument(
        "--input-memory-mode",
        choices=("predicted", "predicted_shuffled", "action_only"),
        required=True,
    )
    parser.add_argument("--dev-root", type=Path, default=DEFAULT_DEV_ROOT)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--memory-path", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--samples-per-phase", type=int, default=20)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--semantic-residual-gate-init", type=float, default=1.0)
    parser.add_argument("--semantic-residual-dropout-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=260824)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _selected_rows(root: Path, samples_per_phase: int, seed: int) -> list[dict]:
    episodes = _read_jsonl(root / "meta/episodes.jsonl")
    candidates = {phase: [] for phase in range(6)}
    for item in episodes:
        parquet = root / "data/chunk-000" / f"episode_{int(item['episode_index']):06d}.parquet"
        table = pq.read_table(parquet, columns=["frame_index", "phase_id", "action_mask"])
        frame = np.asarray(table["frame_index"], dtype=np.int64)
        phase = np.asarray(table["phase_id"], dtype=np.int64)
        full = np.asarray(table["action_mask"], dtype=bool)
        for row in np.flatnonzero(full):
            candidates[int(phase[row])].append({
                "converted_episode_index": int(item["episode_index"]),
                "source_episode_index": int(item["source_episode_index"]),
                "source_episode_name": item["source_episode_name"],
                "prompt": item["tasks"][0],
                "timestep": int(frame[row]),
                "phase_id": int(phase[row]),
            })
    selected = []
    for phase, rows in candidates.items():
        if len(rows) < samples_per_phase:
            raise ValueError(f"Phase {phase} has only {len(rows)} rows")
        indices = np.random.default_rng(seed + phase).choice(
            len(rows), size=samples_per_phase, replace=False
        )
        selected.extend(rows[int(index)] for index in indices)
    return selected


def _state(obs, source_episode_index: int, root: Path) -> np.ndarray:
    episodes = {int(item["source_episode_index"]): item for item in _read_jsonl(root / "meta/episodes.jsonl")}
    item = episodes[source_episode_index]
    # State goal fields are also in the converter metadata through source HDF5 labels,
    # but the one-hot/count values are recoverable from the natural-language prompt.
    prompt = item["tasks"][0].lower()
    color_name = next(color for color in COLOR_TO_ID if color in prompt)
    color = np.eye(3, dtype=np.float32)[COLOR_TO_ID[color_name]]
    word_to_count = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    count = next((value for word, value in word_to_count.items() if word in prompt), 1)
    eef = np.asarray(obs["eef_state"][()], dtype=np.float32).reshape(6)
    width = np.asarray(obs["gripper_state"][()], dtype=np.float32).sum(keepdims=True)
    return np.concatenate((eef, width, color, np.asarray([count / 5], dtype=np.float32)))


def _policy_input(source, item: dict, root: Path) -> tuple[dict, np.ndarray]:
    episode = source[item["source_episode_name"]]
    timestep = item["timestep"]
    obs = episode[f"timestep_{timestep}/obs"]
    target = np.stack([
        np.asarray(episode[f"timestep_{timestep + offset}/action/eef_action"][()], dtype=np.float32)
        for offset in range(1, 17)
    ])
    inputs = {
        "state_raw": _state(obs, item["source_episode_index"], root),
        "front_rgb_0": image_tools.resize_with_pad(np.asarray(obs["front_rgb"][()]), 224, 224),
        "wrist_rgb_0": image_tools.resize_with_pad(np.asarray(obs["wrist_rgb"][()]), 224, 224),
        "video_frame_valid_mask": {
            "front_rgb": np.ones(1, dtype=np.bool_),
            "wrist_rgb": np.ones(1, dtype=np.bool_),
        },
        "prompt": item["prompt"],
        "episode_index": np.int64(item["converted_episode_index"]),
        "frame_index": np.int64(timestep),
        "episode_T": np.float32(len(episode) - 1),
    }
    return inputs, target


def _load_policy(args):
    config = recipe.make_train_config(
        config_module=training_config,
        dataset_root=args.dev_root,
        memory_path=args.memory_path,
        memory_mode=args.trained_mode,
        exp_name="fixed_dev_eval_only",
        steps=2,
        batch_size=1,
        fsdp_devices=1,
        num_workers=0,
        eval_interval=1,
        eval_batches=1,
        save_interval=1,
        semantic_residual_gate_init=args.semantic_residual_gate_init,
        semantic_residual_dropout_rate=args.semantic_residual_dropout_rate,
    )
    config = dataclasses.replace(
        config,
        fsdp_devices=1,
        data=dataclasses.replace(config.data, memory_mode=args.input_memory_mode),
    )
    return policy_config.create_trained_policy(
        config,
        args.checkpoint,
        sample_kwargs={"num_steps": args.num_sampling_steps},
    )


def _metrics(records: list[dict]) -> dict:
    position = np.asarray([record["position_error_m"] for record in records])
    first_position = np.asarray([record["first_position_error_m"] for record in records])
    rotation = np.asarray([record["rotation_error_rad"] for record in records])
    gripper = np.asarray([record["gripper_accuracy"] for record in records])
    return {
        "samples": len(records),
        "position_error_cm": float(position.mean() * 100),
        "first_position_error_cm": float(first_position.mean() * 100),
        "rotation_error_rad": float(rotation.mean()),
        "gripper_accuracy": float(gripper.mean()),
    }


def main() -> None:
    args = parse_args()
    args.dev_root = args.dev_root.resolve()
    selected = _selected_rows(args.dev_root, args.samples_per_phase, args.seed)
    policy = _load_policy(args)
    records = []
    with h5py.File(args.h5.resolve(), "r") as source:
        for ordinal, item in enumerate(selected, start=1):
            inputs, target = _policy_input(source, item, args.dev_root)
            noise = np.random.default_rng(args.seed + ordinal).standard_normal((16, 32)).astype(np.float32)
            prediction = np.asarray(policy.infer(inputs, noise=noise)["actions"], dtype=np.float32)
            position = np.linalg.norm(prediction[:, :3] - target[:, :3], axis=-1)
            angle_delta = (prediction[:, 3:6] - target[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
            gripper_accuracy = np.mean((prediction[:, 6] < 0) == (target[:, 6] < 0))
            records.append({
                **item,
                "position_error_m": float(position.mean()),
                "first_position_error_m": float(position[0]),
                "rotation_error_rad": float(np.linalg.norm(angle_delta, axis=-1).mean()),
                "gripper_accuracy": float(gripper_accuracy),
            })
            if ordinal % 20 == 0:
                print(f"[{ordinal}/{len(selected)}]", flush=True)
    by_phase = {
        str(phase): _metrics([record for record in records if record["phase_id"] == phase])
        for phase in range(6)
    }
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "trained_mode": args.trained_mode,
        "input_memory_mode": args.input_memory_mode,
        "dev_root": str(args.dev_root),
        "frozen_test_accessed": False,
        "num_sampling_steps": args.num_sampling_steps,
        "semantic_residual_gate_init": args.semantic_residual_gate_init,
        "semantic_residual_dropout_rate": args.semantic_residual_dropout_rate,
        "samples_per_phase": args.samples_per_phase,
        "metrics": _metrics(records),
        "by_phase": by_phase,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": payload["metrics"], "by_phase": by_phase}, indent=2))


if __name__ == "__main__":
    main()
