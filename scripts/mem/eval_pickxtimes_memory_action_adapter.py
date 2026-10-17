#!/usr/bin/env python3
"""Evaluate a PickXtimes memory-action adapter on the fixed dev split."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.tasks.robomme.pickxtimes import eef_action_adapter
from openpi.training.mem import robomme_pickxtimes_action_dataset as action_data

PHASE_NAMES = ("PICK", "PLACE", "PRESS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--memory-control", choices=("normal", "shuffled", "zero"), default="normal")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or args.run_dir / "checkpoints/best.msgpack"
    split = json.loads(pathlib.Path(config["split"]).read_text(encoding="utf-8"))
    dev_indices = [int(value) for value in split["val_episode_indices"]]
    raw = action_data.load_action_arrays(
        config["cache"], episode_indices=dev_indices, memory_mode=config["memory_mode"]
    )
    stats = action_data.ActionNormalization.from_json(config["normalization"])
    arrays = action_data.normalize_arrays(raw, stats)
    if args.memory_control == "shuffled":
        rng = np.random.default_rng(260824)
        arrays = dataclasses.replace(arrays, memory_indices=rng.permutation(arrays.memory_indices))
    elif args.memory_control == "zero":
        arrays = dataclasses.replace(
            arrays,
            memory_bank=np.zeros(
                (1, eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH), dtype=np.float16
            ),
            memory_indices=np.zeros(len(arrays), dtype=np.int32),
        )

    model = eef_action_adapter.PickXtimesEEFActionAdapter(
        hidden_width=int(config["hidden_width"]),
        depth=int(config["depth"]),
        memory_query_tokens=int(config["memory_query_tokens"]),
        use_memory=config["memory_mode"] != "action_only",
    )
    variables = model.init(
        jax.random.key(0),
        jnp.zeros((1, eef_action_adapter.VISUAL_FEATURE_DIM), dtype=jnp.float16),
        jnp.zeros((1, eef_action_adapter.ROBOT_GOAL_DIM), dtype=jnp.float32),
        jnp.zeros((1, eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH), dtype=jnp.float16),
        train=False,
    )
    params = flax.serialization.from_bytes(variables["params"], checkpoint.read_bytes())

    @jax.jit
    def predict(visual, robot_goal, memory):
        return model.apply({"params": params}, visual, robot_goal, memory, train=False)

    pose_predictions = []
    close_logits = []
    phase_predictions = []
    for start in range(0, len(arrays), args.batch_size):
        indices = np.arange(start, min(start + args.batch_size, len(arrays)))
        memory = arrays.memory_bank[arrays.memory_indices[indices]]
        outputs = jax.device_get(
            predict(
                jnp.asarray(arrays.visual_features[indices]),
                jnp.asarray(arrays.robot_goal[indices]),
                jnp.asarray(memory),
            )
        )
        pose_predictions.append(np.asarray(outputs["normalized_pose"]))
        close_logits.append(np.asarray(outputs["close_logit"]))
        phase_predictions.append(np.argmax(np.asarray(outputs["phase_logits"]), axis=-1))
    normalized_pose = np.concatenate(pose_predictions)
    close_logit = np.concatenate(close_logits)
    predicted_phase = np.concatenate(phase_predictions)
    physical_error = (normalized_pose - arrays.poses) * stats.pose_std
    close_prediction = close_logit > 0.0
    close_target = arrays.close_targets > 0.5

    def metrics(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "rows": int(mask.sum()),
            "position_mae_cm": float(np.mean(np.abs(physical_error[mask, :3])) * 100.0),
            "rotation_mae_deg": float(np.mean(np.abs(physical_error[mask, 3:])) * (180.0 / np.pi)),
            "gripper_accuracy": float(np.mean(close_prediction[mask] == close_target[mask])),
            "phase_accuracy": float(np.mean(predicted_phase[mask] == arrays.phase_targets[mask])),
        }

    pose_loss = float(np.mean(np.square(normalized_pose - arrays.poses)))
    gripper_bce = np.asarray(
        optax.sigmoid_binary_cross_entropy(jnp.asarray(close_logit), jnp.asarray(arrays.close_targets))
    )
    positive_weight = float(config["positive_gripper_weight"])
    gripper_weights = np.where(arrays.close_targets > 0.5, positive_weight, 1.0)
    gripper_loss = float(np.sum(gripper_bce * gripper_weights) / np.sum(gripper_weights))
    result = {
        "schema_version": 1,
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "selected_training_step": int(summary["best_step"]),
        "memory_mode": config["memory_mode"],
        "memory_control": args.memory_control,
        "frozen_memory": True,
        "frozen_test_accessed": False,
        "pose_loss": pose_loss,
        "gripper_loss": gripper_loss,
        "overall": metrics(np.ones(len(arrays), dtype=np.bool_)),
        "by_phase": {
            name: metrics(arrays.phase_targets == phase) for phase, name in enumerate(PHASE_NAMES)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
