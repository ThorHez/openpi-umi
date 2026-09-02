#!/usr/bin/env python3
"""Train a frozen-memory PickXtimes EEF7 action-chunk adapter."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.tasks.robomme.pickxtimes import eef_action_adapter
from openpi.tasks.robomme.pickxtimes import eef_action_chunk_adapter
from openpi.training.mem import robomme_pickxtimes_action_chunk_dataset as chunk_data
from openpi.training.mem import robomme_pickxtimes_action_dataset as action_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=pathlib.Path, required=True)
    parser.add_argument("--chunk-cache", type=pathlib.Path, required=True)
    parser.add_argument("--spatial-cache", type=pathlib.Path)
    parser.add_argument("--split", type=pathlib.Path, required=True)
    parser.add_argument("--memory-mode", choices=action_data.MEMORY_MODES, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--memory-query-tokens", type=int, default=8)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--phase-loss-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=260825)
    return parser.parse_args()


def _batch(arrays: chunk_data.ActionChunkArrays, indices: np.ndarray) -> dict[str, jax.Array]:
    return {
        "visual_features": jnp.asarray(arrays.visual_features[indices]),
        "robot_goal": jnp.asarray(arrays.robot_goal[indices]),
        "memory": jnp.asarray(arrays.memory_bank[arrays.memory_indices[indices]]),
        "poses": jnp.asarray(arrays.poses[indices]),
        "close_targets": jnp.asarray(arrays.close_targets[indices]),
        "action_mask": jnp.asarray(arrays.action_mask[indices]),
        "phase_targets": jnp.asarray(arrays.phase_targets[indices]),
    }


def _phase_balanced_indices(rng, targets: np.ndarray, batch_size: int) -> np.ndarray:
    groups = [np.flatnonzero(targets == phase) for phase in range(eef_action_adapter.NUM_PHASES)]
    counts = [batch_size // len(groups)] * len(groups)
    for index in range(batch_size % len(groups)):
        counts[index] += 1
    selected = np.concatenate(
        [rng.choice(group, size=count, replace=len(group) < count) for group, count in zip(groups, counts, strict=True)]
    )
    rng.shuffle(selected)
    return selected


def _save(params, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flax.serialization.to_bytes(jax.device_get(params)))


def main() -> None:
    args = parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_episodes = [int(value) for value in split["train_episode_indices"]]
    dev_episodes = [int(value) for value in split["val_episode_indices"]]
    train_raw = chunk_data.load_action_chunk_arrays(
        args.base_cache,
        args.chunk_cache,
        episode_indices=train_episodes,
        memory_mode=args.memory_mode,
    )
    dev_raw = chunk_data.load_action_chunk_arrays(
        args.base_cache,
        args.chunk_cache,
        episode_indices=dev_episodes,
        memory_mode=args.memory_mode,
    )
    if args.spatial_cache is not None:
        train_raw = dataclasses.replace(
            train_raw,
            visual_features=chunk_data.load_spatial_visual_features(
                args.spatial_cache,
                episode_indices=train_episodes,
                expected_episodes=train_raw.episode_indices,
                expected_timesteps=train_raw.timesteps,
            ),
        )
        dev_raw = dataclasses.replace(
            dev_raw,
            visual_features=chunk_data.load_spatial_visual_features(
                args.spatial_cache,
                episode_indices=dev_episodes,
                expected_episodes=dev_raw.episode_indices,
                expected_timesteps=dev_raw.timesteps,
            ),
        )
    stats = chunk_data.compute_normalization(train_raw)
    train = chunk_data.normalize_arrays(train_raw, stats)
    dev = chunk_data.normalize_arrays(dev_raw, stats)
    action_horizon = train.poses.shape[1]
    spatial_visual_tokens = train.visual_features.shape[1] if train.visual_features.ndim == 3 else 0
    close_rate = float(np.sum(train.close_targets * train.action_mask) / np.sum(train.action_mask))
    positive_gripper_weight = (1.0 - close_rate) / max(close_rate, 1e-6)
    print(
        f"mode={args.memory_mode} horizon={action_horizon} train={len(train)} dev={len(dev)} "
        f"close_rate={close_rate:.3f}",
        flush=True,
    )

    model = eef_action_chunk_adapter.PickXtimesEEFActionChunkAdapter(
        action_horizon=action_horizon,
        hidden_width=args.hidden_width,
        depth=args.depth,
        memory_query_tokens=args.memory_query_tokens,
        use_memory=args.memory_mode != "action_only",
        spatial_visual_tokens=spatial_visual_tokens,
    )
    init_rng, dropout_rng = jax.random.split(jax.random.key(args.seed))
    variables = model.init(
        {"params": init_rng, "dropout": init_rng},
        jnp.zeros(
            (args.batch_size, spatial_visual_tokens, eef_action_adapter.VISUAL_FEATURE_DIM)
            if spatial_visual_tokens
            else (args.batch_size, eef_action_adapter.VISUAL_FEATURE_DIM),
            dtype=jnp.float16,
        ),
        jnp.zeros((args.batch_size, eef_action_adapter.ROBOT_GOAL_DIM), dtype=jnp.float32),
        jnp.zeros(
            (args.batch_size, eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH),
            dtype=jnp.float16,
        ),
        train=True,
    )
    params = variables["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        args.learning_rate,
        min(args.warmup_steps, args.steps),
        args.steps,
        end_value=args.learning_rate * 0.1,
    )
    optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)
    pose_std = jnp.asarray(stats.pose_std)

    def objective(current_params, batch, *, train_mode: bool, rng=None):
        outputs = model.apply(
            {"params": current_params},
            batch["visual_features"],
            batch["robot_goal"],
            batch["memory"],
            train=train_mode,
            rngs={"dropout": rng} if rng is not None else None,
        )
        mask = batch["action_mask"]
        pose_error = outputs["normalized_poses"] - batch["poses"]
        pose_loss = jnp.sum(jnp.mean(jnp.square(pose_error), axis=-1) * mask) / jnp.sum(mask)
        gripper_bce = optax.sigmoid_binary_cross_entropy(outputs["close_logits"], batch["close_targets"])
        gripper_weights = jnp.where(batch["close_targets"] > 0.5, positive_gripper_weight, 1.0) * mask
        gripper_loss = jnp.sum(gripper_bce * gripper_weights) / jnp.sum(gripper_weights)
        phase_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(outputs["phase_logits"], batch["phase_targets"])
        )
        physical_error = pose_error * pose_std
        element_mask = mask[..., None]
        denominator = jnp.sum(mask) * 3.0
        metrics = {
            "loss": pose_loss + args.gripper_loss_weight * gripper_loss + args.phase_loss_weight * phase_loss,
            "pose_loss": pose_loss,
            "gripper_loss": gripper_loss,
            "phase_loss": phase_loss,
            "chunk_position_mae_cm": jnp.sum(jnp.abs(physical_error[..., :3]) * element_mask) / denominator * 100.0,
            "first_position_mae_cm": jnp.mean(jnp.abs(physical_error[:, 0, :3])) * 100.0,
            "chunk_rotation_mae_deg": jnp.sum(jnp.abs(physical_error[..., 3:]) * element_mask)
            / denominator
            * (180.0 / jnp.pi),
            "gripper_accuracy": jnp.sum(
                ((outputs["close_logits"] > 0.0) == (batch["close_targets"] > 0.5)) * mask
            )
            / jnp.sum(mask),
            "phase_accuracy": jnp.mean(jnp.argmax(outputs["phase_logits"], axis=-1) == batch["phase_targets"]),
        }
        return metrics["loss"], metrics

    @jax.jit
    def train_step(current_params, current_opt_state, batch, rng):
        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
            current_params, batch, train_mode=True, rng=rng
        )
        updates, next_opt_state = optimizer.update(gradients, current_opt_state, current_params)
        return optax.apply_updates(current_params, updates), next_opt_state, metrics

    @jax.jit
    def eval_step(current_params, batch):
        return objective(current_params, batch, train_mode=False)[1]

    def evaluate(current_params):
        weighted = {}
        total = 0
        for start in range(0, len(dev), args.eval_batch_size):
            indices = np.arange(start, min(start + args.eval_batch_size, len(dev)))
            metrics = jax.device_get(eval_step(current_params, _batch(dev, indices)))
            total += len(indices)
            for key, value in metrics.items():
                weighted[key] = weighted.get(key, 0.0) + float(value) * len(indices)
        return {key: value / total for key, value in weighted.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(int(np.size(value)) for value in jax.tree_util.tree_leaves(params))
    config = vars(args).copy()
    config.update(
        action_horizon=action_horizon,
        spatial_visual_tokens=spatial_visual_tokens,
        train_episode_indices=train_episodes,
        dev_episode_indices=dev_episodes,
        train_rows=len(train),
        dev_rows=len(dev),
        close_rate=close_rate,
        positive_gripper_weight=positive_gripper_weight,
        parameter_count=parameter_count,
        frozen_memory=True,
        frozen_test_accessed=False,
        normalization=stats.to_json(),
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_step = 0
    started = time.monotonic()
    with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            indices = _phase_balanced_indices(rng, train.phase_targets, args.batch_size)
            dropout_rng, step_rng = jax.random.split(dropout_rng)
            params, opt_state, metrics = train_step(params, opt_state, _batch(train, indices), step_rng)
            if step == 1 or step % 20 == 0:
                host = {key: float(value) for key, value in jax.device_get(metrics).items()}
                metrics_file.write(json.dumps({"step": step, "split": "train", **host}) + "\n")
                metrics_file.flush()
                print(
                    f"step={step}/{args.steps} loss={host['loss']:.4f} "
                    f"chunk_pos={host['chunk_position_mae_cm']:.2f}cm "
                    f"phase={host['phase_accuracy']:.3f} elapsed={(time.monotonic() - started) / 60:.1f}m",
                    flush=True,
                )
            if step % args.eval_every == 0 or step == args.steps:
                host = evaluate(params)
                metrics_file.write(json.dumps({"step": step, "split": "dev", **host}) + "\n")
                metrics_file.flush()
                print(
                    f"DEV step={step} loss={host['loss']:.4f} "
                    f"chunk_pos={host['chunk_position_mae_cm']:.2f}cm "
                    f"first_pos={host['first_position_mae_cm']:.2f}cm "
                    f"grip={host['gripper_accuracy']:.3f} phase={host['phase_accuracy']:.3f}",
                    flush=True,
                )
                if host["loss"] < best_loss:
                    best_loss = host["loss"]
                    best_step = step
                    _save(params, args.output_dir / "checkpoints/best.msgpack")
            if step % args.save_every == 0 or step == args.steps:
                _save(params, args.output_dir / f"checkpoints/step_{step}.msgpack")
    (args.output_dir / "summary.json").write_text(
        json.dumps({"memory_mode": args.memory_mode, "best_step": best_step, "best_dev_loss": best_loss}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
