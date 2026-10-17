#!/usr/bin/env python3
"""Train a frozen-memory PickXtimes next-step EEF7 action adapter."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.tasks.robomme.pickxtimes import eef_action_adapter
from openpi.training.mem import robomme_pickxtimes_action_dataset as action_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=pathlib.Path, required=True)
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
    parser.add_argument("--seed", type=int, default=260824)
    return parser.parse_args()


def _batch(arrays: action_data.ActionArrays, indices: np.ndarray) -> dict[str, jax.Array]:
    memory_indices = arrays.memory_indices[indices]
    return {
        "visual_features": jnp.asarray(arrays.visual_features[indices]),
        "robot_goal": jnp.asarray(arrays.robot_goal[indices]),
        "memory": jnp.asarray(arrays.memory_bank[memory_indices]),
        "poses": jnp.asarray(arrays.poses[indices]),
        "close_targets": jnp.asarray(arrays.close_targets[indices]),
        "phase_targets": jnp.asarray(arrays.phase_targets[indices]),
    }


def _phase_balanced_indices(
    rng: np.random.Generator,
    phase_targets: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    groups = [np.flatnonzero(phase_targets == phase) for phase in range(eef_action_adapter.NUM_PHASES)]
    if any(len(group) == 0 for group in groups):
        raise ValueError(f"Empty PickXtimes phase: {[len(group) for group in groups]}")
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
    if min(
        args.steps,
        args.batch_size,
        args.warmup_steps,
        args.eval_every,
        args.eval_batch_size,
        args.save_every,
        args.hidden_width,
        args.depth,
        args.memory_query_tokens,
    ) < 1:
        raise ValueError("Training count and width arguments must be positive")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_indices = [int(value) for value in split["train_episode_indices"]]
    dev_indices = [int(value) for value in split["val_episode_indices"]]
    train_raw = action_data.load_action_arrays(
        args.cache, episode_indices=train_indices, memory_mode=args.memory_mode
    )
    dev_raw = action_data.load_action_arrays(
        args.cache, episode_indices=dev_indices, memory_mode=args.memory_mode
    )
    stats = action_data.compute_normalization(train_raw)
    train = action_data.normalize_arrays(train_raw, stats)
    dev = action_data.normalize_arrays(dev_raw, stats)
    phase_counts = np.bincount(train.phase_targets, minlength=eef_action_adapter.NUM_PHASES)
    close_rate = float(train.close_targets.mean())
    positive_gripper_weight = (1.0 - close_rate) / max(close_rate, 1e-6)
    print(
        f"mode={args.memory_mode} train={len(train)} dev={len(dev)} "
        f"phase={phase_counts.tolist()} close_rate={close_rate:.3f}",
        flush=True,
    )

    model = eef_action_adapter.PickXtimesEEFActionAdapter(
        hidden_width=args.hidden_width,
        depth=args.depth,
        memory_query_tokens=args.memory_query_tokens,
        use_memory=args.memory_mode != "action_only",
    )
    init_rng, dropout_rng = jax.random.split(jax.random.key(args.seed))
    variables = model.init(
        {"params": init_rng, "dropout": init_rng},
        jnp.zeros((args.batch_size, eef_action_adapter.VISUAL_FEATURE_DIM), dtype=jnp.float16),
        jnp.zeros((args.batch_size, eef_action_adapter.ROBOT_GOAL_DIM), dtype=jnp.float32),
        jnp.zeros(
            (args.batch_size, eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH),
            dtype=jnp.float16,
        ),
        train=True,
    )
    params = variables["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=min(args.warmup_steps, args.steps),
        decay_steps=args.steps,
        end_value=args.learning_rate * 0.1,
    )
    optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)
    pose_std = jnp.asarray(stats.pose_std)

    def objective(current_params, batch, *, train_mode: bool, rng=None):
        rngs = {"dropout": rng} if rng is not None else None
        outputs = model.apply(
            {"params": current_params},
            batch["visual_features"],
            batch["robot_goal"],
            batch["memory"],
            train=train_mode,
            rngs=rngs,
        )
        pose_error = outputs["normalized_pose"] - batch["poses"]
        pose_loss = jnp.mean(jnp.square(pose_error))
        gripper_bce = optax.sigmoid_binary_cross_entropy(outputs["close_logit"], batch["close_targets"])
        gripper_weights = jnp.where(batch["close_targets"] > 0.5, positive_gripper_weight, 1.0)
        gripper_loss = jnp.sum(gripper_bce * gripper_weights) / jnp.sum(gripper_weights)
        phase_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(outputs["phase_logits"], batch["phase_targets"])
        )
        physical_pose_error = pose_error * pose_std
        metrics = {
            "loss": pose_loss + args.gripper_loss_weight * gripper_loss + args.phase_loss_weight * phase_loss,
            "pose_loss": pose_loss,
            "gripper_loss": gripper_loss,
            "phase_loss": phase_loss,
            "position_mae_cm": jnp.mean(jnp.abs(physical_pose_error[:, :3])) * 100.0,
            "rotation_mae_deg": jnp.mean(jnp.abs(physical_pose_error[:, 3:])) * (180.0 / jnp.pi),
            "gripper_accuracy": jnp.mean(
                (outputs["close_logit"] > 0.0) == (batch["close_targets"] > 0.5)
            ),
            "phase_accuracy": jnp.mean(jnp.argmax(outputs["phase_logits"], axis=-1) == batch["phase_targets"]),
        }
        return metrics["loss"], metrics

    @jax.jit
    def train_step(current_params, current_opt_state, batch, rng):
        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
            current_params, batch, train_mode=True, rng=rng
        )
        updates, next_opt_state = optimizer.update(gradients, current_opt_state, current_params)
        return (
            optax.apply_updates(current_params, updates),
            next_opt_state,
            {**metrics, "gradient_norm": optax.global_norm(gradients)},
        )

    @jax.jit
    def eval_step(current_params, batch):
        return objective(current_params, batch, train_mode=False)[1]

    def evaluate(current_params) -> dict[str, float]:
        weighted: dict[str, float] = {}
        total = 0
        for start in range(0, len(dev), args.eval_batch_size):
            indices = np.arange(start, min(start + args.eval_batch_size, len(dev)))
            metrics = jax.device_get(eval_step(current_params, _batch(dev, indices)))
            count = len(indices)
            total += count
            for key, value in metrics.items():
                weighted[key] = weighted.get(key, 0.0) + float(value) * count
        return {key: value / total for key, value in weighted.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(int(np.size(value)) for value in jax.tree_util.tree_leaves(params))
    config = vars(args).copy()
    config.update(
        train_episode_indices=train_indices,
        dev_episode_indices=dev_indices,
        train_rows=len(train),
        dev_rows=len(dev),
        train_phase_counts=phase_counts.tolist(),
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
    metrics_path = args.output_dir / "metrics.jsonl"
    numpy_rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_step = 0
    started = time.monotonic()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            indices = _phase_balanced_indices(numpy_rng, train.phase_targets, args.batch_size)
            dropout_rng, step_rng = jax.random.split(dropout_rng)
            params, opt_state, metrics = train_step(params, opt_state, _batch(train, indices), step_rng)
            if step == 1 or step % 20 == 0:
                host = {key: float(value) for key, value in jax.device_get(metrics).items()}
                metrics_file.write(json.dumps({"step": step, "split": "train", **host}) + "\n")
                metrics_file.flush()
                print(
                    f"step={step}/{args.steps} loss={host['loss']:.4f} "
                    f"pos={host['position_mae_cm']:.2f}cm phase={host['phase_accuracy']:.3f} "
                    f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                    flush=True,
                )
            if step % args.eval_every == 0 or step == args.steps:
                host = evaluate(params)
                metrics_file.write(json.dumps({"step": step, "split": "dev", **host}) + "\n")
                metrics_file.flush()
                print(
                    f"DEV step={step} loss={host['loss']:.4f} pos={host['position_mae_cm']:.2f}cm "
                    f"rot={host['rotation_mae_deg']:.2f}deg grip={host['gripper_accuracy']:.3f} "
                    f"phase={host['phase_accuracy']:.3f}",
                    flush=True,
                )
                if host["loss"] < best_loss:
                    best_loss = host["loss"]
                    best_step = step
                    _save(params, args.output_dir / "checkpoints" / "best.msgpack")
            if step % args.save_every == 0 or step == args.steps:
                _save(params, args.output_dir / "checkpoints" / f"step_{step}.msgpack")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"memory_mode": args.memory_mode, "best_step": best_step, "best_dev_loss": best_loss}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
