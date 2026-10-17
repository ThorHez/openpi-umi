#!/usr/bin/env python3
"""Train a frozen-memory-target to next-step EEF7 adapter for VideoUnmask."""

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

from openpi.tasks.robomme.videounmask import eef_action_adapter
from openpi.training.mem import robomme_videounmask_action_dataset as action_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260823)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--progress-steps", type=int, default=128)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--target-mode",
        choices=("absolute", "delta", "phase_waypoint_delta"),
        default="absolute",
        help="Train on the next absolute EEF pose or the adjacent wrapped EEF delta.",
    )
    parser.add_argument("--phase-conditioned", action="store_true")
    parser.add_argument("--phase-balanced", action="store_true")
    return parser.parse_args()


def _batch(arrays: action_data.ActionArrays, indices: np.ndarray) -> dict[str, jax.Array]:
    return {
        "features": jnp.asarray(arrays.features[indices]),
        "target_crops": jnp.asarray(arrays.target_crops[indices]),
        "poses": jnp.asarray(arrays.poses[indices]),
        "close_targets": jnp.asarray(arrays.close_targets[indices]),
    }


def _save(params, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flax.serialization.to_bytes(jax.device_get(params)))


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.save_every, args.progress_steps) < 1:
        raise ValueError("step, batch, evaluation, save, and progress arguments must be positive")
    payload = json.loads(args.labels.read_text(encoding="utf-8"))
    train_indices = [int(value) for value in payload["train_episode_indices"]]
    val_indices = [int(value) for value in payload["val_episode_indices"]]
    train_raw = action_data.load_single_target_action_arrays(
        args.h5,
        args.labels,
        episode_indices=train_indices,
        progress_steps=args.progress_steps,
        target_mode=args.target_mode,
        phase_conditioned=args.phase_conditioned,
    )
    val_raw = action_data.load_single_target_action_arrays(
        args.h5,
        args.labels,
        episode_indices=val_indices,
        progress_steps=args.progress_steps,
        target_mode=args.target_mode,
        phase_conditioned=args.phase_conditioned,
    )
    stats = action_data.compute_normalization(train_raw)
    train = action_data.normalize_arrays(train_raw, stats)
    val = action_data.normalize_arrays(val_raw, stats)
    print(
        f"Loaded {len(train.features)} train rows from {len(np.unique(train.episode_indices))} episodes; "
        f"{len(val.features)} val rows from {len(np.unique(val.episode_indices))} episodes",
        flush=True,
    )

    model = eef_action_adapter.VideoUnmaskEEFActionAdapter(
        hidden_width=args.hidden_width,
        depth=args.depth,
        feature_dim=int(train.features.shape[-1]),
    )
    init_rng, dropout_rng = jax.random.split(jax.random.key(args.seed))
    variables = model.init(
        {"params": init_rng, "dropout": init_rng},
        jnp.zeros((args.batch_size, train.features.shape[-1]), dtype=jnp.float32),
        jnp.zeros(
            (
                args.batch_size,
                eef_action_adapter.TARGET_CROP_SIZE,
                eef_action_adapter.TARGET_CROP_SIZE,
                3,
            ),
            dtype=jnp.uint8,
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
    close_rate = float(train.close_targets.mean())
    positive_weight = (1.0 - close_rate) / max(close_rate, 1e-6)
    pose_std = jnp.asarray(stats.pose_std)

    def objective(current_params, batch, *, train_mode: bool, rng=None):
        rngs = {"dropout": rng} if rng is not None else None
        outputs = model.apply(
            {"params": current_params},
            batch["features"],
            batch["target_crops"],
            train=train_mode,
            rngs=rngs,
        )
        pose_error = outputs["normalized_pose"] - batch["poses"]
        pose_loss = jnp.mean(jnp.square(pose_error))
        gripper_bce = optax.sigmoid_binary_cross_entropy(outputs["close_logit"], batch["close_targets"])
        gripper_weights = jnp.where(batch["close_targets"] > 0.5, positive_weight, 1.0)
        gripper_loss = jnp.sum(gripper_bce * gripper_weights) / jnp.sum(gripper_weights)
        physical_pose_error = pose_error * pose_std
        metrics = {
            "loss": pose_loss + args.gripper_loss_weight * gripper_loss,
            "pose_loss": pose_loss,
            "gripper_loss": gripper_loss,
            "position_mae_cm": jnp.mean(jnp.abs(physical_pose_error[:, :3])) * 100.0,
            "rotation_mae_deg": jnp.mean(jnp.abs(physical_pose_error[:, 3:])) * (180.0 / jnp.pi),
            "gripper_accuracy": jnp.mean((outputs["close_logit"] > 0.0) == (batch["close_targets"] > 0.5)),
        }
        return metrics["loss"], metrics

    @jax.jit
    def train_step(current_params, current_opt_state, batch, rng):
        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
            current_params,
            batch,
            train_mode=True,
            rng=rng,
        )
        updates, next_opt_state = optimizer.update(gradients, current_opt_state, current_params)
        metrics = {**metrics, "gradient_norm": optax.global_norm(gradients)}
        return optax.apply_updates(current_params, updates), next_opt_state, metrics

    @jax.jit
    def eval_step(current_params, batch):
        return objective(current_params, batch, train_mode=False)[1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        train_episode_indices=sorted(np.unique(train.episode_indices).astype(int).tolist()),
        val_episode_indices=sorted(np.unique(val.episode_indices).astype(int).tolist()),
        train_rows=len(train.features),
        val_rows=len(val.features),
        close_rate=close_rate,
        positive_gripper_weight=positive_weight,
        feature_dim=int(train.features.shape[-1]),
        phase_counts={
            action_data.PHASES[index]: int(np.sum(train.phases == index))
            for index in range(len(action_data.PHASES))
        },
        normalization=stats.to_json(),
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    numpy_rng = np.random.default_rng(args.seed)
    val_indices_all = np.arange(len(val.features))
    started = time.monotonic()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            if args.phase_balanced:
                sampled_phases = numpy_rng.integers(
                    0, len(action_data.PHASES), size=args.batch_size
                )
                indices = np.asarray(
                    [numpy_rng.choice(np.flatnonzero(train.phases == phase)) for phase in sampled_phases]
                )
            else:
                indices = numpy_rng.integers(0, len(train.features), size=args.batch_size)
            dropout_rng, step_rng = jax.random.split(dropout_rng)
            params, opt_state, metrics = train_step(params, opt_state, _batch(train, indices), step_rng)
            if step == 1 or step % 20 == 0:
                host = {key: float(value) for key, value in jax.device_get(metrics).items()}
                metrics_file.write(json.dumps({"step": step, "split": "train", **host}) + "\n")
                metrics_file.flush()
                print(
                    f"step={step}/{args.steps} loss={host['loss']:.4f} "
                    f"pos={host['position_mae_cm']:.2f}cm grip={host['gripper_accuracy']:.3f} "
                    f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                    flush=True,
                )
            if step % args.eval_every == 0 or step == args.steps:
                val_metrics = eval_step(params, _batch(val, val_indices_all))
                host = {key: float(value) for key, value in jax.device_get(val_metrics).items()}
                metrics_file.write(json.dumps({"step": step, "split": "val", **host}) + "\n")
                metrics_file.flush()
                print(
                    f"VAL step={step} loss={host['loss']:.4f} pos={host['position_mae_cm']:.2f}cm "
                    f"rot={host['rotation_mae_deg']:.2f}deg grip={host['gripper_accuracy']:.3f}",
                    flush=True,
                )
            if step % args.save_every == 0 or step == args.steps:
                _save(params, args.output_dir / "checkpoints" / f"step_{step}.msgpack")


if __name__ == "__main__":
    main()
