#!/usr/bin/env python3
"""Train the PickXtimes PRESS residual on dense post-PLACE timelines."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import flax
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--init-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--localization-weight", type=float, default=0.5)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=260903)
    return parser.parse_args()


def load_split(features: pathlib.Path):
    with h5py.File(features, "r") as handle:
        train_indices = [int(value) for value in json.loads(handle.attrs["train_episode_indices"])]
        dev_indices = [int(value) for value in json.loads(handle.attrs["dev_episode_indices"])]
        selected = train_indices + dev_indices
        max_length = max(len(handle[str(index)]["starts"]) for index in selected)
        feature_width = int(handle[str(selected[0])]["semantic_features"].shape[-1])
        semantic = np.zeros((len(selected), max_length, feature_width), dtype=np.float32)
        base_logits = np.zeros((len(selected), max_length), dtype=np.float32)
        valid_mask = np.zeros((len(selected), max_length), dtype=np.bool_)
        positive_mask = np.zeros((len(selected), max_length), dtype=np.bool_)
        for row, episode_index in enumerate(selected):
            group = handle[str(episode_index)]
            length = len(group["starts"])
            semantic[row, :length] = np.asarray(group["semantic_features"], dtype=np.float32)
            base_logits[row, :length] = np.asarray(group["base_event_logits"], dtype=np.float32)
            valid_mask[row, :length] = True
            positive_mask[row, :length] = np.asarray(group["positive_mask"], dtype=np.bool_)
    train_rows = np.arange(len(train_indices), dtype=np.int32)
    dev_rows = np.arange(len(train_indices), len(selected), dtype=np.int32)
    return semantic, base_logits, valid_mask, positive_mask, train_rows, dev_rows, train_indices, dev_indices


def dense_losses(logits, valid_mask, positive_mask, *, ranking_margin, temperature):
    negative_mask = valid_mask & ~positive_mask
    element_losses = optax.sigmoid_binary_cross_entropy(logits, positive_mask.astype(jnp.float32))

    def masked_mean(values, mask):
        mask = mask.astype(jnp.float32)
        return jnp.sum(values * mask, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1), 1.0)

    bce = 0.5 * (masked_mean(element_losses, positive_mask) + masked_mean(element_losses, negative_mask))
    masked_positive = jnp.where(positive_mask, logits / temperature, -1e9)
    masked_negative = jnp.where(negative_mask, logits / temperature, -1e9)
    masked_valid = jnp.where(valid_mask, logits / temperature, -1e9)
    positive_peak = temperature * jax.nn.logsumexp(masked_positive, axis=-1)
    negative_peak = temperature * jax.nn.logsumexp(masked_negative, axis=-1)
    ranking = jax.nn.softplus(ranking_margin + negative_peak - positive_peak)
    localization = -(jax.nn.logsumexp(masked_positive, axis=-1) - jax.nn.logsumexp(masked_valid, axis=-1))
    predicted_position = jnp.argmax(jnp.where(valid_mask, logits, -1e9), axis=-1)
    batch_axis = jnp.arange(logits.shape[0])
    return {
        "bce_loss": jnp.mean(bce),
        "ranking_loss": jnp.mean(ranking),
        "localization_loss": jnp.mean(localization),
        "global_peak_in_positive_accuracy": jnp.mean(positive_mask[batch_axis, predicted_position]),
        "positive_logit_mean": jnp.sum(logits * positive_mask) / jnp.maximum(jnp.sum(positive_mask), 1),
        "negative_logit_mean": jnp.sum(logits * negative_mask) / jnp.maximum(jnp.sum(negative_mask), 1),
    }


def replace_press_params(full_params, press_params):
    mutable = flax.core.unfreeze(full_params)
    mutable["press_gate_fusion"]["press_residual"]["kernel"] = press_params["kernel"]
    mutable["press_gate_fusion"]["press_residual"]["bias"] = press_params["bias"]
    return flax.core.freeze(mutable) if isinstance(full_params, flax.core.FrozenDict) else mutable


def main() -> None:
    args = parse_args()
    if args.steps < 2 or args.batch_size < 1:
        raise ValueError("--steps must be at least 2 and --batch-size must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    semantic, base_logits, valid_mask, positive_mask, train_rows, dev_rows, train_indices, dev_indices = load_split(
        args.features
    )
    semantic = jnp.asarray(semantic)
    base_logits = jnp.asarray(base_logits)
    valid_mask = jnp.asarray(valid_mask)
    positive_mask = jnp.asarray(positive_mask)

    full_params = flax.serialization.msgpack_restore(args.init_checkpoint.read_bytes())
    press_source = full_params["press_gate_fusion"]["press_residual"]
    press_params = {
        "kernel": jnp.asarray(press_source["kernel"], dtype=jnp.float32),
        "bias": jnp.asarray(press_source["bias"], dtype=jnp.float32),
    }
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=min(args.warmup_steps, args.steps - 1),
        decay_steps=args.steps,
        end_value=args.learning_rate * 0.1,
    )
    optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
    opt_state = optimizer.init(press_params)

    def apply_press(current_params, rows):
        residual = jnp.einsum("btd,do->bto", semantic[rows], current_params["kernel"])[..., 0]
        return base_logits[rows] + residual + current_params["bias"][0]

    def objective(current_params, rows):
        logits = apply_press(current_params, rows)
        metrics = dense_losses(
            logits,
            valid_mask[rows],
            positive_mask[rows],
            ranking_margin=args.ranking_margin,
            temperature=args.temperature,
        )
        loss = (
            metrics["bce_loss"]
            + args.ranking_weight * metrics["ranking_loss"]
            + args.localization_weight * metrics["localization_loss"]
        )
        return loss, {**metrics, "loss": loss}

    @jax.jit
    def train_step(current_params, current_opt_state, rows):
        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(current_params, rows)
        updates, next_opt_state = optimizer.update(gradients, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, {**metrics, "gradient_norm": optax.global_norm(gradients)}

    @jax.jit
    def eval_step(current_params, rows):
        _, metrics = objective(current_params, rows)
        return metrics

    def save_checkpoint(current_params, step):
        checkpoint = replace_press_params(full_params, jax.device_get(current_params))
        path = args.output_dir / "checkpoints" / f"step_{step}.msgpack"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(flax.serialization.to_bytes(checkpoint))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(train_episode_indices=train_indices, dev_episode_indices=dev_indices)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log_file:
        for step in range(1, args.steps + 1):
            rows = rng.choice(train_rows, size=args.batch_size, replace=len(train_rows) < args.batch_size)
            press_params, opt_state, metrics = train_step(press_params, opt_state, jnp.asarray(rows))
            if step == 1 or step % 10 == 0:
                host = {key: float(value) for key, value in jax.device_get(metrics).items()}
                log_file.write(json.dumps({"step": step, "split": "train", **host}) + "\n")
                log_file.flush()
                print(
                    f"step={step}/{args.steps} loss={host['loss']:.4f} "
                    f"peak_acc={host['global_peak_in_positive_accuracy']:.3f} "
                    f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                    flush=True,
                )
            if step % args.eval_every == 0 or step == args.steps:
                metrics = eval_step(press_params, jnp.asarray(dev_rows))
                host = {key: float(value) for key, value in jax.device_get(metrics).items()}
                log_file.write(json.dumps({"step": step, "split": "dev", **host}) + "\n")
                log_file.flush()
                print(
                    f"DEV step={step} loss={host['loss']:.4f} peak_acc={host['global_peak_in_positive_accuracy']:.3f}",
                    flush=True,
                )
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(press_params, step)


if __name__ == "__main__":
    main()
