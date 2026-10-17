#!/usr/bin/env python3
"""Train and causally audit VideoUnmask semantic memory on frozen features."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import pathlib
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from openpi.tasks.robomme.videounmask import semantic_memory
from openpi.tasks.robomme.videounmask import semantic_memory_target_event
from openpi.training.mem import robomme_videounmask_dataset
from openpi.training.mem.recipes import robomme_videounmask_semantic_memory_pretrain as loss_recipe
from openpi.training.mem.recipes import robomme_videounmask_target_event_pretrain as target_event_loss_recipe

INPUT_KEYS = ("demo_patch_tokens", "prompt_tokens", "prompt_mask", "frame_mask")
TARGET_KEYS = ("target_point", "target_cell", "target_color")
ABLATIONS = ("full", "late_only", "zero_video", "wrong_video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260823)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encoder-width", type=int, default=64)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--memory-width", type=int, default=64)
    parser.add_argument("--memory-depth", type=int, default=1)
    parser.add_argument("--memory-tokens", type=int, default=32)
    parser.add_argument(
        "--architecture",
        choices=("recurrent_frames", "target_event"),
        default="recurrent_frames",
    )
    return parser.parse_args()


def jax_batch(batch: Mapping[str, object], keys: tuple[str, ...]) -> dict[str, jax.Array]:
    result = {}
    for key in keys:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        result[key] = jnp.asarray(value)
    return result


def make_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def save_params(params, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flax.serialization.to_bytes(jax.device_get(params)))


def transform_video(inputs: dict[str, jax.Array], ablation: str) -> dict[str, jax.Array]:
    if ablation == "full":
        return inputs
    transformed = dict(inputs)
    video = inputs["demo_patch_tokens"]
    if ablation == "late_only":
        transformed["demo_patch_tokens"] = jnp.repeat(video[:, -1:], video.shape[1], axis=1)
    elif ablation == "zero_video":
        transformed["demo_patch_tokens"] = jnp.zeros_like(video)
    elif ablation == "wrong_video":
        transformed["demo_patch_tokens"] = jnp.roll(video, 1, axis=0)
    else:
        raise ValueError(ablation)
    return transformed


def mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def compute_coordinate_baselines(payload: dict) -> dict:
    by_index = {int(item["episode_index"]): item for item in payload["episodes"]}
    train = [by_index[int(index)] for index in payload["train_episode_indices"]]
    val = [by_index[int(index)] for index in payload["val_episode_indices"]]
    global_mean = np.mean([item["target_point_yx"] for item in train], axis=0)
    color_means = {
        color: np.mean([item["target_point_yx"] for item in train if item["target_color"] == color], axis=0)
        for color in ("red", "green", "blue")
    }

    def score(selector):
        errors = np.asarray(
            [np.linalg.norm(np.asarray(selector(item)) - np.asarray(item["target_point_yx"])) for item in val]
        )
        return {
            "point_distance_px": float(np.mean(errors)),
            "within_10px": float(np.mean(errors <= 10)),
            "within_20px": float(np.mean(errors <= 20)),
            "within_30px": float(np.mean(errors <= 30)),
        }

    return {
        "global_train_mean": score(lambda _: global_mean),
        "prompt_color_train_mean": score(lambda item: color_means[item["target_color"]]),
    }


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.save_every) < 1:
        raise ValueError("steps, batch-size, eval-every, and save-every must be positive")
    payload = json.loads(args.labels.read_text(encoding="utf-8"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_indices = [int(value) for value in payload["train_episode_indices"]]
    val_indices = [int(value) for value in payload["val_episode_indices"]]
    train_dataset = robomme_videounmask_dataset.VideoUnmaskFeatureDataset(
        args.features, args.labels, episode_indices=train_indices
    )
    val_dataset = robomme_videounmask_dataset.VideoUnmaskFeatureDataset(
        args.features, args.labels, episode_indices=val_indices
    )
    train_loader = make_loader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    # Keep all held-out episodes in one batch so wrong-video is a true episode permutation.
    val_loader = make_loader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0)
    train_iterator = iter(train_loader)

    model_kwargs = {
        "encoder_width": args.encoder_width,
        "encoder_depth": args.encoder_depth,
        "memory_width": args.memory_width,
        "memory_depth": args.memory_depth,
        "num_memory_tokens": args.memory_tokens,
    }
    if args.architecture == "target_event":
        model = semantic_memory_target_event.VideoUnmaskTargetEventMemory(**model_kwargs)
        objective_recipe = target_event_loss_recipe
    else:
        model = semantic_memory.VideoUnmaskSemanticMemory(**model_kwargs)
        objective_recipe = loss_recipe
    example_raw = next(train_iterator)
    example_inputs = jax_batch(example_raw, INPUT_KEYS)
    init_rng, train_rng = jax.random.split(jax.random.key(args.seed))
    variables = model.init(init_rng, **example_inputs, train=True)
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

    @jax.jit
    def train_step(params, opt_state, inputs, targets, rng):
        def objective(current_params):
            outputs = model.apply({"params": current_params}, **inputs, train=True, rngs={"dropout": rng})
            return objective_recipe.compute_losses(outputs, targets)

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        metrics = {**metrics, "gradient_norm": optax.global_norm(gradients)}
        return optax.apply_updates(params, updates), next_opt_state, metrics

    @jax.jit
    def eval_step(params, inputs, targets):
        outputs = model.apply({"params": params}, **inputs, train=False)
        _, metrics = objective_recipe.compute_losses(outputs, targets)
        return metrics

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(train_episode_indices=train_indices, val_episode_indices=val_indices)
    args.output_dir.joinpath("config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    args.output_dir.joinpath("baselines.json").write_text(
        json.dumps(compute_coordinate_baselines(payload), indent=2) + "\n", encoding="utf-8"
    )
    log_path = args.output_dir / "metrics.jsonl"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, args.steps + 1):
            try:
                raw = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                raw = next(train_iterator)
            inputs = jax_batch(raw, INPUT_KEYS)
            targets = jax_batch(raw, TARGET_KEYS)
            train_rng, step_rng = jax.random.split(train_rng)
            params, opt_state, metrics = train_step(params, opt_state, inputs, targets, step_rng)
            if step == 1 or step % 10 == 0:
                host = {key: float(value) for key, value in jax.device_get(metrics).items()}
                log_file.write(json.dumps({"step": step, "split": "train", **host}) + "\n")
                log_file.flush()
                print(
                    f"step={step}/{args.steps} loss={host['loss']:.4f} "
                    f"distance={host['point_distance_px']:.1f}px elapsed={(time.monotonic()-started)/60:.1f}m",
                    flush=True,
                )
            if step % args.eval_every == 0 or step == args.steps:
                val_raw = next(iter(val_loader))
                base_inputs = jax_batch(val_raw, INPUT_KEYS)
                val_targets = jax_batch(val_raw, TARGET_KEYS)
                for ablation in ABLATIONS:
                    result = eval_step(params, transform_video(base_inputs, ablation), val_targets)
                    host = {key: float(value) for key, value in jax.device_get(result).items()}
                    log_file.write(
                        json.dumps({"step": step, "split": "val", "ablation": ablation, **host}) + "\n"
                    )
                    print(
                        f"VAL step={step} {ablation} distance={host['point_distance_px']:.1f}px "
                        f"within20={host['within_20px']:.3f} color={host['target_color_accuracy']:.3f}",
                        flush=True,
                    )
                log_file.flush()
            if step % args.save_every == 0 or step == args.steps:
                save_params(params, args.output_dir / "checkpoints" / f"step_{step}.msgpack")


if __name__ == "__main__":
    main()
