#!/usr/bin/env python3
"""Train PickXTimes event classifiers on centered RGB/proprio modality ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import flax
import flax.linen as nn
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.tasks.robomme.unified_visual_student import VisualWindowEncoder


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    _ROOT / "artifacts/pickxtimes_boundary_centered_multimodal_6pre6post_v1_260827"
)
MODES = (
    "rgb",
    "gripper_state",
    "gripper_command",
    "eef_z",
    "proprio",
    "rgb_proprio",
)
EVENT_NAMES = ("pick_complete", "place_complete", "press_complete")
PROPRIO_LAYOUT = ("gripper_left", "gripper_right", "gripper_command", "eef_z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--end-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260834)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _uses_rgb(mode: str) -> bool:
    return mode in {"rgb", "rgb_proprio"}


def _proprio_indices(mode: str) -> tuple[int, ...]:
    return {
        "rgb": (),
        "gripper_state": (0, 1),
        "gripper_command": (2,),
        "eef_z": (3,),
        "proprio": (0, 1, 2, 3),
        "rgb_proprio": (0, 1, 2, 3),
    }[mode]


class EventDataset:
    def __init__(
        self,
        path: Path,
        *,
        load_rgb: bool,
        proprio_mean: np.ndarray | None = None,
        proprio_std: np.ndarray | None = None,
    ):
        with h5py.File(path, "r") as payload:
            self.event_targets = np.asarray(payload["event_target"], dtype=np.int32)
            self.episode_index = np.asarray(payload["episode_index"], dtype=np.int32)
            self.proprio = np.asarray(payload["proprio"], dtype=np.float32)
            self.patch_tokens = (
                np.asarray(payload["patch_tokens"], dtype=np.float16)
                if load_rgb
                else np.zeros((len(self.event_targets), 1), dtype=np.float16)
            )
        if proprio_mean is None:
            proprio_mean = self.proprio.mean(axis=(0, 1))
        if proprio_std is None:
            proprio_std = self.proprio.std(axis=(0, 1))
        self.proprio_mean = np.asarray(proprio_mean, dtype=np.float32)
        self.proprio_std = np.maximum(np.asarray(proprio_std, dtype=np.float32), 1e-5)
        self.proprio = (self.proprio - self.proprio_mean) / self.proprio_std

    def __len__(self) -> int:
        return len(self.event_targets)

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "patch_tokens": self.patch_tokens[indices],
            "proprio": self.proprio[indices],
            "event_targets": self.event_targets[indices],
        }


class CenteredMultimodalEventClassifier(nn.Module):
    mode: str
    width: int = 64
    encoder_width: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 8

    @nn.compact
    def __call__(
        self,
        patch_tokens: jnp.ndarray,
        proprio: jnp.ndarray,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        batch = proprio.shape[0]
        features = []
        if _uses_rgb(self.mode):
            expected = (batch, 12, 16, 1152)
            if patch_tokens.shape != expected:
                raise ValueError(f"Expected RGB features {expected}, got {patch_tokens.shape}")
            encoded = VisualWindowEncoder(
                name="visual_window_encoder",
                frames=12,
                spatial_tokens=16,
                input_width=1152,
                width=self.width,
                encoder_width=self.encoder_width,
                depth=self.encoder_depth,
                num_heads=self.encoder_heads,
                dtype_mm="bfloat16",
            )(patch_tokens, train=train).reshape(batch, 12, 16, self.width)
            normalized = nn.LayerNorm(name="rgb_evidence_ln", dtype=jnp.float32)(
                encoded.astype(jnp.float32)
            )
            early = jnp.mean(normalized[:, :6], axis=(1, 2))
            late = jnp.mean(normalized[:, 6:], axis=(1, 2))
            whole = jnp.mean(normalized, axis=(1, 2))
            rgb_features = jnp.concatenate(
                (whole, early, late, late - early, jnp.abs(late - early)), axis=-1
            )
            features.append(
                nn.gelu(
                    nn.Dense(self.width * 2, name="rgb_summary", dtype=jnp.float32)(
                        rgb_features
                    )
                )
            )

        indices = _proprio_indices(self.mode)
        if indices:
            x = proprio[..., indices]
            x = nn.Dense(self.width, name="proprio_input", dtype=jnp.float32)(x)
            position = self.param(
                "proprio_temporal_position",
                nn.initializers.normal(stddev=0.02),
                (1, 12, self.width),
                jnp.float32,
            )
            x = x + position
            for layer in range(2):
                normalized = nn.LayerNorm(
                    name=f"proprio_attn_ln_{layer}", dtype=jnp.float32
                )(x)
                x = x + nn.SelfAttention(
                    name=f"proprio_attn_{layer}",
                    num_heads=4,
                    dropout_rate=0.0,
                    deterministic=True,
                    dtype=jnp.float32,
                )(normalized)
                normalized = nn.LayerNorm(
                    name=f"proprio_mlp_ln_{layer}", dtype=jnp.float32
                )(x)
                hidden = nn.gelu(
                    nn.Dense(
                        self.width * 2,
                        name=f"proprio_mlp_in_{layer}",
                        dtype=jnp.float32,
                    )(normalized)
                )
                x = x + nn.Dense(
                    self.width,
                    name=f"proprio_mlp_out_{layer}",
                    dtype=jnp.float32,
                )(hidden)
            x = nn.LayerNorm(name="proprio_output_ln", dtype=jnp.float32)(x)
            proprio_summary = jnp.concatenate(
                (
                    jnp.mean(x, axis=1),
                    jnp.std(x, axis=1),
                    x[:, 0],
                    x[:, 5],
                    x[:, 6],
                    x[:, -1],
                    x[:, -1] - x[:, 0],
                ),
                axis=-1,
            )
            features.append(
                nn.gelu(
                    nn.Dense(
                        self.width * 2,
                        name="proprio_summary",
                        dtype=jnp.float32,
                    )(proprio_summary)
                )
            )

        if not features:
            raise ValueError(f"Mode {self.mode} has no active input")
        fused = features[0] if len(features) == 1 else jnp.concatenate(features, axis=-1)
        fused = nn.gelu(
            nn.Dense(self.width * 2, name="fusion_hidden", dtype=jnp.float32)(fused)
        )
        return nn.Dense(len(EVENT_NAMES), name="event_classifier", dtype=jnp.float32)(fused)


def _inputs(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {
        "patch_tokens": jnp.asarray(batch["patch_tokens"]),
        "proprio": jnp.asarray(batch["proprio"]),
    }


def _metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    predictions = np.argmax(logits, axis=-1)
    confusion = np.zeros((len(EVENT_NAMES), len(EVENT_NAMES)), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    precisions = np.diag(confusion) / np.maximum(confusion.sum(axis=0), 1)
    f1 = 2.0 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
    return {
        "event_accuracy": float(np.mean(targets == predictions)),
        "event_macro_recall": float(np.mean(recalls)),
        "event_macro_f1": float(np.mean(f1)),
        "minimum_pick_place_recall": float(min(recalls[0], recalls[1])),
        "event_per_class_recall": {
            name: float(value) for name, value in zip(EVENT_NAMES, recalls, strict=True)
        },
        "event_confusion": confusion.tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_rgb = _uses_rgb(args.mode)
    train = EventDataset(args.data_dir / "train.h5", load_rgb=load_rgb)
    dev = EventDataset(
        args.data_dir / "dev.h5",
        load_rgb=load_rgb,
        proprio_mean=train.proprio_mean,
        proprio_std=train.proprio_std,
    )
    test = EventDataset(
        args.data_dir / "test.h5",
        load_rgb=load_rgb,
        proprio_mean=train.proprio_mean,
        proprio_std=train.proprio_std,
    )
    rng = np.random.default_rng(args.seed)
    model = CenteredMultimodalEventClassifier(
        mode=args.mode,
        width=args.width,
        encoder_width=args.encoder_width,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
    )
    initial = train.batch(np.arange(min(args.batch_size, len(train))))
    params = model.init(jax.random.key(args.seed), **_inputs(initial), train=False)["params"]
    class_counts = np.bincount(train.event_targets, minlength=len(EVENT_NAMES)).astype(
        np.float32
    )
    class_weights = class_counts.sum() / (class_counts * len(class_counts))
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        args.learning_rate,
        min(args.warmup_steps, max(args.steps - 1, 0)),
        args.steps,
        end_value=args.end_learning_rate,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4)
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        def objective(current_params):
            logits = model.apply({"params": current_params}, **_inputs(batch), train=True)
            targets = jnp.asarray(batch["event_targets"])
            losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            weights = jnp.asarray(class_weights)[targets]
            loss = jnp.sum(losses * weights) / jnp.sum(weights)
            return loss, {
                "loss": loss,
                "accuracy": jnp.mean(jnp.argmax(logits, axis=-1) == targets),
            }

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, metrics

    @jax.jit
    def infer(params, inputs):
        return model.apply({"params": params}, **inputs, train=False)

    def evaluate(params, dataset: EventDataset) -> dict[str, object]:
        full = dataset.batch(np.arange(len(dataset)))
        logits = np.asarray(infer(params, _inputs(full)))
        return _metrics(logits, full["event_targets"])

    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_events": len(train),
        "dev_events": len(dev),
        "test_events": len(test),
        "train_event_counts": {
            name: int(value)
            for name, value in zip(EVENT_NAMES, class_counts, strict=True)
        },
        "proprio_layout": PROPRIO_LAYOUT,
        "active_proprio_fields": tuple(PROPRIO_LAYOUT[index] for index in _proprio_indices(args.mode)),
        "proprio_mean": train.proprio_mean.tolist(),
        "proprio_std": train.proprio_std.tolist(),
        "oracle_boundary_used": True,
        "window_contract": "6_pre_plus_6_post",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    best_score = (-1.0, -1.0, -1.0)
    best_step = 0
    best_params = params
    started = time.monotonic()
    with (args.output_dir / "metrics.jsonl").open("w") as metrics_file:
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                dev_metrics = evaluate(params, dev)
                row = {"split": "dev", "step": step, **dev_metrics}
                metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                metrics_file.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                score = (
                    float(dev_metrics["minimum_pick_place_recall"]),
                    float(dev_metrics["event_macro_recall"]),
                    float(dev_metrics["event_macro_f1"]),
                )
                if score > best_score:
                    best_score = score
                    best_step = step
                    best_params = jax.device_get(params)
            if step == args.steps:
                break
            indices = rng.choice(len(train), args.batch_size, replace=True)
            batch = train.batch(indices)
            params, opt_state, train_metrics = train_step(params, opt_state, batch)
            if step == 0 or (step + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "split": "train",
                            "step": step + 1,
                            **{key: float(value) for key, value in train_metrics.items()},
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    (args.output_dir / "best").mkdir(exist_ok=True)
    (args.output_dir / "best/params").write_bytes(
        flax.serialization.to_bytes(best_params)
    )
    result = {
        "mode": args.mode,
        "best_step": best_step,
        "best_dev_score": best_score,
        "test": evaluate(best_params, test),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
