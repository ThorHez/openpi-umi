#!/usr/bin/env python3
"""Ablate centered RGB, gripper, EEF-Z, and RGB+proprio for Pick events."""

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

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib
from openpi.tasks.robomme.unified_visual_student import VisualWindowEncoder


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = _ROOT / "artifacts/pickxtimes_boundary_centered_multimodal_events_v1_260827"
MODES = ("rgb", "gripper", "eef_z", "rgb_proprio")
EVENT_NAMES = ("pick_complete", "place_complete", "press_complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=48)
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


class CenteredEventDataset:
    def __init__(self, split: str, args: argparse.Namespace, summary: dict):
        self.meta = _load_npz(args.data_dir / f"{split}.npz")
        with h5py.File(args.data_dir / f"{split}.h5", "r") as payload:
            count = len(self.meta["event_targets"])
            # Keep unused high-dimensional RGB off host memory for low-dimensional ablations.
            self.rgb = (
                np.asarray(payload["patch_tokens"])
                if args.mode in {"rgb", "rgb_proprio"}
                else np.zeros((count, 12, 16, 1), dtype=np.float16)
            )
            self.gripper = (
                np.asarray(payload["gripper"], dtype=np.float32)
                if args.mode in {"gripper", "rgb_proprio"}
                else np.zeros((count, 12, 4), dtype=np.float32)
            )
            self.eef_z = (
                np.asarray(payload["eef_z"], dtype=np.float32)
                if args.mode in {"eef_z", "rgb_proprio"}
                else np.zeros((count, 12, 2), dtype=np.float32)
            )
        for key, values in (("gripper", self.gripper), ("eef_z", self.eef_z)):
            if args.mode == "rgb" or (args.mode == "gripper" and key == "eef_z") or (
                args.mode == "eef_z" and key == "gripper"
            ):
                continue
            norm = summary[f"{key}_normalization"]
            mean = np.asarray(norm["mean"], dtype=np.float32)
            std = np.asarray(norm["std"], dtype=np.float32)
            values -= mean
            values /= std

    def __len__(self) -> int:
        return len(self.meta["event_targets"])

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "rgb": self.rgb[indices],
            "gripper": self.gripper[indices],
            "eef_z": self.eef_z[indices],
            "targets": self.meta["event_targets"][indices],
        }


class LowDimTemporalEncoder(nn.Module):
    width: int = 64
    name_prefix: str = "temporal"

    @nn.compact
    def __call__(self, values: jnp.ndarray) -> jnp.ndarray:
        if values.ndim != 3 or values.shape[1] != 12:
            raise ValueError(f"Expected low-dimensional sequence [B,12,D], got {values.shape}")
        projected = nn.Dense(self.width, name=f"{self.name_prefix}_input")(values)
        position = self.param(
            f"{self.name_prefix}_position",
            nn.initializers.normal(stddev=0.02),
            (1, 12, self.width),
            jnp.float32,
        )
        x = projected + position
        residual = nn.gelu(nn.Dense(self.width * 2, name=f"{self.name_prefix}_hidden")(x))
        x = x + nn.Dense(self.width, name=f"{self.name_prefix}_out")(residual)
        early = jnp.mean(x[:, :6], axis=1)
        late = jnp.mean(x[:, 6:], axis=1)
        delta = late - early
        summary = jnp.concatenate(
            (jnp.mean(x, axis=1), early, late, delta, jnp.abs(delta), x[:, 0], x[:, -1]),
            axis=-1,
        )
        return nn.gelu(
            nn.Dense(self.width * 2, name=f"{self.name_prefix}_summary")(summary)
        )


class CenteredModalityEventClassifier(nn.Module):
    mode: str = "rgb"
    width: int = 64
    encoder_width: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 8

    @nn.compact
    def __call__(
        self,
        rgb: jnp.ndarray,
        gripper: jnp.ndarray,
        eef_z: jnp.ndarray,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        features = []
        if self.mode in {"rgb", "rgb_proprio"}:
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
            )(rgb, train=train).reshape(rgb.shape[0], 12, 16, self.width)
            encoded = nn.LayerNorm(name="rgb_output_ln")(encoded.astype(jnp.float32))
            early = jnp.mean(encoded[:, :6], axis=(1, 2))
            late = jnp.mean(encoded[:, 6:], axis=(1, 2))
            delta = late - early
            rgb_summary = jnp.concatenate(
                (jnp.mean(encoded, axis=(1, 2)), early, late, delta, jnp.abs(delta)), axis=-1
            )
            features.append(nn.gelu(nn.Dense(self.width * 2, name="rgb_summary")(rgb_summary)))
        if self.mode in {"gripper", "rgb_proprio"}:
            features.append(
                LowDimTemporalEncoder(width=self.width, name_prefix="gripper")(gripper)
            )
        if self.mode in {"eef_z", "rgb_proprio"}:
            features.append(LowDimTemporalEncoder(width=self.width, name_prefix="eef_z")(eef_z))
        if not features:
            raise ValueError(self.mode)
        fused = features[0] if len(features) == 1 else jnp.concatenate(features, axis=-1)
        hidden = nn.gelu(nn.Dense(self.width * 2, name="classifier_hidden")(fused))
        return nn.Dense(len(EVENT_NAMES), name="classifier_out")(hidden)


def _inputs(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(batch[key]) for key in ("rgb", "gripper", "eef_z")}


def _classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    predictions = np.argmax(logits, axis=-1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    precisions = np.diag(confusion) / np.maximum(confusion.sum(axis=0), 1)
    f1 = 2.0 * recalls * precisions / np.maximum(recalls + precisions, 1e-12)
    return {
        "event_accuracy": float(np.mean(targets == predictions)),
        "event_macro_recall": float(np.mean(recalls)),
        "event_macro_f1": float(np.mean(f1)),
        "pick_place_min_recall": float(min(recalls[0], recalls[1])),
        "event_per_class_recall": {
            name: float(value) for name, value in zip(EVENT_NAMES, recalls, strict=True)
        },
        "event_confusion": confusion.tolist(),
    }


def _rollout(events: np.ndarray, required_count: int) -> np.ndarray:
    completed = 0
    holding = 0
    ready = 0
    done = 0
    states = []
    for event in events:
        if event == 0:  # pick_complete
            holding = 1
        elif event == 1:  # place_complete
            if holding:
                completed = min(required_count, completed + 1)
            holding = 0
            ready = int(completed >= required_count)
        elif event == 2:  # press_complete
            done = int(bool(ready))
            if done:
                ready = 0
        states.append((completed, holding, ready, done))
    return np.asarray(states, dtype=np.int32)


def _rollout_metrics(predictions: np.ndarray, dataset: CenteredEventDataset) -> dict[str, float]:
    offsets = dataset.meta["episode_offsets"]
    required = dataset.meta["required_counts"]
    targets = dataset.meta["event_targets"]
    state_correct = 0
    state_total = 0
    sequence_correct = []
    final_correct = []
    for episode in range(len(required)):
        start, end = int(offsets[episode]), int(offsets[episode + 1])
        predicted_states = _rollout(predictions[start:end], int(required[episode]))
        target_states = _rollout(targets[start:end], int(required[episode]))
        exact = np.all(predicted_states == target_states, axis=-1)
        state_correct += int(exact.sum())
        state_total += len(exact)
        sequence_correct.append(bool(np.all(exact)))
        final_correct.append(bool(exact[-1]))
    return {
        "symbolic_state_exact_accuracy": state_correct / state_total,
        "symbolic_sequence_exact_accuracy": float(np.mean(sequence_correct)),
        "symbolic_final_exact_accuracy": float(np.mean(final_correct)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.data_dir / "summary.json").read_text())
    datasets = {
        split: CenteredEventDataset(split, args, summary) for split in ("train", "dev", "test")
    }
    model = CenteredModalityEventClassifier(
        mode=args.mode,
        width=args.width,
        encoder_width=args.encoder_width,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
    )
    initial = datasets["train"].batch(np.arange(min(args.batch_size, len(datasets["train"]))))
    params = model.init(jax.random.key(args.seed), **_inputs(initial), train=False)["params"]
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
    rng = np.random.default_rng(args.seed)
    class_rows = [
        np.flatnonzero(datasets["train"].meta["event_targets"] == class_id)
        for class_id in range(3)
    ]

    @jax.jit
    def train_step(params, opt_state, batch):
        def objective(current_params):
            logits = model.apply({"params": current_params}, **_inputs(batch), train=True)
            targets = jnp.asarray(batch["targets"])
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == targets)
            return loss, {"loss": loss, "accuracy": accuracy}

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, metrics

    @jax.jit
    def infer(params, inputs):
        return model.apply({"params": params}, **inputs, train=False)

    def evaluate(params, split: str) -> dict[str, object]:
        dataset = datasets[split]
        batch = dataset.batch(np.arange(len(dataset)))
        logits = np.asarray(infer(params, _inputs(batch)))
        predictions = np.argmax(logits, axis=-1)
        return {
            **_classification_metrics(logits, batch["targets"]),
            **_rollout_metrics(predictions, dataset),
        }

    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_events": len(datasets["train"]),
        "dev_events": len(datasets["dev"]),
        "test_events": len(datasets["test"]),
        "window_relative_indices": summary["window"]["relative_indices"],
        "oracle_boundary_used": True,
        "teacher_latent_used": False,
        "selection_metric": "min(Pick recall, Place recall), then macro recall",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    best_score = (-1.0, -1.0, -1.0)
    best_step = 0
    best_params = params
    started = time.monotonic()
    with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                dev = evaluate(params, "dev")
                row = {"split": "dev", "step": step, **dev}
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                score = (
                    float(dev["pick_place_min_recall"]),
                    float(dev["event_macro_recall"]),
                    float(dev["event_accuracy"]),
                )
                if score > best_score:
                    best_score = score
                    best_step = step
                    best_params = jax.device_get(params)
            if step == args.steps:
                break
            per_class = args.batch_size // 3
            remainder = args.batch_size - 3 * per_class
            parts = [
                rng.choice(rows, per_class + int(class_id < remainder), replace=True)
                for class_id, rows in enumerate(class_rows)
            ]
            indices = np.concatenate(parts)
            rng.shuffle(indices)
            batch = datasets["train"].batch(indices)
            params, opt_state, metrics = train_step(params, opt_state, batch)
            if step == 0 or (step + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "split": "train",
                            "step": step + 1,
                            **{key: float(value) for key, value in metrics.items()},
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    (args.output_dir / "best").mkdir(exist_ok=True)
    (args.output_dir / "best/params").write_bytes(flax.serialization.to_bytes(best_params))
    result = {
        "mode": args.mode,
        "best_step": best_step,
        "best_dev_score": best_score,
        "test": evaluate(best_params, "test"),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
