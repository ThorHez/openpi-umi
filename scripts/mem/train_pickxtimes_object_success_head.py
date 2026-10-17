#!/usr/bin/env python3
"""Distill simulator pick/place success into a deployable visual predicate head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import flax
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.tasks.robomme.pickxtimes.explicit_event_count_memory import (  # noqa: E402
    PickObjectSuccessHead,
)


DEFAULT_CACHE = ROOT / "artifacts/pickxtimes_privileged_success_rollouts_train20_seed7_260831"
DEFAULT_INIT = ROOT / "checkpoints/pickxtimes_event_front_wrist_seed260833_260831/params.msgpack"
DEFAULT_PROPRIO = ROOT / "artifacts/pickxtimes_fixed_chunk_proprio_v1_260828/summary.json"
DEFAULT_OUTPUT = ROOT / "checkpoints/pickxtimes_object_success_privileged_distill_seed260835_260831"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--additional-cache-dir",
        type=Path,
        action="append",
        default=[],
        help="On-policy rollout_event_caches directory; added to training only.",
    )
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--proprio-summary", type=Path, default=DEFAULT_PROPRIO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260835)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def restore_matching(params, path: Path):
    source = flax.serialization.msgpack_restore(path.read_bytes())
    target_flat = traverse_util.flatten_dict(params)
    source_flat = traverse_util.flatten_dict(source)
    loaded = []
    for key, value in source_flat.items():
        if key in target_flat and np.shape(value) == np.shape(target_flat[key]):
            target_flat[key] = jnp.asarray(value, dtype=target_flat[key].dtype)
            loaded.append("/".join(key))
    return traverse_util.unflatten_dict(target_flat), loaded


def load_rows(
    cache_dir: Path,
    proprio_mean: np.ndarray,
    proprio_std: np.ndarray,
    *,
    episode_offset: int = 0,
):
    episodes = []
    paths = sorted(cache_dir.glob("episode_*_oracle.npz"))
    onpolicy = False
    if not paths:
        paths = sorted(cache_dir.glob("PickXtimes_ep*.npz"))
        onpolicy = True
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            rgb_key = "rgb_tokens" if onpolicy else "patch_tokens"
            required = {
                rgb_key,
                "wrist_tokens",
                "proprio",
                "privileged_pick_success",
                "privileged_place_success",
            }
            missing = required - set(payload.files)
            if missing:
                raise ValueError(f"{path} is missing privileged fields: {sorted(missing)}")
            count = len(payload[rgb_key])
            if onpolicy:
                match = re.search(r"ep(\d+)", path.stem)
                episode_index = int(match.group(1))
                color_id = int(payload["target_color_id"])
            else:
                episode_index = int(payload["episode_index"])
                color_id = int(payload["goal_color_id"])
            episodes.append(
                {
                    "episode": episode_offset + episode_index,
                    "success_flag": str(payload["success_flag"]),
                    "target_color_id": np.full(
                        count, color_id, dtype=np.int32
                    ),
                    "rgb": np.asarray(payload[rgb_key], dtype=np.float16),
                    "wrist": np.asarray(payload["wrist_tokens"], dtype=np.float16),
                    "proprio": (
                        np.asarray(payload["proprio"], dtype=np.float32)
                        - proprio_mean[None, None]
                    )
                    / proprio_std[None, None],
                    # A success is an event observed anywhere in the causal
                    # window, rather than merely a final-frame gripper cue.
                    "labels": np.stack(
                        (
                            np.asarray(payload["privileged_pick_success"]).any(axis=1),
                            np.asarray(payload["privileged_place_success"]).any(axis=1),
                        ),
                        axis=-1,
                    ).astype(np.float32),
                }
            )
    if not episodes:
        raise ValueError(f"No privileged cached episodes found in {cache_dir}")
    return episodes


def split_episodes(episodes: list[dict]):
    ordered = sorted(episodes, key=lambda row: row["episode"])
    count = len(ordered)
    train_stop = max(1, int(round(0.70 * count)))
    dev_stop = max(train_stop + 1, int(round(0.85 * count)))
    dev_stop = min(dev_stop, count - 1)
    return {
        "train": ordered[:train_stop],
        "dev": ordered[train_stop:dev_stop],
        "test": ordered[dev_stop:],
    }


def concatenate(rows: list[dict]):
    keys = ("target_color_id", "rgb", "wrist", "proprio", "labels")
    result = {key: np.concatenate([row[key] for row in rows], axis=0) for key in keys}
    result["episode_ids"] = [row["episode"] for row in rows]
    result["episode_success"] = [row["success_flag"] for row in rows]
    return result


def model_inputs(batch):
    return {
        "target_color_id": jnp.asarray(batch["target_color_id"]),
        "rgb": jnp.asarray(batch["rgb"]),
        "wrist": jnp.asarray(batch["wrist"]),
        "proprio": jnp.asarray(batch["proprio"]),
    }


def metric_summary(logits: np.ndarray, labels: np.ndarray):
    predicted = logits >= 0.0
    target = labels.astype(bool)
    result = {}
    balanced = []
    for index, name in enumerate(("pick_success", "place_success")):
        tp = int(np.sum(predicted[:, index] & target[:, index]))
        tn = int(np.sum(~predicted[:, index] & ~target[:, index]))
        fp = int(np.sum(predicted[:, index] & ~target[:, index]))
        fn = int(np.sum(~predicted[:, index] & target[:, index]))
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        score = 0.5 * (recall + specificity)
        balanced.append(score)
        result[name] = {
            "positive_windows": int(np.sum(target[:, index])),
            "negative_windows": int(np.sum(~target[:, index])),
            "accuracy": float((tp + tn) / len(target)),
            "balanced_accuracy": float(score),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
    result["min_balanced_accuracy"] = float(min(balanced))
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalization = json.loads(args.proprio_summary.read_text())["normalization"]
    proprio_mean = np.asarray(normalization["mean"], dtype=np.float32)
    proprio_std = np.asarray(normalization["std"], dtype=np.float32)
    split_rows = split_episodes(load_rows(args.cache_dir, proprio_mean, proprio_std))
    for index, cache_dir in enumerate(args.additional_cache_dir):
        split_rows["train"].extend(
            load_rows(
                cache_dir,
                proprio_mean,
                proprio_std,
                episode_offset=1000 * (index + 1),
            )
        )
    splits = {name: concatenate(rows) for name, rows in split_rows.items()}
    print(
        json.dumps(
            {
                name: {
                    "episodes": data["episode_ids"],
                    "windows": len(data["labels"]),
                    "positive_windows": data["labels"].sum(axis=0).astype(int).tolist(),
                }
                for name, data in splits.items()
            }
        ),
        flush=True,
    )

    model = PickObjectSuccessHead()
    dummy = {key: value[:1] for key, value in splits["train"].items() if key in {
        "target_color_id", "rgb", "wrist", "proprio", "labels"
    }}
    params = model.init(jax.random.key(args.seed), **model_inputs(dummy), train=False)["params"]
    params, loaded = restore_matching(params, args.init_checkpoint)
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        args.learning_rate,
        min(100, args.steps - 1),
        args.steps,
        end_value=1e-5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_opt, batch):
        def objective(p):
            logits = model.apply({"params": p}, **model_inputs(batch), train=True)
            labels = jnp.asarray(batch["labels"])
            loss_matrix = optax.sigmoid_binary_cross_entropy(logits, labels)
            loss = loss_matrix.mean()
            return loss, {
                "loss": loss,
                "pick_accuracy": jnp.mean((logits[:, 0] >= 0) == (labels[:, 0] > 0.5)),
                "place_accuracy": jnp.mean((logits[:, 1] >= 0) == (labels[:, 1] > 0.5)),
            }
        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(current_params)
        updates, next_opt = optimizer.update(gradients, current_opt, current_params)
        return optax.apply_updates(current_params, updates), next_opt, metrics

    @jax.jit
    def infer(current_params, batch):
        return model.apply({"params": current_params}, **model_inputs(batch), train=False)

    def evaluate(current_params, split: str):
        data = splits[split]
        outputs = []
        for start in range(0, len(data["labels"]), args.batch_size):
            stop = min(start + args.batch_size, len(data["labels"]))
            batch = {key: data[key][start:stop] for key in (
                "target_color_id", "rgb", "wrist", "proprio", "labels"
            )}
            outputs.append(np.asarray(infer(current_params, batch)))
        return metric_summary(np.concatenate(outputs), data["labels"])

    rng = np.random.default_rng(args.seed)
    train = splits["train"]
    positive = [np.flatnonzero(train["labels"][:, index] > 0.5) for index in range(2)]
    negative = [np.flatnonzero(train["labels"][:, index] <= 0.5) for index in range(2)]
    best_params = params
    best_step = 0
    best_score = -1.0
    history = []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        per_group = max(args.batch_size // 4, 1)
        indices = np.concatenate(
            [
                rng.choice(group, per_group, replace=True)
                for groups in zip(positive, negative, strict=True)
                for group in groups
            ]
        )
        if len(indices) < args.batch_size:
            indices = np.concatenate(
                (indices, rng.choice(len(train["labels"]), args.batch_size - len(indices)))
            )
        rng.shuffle(indices)
        batch = {key: train[key][indices] for key in (
            "target_color_id", "rgb", "wrist", "proprio", "labels"
        )}
        params, opt_state, metrics = train_step(params, opt_state, batch)
        if step % args.eval_every == 0 or step == args.steps:
            dev = evaluate(params, "dev")
            score = dev["min_balanced_accuracy"]
            if score > best_score:
                best_score = score
                best_step = step
                best_params = jax.device_get(params)
            row = {
                "step": step,
                "train_batch": {key: float(value) for key, value in metrics.items()},
                "dev": dev,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    result = {
        "schema_version": 1,
        "experiment": "pickxtimes_privileged_object_success_distillation",
        "privileged_training_labels": [
            "target object lifted AND simulator is_grasping",
            "target object in target XY region AND released AND EEF lifted",
        ],
        "deployment_inputs": ["front RGB", "wrist RGB", "gripper/command", "EEF Z", "target color"],
        "best_step": best_step,
        "best_score": best_score,
        "loaded_initialization_leaves": len(loaded),
        "split_episode_ids": {name: data["episode_ids"] for name, data in splits.items()},
        "metrics": {name: evaluate(best_params, name) for name in splits},
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }
    (args.output_dir / "params.msgpack").write_bytes(
        flax.serialization.to_bytes(best_params)
    )
    def jsonable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    config = {key: jsonable(value) for key, value in vars(args).items()}
    config.update(
        {
            "model": type(model).__name__,
            "proprio_mean": proprio_mean.tolist(),
            "proprio_std": proprio_std.tolist(),
            "threshold": 0.5,
        }
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
