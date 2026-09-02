#!/usr/bin/env python3
"""Probe whether final-cup semantics survive the action memory conditioner.

The probe never executes the action expert.  It feeds zero action tokens into
the trained conditioner, extracts the resulting memory-driven residual, and
fits a held-out linear ridge classifier.  Comparing this representation with
raw memory separates resampler information loss from downstream action-expert
usage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.training.mem.recipes import shellgame_qwen_event_memory_action as action_recipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260826)
    return parser.parse_args()


def _ridge_accuracy(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    split = int(0.8 * len(order))
    train, test = order[:split], order[split:]
    mean = features[train].mean(axis=0, keepdims=True)
    std = features[train].std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    normalized = (features - mean) / std
    train_x = np.concatenate(
        (normalized[train], np.ones((len(train), 1), dtype=np.float32)), axis=1
    ).astype(np.float64)
    test_x = np.concatenate(
        (normalized[test], np.ones((len(test), 1), dtype=np.float32)), axis=1
    ).astype(np.float64)
    targets = np.eye(3, dtype=np.float64)[labels[train]]
    regularizer = np.eye(train_x.shape[1], dtype=np.float64) * 1e-2
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(train_x.T @ train_x + regularizer, train_x.T @ targets)
    prediction = np.argmax(test_x @ weights, axis=1)
    return float(np.mean(prediction == labels[test]))


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    memory_path = args.memory.expanduser().resolve()
    with np.load(memory_path, allow_pickle=False) as source:
        memory = np.asarray(source["final_memory"], dtype=np.float32)
        labels = np.asarray(source["final_label"], dtype=np.int32)
        correct = np.asarray(source["final_prediction"], dtype=np.int32) == labels
    memory = memory[correct]
    labels = labels[correct]

    config = action_recipe.make_model_config()
    model = config.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    model.eval()

    @jax.jit
    def conditioned(batch):
        action_tokens = jnp.zeros(
            (batch.shape[0], config.action_horizon, 1024), dtype=jnp.bfloat16
        )
        output = model.SemanticMemoryActionConditioner(action_tokens, batch)
        return output.astype(jnp.float32)

    residuals = []
    for start in range(0, len(memory), args.batch_size):
        batch = jnp.asarray(memory[start : start + args.batch_size])
        residuals.append(np.asarray(conditioned(batch)))
    residual = np.concatenate(residuals, axis=0)

    raw_mean = memory.mean(axis=1)
    residual_mean = residual.mean(axis=1)
    residual_first = residual[:, 0]
    print(f"examples={len(labels)} residual_shape={residual.shape}")
    print(f"raw_mean_ridge_accuracy={_ridge_accuracy(raw_mean, labels, args.seed):.6f}")
    print(
        "conditioner_mean_ridge_accuracy="
        f"{_ridge_accuracy(residual_mean, labels, args.seed):.6f}"
    )
    print(
        "conditioner_first_ridge_accuracy="
        f"{_ridge_accuracy(residual_first, labels, args.seed):.6f}"
    )
    print(f"conditioner_feature_std={float(np.mean(np.std(residual_mean, axis=0))):.8f}")


if __name__ == "__main__":
    main()
