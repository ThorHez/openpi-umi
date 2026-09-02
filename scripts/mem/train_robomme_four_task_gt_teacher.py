#!/usr/bin/env python3
"""Train and evaluate the unified four-task RoboMME teacher from GT events."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import time
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = _ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
DEFAULT_OUTPUT = _ROOT / "checkpoints/robomme_four_task_unified_gt_teacher_v1_260826"

INPUT_KEYS = (
    "task_ids",
    "goal_color_ids",
    "required_counts",
    "queried_ordinals",
    "num_regions",
    "event_ids",
    "entity_ids",
    "region_a_ids",
    "region_b_ids",
    "step_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--end-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--memory-width", type=int, default=64)
    parser.add_argument("--memory-tokens", type=int, default=128)
    parser.add_argument("--memory-depth", type=int, default=2)
    parser.add_argument("--memory-heads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    # Builder names are singular on disk for readability; model inputs use
    # batch-plural names.
    for singular, plural in (
        ("task_id", "task_ids"),
        ("required_count", "required_counts"),
        ("queried_ordinal", "queried_ordinals"),
    ):
        arrays[plural] = arrays.pop(singular)
    return arrays


def _jax_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, jax.Array]:
    return {key: jnp.asarray(arrays[key][indices]) for key in INPUT_KEYS}


def _targets(arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> tuple[jax.Array, jax.Array]:
    return (
        jnp.asarray(arrays["state_targets"][indices]),
        jnp.asarray(arrays["state_field_mask"][indices]),
    )


def _balanced_indices(
    arrays: Mapping[str, np.ndarray], rng: np.random.Generator, batch_size: int
) -> np.ndarray:
    if batch_size % len(teacher_lib.TASKS):
        raise ValueError(f"--batch-size must be divisible by {len(teacher_lib.TASKS)}")
    per_task = batch_size // len(teacher_lib.TASKS)
    sampled = []
    for task_id in range(len(teacher_lib.TASKS)):
        candidates = np.flatnonzero(arrays["task_ids"] == task_id)
        if not len(candidates):
            raise ValueError(f"Training split has no samples for task {task_id}")
        sampled.extend(rng.choice(candidates, per_task, replace=len(candidates) < per_task))
    rng.shuffle(sampled)
    return np.asarray(sampled, dtype=np.int64)


def _host_summary(
    logits: np.ndarray,
    targets: np.ndarray,
    field_mask: np.ndarray,
    task_ids: np.ndarray,
) -> dict[str, Any]:
    predictions = np.argmax(logits, axis=-1)
    state_valid = np.any(field_mask, axis=-1)
    state_exact = np.all((predictions == targets) | ~field_mask, axis=-1) & state_valid
    fields_correct = (predictions == targets) & field_mask

    def summarize(indices: np.ndarray) -> dict[str, Any]:
        valid = state_valid[indices]
        exact = state_exact[indices]
        mask = field_mask[indices]
        correct = fields_correct[indices]
        lengths = valid.sum(axis=1)
        final = exact[np.arange(len(indices)), np.maximum(lengths - 1, 0)]
        return {
            "episodes": len(indices),
            "field_accuracy": float(correct.sum() / max(mask.sum(), 1)),
            "state_exact_accuracy": float(exact.sum() / max(valid.sum(), 1)),
            "sequence_exact_accuracy": float(np.mean(np.all(exact | ~valid, axis=1))),
            "final_state_exact_accuracy": float(np.mean(final)),
        }

    result = {"overall": summarize(np.arange(len(task_ids)))}
    for task_id, task_name in enumerate(teacher_lib.TASKS):
        result[task_name] = summarize(np.flatnonzero(task_ids == task_id))
    result["field_accuracy"] = {}
    for field_index, field_name in enumerate(teacher_lib.STATE_FIELDS):
        mask = field_mask[..., field_index]
        result["field_accuracy"][field_name] = float(
            fields_correct[..., field_index].sum() / max(mask.sum(), 1)
        )
    return result


def _save_params(params, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "params").write_bytes(flax.serialization.to_bytes(jax.device_get(params)))


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.save_every) < 1:
        raise ValueError("steps, batch-size, eval-every, and save-every must be positive")
    if args.memory_width % args.memory_heads:
        raise ValueError("memory-width must be divisible by memory-heads")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = _load(args.data_dir / "train.npz")
    dev = _load(args.data_dir / "dev.npz")
    test = _load(args.data_dir / "test.npz")
    rng = np.random.default_rng(args.seed)
    model = teacher_lib.UnifiedRoboMMEGTTeacher(
        width=args.memory_width,
        num_memory_tokens=args.memory_tokens,
        memory_depth=args.memory_depth,
        memory_heads=args.memory_heads,
        readout_heads=args.memory_heads,
    )
    initial_indices = _balanced_indices(train, rng, args.batch_size)
    init_inputs = _jax_batch(train, initial_indices)
    init_targets, init_mask = _targets(train, initial_indices)
    variables = model.init(
        jax.random.key(args.seed),
        **init_inputs,
        teacher_state_targets=init_targets,
        teacher_state_field_mask=init_mask,
    )
    params = variables["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=min(args.warmup_steps, args.steps),
        decay_steps=args.steps,
        end_value=args.end_learning_rate,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, inputs, state_targets, state_field_mask):
        def objective(current_params):
            outputs = model.apply(
                {"params": current_params},
                **inputs,
                teacher_state_targets=state_targets,
                teacher_state_field_mask=state_field_mask,
            )
            return teacher_lib.compute_teacher_losses(outputs, state_targets, state_field_mask)

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        metrics = {
            **metrics,
            "gradient_norm": optax.global_norm(gradients),
        }
        return optax.apply_updates(params, updates), next_opt_state, metrics

    @jax.jit
    def evaluate(params, inputs, state_targets, state_field_mask):
        outputs = model.apply(
            {"params": params},
            **inputs,
            teacher_state_targets=state_targets,
            teacher_state_field_mask=state_field_mask,
        )
        _, metrics = teacher_lib.compute_teacher_losses(outputs, state_targets, state_field_mask)
        return outputs["state_logits"], outputs["gt_state_logits"], metrics

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "teacher_uses_qwen_predictions": False,
            "train_episodes": len(train["task_ids"]),
            "dev_episodes": len(dev["task_ids"]),
            "test_episodes": len(test["task_ids"]),
            "state_fields": list(teacher_lib.STATE_FIELDS),
        }
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    all_dev = np.arange(len(dev["task_ids"]), dtype=np.int64)
    all_test = np.arange(len(test["task_ids"]), dtype=np.int64)
    dev_inputs = _jax_batch(dev, all_dev)
    dev_targets, dev_mask = _targets(dev, all_dev)
    best_params = params
    best_dev_final = -1.0
    best_step = 0
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            indices = _balanced_indices(train, rng, args.batch_size)
            inputs = _jax_batch(train, indices)
            state_targets, state_field_mask = _targets(train, indices)
            params, opt_state, metrics = train_step(
                params,
                opt_state,
                inputs,
                state_targets,
                state_field_mask,
            )
            if step == 1 or step % 10 == 0:
                record = {
                    "step": step,
                    "split": "train",
                    "elapsed_seconds": time.perf_counter() - started,
                    "learning_rate": float(schedule(step)),
                    **{
                        key: float(value)
                        for key, value in jax.device_get(metrics).items()
                        if not key.startswith("field/")
                    },
                }
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                if step == 1 or step % 50 == 0:
                    print(json.dumps(record, sort_keys=True), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                dev_rollout_logits, dev_canonical_logits, dev_metrics = evaluate(
                    params, dev_inputs, dev_targets, dev_mask
                )
                dev_rollout_summary = _host_summary(
                    np.asarray(jax.device_get(dev_rollout_logits)),
                    dev["state_targets"],
                    dev["state_field_mask"],
                    dev["task_ids"],
                )
                dev_canonical_summary = _host_summary(
                    np.asarray(jax.device_get(dev_canonical_logits)),
                    dev["state_targets"],
                    dev["state_field_mask"],
                    dev["task_ids"],
                )
                record = {
                    "step": step,
                    "split": "dev",
                    **{key: float(value) for key, value in jax.device_get(dev_metrics).items()},
                    "rollout_strict": dev_rollout_summary,
                    "canonical_strict": dev_canonical_summary,
                }
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                print(
                    json.dumps(
                        {
                            "step": step,
                            "dev_loss": record["loss"],
                            **{
                                f"rollout_{key}": value
                                for key, value in dev_rollout_summary["overall"].items()
                                if key != "episodes"
                            },
                            "canonical_final_state_exact_accuracy": dev_canonical_summary[
                                "overall"
                            ]["final_state_exact_accuracy"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                score = min(
                    dev_rollout_summary["overall"]["final_state_exact_accuracy"],
                    dev_canonical_summary["overall"]["final_state_exact_accuracy"],
                )
                if score > best_dev_final:
                    best_dev_final = score
                    best_step = step
                    best_params = jax.tree.map(lambda value: value.copy(), params)
                    _save_params(best_params, args.output_dir / "best")
                    (args.output_dir / "best/dev_summary.json").write_text(
                        json.dumps(
                            {
                                "rollout": dev_rollout_summary,
                                "canonical": dev_canonical_summary,
                            },
                            indent=2,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            if step % args.save_every == 0:
                _save_params(params, args.output_dir / f"checkpoint-{step:06d}")

    test_inputs = _jax_batch(test, all_test)
    test_targets, test_mask = _targets(test, all_test)
    test_rollout_logits, test_canonical_logits, test_metrics = evaluate(
        best_params, test_inputs, test_targets, test_mask
    )
    test_rollout_summary = _host_summary(
        np.asarray(jax.device_get(test_rollout_logits)),
        test["state_targets"],
        test["state_field_mask"],
        test["task_ids"],
    )
    test_canonical_summary = _host_summary(
        np.asarray(jax.device_get(test_canonical_logits)),
        test["state_targets"],
        test["state_field_mask"],
        test["task_ids"],
    )
    _save_params(params, args.output_dir / "final")
    result = {
        "status": "complete",
        "best_step": best_step,
        "best_dev_final_state_exact_accuracy": best_dev_final,
        "teacher_uses_qwen_predictions": False,
        "test_metrics": {key: float(value) for key, value in jax.device_get(test_metrics).items()},
        "test_rollout_strict": test_rollout_summary,
        "test_canonical_strict": test_canonical_summary,
        "wall_time_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
