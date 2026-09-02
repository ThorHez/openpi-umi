#!/usr/bin/env python3
"""Test whether PickXTimes privileged event information can train a recurrent state model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = _ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
MODES = ("rich", "event_type", "binary_boundary", "goal_only")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--end-learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260832)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--memory-tokens", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_pick(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    pick_id = teacher_lib.TASKS.index("pickxtimes_local_event")
    keep = arrays["task_id"] == pick_id
    arrays = {key: value[keep] for key, value in arrays.items()}
    for singular, plural in (
        ("task_id", "task_ids"),
        ("required_count", "required_counts"),
        ("queried_ordinal", "queried_ordinals"),
    ):
        arrays[plural] = arrays.pop(singular)
    return arrays


def _ablate(arrays: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray]:
    arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
    valid = arrays["step_mask"]
    if mode in {"event_type", "binary_boundary", "goal_only"}:
        arrays["entity_ids"].fill(0)
        arrays["region_a_ids"].fill(0)
        arrays["region_b_ids"].fill(0)
    if mode == "binary_boundary":
        generic_event = teacher_lib.EVENTS.index("pick_complete")
        arrays["event_ids"] = np.where(valid, generic_event, 0).astype(np.int32)
    elif mode == "goal_only":
        arrays["event_ids"].fill(0)
    elif mode not in {"rich", "event_type"}:
        raise ValueError(mode)
    return arrays


def _inputs(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, jax.Array]:
    return {key: jnp.asarray(arrays[key][indices]) for key in INPUT_KEYS}


def _targets(
    arrays: dict[str, np.ndarray], indices: np.ndarray
) -> tuple[jax.Array, jax.Array]:
    return (
        jnp.asarray(arrays["state_targets"][indices]),
        jnp.asarray(arrays["state_field_mask"][indices]),
    )


def _summary(logits: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    targets = arrays["state_targets"]
    mask = arrays["state_field_mask"]
    predictions = np.argmax(logits, axis=-1)
    valid = np.any(mask, axis=-1)
    exact = np.all((predictions == targets) | ~mask, axis=-1) & valid
    lengths = valid.sum(axis=1)
    final = exact[np.arange(len(exact)), lengths - 1]
    post_event = np.concatenate(
        (np.zeros((len(exact), 1), dtype=np.bool_), arrays["step_mask"]), axis=1
    ) & valid
    field_correct = (predictions == targets) & mask
    result: dict[str, float | int] = {
        "episodes": len(exact),
        "field_accuracy": float(field_correct.sum() / mask.sum()),
        "state_exact_accuracy": float(exact.sum() / valid.sum()),
        "post_event_state_exact_accuracy": float(
            (exact & post_event).sum() / post_event.sum()
        ),
        "sequence_exact_accuracy": float(np.mean(np.all(exact | ~valid, axis=1))),
        "final_state_exact_accuracy": float(np.mean(final)),
    }
    for name in (
        "required_count",
        "completed_count",
        "holding",
        "ready_to_press",
        "done",
    ):
        field = teacher_lib.STATE_FIELDS.index(name)
        field_mask = mask[..., field]
        result[f"field/{name}_accuracy"] = float(
            field_correct[..., field].sum() / field_mask.sum()
        )
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = _ablate(_load_pick(args.data_dir / "train.npz"), args.mode)
    dev = _ablate(_load_pick(args.data_dir / "dev.npz"), args.mode)
    test = _ablate(_load_pick(args.data_dir / "test.npz"), args.mode)
    rng = np.random.default_rng(args.seed)
    model = teacher_lib.UnifiedRoboMMEGTTeacher(
        width=args.width,
        num_memory_tokens=args.memory_tokens,
        memory_depth=args.depth,
        memory_heads=args.heads,
        readout_heads=args.heads,
    )
    initial_indices = rng.choice(len(train["task_ids"]), args.batch_size, replace=True)
    variables = model.init(jax.random.key(args.seed), **_inputs(train, initial_indices))
    params = variables["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        args.learning_rate,
        min(args.warmup_steps, args.steps),
        args.steps,
        end_value=args.end_learning_rate,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, inputs, targets, field_mask):
        def objective(current_params):
            outputs = model.apply({"params": current_params}, **inputs)
            return teacher_lib.compute_teacher_losses(outputs, targets, field_mask)

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, metrics

    @jax.jit
    def infer(params, inputs):
        return model.apply({"params": params}, **inputs)["state_logits"]

    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(
        {
            "train_episodes": len(train["task_ids"]),
            "dev_episodes": len(dev["task_ids"]),
            "test_episodes": len(test["task_ids"]),
            "pixels_used": False,
            "qwen_used": False,
            "teacher_state_encoder_used": False,
            "recurrent_rollout_only": True,
        }
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    dev_indices = np.arange(len(dev["task_ids"]))
    test_indices = np.arange(len(test["task_ids"]))
    best_score = (-1.0, -1.0, -1.0)
    best_params = params
    best_step = 0
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.monotonic()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                dev_logits = np.asarray(infer(params, _inputs(dev, dev_indices)))
                summary = _summary(dev_logits, dev)
                row = {"split": "dev", "step": step, **summary}
                metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                metrics_file.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                score = (
                    float(summary["state_exact_accuracy"]),
                    float(summary["final_state_exact_accuracy"]),
                    float(summary["sequence_exact_accuracy"]),
                )
                if score > best_score:
                    best_score = score
                    best_params = jax.device_get(params)
                    best_step = step
            if step == args.steps:
                break
            indices = rng.choice(len(train["task_ids"]), args.batch_size, replace=True)
            targets, field_mask = _targets(train, indices)
            params, opt_state, train_metrics = train_step(
                params, opt_state, _inputs(train, indices), targets, field_mask
            )
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
    (args.output_dir / "best/params").write_bytes(flax.serialization.to_bytes(best_params))
    test_logits = np.asarray(infer(best_params, _inputs(test, test_indices)))
    result = {
        "mode": args.mode,
        "best_step": best_step,
        "best_dev_score": best_score,
        "test": _summary(test_logits, test),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

