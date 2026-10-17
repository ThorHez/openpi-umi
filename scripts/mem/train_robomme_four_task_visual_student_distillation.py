#!/usr/bin/env python3
"""Distill the unified canonical GT teacher into one visual recurrent student."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import time
from typing import Any

import flax
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib
from openpi.tasks.robomme import unified_visual_student as student_lib

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE = _ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
DEFAULT_FEATURES = _ROOT / "artifacts/robomme_four_task_visual_features_4x4_v1_260826"
DEFAULT_MEMORY = _ROOT / "artifacts/robomme_four_task_gt_teacher_memory_v2_260826"
DEFAULT_TEACHER_TRAINING = _ROOT / "checkpoints/robomme_four_task_unified_gt_teacher_canonical_v2_260826"
DEFAULT_TEACHER_CHECKPOINT = DEFAULT_TEACHER_TRAINING / "best/params"
DEFAULT_OUTPUT = _ROOT / "checkpoints/robomme_four_task_visual_student_distilled_v1_260826"
SPLITS = ("train", "dev", "test")

GOAL_KEYS = (
    "task_ids",
    "goal_color_ids",
    "required_counts",
    "queried_ordinals",
    "num_regions",
    "step_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--teacher-memory-dir", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--teacher-training-dir", type=Path, default=DEFAULT_TEACHER_TRAINING)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--end-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--memory-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.5)
    parser.add_argument("--semantic-token-weight", type=float, default=4.0)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    for singular, plural in (
        ("task_id", "task_ids"),
        ("required_count", "required_counts"),
        ("queried_ordinal", "queried_ordinals"),
    ):
        if singular in arrays:
            arrays[plural] = arrays.pop(singular)
    return arrays


class SplitDataset:
    """Join frozen visual features, goal fields, canonical memory, and states."""

    def __init__(self, split: str, args: argparse.Namespace):
        self.split = split
        self.goals = _load_npz(args.sequence_dir / f"{split}.npz")
        self.teacher = _load_npz(args.teacher_memory_dir / f"{split}.npz")
        self.features = h5py.File(args.feature_dir / f"{split}.h5", "r")
        self.length = len(self.goals["task_ids"])
        if self.length != len(self.teacher["teacher_memory"]):
            raise ValueError(f"Teacher/goal length mismatch on {split}")
        for index in range(self.length):
            name = f"episode_{index:06d}"
            if name not in self.features or not bool(self.features[name].attrs.get("complete", False)):
                raise ValueError(f"Missing visual feature group {self.features.filename}:{name}")
            if int(self.features[name].attrs["episode_index"]) != int(self.teacher["episode_index"][index]):
                raise ValueError(f"Episode alignment mismatch at {split}:{index}")

    def close(self) -> None:
        self.features.close()

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        batch = len(indices)
        patches = np.zeros((batch, 12, 12, 16, 1152), dtype=np.float16)
        for batch_index, episode_index in enumerate(indices):
            event_patches = self.features[f"episode_{int(episode_index):06d}/patch_tokens"][:]
            patches[batch_index, : len(event_patches)] = event_patches
        result = {
            "patch_tokens": patches,
            "teacher_memory": self.teacher["teacher_memory"][indices],
            "state_targets": self.teacher["state_targets"][indices],
            "state_field_mask": self.teacher["state_field_mask"][indices],
        }
        result.update({key: self.goals[key][indices] for key in GOAL_KEYS})
        return result


def _balanced_indices(arrays: Mapping[str, np.ndarray], rng: np.random.Generator, batch_size: int):
    if batch_size % len(teacher_lib.TASKS):
        raise ValueError(f"--batch-size must be divisible by {len(teacher_lib.TASKS)}")
    sampled = []
    per_task = batch_size // len(teacher_lib.TASKS)
    for task_id in range(len(teacher_lib.TASKS)):
        candidates = np.flatnonzero(arrays["task_ids"] == task_id)
        sampled.extend(rng.choice(candidates, per_task, replace=len(candidates) < per_task))
    rng.shuffle(sampled)
    return np.asarray(sampled, dtype=np.int64)


def _student_inputs(batch: Mapping[str, np.ndarray | jax.Array]) -> dict[str, jax.Array]:
    return {
        "patch_tokens": jnp.asarray(batch["patch_tokens"]),
        **{key: jnp.asarray(batch[key]) for key in GOAL_KEYS},
    }


def _load_teacher_readout(args: argparse.Namespace) -> tuple[teacher_lib.UnifiedStateReadout, Any]:
    config = json.loads((args.teacher_training_dir / "training_config.json").read_text())
    sequence = _load_npz(args.sequence_dir / "train.npz")
    teacher = teacher_lib.UnifiedRoboMMEGTTeacher(
        width=int(config["memory_width"]),
        num_memory_tokens=int(config["memory_tokens"]),
        memory_depth=int(config["memory_depth"]),
        memory_heads=int(config["memory_heads"]),
        readout_heads=int(config["memory_heads"]),
    )
    one = np.asarray([0])
    teacher_inputs = {
        key: jnp.asarray(sequence[key][one])
        for key in (
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
    }
    template = teacher.init(
        jax.random.key(int(config["seed"])),
        **teacher_inputs,
        teacher_state_targets=jnp.asarray(sequence["state_targets"][one]),
        teacher_state_field_mask=jnp.asarray(sequence["state_field_mask"][one]),
    )["params"]
    params = flax.serialization.from_bytes(template, args.teacher_checkpoint.read_bytes())
    readout = teacher_lib.UnifiedStateReadout(
        width=int(config["memory_width"]),
        num_heads=int(config["memory_heads"]),
    )
    return readout, params["unified_state_readout"]


def _host_summary(
    logits: np.ndarray,
    targets: np.ndarray,
    field_mask: np.ndarray,
    task_ids: np.ndarray,
) -> dict[str, Any]:
    predictions = np.argmax(logits, axis=-1)
    valid = np.any(field_mask, axis=-1)
    exact = np.all((predictions == targets) | ~field_mask, axis=-1) & valid

    def summarize(indices: np.ndarray) -> dict[str, float | int]:
        subset_valid = valid[indices]
        subset_exact = exact[indices]
        lengths = subset_valid.sum(axis=1)
        final = subset_exact[np.arange(len(indices)), np.maximum(lengths - 1, 0)]
        fields = field_mask[indices]
        return {
            "episodes": len(indices),
            "field_accuracy": float(
                (((predictions[indices] == targets[indices]) & fields).sum()) / max(fields.sum(), 1)
            ),
            "state_exact_accuracy": float(subset_exact.sum() / max(subset_valid.sum(), 1)),
            "sequence_exact_accuracy": float(np.mean(np.all(subset_exact | ~subset_valid, axis=1))),
            "final_state_exact_accuracy": float(np.mean(final)),
        }

    output = {"overall": summarize(np.arange(len(task_ids)))}
    for task_id, task_name in enumerate(teacher_lib.TASKS):
        output[task_name] = summarize(np.flatnonzero(task_ids == task_id))
    return output


def _save_params(params, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "params").write_bytes(flax.serialization.to_bytes(jax.device_get(params)))


def main() -> None:
    args = parse_args()
    if args.self_test_only:
        model = student_lib.UnifiedVisualRecurrentStudent(
            max_steps=2,
            frames=2,
            spatial_tokens=4,
            input_width=16,
            width=16,
            num_memory_tokens=24,
            encoder_width=16,
            encoder_depth=1,
            encoder_heads=4,
            memory_depth=1,
            memory_heads=4,
            dtype_mm="float32",
        )
        inputs = {
            "patch_tokens": jnp.zeros((1, 2, 2, 4, 16)),
            "task_ids": jnp.zeros((1,), dtype=jnp.int32),
            "goal_color_ids": jnp.zeros((1, 2), dtype=jnp.int32),
            "required_counts": jnp.zeros((1,), dtype=jnp.int32),
            "queried_ordinals": jnp.zeros((1,), dtype=jnp.int32),
            "num_regions": jnp.zeros((1,), dtype=jnp.int32),
            "step_mask": jnp.ones((1, 2), dtype=jnp.bool_),
        }
        output = model.apply(model.init(jax.random.key(0), **inputs), **inputs)
        print(f"self-test passed: all_memories={output['all_memories'].shape}")
        return
    if args.batch_size % len(teacher_lib.TASKS):
        raise ValueError("--batch-size must be divisible by four")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {split: SplitDataset(split, args) for split in SPLITS}
    rng = np.random.default_rng(args.seed)
    model = student_lib.UnifiedVisualRecurrentStudent(
        encoder_width=args.encoder_width,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
    )
    init_batch = datasets["train"].batch(_balanced_indices(datasets["train"].goals, rng, args.batch_size))
    params = model.init(jax.random.key(args.seed), **_student_inputs(init_batch), train=False)["params"]
    teacher_readout, teacher_readout_params = _load_teacher_readout(args)
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        args.learning_rate,
        min(args.warmup_steps, max(args.steps - 1, 0)),
        args.steps,
        end_value=args.end_learning_rate,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(params)

    def objective(current_params, batch, *, train: bool):
        output = model.apply({"params": current_params}, **_student_inputs(batch), train=train)
        student_memory = output["all_memories"]
        teacher_memory = jnp.asarray(batch["teacher_memory"])
        targets = jnp.asarray(batch["state_targets"])
        field_mask = jnp.asarray(batch["state_field_mask"])
        valid = jnp.any(field_mask, axis=-1)
        memory_loss, memory_metrics = student_lib.memory_distillation_loss(
            student_memory,
            teacher_memory,
            valid,
            semantic_token_weight=args.semantic_token_weight,
        )
        flat = student_memory.reshape(-1, student_memory.shape[-2], student_memory.shape[-1])
        logits = teacher_readout.apply({"params": teacher_readout_params}, flat).reshape(
            *student_memory.shape[:2], len(teacher_lib.STATE_FIELDS), teacher_lib.MAX_FIELD_CLASSES
        )
        state_loss, state_metrics = teacher_lib.compute_teacher_losses(
            {"state_logits": logits, "all_memories": student_memory}, targets, field_mask
        )
        loss = args.memory_loss_weight * memory_loss + args.state_loss_weight * state_loss
        metrics = {
            "loss": loss,
            **memory_metrics,
            "state_loss": state_loss,
            "field_accuracy": state_metrics["field_accuracy"],
            "state_exact_accuracy": state_metrics["state_exact_accuracy"],
            "sequence_exact_accuracy": state_metrics["sequence_exact_accuracy"],
            "final_state_exact_accuracy": state_metrics["final_state_exact_accuracy"],
        }
        return loss, (metrics, logits)

    @jax.jit
    def train_step(params, opt_state, batch):
        (_, (metrics, _)), gradients = jax.value_and_grad(objective, has_aux=True)(
            params, batch, train=True
        )
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        metrics = {**metrics, "gradient_norm": optax.global_norm(gradients)}
        return optax.apply_updates(params, updates), next_opt_state, metrics

    @jax.jit
    def eval_step(params, batch):
        _, (metrics, logits) = objective(params, batch, train=False)
        return metrics, logits

    def evaluate(split: str, params) -> tuple[dict[str, Any], dict[str, float]]:
        dataset = datasets[split]
        logits, metric_rows = [], []
        for start in range(0, dataset.length, args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, dataset.length))
            # Keep a static shape for JIT by repeating the final episode, then trim outputs.
            real_count = len(indices)
            if real_count < args.batch_size:
                indices = np.pad(indices, (0, args.batch_size - real_count), mode="edge")
            batch = dataset.batch(indices)
            metrics, batch_logits = eval_step(params, batch)
            logits.append(np.asarray(batch_logits)[:real_count])
            metric_rows.append({key: float(value) for key, value in metrics.items()})
        logits = np.concatenate(logits)
        summary = _host_summary(
            logits,
            dataset.teacher["state_targets"],
            dataset.teacher["state_field_mask"],
            dataset.goals["task_ids"],
        )
        averaged = {key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]}
        return summary, averaged

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "student_inputs": "goal fields + causal 12-frame frozen visual tokens + previous student memory",
            "forbidden_student_inputs": ["GT event", "GT state", "Qwen output", "teacher memory at inference"],
            "distillation_target": "canonical teacher_memory",
            "teacher_readout_frozen": True,
        }
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    best_params = params
    best_score = (-1.0, -1.0, -1.0)
    best_step = 0
    started = time.perf_counter()
    try:
        with metrics_path.open("w", encoding="utf-8") as log:
            for step in range(1, args.steps + 1):
                indices = _balanced_indices(datasets["train"].goals, rng, args.batch_size)
                batch = datasets["train"].batch(indices)
                params, opt_state, metrics = train_step(params, opt_state, batch)
                if step == 1 or step % 10 == 0:
                    record = {
                        "step": step,
                        "split": "train",
                        "elapsed_seconds": time.perf_counter() - started,
                        "learning_rate": float(schedule(step)),
                        **{key: float(value) for key, value in metrics.items()},
                    }
                    log.write(json.dumps(record, sort_keys=True) + "\n")
                    log.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    dev_summary, dev_metrics = evaluate("dev", params)
                    overall = dev_summary["overall"]
                    score = (
                        overall["final_state_exact_accuracy"],
                        overall["sequence_exact_accuracy"],
                        overall["state_exact_accuracy"],
                    )
                    record = {
                        "step": step,
                        "split": "dev",
                        "elapsed_seconds": time.perf_counter() - started,
                        **dev_metrics,
                        "strict": dev_summary,
                    }
                    log.write(json.dumps(record, sort_keys=True) + "\n")
                    log.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)
                    if score > best_score:
                        best_score, best_step, best_params = score, step, jax.device_get(params)
                        _save_params(best_params, args.output_dir / "best")
                if step % args.save_every == 0 or step == args.steps:
                    _save_params(params, args.output_dir / str(step))
        test_summary, test_metrics = evaluate("test", best_params)
        result = {
            "best_step": best_step,
            "best_dev_score": best_score,
            "test": test_summary,
            "test_losses": test_metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
