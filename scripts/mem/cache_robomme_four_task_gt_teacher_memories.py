#!/usr/bin/env python3
"""Cache canonical four-task GT teacher memories for visual-MEM distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = _ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
DEFAULT_TRAINING = _ROOT / "checkpoints/robomme_four_task_unified_gt_teacher_canonical_v2_260826"
DEFAULT_CHECKPOINT = DEFAULT_TRAINING / "best/params"
DEFAULT_OUTPUT = _ROOT / "artifacts/robomme_four_task_gt_teacher_memory_v2_260826"
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    for singular, plural in (
        ("task_id", "task_ids"),
        ("required_count", "required_counts"),
        ("queried_ordinal", "queried_ordinals"),
    ):
        arrays[plural] = arrays.pop(singular)
    return arrays


def _inputs(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, jax.Array]:
    return {
        key: jnp.asarray(arrays[key][indices])
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


def _strict_summary(
    logits: np.ndarray,
    targets: np.ndarray,
    field_mask: np.ndarray,
    task_ids: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    predictions = np.argmax(logits, axis=-1)
    valid = np.any(field_mask, axis=-1)
    exact = np.all((predictions == targets) | ~field_mask, axis=-1) & valid

    def score(indices: np.ndarray) -> dict[str, float | int]:
        subset_valid = valid[indices]
        subset_exact = exact[indices]
        lengths = subset_valid.sum(axis=1)
        final = subset_exact[np.arange(len(indices)), np.maximum(lengths - 1, 0)]
        return {
            "episodes": len(indices),
            "state_exact_accuracy": float(subset_exact.sum() / max(subset_valid.sum(), 1)),
            "sequence_exact_accuracy": float(np.mean(np.all(subset_exact | ~subset_valid, axis=1))),
            "final_state_exact_accuracy": float(np.mean(final)),
        }

    result = {"overall": score(np.arange(len(task_ids)))}
    for task_id, task_name in enumerate(teacher_lib.TASKS):
        result[task_name] = score(np.flatnonzero(task_ids == task_id))
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / f"{split}.npz" for split in SPLITS]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Output exists under {args.output_dir}; pass --overwrite")

    config = json.loads((args.training_dir / "training_config.json").read_text(encoding="utf-8"))
    model = teacher_lib.UnifiedRoboMMEGTTeacher(
        width=int(config["memory_width"]),
        num_memory_tokens=int(config["memory_tokens"]),
        memory_depth=int(config["memory_depth"]),
        memory_heads=int(config["memory_heads"]),
        readout_heads=int(config["memory_heads"]),
    )
    template_data = _load(args.data_dir / "train.npz")
    one = np.asarray([0], dtype=np.int64)
    template_targets = jnp.asarray(template_data["state_targets"][one])
    template_mask = jnp.asarray(template_data["state_field_mask"][one])
    template = model.init(
        jax.random.key(int(config["seed"])),
        **_inputs(template_data, one),
        teacher_state_targets=template_targets,
        teacher_state_field_mask=template_mask,
    )["params"]
    params = flax.serialization.from_bytes(template, args.checkpoint.read_bytes())

    @jax.jit
    def infer(inputs, targets, field_mask):
        outputs = model.apply(
            {"params": params},
            **inputs,
            teacher_state_targets=targets,
            teacher_state_field_mask=field_mask,
        )
        return (
            outputs["gt_state_memories"],
            outputs["all_memories"],
            outputs["gt_state_logits"],
            outputs["state_logits"],
        )

    summaries = {}
    for split in SPLITS:
        arrays = template_data if split == "train" else _load(args.data_dir / f"{split}.npz")
        canonical_memories, rollout_memories = [], []
        canonical_logits, rollout_logits = [], []
        for start in range(0, len(arrays["task_ids"]), args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, len(arrays["task_ids"])))
            result = infer(
                _inputs(arrays, indices),
                jnp.asarray(arrays["state_targets"][indices]),
                jnp.asarray(arrays["state_field_mask"][indices]),
            )
            canonical, rollout, canonical_logit, rollout_logit = jax.device_get(result)
            canonical_memories.append(np.asarray(canonical, dtype=np.float16))
            rollout_memories.append(np.asarray(rollout, dtype=np.float16))
            canonical_logits.append(np.asarray(canonical_logit))
            rollout_logits.append(np.asarray(rollout_logit))
        canonical_memory = np.concatenate(canonical_memories)
        rollout_memory = np.concatenate(rollout_memories)
        canonical_logit = np.concatenate(canonical_logits)
        rollout_logit = np.concatenate(rollout_logits)
        canonical_summary = _strict_summary(
            canonical_logit,
            arrays["state_targets"],
            arrays["state_field_mask"],
            arrays["task_ids"],
        )
        rollout_summary = _strict_summary(
            rollout_logit,
            arrays["state_targets"],
            arrays["state_field_mask"],
            arrays["task_ids"],
        )
        if canonical_summary["overall"]["sequence_exact_accuracy"] != 1.0:
            raise ValueError(f"Canonical teacher is not exact on {split}: {canonical_summary}")
        np.savez_compressed(
            args.output_dir / f"{split}.npz",
            teacher_memory=canonical_memory,
            diagnostic_rollout_memory=rollout_memory,
            task_id=arrays["task_ids"],
            episode_index=arrays["episode_index"],
            source=arrays["source"],
            step_mask=arrays["step_mask"],
            state_targets=arrays["state_targets"],
            state_field_mask=arrays["state_field_mask"],
        )
        summaries[split] = {
            "episodes": len(arrays["task_ids"]),
            "teacher_memory_shape": list(canonical_memory.shape),
            "teacher_memory_dtype": str(canonical_memory.dtype),
            "canonical": canonical_summary,
            "diagnostic_event_rollout": rollout_summary,
        }
        print(json.dumps({split: summaries[split]}, ensure_ascii=False, sort_keys=True), flush=True)
    summary = {
        "schema_version": 1,
        "data_dir": str(args.data_dir.resolve()),
        "training_dir": str(args.training_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "teacher_uses_qwen_predictions": False,
        "student_distillation_target": "teacher_memory (canonical GT-state memory)",
        "diagnostic_only": "diagnostic_rollout_memory",
        "splits": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

