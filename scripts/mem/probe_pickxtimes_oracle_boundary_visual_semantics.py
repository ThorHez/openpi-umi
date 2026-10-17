#!/usr/bin/env python3
"""Probe visual event semantics at privileged PickXTimes event boundaries.

This deliberately removes teacher-latent distillation.  Native simulator
boundaries select event-containing 12-frame chunks; a visual-only head predicts
pick/place/press, and a recurrent memory receives only that predicted semantic
token plus the goal.  A newly initialized readout directly predicts the Pick
symbolic state after every event.
"""

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

from openpi.models import siglip_mem_semantic as memory_core
from openpi.tasks.robomme import unified_gt_teacher as teacher_lib
from openpi.tasks.robomme.unified_visual_student import VisualWindowEncoder


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE = (
    _ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_pick_native_v1_260827"
)
DEFAULT_FEATURES = _ROOT / "artifacts/robomme_four_task_fixed_chunk_features_4x4_v1_260826"
DEFAULT_TEACHER = _ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
PICK_TASK = teacher_lib.TASKS.index("pickxtimes_local_event")
PICK_EVENT_IDS = np.asarray(
    [
        teacher_lib.EVENTS.index("pick_complete"),
        teacher_lib.EVENTS.index("place_complete"),
        teacher_lib.EVENTS.index("press_complete"),
    ],
    dtype=np.int32,
)
PICK_EVENT_NAMES = tuple(teacher_lib.EVENTS[index] for index in PICK_EVENT_IDS)
MAX_EVENTS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-mode", choices=("visual", "zero"), default="visual")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--end-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--event-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--memory-tokens", type=int, default=32)
    parser.add_argument("--memory-depth", type=int, default=1)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260833)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


class PickOracleEventDataset:
    """Event-only episodes selected by native simulator boundaries."""

    def __init__(self, split: str, args: argparse.Namespace):
        fixed = _load_npz(args.sequence_dir / f"{split}.npz")
        teacher = _load_npz(args.teacher_dir / f"{split}.npz")
        rows = np.flatnonzero(fixed["task_ids"] == PICK_TASK)
        if not np.array_equal(fixed["episode_index"][rows], teacher["episode_index"][rows]):
            raise ValueError(f"Fixed/teacher episode mismatch on {split}")
        count = len(rows)
        patches = np.zeros(
            (count, MAX_EVENTS, 12, 16, 1152), dtype=np.float16
        )
        event_targets = np.zeros((count, MAX_EVENTS), dtype=np.int32)
        event_mask = np.zeros((count, MAX_EVENTS), dtype=np.bool_)
        with h5py.File(args.feature_dir / f"{split}.h5", "r") as features:
            for output_row, source_row in enumerate(rows):
                change_chunks = np.flatnonzero(fixed["state_change_mask"][source_row])
                num_events = int(teacher["step_mask"][source_row].sum())
                if len(change_chunks) != num_events:
                    raise ValueError(
                        f"{split}:{source_row} has {len(change_chunks)} native chunks but "
                        f"{num_events} semantic events"
                    )
                state_indices = fixed["teacher_state_index"][source_row]
                if not np.array_equal(
                    state_indices[change_chunks + 1], np.arange(1, num_events + 1)
                ):
                    raise ValueError(f"Non-unit event transition on {split}:{source_row}")
                tokens = features[f"episode_{source_row:06d}/patch_tokens"]
                patches[output_row, :num_events] = tokens[change_chunks]
                global_events = teacher["event_ids"][source_row, :num_events]
                local_events = np.searchsorted(PICK_EVENT_IDS, global_events)
                if np.any(PICK_EVENT_IDS[local_events] != global_events):
                    raise ValueError(f"Unexpected Pick event on {split}:{source_row}")
                event_targets[output_row, :num_events] = local_events
                event_mask[output_row, :num_events] = True
        self.arrays = {
            "patch_tokens": patches,
            "event_targets": event_targets,
            "event_mask": event_mask,
            "task_ids": fixed["task_ids"][rows].astype(np.int32),
            "goal_color_ids": fixed["goal_color_ids"][rows].astype(np.int32),
            "required_counts": fixed["required_counts"][rows].astype(np.int32),
            "state_targets": teacher["state_targets"][rows].astype(np.int32),
            "state_field_mask": teacher["state_field_mask"][rows].astype(np.bool_),
            "episode_index": fixed["episode_index"][rows].astype(np.int32),
        }
        # Ensure padded states cannot influence either optimization or metrics.
        state_valid = np.concatenate(
            (np.ones((count, 1), dtype=np.bool_), event_mask), axis=1
        )
        self.arrays["state_field_mask"] &= state_valid[..., None]

    def __len__(self) -> int:
        return len(self.arrays["task_ids"])

    def batch(self, indices: np.ndarray, *, input_mode: str) -> dict[str, np.ndarray]:
        result = {key: value[indices] for key, value in self.arrays.items()}
        if input_mode == "zero":
            result["patch_tokens"] = np.zeros_like(result["patch_tokens"])
        return result


class OracleBoundaryVisualSemanticMemory(nn.Module):
    """Visual event classifier feeding a semantic-bottleneck recurrent MEM."""

    width: int = 64
    num_memory_tokens: int = 32
    memory_depth: int = 1
    encoder_width: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 8

    @nn.compact
    def __call__(
        self,
        patch_tokens: jnp.ndarray,
        task_ids: jnp.ndarray,
        goal_color_ids: jnp.ndarray,
        required_counts: jnp.ndarray,
        event_mask: jnp.ndarray,
        *,
        train: bool = False,
    ) -> dict[str, jnp.ndarray]:
        batch = patch_tokens.shape[0]
        expected = (batch, MAX_EVENTS, 12, 16, 1152)
        if patch_tokens.shape != expected:
            raise ValueError(f"Expected {expected}, got {patch_tokens.shape}")
        flat = patch_tokens.reshape(batch * MAX_EVENTS, 12, 16, 1152)
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
        )(flat, train=train).reshape(batch, MAX_EVENTS, 12, 16, self.width)

        normalized = nn.LayerNorm(name="event_evidence_ln", dtype=jnp.float32)(
            encoded.astype(jnp.float32)
        )
        early = jnp.mean(normalized[:, :, :6], axis=(2, 3))
        late = jnp.mean(normalized[:, :, 6:], axis=(2, 3))
        whole = jnp.mean(normalized, axis=(2, 3))
        visual_features = jnp.concatenate(
            (whole, early, late, late - early, jnp.abs(late - early)), axis=-1
        )
        event_hidden = nn.gelu(
            nn.Dense(self.width * 2, name="event_hidden", dtype=jnp.float32)(visual_features)
        )
        event_logits = nn.Dense(
            len(PICK_EVENT_NAMES), name="event_classifier", dtype=jnp.float32
        )(event_hidden)
        event_probabilities = jax.nn.softmax(event_logits, axis=-1)
        semantic_table = self.param(
            "event_semantic_embedding",
            nn.initializers.normal(stddev=0.02),
            (len(PICK_EVENT_NAMES), self.width),
            jnp.float32,
        )
        # This is the only per-event evidence received by recurrent memory.
        semantic_steps = jnp.einsum("bsc,cw->bsw", event_probabilities, semantic_table)
        semantic_steps = semantic_steps[:, :, None, :]

        task_embed = nn.Embed(len(teacher_lib.TASKS), self.width, name="task_embedding")
        color_embed = nn.Embed(len(teacher_lib.COLORS), self.width, name="color_embedding")
        count_embed = nn.Embed(6, self.width, name="count_embedding")
        goal_tokens = jnp.stack(
            (
                task_embed(task_ids),
                color_embed(goal_color_ids[:, 0]),
                count_embed(required_counts),
            ),
            axis=1,
        )
        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.width),
            jnp.float32,
        )
        base_memory = jnp.tile(base_memory, (batch, 1, 1))
        initial_memory, _ = memory_core.RecurrentMemoryUpdater(
            name="goal_memory_initializer",
            width=self.width,
            depth=self.memory_depth,
            num_heads=4,
            dtype_mm="float32",
        )(base_memory, goal_tokens[:, None])
        _, event_memories = memory_core.RecurrentMemoryUpdater(
            name="semantic_recurrent_updater",
            width=self.width,
            depth=self.memory_depth,
            num_heads=4,
            dtype_mm="float32",
        )(initial_memory, semantic_steps, step_mask=event_mask)
        all_memories = jnp.concatenate((initial_memory[:, None], event_memories), axis=1)
        flat_memory = all_memories.reshape(
            batch * (MAX_EVENTS + 1), self.num_memory_tokens, self.width
        )
        state_logits = teacher_lib.UnifiedStateReadout(
            name="direct_state_readout", width=self.width, num_heads=4
        )(flat_memory).reshape(
            batch,
            MAX_EVENTS + 1,
            len(teacher_lib.STATE_FIELDS),
            teacher_lib.MAX_FIELD_CLASSES,
        )
        return {
            "event_logits": event_logits,
            "event_probabilities": event_probabilities,
            "all_memories": all_memories,
            "state_logits": state_logits,
        }


def _inputs(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {
        key: jnp.asarray(batch[key])
        for key in (
            "patch_tokens",
            "task_ids",
            "goal_color_ids",
            "required_counts",
            "event_mask",
        )
    }


def _event_metrics(
    logits: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> dict[str, object]:
    predictions = np.argmax(logits, axis=-1)
    labels = targets[mask]
    guesses = predictions[mask]
    confusion = np.zeros((len(PICK_EVENT_NAMES), len(PICK_EVENT_NAMES)), dtype=np.int64)
    np.add.at(confusion, (labels, guesses), 1)
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    precisions = np.diag(confusion) / np.maximum(confusion.sum(axis=0), 1)
    f1 = 2.0 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
    return {
        "event_accuracy": float(np.mean(labels == guesses)),
        "event_macro_recall": float(np.mean(recalls)),
        "event_macro_f1": float(np.mean(f1)),
        "event_per_class_recall": {
            name: float(value) for name, value in zip(PICK_EVENT_NAMES, recalls, strict=True)
        },
        "event_confusion": confusion.tolist(),
    }


def _state_metrics(
    logits: np.ndarray, targets: np.ndarray, field_mask: np.ndarray
) -> dict[str, float]:
    predictions = np.argmax(logits, axis=-1)
    valid = np.any(field_mask, axis=-1)
    exact = np.all((predictions == targets) | ~field_mask, axis=-1) & valid
    lengths = valid.sum(axis=1)
    final = exact[np.arange(len(exact)), lengths - 1]
    dynamic_fields = tuple(
        teacher_lib.STATE_FIELDS.index(name)
        for name in ("required_count", "completed_count", "holding", "ready_to_press", "done")
    )
    dynamic_mask = field_mask[..., dynamic_fields]
    dynamic_correct = predictions[..., dynamic_fields] == targets[..., dynamic_fields]
    result = {
        "state_field_accuracy": float(
            (((predictions == targets) & field_mask).sum()) / field_mask.sum()
        ),
        "state_exact_accuracy": float(exact.sum() / valid.sum()),
        "post_event_state_exact_accuracy": float(exact[:, 1:].sum() / valid[:, 1:].sum()),
        "state_sequence_exact_accuracy": float(np.mean(np.all(exact | ~valid, axis=1))),
        "state_final_exact_accuracy": float(np.mean(final)),
        "dynamic_field_accuracy": float(
            (dynamic_correct & dynamic_mask).sum() / dynamic_mask.sum()
        ),
    }
    for field in dynamic_fields:
        name = teacher_lib.STATE_FIELDS[field]
        mask = field_mask[..., field]
        result[f"field/{name}_accuracy"] = float(
            ((predictions[..., field] == targets[..., field]) & mask).sum() / mask.sum()
        )
    return result


def _corrupt_visuals(
    batch: dict[str, np.ndarray], mode: str, seed: int
) -> dict[str, np.ndarray]:
    result = {key: np.array(value, copy=True) for key, value in batch.items()}
    if mode == "zero":
        result["patch_tokens"].fill(0)
    elif mode == "permuted":
        valid = result["event_mask"]
        values = np.array(result["patch_tokens"][valid], copy=True)
        rng = np.random.default_rng(seed)
        result["patch_tokens"][valid] = values[rng.permutation(len(values))]
    elif mode != "native":
        raise ValueError(mode)
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: PickOracleEventDataset(split, args) for split in ("train", "dev", "test")}
    rng = np.random.default_rng(args.seed)
    model = OracleBoundaryVisualSemanticMemory(
        width=args.width,
        num_memory_tokens=args.memory_tokens,
        memory_depth=args.memory_depth,
        encoder_width=args.encoder_width,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
    )
    initial = datasets["train"].batch(np.arange(min(args.batch_size, len(datasets["train"]))), input_mode=args.input_mode)
    params = model.init(jax.random.key(args.seed), **_inputs(initial), train=False)["params"]
    train_events = datasets["train"].arrays["event_targets"][
        datasets["train"].arrays["event_mask"]
    ]
    class_counts = np.bincount(train_events, minlength=len(PICK_EVENT_NAMES)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts * len(class_counts), 1.0)
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
            outputs = model.apply({"params": current_params}, **_inputs(batch), train=True)
            event_log_probs = jax.nn.log_softmax(outputs["event_logits"], axis=-1)
            event_targets = jnp.asarray(batch["event_targets"])
            event_mask = jnp.asarray(batch["event_mask"], dtype=jnp.float32)
            event_ce = -jnp.take_along_axis(
                event_log_probs, event_targets[..., None], axis=-1
            )[..., 0]
            event_weights = jnp.asarray(class_weights)[event_targets] * event_mask
            event_loss = jnp.sum(event_ce * event_weights) / jnp.maximum(
                jnp.sum(event_weights), 1.0
            )
            state_loss, state_metrics = teacher_lib.compute_teacher_losses(
                outputs,
                jnp.asarray(batch["state_targets"]),
                jnp.asarray(batch["state_field_mask"]),
            )
            loss = args.event_loss_weight * event_loss + args.state_loss_weight * state_loss
            return loss, {
                "loss": loss,
                "event_loss": event_loss,
                "state_loss": state_loss,
                "state_exact_accuracy": state_metrics["state_exact_accuracy"],
                "state_final_exact_accuracy": state_metrics["final_state_exact_accuracy"],
            }

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, metrics

    @jax.jit
    def infer(params, inputs):
        outputs = model.apply({"params": params}, **inputs, train=False)
        return outputs["event_logits"], outputs["state_logits"]

    def evaluate(params, split: str, corruption: str = "native") -> dict[str, object]:
        dataset = datasets[split]
        full = dataset.batch(np.arange(len(dataset)), input_mode=args.input_mode)
        if args.input_mode == "visual":
            full = _corrupt_visuals(full, corruption, args.seed + 91)
        event_logits, state_logits = infer(params, _inputs(full))
        return {
            **_event_metrics(
                np.asarray(event_logits), full["event_targets"], full["event_mask"]
            ),
            **_state_metrics(
                np.asarray(state_logits), full["state_targets"], full["state_field_mask"]
            ),
        }

    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_episodes": len(datasets["train"]),
        "dev_episodes": len(datasets["dev"]),
        "test_episodes": len(datasets["test"]),
        "train_event_counts": {
            name: int(value)
            for name, value in zip(PICK_EVENT_NAMES, class_counts, strict=True)
        },
        "oracle_boundary_used": True,
        "teacher_latent_used": False,
        "teacher_readout_used": False,
        "recurrent_input": "predicted_pick_place_press_probability_token",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    best_score = (-1.0, -1.0, -1.0)
    best_step = 0
    best_params = params
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.monotonic()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                dev = evaluate(params, "dev")
                row = {"split": "dev", "step": step, **dev}
                metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                metrics_file.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                score = (
                    float(dev["state_sequence_exact_accuracy"]),
                    float(dev["state_final_exact_accuracy"]),
                    float(dev["event_macro_recall"]),
                )
                if score > best_score:
                    best_score = score
                    best_step = step
                    best_params = jax.device_get(params)
            if step == args.steps:
                break
            indices = rng.choice(
                len(datasets["train"]), args.batch_size, replace=True
            )
            batch = datasets["train"].batch(indices, input_mode=args.input_mode)
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
    (args.output_dir / "best/params").write_bytes(
        flax.serialization.to_bytes(best_params)
    )
    test = evaluate(best_params, "test")
    result: dict[str, object] = {
        "input_mode": args.input_mode,
        "best_step": best_step,
        "best_dev_score": best_score,
        "test": test,
        "elapsed_seconds": time.monotonic() - started,
    }
    if args.input_mode == "visual":
        result["test_zero_visual"] = evaluate(best_params, "test", "zero")
        result["test_permuted_visual"] = evaluate(best_params, "test", "permuted")
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
