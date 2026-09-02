#!/usr/bin/env python3
"""Distill the decomposed visual ceiling into recurrent semantic region MEM."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import flax
from flax import traverse_util
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.tasks.robomme import unified_gt_teacher as contract  # noqa: E402
from openpi.tasks.robomme.decomposed_region_recurrent_memory import DecomposedRegionRecurrentMemory  # noqa: E402
from openpi.tasks.robomme.decomposed_region_recurrent_memory import SWAP_PAIRS  # noqa: E402


TASKS = contract.TASKS[:3]
DEFAULT_FIXED = ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_TEACHER = ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
DEFAULT_FEATURES = ROOT / "artifacts/robomme_four_task_fixed_chunk_features_4x4_v1_260826"
DEFAULT_INIT = ROOT / "checkpoints/robomme_four_task_fixed_chunk_soft_gate_final_ft_v1_260827/best/params"
DEFAULT_OUTPUT = ROOT / "checkpoints/robomme_decomposed_region_distillation_seed260828_260828"
TABLE_FIELDS = tuple(
    contract.STATE_FIELDS.index(name)
    for name in (
        "red_cell",
        "green_cell",
        "blue_cell",
        "ordered_cell_0",
        "ordered_cell_1",
        "ordered_cell_2",
        "ordered_cell_3",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--operation-pretrain-steps",
        type=int,
        default=800,
        help="Train the ceiling-decomposed operation heads before recurrent rollout losses.",
    )
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--end-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--event-type-weight", type=float, default=1.0)
    parser.add_argument("--write-entity-weight", type=float, default=0.5)
    parser.add_argument("--write-region-weight", type=float, default=1.0)
    parser.add_argument("--swap-pair-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-weight", type=float, default=2.0)
    parser.add_argument("--final-weight", type=float, default=4.0)
    parser.add_argument("--hold-weight", type=float, default=0.2)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--gate-temperature", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260828)
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


class RegionDataset:
    def __init__(self, split: str, args: argparse.Namespace):
        self.split = split
        self.fixed = _load(args.fixed_dir / f"{split}.npz")
        self.teacher = _load(args.teacher_dir / f"{split}.npz")
        self.features = h5py.File(args.feature_dir / f"{split}.h5", "r")
        self.rows = np.flatnonzero(self.fixed["task_ids"] < len(TASKS))
        self.max_steps = self.fixed["step_mask"].shape[1]
        count = len(self.fixed["task_ids"])
        self.event_type = np.zeros((count, self.max_steps, 2), dtype=np.int32)
        self.write_entity = np.zeros_like(self.event_type)
        self.write_region = np.zeros_like(self.event_type)
        self.swap_pair = np.zeros_like(self.event_type)
        self.write_mask = np.zeros_like(self.event_type, dtype=np.bool_)
        self.swap_mask = np.zeros_like(self.event_type, dtype=np.bool_)
        self.micro_mask = np.repeat(
            self.fixed["step_mask"][:, :, None], 2, axis=2
        )
        self.table_targets = np.zeros(
            (count, self.max_steps + 1, 7), dtype=np.int32
        )
        self.table_mask = np.zeros_like(self.table_targets, dtype=np.bool_)
        self._build_labels()
        valid_types = self.event_type[self.micro_mask & (self.fixed["task_ids"][:, None, None] < 3)]
        type_counts = np.bincount(valid_types, minlength=3).astype(np.float32)
        weights = type_counts.sum() / np.maximum(3.0 * type_counts, 1.0)
        self.event_type_weights = weights / weights.mean()

    def close(self) -> None:
        self.features.close()

    def _build_labels(self) -> None:
        written_field = contract.STATE_FIELDS.index("written_count")
        pair_lookup = {pair: index for index, pair in enumerate(SWAP_PAIRS)}
        for row in self.rows:
            row = int(row)
            state_index = self.fixed["teacher_state_index"][row]
            canonical_targets = self.teacher["state_targets"][row]
            canonical_mask = self.teacher["state_field_mask"][row]
            self.table_targets[row] = canonical_targets[state_index][:, TABLE_FIELDS]
            self.table_mask[row] = canonical_mask[state_index][:, TABLE_FIELDS]
            length = int(self.fixed["step_mask"][row].sum())
            for chunk in range(length):
                before = int(state_index[chunk])
                after = int(state_index[chunk + 1])
                if after - before > 2:
                    raise ValueError(f"More than two events in {self.split}:{row}:{chunk}")
                for micro, event_index in enumerate(range(before, after)):
                    event_id = int(self.teacher["event_ids"][row, event_index])
                    event = contract.EVENTS[event_id]
                    if event in ("target_visible", "target_covered"):
                        self.event_type[row, chunk, micro] = 1
                        self.write_entity[row, chunk, micro] = (
                            int(self.teacher["entity_ids"][row, event_index]) - 1
                        )
                        self.write_region[row, chunk, micro] = (
                            int(self.teacher["region_a_ids"][row, event_index]) - 1
                        )
                        self.write_mask[row, chunk, micro] = True
                    elif event == "place_complete":
                        self.event_type[row, chunk, micro] = 1
                        post_written = int(canonical_targets[event_index + 1, written_field])
                        self.write_entity[row, chunk, micro] = 3 + post_written - 1
                        self.write_region[row, chunk, micro] = (
                            int(self.teacher["region_a_ids"][row, event_index]) - 1
                        )
                        self.write_mask[row, chunk, micro] = True
                    elif event == "swap_complete":
                        self.event_type[row, chunk, micro] = 2
                        pair = tuple(
                            sorted(
                                (
                                    int(self.teacher["region_a_ids"][row, event_index]) - 1,
                                    int(self.teacher["region_b_ids"][row, event_index]) - 1,
                                )
                            )
                        )
                        self.swap_pair[row, chunk, micro] = pair_lookup[pair]
                        self.swap_mask[row, chunk, micro] = True
                    else:
                        raise ValueError(f"Unexpected region event {event}")

    def sample(self, rng: np.random.Generator, batch_size: int) -> np.ndarray:
        if batch_size % len(TASKS):
            raise ValueError("batch size must be divisible by three")
        per_task = batch_size // len(TASKS)
        result = []
        for task_id in range(len(TASKS)):
            candidates = self.rows[self.fixed["task_ids"][self.rows] == task_id]
            result.extend(rng.choice(candidates, per_task, replace=False))
        rng.shuffle(result)
        return np.asarray(result, dtype=np.int32)

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        batch_size = len(indices)
        patches = np.zeros(
            (batch_size, self.max_steps, 12, 16, 1152), dtype=np.float16
        )
        for batch_row, row in enumerate(indices):
            values = self.features[f"episode_{int(row):06d}/patch_tokens"][()]
            patches[batch_row, : len(values)] = values
        return {
            "patch_tokens": patches,
            "sequence_mask": self.fixed["step_mask"][indices],
            "task_ids": self.fixed["task_ids"][indices],
            "goal_color_ids": self.fixed["goal_color_ids"][indices],
            "queried_ordinals": self.fixed["queried_ordinals"][indices],
            "num_regions": self.fixed["num_regions"][indices],
            "episode_index": self.fixed["episode_index"][indices],
            "event_type": self.event_type[indices],
            "write_entity": self.write_entity[indices],
            "write_region": self.write_region[indices],
            "swap_pair": self.swap_pair[indices],
            "write_mask": self.write_mask[indices],
            "swap_mask": self.swap_mask[indices],
            "micro_mask": self.micro_mask[indices],
            "table_targets": self.table_targets[indices],
            "table_mask": self.table_mask[indices],
            "state_change_mask": self.fixed["state_change_mask"][indices],
        }


def _model_inputs(batch: dict[str, Any]) -> dict[str, jax.Array]:
    return {
        key: jnp.asarray(batch[key])
        for key in (
            "patch_tokens",
            "sequence_mask",
            "task_ids",
            "goal_color_ids",
            "queried_ordinals",
            "num_regions",
        )
    }


def _masked_ce(logits, targets, mask, class_weights=None):
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits, targets.astype(jnp.int32)
    )
    weights = mask.astype(jnp.float32)
    # Normalize by the effective weight rather than the raw sample count.
    # Otherwise inverse-frequency weights make the entire event loss roughly
    # ten times smaller on the very hold-heavy fixed-chunk corpus.
    if class_weights is not None:
        weights = weights * class_weights[targets.astype(jnp.int32)]
    return jnp.sum(losses * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def _restore_matching(params, path: Path):
    restored = flax.serialization.msgpack_restore(path.read_bytes())
    flat = traverse_util.flatten_dict(params)
    source = traverse_util.flatten_dict(restored)
    loaded = []
    for key, value in source.items():
        if key in flat and np.shape(value) == np.shape(flat[key]):
            flat[key] = jnp.asarray(value, dtype=flat[key].dtype)
            loaded.append("/".join(key))
    print(json.dumps({"initialized_from": str(path), "loaded_leaves": len(loaded)}))
    return traverse_util.unflatten_dict(flat)


def _summary(output: dict[str, np.ndarray], batch: dict[str, np.ndarray]) -> dict[str, Any]:
    tables = np.argmax(output["all_tables"], axis=-1)
    event_type = np.argmax(output["event_type_logits"], axis=-1)
    write_entity = np.argmax(output["write_entity_logits"], axis=-1)
    write_region = np.argmax(output["write_region_logits"], axis=-1)
    swap_pair = np.argmax(output["swap_pair_logits"], axis=-1)
    result: dict[str, Any] = {}
    for task_id, task in enumerate(TASKS):
        rows = np.flatnonzero(batch["task_ids"] == task_id)
        query_correct = []
        episode_correct = []
        state_exact_values = []
        for row in rows:
            length = int(batch["sequence_mask"][row].sum())
            if task_id in (0, 1):
                fields = [int(color) - 1 for color in batch["goal_color_ids"][row] if int(color) > 0]
            else:
                fields = [3 + int(batch["queried_ordinals"][row]) - 1]
            values = tables[row, length, fields] == batch["table_targets"][row, length, fields]
            query_correct.extend(values.tolist())
            episode_correct.append(bool(values.all()))
            mask = batch["table_mask"][row, : length + 1]
            exact = np.all(
                (tables[row, : length + 1] == batch["table_targets"][row, : length + 1]) | ~mask,
                axis=-1,
            )
            state_exact_values.extend(exact.tolist())
        result[task] = {
            "episodes": len(rows),
            "queries": len(query_correct),
            "final_query_accuracy": float(np.mean(query_correct)),
            "final_episode_exact_accuracy": float(np.mean(episode_correct)),
            "region_state_trajectory_exact_accuracy": float(np.mean(state_exact_values)),
        }
    micro = batch["micro_mask"]
    write = batch["write_mask"]
    swap = batch["swap_mask"]
    result["operation"] = {
        "event_type_accuracy": float((event_type[micro] == batch["event_type"][micro]).mean()),
        "write_entity_accuracy": float((write_entity[write] == batch["write_entity"][write]).mean()),
        "write_region_accuracy": float((write_region[write] == batch["write_region"][write]).mean()),
        "swap_pair_accuracy": float((swap_pair[swap] == batch["swap_pair"][swap]).mean()),
        "write_events": int(write.sum()),
        "swap_events": int(swap.sum()),
    }
    task_values = [result[task]["final_query_accuracy"] for task in TASKS]
    result["overall"] = {
        "min_task_final_query_accuracy": float(min(task_values)),
        "mean_task_final_query_accuracy": float(np.mean(task_values)),
        "mean_task_trajectory_accuracy": float(
            np.mean([result[task]["region_state_trajectory_exact_accuracy"] for task in TASKS])
        ),
    }
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size % 3 or args.eval_batch_size % 3:
        raise ValueError("train and eval batch sizes must be divisible by three")
    datasets = {split: RegionDataset(split, args) for split in ("train", "dev", "test")}
    try:
        model = DecomposedRegionRecurrentMemory(
            encoder_width=args.encoder_width,
            encoder_depth=args.encoder_depth,
            encoder_heads=args.encoder_heads,
            gate_temperature=args.gate_temperature,
        )
        rng = np.random.default_rng(args.seed)
        initial = datasets["train"].batch(datasets["train"].sample(rng, args.batch_size))
        params = model.init(jax.random.key(args.seed), **_model_inputs(initial), train=False)["params"]
        if args.init_checkpoint is not None:
            params = _restore_matching(params, args.init_checkpoint)
        schedule = optax.warmup_cosine_decay_schedule(
            0.0,
            args.learning_rate,
            min(args.warmup_steps, args.steps - 1),
            args.steps,
            end_value=args.end_learning_rate,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adamw(schedule, weight_decay=args.weight_decay),
        )
        opt_state = optimizer.init(params)
        type_weights = jnp.asarray(datasets["train"].event_type_weights)

        def objective(current_params, batch, recurrent_weight, *, train: bool):
            output = model.apply({"params": current_params}, **_model_inputs(batch), train=train)
            event_loss = _masked_ce(
                output["event_type_logits"],
                jnp.asarray(batch["event_type"]),
                jnp.asarray(batch["micro_mask"]),
                type_weights,
            )
            entity_loss = _masked_ce(
                output["write_entity_logits"],
                jnp.asarray(batch["write_entity"]),
                jnp.asarray(batch["write_mask"]),
            )
            region_loss = _masked_ce(
                output["write_region_logits"],
                jnp.asarray(batch["write_region"]),
                jnp.asarray(batch["write_mask"]),
            )
            pair_loss = _masked_ce(
                output["swap_pair_logits"],
                jnp.asarray(batch["swap_pair"]),
                jnp.asarray(batch["swap_mask"]),
            )
            probabilities = jnp.clip(output["all_tables"], 1e-6, 1.0)
            targets = jnp.asarray(batch["table_targets"], dtype=jnp.int32)
            table_ce = -jnp.log(
                jnp.take_along_axis(probabilities, targets[..., None], axis=-1)[..., 0]
            )
            table_mask = jnp.asarray(batch["table_mask"], dtype=jnp.float32)
            valid_states = jnp.concatenate(
                (jnp.ones((table_mask.shape[0], 1)), jnp.asarray(batch["sequence_mask"], dtype=jnp.float32)),
                axis=1,
            )
            trajectory_loss = jnp.sum(table_ce * table_mask) / jnp.maximum(jnp.sum(table_mask), 1.0)
            lengths = jnp.sum(valid_states, axis=1).astype(jnp.int32) - 1
            final_ce = table_ce[jnp.arange(table_ce.shape[0]), lengths]
            final_mask = table_mask[jnp.arange(table_mask.shape[0]), lengths]
            final_loss = jnp.sum(final_ce * final_mask) / jnp.maximum(jnp.sum(final_mask), 1.0)
            hold = jnp.asarray(batch["sequence_mask"] & ~batch["state_change_mask"], dtype=jnp.float32)
            table_delta = jnp.mean(
                jnp.square(output["all_tables"][:, 1:] - output["all_tables"][:, :-1]),
                axis=(-2, -1),
            )
            hold_loss = jnp.sum(table_delta * hold) / jnp.maximum(jnp.sum(hold), 1.0)
            operation_loss = (
                args.event_type_weight * event_loss
                + args.write_entity_weight * entity_loss
                + args.write_region_weight * region_loss
                + args.swap_pair_weight * pair_loss
            )
            recurrent_loss = (
                args.trajectory_weight * trajectory_loss
                + args.final_weight * final_loss
                + args.hold_weight * hold_loss
            )
            loss = operation_loss + recurrent_weight * recurrent_loss
            metrics = {
                "loss": loss,
                "operation_loss": operation_loss,
                "recurrent_loss": recurrent_loss,
                "event_type_loss": event_loss,
                "write_entity_loss": entity_loss,
                "write_region_loss": region_loss,
                "swap_pair_loss": pair_loss,
                "trajectory_loss": trajectory_loss,
                "final_loss": final_loss,
                "hold_loss": hold_loss,
            }
            return loss, (metrics, output)

        @jax.jit
        def train_step(current_params, current_opt, batch, recurrent_weight):
            (loss, (metrics, _)), grads = jax.value_and_grad(objective, has_aux=True)(
                current_params, batch, recurrent_weight, train=True
            )
            updates, next_opt = optimizer.update(grads, current_opt, current_params)
            return optax.apply_updates(current_params, updates), next_opt, metrics

        @jax.jit
        def infer(current_params, batch):
            return model.apply({"params": current_params}, **_model_inputs(batch), train=False)

        def evaluate(current_params, split: str) -> dict[str, Any]:
            outputs: dict[str, list[np.ndarray]] = defaultdict(list)
            batches = []
            rows = datasets[split].rows
            for start in range(0, len(rows), args.eval_batch_size):
                indices = rows[start : start + args.eval_batch_size]
                if len(indices) < args.eval_batch_size:
                    indices = np.pad(indices, (0, args.eval_batch_size - len(indices)), mode="edge")
                    valid = len(rows) - start
                else:
                    valid = len(indices)
                batch = datasets[split].batch(indices)
                output = jax.device_get(infer(current_params, batch))
                for key, value in output.items():
                    outputs[key].append(np.asarray(value)[:valid])
                batches.append({key: value[:valid] for key, value in batch.items()})
            merged_output = {key: np.concatenate(value) for key, value in outputs.items()}
            merged_batch = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
            return _summary(merged_output, merged_batch)

        best_params = params
        best_score = (-1.0, -1.0, -1.0)
        best_step = 0
        history = []
        for step_index in range(1, args.steps + 1):
            batch = datasets["train"].batch(datasets["train"].sample(rng, args.batch_size))
            # Stage 1 copies the ceiling's decomposed local decisions.  Stage 2
            # turns on free recurrent rollout only after the gate/payload heads
            # have a usable initialization.  A short ramp avoids an abrupt loss
            # scale change at the boundary.
            recurrent_weight = np.clip(
                (step_index - args.operation_pretrain_steps) / 100.0, 0.0, 1.0
            ).astype(np.float32)
            params, opt_state, metrics = train_step(
                params, opt_state, batch, jnp.asarray(recurrent_weight)
            )
            if step_index % args.eval_every == 0 or step_index == args.steps:
                dev = evaluate(params, "dev")
                score = (
                    dev["overall"]["mean_task_final_query_accuracy"],
                    dev["overall"]["min_task_final_query_accuracy"],
                    dev["overall"]["mean_task_trajectory_accuracy"],
                )
                if score > best_score:
                    best_score = score
                    best_step = step_index
                    best_params = jax.device_get(params)
                row = {
                    "step": step_index,
                    "recurrent_weight": float(recurrent_weight),
                    **{key: float(value) for key, value in metrics.items()},
                    "selection_score": score,
                    "dev": dev,
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        result = {
            "schema_version": 1,
            "experiment": "ceiling_decomposition_to_recurrent_semantic_table_mem",
            "best_step": best_step,
            "best_score": best_score,
            "metrics": {
                split: evaluate(best_params, split) for split in ("train", "dev", "test")
            },
            "history": history,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "params.msgpack").write_bytes(flax.serialization.to_bytes(best_params))
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output_dir / "training_config.json").write_text(
            json.dumps(
                {
                    **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    "event_type_weights": datasets["train"].event_type_weights.tolist(),
                    "jax_devices": [str(device) for device in jax.devices()],
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(result["metrics"], indent=2), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
