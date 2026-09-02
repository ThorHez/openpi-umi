#!/usr/bin/env python3
"""Train the context-matched anchor-conditioned decomposed recurrent MEM."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

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

from openpi.tasks.robomme.anchor_conditioned_decomposed_memory import (  # noqa: E402
    AnchorConditionedDecomposedMemory,
)
from scripts.mem import train_robomme_decomposed_region_distillation as base  # noqa: E402


DEFAULT_ANCHORS = ROOT / "checkpoints/robomme_anchor_pointer_probe_v1_260828/data"
DEFAULT_INIT = (
    ROOT
    / "checkpoints/robomme_decomposed_region_operation_only_seed260830_260828/params.msgpack"
)
DEFAULT_OUTPUT = ROOT / "checkpoints/robomme_anchor_conditioned_decomposition_seed260831_260828"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, default=base.DEFAULT_FIXED)
    parser.add_argument("--teacher-dir", type=Path, default=base.DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=base.DEFAULT_FEATURES)
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--end-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--event-type-weight", type=float, default=1.0)
    parser.add_argument("--write-entity-weight", type=float, default=0.5)
    parser.add_argument("--write-region-weight", type=float, default=2.0)
    parser.add_argument("--swap-pair-weight", type=float, default=2.0)
    parser.add_argument("--gate-temperature", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260831)
    return parser.parse_args()


class AnchorRegionDataset(base.RegionDataset):
    def __init__(self, split: str, args: argparse.Namespace):
        super().__init__(split, args)
        pointer = base._load(args.anchor_dir / f"{split}.npz")
        count = len(self.fixed["task_ids"])
        self.anchor_yx = np.zeros((count, 4, 2), dtype=np.float32)
        self.anchor_mask = np.zeros((count, 4), dtype=np.bool_)
        lookup: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for index in range(len(pointer["task_ids"])):
            key = (
                int(pointer["task_ids"][index]),
                int(pointer["episode_index"][index]),
            )
            value = (
                np.asarray(pointer["anchor_yx"][index], dtype=np.float32),
                np.asarray(pointer["anchor_mask"][index], dtype=np.bool_),
            )
            if key in lookup:
                if not (
                    np.allclose(lookup[key][0], value[0])
                    and np.array_equal(lookup[key][1], value[1])
                ):
                    raise ValueError(f"Inconsistent duplicate anchor record {split}:{key}")
            else:
                lookup[key] = value
        for row in self.rows:
            row = int(row)
            key = (
                int(self.fixed["task_ids"][row]),
                int(self.fixed["episode_index"][row]),
            )
            if key not in lookup:
                raise KeyError(f"Missing anchor record {split}:{key}")
            self.anchor_yx[row], self.anchor_mask[row] = lookup[key]
            expected = int(self.fixed["num_regions"][row])
            if int(self.anchor_mask[row].sum()) != expected:
                raise ValueError(
                    f"Anchor count mismatch {split}:{key}: "
                    f"{self.anchor_mask[row].sum()} != {expected}"
                )

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        result = super().batch(indices)
        result["anchor_yx"] = self.anchor_yx[indices]
        result["anchor_mask"] = self.anchor_mask[indices]
        return result


def _model_inputs(batch: dict[str, Any]) -> dict[str, jax.Array]:
    result = base._model_inputs(batch)
    result["anchor_yx"] = jnp.asarray(batch["anchor_yx"])
    result["anchor_mask"] = jnp.asarray(batch["anchor_mask"])
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size % 3 or args.eval_batch_size % 3:
        raise ValueError("train and eval batch sizes must be divisible by three")
    datasets = {
        split: AnchorRegionDataset(split, args) for split in ("train", "dev", "test")
    }
    try:
        model = AnchorConditionedDecomposedMemory(
            gate_temperature=args.gate_temperature
        )
        rng = np.random.default_rng(args.seed)
        initial = datasets["train"].batch(
            datasets["train"].sample(rng, args.batch_size)
        )
        params = model.init(
            jax.random.key(args.seed), **_model_inputs(initial), train=False
        )["params"]
        loaded_leaves = []
        if args.init_checkpoint is not None:
            params, loaded_leaves = _restore_matching_with_names(
                params, args.init_checkpoint
            )
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

        def objective(current_params, batch, *, train: bool):
            output = model.apply(
                {"params": current_params}, **_model_inputs(batch), train=train
            )
            event_loss = base._masked_ce(
                output["event_type_logits"],
                jnp.asarray(batch["event_type"]),
                jnp.asarray(batch["micro_mask"]),
                type_weights,
            )
            entity_loss = base._masked_ce(
                output["write_entity_logits"],
                jnp.asarray(batch["write_entity"]),
                jnp.asarray(batch["write_mask"]),
            )
            region_loss = base._masked_ce(
                output["write_region_logits"],
                jnp.asarray(batch["write_region"]),
                jnp.asarray(batch["write_mask"]),
            )
            pair_loss = base._masked_ce(
                output["swap_pair_logits"],
                jnp.asarray(batch["swap_pair"]),
                jnp.asarray(batch["swap_mask"]),
            )
            loss = (
                args.event_type_weight * event_loss
                + args.write_entity_weight * entity_loss
                + args.write_region_weight * region_loss
                + args.swap_pair_weight * pair_loss
            )
            return loss, {
                "loss": loss,
                "event_type_loss": event_loss,
                "write_entity_loss": entity_loss,
                "write_region_loss": region_loss,
                "swap_pair_loss": pair_loss,
            }

        @jax.jit
        def train_step(current_params, current_opt, batch):
            (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(
                current_params, batch, train=True
            )
            updates, next_opt = optimizer.update(grads, current_opt, current_params)
            return optax.apply_updates(current_params, updates), next_opt, metrics

        @jax.jit
        def infer(current_params, batch):
            return model.apply(
                {"params": current_params}, **_model_inputs(batch), train=False
            )

        def evaluate(current_params, split: str) -> dict[str, Any]:
            outputs: dict[str, list[np.ndarray]] = defaultdict(list)
            batches = []
            rows = datasets[split].rows
            for start in range(0, len(rows), args.eval_batch_size):
                indices = rows[start : start + args.eval_batch_size]
                valid = len(indices)
                if valid < args.eval_batch_size:
                    indices = np.pad(
                        indices, (0, args.eval_batch_size - valid), mode="edge"
                    )
                batch = datasets[split].batch(indices)
                output = jax.device_get(infer(current_params, batch))
                for key, value in output.items():
                    if key != "all_memories":
                        outputs[key].append(np.asarray(value)[:valid])
                batches.append({key: value[:valid] for key, value in batch.items()})
            merged_output = {
                key: np.concatenate(value) for key, value in outputs.items()
            }
            merged_batch = {
                key: np.concatenate([batch[key] for batch in batches])
                for key in batches[0]
            }
            return base._summary(merged_output, merged_batch)

        best_params = params
        best_score = (-1.0, -1.0, -1.0)
        best_step = 0
        history = []
        for step_index in range(1, args.steps + 1):
            batch = datasets["train"].batch(
                datasets["train"].sample(rng, args.batch_size)
            )
            params, opt_state, metrics = train_step(params, opt_state, batch)
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
                    **{key: float(value) for key, value in metrics.items()},
                    "selection_score": score,
                    "dev": dev,
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        result = {
            "schema_version": 1,
            "experiment": "context_matched_anchor_conditioned_operation_distillation",
            "best_step": best_step,
            "best_score": best_score,
            "metrics": {
                split: evaluate(best_params, split)
                for split in ("train", "dev", "test")
            },
            "history": history,
            "input_contract": {
                "visual": "fixed non-overlapping 12-frame 4x4 SigLIP patch tokens",
                "anchors": "episode-local detector/oracle coordinates from existing pointer cache",
                "operation_features": "per-anchor bilinear ROI temporal evidence",
                "write": "shared entity-to-anchor pointer",
                "swap": "shared six-anchor-pair scorer",
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "params.msgpack").write_bytes(
            flax.serialization.to_bytes(best_params)
        )
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        (args.output_dir / "training_config.json").write_text(
            json.dumps(
                {
                    **{
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "event_type_weights": datasets[
                        "train"
                    ].event_type_weights.tolist(),
                    "initialized_leaves": loaded_leaves,
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


def _restore_matching_with_names(params, path: Path):
    restored = flax.serialization.msgpack_restore(path.read_bytes())
    flat = traverse_util.flatten_dict(params)
    source = traverse_util.flatten_dict(restored)
    loaded = []
    for key, value in source.items():
        if key in flat and np.shape(value) == np.shape(flat[key]):
            flat[key] = jnp.asarray(value, dtype=flat[key].dtype)
            loaded.append("/".join(key))
    print(
        json.dumps({"initialized_from": str(path), "loaded_leaves": len(loaded)}),
        flush=True,
    )
    return traverse_util.unflatten_dict(flat), loaded


if __name__ == "__main__":
    main()
