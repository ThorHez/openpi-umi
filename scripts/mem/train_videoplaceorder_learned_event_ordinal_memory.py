#!/usr/bin/env python3
"""Train a deployable RGB+proprio event head with deterministic ordinal memory."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import flax
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

from openpi.tasks.robomme.explicit_event_bottleneck_memory import ExplicitEventBottleneckMemory  # noqa: E402
from scripts.mem import train_robomme_anchor_conditioned_decomposition as anchor_base  # noqa: E402
from scripts.mem import train_robomme_decomposed_region_distillation as base  # noqa: E402
from scripts.mem import train_robomme_explicit_event_bottleneck_ablation as explicit_base  # noqa: E402


DEFAULT_TRAJECTORIES = ROOT / "artifacts/videoplaceorder_observable_event_trajectories_v1_260831"
DEFAULT_FEATURES = ROOT / "artifacts/robomme_fixed_chunk_rgb_grid8_v1_260829"
DEFAULT_INIT = ROOT / "checkpoints/robomme_explicit_event_pooled_soft_causal_seed260908_260829/params.msgpack"
DEFAULT_OUTPUT = ROOT / "checkpoints/videoplaceorder_learned_event_ordinal_seed260831_260831"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supervision-mode",
        choices=("full", "terminal_only"),
        default="full",
        help=(
            "full uses privileged event/state trajectories; terminal_only is the "
            "No-Teacher condition and supervises only the final queried ordinal."
        ),
    )
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--operation-pretrain-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--end-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260831)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class PlaceTrajectoryDataset:
    def __init__(self, split: str, args: argparse.Namespace):
        with np.load(args.trajectory_dir / f"{split}.npz", allow_pickle=False) as source:
            self.data = {key: np.asarray(source[key]) for key in source.files}
        self.features = h5py.File(args.feature_dir / f"{split}.h5", "r")
        self.rows = np.arange(len(self.data["episode_index"]), dtype=np.int32)
        self.max_steps = self.data["sequence_mask"].shape[1]
        self.spatial_tokens = int(self.features.attrs["spatial_tokens"])
        self.patch_width = int(self.features.attrs["patch_width"])
        valid = self.data["micro_mask"]
        counts = np.bincount(self.data["event_type"][valid], minlength=3).astype(np.float32)
        weights = np.sqrt(counts.sum() / np.maximum(3.0 * counts, 1.0))
        self.event_type_weights = weights / weights.mean()

    def close(self) -> None:
        self.features.close()

    def sample(self, rng: np.random.Generator, batch_size: int) -> np.ndarray:
        return rng.choice(self.rows, batch_size, replace=True).astype(np.int32)

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        patches = np.zeros(
            (len(indices), self.max_steps, 12, self.spatial_tokens, self.patch_width),
            dtype=np.float16,
        )
        for batch_row, row in enumerate(indices):
            group = int(self.data["patch_group_index"][row])
            values = self.features[f"episode_{group:06d}/patch_tokens"][()]
            patches[batch_row, : len(values)] = values
        result = {key: value[indices] for key, value in self.data.items()}
        result["patch_tokens"] = patches
        return result


def model_inputs(batch: dict[str, Any], teacher_force_mask) -> dict[str, jax.Array]:
    return {
        "patch_tokens": jnp.asarray(batch["patch_tokens"]),
        "sequence_mask": jnp.asarray(batch["sequence_mask"]),
        "task_ids": jnp.asarray(batch["task_ids"]),
        "goal_color_ids": jnp.asarray(batch["goal_color_ids"]),
        "queried_ordinals": jnp.asarray(batch["queried_ordinals"]),
        "num_regions": jnp.asarray(batch["num_regions"]),
        "anchor_yx": jnp.asarray(batch["anchor_yx"]),
        "anchor_mask": jnp.asarray(batch["anchor_mask"]),
        "teacher_previous_tables": jnp.asarray(batch["table_targets"][:, :-1]),
        "teacher_force_mask": jnp.asarray(teacher_force_mask),
        "proprio_tokens": jnp.asarray(batch["proprio_tokens"]),
    }


def summarize(output: dict[str, np.ndarray], batch: dict[str, np.ndarray]) -> dict[str, float]:
    tables = np.argmax(output["all_tables"], axis=-1)
    events = np.argmax(output["event_type_logits"], axis=-1)
    regions = np.argmax(output["write_region_logits"], axis=-1)
    pairs = np.argmax(output["swap_pair_logits"], axis=-1)
    valid = batch["micro_mask"]
    gt_event = batch["event_type"]
    predicted_update = (events != 0) & valid
    gt_update = (gt_event != 0) & valid
    true_update = predicted_update & gt_update & (events == gt_event)
    transition = []
    hold = []
    trajectory = []
    final = []
    sequence = []
    for row in range(len(batch["episode_index"])):
        length = int(batch["sequence_mask"][row].sum())
        mask = batch["table_mask"][row, : length + 1]
        exact = np.all(
            (tables[row, : length + 1] == batch["table_targets"][row, : length + 1]) | ~mask,
            axis=-1,
        )
        changes = batch["state_change_mask"][row, :length]
        transition.extend(exact[1:][changes].tolist())
        hold.extend(exact[1:][~changes].tolist())
        trajectory.extend(exact[1:].tolist())
        sequence.append(bool(exact.all()))
        field = 3 + int(batch["queried_ordinals"][row]) - 1
        final.append(
            int(tables[row, length, field])
            == int(batch["table_targets"][row, length, field])
        )
    write_mask = batch["write_mask"]
    swap_mask = batch["swap_mask"]
    return {
        "final_ordinal_accuracy": float(np.mean(final)),
        "transition_state_exact_accuracy": float(np.mean(transition)),
        "hold_state_exact_accuracy": float(np.mean(hold)),
        "trajectory_state_exact_accuracy": float(np.mean(trajectory)),
        "full_sequence_exact_accuracy": float(np.mean(sequence)),
        "event_update_precision": float(true_update.sum() / max(predicted_update.sum(), 1)),
        "event_update_recall": float(true_update.sum() / max(gt_update.sum(), 1)),
        "write_region_accuracy": float(np.mean(regions[write_mask] == batch["write_region"][write_mask])),
        "swap_pair_accuracy": float(np.mean(pairs[swap_mask] == batch["swap_pair"][swap_mask])) if swap_mask.any() else 1.0,
        "predicted_updates": int(predicted_update.sum()),
        "gt_updates": int(gt_update.sum()),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: PlaceTrajectoryDataset(split, args) for split in ("train", "dev")}
    try:
        data = datasets["train"]
        model = ExplicitEventBottleneckMemory(
            max_steps=data.max_steps,
            spatial_tokens=data.spatial_tokens,
            input_width=data.patch_width,
            temporal_encoder="pooled",
            deterministic_updater=True,
            causal_evidence_state=True,
            gate_temperature=0.25,
        )
        rng = np.random.default_rng(args.seed)
        initial = data.batch(data.sample(rng, args.batch_size))
        initial_force = initial["sequence_mask"]
        if args.supervision_mode == "terminal_only":
            initial_force = np.zeros_like(initial_force)
        params = model.init(
            jax.random.key(args.seed),
            **model_inputs(initial, initial_force),
        )["params"]
        loaded_leaves = []
        initialization_checkpoint_used = None
        if args.supervision_mode == "full" and args.init_checkpoint is not None:
            params, loaded_leaves = anchor_base._restore_matching_with_names(  # noqa: SLF001
                params, args.init_checkpoint
            )
            initialization_checkpoint_used = str(args.init_checkpoint)
        schedule = optax.warmup_cosine_decay_schedule(
            0.0,
            args.learning_rate,
            min(args.warmup_steps, args.steps - 1),
            args.steps,
            end_value=args.end_learning_rate,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4)
        )
        opt_state = optimizer.init(params)
        type_weights = jnp.asarray(data.event_type_weights)

        def objective(current_params, batch, force, recurrent_weight):
            output = model.apply({"params": current_params}, **model_inputs(batch, force))
            terminal_loss, terminal_accuracy = explicit_base._terminal_answer_loss(  # noqa: SLF001
                output, batch
            )
            if args.supervision_mode == "terminal_only":
                return terminal_loss, {
                    "loss": terminal_loss,
                    "terminal_answer_loss": terminal_loss,
                    "terminal_answer_exact_accuracy": terminal_accuracy,
                }
            event = base._masked_ce(output["event_type_logits"], jnp.asarray(batch["event_type"]), jnp.asarray(batch["micro_mask"]), type_weights)  # noqa: SLF001
            entity = base._masked_ce(output["write_entity_logits"], jnp.asarray(batch["write_entity"]), jnp.asarray(batch["write_mask"]))  # noqa: SLF001
            region = base._masked_ce(output["write_region_logits"], jnp.asarray(batch["write_region"]), jnp.asarray(batch["write_mask"]))  # noqa: SLF001
            pair = base._masked_ce(output["swap_pair_logits"], jnp.asarray(batch["swap_pair"]), jnp.asarray(batch["swap_mask"]))  # noqa: SLF001
            table = explicit_base._smoothed_table_losses(output, batch)  # noqa: SLF001
            query = explicit_base._query_losses(output, batch)  # noqa: SLF001
            operation = 2.0 * event + 0.5 * entity + 3.0 * region + 3.0 * pair
            recurrent = (
                2.0 * table["transition_loss"]
                + 1.0 * table["no_change_loss"]
                + 3.0 * table["delta_loss"]
                + 3.0 * table["final_loss"]
                + 0.1 * table["trajectory_loss"]
                + 3.0 * query["ordinal_binding_loss"]
            )
            loss = operation + recurrent_weight * recurrent
            return loss, {
                "loss": loss,
                "operation_loss": operation,
                "event_loss": event,
                "region_loss": region,
                "pair_loss": pair,
                "terminal_answer_loss": terminal_loss,
                "terminal_answer_exact_accuracy": terminal_accuracy,
                **table,
                **query,
            }

        @jax.jit
        def train_step(current_params, current_opt, batch, force, recurrent_weight):
            (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(
                current_params, batch, force, recurrent_weight
            )
            updates, next_opt = optimizer.update(grads, current_opt, current_params)
            return optax.apply_updates(current_params, updates), next_opt, metrics

        @jax.jit
        def infer(current_params, batch, force):
            return model.apply({"params": current_params}, **model_inputs(batch, force))

        def evaluate(current_params, split: str) -> dict[str, float]:
            outputs: dict[str, list[np.ndarray]] = defaultdict(list)
            batches = []
            split_data = datasets[split]
            for start in range(0, len(split_data.rows), args.eval_batch_size):
                indices = split_data.rows[start : start + args.eval_batch_size]
                valid_count = len(indices)
                if valid_count < args.eval_batch_size:
                    indices = np.pad(indices, (0, args.eval_batch_size - valid_count), mode="edge")
                batch = split_data.batch(indices)
                output = jax.device_get(infer(current_params, batch, np.zeros_like(batch["sequence_mask"])))
                for key, value in output.items():
                    if key != "all_memories":
                        outputs[key].append(np.asarray(value)[:valid_count])
                batches.append({key: np.asarray(value)[:valid_count] for key, value in batch.items()})
            merged_output = {key: np.concatenate(values) for key, values in outputs.items()}
            merged_batch = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
            return summarize(merged_output, merged_batch)

        best_params = params
        best_step = 0
        best_score = (-1.0,) * 5
        history = []
        started = time.monotonic()
        for step in range(1, args.steps + 1):
            batch = data.batch(data.sample(rng, args.batch_size))
            if args.supervision_mode == "terminal_only":
                ratio, recurrent_weight = 0.0, 1.0
            elif step <= args.operation_pretrain_steps:
                ratio, recurrent_weight = 1.0, 0.0
            else:
                progress = (step - args.operation_pretrain_steps) / max(args.steps - args.operation_pretrain_steps, 1)
                ratio = max(0.0, 1.0 - progress)
                recurrent_weight = min(1.0, progress / 0.2)
            force = batch["sequence_mask"] & (rng.random(batch["sequence_mask"].shape) < ratio)
            params, opt_state, train_metrics = train_step(
                params, opt_state, batch, force, jnp.asarray(recurrent_weight, jnp.float32)
            )
            if step % args.eval_every == 0 or step == args.steps:
                dev = evaluate(params, "dev")
                if args.supervision_mode == "terminal_only":
                    # Trajectory labels are diagnostics only in this condition.
                    # Checkpoint selection is based solely on the non-privileged
                    # dev terminal answer, retaining the earliest checkpoint on ties.
                    score = (dev["final_ordinal_accuracy"],)
                    eligible = True
                else:
                    balance = min(dev["transition_state_exact_accuracy"], dev["hold_state_exact_accuracy"], dev["final_ordinal_accuracy"])
                    score = (balance, dev["final_ordinal_accuracy"], dev["transition_state_exact_accuracy"], dev["hold_state_exact_accuracy"], dev["event_update_recall"])
                    eligible = step >= args.operation_pretrain_steps
                if eligible and score > best_score:
                    best_score, best_step, best_params = score, step, jax.device_get(params)
                row = {"step": step, "teacher_force_ratio": ratio, "recurrent_weight": recurrent_weight, "score": score, "train": {key: float(value) for key, value in train_metrics.items()}, "dev_free_rollout": dev}
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        metrics = {split: evaluate(best_params, split) for split in ("train", "dev")}
        result = {
            "schema_version": 1,
            "experiment": "videoplaceorder_learned_event_head_deterministic_ordinal_updater",
            "deployment_inputs": ["RGB", "joint_state", "gripper_state", "goal ordinal", "visual anchors"],
            "supervision_mode": args.supervision_mode,
            "training_only_teacher": (
                "privileged event trajectory"
                if args.supervision_mode == "full"
                else "none; final queried ordinal only"
            ),
            "privileged_trajectory_teacher_used": args.supervision_mode == "full",
            "initialization_checkpoint_used": initialization_checkpoint_used,
            "test_used_for_selection": False,
            "best_step": best_step,
            "best_score": best_score,
            "loaded_initialization_leaves": len(loaded_leaves),
            "elapsed_seconds": time.monotonic() - started,
            "metrics": metrics,
            "history": history,
        }
        (args.output_dir / "params.msgpack").write_bytes(flax.serialization.to_bytes(best_params))
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update({"variant": "pooled_hard_causal_proprio", "max_steps": data.max_steps, "spatial_tokens": data.spatial_tokens, "patch_width": data.patch_width})
        (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
