#!/usr/bin/env python3
"""Train PickXtimes goal-conditioned sliding-window event memory."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import pathlib
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from openpi.tasks.robomme.pickxtimes import semantic_memory_event
from openpi.training.mem import robomme_pickxtimes_dataset
from openpi.training.mem.recipes import robomme_pickxtimes_event_semantic_memory_pretrain as loss_recipe

INPUT_KEYS = (
    "window_patch_tokens",
    "window_gripper_closed",
    "prompt_tokens",
    "prompt_mask",
    "sequence_positions",
    "sequence_mask",
    "candidate_valid_mask",
)
TARGET_KEYS = (
    "candidate_valid_mask",
    "event_targets",
    "event_type_targets",
    "event_type_mask",
    "goal_color",
    "goal_required_count",
    "sequence_mask",
    "completed_count_targets",
    "holding_targets",
    "remaining_count_targets",
    "should_press_targets",
    "done_targets",
    "next_event_targets",
    "next_event_mask",
    "teacher_event_logits",
    "teacher_event_type_logits",
    "teacher_distillation_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=pathlib.Path, required=True)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--split", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--init-checkpoint", type=pathlib.Path)
    parser.add_argument("--window-classifier-checkpoint", type=pathlib.Path)
    parser.add_argument("--causal-records", type=pathlib.Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260822)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encoder-width", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--memory-width", type=int, default=64)
    parser.add_argument("--memory-depth", type=int, default=2)
    parser.add_argument("--memory-tokens", type=int, default=128)
    parser.add_argument("--evidence-tokens-per-event", type=int, default=640)
    parser.add_argument("--pick-type-weight", type=float, default=1.0)
    parser.add_argument("--place-type-weight", type=float, default=1.0)
    parser.add_argument("--press-type-weight", type=float, default=1.0)
    parser.add_argument("--train-gripper-fusion-only", action="store_true")
    parser.add_argument("--train-gripper-gate-fusion-only", action="store_true")
    parser.add_argument("--event-type-only-loss", action="store_true")
    parser.add_argument("--train-event-gate-head-only", action="store_true")
    parser.add_argument("--event-gate-only-loss", action="store_true")
    parser.add_argument("--gate-neighborhood-sampling", action="store_true")
    parser.add_argument("--gate-logit-records", type=pathlib.Path)
    parser.add_argument("--distillation-records", type=pathlib.Path)
    parser.add_argument("--event-logit-distill-weight", type=float, default=0.0)
    parser.add_argument("--event-type-logit-distill-weight", type=float, default=0.0)
    parser.add_argument("--train-event-temporal-readout-only", action="store_true")
    parser.add_argument("--use-sequence-event-types", action="store_true")
    parser.add_argument("--train-memory-updater-readout-only", action="store_true")
    parser.add_argument("--memory-state-only-loss", action="store_true")
    parser.add_argument("--train-press-gate-fusion-only", action="store_true")
    parser.add_argument("--press-gate-only-loss", action="store_true")
    parser.add_argument("--press-gate-sampling", action="store_true")
    return parser.parse_args()


def replace_window_classifier(params, source_params):
    """Replace only the event detector/type head while preserving memory."""
    was_frozen = isinstance(params, flax.core.FrozenDict)
    mutable = flax.core.unfreeze(params)
    source_mutable = flax.core.unfreeze(source_params)
    mutable["window_classifier"] = source_mutable["window_classifier"]
    return flax.core.freeze(mutable) if was_frozen else mutable


def restore_compatible_params(target_params, checkpoint: pathlib.Path):
    """Restore shape-compatible leaves and keep newly added leaves initialized."""
    was_frozen = isinstance(target_params, flax.core.FrozenDict)
    target = flax.traverse_util.flatten_dict(flax.core.unfreeze(target_params))
    source = flax.traverse_util.flatten_dict(flax.serialization.msgpack_restore(checkpoint.read_bytes()))
    restored = 0
    for key, reference in target.items():
        candidate = source.get(key)
        if candidate is not None and np.shape(candidate) == np.shape(reference):
            target[key] = jnp.asarray(candidate, dtype=reference.dtype)
            restored += 1
    result = flax.traverse_util.unflatten_dict(target)
    return (flax.core.freeze(result) if was_frozen else result), restored, len(target) - restored


def keep_only_gripper_fusion_updates(updates):
    return {
        key: (value if key == "gripper_type_fusion" else jax.tree_util.tree_map(jnp.zeros_like, value))
        for key, value in updates.items()
    }


def keep_only_gripper_gate_fusion_updates(updates):
    return {
        key: (value if key == "gripper_gate_fusion" else jax.tree_util.tree_map(jnp.zeros_like, value))
        for key, value in updates.items()
    }


def keep_only_event_gate_head_updates(updates):
    """Freeze the encoder/type head and retain only the binary gate Dense."""
    was_frozen = isinstance(updates, flax.core.FrozenDict)
    mutable = flax.core.unfreeze(updates)
    filtered = jax.tree_util.tree_map(jnp.zeros_like, mutable)
    filtered["window_classifier"]["event_classifier"] = mutable["window_classifier"]["event_classifier"]
    return flax.core.freeze(filtered) if was_frozen else filtered


def keep_only_event_temporal_readout_updates(updates):
    """Update temporal/readout encoder leaves and the event gate only."""
    was_frozen = isinstance(updates, flax.core.FrozenDict)
    flat = flax.traverse_util.flatten_dict(flax.core.unfreeze(updates))
    kept = {}
    for key, value in flat.items():
        path = "/".join(key)
        trainable = path.startswith("window_classifier/event_classifier/") or (
            path.startswith("window_classifier/semantic_encoder/")
            and (
                "/temporal_attn/" in path
                or "/temporal_ln/" in path
                or "/readout_attention/" in path
                or "/readout_ln/" in path
                or "/output_ln/" in path
                or path.endswith(("/readout_query", "/relative_temporal_pos_embedding"))
            )
        )
        kept[key] = value if trainable else jnp.zeros_like(value)
    result = flax.traverse_util.unflatten_dict(kept)
    return flax.core.freeze(result) if was_frozen else result


def keep_only_memory_updater_readout_updates(updates):
    """Freeze event/goal modules and update only recurrent state handling."""
    was_frozen = isinstance(updates, flax.core.FrozenDict)
    flat = flax.traverse_util.flatten_dict(flax.core.unfreeze(updates))
    kept = {}
    for key, value in flat.items():
        path = "/".join(key)
        trainable = path.startswith(
            (
                "goal_conditioned_recurrent_memory/recurrent_memory_updater/",
                "memory_state_readout/",
            )
        )
        kept[key] = value if trainable else jnp.zeros_like(value)
    result = flax.traverse_util.unflatten_dict(kept)
    return flax.core.freeze(result) if was_frozen else result


def keep_only_press_gate_fusion_updates(updates):
    return {
        key: (value if key == "press_gate_fusion" else jax.tree_util.tree_map(jnp.zeros_like, value))
        for key, value in updates.items()
    }


def press_gate_loss(outputs, targets):
    valid = targets["candidate_valid_mask"].astype(jnp.bool_)
    positive = (targets["event_targets"] > 0) & valid
    negative = (targets["event_targets"] <= 0) & valid
    losses = optax.sigmoid_binary_cross_entropy(
        outputs["press_event_logits"].astype(jnp.float32),
        targets["event_targets"].astype(jnp.float32),
    )

    def masked_mean(values, mask):
        mask = mask.astype(jnp.float32)
        return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    loss = 0.5 * (masked_mean(losses, positive) + masked_mean(losses, negative))
    predictions = outputs["press_event_logits"] > 0
    return loss, {
        "press_gate_loss": loss,
        "press_recall_at_0p5": masked_mean(predictions, positive),
        "press_negative_rejection_at_0p5": masked_mean(~predictions, negative),
    }


def distillation_losses(outputs, targets):
    mask = targets["teacher_distillation_mask"].astype(jnp.float32)
    denominator = jnp.maximum(jnp.sum(mask), 1.0)
    event_loss = (
        jnp.sum(
            optax.huber_loss(
                outputs["event_logits"].astype(jnp.float32),
                targets["teacher_event_logits"].astype(jnp.float32),
                delta=1.0,
            )
            * mask
        )
        / denominator
    )
    type_mask = mask * targets["event_type_mask"].astype(jnp.float32)
    type_denominator = jnp.maximum(jnp.sum(type_mask) * outputs["event_type_logits"].shape[-1], 1.0)
    type_loss = (
        jnp.sum(
            optax.huber_loss(
                outputs["event_type_logits"].astype(jnp.float32),
                targets["teacher_event_type_logits"].astype(jnp.float32),
                delta=1.0,
            )
            * type_mask[..., None]
        )
        / type_denominator
    )
    return event_loss, type_loss


def jax_batch(batch: Mapping[str, object], keys: tuple[str, ...]) -> dict[str, jax.Array]:
    result = {}
    for key in keys:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        result[key] = jnp.asarray(value)
    return result


def make_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def save_params(params, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flax.serialization.to_bytes(jax.device_get(params)))


def mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive")
    if args.event_type_only_loss and args.event_gate_only_loss:
        raise ValueError("Only one of --event-type-only-loss and --event-gate-only-loss may be set")
    focused_update_modes = sum(
        (
            args.train_gripper_fusion_only,
            args.train_gripper_gate_fusion_only,
            args.train_event_gate_head_only,
            args.train_event_temporal_readout_only,
            args.train_memory_updater_readout_only,
            args.train_press_gate_fusion_only,
        )
    )
    if focused_update_modes > 1:
        raise ValueError("Only one focused update mode may be selected")
    if (args.event_logit_distill_weight or args.event_type_logit_distill_weight) and not args.distillation_records:
        raise ValueError("Nonzero distillation weights require --distillation-records")
    if args.memory_state_only_loss and not args.train_memory_updater_readout_only:
        raise ValueError("--memory-state-only-loss requires --train-memory-updater-readout-only")
    if args.use_sequence_event_types and not args.causal_records:
        raise ValueError("--use-sequence-event-types requires --causal-records")
    press_options = (
        args.train_press_gate_fusion_only,
        args.press_gate_only_loss,
        args.press_gate_sampling,
    )
    if any(press_options) and not all(press_options):
        raise ValueError(
            "PRESS training requires --train-press-gate-fusion-only, "
            "--press-gate-only-loss, and --press-gate-sampling together"
        )
    input_keys = INPUT_KEYS + (("sequence_event_types",) if args.use_sequence_event_types else ())
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_indices = [int(value) for value in split["train_episode_indices"]]
    val_indices = [int(value) for value in split["val_episode_indices"]]
    train_dataset = robomme_pickxtimes_dataset.PickXtimesWindowDataset(
        args.h5,
        args.labels,
        feature_h5_path=args.features,
        causal_records_path=args.causal_records,
        episode_indices=train_indices,
        random_seed=args.seed,
        randomize=True,
        gate_neighborhood_sampling=args.gate_neighborhood_sampling,
        gate_logit_records_path=args.gate_logit_records,
        distillation_records_path=args.distillation_records,
        use_decoded_sequence_event_types=args.use_sequence_event_types,
        memory_only=args.memory_state_only_loss,
        press_gate_sampling=args.press_gate_sampling,
    )
    val_dataset = robomme_pickxtimes_dataset.PickXtimesWindowDataset(
        args.h5,
        args.labels,
        feature_h5_path=args.features,
        causal_records_path=args.causal_records,
        episode_indices=val_indices,
        random_seed=args.seed,
        randomize=False,
        gate_neighborhood_sampling=args.gate_neighborhood_sampling,
        gate_logit_records_path=args.gate_logit_records,
        distillation_records_path=args.distillation_records,
        use_decoded_sequence_event_types=args.use_sequence_event_types,
        memory_only=args.memory_state_only_loss,
        press_gate_sampling=args.press_gate_sampling,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    train_iterator = iter(train_loader)

    tracker = semantic_memory_event.PickXtimesSlidingWindowEventMemoryTracker(
        encoder_width=args.encoder_width,
        encoder_depth=args.encoder_depth,
        memory_width=args.memory_width,
        memory_depth=args.memory_depth,
        num_memory_tokens=args.memory_tokens,
        evidence_tokens_per_event=args.evidence_tokens_per_event,
    )
    example_raw = next(train_iterator)
    example_inputs = jax_batch(example_raw, input_keys)
    init_rng, train_rng = jax.random.split(jax.random.key(args.seed))
    variables = tracker.init(
        init_rng,
        **example_inputs,
        causal_selection=False,
        train=True,
    )
    params = variables["params"]
    if args.init_checkpoint is not None:
        params, restored, initialized = restore_compatible_params(params, args.init_checkpoint)
        print(
            f"Restored initial parameters from {args.init_checkpoint} (restored={restored}, initialized={initialized})",
            flush=True,
        )
    if args.window_classifier_checkpoint is not None:
        classifier_source, _, _ = restore_compatible_params(params, args.window_classifier_checkpoint)
        params = replace_window_classifier(params, classifier_source)
        print(
            f"Restored window classifier from {args.window_classifier_checkpoint}",
            flush=True,
        )
    focused_loss = (
        args.event_type_only_loss
        or args.event_gate_only_loss
        or args.memory_state_only_loss
        or args.press_gate_only_loss
    )
    train_state_losses = args.memory_state_only_loss or not focused_loss
    loss_weights = loss_recipe.PickXtimesLossWeights(
        event=1.0 if args.event_gate_only_loss else (0.0 if focused_loss else 0.5),
        event_type=1.0 if args.event_type_only_loss else (0.0 if focused_loss else 0.5),
        goal=0.0 if focused_loss else 0.5,
        completed_count=1.0 if train_state_losses else 0.0,
        remaining_count=1.0 if train_state_losses else 0.0,
        holding=0.5 if train_state_losses else 0.0,
        should_press=0.5 if train_state_losses else 0.0,
        done=0.25 if train_state_losses else 0.0,
        next_event=0.25 if train_state_losses else 0.0,
        pick_type=args.pick_type_weight,
        place_type=args.place_type_weight,
        press_type=args.press_type_weight,
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=min(args.warmup_steps, args.steps),
        decay_steps=args.steps,
        end_value=args.learning_rate * 0.1,
    )
    optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_inputs, targets, rng):
        def objective(current_params):
            outputs = tracker.apply(
                {"params": current_params},
                **batch_inputs,
                causal_selection=False,
                train=True,
                rngs={"dropout": rng},
            )
            supervised_loss, metrics = loss_recipe.compute_losses(outputs, targets, loss_weights)
            if args.press_gate_only_loss:
                supervised_loss, press_metrics = press_gate_loss(outputs, targets)
                metrics = {**metrics, **press_metrics}
            event_distill, type_distill = distillation_losses(outputs, targets)
            loss = (
                supervised_loss
                + args.event_logit_distill_weight * event_distill
                + args.event_type_logit_distill_weight * type_distill
            )
            return loss, {
                **metrics,
                "loss": loss,
                "supervised_loss": supervised_loss,
                "event_logit_distill_loss": event_distill,
                "event_type_logit_distill_loss": type_distill,
            }

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        if args.train_gripper_fusion_only:
            updates = keep_only_gripper_fusion_updates(updates)
        elif args.train_gripper_gate_fusion_only:
            updates = keep_only_gripper_gate_fusion_updates(updates)
        elif args.train_event_gate_head_only:
            updates = keep_only_event_gate_head_updates(updates)
        elif args.train_event_temporal_readout_only:
            updates = keep_only_event_temporal_readout_updates(updates)
        elif args.train_memory_updater_readout_only:
            updates = keep_only_memory_updater_readout_updates(updates)
        elif args.train_press_gate_fusion_only:
            updates = keep_only_press_gate_fusion_updates(updates)
        next_params = optax.apply_updates(params, updates)
        metrics = {**metrics, "gradient_norm": optax.global_norm(gradients)}
        return next_params, next_opt_state, metrics

    @jax.jit
    def eval_step(params, batch_inputs, targets):
        outputs = tracker.apply(
            {"params": params},
            **batch_inputs,
            causal_selection=False,
            train=False,
        )
        supervised_loss, metrics = loss_recipe.compute_losses(outputs, targets, loss_weights)
        if args.press_gate_only_loss:
            supervised_loss, press_metrics = press_gate_loss(outputs, targets)
            metrics = {**metrics, **press_metrics}
        event_distill, type_distill = distillation_losses(outputs, targets)
        loss = (
            supervised_loss
            + args.event_logit_distill_weight * event_distill
            + args.event_type_logit_distill_weight * type_distill
        )
        return {
            **metrics,
            "loss": loss,
            "supervised_loss": supervised_loss,
            "event_logit_distill_loss": event_distill,
            "event_type_logit_distill_loss": type_distill,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(train_episode_indices=train_indices, val_episode_indices=val_indices)
    args.output_dir.joinpath("config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    log_path = args.output_dir / "metrics.jsonl"
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log_file:
        for step in range(1, args.steps + 1):
            try:
                raw_batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                raw_batch = next(train_iterator)
            batch_inputs = jax_batch(raw_batch, input_keys)
            targets = jax_batch(raw_batch, TARGET_KEYS)
            train_rng, step_rng = jax.random.split(train_rng)
            params, opt_state, metrics = train_step(params, opt_state, batch_inputs, targets, step_rng)
            if step == 1 or step % 10 == 0:
                host_metrics = {key: float(value) for key, value in jax.device_get(metrics).items()}
                record = {"step": step, "split": "train", **host_metrics}
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(
                    f"step={step}/{args.steps} loss={host_metrics['loss']:.4f} "
                    f"event_type_acc={host_metrics['event_type_accuracy']:.3f} "
                    f"count_acc={host_metrics['stage_count_accuracy']:.3f} "
                    f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                    flush=True,
                )
            if step % args.eval_every == 0 or step == args.steps:
                validation = []
                for val_raw in val_loader:
                    val_inputs = jax_batch(val_raw, input_keys)
                    val_targets = jax_batch(val_raw, TARGET_KEYS)
                    result = eval_step(params, val_inputs, val_targets)
                    validation.append({key: float(value) for key, value in jax.device_get(result).items()})
                val_metrics = mean_metrics(validation)
                record = {"step": step, "split": "val", **val_metrics}
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(
                    f"VAL step={step} loss={val_metrics['loss']:.4f} "
                    f"event_type_acc={val_metrics['event_type_accuracy']:.3f} "
                    f"count_acc={val_metrics['stage_count_accuracy']:.3f}",
                    flush=True,
                )
            if step % args.save_every == 0 or step == args.steps:
                save_params(params, args.output_dir / "checkpoints" / f"step_{step}.msgpack")


if __name__ == "__main__":
    main()
