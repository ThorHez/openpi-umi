#!/usr/bin/env python3
"""Train the unified semantic-feedback student on PickXTimes fixed chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib  # noqa: E402
from openpi.tasks.robomme import unified_semantic_feedback_student as model_lib  # noqa: E402
from scripts.mem import train_robomme_four_task_fixed_chunk_distillation as data_lib  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=_ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=_ROOT / "artifacts/robomme_four_task_fixed_chunk_features_4x4_v1_260826",
    )
    parser.add_argument(
        "--proprio-dir",
        type=Path,
        default=_ROOT / "artifacts/pickxtimes_fixed_chunk_proprio_v1_260828",
    )
    parser.add_argument(
        "--teacher-memory-dir",
        type=Path,
        default=_ROOT / "artifacts/robomme_four_task_gt_teacher_memory_v2_260826",
    )
    parser.add_argument(
        "--teacher-sequence-dir",
        type=Path,
        default=_ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--teacher-forcing-steps", type=int, default=400)
    parser.add_argument(
        "--supervision-mode",
        choices=("full", "terminal_only"),
        default="full",
        help=(
            "terminal_only is the strict no-teacher ablation: no intermediate "
            "state loss and no previous-state teacher forcing."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--end-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260868)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--straight-through-hard-feedback",
        action="store_true",
        help="Use argmax semantic feedback in the forward pass with straight-through gradients.",
    )
    return parser.parse_args()


def _prepare_dataset_args(args: argparse.Namespace) -> None:
    args.task = "pickxtimes_local_event"
    args.pick_transition_balanced_loss = False
    args.groundsg_action_teacher_dir = None
    args.groundsg_action_min_confidence = 0.0


def _inputs(batch: dict[str, np.ndarray], *, teacher_forcing: bool) -> dict[str, jax.Array]:
    sequence_mask = jnp.asarray(batch["sequence_mask"])
    teacher_force_mask = sequence_mask & jnp.asarray(teacher_forcing, dtype=jnp.bool_)
    return {
        "patch_tokens": jnp.asarray(batch["patch_tokens"]),
        "proprio": jnp.asarray(batch["proprio"]),
        "sequence_mask": sequence_mask,
        "initial_state_targets": jnp.asarray(batch["state_targets"][:, 0]),
        "state_field_mask": jnp.asarray(batch["state_field_mask"][:, 1:]),
        "teacher_previous_targets": jnp.asarray(batch["state_targets"][:, :-1]),
        "teacher_force_mask": jnp.asarray(teacher_force_mask),
    }


def _loss_and_metrics(
    logits: jax.Array,
    batch: dict[str, np.ndarray],
    *,
    supervision_mode: str = "full",
):
    targets = jnp.asarray(batch["state_targets"][:, 1:])
    field_mask = jnp.asarray(batch["state_field_mask"][:, 1:]).astype(jnp.float32)
    valid = jnp.asarray(batch["sequence_mask"]).astype(jnp.float32)
    transition = jnp.asarray(batch["state_change_mask"]).astype(jnp.float32) * valid
    hold = (1.0 - jnp.asarray(batch["state_change_mask"]).astype(jnp.float32)) * valid
    token_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(logits[:, 1:], axis=-1), targets[..., None], axis=-1
    )[..., 0]
    state_loss = jnp.sum(token_loss * field_mask, axis=-1) / jnp.maximum(
        jnp.sum(field_mask, axis=-1), 1.0
    )

    def mean_on(mask):
        return jnp.sum(state_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    transition_loss = mean_on(transition)
    hold_loss = mean_on(hold)
    overall_loss = mean_on(valid)
    lengths = jnp.sum(jnp.asarray(batch["sequence_mask"], dtype=jnp.int32), axis=1) - 1
    rows = jnp.arange(logits.shape[0])
    final_logits = logits[:, 1:][rows, lengths]
    final_targets = targets[rows, lengths]
    answer_fields = jnp.asarray(
        tuple(
            teacher_lib.STATE_FIELDS.index(name)
            for name in ("completed_count", "done")
        ),
        dtype=jnp.int32,
    )
    final_answer_logits = final_logits[:, answer_fields]
    final_answer_targets = final_targets[:, answer_fields]
    terminal_answer_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(final_answer_logits, axis=-1),
        final_answer_targets[..., None],
        axis=-1,
    )[..., 0].mean()
    terminal_prediction = jnp.argmax(final_answer_logits, axis=-1)
    terminal_answer_exact = jnp.mean(
        jnp.all(terminal_prediction == final_answer_targets, axis=-1).astype(jnp.float32)
    )
    if supervision_mode == "terminal_only":
        loss = terminal_answer_loss
    else:
        loss = 0.5 * transition_loss + 0.5 * hold_loss + 0.1 * overall_loss
    prediction = jnp.argmax(logits[:, 1:], axis=-1)
    state_exact = jnp.all((prediction == targets) | ~field_mask.astype(jnp.bool_), axis=-1)

    def accuracy_on(mask):
        return jnp.sum(state_exact.astype(jnp.float32) * mask) / jnp.maximum(
            jnp.sum(mask), 1.0
        )

    return loss, {
        "loss": loss,
        "transition_loss": transition_loss,
        "hold_loss": hold_loss,
        "overall_loss": overall_loss,
        "transition_state_exact_accuracy": accuracy_on(transition),
        "no_change_state_exact_accuracy": accuracy_on(hold),
        "state_exact_accuracy": accuracy_on(valid),
        "terminal_answer_loss": terminal_answer_loss,
        "terminal_answer_exact_accuracy": terminal_answer_exact,
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.teacher_forcing_steps <= args.steps:
        raise ValueError("--teacher-forcing-steps must be between 0 and --steps")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_dataset_args(args)
    datasets = {split: data_lib.SplitDataset(split, args) for split in data_lib.SPLITS}
    task_id = teacher_lib.TASKS.index(args.task)
    pools = {
        split: np.flatnonzero(dataset.sequence["task_ids"] == task_id)
        for split, dataset in datasets.items()
    }
    model = model_lib.UnifiedSemanticFeedbackStudent(
        max_steps=datasets["train"].max_steps,
        proprio_dim=datasets["train"].proprio_dim,
        width=args.width,
        encoder_width=args.encoder_width,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        straight_through_hard_feedback=args.straight_through_hard_feedback,
    )
    rng = np.random.default_rng(args.seed)
    initial_indices = rng.choice(pools["train"], args.batch_size, replace=True)
    initial_batch = datasets["train"].batch(
        initial_indices, change_state_weight=1.0, final_state_weight=1.0
    )
    params = model.init(
        jax.random.key(args.seed),
        **_inputs(
            initial_batch,
            teacher_forcing=args.supervision_mode == "full",
        ),
        train=False,
    )["params"]
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
    def train_step(params, opt_state, batch, teacher_forcing):
        def objective(current_params):
            outputs = model.apply(
                {"params": current_params},
                **_inputs(batch, teacher_forcing=teacher_forcing),
                train=True,
            )
            return _loss_and_metrics(
                outputs["state_logits"],
                batch,
                supervision_mode=args.supervision_mode,
            )

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, metrics

    @jax.jit
    def infer(params, batch, teacher_forcing):
        return model.apply(
            {"params": params},
            **_inputs(batch, teacher_forcing=teacher_forcing),
            train=False,
        )["state_logits"]

    dynamic_fields = tuple(
        teacher_lib.STATE_FIELDS.index(name)
        for name in ("completed_count", "holding", "ready_to_press", "done")
    )

    def evaluate(split: str, current_params, *, teacher_forcing: bool):
        dataset = datasets[split]
        logits, targets, masks, changes, lengths = [], [], [], [], []
        for start in range(0, len(pools[split]), args.batch_size):
            indices = pools[split][start : start + args.batch_size]
            real_count = len(indices)
            if real_count < args.batch_size:
                indices = np.pad(indices, (0, args.batch_size - real_count), mode="edge")
            batch = dataset.batch(indices, change_state_weight=1.0, final_state_weight=1.0)
            logits.append(np.asarray(infer(current_params, batch, teacher_forcing))[:real_count])
            targets.append(batch["state_targets"][:real_count])
            masks.append(batch["state_field_mask"][:real_count])
            changes.append(batch["state_change_mask"][:real_count])
            lengths.append(batch["sequence_mask"][:real_count].sum(axis=1))
        logits = np.concatenate(logits)
        targets = np.concatenate(targets)
        masks = np.concatenate(masks)
        changes = np.concatenate(changes)
        lengths = np.concatenate(lengths).astype(np.int64)
        summary = data_lib._host_summary(  # noqa: SLF001
            logits,
            targets,
            masks,
            dataset.sequence["task_ids"][pools[split]],
            changes,
        )["overall"]
        summary["dynamic_state"] = data_lib._host_summary(  # noqa: SLF001
            logits[..., dynamic_fields, :],
            targets[..., dynamic_fields],
            masks[..., dynamic_fields],
            dataset.sequence["task_ids"][pools[split]],
            changes,
        )["overall"]
        rows = np.arange(len(logits))
        answer_fields = tuple(
            teacher_lib.STATE_FIELDS.index(name)
            for name in ("completed_count", "done")
        )
        answer_prediction = np.argmax(
            logits[rows, lengths][:, answer_fields], axis=-1
        )
        answer_targets = targets[rows, lengths][:, answer_fields]
        summary["terminal_answer_exact_accuracy"] = float(
            np.all(answer_prediction == answer_targets, axis=-1).mean()
        )
        return summary

    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "architecture": "unified_19_field_semantic_feedback",
        "semantic_feedback_mode": (
            "straight_through_hard" if args.straight_through_hard_feedback else "soft"
        ),
        "task_specific_head": False,
        "teacher_latent_used": False,
        "supervision_mode": args.supervision_mode,
        "privileged_trajectory_teacher_used": args.supervision_mode == "full",
        "initial_state_source": "prompt-derivable GT initial semantic state",
        "checkpoint_selection": (
            "free-rollout terminal-answer accuracy"
            if args.supervision_mode == "terminal_only"
            else "free-rollout min(transition,no-change)"
        ),
        "train_episodes": len(pools["train"]),
        "dev_episodes": len(pools["dev"]),
        "test_episodes": len(pools["test"]),
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    best_params = params
    best_score = (-1.0, -1.0, -1.0, -1.0, -1.0)
    best_step = 0
    started = time.monotonic()
    try:
        with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
            for step in range(1, args.steps + 1):
                indices = rng.choice(pools["train"], args.batch_size, replace=True)
                batch = datasets["train"].batch(
                    indices, change_state_weight=1.0, final_state_weight=1.0
                )
                teacher_forcing = (
                    args.supervision_mode == "full"
                    and step <= args.teacher_forcing_steps
                )
                params, opt_state, metrics = train_step(
                    params, opt_state, batch, teacher_forcing
                )
                if step == 1 or step % 10 == 0:
                    row = {
                        "split": "train",
                        "step": step,
                        "phase": "teacher_forced" if teacher_forcing else "free_rollout",
                        "learning_rate": float(schedule(step)),
                        **{key: float(value) for key, value in metrics.items()},
                    }
                    metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                    metrics_file.flush()
                    print(json.dumps(row, sort_keys=True), flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    dev = evaluate("dev", params, teacher_forcing=False)
                    if args.supervision_mode == "terminal_only":
                        score = (
                            dev["terminal_answer_exact_accuracy"],
                            dev["final_state_exact_accuracy"],
                            dev["state_exact_accuracy"],
                            dev["transition_state_exact_accuracy"],
                            dev["no_change_state_exact_accuracy"],
                        )
                    else:
                        score = (
                            min(
                                dev["transition_state_exact_accuracy"],
                                dev["no_change_state_exact_accuracy"],
                            ),
                            dev["transition_state_exact_accuracy"],
                            dev["no_change_state_exact_accuracy"],
                            dev["state_exact_accuracy"],
                            dev["final_state_exact_accuracy"],
                        )
                    row = {"split": "dev_free_rollout", "step": step, **dev}
                    metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                    metrics_file.flush()
                    print(json.dumps(row, sort_keys=True), flush=True)
                    if score > best_score:
                        best_score = score
                        best_step = step
                        best_params = jax.device_get(params)
                if step % args.save_every == 0 or step == args.steps:
                    path = args.output_dir / str(step)
                    path.mkdir(exist_ok=True)
                    (path / "params").write_bytes(flax.serialization.to_bytes(params))
        (args.output_dir / "best").mkdir(exist_ok=True)
        (args.output_dir / "best/params").write_bytes(flax.serialization.to_bytes(best_params))
        result = {
            "best_step": best_step,
            "best_dev_score": best_score,
            "dev": {
                "teacher_forced": evaluate("dev", best_params, teacher_forcing=True),
                "free_rollout": evaluate("dev", best_params, teacher_forcing=False),
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        if not args.skip_test:
            result["test"] = {
                "teacher_forced": evaluate("test", best_params, teacher_forcing=True),
                "free_rollout": evaluate("test", best_params, teacher_forcing=False),
            }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
