#!/usr/bin/env python3
"""Factorial ablation of temporal evidence and deterministic event execution."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
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
from scripts.mem import eval_robomme_transition_causal_ablation as replay  # noqa: E402
from scripts.mem import train_robomme_anchor_transition_curriculum as transition  # noqa: E402
from scripts.mem import train_robomme_decomposed_region_distillation as base  # noqa: E402
from scripts.mem import train_robomme_visual_operation_parser_ablation as parser_base  # noqa: E402

DEFAULT_FEATURES = ROOT / "artifacts/robomme_fixed_chunk_rgb_grid8_v1_260829"
DEFAULT_OUTPUT = ROOT / "checkpoints/robomme_explicit_event_bottleneck_ablation_260829"
DEFAULT_UNMASK_H5 = ROOT / "data/robomme_extracted/record_dataset_VideoUnmask.h5"
UNMASK_PROMPT_PATTERN = re.compile(r"hiding the (red|green|blue) cube")
VARIANTS = {
    "pooled_soft": ("pooled", False, False),
    "pooled_hard": ("pooled", True, False),
    "pooled_soft_causal": ("pooled", False, True),
    "relational_soft": ("relational", False, False),
    "relational_hard": ("relational", True, False),
    "relational_soft_causal": ("relational", False, True),
    "relational_hard_causal": ("relational", True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument(
        "--supervision-mode",
        choices=("full", "terminal_only"),
        default="full",
        help=(
            "full uses privileged event/state trajectories; terminal_only is the "
            "strict no-teacher ablation and supervises only the episode answer."
        ),
    )
    parser.add_argument("--fixed-dir", type=Path, default=base.DEFAULT_FIXED)
    parser.add_argument("--teacher-dir", type=Path, default=base.DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--anchor-dir", type=Path, default=parser_base.anchor_base.DEFAULT_ANCHORS)
    parser.add_argument("--unmask-h5", type=Path, default=DEFAULT_UNMASK_H5)
    parser.add_argument(
        "--unmask-binding-labels",
        choices=("original", "native_single", "native_full"),
        default="original",
        help=(
            "native_single corrects the existing VideoUnmask target binding from "
            "the simulator-native first-frame geometry; native_full also restores "
            "all target colors from the original one/two-target instruction."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--operation-pretrain-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--end-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--event-type-weight", type=float, default=1.0)
    parser.add_argument("--write-entity-weight", type=float, default=0.5)
    parser.add_argument("--write-region-weight", type=float, default=2.0)
    parser.add_argument("--swap-pair-weight", type=float, default=2.0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--no-change-weight", type=float, default=1.0)
    parser.add_argument("--delta-weight", type=float, default=2.0)
    parser.add_argument("--final-weight", type=float, default=2.0)
    parser.add_argument("--trajectory-weight", type=float, default=0.1)
    parser.add_argument(
        "--ordinal-binding-weight",
        type=float,
        default=0.0,
        help="Final queried-ordinal CE weight for VideoPlaceOrder.",
    )
    parser.add_argument(
        "--completeness-weight",
        type=float,
        default=0.0,
        help="Known-region penalty for every requested color in multi-target episodes.",
    )
    parser.add_argument("--gate-temperature", type=float, default=0.25)
    parser.add_argument("--temporal-depth", type=int, default=2)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260908)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _decode_h5(value) -> str:
    values = np.asarray(value).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"Expected one H5 string, got {len(values)}")
    item = values[0]
    return item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)


def _color_centers(image: np.ndarray) -> dict[int, np.ndarray]:
    """Return simulator-native red/green/blue centers as 1-based color ids."""

    image = np.asarray(image, dtype=np.uint8)
    red, green, blue = (image[..., index] for index in range(3))
    masks = (
        (red > 180) & (green < 70) & (blue < 70),
        (green > 180) & (red < 70) & (blue < 70),
        (blue > 180) & (red < 70) & (green < 70),
    )
    result = {}
    for color_id, mask in enumerate(masks, start=1):
        y, x = np.nonzero(mask)
        if len(y) < 8:
            raise ValueError(f"Could not segment color id {color_id}")
        result[color_id] = np.asarray(
            [np.median(y), np.median(x)], dtype=np.float32
        )
    return result


class NativeUnmaskBindingDataset(parser_base.FixedParserDataset):
    """Optionally replace noisy VideoUnmask teacher bindings with native GT.

    The first fixed chunk causally observes all colored cubes.  Region anchors
    are those same episode-local locations, so simulator-native color centers
    provide exact training-only entity/region events without exposing anything
    unavailable at inference.  The recurrent model and updater are unchanged.
    """

    def __init__(self, split: str, args: argparse.Namespace):
        super().__init__(split, args)
        self.native_binding_audit = {
            "mode": args.unmask_binding_labels,
            "episodes": 0,
            "dual_goal_episodes": 0,
            "corrected_original_targets": 0,
        }
        if args.unmask_binding_labels != "original":
            self._apply_native_unmask_bindings(args)

    def _apply_native_unmask_bindings(self, args: argparse.Namespace) -> None:
        with h5py.File(args.unmask_h5, "r") as source:
            task_rows = self.rows[self.fixed["task_ids"][self.rows] == 0]
            for row_value in task_rows:
                row = int(row_value)
                episode_index = int(self.fixed["episode_index"][row])
                episode = source[f"episode_{episode_index}"]
                prompt = _decode_h5(episode["setup/task_goal"][()])
                prompt_colors = [
                    base.contract.COLORS.index(color)
                    for color in UNMASK_PROMPT_PATTERN.findall(prompt)
                ]
                if not 1 <= len(prompt_colors) <= 2:
                    raise ValueError(
                        f"Expected one/two VideoUnmask targets in {prompt!r}"
                    )
                original_color = int(self.fixed["goal_color_ids"][row, 0])
                selected_colors = (
                    [original_color]
                    if args.unmask_binding_labels == "native_single"
                    else prompt_colors
                )
                if original_color not in prompt_colors:
                    raise ValueError(
                        f"Sequence target {original_color} absent from native prompt {prompt!r}"
                    )

                centers = _color_centers(episode["timestep_0/obs/front_rgb"][()])
                anchors = (
                    self.anchor_yx[row, self.anchor_mask[row]] + 1.0
                ) * 127.5
                color_regions = {
                    color_id: int(
                        np.argmin(np.linalg.norm(anchors - center, axis=-1))
                    )
                    for color_id, center in centers.items()
                }
                if len(set(color_regions.values())) != len(color_regions):
                    raise ValueError(
                        f"Non-unique native binding for episode {episode_index}: "
                        f"{color_regions}"
                    )

                length = int(self.fixed["step_mask"][row].sum())
                old_target = int(
                    self.table_targets[row, length, original_color - 1]
                )
                new_target = color_regions[original_color] + 1
                self.native_binding_audit["corrected_original_targets"] += int(
                    old_target != new_target
                )

                padded = selected_colors + [0] * (2 - len(selected_colors))
                self.fixed["goal_color_ids"][row] = np.asarray(
                    padded, dtype=np.int32
                )
                self.table_targets[row, :, :3] = 0
                self.table_mask[row, :, :3] = False
                for color_id in selected_colors:
                    field = color_id - 1
                    self.table_mask[row, :, field] = True
                    self.table_targets[row, 1 : length + 1, field] = (
                        color_regions[color_id] + 1
                    )

                # Chunk zero observes all visible target colors.  It previously
                # carried two supervised hold slots, so the native writes do
                # not overwrite another labeled operation.
                if np.any(self.event_type[row, 0] != 0):
                    raise ValueError(
                        f"Expected empty initial event slots for {self.split}:{row}"
                    )
                self.event_type[row, 0] = 0
                self.write_entity[row, 0] = 0
                self.write_region[row, 0] = 0
                self.write_mask[row, 0] = False
                self.swap_pair[row, 0] = 0
                self.swap_mask[row, 0] = False
                for micro, color_id in enumerate(selected_colors):
                    self.event_type[row, 0, micro] = 1
                    self.write_entity[row, 0, micro] = color_id - 1
                    self.write_region[row, 0, micro] = color_regions[color_id]
                    self.write_mask[row, 0, micro] = True

                # Correct all later payloads into the same runtime anchor
                # vocabulary, including non-query visual events.
                for chunk, micro in np.argwhere(self.write_mask[row]):
                    entity = int(self.write_entity[row, chunk, micro]) + 1
                    self.write_region[row, chunk, micro] = color_regions[entity]

                changes = np.any(
                    self.table_targets[row, 1 : length + 1]
                    != self.table_targets[row, :length],
                    axis=-1,
                )
                self.fixed["state_change_mask"][row, :length] = changes
                self.native_binding_audit["episodes"] += 1
                self.native_binding_audit["dual_goal_episodes"] += int(
                    len(selected_colors) == 2
                )


def _inputs(batch: dict[str, Any], teacher_force_mask: np.ndarray | jax.Array):
    result = parser_base._model_inputs(batch)  # noqa: SLF001
    result.pop("previous_tables")
    result["teacher_previous_tables"] = jnp.asarray(batch["table_targets"][:, :-1])
    result["teacher_force_mask"] = jnp.asarray(teacher_force_mask)
    return result


def _model(args: argparse.Namespace, data: parser_base.FixedParserDataset):
    encoder, deterministic, causal = VARIANTS[args.variant]
    return ExplicitEventBottleneckMemory(
        max_steps=data.max_parser_steps,
        spatial_tokens=data.spatial_tokens,
        input_width=data.patch_width,
        temporal_encoder=encoder,
        temporal_depth=args.temporal_depth,
        temporal_heads=args.temporal_heads,
        deterministic_updater=deterministic,
        causal_evidence_state=causal,
        gate_temperature=args.gate_temperature,
    )


def _smoothed_table_losses(output: dict[str, jax.Array], batch: dict[str, Any]):
    """State losses that retain ST gradients for categorical forward tables."""

    probabilities = output["all_tables"][:, 1:].astype(jnp.float32)
    probabilities = probabilities * 0.999 + 0.001 / probabilities.shape[-1]
    targets = jnp.asarray(batch["table_targets"][:, 1:], dtype=jnp.int32)
    previous = jnp.asarray(batch["table_targets"][:, :-1], dtype=jnp.int32)
    field_mask = jnp.asarray(batch["table_mask"][:, 1:], dtype=jnp.float32)
    valid = jnp.asarray(batch["sequence_mask"], dtype=jnp.float32)
    transition_mask = jnp.asarray(batch["state_change_mask"], dtype=jnp.float32) * valid
    hold_mask = (1.0 - jnp.asarray(batch["state_change_mask"], dtype=jnp.float32)) * valid
    ce = -jnp.log(
        jnp.take_along_axis(probabilities, targets[..., None], axis=-1)[..., 0]
    )
    state_ce = jnp.sum(ce * field_mask, axis=-1) / jnp.maximum(
        jnp.sum(field_mask, axis=-1), 1.0
    )

    def mean_on(values, mask):
        return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    changed_fields = (targets != previous).astype(jnp.float32) * field_mask
    lengths = jnp.sum(valid, axis=1).astype(jnp.int32) - 1
    return {
        "transition_loss": mean_on(state_ce, transition_mask),
        "no_change_loss": mean_on(state_ce, hold_mask),
        "delta_loss": jnp.sum(ce * changed_fields)
        / jnp.maximum(jnp.sum(changed_fields), 1.0),
        "final_loss": jnp.mean(state_ce[jnp.arange(state_ce.shape[0]), lengths]),
        "trajectory_loss": mean_on(state_ce, valid),
    }


def _query_losses(output: dict[str, jax.Array], batch: dict[str, Any]):
    """Orthogonal query supervision without adding task-specific model heads."""

    probabilities = output["all_tables"].astype(jnp.float32)
    probabilities = probabilities * 0.999 + 0.001 / probabilities.shape[-1]
    valid = jnp.asarray(batch["sequence_mask"], dtype=jnp.int32)
    lengths = jnp.sum(valid, axis=1)
    rows = jnp.arange(probabilities.shape[0])
    final_probabilities = probabilities[rows, lengths]
    final_targets = jnp.asarray(batch["table_targets"], dtype=jnp.int32)[
        rows, lengths
    ]
    task_ids = jnp.asarray(batch["task_ids"], dtype=jnp.int32)

    # PlaceOrder stores first..fourth demonstrated placements in table fields
    # 3..6.  Only the ordinal named by the shared goal representation is read.
    ordinals = jnp.asarray(batch["queried_ordinals"], dtype=jnp.int32)
    ordinal_fields = jnp.clip(ordinals + 2, 0, 6)
    ordinal_targets = jnp.take_along_axis(
        final_targets, ordinal_fields[:, None], axis=1
    )[:, 0]
    ordinal_probabilities = jnp.take_along_axis(
        final_probabilities,
        ordinal_fields[:, None, None].repeat(final_probabilities.shape[-1], axis=2),
        axis=1,
    )[:, 0]
    ordinal_ce = -jnp.log(
        jnp.take_along_axis(
            ordinal_probabilities, ordinal_targets[:, None], axis=-1
        )[:, 0]
    )
    ordinal_mask = (task_ids == 2) & (ordinals > 0)
    ordinal_binding_loss = jnp.sum(ordinal_ce * ordinal_mask) / jnp.maximum(
        jnp.sum(ordinal_mask), 1
    )

    # Completeness is deliberately not another region-class CE.  It only asks
    # that both requested colors have left the `none` class, keeping it
    # orthogonal to correctness and inactive for single-target episodes.
    colors = jnp.asarray(batch["goal_color_ids"], dtype=jnp.int32)
    color_fields = jnp.clip(colors - 1, 0, 2)
    color_probabilities = jnp.take_along_axis(
        final_probabilities,
        color_fields[:, :, None].repeat(final_probabilities.shape[-1], axis=2),
        axis=1,
    )
    known_probability = jnp.clip(1.0 - color_probabilities[..., 0], 1e-6, 1.0)
    multi_target = (task_ids < 2) & (colors[:, 1] > 0)
    color_mask = multi_target[:, None] & (colors > 0)
    completeness_loss = jnp.sum(-jnp.log(known_probability) * color_mask) / jnp.maximum(
        jnp.sum(color_mask), 1
    )
    return {
        "ordinal_binding_loss": ordinal_binding_loss,
        "completeness_loss": completeness_loss,
    }


def _terminal_answer_loss(output: dict[str, jax.Array], batch: dict[str, Any]):
    """CE on only the task answer at the final valid recurrent step."""

    probabilities = output["all_tables"].astype(jnp.float32)
    probabilities = probabilities * 0.999 + 0.001 / probabilities.shape[-1]
    valid = jnp.asarray(batch["sequence_mask"], dtype=jnp.int32)
    lengths = jnp.sum(valid, axis=1)
    rows = jnp.arange(probabilities.shape[0])
    final_probabilities = probabilities[rows, lengths]
    final_targets = jnp.asarray(batch["table_targets"], dtype=jnp.int32)[rows, lengths]
    task_ids = jnp.asarray(batch["task_ids"], dtype=jnp.int32)

    # Fields 0..2 are red/green/blue.  Supervise only colors named by the prompt.
    colors = jnp.asarray(batch["goal_color_ids"], dtype=jnp.int32)
    color_fields = jnp.clip(colors - 1, 0, 2)
    color_mask = (task_ids[:, None] < 2) & (colors > 0)
    selected = jnp.sum(
        jax.nn.one_hot(color_fields, 7) * color_mask[..., None], axis=1
    )

    # PlaceOrder fields 3..6 store first..fourth demonstrated placements.
    ordinals = jnp.asarray(batch["queried_ordinals"], dtype=jnp.int32)
    ordinal_fields = jnp.clip(ordinals + 2, 3, 6)
    ordinal_mask = (task_ids == 2) & (ordinals > 0)
    selected = selected + jax.nn.one_hot(ordinal_fields, 7) * ordinal_mask[:, None]
    selected = jnp.clip(selected, 0.0, 1.0)

    ce = -jnp.log(
        jnp.take_along_axis(final_probabilities, final_targets[..., None], axis=-1)[
            ..., 0
        ]
    )
    loss = jnp.sum(ce * selected) / jnp.maximum(jnp.sum(selected), 1.0)
    prediction = jnp.argmax(final_probabilities, axis=-1)
    exact = jnp.all((prediction == final_targets) | ~selected.astype(jnp.bool_), axis=-1)
    return loss, jnp.mean(exact.astype(jnp.float32))


def _summary(output: dict[str, np.ndarray], batch: dict[str, np.ndarray]):
    result = transition._summary(output, batch)  # noqa: SLF001
    result["routing"] = parser_base._metrics(output, batch)  # noqa: SLF001
    tables = np.argmax(output["all_tables"], axis=-1)
    for task_id, task in enumerate(base.TASKS):
        query_values = []
        complete_values = []
        for row in np.flatnonzero(batch["task_ids"] == task_id):
            length = int(batch["sequence_mask"][row].sum())
            if task_id < 2:
                fields = [
                    int(color) - 1
                    for color in batch["goal_color_ids"][row]
                    if int(color) > 0
                ]
            else:
                fields = [3 + int(batch["queried_ordinals"][row]) - 1]
            known = tables[row, length, fields] != 0
            query_values.extend(known.tolist())
            complete_values.append(bool(known.all()))
        result[task]["final_query_missing_rate"] = float(
            1.0 - np.mean(query_values)
        )
        result[task]["final_episode_complete_accuracy"] = float(
            np.mean(complete_values)
        )
    return result


def _score(summary: dict[str, Any]) -> tuple[float, ...]:
    overall = summary["overall"]
    routing = summary["routing"]
    balance = min(
        overall["transition_state_exact_accuracy"],
        overall["no_change_state_exact_accuracy"],
        overall["mean_task_final_query_accuracy"],
    )
    return (
        balance,
        overall["mean_task_final_query_accuracy"],
        overall["transition_state_exact_accuracy"],
        overall["no_change_state_exact_accuracy"],
        routing["full_update_recall"],
    )


def main() -> None:
    args = parse_args()
    if args.batch_size % 3 or args.eval_batch_size % 3:
        raise ValueError("train and eval batch sizes must be divisible by three")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: NativeUnmaskBindingDataset(split, args)
        for split in ("train", "dev", "test")
    }
    try:
        train_data = datasets["train"]
        model = _model(args, train_data)
        rng = np.random.default_rng(args.seed)
        initial = train_data.parser_batch(train_data.sample(rng, args.batch_size))
        force_all = initial["sequence_mask"].copy()
        if args.supervision_mode == "terminal_only":
            force_all[:] = False
        params = model.init(jax.random.key(args.seed), **_inputs(initial, force_all))["params"]
        parameter_count = sum(value.size for value in jax.tree_util.tree_leaves(params))
        schedule = optax.warmup_cosine_decay_schedule(
            0.0,
            args.learning_rate,
            min(args.warmup_steps, max(args.steps - 1, 1)),
            args.steps,
            end_value=args.end_learning_rate,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=args.weight_decay),
        )
        opt_state = optimizer.init(params)
        type_weights = np.sqrt(train_data.event_type_weights)
        type_weights = jnp.asarray(type_weights / type_weights.mean())

        def objective(current_params, batch, teacher_force_mask, recurrent_weight):
            output = model.apply(
                {"params": current_params}, **_inputs(batch, teacher_force_mask)
            )
            terminal_loss, terminal_accuracy = _terminal_answer_loss(output, batch)
            if args.supervision_mode == "terminal_only":
                return terminal_loss, {
                    "loss": terminal_loss,
                    "terminal_answer_loss": terminal_loss,
                    "terminal_answer_exact_accuracy": terminal_accuracy,
                }
            event = base._masked_ce(  # noqa: SLF001
                output["event_type_logits"],
                jnp.asarray(batch["event_type"]),
                jnp.asarray(batch["micro_mask"]),
                type_weights,
            )
            entity = base._masked_ce(  # noqa: SLF001
                output["write_entity_logits"],
                jnp.asarray(batch["write_entity"]),
                jnp.asarray(batch["write_mask"]),
            )
            region = base._masked_ce(  # noqa: SLF001
                output["write_region_logits"],
                jnp.asarray(batch["write_region"]),
                jnp.asarray(batch["write_mask"]),
            )
            pair = base._masked_ce(  # noqa: SLF001
                output["swap_pair_logits"],
                jnp.asarray(batch["swap_pair"]),
                jnp.asarray(batch["swap_mask"]),
            )
            table = _smoothed_table_losses(output, batch)
            query = _query_losses(output, batch)
            operation_loss = (
                args.event_type_weight * event
                + args.write_entity_weight * entity
                + args.write_region_weight * region
                + args.swap_pair_weight * pair
            )
            loss = operation_loss + recurrent_weight * (
                args.transition_weight * table["transition_loss"]
                + args.no_change_weight * table["no_change_loss"]
                + args.delta_weight * table["delta_loss"]
                + args.final_weight * table["final_loss"]
                + args.trajectory_weight * table["trajectory_loss"]
                + args.ordinal_binding_weight * query["ordinal_binding_loss"]
                + args.completeness_weight * query["completeness_loss"]
            )
            return loss, {
                "loss": loss,
                "operation_loss": operation_loss,
                "event_type_loss": event,
                "write_entity_loss": entity,
                "write_region_loss": region,
                "swap_pair_loss": pair,
                "terminal_answer_loss": terminal_loss,
                "terminal_answer_exact_accuracy": terminal_accuracy,
                **table,
                **query,
            }

        @jax.jit
        def train_step(current_params, current_opt, batch, teacher_force_mask, recurrent_weight):
            (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(
                current_params, batch, teacher_force_mask, recurrent_weight
            )
            updates, next_opt = optimizer.update(grads, current_opt, current_params)
            return optax.apply_updates(current_params, updates), next_opt, metrics

        @jax.jit
        def infer(current_params, batch, teacher_force_mask):
            return model.apply(
                {"params": current_params}, **_inputs(batch, teacher_force_mask)
            )

        def infer_split(current_params, split: str, *, teacher_forcing: bool):
            outputs: dict[str, list[np.ndarray]] = defaultdict(list)
            batches = []
            data = datasets[split]
            for start in range(0, len(data.rows), args.eval_batch_size):
                indices = data.rows[start : start + args.eval_batch_size]
                valid_count = len(indices)
                if valid_count < args.eval_batch_size:
                    indices = np.pad(
                        indices, (0, args.eval_batch_size - valid_count), mode="edge"
                    )
                batch = data.parser_batch(indices)
                force = batch["sequence_mask"] & bool(teacher_forcing)
                output = jax.device_get(infer(current_params, batch, force))
                for key, value in output.items():
                    if key != "all_memories":
                        outputs[key].append(np.asarray(value)[:valid_count])
                batches.append(
                    {key: np.asarray(value)[:valid_count] for key, value in batch.items()}
                )
            merged_output = {key: np.concatenate(values) for key, values in outputs.items()}
            merged_batch = {
                key: np.concatenate([batch[key] for batch in batches])
                for key in batches[0]
            }
            return merged_output, merged_batch

        def evaluate(current_params, split: str, *, teacher_forcing: bool):
            output, batch = infer_split(
                current_params, split, teacher_forcing=teacher_forcing
            )
            return _summary(output, batch)

        best_params = params
        best_step = 0
        best_score = (-1.0,) * 5
        history = []
        started = time.monotonic()
        for step in range(1, args.steps + 1):
            batch = train_data.parser_batch(train_data.sample(rng, args.batch_size))
            if args.supervision_mode == "terminal_only":
                ratio = 0.0
                recurrent_weight = 1.0
            elif step <= args.operation_pretrain_steps:
                ratio = 1.0
                recurrent_weight = 0.0
            else:
                recurrent_step = step - args.operation_pretrain_steps
                recurrent_steps = max(args.steps - args.operation_pretrain_steps, 1)
                ratio = transition._curriculum_ratio(  # noqa: SLF001
                    recurrent_step, recurrent_steps
                )
                recurrent_weight = min(
                    1.0, recurrent_step / max(0.2 * recurrent_steps, 1.0)
                )
            teacher_force_mask = batch["sequence_mask"] & (
                rng.random(batch["sequence_mask"].shape) < ratio
            )
            params, opt_state, train_metrics = train_step(
                params,
                opt_state,
                batch,
                teacher_force_mask,
                jnp.asarray(recurrent_weight, dtype=jnp.float32),
            )
            if step % args.eval_every == 0 or step == args.steps:
                dev = evaluate(params, "dev", teacher_forcing=False)
                if args.supervision_mode == "terminal_only":
                    overall = dev["overall"]
                    score = (
                        overall["mean_task_final_query_accuracy"],
                        overall["min_task_final_query_accuracy"],
                    )
                    eligible = True
                else:
                    score = _score(dev)
                    eligible = step >= args.operation_pretrain_steps
                if eligible and score > best_score:
                    best_score = score
                    best_step = step
                    best_params = jax.device_get(params)
                row = {
                    "step": step,
                    "teacher_force_ratio": ratio,
                    "recurrent_weight": recurrent_weight,
                    "selection_score": score,
                    "train_batch": {
                        key: float(value) for key, value in train_metrics.items()
                    },
                    "dev_free_rollout": dev,
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        metrics = {}
        for split in ("train", "dev", "test"):
            teacher_output, batch = infer_split(
                best_params, split, teacher_forcing=True
            )
            free_output, _ = infer_split(best_params, split, teacher_forcing=False)
            hard_replay = replay._replay(  # noqa: SLF001
                teacher_output, batch, oracle_event=False, oracle_payload=False
            )
            metrics[split] = {
                "teacher_forced": _summary(teacher_output, batch),
                "teacher_forced_hard_replay": _summary(hard_replay, batch),
                "free_rollout": _summary(free_output, batch),
            }
        result = {
            "schema_version": 1,
            "experiment": "robomme_explicit_event_bottleneck_factorial_ablation",
            "variant": args.variant,
            "components": {
                "temporal_encoder": VARIANTS[args.variant][0],
                "deterministic_updater": VARIANTS[args.variant][1],
                "causal_evidence_state": VARIANTS[args.variant][2],
                "ordinal_binding_weight": args.ordinal_binding_weight,
                "completeness_weight": args.completeness_weight,
                "supervision_mode": args.supervision_mode,
                "privileged_trajectory_teacher_used": args.supervision_mode == "full",
                "unmask_binding_labels": args.unmask_binding_labels,
                "unmask_native_binding_audit": {
                    split: data.native_binding_audit
                    for split, data in datasets.items()
                },
            },
            "parameter_count": parameter_count,
            "best_step": best_step,
            "best_score": best_score,
            "elapsed_seconds": time.monotonic() - started,
            "metrics": metrics,
            "history": history,
        }
        (args.output_dir / "params.msgpack").write_bytes(
            flax.serialization.to_bytes(best_params)
        )
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output_dir / "training_config.json").write_text(
            json.dumps(
                {
                    **{
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "jax_devices": [str(device) for device in jax.devices()],
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(metrics, indent=2), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
