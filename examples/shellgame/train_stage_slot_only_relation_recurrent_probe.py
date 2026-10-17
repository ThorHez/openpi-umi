"""Train visual swap evidence using only recurrent stage-slot supervision.

This is the strict ablation for asking whether explicit swap-pair labels are
necessary.  The already validated M128 recurrent updater/readout is frozen and
anchors the meanings of its three relation-code channels.  The complete visual
relation classifier is randomly initialized and emits differentiable softmax
probabilities; no ground-truth relation is fed to memory and no relation
cross-entropy is present.  Ground-truth initial cup and the ball slot after
each of the three swaps provide the only task loss.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.shellgame import eval_three_swap_fully_visual_relation_memory_probe as _fully
from openpi.models import model as _model
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import optimizer as _optimizer
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as _recipe
from scripts.mem import train_semantic_memory as _trainer

DEFAULT_VISUAL_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_pair_fixed_grid_probe_260809/"
    "swap_pair_full600_b18_260809/599/params"
)
DEFAULT_INITIAL_CHECKPOINT = _fully.DEFAULT_INITIAL_CHECKPOINT
DEFAULT_MEMORY_CHECKPOINT = _fully.DEFAULT_MEMORY_CHECKPOINT


@dataclasses.dataclass(frozen=True)
class MemoryAnchoredRandomRelationLoader:
    """Restore base/initial/recurrent weights but randomize visual relations."""

    visual_params_path: str
    initial_params_path: str
    memory_params_path: str

    def load(self, params):
        visual = flax.traverse_util.flatten_dict(
            _model.restore_params(self.visual_params_path, restore_type=np.ndarray),
            sep="/",
        )
        initial = flax.traverse_util.flatten_dict(
            _model.restore_params(self.initial_params_path, restore_type=np.ndarray),
            sep="/",
        )
        memory = flax.traverse_util.flatten_dict(
            _model.restore_params(self.memory_params_path, restore_type=np.ndarray),
            sep="/",
        )
        target = flax.traverse_util.flatten_dict(params, sep="/")

        initial_root = "HistoryFrame0InitialCupClassifier/"
        tracker_root = "HistoryThreeSwapVisualRelationMemoryTracker/"
        relation_root = tracker_root + "swap_relation_classifier/"
        memory_root = "HistoryThreeSwapOraclePairRecurrentMemoryTracker/"
        initial_name_map = {
            "initial_ln/bias": "HistoryClassifierNorm/bias",
            "initial_ln/scale": "HistoryClassifierNorm/scale",
            "initial_head/bias": "HistoryClassifierHead/bias",
            "initial_head/kernel": "HistoryClassifierHead/kernel",
        }

        result = {}
        counts = {
            "base": 0,
            "initial": 0,
            "memory": 0,
            "random_relation": 0,
            "initialized_unused": 0,
        }
        missing_memory = []
        for key, reference in target.items():
            candidate = None
            kind = "base"
            if key.startswith(relation_root):
                result[key] = reference
                counts["random_relation"] += 1
                continue
            if key.startswith(initial_root):
                relative = key.removeprefix(initial_root)
                source_key = initial_name_map.get(relative)
                candidate = initial.get(source_key) if source_key else None
                kind = "initial"
            elif key.startswith(tracker_root):
                relative = key.removeprefix(tracker_root)
                if relative.startswith("shared_swap_memory_updater/"):
                    relative = "shared_segment_memory_updater/" + relative.removeprefix("shared_swap_memory_updater/")
                candidate = memory.get(memory_root + relative)
                kind = "memory"
            else:
                candidate = visual.get(key)

            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                counts[kind] += 1
            else:
                result[key] = reference
                if kind == "memory":
                    missing_memory.append(key)
                else:
                    counts["initialized_unused"] += 1

        if missing_memory:
            raise ValueError(f"Frozen recurrent restore incomplete: {missing_memory[:8]}")
        print(
            "MemoryAnchoredRandomRelationLoader: "
            + ", ".join(f"{key}={value}" for key, value in counts.items())
            + ", missing_memory=0"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def stage_slot_only_objective(
    config,
    model,
    rng,
    observation,
    label_table,
    *,
    train: bool,
):
    """Use GT initial/stage slots, but never expose GT relation identities."""
    if observation.episode_index is None:
        raise ValueError("Stage-only training requires episode_index")
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode_index]
    initial_labels = labels[:, 0]
    relation_labels = labels[:, 1 : 1 + _recipe.NUM_STAGES]
    stage_labels = labels[:, 1 + _recipe.NUM_STAGES :]

    processed = _model.preprocess_observation(
        rng,
        observation,
        train=train and config.memory_train_augmentation,
    )
    outputs = model._track_history(  # noqa: SLF001 - intentional probe seam
        processed,
        initial_slots_override=initial_labels,
        relation_ids_override=None,
    )
    stage_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            outputs["stage_logits"].astype(jnp.float32),
            stage_labels,
        )
    )
    relation_probabilities = jax.nn.softmax(
        outputs["relation_logits"].astype(jnp.float32),
        axis=-1,
    )
    relation_entropy = -jnp.mean(
        jnp.sum(
            relation_probabilities * jnp.log(jnp.maximum(relation_probabilities, 1e-8)),
            axis=-1,
        )
    )
    metrics = {
        "loss": stage_loss,
        "stage_memory_loss": stage_loss,
        "stage_memory_accuracy": jnp.mean(jnp.argmax(outputs["stage_logits"], axis=-1) == stage_labels),
        "final_memory_accuracy": jnp.mean(jnp.argmax(outputs["stage_logits"][:, -1], axis=-1) == stage_labels[:, -1]),
        "relation_accuracy_diagnostic": jnp.mean(outputs["relation_ids"] == relation_labels),
        "relation_sequence_accuracy_diagnostic": jnp.mean(jnp.all(outputs["relation_ids"] == relation_labels, axis=1)),
        "relation_entropy": relation_entropy,
    }
    for stage_index in range(_recipe.NUM_STAGES):
        metrics[f"relation_{stage_index}_accuracy_diagnostic"] = jnp.mean(
            outputs["relation_ids"][:, stage_index] == relation_labels[:, stage_index]
        )
        metrics[f"slot_{stage_index}_accuracy"] = jnp.mean(
            jnp.argmax(outputs["stage_logits"][:, stage_index], axis=-1) == stage_labels[:, stage_index]
        )
    return stage_loss, metrics


def build_config(args: argparse.Namespace):
    base = _recipe.make_train_config()
    model = dataclasses.replace(base.model, relation_mode="probabilities")
    relation = nnx_utils.PathRegex(
        r".*HistoryThreeSwapVisualRelationMemoryTracker/"
        r"swap_relation_classifier.*"
    )
    return dataclasses.replace(
        base,
        name="shellgame_stage_slot_only_relation_recurrent_probe",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=nnx.Not(relation),
        weight_loader=MemoryAnchoredRandomRelationLoader(
            visual_params_path=args.visual_checkpoint,
            initial_params_path=args.initial_checkpoint,
            memory_params_path=args.memory_checkpoint,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 1),
            decay_lr=args.peak_lr * 0.1,
        ),
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=args.save_interval,
        keep_period=args.save_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        initial_loss_weight=0.0,
        relation_loss_weight=0.0,
        stage_memory_loss_weight=1.0,
        wandb_enabled=False,
        overwrite=args.overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--visual-checkpoint", default=DEFAULT_VISUAL_CHECKPOINT)
    parser.add_argument("--initial-checkpoint", default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument("--memory-checkpoint", default=DEFAULT_MEMORY_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _recipe.compute_objective = stage_slot_only_objective
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
