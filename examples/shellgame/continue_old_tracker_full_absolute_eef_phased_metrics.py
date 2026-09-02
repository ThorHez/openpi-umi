"""Continue absolute-EEF7 action training to global step 6000.

This is a strict training-duration control: it restores the full step-1999
train state (including AdamW state), preserves the original balanced phase
sampler and frozen old tracker, and only extends the low-learning-rate tail.
Validation additionally reports action flow loss by the five sampler anchors:
selection, approach, descend, grasp, and lift.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from examples.shellgame import train_old_tracker_full_absolute_eef as full_eef
from examples.shellgame import train_old_tracker_full_joint_grasp as full_joint
from openpi.training import config_pi0_mem
from openpi.training import optimizer
from scripts.mem import train_pi0_mem_compress as trainer


CONTINUATION_PEAK_LR = 3.5e-6
CONTINUATION_END_LR = 1.0e-6
GLOBAL_END_STEP = 6_000

# These are observation-frame anchors. Every row predicts actions beginning at
# frame_index + 1. They exactly match the deterministic 20/20/20/20/20 sampler
# in train_old_tracker_full_joint_grasp._balanced_full_action_indices.
PHASE_ANCHORS = {
    "selection": (59, 59),
    "approach": (60, 88),
    "descend": (89, 108),
    "grasp": (109, 118),
    "lift": (119, 153),
}

_ORIGINAL_EVAL_STEP = trainer.eval_step


def _phased_eval_step(
    config,
    rng,
    state,
    batch,
    *,
    class_labels_by_episode=None,
):
    """Add exact sample-weighted phase sums to the normal validation output."""
    metrics = _ORIGINAL_EVAL_STEP(
        config,
        rng,
        state,
        batch,
        class_labels_by_episode=class_labels_by_episode,
    )
    observation, actions = batch
    if observation.frame_index is None:
        raise ValueError("Phased validation requires observation.frame_index")

    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    chunked_loss, _ = model.compute_loss_with_memory_aux(
        rng, observation, actions, train=False
    )
    # compute_loss_with_memory_aux scales terminal chunks so this mean is the
    # mean over real, non-padding future actions for every anchor sample.
    per_sample_loss = jnp.mean(chunked_loss, axis=-1)
    frame_index = jnp.asarray(observation.frame_index, dtype=jnp.int32).reshape(-1)
    per_sample_loss = jnp.asarray(per_sample_loss).reshape(-1)

    for phase, (start, end) in PHASE_ANCHORS.items():
        valid = (frame_index >= start) & (frame_index <= end)
        valid_f = valid.astype(jnp.float32)
        metrics[f"_val/phase_{phase}_loss_sum"] = jnp.sum(per_sample_loss * valid_f)
        metrics[f"_val/phase_{phase}_count"] = jnp.sum(valid_f)
    return metrics


def _phased_run_evaluation(
    peval_step,
    eval_rng,
    val_iter,
    config,
    mesh,
    state,
):
    """Reduce phase metrics by sample count rather than by batch count."""
    eval_infos = []
    for batch_index in range(config.eval_batches):
        batch = next(val_iter)
        batch_rng = jax.random.fold_in(eval_rng, batch_index)
        with trainer.sharding.set_mesh(mesh):
            eval_infos.append(peval_step(batch_rng, state, batch))

    stacked = jax.device_get(trainer.common_utils.stack_forest(eval_infos))
    reduced = jax.tree.map(jnp.mean, stacked)
    output = {
        key: float(value)
        for key, value in reduced.items()
        if not key.startswith("_val/")
    }
    total_suffix_sum = 0.0
    total_suffix_count = 0.0
    for phase in PHASE_ANCHORS:
        loss_sum = float(jnp.sum(stacked[f"_val/phase_{phase}_loss_sum"]))
        count = float(jnp.sum(stacked[f"_val/phase_{phase}_count"]))
        if count <= 0:
            raise RuntimeError(
                f"Validation batches contained no {phase} anchor samples; "
                "increase --eval-batches or check the balanced sampler."
            )
        output[f"val/phase_{phase}_action_loss"] = loss_sum / count
        output[f"val/phase_{phase}_samples"] = count
        if phase in {"descend", "grasp", "lift"}:
            total_suffix_sum += loss_sum
            total_suffix_count += count
    output["val/critical_suffix_action_loss"] = total_suffix_sum / total_suffix_count
    return output


def main() -> None:
    args = full_joint.parse_args()
    if args.steps != GLOBAL_END_STEP:
        raise ValueError(
            f"Continuation requires --steps {GLOBAL_END_STEP}; got {args.steps}. "
            "The restored checkpoint already carries global step 1999."
        )
    if args.save_interval != 500 or args.keep_period != 1_000:
        raise ValueError(
            "Use --save-interval 500 --keep-period 1000 so global checkpoints "
            "3000, 4000, 5000, and 5999 are retained."
        )

    config = full_eef.build_config(args)
    config = dataclasses.replace(
        config,
        resume=True,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=CONTINUATION_PEAK_LR,
            decay_steps=GLOBAL_END_STEP,
            decay_lr=CONTINUATION_END_LR,
        ),
    )
    logging.info(
        "Absolute-EEF duration-control continuation: restored global step; "
        "peak_lr=%g decay_lr=%g global_end=%d phase_metrics=%s",
        CONTINUATION_PEAK_LR,
        CONTINUATION_END_LR,
        GLOBAL_END_STEP,
        tuple(PHASE_ANCHORS),
    )

    config_pi0_mem.VideoFrameDataset = full_joint.FixedPrefixCurrentVideoDataset
    trainer._filter_memory_classifier_frame_range = (  # noqa: SLF001
        full_joint._balanced_full_action_indices  # noqa: SLF001
    )
    trainer.eval_step = _phased_eval_step
    trainer.run_evaluation = _phased_run_evaluation
    trainer.main(config)


if __name__ == "__main__":
    main()
