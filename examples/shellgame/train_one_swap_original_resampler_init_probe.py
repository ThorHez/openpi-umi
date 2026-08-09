"""Test whether the original flatten-then-cross-attention path mainly fails from initialization.

The student is the original Pi0MemCompress visual history path:

    30 x 256 patches + temporal position
      -> flatten
      -> HistoryResampler cross-attention (128 tokens)
      -> current-frame ViT blocks repeatedly read the memory
      -> attention-pooling three-way classifier

The frozen teacher is the staged fixed-grid temporal-memory probe that reached
100% one-swap validation accuracy.  The script supports a causal comparison:

* ``random_ce``: reset the complete student history path and train with CE.
* ``distill``: reset the student resampler and match the teacher memory.
* ``shuffled_distill``: identical to distill, but roll teacher targets across
  the batch as a negative control.
* ``reader_ce``: load a distilled checkpoint, freeze the resampler, and train
  only the original memory reader plus classifier.
* ``joint_ce``: load a warm checkpoint and jointly fine-tune the complete
  student history path.

All policy-independent diagnostic code lives in this file.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.shellgame.train_one_swap_fixed_grid_temporal_memory_probe import (
    FixedGridTemporalMemoryProbe,
)
from examples.shellgame.train_one_swap_history_probe import build_one_swap_labels
from examples.shellgame.train_one_swap_history_probe import DATASET_ROOT
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer


TEACHER_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_one_swap_fixed_grid_staged_joint_probe_260808/"
    "one_swap_fixed_grid_stage3_joint_260808/199/params"
)


class OriginalFullPathReadout(nn.Module):
    """Read current tokens after they have consumed original history memory."""

    input_width: int = 1152
    width: int = 256
    num_classes: int = 3
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, current_tokens):
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(current_tokens)
        x = nn.Dense(self.width, name="projection", dtype=self.dtype_mm)(x)
        x = nn.tanh(x)
        scores = nn.Dense(1, name="attention", dtype=jnp.float32)(x.astype(jnp.float32))
        weights = nn.softmax(scores, axis=1)
        pooled = jnp.sum(weights * x.astype(jnp.float32), axis=1)
        pooled = nn.LayerNorm(name="pooled_ln", dtype=jnp.float32)(pooled)
        logits = nn.Dense(self.num_classes, name="classifier", dtype=jnp.float32)(pooled)
        return logits, pooled


@dataclasses.dataclass(frozen=True)
class OriginalResamplerInitProbeConfig(_base_model.Pi0MemCompressConfig):
    distill_shuffle_targets: bool = False
    eval_history_mode: str = "normal"

    def create(self, rng: at.KeyArrayLike) -> OriginalResamplerInitProbeModel:
        return OriginalResamplerInitProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_for_phase(self, phase: str) -> nnx.filterlib.Filter:
        resampler = nnx_utils.PathRegex(
            r".*PaliGemma/img/Transformer/HistoryResampler_0.*"
        )
        reader = nnx_utils.PathRegex(
            r".*PaliGemma/img/Transformer/encoderblock/"
            r"(HistoryLayerNorm_0|HistoryMultiHeadDotProductAttention_0).*"
        )
        readout = nnx_utils.PathRegex(r".*HistoryOriginalFullPathReadout.*")
        if phase in ("distill", "shuffled_distill"):
            trainable = resampler
        elif phase == "reader_ce":
            trainable = nnx.Any(reader, readout)
        elif phase in ("random_ce", "joint_ce"):
            trainable = nnx.Any(resampler, reader, readout)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return nnx.Not(trainable)


class OriginalResamplerInitProbeModel(_base_model.Pi0MemCompress):
    def __init__(self, config: OriginalResamplerInitProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.HistoryFixedGridTemporalMemory = nnx_bridge.ToNNX(
            FixedGridTemporalMemoryProbe(
                num_frames=30,
                input_grid_size=16,
                pool_factor=2,
                input_width=1152,
                width=256,
                temporal_depth=2,
                num_heads=8,
                memory_tokens=128,
                output_width=1152,
                num_classes=3,
                dtype_mm=config.dtype,
            )
        )
        self.HistoryOriginalFullPathReadout = nnx_bridge.ToNNX(
            OriginalFullPathReadout(
                input_width=1152,
                width=256,
                num_classes=config.history_classifier_num_classes,
                dtype_mm=config.dtype,
            )
        )
        fake_history = jnp.zeros((1, 30, 256, 1152), dtype=jnp.bfloat16)
        fake_current = jnp.zeros((1, 256, 1152), dtype=jnp.bfloat16)
        self.HistoryFixedGridTemporalMemory.lazy_init(fake_history, train=False, rngs=rngs)
        self.HistoryOriginalFullPathReadout.lazy_init(fake_current, rngs=rngs)

    def compute_history_classification(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ):
        observation = _model.preprocess_observation(rng, observation, train=train)
        image = observation.images["base_rgb"]
        if image.ndim == 4:
            image = image[:, None, ...]
        if image.ndim != 5 or image.shape[1] != 31:
            raise ValueError(f"Original-path probe expects [B,31,H,W,C], got {image.shape}")

        _, encoder_out = self.PaliGemma.img(image, train=False)
        student_memory = encoder_out["encoder"]["history_mem"]
        current_tokens = encoder_out["encoder"]["pre_ln"]
        logits, pooled = self.HistoryOriginalFullPathReadout(current_tokens)

        history_patches = encoder_out["with_posemb"][:, :30]
        teacher_logits, teacher_memory = self.HistoryFixedGridTemporalMemory(
            history_patches,
            train=False,
        )
        teacher_memory = jax.lax.stop_gradient(teacher_memory)
        teacher_logits = jax.lax.stop_gradient(teacher_logits)
        return logits, {
            "history_mem": student_memory,
            "encoder_auxes": (encoder_out["encoder"],),
            "student_memory": student_memory,
            "teacher_memory": teacher_memory,
            "teacher_logits": teacher_logits,
            "student_pooled": pooled,
        }


@dataclasses.dataclass(frozen=True)
class ProbeCheckpointLoader:
    """Load the teacher/base while optionally resetting only the student path."""

    params_path: str
    reset_student: bool

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")
        result = dict(flat_ref)
        reset_pattern = re.compile(
            r"(?:PaliGemma/img/Transformer/HistoryResampler_0|"
            r"PaliGemma/img/Transformer/encoderblock/"
            r"(?:HistoryLayerNorm_0|HistoryMultiHeadDotProductAttention_0)|"
            r"HistoryOriginalFullPathReadout)"
        )
        loaded_count = 0
        reset_count = 0
        teacher_count = 0
        teacher_expected = 0
        for key, reference in flat_ref.items():
            if key.startswith("HistoryFixedGridTemporalMemory/"):
                teacher_expected += 1
            if self.reset_student and reset_pattern.search(key):
                reset_count += 1
                continue
            candidate = flat_loaded.get(key)
            if candidate is None or np.shape(candidate) != np.shape(reference):
                continue
            result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
            loaded_count += 1
            if key.startswith("HistoryFixedGridTemporalMemory/"):
                teacher_count += 1

        if teacher_count != teacher_expected:
            raise ValueError(
                f"Teacher checkpoint mapping incomplete: {teacher_count}/{teacher_expected}"
            )
        if not self.reset_student:
            missing = [
                key
                for key, value in flat_ref.items()
                if key not in flat_loaded or np.shape(flat_loaded[key]) != np.shape(value)
            ]
            if missing:
                raise ValueError(f"Warm checkpoint is missing {len(missing)} parameters: {missing[:5]}")
        print(
            f"ProbeCheckpointLoader: loaded={loaded_count}, reset_student={reset_count}, "
            f"teacher={teacher_count}/{teacher_expected}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def _label_metrics(logits, observation, labels_by_episode):
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    safe_index = jnp.clip(episode_index, 0, labels_by_episode.shape[0] - 1)
    labels = labels_by_episode[safe_index]
    valid = (episode_index >= 0) & (episode_index < labels_by_episode.shape[0]) & (labels >= 0)
    valid_f = valid.astype(jnp.float32)
    count = jnp.sum(valid_f)
    safe_labels = jnp.maximum(labels, 0)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, safe_labels)
    loss_sum = jnp.sum(losses * valid_f)
    correct = jnp.sum((jnp.argmax(logits, axis=-1) == safe_labels) * valid_f)
    return loss_sum / jnp.maximum(count, 1.0), correct / jnp.maximum(count, 1.0), count


def _memory_distillation_loss(student_memory, teacher_memory, *, shuffled: bool):
    teacher_memory = jnp.asarray(teacher_memory, dtype=jnp.float32)
    student_memory = jnp.asarray(student_memory, dtype=jnp.float32)
    if shuffled:
        teacher_memory = jnp.roll(teacher_memory, shift=1, axis=0)
    student_unit = student_memory / jnp.maximum(
        jnp.linalg.norm(student_memory, axis=-1, keepdims=True), 1e-6
    )
    teacher_unit = teacher_memory / jnp.maximum(
        jnp.linalg.norm(teacher_memory, axis=-1, keepdims=True), 1e-6
    )
    cosine_loss = jnp.mean(1.0 - jnp.sum(student_unit * teacher_unit, axis=-1))
    mse_loss = jnp.mean(jnp.square(student_memory - teacher_memory))
    return cosine_loss + 0.1 * mse_loss, cosine_loss, mse_loss


def distill_train_step(
    config,
    rng,
    state,
    batch,
    *,
    class_labels_by_episode=None,
):
    del class_labels_by_episode
    model = nnx.merge(state.model_def, state.params)
    model.train()
    shuffled = bool(getattr(config.model, "distill_shuffle_targets", False))

    def loss_fn(model, step_rng, observation, actions):
        del actions
        logits, aux = model.compute_history_classification(step_rng, observation, train=False)
        loss, cosine_loss, mse_loss = _memory_distillation_loss(
            aux["student_memory"], aux["teacher_memory"], shuffled=shuffled
        )
        return loss, (logits, aux, cosine_loss, mse_loss)

    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, (logits, aux, cosine_loss, mse_loss)), grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state, has_aux=True
    )(model, jax.random.fold_in(rng, state.step), observation, actions)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    new_params = nnx.state(model)
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    teacher_agreement = jnp.mean(
        jnp.argmax(logits, axis=-1) == jnp.argmax(aux["teacher_logits"], axis=-1)
    )
    return new_state, {
        "loss": loss,
        "action_loss": jnp.asarray(0.0),
        "normalized_action_loss": jnp.asarray(0.0),
        "diversity_loss": loss,
        "diversity_weight": jnp.asarray(1.0),
        "history_classifier_loss": jnp.asarray(0.0),
        "history_classifier_accuracy": teacher_agreement,
        "history_classifier_valid_fraction": jnp.asarray(1.0),
        "history_classifier_weight": jnp.asarray(0.0),
        "distill_cosine_loss": cosine_loss,
        "distill_mse_loss": mse_loss,
        "grad_norm": optax.global_norm(grads),
    }


def distill_eval_step(
    config,
    rng,
    state,
    batch,
    *,
    class_labels_by_episode=None,
):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, _ = batch
    logits, aux = model.compute_history_classification(rng, observation, train=False)
    distill_loss, cosine_loss, mse_loss = _memory_distillation_loss(
        aux["student_memory"],
        aux["teacher_memory"],
        shuffled=bool(getattr(config.model, "distill_shuffle_targets", False)),
    )
    classifier_loss, accuracy, count = _label_metrics(
        logits, observation, class_labels_by_episode
    )
    teacher_loss, teacher_accuracy, _ = _label_metrics(
        aux["teacher_logits"], observation, class_labels_by_episode
    )
    del teacher_loss
    return {
        "val/loss": distill_loss,
        "val/action_loss": jnp.asarray(0.0),
        "val/diversity_loss": distill_loss,
        "val/history_classifier_loss": classifier_loss,
        "val/history_classifier_accuracy": accuracy,
        "val/history_classifier_valid_fraction": jnp.asarray(1.0),
        "val/distill_cosine_loss": cosine_loss,
        "val/distill_mse_loss": mse_loss,
        "val/teacher_accuracy": teacher_accuracy,
        "_val/history_classifier_loss_sum": classifier_loss * count,
        "_val/history_classifier_correct_count": accuracy * count,
        "_val/history_classifier_valid_count": count,
    }


def _ablate_eval_history(observation, mode: str):
    """Change only frames 0:30; frame 30 and all non-image inputs stay fixed."""
    if mode == "normal":
        return observation
    images = dict(observation.images)
    clip = images["base_rgb"]
    history, current = clip[:, :30], clip[:, 30:]
    if mode == "shuffle_batch":
        history = jnp.roll(history, shift=1, axis=0)
    elif mode == "reverse_time":
        history = history[:, ::-1]
    elif mode == "zero_history":
        history = jnp.zeros_like(history)
    else:
        raise ValueError(f"Unknown eval history mode: {mode}")
    images["base_rgb"] = jnp.concatenate((history, current), axis=1)
    return observation.replace(images=images)


_BASE_EVAL_STEP = _trainer.eval_step


def ablation_eval_step(
    config,
    rng,
    state,
    batch,
    *,
    class_labels_by_episode=None,
):
    observation, actions = batch
    observation = _ablate_eval_history(observation, config.model.eval_history_mode)
    return _BASE_EVAL_STEP(
        config,
        rng,
        state,
        (observation, actions),
        class_labels_by_episode=class_labels_by_episode,
    )


def build_config(args, labels_path):
    is_distill = args.phase in ("distill", "shuffled_distill")
    model = OriginalResamplerInitProbeConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        num_frames=31,
        memory_every=1,
        current_frame_index=-1,
        history_memory_tokens=128,
        history_resampler_depth=1,
        history_use_current_condition=True,
        history_gate_fixed=1.0,
        diversity_weight=1.0 if is_distill else 0.0,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=3,
        distill_shuffle_targets=args.phase == "shuffled_distill",
        eval_history_mode=args.eval_history_mode,
    )
    return _config.TrainConfig(
        name=f"pi0_shellgame_original_resampler_init_{args.phase}_260808",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_for_phase(args.phase),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=str(DATASET_ROOT),
                    assets=_config.AssetsConfig(
                        asset_id=".", assets_dir=str(DATASET_ROOT)
                    ),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=31,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=ProbeCheckpointLoader(
            args.init_checkpoint,
            reset_student=args.reset_student,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 2),
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=10.0),
        ema_decay=None,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=max(args.steps, 1),
        keep_period=max(args.steps, 1),
        val_ratio=0.1,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        wandb_enabled=False,
        overwrite=args.overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(
            enabled=True,
            episodes_metadata_path=str(labels_path),
            label_key="after_first_swap_ball_cup",
            classes=("left", "middle", "right"),
            min_frame_index=30,
            max_frame_index=30,
            loss_weight=0.0 if is_distill else 1.0,
            action_loss_weight=0.0,
            disable_train_augmentation=True,
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("random_ce", "distill", "shuffled_distill", "reader_ce", "joint_ce"),
        required=True,
    )
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=TEACHER_CHECKPOINT)
    parser.add_argument("--reset-student", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument(
        "--eval-history-mode",
        choices=("normal", "shuffle_batch", "reverse_time", "zero_history"),
        default="normal",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.phase in ("distill", "shuffled_distill"):
        _trainer.train_step = distill_train_step
        _trainer.eval_step = distill_eval_step
    elif args.eval_history_mode != "normal":
        _trainer.eval_step = ablation_eval_step
    _trainer.main(build_config(args, build_one_swap_labels()))


if __name__ == "__main__":
    main()
