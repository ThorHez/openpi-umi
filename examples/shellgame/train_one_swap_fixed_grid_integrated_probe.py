"""Validate the optimized fixed-grid temporal memory inside the real Pi0 reader.

The proven Stage-3 K64/M128 history weights are transplanted into the new
integrated visual encoder.  ``reader_ce`` freezes that encoder and trains only
the periodic Pi0 memory-reader branches plus a classifier over current tokens.
``joint_ce`` then optionally fine-tunes the complete history-to-reader path.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax.numpy as jnp
import numpy as np

from examples.shellgame.train_one_swap_history_probe import DATASET_ROOT
from examples.shellgame.train_one_swap_history_probe import build_one_swap_labels
from openpi.models import model as _model
from openpi.models import pi0_mem_fixed_grid_temporal as _fixed_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer

STAGE3_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_one_swap_fixed_grid_staged_joint_probe_260808/"
    "one_swap_fixed_grid_stage3_joint_260808/199/params"
)


class IntegratedCurrentReadout(nn.Module):
    """Small diagnostic head over current tokens after memory cross-attention."""

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
        return nn.Dense(self.num_classes, name="classifier", dtype=jnp.float32)(pooled)


@dataclasses.dataclass(frozen=True)
class IntegratedProbeConfig(_fixed_model.Pi0MemFixedGridTemporalConfig):
    eval_history_mode: str = "normal"

    def create(self, rng: at.KeyArrayLike) -> IntegratedProbeModel:
        return IntegratedProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_for_phase(self, phase: str) -> nnx.filterlib.Filter:
        history = nnx_utils.PathRegex(r".*PaliGemma/img/Transformer/FixedGridTemporalHistory_0.*")
        reader = nnx_utils.PathRegex(
            r".*PaliGemma/img/Transformer/encoderblock/"
            r"(?:HistoryLayerNorm_0|HistoryMultiHeadDotProductAttention_0).*"
        )
        readout = nnx_utils.PathRegex(r".*HistoryIntegratedCurrentReadout.*")
        if phase == "reader_ce":
            trainable = nnx.Any(reader, readout)
        elif phase == "joint_ce":
            trainable = nnx.Any(history, reader, readout)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return nnx.Not(trainable)


class IntegratedProbeModel(_fixed_model.Pi0MemFixedGridTemporal):
    def __init__(self, config: IntegratedProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.HistoryIntegratedCurrentReadout = nnx_bridge.ToNNX(
            IntegratedCurrentReadout(
                input_width=1152,
                width=config.temporal_width,
                num_classes=config.history_classifier_num_classes,
                dtype_mm=config.dtype,
            )
        )
        fake_current = jnp.zeros((1, 256, 1152), dtype=jnp.bfloat16)
        self.HistoryIntegratedCurrentReadout.lazy_init(fake_current, rngs=rngs)

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
            image = image[:, None]
        if image.ndim != 5 or image.shape[1] != 31:
            raise ValueError(f"Integrated probe expects [B,31,H,W,C], got {image.shape}")
        _, encoder_out = self.PaliGemma.img(image, train=train)
        encoder_aux = encoder_out["encoder"]
        logits = self.HistoryIntegratedCurrentReadout(encoder_aux["pre_ln"])
        return logits, {
            "history_mem": encoder_aux["history_mem"],
            "encoder_auxes": (encoder_aux,),
        }


@dataclasses.dataclass(frozen=True)
class IntegratedCheckpointLoader:
    """Load normal Pi0 leaves and transplant the proven fixed-grid subtree."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        result = dict(target)
        target_prefix = "PaliGemma/img/Transformer/FixedGridTemporalHistory_0/"
        source_prefix = "HistoryFixedGridTemporalMemory/"
        mapped_history = 0
        expected_history = 0
        exact = 0
        shape_mismatch = []
        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
                if key.startswith(target_prefix):
                    mapped_history += 1
                continue
            if key.startswith(target_prefix):
                expected_history += 1
                source_key = source_prefix + key.removeprefix(target_prefix)
                candidate = source.get(source_key)
                if candidate is not None and np.shape(candidate) == np.shape(reference):
                    result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                    mapped_history += 1
                else:
                    shape_mismatch.append((key, source_key, np.shape(reference)))

        total_history = sum(key.startswith(target_prefix) for key in target)
        if mapped_history != total_history:
            raise ValueError(
                "Fixed-grid checkpoint transplant incomplete: "
                f"{mapped_history}/{total_history}; examples={shape_mismatch[:3]}"
            )
        print(
            "IntegratedCheckpointLoader: "
            f"exact={exact}, fixed_grid={mapped_history}/{total_history}, "
            f"transplanted={expected_history}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def _ablate_history(observation: _model.Observation, mode: str) -> _model.Observation:
    """Modify only history frames 0:30; preserve frame 30 and every label/input."""
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
    observation = _ablate_history(observation, config.model.eval_history_mode)
    return _BASE_EVAL_STEP(
        config,
        rng,
        state,
        (observation, actions),
        class_labels_by_episode=class_labels_by_episode,
    )


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    model = IntegratedProbeConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        num_frames=31,
        memory_every=4,
        current_frame_index=-1,
        history_memory_tokens=128,
        history_resampler_depth=1,
        history_use_current_condition=False,
        history_gate_fixed=1.0,
        diversity_weight=0.0,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=3,
        temporal_width=256,
        temporal_depth=2,
        temporal_heads=8,
        spatial_pool_factor=2,
        eval_history_mode=args.eval_history_mode,
    )
    return _config.TrainConfig(
        name=f"pi0_shellgame_fixed_grid_integrated_{args.phase}_260808",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_for_phase(args.phase),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=str(DATASET_ROOT),
                    assets=_config.AssetsConfig(asset_id=".", assets_dir=str(DATASET_ROOT)),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=31,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=IntegratedCheckpointLoader(args.init_checkpoint),
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
            loss_weight=1.0,
            action_loss_weight=0.0,
            disable_train_augmentation=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("reader_ce", "joint_ce"), required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=STAGE3_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument(
        "--eval-history-mode",
        choices=("normal", "shuffle_batch", "reverse_time", "zero_history"),
        default="normal",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_history_mode != "normal":
        _trainer.eval_step = ablation_eval_step
    _trainer.main(build_config(args, build_one_swap_labels()))
