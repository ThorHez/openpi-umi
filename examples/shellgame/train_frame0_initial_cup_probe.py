"""Frozen-SigLIP probe: frame 0 base image -> initial ball cup.

This diagnostic deliberately bypasses history compression, language, robot
state, actions, and the wrist camera.  A linear classifier reads the flattened
16x16 grid of frozen SigLIP patch features from the base image at frame 0.
Validation is split by episode through the existing Pi0Mem training pipeline.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

# Running a file below ``examples/`` puts only that directory on sys.path;
# expose the repository root so the existing training entry point is importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from scripts.mem import train_pi0_mem_compress as _trainer


DATASET_ROOT = (
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_absolute_joint"
)
SOURCE_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_mem_compress_evan_shellgame_openpi_joint_260727/"
    "my_experiment_30f_s2_6gpu/23000/params"
)


@dataclasses.dataclass(frozen=True)
class Frame0InitialCupProbeConfig(_base_model.Pi0MemCompressConfig):
    """Single-frame probe configuration kept separate from policy configs."""

    probe_stream: str = "base_rgb"

    def create(self, rng: at.KeyArrayLike) -> Frame0InitialCupProbe:
        return Frame0InitialCupProbe(self, rngs=nnx.Rngs(rng))


class Frame0InitialCupProbe(_base_model.Pi0MemCompress):
    """Classify frozen, spatially ordered SigLIP features from one image."""

    def __init__(self, config: Frame0InitialCupProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.probe_stream = config.probe_stream

    def compute_history_classification(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ):
        # ``disable_train_augmentation=True`` in the experiment config means
        # train is False here. Keeping the argument makes the method conform to
        # the existing classification-only trainer contract.
        observation = _model.preprocess_observation(rng, observation, train=train)
        if self.probe_stream not in observation.images:
            raise KeyError(
                f"Missing probe stream {self.probe_stream!r}; "
                f"available streams are {tuple(observation.images)}"
            )

        image = observation.images[self.probe_stream]
        if image.ndim == 4:
            image = image[:, None, ...]
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(f"Frame-0 probe expects [B,1,H,W,C], got {image.shape}")

        _, encoder_out = self.PaliGemma.img(image, train=False)
        # [B, 256, 1152] for 224x224 input with So400m/14. Flattening retains
        # spatial location; global averaging would unnecessarily dilute a ball
        # that occupies only a few pixels in one patch.
        patch_features = encoder_out["encoded"]
        patch_features = self.HistoryClassifierNorm(patch_features)
        logits = self.HistoryClassifierHead(
            patch_features.reshape(patch_features.shape[0], -1)
        )

        # Reuse the trainer's generic monitoring contract. Here these are
        # image-patch features, not compressed history memory.
        return logits, {
            "history_mem": jnp.asarray(patch_features),
            "encoder_auxes": (encoder_out["encoder"],),
        }


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    model = Frame0InitialCupProbeConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        num_frames=1,
        memory_every=0,
        current_frame_index=0,
        history_memory_tokens=256,
        history_resampler_depth=1,
        history_use_current_condition=False,
        history_gate_fixed=0.0,
        diversity_weight=0.0,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=3,
        probe_stream="base_rgb",
    )

    return _config.TrainConfig(
        name="pi0_shellgame_frame0_initial_cup_probe_260807",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_history_classifier_probe(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=DATASET_ROOT,
                    assets=_config.AssetsConfig(asset_id=".", assets_dir=DATASET_ROOT),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=1,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            SOURCE_CHECKPOINT
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=20,
            peak_lr=1e-4,
            decay_steps=args.steps,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=10.0),
        ema_decay=None,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=args.steps,
        keep_period=args.steps,
        val_ratio=0.1,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        wandb_enabled=False,
        overwrite=args.overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(
            enabled=True,
            episodes_metadata_path=f"{DATASET_ROOT}/meta/episodes.jsonl",
            label_key="initial_ball_cup",
            classes=("left", "middle", "right"),
            min_frame_index=0,
            max_frame_index=0,
            loss_weight=1.0,
            action_loss_weight=0.0,
            disable_train_augmentation=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="frame0_initial_cup_linear")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    _trainer.main(build_config(parse_args()))
