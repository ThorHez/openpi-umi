"""One-swap probe with topology-preserving fixed-grid spatial reduction.

Pipeline:
    30 x (16 x 16) frozen SigLIP patch tokens
      -> parameter-free 2 x 2 average pooling per frame
      -> 30 x (8 x 8) fixed-position tokens
      -> the same factorized space-time Transformer and readout used by the
         successful full 256-patch probe.

This is a controlled K=64 comparison against learned-query framewise
compression.  Unlike learned slots, grid cell k always denotes the same image
region across time.
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
import jax.numpy as jnp
import numpy as np

from examples.shellgame.train_one_swap_history_probe import build_one_swap_labels
from examples.shellgame.train_one_swap_history_probe import DATASET_ROOT
from examples.shellgame.train_one_swap_history_probe import SOURCE_CHECKPOINT
from examples.shellgame.train_one_swap_temporal_transformer_probe import (
    FactorizedSpaceTimeBlock,
)
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from scripts.mem import train_pi0_mem_compress as _trainer


@dataclasses.dataclass(frozen=True)
class FixedGridTemporalCheckpointLoader:
    """Restore the source policy while randomly initializing the probe."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(self.params_path, restore_type=np.ndarray)
        return weight_loaders._merge_params(
            loaded_params,
            params,
            missing_regex=(
                r".*(lora|HistoryResampler|HistoryLayerNorm_0|"
                r"HistoryMultiHeadDotProductAttention_0|HistoryOutProj|"
                r"history_memory_gate_logit|HistoryClassifier|"
                r"HistoryFixedGridTemporalProbe).*"
            ),
        )


class FixedGridTemporalProbe(nn.Module):
    """Classify one swap after fixed, topology-preserving spatial pooling."""

    num_frames: int = 30
    input_grid_size: int = 16
    pool_factor: int = 2
    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    num_classes: int = 3
    dropout: float = 0.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens, *, train: bool):
        if patch_tokens.ndim != 4:
            raise ValueError(f"Expected [B,T,N,D], got {patch_tokens.shape}")
        b, t, n, d = patch_tokens.shape
        expected_patches = self.input_grid_size**2
        if t != self.num_frames or n != expected_patches or d != self.input_width:
            raise ValueError(
                f"Expected [B,{self.num_frames},{expected_patches},{self.input_width}], "
                f"got {patch_tokens.shape}"
            )
        if self.input_grid_size % self.pool_factor != 0:
            raise ValueError(
                f"input_grid_size={self.input_grid_size} must be divisible by "
                f"pool_factor={self.pool_factor}"
            )

        # Preserve a stable 2D coordinate system.  Because ``patch_tokens`` are
        # SigLIP ``with_posemb`` features, averaging a local block also retains
        # its corresponding local 2D positional signal.
        output_grid_size = self.input_grid_size // self.pool_factor
        x = patch_tokens.reshape(
            b,
            t,
            output_grid_size,
            self.pool_factor,
            output_grid_size,
            self.pool_factor,
            d,
        )
        x = jnp.mean(x, axis=(3, 5))
        x = x.reshape(b, t, output_grid_size**2, d)

        # Everything below is identical to FullHistoryTemporalProbe.
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(x)
        x = nn.Dense(self.width, name="input_projection", dtype=self.dtype_mm)(x)
        temporal_pos = self.param(
            "temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_frames, 1, self.width),
            x.dtype,
        )
        x = x + temporal_pos

        for block_index in range(self.depth):
            x = FactorizedSpaceTimeBlock(
                name=f"block_{block_index}",
                width=self.width,
                num_heads=self.num_heads,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
            )(x, train=train)

        flat = x.reshape(b, -1, self.width)
        flat = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(flat)
        readout_query = self.param(
            "readout_query",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.width),
            flat.dtype,
        )
        query = jnp.tile(readout_query, (b, 1, 1))
        pooled = nn.MultiHeadDotProductAttention(
            name="readout_attention",
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
            dtype=self.dtype_mm,
        )(query, flat)
        pooled = nn.LayerNorm(name="readout_ln", dtype=self.dtype_mm)(pooled[:, 0])
        logits = nn.Dense(
            self.num_classes,
            name="classifier",
            dtype=self.dtype_mm,
        )(pooled)
        return logits, pooled


@dataclasses.dataclass(frozen=True)
class OneSwapFixedGridTemporalConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2

    def create(self, rng: at.KeyArrayLike) -> OneSwapFixedGridTemporalModel:
        return OneSwapFixedGridTemporalModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_fixed_grid_probe(self) -> nnx.filterlib.Filter:
        probe = nnx_utils.PathRegex(r".*HistoryFixedGridTemporalProbe.*")
        return nnx.Not(probe)


class OneSwapFixedGridTemporalModel(_base_model.Pi0MemCompress):
    def __init__(self, config: OneSwapFixedGridTemporalConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.HistoryFixedGridTemporalProbe = nnx_bridge.ToNNX(
            FixedGridTemporalProbe(
                num_frames=30,
                input_grid_size=16,
                pool_factor=config.spatial_pool_factor,
                input_width=1152,
                width=config.temporal_width,
                depth=config.temporal_depth,
                num_heads=config.temporal_heads,
                num_classes=config.history_classifier_num_classes,
                dropout=0.0,
                dtype_mm=config.dtype,
            )
        )
        fake_tokens = jnp.zeros((1, 30, 256, 1152), dtype=jnp.bfloat16)
        self.HistoryFixedGridTemporalProbe.lazy_init(
            fake_tokens,
            train=False,
            rngs=rngs,
        )

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
            raise ValueError(
                f"Fixed-grid probe expects [B,31,H,W,C], got {image.shape}"
            )

        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, :30]
        logits, pooled = self.HistoryFixedGridTemporalProbe(
            history_patches,
            train=train,
        )
        return logits, {
            "history_mem": pooled[:, None, :],
            "encoder_auxes": (),
        }


def build_config(
    args: argparse.Namespace,
    labels_path: pathlib.Path,
) -> _config.TrainConfig:
    model = OneSwapFixedGridTemporalConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        num_frames=31,
        memory_every=1,
        current_frame_index=-1,
        history_memory_tokens=256,
        history_resampler_depth=1,
        history_use_current_condition=True,
        history_gate_fixed=1.0,
        diversity_weight=0.0,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=3,
        temporal_width=args.temporal_width,
        temporal_depth=args.temporal_depth,
        temporal_heads=args.temporal_heads,
        spatial_pool_factor=args.spatial_pool_factor,
    )
    return _config.TrainConfig(
        name="pi0_shellgame_one_swap_fixed_grid_temporal_probe_260808",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_fixed_grid_probe(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=str(DATASET_ROOT),
                    assets=_config.AssetsConfig(
                        asset_id=".",
                        assets_dir=str(DATASET_ROOT),
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
        weight_loader=FixedGridTemporalCheckpointLoader(SOURCE_CHECKPOINT),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=args.steps,
            decay_lr=args.peak_lr * 0.1,
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
        resume=args.resume,
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
    parser.add_argument(
        "--exp-name",
        default="one_swap_fixed_grid_k64_260808",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--temporal-width", type=int, default=256)
    parser.add_argument("--temporal-depth", type=int, default=2)
    parser.add_argument("--temporal-heads", type=int, default=8)
    parser.add_argument("--spatial-pool-factor", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    _trainer.main(build_config(parsed_args, build_one_swap_labels()))
