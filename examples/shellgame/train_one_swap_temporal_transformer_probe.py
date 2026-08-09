"""One-swap probe without history-memory compression.

The frozen Pi0Mem visual stem supplies all 30 x 256 Base-camera patch tokens
from frames 0..29. A small trainable factorized space-time Transformer keeps
the full token grid, then uses one learned readout query for three-way cup
classification. It never reads ``history_mem``.
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
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from scripts.mem import train_pi0_mem_compress as _trainer


@dataclasses.dataclass(frozen=True)
class TemporalProbeCheckpointLoader:
    """Restore the source policy while randomly initializing the new probe."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(self.params_path, restore_type=np.ndarray)
        return weight_loaders._merge_params(
            loaded_params,
            params,
            missing_regex=(
                r".*(lora|HistoryResampler|HistoryLayerNorm_0|"
                r"HistoryMultiHeadDotProductAttention_0|HistoryOutProj|"
                r"history_memory_gate_logit|HistoryClassifier|HistoryTemporalProbe).*"
            ),
        )


class FactorizedSpaceTimeBlock(nn.Module):
    """Temporal attention per patch, then spatial attention per frame."""

    width: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, x, *, train: bool):
        b, t, n, d = x.shape
        if d != self.width:
            raise ValueError(f"Expected width {self.width}, got {d}")

        y = nn.LayerNorm(name="temporal_ln", dtype=self.dtype_mm)(x)
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(b * n, t, d)
        y = nn.MultiHeadDotProductAttention(
            name="temporal_attn",
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
            dtype=self.dtype_mm,
        )(y, y)
        y = y.reshape(b, n, t, d).transpose(0, 2, 1, 3)
        x = x + y

        y = nn.LayerNorm(name="spatial_ln", dtype=self.dtype_mm)(x)
        y = y.reshape(b * t, n, d)
        y = nn.MultiHeadDotProductAttention(
            name="spatial_attn",
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
            dtype=self.dtype_mm,
        )(y, y)
        y = y.reshape(b, t, n, d)
        x = x + y

        y = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(x)
        y = nn.Dense(
            self.width * self.mlp_ratio,
            name="mlp_in",
            dtype=self.dtype_mm,
        )(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=not train)
        y = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(y)
        return x + y


class FullHistoryTemporalProbe(nn.Module):
    """Retain every historical patch token until the final readout query."""

    num_frames: int = 30
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
        _, t, _, d = patch_tokens.shape
        if t != self.num_frames or d != self.input_width:
            raise ValueError(
                f"Expected T={self.num_frames}, D={self.input_width}; got {patch_tokens.shape}"
            )

        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(patch_tokens)
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

        b = x.shape[0]
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
        logits = nn.Dense(self.num_classes, name="classifier", dtype=self.dtype_mm)(pooled)
        return logits, pooled


@dataclasses.dataclass(frozen=True)
class OneSwapTemporalProbeConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8

    def create(self, rng: at.KeyArrayLike) -> OneSwapTemporalProbe:
        return OneSwapTemporalProbe(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_temporal_probe(self) -> nnx.filterlib.Filter:
        temporal_probe = nnx_utils.PathRegex(r".*HistoryTemporalProbe.*")
        return nnx.Not(temporal_probe)


class OneSwapTemporalProbe(_base_model.Pi0MemCompress):
    def __init__(self, config: OneSwapTemporalProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.HistoryTemporalProbe = nnx_bridge.ToNNX(
            FullHistoryTemporalProbe(
                num_frames=30,
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
        self.HistoryTemporalProbe.lazy_init(fake_tokens, train=False, rngs=rngs)

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
            raise ValueError(f"Temporal probe expects [B,31,H,W,C], got {image.shape}")

        # ``with_posemb`` is the full [B,T,256,1152] patch grid immediately
        # before the standard HistoryResampler. Bypass history_mem entirely.
        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, :30]
        logits, pooled = self.HistoryTemporalProbe(history_patches, train=train)
        return logits, {
            "history_mem": pooled[:, None, :],
            "encoder_auxes": (),
        }


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    model = OneSwapTemporalProbeConfig(
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
    )
    return _config.TrainConfig(
        name="pi0_shellgame_one_swap_temporal_transformer_probe_260807",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_temporal_probe(),
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
        weight_loader=TemporalProbeCheckpointLoader(SOURCE_CHECKPOINT),
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
    parser.add_argument("--exp-name", default="one_swap_temporal_transformer")
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    _trainer.main(build_config(parsed_args, build_one_swap_labels()))
