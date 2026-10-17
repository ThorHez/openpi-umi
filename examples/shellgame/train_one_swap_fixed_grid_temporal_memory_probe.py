"""One-swap probe with fixed-grid temporal reasoning and final Pi0 memory.

Pipeline:
    30 x (16 x 16) frozen SigLIP patch tokens
      -> parameter-free 2 x 2 average pooling per frame
      -> factorized temporal/spatial Transformer over [B, 30, 64, 256]
      -> learned-query compression to [B, 128, 1152] Pi0-width memory
      -> attention-pooling three-way cup classifier.

This experiment tests whether final memory compression preserves the tracking
signal that the pre-memory fixed-grid probe learned to classify perfectly.
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

from examples.shellgame.train_one_swap_framewise_temporal_memory_probe import (
    FinalMemoryCompressor,
)
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
class FixedGridTemporalMemoryCheckpointLoader:
    """Restore the source policy and initialize the new memory path."""

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
                r"HistoryFixedGridTemporalMemory).*"
            ),
        )


class FixedGridTemporalMemoryProbe(nn.Module):
    """Track on a fixed grid, then compress the contextualized history."""

    num_frames: int = 30
    input_grid_size: int = 16
    pool_factor: int = 2
    input_width: int = 1152
    width: int = 256
    temporal_depth: int = 2
    num_heads: int = 8
    memory_tokens: int = 128
    output_width: int = 1152
    num_classes: int = 3
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

        # Identical fixed-grid temporal path to the successful pre-memory probe.
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(x)
        x = nn.Dense(self.width, name="input_projection", dtype=self.dtype_mm)(x)
        temporal_pos = self.param(
            "temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_frames, 1, self.width),
            x.dtype,
        )
        x = x + temporal_pos
        for block_index in range(self.temporal_depth):
            x = FactorizedSpaceTimeBlock(
                name=f"temporal_block_{block_index}",
                width=self.width,
                num_heads=self.num_heads,
                dropout=0.0,
                dtype_mm=self.dtype_mm,
            )(x, train=train)

        memory = FinalMemoryCompressor(
            name="final_memory_compressor",
            width=self.width,
            output_width=self.output_width,
            num_memory_tokens=self.memory_tokens,
            num_heads=self.num_heads,
            dtype_mm=self.dtype_mm,
        )(x)

        # Read only the final Pi0-width memory.  This is the stable shared
        # attention pool used by the earlier final-memory diagnostic.
        readout_tokens = nn.Dense(
            self.width,
            name="readout_projection",
            dtype=self.dtype_mm,
        )(memory)
        readout_tokens = nn.tanh(readout_tokens)
        attention_logits = nn.Dense(
            1,
            name="readout_attention",
            dtype=jnp.float32,
        )(readout_tokens.astype(jnp.float32))
        attention_weights = nn.softmax(attention_logits, axis=1)
        pooled = jnp.sum(
            attention_weights * readout_tokens.astype(jnp.float32),
            axis=1,
        )
        pooled = nn.LayerNorm(name="readout_ln", dtype=jnp.float32)(pooled)
        logits = nn.Dense(
            self.num_classes,
            name="classifier",
            dtype=jnp.float32,
        )(pooled)
        return logits, memory


@dataclasses.dataclass(frozen=True)
class OneSwapFixedGridTemporalMemoryConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2
    final_memory_tokens: int = 128

    def create(self, rng: at.KeyArrayLike) -> OneSwapFixedGridTemporalMemoryModel:
        return OneSwapFixedGridTemporalMemoryModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_fixed_grid_memory(self) -> nnx.filterlib.Filter:
        probe = nnx_utils.PathRegex(r".*HistoryFixedGridTemporalMemory.*")
        return nnx.Not(probe)


class OneSwapFixedGridTemporalMemoryModel(_base_model.Pi0MemCompress):
    def __init__(
        self,
        config: OneSwapFixedGridTemporalMemoryConfig,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, rngs)
        self.HistoryFixedGridTemporalMemory = nnx_bridge.ToNNX(
            FixedGridTemporalMemoryProbe(
                num_frames=30,
                input_grid_size=16,
                pool_factor=config.spatial_pool_factor,
                input_width=1152,
                width=config.temporal_width,
                temporal_depth=config.temporal_depth,
                num_heads=config.temporal_heads,
                memory_tokens=config.final_memory_tokens,
                output_width=1152,
                num_classes=config.history_classifier_num_classes,
                dtype_mm=config.dtype,
            )
        )
        fake_tokens = jnp.zeros((1, 30, 256, 1152), dtype=jnp.bfloat16)
        self.HistoryFixedGridTemporalMemory.lazy_init(
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
                f"Fixed-grid memory probe expects [B,31,H,W,C], got {image.shape}"
            )

        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, :30]
        logits, memory = self.HistoryFixedGridTemporalMemory(
            history_patches,
            train=train,
        )
        return logits, {
            "history_mem": memory,
            "encoder_auxes": (),
        }


def build_config(
    args: argparse.Namespace,
    labels_path: pathlib.Path,
) -> _config.TrainConfig:
    model = OneSwapFixedGridTemporalMemoryConfig(
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
        diversity_weight=args.diversity_weight,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=3,
        temporal_width=args.temporal_width,
        temporal_depth=args.temporal_depth,
        temporal_heads=args.temporal_heads,
        spatial_pool_factor=args.spatial_pool_factor,
        final_memory_tokens=args.final_memory_tokens,
    )
    return _config.TrainConfig(
        name="pi0_shellgame_one_swap_fixed_grid_temporal_memory_probe_260808",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_fixed_grid_memory(),
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
        weight_loader=FixedGridTemporalMemoryCheckpointLoader(SOURCE_CHECKPOINT),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=args.steps,
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=args.clip_gradient_norm),
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
        default="one_swap_fixed_grid_k64_memory128_260808",
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
    parser.add_argument("--final-memory-tokens", type=int, default=128)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--clip-gradient-norm", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    _trainer.main(build_config(parsed_args, build_one_swap_labels()))
