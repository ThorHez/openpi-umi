"""Locate where the fixed-grid history encoder loses cup identity over three swaps.

This is the non-causal production history encoder used by the successful
one-swap interface, not the earlier causal diagnostic.  To prevent future
leakage while keeping its fixed 60-position input, the same shared encoder sees
three endpoint-padded histories:

    swap_0: raw 0..29, then repeat frame 29 through position 59
    swap_1: raw 0..39, then repeat frame 39 through position 59
    swap_2: raw 0..49, then repeat frame 49 through position 59

Each history follows the exact K64 -> depth-2 factorized Transformer -> M128
pipeline.  One shared classifier predicts the cup slot at all three endpoints.
The factorized 27-way joint logits make the existing trainer optimize the sum
of the three endpoint cross-entropies.  Pi0 action loss is disabled.
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

from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import IntegratedCurrentReadout
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import DATASET_ROOT
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import JOINT_CLASSES
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import multistage_eval_step
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.models import siglip_mem_fixed_grid_temporal as _fixed_siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer

DEFAULT_INIT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_full_integrated_history_readout_control_260808/"
    "three_swap_full60_integrated_history_readout_260808/599/params"
)
ENDPOINTS = (29, 39, 49)


class SharedEndpointFixedGridTracker(nn.Module):
    """Run one shared full history encoder and readout at three clean prefixes."""

    num_frames: int = 60
    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    spatial_pool_factor: int = 2
    num_memory_tokens: int = 128
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens):
        b, t, n, d = patch_tokens.shape
        if (t, n, d) != (self.num_frames, 256, self.input_width):
            raise ValueError(
                f"Expected [B,{self.num_frames},256,{self.input_width}], got {patch_tokens.shape}"
            )

        frame_ids = jnp.arange(self.num_frames)[None, None, :, None, None]
        endpoints = jnp.asarray(ENDPOINTS)[None, :, None, None, None]
        source = patch_tokens[:, None, :, :, :]
        endpoint_tokens = jnp.stack(
            [patch_tokens[:, endpoint] for endpoint in ENDPOINTS],
            axis=1,
        )[:, :, None, :, :]
        padded = jnp.where(frame_ids <= endpoints, source, endpoint_tokens)
        padded = padded.reshape(b * len(ENDPOINTS), self.num_frames, n, d)

        shared_history = _fixed_siglip.FixedGridTemporalHistory(
            name="shared_history",
            input_width=self.input_width,
            temporal_width=self.width,
            temporal_depth=self.depth,
            temporal_heads=self.num_heads,
            spatial_pool_factor=self.spatial_pool_factor,
            num_memory_tokens=self.num_memory_tokens,
            output_width=self.input_width,
            dropout=0.0,
            dtype_mm=self.dtype_mm,
        )(padded, deterministic=True)
        shared_readout = IntegratedCurrentReadout(
            name="shared_readout",
            input_width=self.input_width,
            width=self.width,
            num_classes=3,
            dtype_mm=self.dtype_mm,
        )(shared_history)
        stage_logits = shared_readout.reshape(b, len(ENDPOINTS), 3)

        logits_0, logits_1, logits_2 = (
            stage_logits[:, 0],
            stage_logits[:, 1],
            stage_logits[:, 2],
        )
        joint_logits = (
            logits_0[:, :, None, None]
            + logits_1[:, None, :, None]
            + logits_2[:, None, None, :]
        ).reshape(b, 27)
        return joint_logits, stage_logits


@dataclasses.dataclass(frozen=True)
class SharedEndpointProbeConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2
    endpoint_memory_tokens: int = 128

    def create(self, rng: at.KeyArrayLike) -> SharedEndpointProbeModel:
        return SharedEndpointProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_for_phase(self, phase: str) -> nnx.filterlib.Filter:
        history = nnx_utils.PathRegex(
            r".*HistoryThreeSwapSharedEndpointTracker/shared_history.*"
        )
        readout = nnx_utils.PathRegex(
            r".*HistoryThreeSwapSharedEndpointTracker/shared_readout.*"
        )
        if phase == "readout_only":
            trainable = readout
        elif phase == "joint_aux":
            trainable = nnx.Any(history, readout)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return nnx.Not(trainable)


class SharedEndpointProbeModel(_base_model.Pi0MemCompress):
    def __init__(self, config: SharedEndpointProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.HistoryThreeSwapSharedEndpointTracker = nnx_bridge.ToNNX(
            SharedEndpointFixedGridTracker(
                num_frames=60,
                input_width=1152,
                width=config.temporal_width,
                depth=config.temporal_depth,
                num_heads=config.temporal_heads,
                spatial_pool_factor=config.spatial_pool_factor,
                num_memory_tokens=config.endpoint_memory_tokens,
                dtype_mm=config.dtype,
            )
        )
        fake_tokens = jnp.zeros((1, 60, 256, 1152), dtype=jnp.bfloat16)
        self.HistoryThreeSwapSharedEndpointTracker.lazy_init(fake_tokens, rngs=rngs)

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
        if image.ndim != 5 or image.shape[1] != 61:
            raise ValueError(f"Shared endpoint probe expects [B,61,H,W,C], got {image.shape}")

        # The frozen Pi0 visual backbone supplies the exact pretrained patch
        # embedding.  Only raw frames 0..59 enter the endpoint tracker.
        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, :60]
        joint_logits, stage_logits = self.HistoryThreeSwapSharedEndpointTracker(
            history_patches
        )
        return joint_logits, {
            "history_mem": stage_logits,
            "encoder_auxes": (),
        }


@dataclasses.dataclass(frozen=True)
class SharedEndpointCheckpointLoader:
    """Restore an exact probe or transplant the proven full-interface history."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        result = {}
        exact = 0
        exact_history = 0
        mapped_history = 0
        mapped_readout = 0
        initialized = []
        target_tracker = "HistoryThreeSwapSharedEndpointTracker/"
        target_history = target_tracker + "shared_history/"
        target_readout = target_tracker + "shared_readout/"
        source_history = "PaliGemma/img/Transformer/FixedGridTemporalHistory_0/"
        source_readout = "HistoryIntegratedCurrentReadout/"

        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
                if key.startswith(target_history):
                    exact_history += 1
                continue

            mapped_candidate = None
            mapping_kind = None
            if key.startswith(target_history):
                mapped_candidate = source.get(source_history + key.removeprefix(target_history))
                mapping_kind = "history"
            elif key.startswith(target_readout):
                mapped_candidate = source.get(source_readout + key.removeprefix(target_readout))
                mapping_kind = "readout"

            if mapped_candidate is not None and np.shape(mapped_candidate) == np.shape(reference):
                result[key] = np.asarray(mapped_candidate, dtype=np.dtype(reference.dtype))
                if mapping_kind == "history":
                    mapped_history += 1
                else:
                    mapped_readout += 1
            else:
                result[key] = reference
                initialized.append(key)

        history_count = sum(key.startswith(target_history) for key in target)
        if mapped_history + exact_history != history_count:
            missing = [key for key in initialized if key.startswith(target_history)]
            raise ValueError(
                f"Shared history restore incomplete: mapped={mapped_history}/{history_count}, "
                f"missing={missing[:5]}"
            )
        print(
            "SharedEndpointCheckpointLoader: "
            f"exact={exact}, history={mapped_history}+{exact_history}/{history_count}, "
            f"mapped_readout={mapped_readout}, initialized={len(initialized)}, "
            f"examples={initialized[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    model = SharedEndpointProbeConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        num_frames=61,
        memory_every=0,
        current_frame_index=-1,
        # The unused legacy resampler needs a non-empty shape.  M=1 and
        # memory_every=0 keep it inert and cheap.
        history_memory_tokens=1,
        history_resampler_depth=1,
        history_use_current_condition=False,
        history_gate_fixed=0.0,
        diversity_weight=0.0,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=27,
        temporal_width=256,
        temporal_depth=2,
        temporal_heads=8,
        spatial_pool_factor=2,
        endpoint_memory_tokens=128,
    )
    return _config.TrainConfig(
        name=f"pi0_shellgame_three_swap_shared_endpoint_{args.phase}_260809",
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
                    num_frames=61,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=SharedEndpointCheckpointLoader(args.init_checkpoint),
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
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(
            enabled=True,
            episodes_metadata_path=str(labels_path),
            label_key="swap_track_code",
            classes=JOINT_CLASSES,
            min_frame_index=60,
            max_frame_index=60,
            loss_weight=1.0,
            action_loss_weight=0.0,
            disable_train_augmentation=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("readout_only", "joint_aux"), required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _trainer.eval_step = multistage_eval_step
    _trainer.main(build_config(args, build_three_swap_labels()))


if __name__ == "__main__":
    main()
