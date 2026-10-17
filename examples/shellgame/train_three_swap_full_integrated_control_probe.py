"""Controlled three-swap classification through the proven full memory interface.

Only the temporal extent and label differ from the successful one-swap
integrated-reader experiment.  The exact production-style visual path is:

    raw history frames 0..59
      -> frozen SigLIP patch embedding
      -> fixed 2x2 grid pooling (K=64)
      -> non-causal depth-2 factorized temporal/spatial Transformer
      -> M=128, D=1152 final memory compressor
      -> current-frame 27-layer SigLIP with memory reads every four layers
      -> current-token classification readout

Frame 60 is used only as the dataset anchor.  Before encoding, its image is
replaced with frame 59 so the first robot-approach frame cannot leak the target.
Pi0 action loss is disabled; the sole target is ``final_ball_cup``.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import flax.traverse_util
import numpy as np

from examples.shellgame.train_fixed_grid_action60_probe import INTEGRATED_READER_CHECKPOINT
from examples.shellgame.train_fixed_grid_action60_probe import LEROBOT_ROOT
from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import IntegratedProbeConfig
from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import IntegratedProbeModel
from examples.shellgame.train_one_swap_history_probe import build_one_swap_labels
from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer


@dataclasses.dataclass(frozen=True)
class FullIntegratedControlConfig(IntegratedProbeConfig):
    def create(self, rng: at.KeyArrayLike) -> FullIntegratedControlModel:
        return FullIntegratedControlModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_for_control_phase(self, phase: str) -> nnx.filterlib.Filter:
        history = nnx_utils.PathRegex(
            r".*PaliGemma/img/Transformer/FixedGridTemporalHistory_0.*"
        )
        reader = nnx_utils.PathRegex(
            r".*PaliGemma/img/Transformer/encoderblock/"
            r"(?:HistoryLayerNorm_0|HistoryMultiHeadDotProductAttention_0).*"
        )
        readout = nnx_utils.PathRegex(r".*HistoryIntegratedCurrentReadout.*")
        if phase == "readout_only":
            trainable = readout
        elif phase == "reader_ce":
            trainable = nnx.Any(reader, readout)
        elif phase == "history_readout":
            trainable = nnx.Any(history, readout)
        elif phase == "joint_ce":
            trainable = nnx.Any(history, reader, readout)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return nnx.Not(trainable)


class FullIntegratedControlModel(IntegratedProbeModel):
    """Run the full integrated reader while excluding frame-60 action leakage."""

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
            raise ValueError(f"Full integrated control expects [B,61,H,W,C], got {image.shape}")

        # History remains raw 0..59.  The visual current frame is a duplicate
        # of settle frame 59, so no robot-approach cue can reveal the target.
        image = image.at[:, 60].set(image[:, 59])
        _, encoder_out = self.PaliGemma.img(image, train=train)
        encoder_aux = encoder_out["encoder"]
        logits = self.HistoryIntegratedCurrentReadout(encoder_aux["pre_ln"])
        return logits, {
            "history_mem": encoder_aux["history_mem"],
            "encoder_auxes": (encoder_aux,),
        }


@dataclasses.dataclass(frozen=True)
class Full60IntegratedCheckpointLoader:
    """Restore the successful one-swap full interface with a 60-position extension."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        result = {}
        exact = 0
        initialized = []
        temporal_key = (
            "PaliGemma/img/Transformer/FixedGridTemporalHistory_0/"
            "temporal_pos_embedding"
        )
        temporal_prefix_copied = False

        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
                if key == temporal_key:
                    temporal_prefix_copied = True
                continue
            if key == temporal_key and candidate is not None:
                candidate_array = np.asarray(candidate)
                reference_array = np.zeros(reference.shape, dtype=np.dtype(reference.dtype))
                if (
                    candidate_array.shape[0] == reference_array.shape[0] == 1
                    and candidate_array.shape[2:] == reference_array.shape[2:]
                    and candidate_array.shape[1] == 30
                    and reference_array.shape[1] == 60
                ):
                    # Raw positions 0..29 retain exactly their learned one-swap
                    # embeddings. New raw positions 30..59 keep target init.
                    reference_array[:, :30] = candidate_array
                    result[key] = reference_array.astype(reference.dtype)
                    temporal_prefix_copied = True
                    continue
            result[key] = reference
            initialized.append(key)

        if not temporal_prefix_copied:
            raise ValueError("Failed to extend the one-swap temporal position embedding 30 -> 60")
        print(
            "Full60IntegratedCheckpointLoader: "
            f"exact={exact}, temporal_loaded=True, "
            f"initialized={len(initialized)}, examples={initialized[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    model = FullIntegratedControlConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        num_frames=61,
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
        eval_history_mode="normal",
    )
    return _config.TrainConfig(
        name=f"pi0_shellgame_three_swap_full_integrated_{args.phase}_control_260808",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_for_control_phase(args.phase),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=str(LEROBOT_ROOT),
                    assets=_config.AssetsConfig(asset_id=".", assets_dir=str(LEROBOT_ROOT)),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=61,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=Full60IntegratedCheckpointLoader(args.init_checkpoint),
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
            label_key="final_ball_cup",
            classes=("left", "middle", "right"),
            min_frame_index=60,
            max_frame_index=60,
            loss_weight=1.0,
            action_loss_weight=0.0,
            disable_train_augmentation=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("readout_only", "reader_ce", "history_readout", "joint_ce"),
        default="history_readout",
    )
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=INTEGRATED_READER_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _trainer.main(build_config(args, build_one_swap_labels()))


if __name__ == "__main__":
    main()
