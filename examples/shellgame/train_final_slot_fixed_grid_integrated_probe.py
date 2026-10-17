"""Classify the final cup from the exact frame-59 action-policy history.

The input is a 31-frame, stride-2 clip ending at raw frame 59.  It therefore
matches the action60 experiments while covering reveal and all three swaps.
The label is the verified ``final_ball_cup`` metadata field.  This is a
diagnostic probe, not an auxiliary objective for the production action model.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx

from examples.shellgame.train_fixed_grid_action60_probe import INTEGRATED_READER_CHECKPOINT
from examples.shellgame.train_fixed_grid_action60_probe import LEROBOT_ROOT
from examples.shellgame.train_fixed_grid_action60_probe import _frame59_only
from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import IntegratedCheckpointLoader
from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import IntegratedProbeConfig
from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import IntegratedProbeModel
from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import ablation_eval_step
from examples.shellgame.train_one_swap_history_probe import build_one_swap_labels
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer


@dataclasses.dataclass(frozen=True)
class FinalSlotProbeConfig(IntegratedProbeConfig):
    def create(self, rng: at.KeyArrayLike) -> IntegratedProbeModel:
        return IntegratedProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_for_phase(self, phase: str) -> nnx.filterlib.Filter:
        history = nnx_utils.PathRegex(r".*PaliGemma/img/Transformer/FixedGridTemporalHistory_0.*")
        reader = nnx_utils.PathRegex(
            r".*PaliGemma/img/Transformer/encoderblock/"
            r"(?:HistoryLayerNorm_0|HistoryMultiHeadDotProductAttention_0).*"
        )
        readout = nnx_utils.PathRegex(r".*HistoryIntegratedCurrentReadout.*")
        if phase == "readout_only":
            trainable = readout
        elif phase == "joint_ce":
            trainable = nnx.Any(history, reader, readout)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return nnx.Not(trainable)


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    model = FinalSlotProbeConfig(
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
        name=f"pi0_shellgame_final_slot_fixed_grid_{args.phase}_260808",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_for_phase(args.phase),
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
                    num_frames=31,
                    frame_stride=2,
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
            label_key="final_ball_cup",
            classes=("left", "middle", "right"),
            min_frame_index=59,
            max_frame_index=59,
            loss_weight=1.0,
            action_loss_weight=0.0,
            disable_train_augmentation=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("readout_only", "joint_ce"), required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=INTEGRATED_READER_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--warmup-steps", type=int, default=20)
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


def main() -> None:
    args = parse_args()
    _trainer._filter_memory_classifier_frame_range = _frame59_only  # noqa: SLF001
    if args.eval_history_mode != "normal":
        _trainer.eval_step = ablation_eval_step
    _trainer.main(build_config(args, build_one_swap_labels()))


if __name__ == "__main__":
    main()
