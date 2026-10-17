"""Train a frame-59 -> frames-60:75 absolute-joint action probe.

The input clip contains 31 frames with stride 2 and ends at raw frame 59, so
it spans the reveal and all scripted swaps.  The LeRobot action chunk stored
at frame 59 starts with raw ``joint_pos[60]``.  The proven K64/M128 history
encoder and integrated Pi0 reader are frozen; only the action expert and
action projections are optimized in the first stage.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import numpy as np

from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import IntegratedCheckpointLoader
from openpi.models import pi0_mem_fixed_grid_temporal as _fixed_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer

INTEGRATED_READER_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_fixed_grid_integrated_reader_ce_260808/"
    "integrated_reader_k64_m128_e4_260808/299/params"
)
LEROBOT_ROOT = pathlib.Path("/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_joint")
RAW_DATASET_ROOT = "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_absolute_joint_dataset"


@dataclasses.dataclass(frozen=True)
class Action60ProbeConfig(_fixed_model.Pi0MemFixedGridTemporalConfig):
    def create(self, rng: at.KeyArrayLike) -> _fixed_model.Pi0MemFixedGridTemporal:
        return _fixed_model.Pi0MemFixedGridTemporal(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_for_phase(self, phase: str) -> nnx.filterlib.Filter:
        action_output = nnx_utils.PathRegex(r".*action_out_proj.*")
        if phase == "eval_only":
            trainable = action_output
        elif phase == "action_expert":
            action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm/.*_1.*")
            action_modules = nnx_utils.PathRegex(r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*")
            trainable = nnx.Any(action_expert, action_modules)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return nnx.Not(trainable)


def _frame59_only(dataset, indices: list[int], _classifier_config) -> list[int]:
    """Restrict both episode-level splits to the first post-swap action query."""
    current = dataset
    hf_dataset = None
    while current is not None:
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            break
        current = getattr(current, "_dataset", None)
    if hf_dataset is None or "frame_index" not in getattr(hf_dataset, "column_names", ()):
        raise ValueError("Action60 probe requires a frame_index column")
    selected = np.asarray(indices, dtype=np.int64)
    frame_indices = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    filtered = selected[frame_indices[selected] == 59].tolist()
    if not filtered:
        raise ValueError("frame_index=59 selected no rows")
    return filtered


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    model = Action60ProbeConfig(
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
        history_classifier_num_classes=0,
        temporal_width=256,
        temporal_depth=2,
        temporal_heads=8,
        spatial_pool_factor=2,
    )
    return _config.TrainConfig(
        name=f"pi0_shellgame_fixed_grid_action60_{args.phase}_260808",
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
        optimizer=_optimizer.AdamW(clip_gradient_norm=args.clip_gradient_norm),
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
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(
            enabled=True,
            raw_dataset_root=RAW_DATASET_ROOT,
            robosuite_root="/data2/hzl_workspace_for_pi_mem/robosuite",
            interval=args.cup_eval_interval,
            num_episodes=args.cup_eval_episodes,
            batch_size=args.cup_eval_batch_size,
            num_sampling_steps=args.num_sampling_steps,
            sample_seed=260808,
            selection_radius=0.06,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("eval_only", "action_expert"), required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=INTEGRATED_READER_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--peak-lr", type=float, default=3e-5)
    parser.add_argument("--clip-gradient-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--cup-eval-interval", type=int, default=50)
    parser.add_argument("--cup-eval-episodes", type=int, default=24)
    parser.add_argument("--cup-eval-batch-size", type=int, default=6)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The ordinary trainer already performs an episode-held-out split.  This
    # hook only applies the frame-59 restriction within each side of that split.
    _trainer._filter_memory_classifier_frame_range = _frame59_only  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
