"""Train absolute EEF7 control while preserving the validated old tracker.

This is the absolute-EEF counterpart of ``train_old_tracker_full_joint_grasp``.
The memory tracker consumes the exact raw frames 0..59, frame ``t`` is appended
as the current observation, and selection/approach/descend/grasp/lift rows are
sampled in a deterministic 20/20/20/20/20 mixture.  The validated tracker,
query resampler, and memory cross-attention are frozen; only Pi0.5's action
expert and action/time projections learn the new absolute EEF7 semantics.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.shellgame import train_old_tracker_full_joint_grasp as _joint
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from scripts.mem import train_pi0_mem_compress as _trainer

LEROBOT_ABSOLUTE_EEF7_ROOT = (
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_absolute_eef_raw7"
)
CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_260812"


def build_config(args):
    """Build the proven full-action recipe with only action semantics changed."""
    # Construct the validated architecture directly.  This intentionally does
    # not rebuild the old classification-probe label tables: they are not used
    # by action training, and the large raw joint-image source may be archived.
    model = _joint.OldTrackerFullJointGraspConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        max_token_len=256,
        num_frames=_joint.TOTAL_INPUT_FRAMES,
        memory_every=0,
        current_frame_index=-1,
        history_memory_tokens=1,
        history_resampler_depth=1,
        history_use_current_condition=False,
        history_gate_fixed=0.0,
        diversity_weight=0.0,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=0,
        encoder_width=args.encoder_width,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        memory_width=args.memory_width,
        memory_depth=args.memory_depth,
        memory_heads=args.memory_heads,
        adapter_heads=args.adapter_heads,
        endpoint_memory_tokens=args.memory_tokens,
        adapter_current_tokens=args.current_tokens,
        adapter_residual_scale=args.residual_scale,
        video_mode=args.video_mode,
        initial_mode=args.initial_mode,
        relation_mode=args.relation_mode,
        raw_memory_mode=args.raw_memory_mode,
        query_tokens=args.query_tokens,
        query_width=args.query_width,
        query_depth=args.query_depth,
        query_heads=args.query_heads,
        action_cross_attention_heads=args.action_cross_attention_heads,
        gripper_loss_weight=args.gripper_loss_weight,
        real_action_dim=7,
        gripper_action_index=6,
        last_episode_frame=_joint.LAST_EPISODE_FRAME,
    )
    data = _config.MultiDataConfigFactory(
        state_pad_dim=96,
        weights=[1.0],
        datasets=[
            _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_AbsoluteEEF7(
                repo_id=LEROBOT_ABSOLUTE_EEF7_ROOT,
                assets=_config.AssetsConfig(
                    asset_id=".",
                    assets_dir=LEROBOT_ABSOLUTE_EEF7_ROOT,
                ),
                base_config=_config.UmiDataConfig(
                    action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
                    robot_type="ARM=1 G=0 H=0",
                ),
                num_frames=_joint.TOTAL_INPUT_FRAMES,
                frame_stride=1,
            )
        ],
    )
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=args.exp_name,
        model=model,
        data=data,
        freeze_filter=model.get_freeze_filter_full_action(),
        weight_loader=weight_loaders.CheckpointWeightLoader(args.init_checkpoint),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 2),
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        val_ratio=0.1,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        wandb_enabled=False,
        overwrite=args.overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(enabled=False),
    )


def main() -> None:
    args = _joint.parse_args()
    # Keep the old tracker's temporal input invariant: raw 0..59 + current t.
    _joint._config_pi0_mem.VideoFrameDataset = (  # noqa: SLF001
        _joint.FixedPrefixCurrentVideoDataset
    )
    _trainer._filter_memory_classifier_frame_range = (  # noqa: SLF001
        _joint._balanced_full_action_indices  # noqa: SLF001
    )
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
