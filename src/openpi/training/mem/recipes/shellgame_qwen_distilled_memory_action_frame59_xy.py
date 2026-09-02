"""Frame-59 planar EEF alignment for frozen recurrent visual memory.

This diagnostic stage uses ordinary action supervision only, but restricts the
flow-matching objective to absolute EEF X/Y.  It tests whether the compact
memory can drive spatial target selection once rotation, height, and gripper
losses no longer dilute the two dimensions that distinguish tabletop slots.
The memory network and current-image encoder remain frozen.
"""

from __future__ import annotations

import dataclasses
import pathlib

from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_frame59_adapter as _adapter
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_v10 as _v10
from openpi.training.mem.recipes import shellgame_qwen_event_memory_action as _model_base


DEFAULT_MEMORY = _adapter.DEFAULT_MEMORY
DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_distilled_memory_action_frame59_joint_eef7_260826/"
    "direct_visual_frame59_joint_continue2000_4gpu_260826/500/params"
)
XY_MASK = (1.0, 1.0) + (0.0,) * 30

filter_frame59_correct_indices = _adapter.filter_frame59_correct_indices


def make_train_config(
    *,
    config_module,
    exp_name: str,
    memory_path: pathlib.Path = DEFAULT_MEMORY,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    steps: int = 500,
    batch_size: int = 12,
    fsdp_devices: int = 4,
    num_workers: int = 8,
    overwrite: bool = False,
):
    memory_path = memory_path.expanduser().resolve()
    init_checkpoint = init_checkpoint.expanduser().resolve()
    if not memory_path.is_file():
        raise FileNotFoundError(memory_path)
    if not init_checkpoint.is_dir():
        raise FileNotFoundError(init_checkpoint)

    model = dataclasses.replace(
        _model_base.make_model_config(),
        action_loss_mask=XY_MASK,
        semantic_residual_dropout_rate=0.0,
    )
    data = _v10._data_config(config_module, _v10.NOMINAL_ROOT, memory_path)  # noqa: SLF001
    data = dataclasses.replace(
        data,
        base_config=config_module.UmiDataConfig(
            action_loss_mask=XY_MASK,
            robot_type="ARM=1 G=0 H=0",
        ),
    )
    return config_module.TrainConfig(
        name="pi0_shellgame_qwen_distilled_memory_action_frame59_xy_eef7_260826",
        exp_name=exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_action_finetune(),
        data=config_module.MultiDataConfigFactory(
            state_pad_dim=96,
            datasets=[data],
            weights=[1.0],
            use_merged_norm_stats=False,
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(
            str(init_checkpoint)
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(50, max(steps - 1, 0)),
            peak_lr=3e-6,
            decay_steps=max(steps, 2),
            decay_lr=3e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        log_interval=10,
        save_interval=250,
        keep_period=250,
        val_ratio=0.1,
        eval_interval=125,
        eval_batches=30,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=config_module.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=config_module.ShellgameCupEvalConfig(enabled=False),
    )
