"""Jointly align frozen visual MEM and Pi's action expert at frame 59.

The adapter-only control showed that lowering flow loss in the small memory
conditioner is insufficient to change target-cup selection.  This follow-up
keeps the visual encoder and external recurrent memory frozen, but lets the
memory conditioner, Pi action expert, and action projections co-adapt on the
true first post-observation action chunk.  It uses no cup labels or auxiliary
classifier: the only supervision is the ordinary action flow-matching loss.
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
    "pi0_shellgame_qwen_distilled_memory_action_frame59_adapter_eef7_260826/"
    "direct_visual_frame59_adapter_only500_4gpu_260826/300/params"
)

# Re-export the exact same causal row filter so the adapter-only and joint
# experiments differ only in which action-side parameters are trainable.
filter_frame59_correct_indices = _adapter.filter_frame59_correct_indices


def make_train_config(
    *,
    config_module,
    exp_name: str,
    memory_path: pathlib.Path = DEFAULT_MEMORY,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    steps: int = 300,
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
        semantic_residual_dropout_rate=0.0,
    )
    return config_module.TrainConfig(
        name="pi0_shellgame_qwen_distilled_memory_action_frame59_joint_eef7_260826",
        exp_name=exp_name,
        model=model,
        # Train only the external-memory conditioner, Pi action expert, and
        # action/time projections.  Current-image encoders and MEM stay fixed.
        freeze_filter=model.get_freeze_filter_action_finetune(),
        data=config_module.MultiDataConfigFactory(
            state_pad_dim=96,
            datasets=[_v10._data_config(config_module, _v10.NOMINAL_ROOT, memory_path)],  # noqa: SLF001
            weights=[1.0],
            use_merged_norm_stats=False,
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(
            str(init_checkpoint)
        ),
        # The action expert is much larger than the adapter, so retain the
        # proven V10 fine-tuning scale instead of the adapter-only 3e-5 LR.
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
        save_interval=100 if steps <= 500 else 500,
        keep_period=100 if steps <= 500 else 500,
        val_ratio=0.1,
        eval_interval=50 if steps <= 500 else 250,
        eval_batches=30,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=config_module.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=config_module.ShellgameCupEvalConfig(enabled=False),
    )
