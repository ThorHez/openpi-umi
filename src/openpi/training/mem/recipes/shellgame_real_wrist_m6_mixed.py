"""Full gradual-suffix M6 training on 25% old306 + 75% cup_0903."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_real_mixed_common as _mixed
from openpi.training.mem.recipes import shellgame_real_wrist_m6 as _m6
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2

CONFIG_NAME = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_mixed_cup0903"


def make_train_config(
    *,
    exp_name: str,
    checkpoint: str,
    steps: int = 10_000,
    warmup_steps: int = 300,
    peak_lr: float = 2e-5,
    batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 16,
    fsdp_devices: int = 8,
    eval_interval: int = 250,
    eval_batches: int = 8,
    save_interval: int = 5_000,
    resume: bool = False,
    overwrite: bool = False,
) -> Any:
    from openpi.training import config as _config

    if batch_size <= 0 or batch_size % fsdp_devices:
        raise ValueError("batch_size must be positive and divisible by fsdp_devices")
    params_path = Path(checkpoint)
    if params_path.name != "params":
        params_path /= "params"
    model = _stage2.MODEL_CONFIG
    data_type = _m6.data_config_type(_config)
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_memory_interface_finetune(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=_mixed.sampler_weights(decision_frame=None),
            norm_weights=list(_mixed.SOURCE_PROBABILITIES),
            datasets=_mixed.make_dataset_factories(
                _config,
                data_type,
                min_frame=_stage2.CURRENT_START_FRAME,
                max_frame=None,
            ),
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoader(str(params_path)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(warmup_steps, max(steps - 1, 0)),
            peak_lr=peak_lr,
            decay_steps=max(steps, 2),
            decay_lr=peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=steps,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        seed=42,
        log_interval=10,
        save_interval=min(save_interval, steps),
        keep_period=steps,
        resume=resume,
        val_ratio=0.1,
        eval_interval=min(eval_interval, steps),
        eval_batches=eval_batches,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(enabled=False),
    )
