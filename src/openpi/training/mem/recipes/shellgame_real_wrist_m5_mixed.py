"""M5 probe on the 25% old306 + 75% cup_0903 real-data mixture."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_real_mixed_common as _mixed
from openpi.training.mem.recipes import shellgame_real_wrist_m5 as _m5
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2

CONFIG_NAME = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5_mixed_cup0903"


def build_model_config(semantic_source: str) -> _m5.RealWristM5Config:
    model = _m5.build_model_config(semantic_source)
    return dataclasses.replace(
        model,
        oracle_final_cups=(_mixed.global_final_cups() if semantic_source == "oracle" else ()),
    )


def make_train_config(
    *,
    semantic_source: str,
    exp_name: str,
    checkpoint: str = str(_mixed.ADAPTED_MEMORY_CHECKPOINT),
    steps: int = 1_000,
    warmup_steps: int = 50,
    peak_lr: float = 3e-4,
    batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 16,
    fsdp_devices: int = 8,
    eval_interval: int = 50,
    eval_batches: int = 2,
    save_interval: int = 5_000,
    overwrite: bool = False,
) -> Any:
    from openpi.training import config as _config

    if batch_size <= 0 or batch_size % fsdp_devices:
        raise ValueError("batch_size must be positive and divisible by fsdp_devices")
    model = build_model_config(semantic_source)
    data_type = _stage2.data_config_type(_config)
    params_path = Path(checkpoint)
    if params_path.name != "params":
        params_path /= "params"
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_m5(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=_mixed.sampler_weights(decision_frame=_stage2.CURRENT_START_FRAME),
            norm_weights=list(_mixed.SOURCE_PROBABILITIES),
            datasets=_mixed.make_dataset_factories(
                _config,
                data_type,
                min_frame=_stage2.CURRENT_START_FRAME,
                max_frame=_stage2.CURRENT_START_FRAME,
            ),
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderReinitialize(
            params_path=str(params_path),
            reinitialize_regex=r".*HistorySemanticJointActionReadout.*",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(warmup_steps, max(steps - 1, 0)),
            peak_lr=peak_lr,
            decay_steps=max(steps, 2),
            decay_lr=peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=10.0),
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
        val_ratio=0.1,
        eval_interval=min(eval_interval, steps),
        eval_batches=eval_batches,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(enabled=False),
    )
