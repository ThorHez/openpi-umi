"""Direction-preserving full-suffix M6 training on old306 + cup_0903.

The flow objective sees the complete gradual-action suffix.  A duplicated,
balanced 241..245 anchor stream keeps the direction objective frequent without
forcing lateral cup motion during the later descend/grasp frames.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_real_mixed_common as _mixed
from openpi.training.mem.recipes import shellgame_real_wrist_m6 as _m6
from openpi.training.mem.recipes import shellgame_real_wrist_m6_direction_stage1_mixed as _stage1_mixed
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2


CONFIG_NAME = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_full_mixed_cup0903"
DIRECTION_FRAME_START = _stage2.CURRENT_START_FRAME
DIRECTION_FRAME_END = DIRECTION_FRAME_START + 4


def _sampler_weights(anchor_fraction: float) -> list[float]:
    if not 0.0 < anchor_fraction < 1.0:
        raise ValueError("anchor_fraction must be strictly between zero and one")
    full = _mixed.sampler_weights(decision_frame=None)
    one_frame_counts = _mixed.balanced_source_row_counts(decision_frame=DIRECTION_FRAME_START)
    anchor_frames = DIRECTION_FRAME_END - DIRECTION_FRAME_START + 1
    anchor = [
        probability / (rows * anchor_frames)
        for probability, rows in zip(_mixed.SOURCE_PROBABILITIES, one_frame_counts, strict=True)
    ]
    return [*(value * (1.0 - anchor_fraction) for value in full), *(value * anchor_fraction for value in anchor)]


def make_train_config(
    *,
    exp_name: str,
    checkpoint: str,
    steps: int = 10_000,
    warmup_steps: int = 300,
    peak_lr: float = 1e-5,
    batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 16,
    fsdp_devices: int = 8,
    eval_interval: int = 250,
    eval_batches: int = 8,
    direction_loss_weight: float = 0.1,
    direction_temperature: float = 5e-4,
    anchor_fraction: float = 0.5,
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
    model = _stage1_mixed.build_model_config(
        direction_loss_weight,
        direction_temperature,
        direction_frame_start=DIRECTION_FRAME_START,
        direction_frame_end=DIRECTION_FRAME_END,
    )
    model = dataclasses.replace(model, direction_early_stop_metric="")
    data_type = _m6.data_config_type(_config)
    full_datasets = _mixed.make_dataset_factories(
        _config,
        data_type,
        min_frame=DIRECTION_FRAME_START,
        max_frame=None,
    )
    anchor_datasets = _mixed.make_dataset_factories(
        _config,
        data_type,
        min_frame=DIRECTION_FRAME_START,
        max_frame=DIRECTION_FRAME_END,
    )
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_memory_interface_finetune(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=_sampler_weights(anchor_fraction),
            norm_weights=[
                *(probability * (1.0 - anchor_fraction) for probability in _mixed.SOURCE_PROBABILITIES),
                *(probability * anchor_fraction for probability in _mixed.SOURCE_PROBABILITIES),
            ],
            datasets=[*full_datasets, *anchor_datasets],
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
