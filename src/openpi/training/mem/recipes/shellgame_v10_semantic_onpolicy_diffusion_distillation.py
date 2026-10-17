"""Adapter-only V10 on-policy diffusion-trajectory distillation recipe."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

from examples.shellgame import v10_exact_semantic_onpolicy_diffusion_distillation as _distill
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_v10_exact_parallel_semantic_adapter as _base

DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_v10_semantic_velocity_distillation_eef7_260827/"
    "v10_exact_semantic_velocity_distill_from_token249_w1_250_b6_3gpu_260827/249/params"
)


def make_index_filter(memory_banks):
    return _base.make_index_filter(memory_banks)


def make_train_config(
    *,
    config_module: Any,
    exp_name: str,
    memory_banks=None,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    steps: int = 250,
    peak_lr: float = 1e-5,
    onpolicy_num_steps: int = 4,
    onpolicy_velocity_weight: float = 1.0,
    batch_size: int = 3,
    fsdp_devices: int = 3,
    num_workers: int = 8,
    checkpoint_period: int = 125,
    overwrite: bool = False,
):
    init_checkpoint = pathlib.Path(init_checkpoint).resolve()
    if not init_checkpoint.is_dir():
        raise FileNotFoundError(init_checkpoint)
    if checkpoint_period <= 0:
        raise ValueError("checkpoint_period must be positive")
    parent = _base.make_train_config(
        config_module=config_module,
        exp_name=exp_name,
        memory_banks=memory_banks,
        steps=steps,
        peak_lr=peak_lr,
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        num_workers=num_workers,
        overwrite=overwrite,
    )
    model = _distill.make_config_from_adapter(
        parent.model,
        onpolicy_num_steps=onpolicy_num_steps,
        onpolicy_velocity_weight=onpolicy_velocity_weight,
    )
    return dataclasses.replace(
        parent,
        name="pi0_shellgame_v10_semantic_onpolicy_diffusion_distillation_eef7_260827",
        model=model,
        weight_loader=_weight_loaders.CheckpointWeightLoader(str(init_checkpoint)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(25, max(steps - 1, 0)),
            peak_lr=peak_lr,
            decay_steps=max(steps, 2),
            decay_lr=peak_lr * 0.1,
        ),
        save_interval=checkpoint_period,
        keep_period=checkpoint_period,
        eval_interval=checkpoint_period,
    )
