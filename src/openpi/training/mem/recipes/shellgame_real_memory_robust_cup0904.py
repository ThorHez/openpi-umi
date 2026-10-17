"""Robust relation-head adaptation for the real ShellGame cup_0904 domain.

The already adapted cup_0903 memory is the initialization.  Only the visual
swap relation classifier is optimized; the image encoder, initial-cup head,
recurrent memory, action policy, and deployment action interface stay frozen.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import flax.nnx as nnx

import openpi.shared.nnx_utils as nnx_utils
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders

SOURCE_MEMORY_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_mem_shellgame_real_relation_adapt_new75_old25/"
    "cup0903_new75_old25_relation_only_lr1e5_b32_seed42_v1/500/params"
)
CONFIG_NAME = "pi0_mem_shellgame_real_relation_robust_cup0904"


def make_train_config(config_module: Any | None = None):
    if config_module is None:
        from openpi.training import config as config_module

    base = _stage2.make_train_config(config_module)
    relation_classifier = nnx_utils.PathRegex(
        r".*HistoryThreeSwapVisualRelationMemoryTracker/"
        r"swap_relation_classifier/.*"
    )
    return dataclasses.replace(
        base,
        name=CONFIG_NAME,
        freeze_filter=nnx.Not(relation_classifier),
        weight_loader=_weight_loaders.CheckpointWeightLoader(
            SOURCE_MEMORY_CHECKPOINT
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=50,
            peak_lr=5e-6,
            decay_steps=1_200,
            decay_lr=5e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=1_200,
        batch_size=32,
        num_workers=0,
        fsdp_devices=2,
        log_interval=10,
        save_interval=5_000,
        keep_period=5_000,
        val_ratio=0.15,
        eval_interval=100,
        eval_batches=1,
        wandb_enabled=False,
    )
