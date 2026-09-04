"""New-environment-dominant adaptation of the real ShellGame visual MEM.

Only the pretrained swap-relation classifier is optimized.  The vision
backbone, frame-0 cup classifier, recurrent updater, readout, and every action
module remain frozen so the resulting relation head can be transplanted into
the validated M6/H16 action policy without changing its memory interface.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import flax.nnx as nnx

import openpi.shared.nnx_utils as nnx_utils
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders

SOURCE_MEMORY_CHECKPOINT = "/data2/hzl_workspace_for_pi_mem/4999/params"
CONFIG_NAME = "pi0_mem_shellgame_real_relation_adapt_new75_old25"


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
        # TrainConfig.freeze_filter denotes frozen leaves.  The double
        # complement in TrainConfig.trainable_filter therefore selects only
        # this exact classifier subtree.
        freeze_filter=nnx.Not(relation_classifier),
        weight_loader=_weight_loaders.CheckpointWeightLoader(SOURCE_MEMORY_CHECKPOINT),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=50,
            peak_lr=1e-5,
            decay_steps=1_500,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=1_500,
        batch_size=32,
        num_workers=0,
        fsdp_devices=8,
        log_interval=10,
        # The dedicated trainer keeps small best-head snapshots in memory and
        # writes one full checkpoint at the selected step.  Keep the global
        # interval at 5000 to respect the workspace disk budget.
        save_interval=5_000,
        keep_period=5_000,
        val_ratio=0.15,
        eval_interval=100,
        eval_batches=1,
        wandb_enabled=False,
    )
