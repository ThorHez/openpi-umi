"""Train a fresh real-ShellGame MEM on all episodes with mild augmentation.

No weights are loaded from ``/4999`` or any later MEM checkpoint.  A standard
Pi0.5 action-policy checkpoint supplies only the generic backbone/container;
every ShellGame-specific memory parameter is randomly reinitialized.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import flax.nnx as nnx

import openpi.shared.nnx_utils as nnx_utils
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders

GENERIC_PI05_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi/.cache/openpi/openpi-assets/"
    "checkpoints/pi05_base/params"
)
CONFIG_NAME = "pi0_mem_shellgame_real_fresh_memory_mild_all"


def make_train_config(config_module: Any | None = None):
    if config_module is None:
        from openpi.training import config as config_module

    base = _stage2.make_train_config(config_module)
    shellgame_memory = nnx_utils.PathRegex(
        r".*(HistoryFrame0InitialCupClassifier|"
        r"HistoryThreeSwapVisualRelationMemoryTracker).*"
    )
    return dataclasses.replace(
        base,
        name=CONFIG_NAME,
        freeze_filter=nnx.Not(shellgame_memory),
        weight_loader=_weight_loaders.CheckpointWeightLoaderReinitialize(
            GENERIC_PI05_CHECKPOINT,
            reinitialize_regex=(
                r".*(HistoryResampler_0|HistoryLayerNorm_0|"
                r"HistoryMultiHeadDotProductAttention_0|HistoryOutProj|"
                r"history_memory_gate_logit|HistoryFrame0InitialCupClassifier|"
                r"HistoryThreeSwapVisualRelationMemoryTracker|"
                r"HistorySemanticJointActionReadout|HistoryRawMemoryQueryResampler|"
                r"ActionMemoryCrossAttention).*"
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=3e-4,
            decay_steps=3_000,
            decay_lr=3e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=3_000,
        batch_size=32,
        num_workers=0,
        fsdp_devices=2,
        log_interval=10,
        save_interval=5_000,
        keep_period=5_000,
        eval_interval=100,
        eval_batches=1,
        wandb_enabled=False,
    )
