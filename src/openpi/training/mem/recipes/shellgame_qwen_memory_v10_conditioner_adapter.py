"""Adapt a minimal semantic conditioner behind the frozen V10 action expert.

Stage 1 adapted the transplanted V10 action branch to a fixed Qwen-memory
conditioner.  This second stage freezes that complete action expert again and
updates only the final semantic-to-action latent mapping:

* ``SemanticMemoryActionConditioner/query_out``;
* ``SemanticMemoryActionConditioner/action_cross_attention/out``; and
* ``SemanticMemoryActionConditioner/gate_delta``.

The semantic memory reader, query extraction, continuous waypoint decoder and
hard XY anchor stay fixed, so target selection cannot be relearned as a
ShellGame-specific shortcut.
"""

from __future__ import annotations

import dataclasses
import pathlib

import flax.nnx as nnx

from openpi.shared import nnx_utils
from openpi.training import optimizer as _optimizer
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_waypoint_grasp_v6 as _grasp

DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_memory_waypoint_v10_action_adapt_eef7_260827/"
    "v10_action_qwen_bridge_adapt_v6mix_1k_lr1e6_b12_6gpu_260827/999/params"
)


def make_train_config(
    *,
    config_module,
    exp_name: str,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    steps: int = 500,
    peak_lr: float = 1e-5,
    batch_size: int = 12,
    fsdp_devices: int = 6,
    num_workers: int = 8,
    overwrite: bool = False,
):
    init_checkpoint = init_checkpoint.expanduser().resolve()
    if not init_checkpoint.is_dir():
        raise FileNotFoundError(init_checkpoint)
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if peak_lr <= 0:
        raise ValueError("peak_lr must be positive")

    parent = _grasp.make_train_config(
        config_module=config_module,
        exp_name=exp_name,
        init_checkpoint=init_checkpoint,
        steps=steps,
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        num_workers=num_workers,
        overwrite=overwrite,
    )
    query_out = nnx_utils.PathRegex(r".*SemanticMemoryActionConditioner/query_out/.*")
    attention_out = nnx_utils.PathRegex(r".*SemanticMemoryActionConditioner/action_cross_attention/out/.*")
    residual_gate = nnx_utils.PathRegex(r".*SemanticMemoryActionConditioner/gate_delta.*")
    adapter = nnx.Any(query_out, attention_out, residual_gate)
    return dataclasses.replace(
        parent,
        name="pi0_shellgame_qwen_memory_v10_conditioner_adapter_eef7_260827",
        freeze_filter=nnx.Not(adapter),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(50, max(steps - 1, 0)),
            peak_lr=peak_lr,
            decay_steps=max(steps, 2),
            decay_lr=peak_lr * 0.1,
        ),
        save_interval=125,
        keep_period=125,
        eval_interval=125,
        eval_batches=20,
    )


make_index_filter = _grasp.make_index_filter
