"""Train a continuous memory-to-waypoint bridge on causal frame-59 rows.

This recipe deliberately adds no ShellGame class or relation labels.  The
auxiliary target is the selected coordinate of the ordinary future action
chunk, so the same bridge applies to any absolute-EEF task with a continuous
spatial goal.
"""

from __future__ import annotations

import dataclasses
import pathlib

import flax.nnx as nnx

from openpi.shared import nnx_utils
from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_action as _waypoint_model
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_frame59_adapter as _adapter
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_v10 as _v10
from openpi.training.mem.recipes import shellgame_qwen_event_memory_action as _base


DEFAULT_MEMORY = _adapter.DEFAULT_MEMORY
DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_distilled_memory_action_frame59_joint_eef7_260826/"
    "direct_visual_frame59_joint_continue2000_4gpu_260826/500/params"
)

filter_frame59_correct_indices = _adapter.filter_frame59_correct_indices


def make_model_config() -> _waypoint_model.Pi0QwenEventMemoryWaypointActionConfig:
    base = dataclasses.replace(_base.make_model_config(), semantic_residual_dropout_rate=0.0)
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    return _waypoint_model.Pi0QwenEventMemoryWaypointActionConfig(
        **values,
        waypoint_action_dims=(0, 1),
        waypoint_action_index=0,
        waypoint_aux_weight=1.0,
        waypoint_injection_scale=1.0,
        waypoint_anchor_strength=1.0,
    )


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

    model = make_model_config()
    memory_bridge = nnx_utils.PathRegex(r".*SemanticMemoryActionConditioner.*")
    return config_module.TrainConfig(
        name="pi0_shellgame_qwen_distilled_memory_action_frame59_waypoint_eef7_260826",
        exp_name=exp_name,
        model=model,
        # Keep the validated visual memory and Pi expert fixed.  Updating the
        # compact conditioner plus new waypoint leaves is only ~5M parameters.
        freeze_filter=nnx.Not(memory_bridge),
        data=config_module.MultiDataConfigFactory(
            state_pad_dim=96,
            datasets=[_v10._data_config(config_module, _v10.NOMINAL_ROOT, memory_path)],  # noqa: SLF001
            weights=[1.0],
            use_merged_norm_stats=False,
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(
            str(init_checkpoint)
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(50, max(steps - 1, 0)),
            peak_lr=3e-5,
            decay_steps=max(steps, 2),
            decay_lr=3e-6,
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
        eval_interval=50,
        eval_batches=30,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=config_module.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=config_module.ShellgameCupEvalConfig(enabled=False),
    )
