"""V2 waypoint bridge with a zero-init memory mean/std correction path."""

from __future__ import annotations

import dataclasses
import pathlib

import flax.nnx as nnx

from openpi.shared import nnx_utils
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_frame59_waypoint as _v1
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_v10 as _v10


DEFAULT_MEMORY = _v1.DEFAULT_MEMORY
DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_distilled_memory_action_frame59_waypoint_eef7_260826/"
    "direct_visual_frame59_waypoint500_4gpu_260826/499/params"
)

filter_frame59_correct_indices = _v1.filter_frame59_correct_indices


def make_model_config():
    return dataclasses.replace(
        _v1.make_model_config(),
        waypoint_use_memory_statistics=True,
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
        name="pi0_shellgame_qwen_distilled_memory_action_frame59_waypoint_stats_eef7_260826",
        exp_name=exp_name,
        model=model,
        freeze_filter=nnx.Not(memory_bridge),
        data=config_module.MultiDataConfigFactory(
            state_pad_dim=96,
            datasets=[_v10._data_config(config_module, _v10.NOMINAL_ROOT, memory_path)],  # noqa: SLF001
            weights=[1.0],
            use_merged_norm_stats=False,
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(str(init_checkpoint)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(30, max(steps - 1, 0)),
            peak_lr=1e-5,
            decay_steps=max(steps, 2),
            decay_lr=1e-6,
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
