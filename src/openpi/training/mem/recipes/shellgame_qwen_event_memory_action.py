"""Absolute EEF7 action recipe with frozen Qwen-driven recurrent memory."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

from openpi.tasks.shellgame import pi0_qwen_event_memory_action as _policy
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_semantic_action as _base
import openpi.transforms as _transforms


DEFAULT_MEMORY_BANK = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/artifacts/"
    "shellgame_qwen_event_final_memory_v1_260825.npz"
)
DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/"
    "absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820/"
    "1000/params"
)
DEFAULT_DATASET_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7"
)


@dataclasses.dataclass(frozen=True)
class FrozenShellGameMemoryLookup:
    templates: np.ndarray
    episode_template_index: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: pathlib.Path):
        with np.load(path, allow_pickle=False) as source:
            templates = np.asarray(source["memory_templates"], dtype=np.float32)
            indices = np.asarray(source["episode_template_index"], dtype=np.int32)
            metadata = json.loads(str(np.asarray(source["metadata_json"]).reshape(())))
        if templates.ndim != 3 or templates.shape[1:] != (128, 64):
            raise ValueError(f"Expected memory templates [N,128,64], got {templates.shape}")
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(templates)):
            raise ValueError("Invalid episode_template_index")
        return cls(templates, indices, metadata)

    def at(self, episode_index: int) -> np.ndarray:
        return self.templates[self.episode_template_index[int(episode_index)]]


@dataclasses.dataclass(frozen=True)
class InjectFrozenShellGameMemory(_transforms.DataTransformFn):
    lookup: FrozenShellGameMemoryLookup

    def __call__(self, data: dict) -> dict:
        episode = int(np.asarray(data["episode_index"]).reshape(()))
        data["semantic_memory"] = self.lookup.at(episode)
        return data


_DATA_CONFIG_TYPES: dict[type, type] = {}


def data_config_type(config_module: Any | None = None) -> type:
    if config_module is None:
        from openpi.training import config as config_module
    base_type = config_module.DataConfigFactory
    if concrete := _DATA_CONFIG_TYPES.get(base_type):
        return concrete
    base_shellgame = _base.data_config_type(config_module)

    @dataclasses.dataclass(frozen=True)
    class ShellGameQwenEventMemoryActionDataConfig(base_shellgame):
        memory_path: str = str(DEFAULT_MEMORY_BANK)

        def create(self, assets_dirs, model_config):
            config = super().create(assets_dirs, model_config)
            lookup = FrozenShellGameMemoryLookup.load(pathlib.Path(self.memory_path).expanduser().resolve())
            return dataclasses.replace(
                config,
                data_transforms=config.data_transforms.push(
                    inputs=[InjectFrozenShellGameMemory(lookup)]
                ),
                model_transforms=config.model_transforms.push(
                    inputs=[_transforms.KeepModelKeys()]
                ),
            )

    ShellGameQwenEventMemoryActionDataConfig.__module__ = __name__
    ShellGameQwenEventMemoryActionDataConfig.__qualname__ = "ShellGameQwenEventMemoryActionDataConfig"
    _DATA_CONFIG_TYPES[base_type] = ShellGameQwenEventMemoryActionDataConfig
    return ShellGameQwenEventMemoryActionDataConfig


def make_model_config() -> _policy.Pi0QwenEventMemoryActionConfig:
    return _policy.Pi0QwenEventMemoryActionConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        max_token_len=256,
        num_frames=1,
        current_frame_index=0,
        memory_every=0,
        history_memory_tokens=1,
        history_resampler_depth=1,
        history_use_current_condition=False,
        history_gate_fixed=0.0,
        diversity_weight=0.0,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=0,
        semantic_memory_tokens=128,
        semantic_memory_width=64,
        semantic_query_tokens=8,
        semantic_hidden_width=256,
        gripper_loss_weight=4.0,
        real_action_dim=7,
        gripper_action_index=6,
        semantic_residual_gate_init=1.0,
        semantic_residual_dropout_rate=0.1,
        last_episode_frame=154,
    )


def make_train_config(
    *,
    config_module: Any,
    exp_name: str,
    memory_path: pathlib.Path = DEFAULT_MEMORY_BANK,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    steps: int = 250,
    batch_size: int = 12,
    fsdp_devices: int = 6,
    num_workers: int = 8,
    overwrite: bool = False,
):
    model = make_model_config()
    data_cls = data_config_type(config_module)
    root = DEFAULT_DATASET_ROOT.resolve()
    return config_module.TrainConfig(
        name="pi0_shellgame_qwen_event_memory_action_eef7_260825",
        exp_name=exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_action_finetune(),
        data=config_module.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                data_cls(
                    repo_id=str(root),
                    memory_path=str(memory_path.resolve()),
                    assets=config_module.AssetsConfig(asset_id=".", assets_dir=str(root)),
                    base_config=config_module.UmiDataConfig(
                        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=1,
                    frame_stride=1,
                    video_layout="sliding",
                    min_frame_index=59,
                    max_frame_index=153,
                )
            ],
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(str(init_checkpoint.resolve())),
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
        # This is a short interface proof; retain only the final checkpoint to
        # protect the nearly-full /data2 volume.
        save_interval=250,
        keep_period=250,
        val_ratio=0.1,
        eval_interval=50,
        eval_batches=10,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=config_module.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=config_module.ShellgameCupEvalConfig(enabled=False),
    )
