"""ShellGame recipe for semantic visual memory and EEF action prediction.

This module owns all ShellGame-specific data contracts, action semantics,
hyperparameters, and checkpoint choices.  The reusable memory components live
under :mod:`openpi.models`; the ShellGame policy adapter lives under
:mod:`openpi.tasks.shellgame`.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, ClassVar

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
from openpi.tasks.shellgame import pi0_mem_semantic_action as _shellgame_model
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
from openpi.transforms import make_bool_mask


@dataclasses.dataclass(frozen=True)
class _ShellGameAbsoluteEEF7DataConfigMixin:
    """Fixed-history ShellGame data with native 7-D absolute EEF actions."""

    num_frames: int = 16
    frame_stride: int = 10
    padding_mode: str = "repeat"
    num_future_frames: int = 0
    future_frame_stride: int = 1
    video_layout: str = "sliding"
    fixed_prefix_frames: int = 0
    min_frame_index: int | None = None
    max_frame_index: int | None = None
    tokenize_prompt: bool = True
    image_keys: tuple[str, ...] = (
        "left_wrist_0_rgb_0",
        "left_wrist_0_rgb_1",
    )

    normalize_masks: ClassVar[dict[str, tuple[bool, ...]]] = {
        "actions": make_bool_mask(7),
        "state": make_bool_mask(10),
    }

    @property
    def total_frames(self) -> int:
        return self.num_frames + self.num_future_frames

    def video_frame_config(self):
        """Return the sampling contract consumed by the memory data loader."""
        from openpi.training.mem.video_dataset import VideoFrameConfig

        return VideoFrameConfig(
            image_keys=tuple(self.image_keys),
            num_frames=self.num_frames,
            frame_stride=self.frame_stride,
            padding_mode=self.padding_mode,
            num_future_frames=self.num_future_frames,
            future_frame_stride=self.future_frame_stride,
            layout=self.video_layout,
            fixed_prefix_frames=self.fixed_prefix_frames,
            min_frame_index=self.min_frame_index,
            max_frame_index=self.max_frame_index,
        )

    def create_base_config(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> Any:
        config = super().create_base_config(assets_dirs, model_config)
        return dataclasses.replace(config, normalize_masks=self.normalize_masks)

    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> Any:
        from openpi import transforms_video as _transforms_video
        import openpi.training.config_pi0_mem as _config_pi0_mem

        per_frame_keys = {
            f"{key}_{t}": f"{key}_{t}"
            for key in self.image_keys
            for t in range(self.total_frames)
        }
        frame_valid_mask_keys = {
            "video_frame_valid_mask": {
                key: f"video_frame_valid_mask/{key}" for key in self.image_keys
            }
        }
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "observation.robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        **per_frame_keys,
                        **frame_valid_mask_keys,
                        "actions": "actions",
                        "prompt": "task",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                _transforms_video.BuildVideoTensor(
                    image_keys=tuple(self.image_keys),
                    num_frames=self.total_frames,
                    output_keys={key: f"{key}_video" for key in self.image_keys},
                ),
                _config_pi0_mem.UmiInputsV4_Shellgame_Video(num_frames=self.total_frames),
            ],
        )
        model_inputs = []
        if self.tokenize_prompt:
            model_inputs.extend(
                [
                    _transforms.InjectDefaultPrompt(None),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=(
                            model_config.discrete_state_input
                            if hasattr(model_config, "discrete_state_input")
                            else False
                        ),
                    ),
                ]
            )
        else:
            # The standalone memory objective does not consume language.  Do
            # not leave the raw string in the batch because JAX cannot shard
            # Unicode arrays.
            model_inputs.append(_transforms.DropKeys(keys=("prompt",)))
        model_inputs.extend(
            [
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ]
        )
        model_transforms = _transforms.Group(
            inputs=model_inputs,
            outputs=[
                _transforms.ChunkActions(target_dim=7),
                _transforms.DropKeys(keys=("state",)),
            ],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


_DATA_CONFIG_TYPES: dict[type, type] = {}


def data_config_type(config_module: Any | None = None) -> type:
    """Create the concrete data-config type without a module import cycle."""
    if config_module is None:
        from openpi.training import config as config_module

    base_type = config_module.DataConfigFactory
    if concrete_type := _DATA_CONFIG_TYPES.get(base_type):
        return concrete_type

    @dataclasses.dataclass(frozen=True)
    class ShellGameAbsoluteEEF7DataConfig(
        _ShellGameAbsoluteEEF7DataConfigMixin,
        base_type,
    ):
        pass

    ShellGameAbsoluteEEF7DataConfig.__module__ = __name__
    ShellGameAbsoluteEEF7DataConfig.__qualname__ = "ShellGameAbsoluteEEF7DataConfig"
    _DATA_CONFIG_TYPES[base_type] = ShellGameAbsoluteEEF7DataConfig
    return ShellGameAbsoluteEEF7DataConfig


MODEL_CONFIG = _shellgame_model.Pi0MemSemanticActionConfig(
    pi05=True,
    action_dim=32,
    action_horizon=16,
    action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
    max_token_len=256,
    num_frames=61,
    current_frame_index=-1,
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
    history_frames=60,
    encoder_width=256,
    encoder_depth=2,
    encoder_heads=8,
    semantic_memory_width=64,
    semantic_memory_depth=2,
    semantic_memory_heads=4,
    semantic_memory_tokens=128,
    diagnostic_current_tokens=256,
    diagnostic_adapter_heads=4,
    diagnostic_residual_scale=1.0,
    video_mode="normal",
    initial_mode="normal",
    relation_mode="one_hot",
    raw_memory_mode="normal",
    query_tokens=16,
    query_width=256,
    query_depth=2,
    query_heads=4,
    action_cross_attention_heads=8,
    gripper_loss_weight=4.0,
    real_action_dim=7,
    gripper_action_index=6,
    last_episode_frame=154,
)


def make_train_config(config_module: Any | None = None) -> Any:
    """Build the registered ShellGame semantic-memory training recipe."""
    if config_module is None:
        from openpi.training import config as config_module

    _config = config_module
    data_config_cls = data_config_type(_config)
    return _config.TrainConfig(
        name="pi0_mem_semantic_action_shellgame_eef7",
        model=MODEL_CONFIG,
        freeze_filter=MODEL_CONFIG.get_freeze_filter_action_finetune(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                data_config_cls(
                    repo_id=(
                        "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                        "shellgame_lerobot_absolute_eef_raw7"
                    ),
                    assets=_config.AssetsConfig(
                        asset_id=".",
                        assets_dir=(
                            "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                            "shellgame_lerobot_absolute_eef_raw7"
                        ),
                    ),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=61,
                    frame_stride=1,
                    video_layout="fixed_prefix_current",
                    fixed_prefix_frames=60,
                    min_frame_index=59,
                    max_frame_index=153,
                ),
            ],
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoader(
            "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
            "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/"
            "absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/"
            "5999/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=3e-5,
            decay_steps=6_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=6_000,
        batch_size=12,
        num_workers=8,
        fsdp_devices=6,
        log_interval=10,
        save_interval=500,
        keep_period=1_000,
        val_ratio=0.1,
        eval_interval=250,
        eval_batches=20,
        wandb_enabled=False,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(enabled=False),
    )
