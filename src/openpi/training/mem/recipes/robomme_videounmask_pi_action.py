"""Action-only Pi0.5 recipe for single-target RoboMME VideoUnmask.

The dataset stores one current front image, one current wrist image, a
9-dimensional state ``[eef_rpy6, gripper_width, target_y, target_x]``, and an
already aligned 16x7 absolute EEF action chunk.  Target coordinates are oracle
labels for this first-stage action gate.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import einops
import numpy as np

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
from openpi.tasks.robomme.videounmask import pi0_point_action as _point_action
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
from openpi.transforms import make_bool_mask

DEFAULT_DATASET_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/data/"
    "robomme_videounmask_lerobot_pi_action_train"
)
DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/"
    "absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/"
    "5999/params"
)


def _parse_video(value) -> np.ndarray:
    video = np.asarray(value)
    if np.issubdtype(video.dtype, np.floating):
        scale = 255.0 if video.size == 0 or float(np.nanmax(video)) <= 1.0 else 1.0
        video = np.clip(video * scale, 0, 255).astype(np.uint8)
    if video.ndim == 4 and video.shape[1] == 3:
        video = einops.rearrange(video, "t c h w -> t h w c")
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Expected video [T,H,W,3] or [T,3,H,W], got {video.shape}")
    return video


@dataclasses.dataclass(frozen=True)
class VideoUnmaskPointActionInputs(_transforms.DataTransformFn):
    num_frames: int = 1
    phase_conditioned: bool = False

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state_raw"], dtype=np.float32).reshape(-1)
        state_dim = 13 if self.phase_conditioned else 9
        if state.shape != (state_dim,):
            raise ValueError(f"Expected VideoUnmask state{state_dim}, got {state.shape}")
        front = _parse_video(data["front_rgb_video"])
        wrist = _parse_video(data["wrist_rgb_video"])
        expected = (self.num_frames, 224, 224, 3)
        if front.shape != expected or wrist.shape != expected:
            raise ValueError(
                f"Expected current videos {expected}, got front={front.shape}, wrist={wrist.shape}"
            )
        # Observation.episode_T is historically typed as a float value field;
        # the action model converts it back to int only for temporal masking.
        data["episode_T"] = np.float32(data["episode_T"])
        data["state"] = state
        data["image"] = {"base_rgb": front, "wrist_rgb": wrist}
        data["image_mask"] = {"base_rgb": np.True_, "wrist_rgb": np.True_}
        frame_mask = data.get("video_frame_valid_mask")
        if frame_mask is not None:
            data["frame_valid_mask"] = {
                "base_rgb": np.asarray(
                    frame_mask.get("front_rgb", np.ones(self.num_frames, dtype=np.bool_))
                ),
                "wrist_rgb": np.asarray(
                    frame_mask.get("wrist_rgb", np.ones(self.num_frames, dtype=np.bool_))
                ),
            }
        return data


@dataclasses.dataclass(frozen=True)
class _VideoUnmaskPointActionDataConfigMixin:
    num_frames: int = 1
    frame_stride: int = 1
    padding_mode: str = "repeat"
    num_future_frames: int = 0
    future_frame_stride: int = 1
    video_layout: str = "sliding"
    image_keys: tuple[str, ...] = ("front_rgb", "wrist_rgb")
    phase_conditioned: bool = False

    def video_frame_config(self):
        from openpi.training.mem.video_dataset import VideoFrameConfig

        return VideoFrameConfig(
            image_keys=self.image_keys,
            num_frames=self.num_frames,
            frame_stride=self.frame_stride,
            padding_mode=self.padding_mode,
            num_future_frames=self.num_future_frames,
            future_frame_stride=self.future_frame_stride,
            layout=self.video_layout,
        )

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig):
        config = super().create_base_config(assets_dirs, model_config)
        normalize_masks = {
            "actions": make_bool_mask(7),
            "state": make_bool_mask(13 if self.phase_conditioned else 9),
        }
        return dataclasses.replace(config, normalize_masks=normalize_masks)

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig):
        from openpi import transforms_video as _transforms_video

        per_frame_keys = {
            f"{key}_{frame}": f"{key}_{frame}"
            for key in self.image_keys
            for frame in range(self.num_frames)
        }
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "state_raw": "observation.state",
                        **per_frame_keys,
                        "video_frame_valid_mask": {
                            key: f"video_frame_valid_mask/{key}" for key in self.image_keys
                        },
                        "actions": "actions",
                        "prompt": "task",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                        "episode_T": "episode_T",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                _transforms_video.BuildVideoTensor(
                    image_keys=self.image_keys,
                    num_frames=self.num_frames,
                    output_keys={key: f"{key}_video" for key in self.image_keys},
                ),
                VideoUnmaskPointActionInputs(
                    num_frames=self.num_frames, phase_conditioned=self.phase_conditioned
                ),
            ]
        )
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=True,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
                _transforms.KeepModelKeys(),
            ],
            outputs=[
                _transforms.ChunkActions(target_dim=7),
                _transforms.DropKeys(keys=("state",)),
            ],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


_DATA_CONFIG_TYPES: dict[type, type] = {}


def data_config_type(config_module: Any | None = None) -> type:
    if config_module is None:
        from openpi.training import config as config_module

    base_type = config_module.DataConfigFactory
    if concrete := _DATA_CONFIG_TYPES.get(base_type):
        return concrete

    @dataclasses.dataclass(frozen=True)
    class VideoUnmaskPointActionDataConfig(_VideoUnmaskPointActionDataConfigMixin, base_type):
        pass

    VideoUnmaskPointActionDataConfig.__module__ = __name__
    VideoUnmaskPointActionDataConfig.__qualname__ = "VideoUnmaskPointActionDataConfig"
    _DATA_CONFIG_TYPES[base_type] = VideoUnmaskPointActionDataConfig
    return VideoUnmaskPointActionDataConfig


def make_model_config(
    *,
    target_point_relative_to_eef: bool = False,
    phase_goal_conditioner: bool = False,
) -> _point_action.Pi0VideoUnmaskPointActionConfig:
    return _point_action.Pi0VideoUnmaskPointActionConfig(
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
        target_point_state_start=7,
        target_point_hidden_width=256,
        target_point_relative_to_eef=target_point_relative_to_eef,
        phase_goal_conditioner=phase_goal_conditioner,
        gripper_loss_weight=4.0,
        real_action_dim=7,
        gripper_action_index=6,
    )


def make_train_config(
    *,
    config_module: Any,
    dataset_root: pathlib.Path = DEFAULT_DATASET_ROOT,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    exp_name: str,
    steps: int = 500,
    batch_size: int = 12,
    fsdp_devices: int = 6,
    num_workers: int = 8,
    peak_lr: float = 3e-5,
    warmup_steps: int = 50,
    eval_interval: int = 100,
    eval_batches: int = 10,
    save_interval: int = 250,
    overwrite: bool = False,
    phase_conditioned: bool = False,
    target_point_relative_to_eef: bool = False,
    phase_goal_conditioner: bool = False,
):
    model = make_model_config(
        target_point_relative_to_eef=target_point_relative_to_eef,
        phase_goal_conditioner=phase_goal_conditioner,
    )
    data_cls = data_config_type(config_module)
    root = dataset_root.expanduser().resolve()
    return config_module.TrainConfig(
        name="pi0_robomme_videounmask_point_action_260823",
        exp_name=exp_name,
        model=model,
        data=data_cls(
            repo_id=str(root),
            assets=config_module.AssetsConfig(asset_id=".", assets_dir=str(root)),
            base_config=config_module.UmiDataConfig(
                action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
                robot_type="ARM=1 G=0 H=0",
            ),
            phase_conditioned=phase_conditioned,
        ),
        freeze_filter=model.get_freeze_filter_action_finetune(),
        weight_loader=_weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(
            str(init_checkpoint.expanduser().resolve())
        ),
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
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        log_interval=10,
        save_interval=save_interval,
        keep_period=max(save_interval, 1),
        val_ratio=0.1,
        eval_interval=eval_interval,
        eval_batches=eval_batches,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=config_module.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=config_module.ShellgameCupEvalConfig(enabled=False),
    )
