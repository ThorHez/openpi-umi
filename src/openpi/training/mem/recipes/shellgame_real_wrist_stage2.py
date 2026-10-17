"""Stage-2-only recipe for the 306-episode real-robot ShellGame dataset.

Contract summary:

* fixed history: episode frames 0..240 (241 images);
* current image: the dynamic action-time wrist frame;
* state: 10-D link6 pose/gripper relative to episode frame 0;
* action: 16x10 future link6 targets, all relative to the current frame;
* initialization: the matching 306-degap memory checkpoint at step 4999;
* trainable parameters: memory/action interface plus Pi0.5 action expert.

The action anchor exactly matches ``eval_arx5_pi_hzl.py``: every waypoint is
decoded as ``T_world_current @ T_action_relative``. The state intentionally
uses a different, episode-first anchor.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, ClassVar

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
from openpi.training.mem.recipes import shellgame_semantic_action as _base_recipe
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
from openpi.transforms import make_bool_mask

DATASET_ROOT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/data/"
    "shellgame_real_306_degap_state_epfirst_action_currentrel_eef10"
)
MEMORY_CHECKPOINT = "/data2/hzl_workspace_for_pi_mem/4999/params"
HISTORY_FRAMES = 241
CURRENT_START_FRAME = 241
ACTION_HORIZON = 16
ACTION_DIM = 10


# The /4999 MEM checkpoint was trained on the complete 240-frame event:
# history index 0 is the initial frame, followed by three disjoint 80-frame
# swaps.  Keeping all 80 frames per stage is also required by its
# relative_temporal_pos_embedding shape (1, 80, 1, 64).
REAL_SWAP_FRAME_INDICES = tuple(
    tuple(range(1 + stage * 80, 1 + (stage + 1) * 80))
    for stage in range(3)
)


@dataclasses.dataclass(frozen=True)
class _RealWristEpisodeFirstEEF10DataConfigMixin:
    """One wrist camera with pre-chunked current-relative 10-D link6 targets."""

    num_frames: int = HISTORY_FRAMES + 1
    frame_stride: int = 1
    padding_mode: str = "repeat"
    num_future_frames: int = 0
    future_frame_stride: int = 1
    video_layout: str = "fixed_prefix_current"
    fixed_prefix_frames: int = HISTORY_FRAMES
    min_frame_index: int | None = CURRENT_START_FRAME
    max_frame_index: int | None = None
    tokenize_prompt: bool = True
    image_keys: tuple[str, ...] = ("left_wrist_0_rgb_0",)

    normalize_masks: ClassVar[dict[str, tuple[bool, ...]]] = {
        "actions": make_bool_mask(ACTION_DIM),
        "state": make_bool_mask(ACTION_DIM),
    }

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
            f"{key}_{frame}": f"{key}_{frame}" for key in self.image_keys for frame in range(self.num_frames)
        }
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "observation.robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": ("observation.robot0_eef_rot_axis_angle"),
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        **per_frame_keys,
                        "video_frame_valid_mask": {key: f"video_frame_valid_mask/{key}" for key in self.image_keys},
                        "actions": "actions",
                        "prompt": "task",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                        "episode_length": "episode_length",
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
                _config_pi0_mem.UmiInputsV4ShellgameRealWristVideo(
                    num_frames=self.num_frames,
                    action_horizon=model_config.action_horizon,
                ),
            ]
        )
        model_inputs: list[Any] = []
        if self.tokenize_prompt:
            model_inputs.extend(
                [
                    _transforms.InjectDefaultPrompt(None),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=getattr(model_config, "discrete_state_input", False),
                    ),
                ]
            )
        model_inputs.extend(
            [
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
                _transforms.KeepModelKeys(),
            ]
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=_transforms.Group(
                inputs=model_inputs,
                outputs=[
                    _transforms.ChunkActions(target_dim=ACTION_DIM),
                    _transforms.DropKeys(keys=("state",)),
                ],
            ),
        )


_DATA_CONFIG_TYPES: dict[type, type] = {}


def data_config_type(config_module: Any | None = None) -> type:
    if config_module is None:
        from openpi.training import config as config_module

    base_type = config_module.DataConfigFactory
    if concrete := _DATA_CONFIG_TYPES.get(base_type):
        return concrete

    @dataclasses.dataclass(frozen=True)
    class ShellGameRealWristEpisodeFirstEEF10DataConfig(
        _RealWristEpisodeFirstEEF10DataConfigMixin,
        base_type,
    ):
        pass

    ShellGameRealWristEpisodeFirstEEF10DataConfig.__module__ = __name__
    _DATA_CONFIG_TYPES[base_type] = ShellGameRealWristEpisodeFirstEEF10DataConfig
    return ShellGameRealWristEpisodeFirstEEF10DataConfig


MODEL_CONFIG = dataclasses.replace(
    _base_recipe.MODEL_CONFIG,
    num_frames=HISTORY_FRAMES + 1,
    history_frames=HISTORY_FRAMES,
    action_horizon=ACTION_HORIZON,
    action_dim=32,
    action_loss_mask=(1.0,) * ACTION_DIM + (0.0,) * (32 - ACTION_DIM),
    real_action_dim=ACTION_DIM,
    gripper_action_index=9,
    # Variable episode lengths are carried in every observation; this is only
    # the backward-compatible fallback used by inference-only callers.
    last_episode_frame=100_000,
    swap_frame_indices=REAL_SWAP_FRAME_INDICES,
)


def make_train_config(config_module: Any | None = None) -> Any:
    if config_module is None:
        from openpi.training import config as config_module

    _config = config_module
    data_cls = data_config_type(_config)
    return _config.TrainConfig(
        name="pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_stage2",
        model=MODEL_CONFIG,
        # The visual tracker stays frozen. The two modules that connect its
        # memory to Pi0.5 must train together with the action expert.
        freeze_filter=MODEL_CONFIG.get_freeze_filter_memory_interface_finetune(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                data_cls(
                    repo_id=DATASET_ROOT,
                    assets=_config.AssetsConfig(asset_id=".", assets_dir=DATASET_ROOT),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * ACTION_DIM + (0.0,) * (32 - ACTION_DIM),
                        robot_type="ARM=1 G=0 H=0",
                    ),
                )
            ],
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoader(MEMORY_CHECKPOINT),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=3e-5,
            decay_steps=21_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=21_000,
        batch_size=4,
        num_workers=8,
        fsdp_devices=4,
        log_interval=10,
        save_interval=500,
        keep_period=1_000,
        val_ratio=0.1,
        eval_interval=250,
        # 64 batches = 256 action rows, large enough to avoid the failed
        # experiment's eight-row validation while keeping 242-frame eval sane.
        eval_batches=64,
        wandb_enabled=False,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(enabled=False),
    )
