"""Real Pi0.5 flow-action recipe with causal frozen PickXtimes MEM tokens."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, ClassVar

import h5py
import numpy as np

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
from openpi.tasks.robomme.pickxtimes import pi0_memory_action as _memory_action
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
from openpi.transforms import make_bool_mask

DEFAULT_DATASET_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/data/"
    "robomme_pickxtimes_lerobot_pi_action_train70_stride2"
)
DEFAULT_MEMORY_BANK = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/data/robomme_extracted/"
    "pickxtimes_action_memory_tokens_round9_train70_dev15.h5"
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
        video = np.transpose(video, (0, 2, 3, 1))
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Expected video [T,H,W,3] or [T,3,H,W], got {video.shape}")
    return video


@dataclasses.dataclass(frozen=True)
class FrozenMemoryLookup:
    banks: tuple[np.ndarray, ...]
    visible_timesteps: tuple[np.ndarray, ...]

    @classmethod
    def load(cls, dataset_root: pathlib.Path, memory_path: pathlib.Path, mode: str):
        if mode not in {"predicted", "predicted_shuffled", "action_only"}:
            raise ValueError(f"Unknown frozen-memory lookup mode {mode!r}")
        episodes = [
            json.loads(line)
            for line in (dataset_root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        banks, visible = [], []
        with h5py.File(memory_path, "r") as source:
            for expected_index, item in enumerate(episodes):
                if int(item["episode_index"]) != expected_index:
                    raise ValueError("Converted episodes must be contiguous and ordered")
                group = source[item["source_episode_name"]]
                initial = np.asarray(group["initial_memory"], dtype=np.float32)
                if mode == "action_only":
                    banks.append(np.zeros_like(initial)[None])
                    visible.append(np.empty((0,), dtype=np.int32))
                else:
                    stages = np.asarray(group["predicted/stage_memories"], dtype=np.float32)
                    banks.append(np.concatenate((initial[None], stages), axis=0))
                    visible.append(np.asarray(group["predicted/visible_timesteps"], dtype=np.int32))
        if mode == "predicted_shuffled" and len(banks) > 1:
            banks = banks[1:] + banks[:1]
            visible = visible[1:] + visible[:1]
        return cls(tuple(banks), tuple(visible))

    def at(self, episode_index: int, timestep: int) -> np.ndarray:
        index = int(episode_index)
        offset = int(np.searchsorted(self.visible_timesteps[index], int(timestep), side="right"))
        return self.banks[index][offset]


@dataclasses.dataclass(frozen=True)
class PickXtimesMemoryActionInputs(_transforms.DataTransformFn):
    memory_lookup: FrozenMemoryLookup
    num_frames: int = 1

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state_raw"], dtype=np.float32).reshape(-1)
        if state.shape != (11,):
            raise ValueError(f"Expected PickXtimes state11, got {state.shape}")
        front, wrist = _parse_video(data["front_rgb_video"]), _parse_video(data["wrist_rgb_video"])
        expected = (self.num_frames, 224, 224, 3)
        if front.shape != expected or wrist.shape != expected:
            raise ValueError(f"Expected current videos {expected}, got {front.shape}, {wrist.shape}")
        episode_index = int(np.asarray(data["episode_index"]).reshape(()))
        frame_index = int(np.asarray(data["frame_index"]).reshape(()))
        data["episode_T"] = np.float32(data["episode_T"])
        data["state"] = state
        data["semantic_memory"] = self.memory_lookup.at(episode_index, frame_index)
        data["image"] = {"base_rgb": front, "wrist_rgb": wrist}
        data["image_mask"] = {"base_rgb": np.True_, "wrist_rgb": np.True_}
        frame_mask = data.get("video_frame_valid_mask")
        if frame_mask is not None:
            data["frame_valid_mask"] = {
                "base_rgb": np.asarray(frame_mask.get("front_rgb", np.ones(1, dtype=np.bool_))),
                "wrist_rgb": np.asarray(frame_mask.get("wrist_rgb", np.ones(1, dtype=np.bool_))),
            }
        return data


@dataclasses.dataclass(frozen=True)
class _PickXtimesMemoryActionDataConfigMixin:
    memory_path: str = str(DEFAULT_MEMORY_BANK)
    memory_mode: str = "predicted"
    num_frames: int = 1
    frame_stride: int = 1
    padding_mode: str = "repeat"
    num_future_frames: int = 0
    future_frame_stride: int = 1
    video_layout: str = "sliding"
    image_keys: tuple[str, ...] = ("front_rgb", "wrist_rgb")

    normalize_masks: ClassVar[dict[str, tuple[bool, ...]]] = {
        "actions": make_bool_mask(7),
        "state": make_bool_mask(11),
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
        )

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig):
        config = super().create_base_config(assets_dirs, model_config)
        return dataclasses.replace(config, normalize_masks=self.normalize_masks)

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig):
        from openpi import transforms_video as _transforms_video

        root = pathlib.Path(self.repo_id).expanduser().resolve()
        lookup = FrozenMemoryLookup.load(
            root, pathlib.Path(self.memory_path).expanduser().resolve(), self.memory_mode
        )
        per_frame_keys = {
            f"{key}_{frame}": f"{key}_{frame}"
            for key in self.image_keys
            for frame in range(self.num_frames)
        }
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "state_raw": "observation.state", **per_frame_keys,
            "video_frame_valid_mask": {
                key: f"video_frame_valid_mask/{key}" for key in self.image_keys
            },
            "actions": "actions", "prompt": "task", "episode_index": "episode_index",
            "frame_index": "frame_index", "episode_T": "episode_T",
        })])
        data_transforms = _transforms.Group(inputs=[
            _transforms_video.BuildVideoTensor(
                image_keys=self.image_keys,
                num_frames=self.num_frames,
                output_keys={key: f"{key}_video" for key in self.image_keys},
            ),
            PickXtimesMemoryActionInputs(memory_lookup=lookup, num_frames=self.num_frames),
        ])
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
            outputs=[_transforms.ChunkActions(target_dim=7), _transforms.DropKeys(keys=("state",))],
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
    class PickXtimesMemoryActionDataConfig(_PickXtimesMemoryActionDataConfigMixin, base_type):
        pass

    PickXtimesMemoryActionDataConfig.__module__ = __name__
    PickXtimesMemoryActionDataConfig.__qualname__ = "PickXtimesMemoryActionDataConfig"
    _DATA_CONFIG_TYPES[base_type] = PickXtimesMemoryActionDataConfig
    return PickXtimesMemoryActionDataConfig


def make_model_config(
    *,
    use_learned_null_memory: bool = False,
    semantic_residual_gate_init: float = 1.0,
    semantic_residual_dropout_rate: float = 0.0,
) -> _memory_action.Pi0PickXtimesMemoryActionConfig:
    return _memory_action.Pi0PickXtimesMemoryActionConfig(
        pi05=True, action_dim=32, action_horizon=16,
        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        max_token_len=256, num_frames=1, current_frame_index=0,
        memory_every=0, history_memory_tokens=1, history_resampler_depth=1,
        history_use_current_condition=False, history_gate_fixed=0.0, diversity_weight=0.0,
        current_frame_corrupt_sample_prob=0.0, current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0, current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=0, semantic_memory_tokens=128,
        semantic_memory_width=64, semantic_query_tokens=8, semantic_hidden_width=256,
        gripper_loss_weight=4.0, real_action_dim=7, gripper_action_index=6,
        use_learned_null_memory=use_learned_null_memory,
        semantic_residual_gate_init=semantic_residual_gate_init,
        semantic_residual_dropout_rate=semantic_residual_dropout_rate,
    )


def make_train_config(*, config_module: Any, dataset_root: pathlib.Path = DEFAULT_DATASET_ROOT,
                      memory_path: pathlib.Path = DEFAULT_MEMORY_BANK, memory_mode: str,
                      init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT, exp_name: str,
                      steps: int = 500, batch_size: int = 12, fsdp_devices: int = 6,
                      num_workers: int = 8, peak_lr: float = 3e-5, warmup_steps: int = 50,
                      eval_interval: int = 100, eval_batches: int = 10,
                      save_interval: int = 250, keep_period: int | None = None,
                      overwrite: bool = False, resume: bool = False,
                      semantic_residual_gate_init: float = 1.0,
                      semantic_residual_dropout_rate: float = 0.0):
    model = make_model_config(
        use_learned_null_memory=memory_mode == "action_only",
        semantic_residual_gate_init=semantic_residual_gate_init,
        semantic_residual_dropout_rate=semantic_residual_dropout_rate,
    )
    data_cls = data_config_type(config_module)
    root = dataset_root.expanduser().resolve()
    return config_module.TrainConfig(
        name=f"pi0_robomme_pickxtimes_{memory_mode}_memory_action_260824",
        exp_name=exp_name,
        model=model,
        data=data_cls(
            repo_id=str(root), memory_path=str(memory_path.expanduser().resolve()), memory_mode=memory_mode,
            assets=config_module.AssetsConfig(asset_id=".", assets_dir=str(root)),
            base_config=config_module.UmiDataConfig(
                action_loss_mask=(1.0,) * 7 + (0.0,) * 25, robot_type="ARM=1 G=0 H=0"
            ),
        ),
        freeze_filter=model.get_freeze_filter_action_finetune(),
        weight_loader=_weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(
            str(init_checkpoint.expanduser().resolve())
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(warmup_steps, max(steps - 1, 0)), peak_lr=peak_lr,
            decay_steps=max(steps, 2), decay_lr=peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0), ema_decay=None,
        num_train_steps=steps, batch_size=batch_size, num_workers=num_workers,
        fsdp_devices=fsdp_devices, log_interval=10, save_interval=save_interval,
        keep_period=(max(save_interval, 1) if keep_period is None else max(keep_period, 1)),
        val_ratio=0.1, eval_interval=eval_interval,
        eval_batches=eval_batches, wandb_enabled=False, overwrite=overwrite,
        resume=resume,
        shellgame_memory_classifier=config_module.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=config_module.ShellgameCupEvalConfig(enabled=False),
    )
