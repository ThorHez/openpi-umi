"""Real-robot ShellGame M6: frozen tracker -> Pi0.5 action with direction prompt.

M5 proved that the frozen 241-frame tracker can causally route a deterministic
EEF10 action head.  M6 removes that diagnostic head and trains the generic
raw-memory query/cross-attention path together with the Pi0.5 action expert.

Training receives an explicit direction prompt derived from the dataset's
ground-truth ``final_cup``.  Deployment must derive the same prompt from the
frozen MEM prediction; no ground-truth cup is available at inference time.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

M6_CONFIG_NAME = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_prompt"
DEFAULT_M5_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5/"
    "real306_m5_memory_seed42_v1/999/params"
)
CUP_NAMES = ("left", "middle", "right")
PROMPT_TEMPLATE = (
    "The shell game has ended. The ball is under the {cup} cup. Move toward, grasp, and lift the {cup} cup."
)


def direction_prompt(cup: int | str) -> str:
    """Return the one canonical prompt used by training and deployment."""
    if isinstance(cup, str):
        name = cup.strip().lower()
        if name not in CUP_NAMES:
            raise ValueError(f"Unknown cup name {cup!r}; expected one of {CUP_NAMES}")
    else:
        index = int(cup)
        if not 0 <= index < len(CUP_NAMES):
            raise ValueError(f"Cup index must be 0, 1, or 2, got {index}")
        name = CUP_NAMES[index]
    return PROMPT_TEMPLATE.format(cup=name.upper())


@dataclasses.dataclass(frozen=True)
class InjectDirectionPrompt(_transforms.DataTransformFn):
    """Replace the generic LeRobot task with a final-cup direction prompt."""

    def __call__(self, data: dict) -> dict:
        if "final_cup" not in data:
            # Deployment/evaluation already supplies the canonical prompt
            # derived from the frozen MEM prediction. ``final_cup`` is a
            # privileged training-only field and is unavailable at inference.
            if "prompt" in data:
                return data
            raise KeyError("M6 direction-prompt training requires final_cup in every row")
        final_cup = int(np.asarray(data["final_cup"]).reshape(()))
        data["prompt"] = np.asarray(direction_prompt(final_cup))
        return data


_DATA_CONFIG_TYPES: dict[type, type] = {}


def data_config_type(config_module: Any | None = None) -> type:
    """Add a training-only direction prompt to the deployed Stage2 contract."""
    if config_module is None:
        from openpi.training import config as config_module

    base_type = _stage2.data_config_type(config_module)
    if concrete := _DATA_CONFIG_TYPES.get(base_type):
        return concrete

    @dataclasses.dataclass(frozen=True)
    class ShellGameRealWristM6DirectionPromptDataConfig(base_type):
        def create(self, assets_dirs, model_config):
            config = super().create(assets_dirs, model_config)

            # The Stage2 repack intentionally retained only the generic task.
            # Rebuild it with final_cup so the prompt can be generated per row.
            per_frame_keys = {
                f"{key}_{frame}": f"{key}_{frame}" for key in self.image_keys for frame in range(self.num_frames)
            }
            repack = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "robot0_eef_pos": "observation.robot0_eef_pos",
                            "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
                            "robot0_gripper_width": "observation.robot0_gripper_width",
                            **per_frame_keys,
                            "video_frame_valid_mask": {key: f"video_frame_valid_mask/{key}" for key in self.image_keys},
                            "actions": "actions",
                            "prompt": "task",
                            "final_cup": "final_cup",
                            "episode_index": "episode_index",
                            "frame_index": "frame_index",
                            "episode_length": "episode_length",
                        }
                    )
                ]
            )
            return dataclasses.replace(
                config,
                repack_transforms=repack,
                # Insert before TokenizePrompt, then KeepModelKeys removes
                # final_cup after the string has been tokenized.
                model_transforms=_transforms.Group(
                    inputs=(InjectDirectionPrompt(), *config.model_transforms.inputs),
                    outputs=config.model_transforms.outputs,
                ),
            )

    ShellGameRealWristM6DirectionPromptDataConfig.__module__ = __name__
    ShellGameRealWristM6DirectionPromptDataConfig.__qualname__ = "ShellGameRealWristM6DirectionPromptDataConfig"
    _DATA_CONFIG_TYPES[base_type] = ShellGameRealWristM6DirectionPromptDataConfig
    return ShellGameRealWristM6DirectionPromptDataConfig


def make_train_config(
    *,
    exp_name: str,
    checkpoint: str = DEFAULT_M5_CHECKPOINT,
    steps: int = 21_000,
    warmup_steps: int = 500,
    peak_lr: float = 3e-5,
    batch_size: int = 4,
    num_workers: int = 8,
    fsdp_devices: int = 4,
    eval_interval: int = 250,
    eval_batches: int = 64,
    save_interval: int = 5_000,
    resume: bool = False,
    overwrite: bool = False,
) -> Any:
    """Build M6 without registering the experiment-only config globally."""
    from openpi.training import config as _config

    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0 or batch_size % fsdp_devices != 0:
        raise ValueError("batch_size must be positive and divisible by fsdp_devices")
    params_path = Path(checkpoint)
    if params_path.name != "params":
        params_path = params_path / "params"

    model = _stage2.MODEL_CONFIG
    data_cls = data_config_type(_config)
    reset_modules = (
        r".*(HistorySemanticJointActionReadout|"
        r"HistoryRawMemoryQueryResampler|ActionMemoryCrossAttention).*"
    )
    return _config.TrainConfig(
        name=M6_CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_memory_interface_finetune(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                data_cls(
                    repo_id=_stage2.DATASET_ROOT,
                    assets=_config.AssetsConfig(
                        asset_id=".",
                        assets_dir=_stage2.DATASET_ROOT,
                    ),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * _stage2.ACTION_DIM + (0.0,) * (32 - _stage2.ACTION_DIM),
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    min_frame_index=_stage2.CURRENT_START_FRAME,
                    max_frame_index=None,
                )
            ],
        ),
        # M5 owns the verified tracker. Its task-specific deterministic head
        # is discarded, and the generic M6 memory/action interface starts new.
        weight_loader=_weight_loaders.CheckpointWeightLoaderReinitialize(
            params_path=str(params_path),
            reinitialize_regex=reset_modules,
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
        seed=42,
        log_interval=10,
        save_interval=min(save_interval, steps),
        keep_period=steps,
        resume=resume,
        val_ratio=0.1,
        eval_interval=min(eval_interval, steps),
        eval_batches=eval_batches,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(enabled=False),
    )
