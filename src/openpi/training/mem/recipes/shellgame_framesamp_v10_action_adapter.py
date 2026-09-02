"""Train a frozen MME FrameSamp memory interface on the exact V10 action graph."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v10_timing_diag as _v10
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.shared import nnx_utils
from openpi.training import optimizer as _optimizer
from openpi.training.mem.recipes import shellgame_semantic_action as _base_data
from openpi.training.mem.recipes import shellgame_v10_exact_parallel_semantic_adapter as _exact
import openpi.transforms as _transforms


MEMORY_TOKENS = 512
MEMORY_WIDTH = 1024
DEFAULT_V10_CHECKPOINT = _exact.DEFAULT_V10_CHECKPOINT
DEFAULT_DATASET_ROOT = Path(_v10.NOMINAL_ROOT).resolve()


@dataclasses.dataclass(frozen=True)
class FrameSampMemoryLookup:
    memories: np.ndarray
    episode_to_row: np.ndarray
    labels: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, root: Path) -> "FrameSampMemoryLookup":
        root = root.expanduser().resolve()
        if not (root / "_COMPLETE").is_file():
            raise FileNotFoundError(f"Incomplete FrameSamp bank: {root}")
        metadata = json.loads((root / "metadata.json").read_text())
        memories = np.load(root / "memory.npy", mmap_mode="r")
        episodes = np.asarray(np.load(root / "episode_index.npy"), dtype=np.int64)
        labels = np.asarray(np.load(root / "final_label.npy"), dtype=np.int8)
        if memories.ndim != 3 or memories.shape[1:] != (MEMORY_TOKENS, MEMORY_WIDTH):
            raise ValueError(
                f"Expected FrameSamp memory [N,{MEMORY_TOKENS},{MEMORY_WIDTH}], got {memories.shape}"
            )
        if len(episodes) != len(memories) or len(labels) != len(memories):
            raise ValueError("FrameSamp bank arrays have inconsistent episode counts")
        if len(np.unique(episodes)) != len(episodes) or np.any(episodes < 0):
            raise ValueError("FrameSamp bank episode indices must be unique and non-negative")
        episode_to_row = np.full(int(np.max(episodes)) + 1, -1, dtype=np.int64)
        episode_to_row[episodes] = np.arange(len(episodes), dtype=np.int64)
        return cls(memories, episode_to_row, labels, metadata)

    def has(self, episode_index: int) -> bool:
        return (
            0 <= episode_index < len(self.episode_to_row)
            and int(self.episode_to_row[episode_index]) >= 0
        )

    def at(self, episode_index: int) -> np.ndarray:
        if not self.has(episode_index):
            raise KeyError(f"Episode {episode_index} is absent from FrameSamp bank")
        row = int(self.episode_to_row[episode_index])
        return np.asarray(self.memories[row], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class InjectFrameSampMemory(_transforms.DataTransformFn):
    lookup: FrameSampMemoryLookup

    def __call__(self, data: dict) -> dict:
        episode = int(np.asarray(data["episode_index"]).reshape(()))
        data["semantic_memory"] = self.lookup.at(episode)
        return data


_DATA_CONFIG_TYPES: dict[type, type] = {}


def data_config_type(config_module: Any) -> type:
    base_type = config_module.DataConfigFactory
    if concrete := _DATA_CONFIG_TYPES.get(base_type):
        return concrete
    base_shellgame = _base_data.data_config_type(config_module)

    @dataclasses.dataclass(frozen=True)
    class ShellGameFrameSampV10DataConfig(base_shellgame):
        memory_bank_dir: str = ""

        def create(self, assets_dirs, model_config):
            config = super().create(assets_dirs, model_config)
            lookup = FrameSampMemoryLookup.load(Path(self.memory_bank_dir))
            return dataclasses.replace(
                config,
                data_transforms=config.data_transforms.push(
                    inputs=[InjectFrameSampMemory(lookup)]
                ),
                model_transforms=config.model_transforms.push(
                    inputs=[_transforms.KeepModelKeys()]
                ),
            )

    ShellGameFrameSampV10DataConfig.__module__ = __name__
    ShellGameFrameSampV10DataConfig.__qualname__ = "ShellGameFrameSampV10DataConfig"
    _DATA_CONFIG_TYPES[base_type] = ShellGameFrameSampV10DataConfig
    return ShellGameFrameSampV10DataConfig


def make_model_config():
    model = _exact.make_model_config()
    return dataclasses.replace(
        model,
        semantic_memory_tokens=MEMORY_TOKENS,
        semantic_memory_width=MEMORY_WIDTH,
        semantic_query_tokens=16,
        semantic_hidden_width=256,
        parallel_semantic_adapter_enabled=True,
        old_memory_condition_strength=0.0,
        semantic_residual_gate_init=0.0,
    )


def make_index_filter(memory_bank_dir: Path):
    lookup = FrameSampMemoryLookup.load(memory_bank_dir)

    def filter_indices(dataset, indices: list[int], classifier_config) -> list[int]:
        hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
        selected = np.asarray(indices, dtype=np.int64)
        episodes = np.asarray(hf["episode_index"], dtype=np.int64)[selected]
        available = np.asarray([lookup.has(int(episode)) for episode in episodes])
        kept = selected[available].tolist()
        logging.info(
            "FrameSamp bank filter rows=%d->%d episodes=%d->%d",
            len(indices),
            len(kept),
            len(np.unique(episodes)),
            len(np.unique(episodes[available])),
        )
        return _v10._indices(dataset, kept, classifier_config)  # noqa: SLF001

    return filter_indices


def make_train_config(
    *,
    config_module: Any,
    exp_name: str,
    memory_bank_dir: Path,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    init_checkpoint: Path = DEFAULT_V10_CHECKPOINT,
    steps: int = 1000,
    peak_lr: float = 1e-4,
    batch_size: int = 8,
    fsdp_devices: int = 4,
    num_workers: int = 4,
    overwrite: bool = False,
):
    memory_bank_dir = memory_bank_dir.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    init_checkpoint = init_checkpoint.expanduser().resolve()
    FrameSampMemoryLookup.load(memory_bank_dir)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not init_checkpoint.is_dir():
        raise FileNotFoundError(init_checkpoint)
    if batch_size % fsdp_devices != 0:
        raise ValueError("batch_size must be divisible by fsdp_devices")

    model = make_model_config()
    adapter = nnx_utils.PathRegex(r".*ParallelSemanticMemoryActionConditioner.*")
    data_cls = data_config_type(config_module)
    logging.info(
        "FrameSamp->V10: memory=%s V10=%s trainable=parallel-action-interface only",
        memory_bank_dir,
        init_checkpoint,
    )
    return config_module.TrainConfig(
        name="pi0_shellgame_framesamp_v10_action_adapter_eef7_v1",
        exp_name=exp_name,
        model=model,
        freeze_filter=nnx.Not(adapter),
        data=config_module.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                data_cls(
                    repo_id=str(dataset_root),
                    memory_bank_dir=str(memory_bank_dir),
                    assets=config_module.AssetsConfig(
                        asset_id=".", assets_dir=str(DEFAULT_DATASET_ROOT)
                    ),
                    base_config=config_module.UmiDataConfig(
                        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=61,
                    frame_stride=1,
                    video_layout="sliding",
                )
            ],
        ),
        weight_loader=_exact.ExactV10ParallelAdapterLoader(str(init_checkpoint)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(100, max(steps - 1, 0)),
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
        # A frozen V10 checkpoint is large. Save the final baseline and retain
        # only the latest snapshot instead of accumulating periodic copies.
        save_interval=max(steps, 1),
        keep_period=None,
        val_ratio=0.1,
        eval_interval=125,
        eval_batches=20,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=config_module.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=config_module.ShellgameCupEvalConfig(enabled=False),
    )
