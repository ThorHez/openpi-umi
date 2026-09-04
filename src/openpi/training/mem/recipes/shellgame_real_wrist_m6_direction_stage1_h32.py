"""Balanced frame-241 M6 direction training with 32-step EEF10 chunks."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from openpi.training.mem.recipes import shellgame_real_wrist_m6 as _m6
from openpi.training.mem.recipes import shellgame_real_wrist_m6_direction_stage1 as _h16
from openpi.training.mem.recipes import shellgame_real_wrist_stage2_h32 as _stage2
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders

CONFIG_NAME = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_stage1_h32"
LABELS_PATH = _h16.LABELS_PATH
DECISION_FRAME = _stage2.CURRENT_START_FRAME

# The model/loss implementation is horizon-generic. This module only binds it
# to the independently converted H32 dataset and its H32 centroids/statistics.
RealM6DirectionStage1Config = _h16.RealM6DirectionStage1Config


def _load_split() -> tuple[list[int], list[int]]:
    root = Path(_stage2.DATASET_ROOT)
    audit = json.loads((root / "conversion_audit.json").read_text(encoding="utf-8"))
    validation = sorted(int(value) for value in audit["validation_episode_ids"])
    validation_set = set(validation)
    training = [episode for episode in range(int(audit["episodes"])) if episode not in validation_set]
    if len(training) != 275 or len(validation) != 31:
        raise ValueError(f"Expected seed-42 split 275/31, got {len(training)}/{len(validation)}")
    return training, validation


def load_final_cups() -> tuple[int, ...]:
    return _h16.load_final_cups()


def _raw_xyz_centroids() -> np.ndarray:
    labels = load_final_cups()
    training, _ = _load_split()
    grouped: list[list[np.ndarray]] = [[], [], []]
    root = Path(_stage2.DATASET_ROOT)
    for episode_id in training:
        path = root / "data" / f"chunk-{episode_id // 1000:03d}" / f"episode_{episode_id:06d}.parquet"
        table = pq.read_table(path, columns=["frame_index", "actions"])
        frames = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)
        rows = np.flatnonzero(frames == DECISION_FRAME)
        if rows.size != 1:
            raise ValueError(f"episode {episode_id} has {rows.size} decision rows")
        action = np.asarray(table.column("actions")[int(rows[0])].as_py(), dtype=np.float32)
        grouped[labels[episode_id]].append(action[..., :3])
    return np.stack([np.mean(np.stack(items), axis=0) for items in grouped]).astype(np.float32)


def normalized_xyz_centroids() -> tuple[tuple[tuple[float, ...], ...], ...]:
    root = Path(_stage2.DATASET_ROOT)
    payload = json.loads((root / "norm_stats.json").read_text(encoding="utf-8"))["norm_stats"]["actions"]
    low = np.asarray(payload["min"], dtype=np.float32)[:3]
    high = np.asarray(payload["max"], dtype=np.float32)[:3]
    centroids = 2.0 * (_raw_xyz_centroids() - low) / np.maximum(high - low, 1e-7) - 1.0
    return tuple(tuple(tuple(float(value) for value in xyz) for xyz in chunk) for chunk in centroids)


def build_model_config(
    direction_loss_weight: float,
    direction_temperature: float,
) -> RealM6DirectionStage1Config:
    fields = {
        field.name: getattr(_stage2.MODEL_CONFIG, field.name) for field in dataclasses.fields(_stage2.MODEL_CONFIG)
    }
    return RealM6DirectionStage1Config(
        **fields,
        final_cups=load_final_cups(),
        direction_xyz_centroids=normalized_xyz_centroids(),
        direction_loss_weight=direction_loss_weight,
        direction_temperature=direction_temperature,
    )


def make_train_config(
    *,
    exp_name: str,
    checkpoint: str = _m6.DEFAULT_M5_CHECKPOINT,
    steps: int = 2_000,
    schedule_steps: int | None = None,
    warmup_steps: int = 100,
    peak_lr: float = 3e-5,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    num_workers: int = 16,
    fsdp_devices: int = 8,
    eval_interval: int = 100,
    eval_batches: int = 3,
    direction_loss_weight: float = 0.1,
    direction_temperature: float = 5e-4,
    enable_direction_early_stop: bool = True,
    save_interval: int = 5_000,
    resume: bool = False,
    overwrite: bool = False,
) -> Any:
    from openpi.training import config as _config

    model = build_model_config(direction_loss_weight, direction_temperature)
    if not enable_direction_early_stop:
        model = dataclasses.replace(model, direction_early_stop_metric="")
    schedule_steps = steps if schedule_steps is None else schedule_steps
    if schedule_steps < steps:
        raise ValueError("schedule_steps must be >= steps")
    data_cls = _m6.data_config_type(_config)
    params_path = Path(checkpoint)
    if params_path.name != "params":
        params_path /= "params"
    reset_modules = r".*(HistorySemanticJointActionReadout|HistoryRawMemoryQueryResampler|ActionMemoryCrossAttention).*"
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_memory_interface_finetune(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                data_cls(
                    repo_id=_stage2.DATASET_ROOT,
                    assets=_config.AssetsConfig(asset_id=".", assets_dir=_stage2.DATASET_ROOT),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * _stage2.ACTION_DIM + (0.0,) * (32 - _stage2.ACTION_DIM),
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    min_frame_index=DECISION_FRAME,
                    max_frame_index=DECISION_FRAME,
                )
            ],
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderReinitialize(
            params_path=str(params_path), reinitialize_regex=reset_modules
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(warmup_steps, max(steps - 1, 0)),
            peak_lr=peak_lr,
            decay_steps=max(schedule_steps, 2),
            decay_lr=peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=steps,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
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


def filter_frame241_balanced_indices(dataset, indices, classifier_config):
    """Downsample each split to equal final-cup counts at frame 241."""
    del classifier_config
    current = dataset
    hf_dataset = None
    sample_indices = None
    while current is not None:
        if sample_indices is None:
            sample_indices = getattr(current, "sample_indices", None)
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            break
        current = getattr(current, "_dataset", None)
    if hf_dataset is None:
        raise ValueError("balanced direction sampler could not find HuggingFace dataset")
    frames = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    episodes = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    if sample_indices is not None:
        mapped = np.asarray(sample_indices, dtype=np.int64)
        frames, episodes = frames[mapped], episodes[mapped]
    selected = np.asarray(indices, dtype=np.int64)
    selected = selected[frames[selected] == DECISION_FRAME]
    labels = np.asarray(load_final_cups(), dtype=np.int64)[episodes[selected]]
    groups = [selected[labels == cup] for cup in range(3)]
    per_class = min(len(group) for group in groups)
    if per_class <= 0:
        raise ValueError("balanced direction sampler found an empty class")
    rng = np.random.default_rng(42 + int(np.min(episodes[selected])))
    balanced = np.concatenate([rng.permutation(group)[:per_class] for group in groups])
    return rng.permutation(balanced).tolist()
