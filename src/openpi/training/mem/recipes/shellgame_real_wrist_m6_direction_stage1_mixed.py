"""Direction-guarded M6 frame-241 training on old306 + cup_0903."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize as _normalize
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_real_mixed_common as _mixed
from openpi.training.mem.recipes import shellgame_real_wrist_m6 as _m6
from openpi.training.mem.recipes import shellgame_real_wrist_m6_direction_stage1 as _base
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2

CONFIG_NAME = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_stage1_mixed_cup0903"


def _source_raw_centroids(
    root: Path,
    labels: tuple[int, ...],
    training: set[int],
    *,
    frame_start: int,
    frame_end: int,
) -> np.ndarray:
    grouped: list[list[np.ndarray]] = [[], [], []]
    for episode in sorted(training):
        path = root / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
        table = pq.read_table(path, columns=["frame_index", "actions"])
        frames = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)
        rows = np.flatnonzero((frames >= frame_start) & (frames <= frame_end))
        if rows.size != frame_end - frame_start + 1:
            raise ValueError(f"{path} has {rows.size} rows in frame range {frame_start}..{frame_end}")
        for row in rows:
            action = np.asarray(table.column("actions")[int(row)].as_py(), dtype=np.float32)
            grouped[labels[episode]].append(action[..., :3])
    return np.stack([np.mean(np.stack(items), axis=0) for items in grouped]).astype(np.float32)


def normalized_xyz_centroids(
    *,
    frame_start: int = _stage2.CURRENT_START_FRAME,
    frame_end: int = _stage2.CURRENT_START_FRAME,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    labels = _mixed.source_final_cups()
    splits = _mixed.source_splits()
    roots = (_mixed.OLD_DATASET_ROOT, _mixed.NEW_DATASET_ROOT)
    raw = sum(
        probability
        * _source_raw_centroids(
            root,
            source_labels,
            training,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        for probability, root, source_labels, (training, _) in zip(
            _mixed.SOURCE_PROBABILITIES,
            roots,
            labels,
            splits,
            strict=True,
        )
    )
    stats = [_normalize.load(root)["actions"] for root in roots]
    merged = _normalize.merge_norm_stats(stats, weights=list(_mixed.SOURCE_PROBABILITIES))
    if merged.min is None or merged.max is None:
        raise ValueError("Mixed action normalization requires min/max statistics")
    low = np.asarray(merged.min, dtype=np.float32)[:3]
    high = np.asarray(merged.max, dtype=np.float32)[:3]
    centroids = 2.0 * (raw - low) / np.maximum(high - low, 1e-7) - 1.0
    return tuple(tuple(tuple(float(value) for value in xyz) for xyz in chunk) for chunk in centroids)


def build_model_config(
    direction_loss_weight: float,
    direction_temperature: float,
    *,
    direction_frame_start: int | None = None,
    direction_frame_end: int | None = None,
) -> _base.RealM6DirectionStage1Config:
    fields = {
        field.name: getattr(_stage2.MODEL_CONFIG, field.name)
        for field in dataclasses.fields(_stage2.MODEL_CONFIG)
    }
    return _base.RealM6DirectionStage1Config(
        **fields,
        final_cups=_mixed.global_final_cups(),
        direction_xyz_centroids=normalized_xyz_centroids(
            frame_start=(
                _stage2.CURRENT_START_FRAME
                if direction_frame_start is None
                else direction_frame_start
            ),
            frame_end=(
                _stage2.CURRENT_START_FRAME
                if direction_frame_end is None
                else direction_frame_end
            ),
        ),
        direction_loss_weight=direction_loss_weight,
        direction_temperature=direction_temperature,
        direction_frame_start=direction_frame_start,
        direction_frame_end=direction_frame_end,
    )


def make_train_config(
    *,
    exp_name: str,
    action_checkpoint: str = str(_mixed.OLD_H16_ACTION_CHECKPOINT),
    memory_checkpoint: str = str(_mixed.ADAPTED_MEMORY_CHECKPOINT),
    steps: int = 2_000,
    schedule_steps: int | None = None,
    warmup_steps: int = 100,
    peak_lr: float = 3e-5,
    batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 16,
    fsdp_devices: int = 8,
    eval_interval: int = 50,
    eval_batches: int = 2,
    direction_loss_weight: float = 0.1,
    direction_temperature: float = 5e-4,
    enable_direction_early_stop: bool = True,
    save_interval: int = 5_000,
    resume: bool = False,
    overwrite: bool = False,
) -> Any:
    from openpi.training import config as _config

    if batch_size <= 0 or batch_size % fsdp_devices:
        raise ValueError("batch_size must be positive and divisible by fsdp_devices")
    model = build_model_config(direction_loss_weight, direction_temperature)
    if not enable_direction_early_stop:
        model = dataclasses.replace(model, direction_early_stop_metric="")
    schedule_steps = steps if schedule_steps is None else schedule_steps
    if schedule_steps < steps:
        raise ValueError("schedule_steps must be >= steps")
    action_path = Path(action_checkpoint)
    memory_path = Path(memory_checkpoint)
    if action_path.name != "params":
        action_path /= "params"
    if memory_path.name != "params":
        memory_path /= "params"
    data_type = _m6.data_config_type(_config)
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_memory_interface_finetune(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=_mixed.sampler_weights(decision_frame=_stage2.CURRENT_START_FRAME),
            norm_weights=list(_mixed.SOURCE_PROBABILITIES),
            datasets=_mixed.make_dataset_factories(
                _config,
                data_type,
                min_frame=_stage2.CURRENT_START_FRAME,
                max_frame=_stage2.CURRENT_START_FRAME,
            ),
        ),
        weight_loader=_weight_loaders.OverlayCheckpointWeightLoader(
            params_path=str(action_path),
            overlay_params_path=str(memory_path),
            overlay_regex=r".*swap_relation_classifier.*",
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
