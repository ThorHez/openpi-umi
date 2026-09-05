"""Paired M6 prompt-only versus prompt+MEM action-conditioning ablation.

Both arms start from the same fresh held-out ShellGame MEM checkpoint and use
the same canonical direction prompt, action data, episode split, optimizer,
and seed.  ``prompt_only`` disables the MEM-to-action cross-attention while
``prompt_memory`` retains the original M6 action-memory injection.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize as _normalize
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_real_mixed_common as _mixed
from openpi.training.mem.recipes import shellgame_real_wrist_m6 as _m6
from openpi.training.mem.recipes import shellgame_real_wrist_m6_direction_stage1 as _direction
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2

CONFIG_NAME = "pi0_mem_shellgame_real_m6_prompt_action_ablation_mixed"
CONDITION_MODES = ("prompt_only", "prompt_memory")
MEMORY_EXPERIMENT_ROOT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_mem_shellgame_real_fresh_memory_mild_all/"
    "freshmem_train383_val44_mildaug_officialbase_b32_seed42_split20260904_v1"
)
DEFAULT_MEMORY_CHECKPOINT = MEMORY_EXPERIMENT_ROOT / "1800"
MEMORY_SPLIT_MANIFEST = MEMORY_EXPERIMENT_ROOT / "training_manifest.json"


def source_episode_splits() -> tuple[tuple[set[int], set[int]], tuple[set[int], set[int]]]:
    """Return the exact old306/cup0903 split used by the fresh MEM run."""
    payload = json.loads(MEMORY_SPLIT_MANIFEST.read_text(encoding="utf-8"))
    split = payload["episode_split"]
    if not split["heldout_validation"] or split["split_seed"] != 20260904:
        raise ValueError("Expected the fixed held-out MEM split with seed 20260904")

    result = []
    for domain, offset, expected_train, expected_validation in (
        ("old306", 0, 275, 31),
        ("cup0903", _mixed.NEW_EPISODE_OFFSET, 90, 10),
    ):
        training = {int(value) - offset for value in split["training"][domain]["global_episode_ids"]}
        validation = {int(value) - offset for value in split["validation"][domain]["global_episode_ids"]}
        if len(training) != expected_train or len(validation) != expected_validation:
            raise ValueError(f"Unexpected {domain} split sizes {len(training)}/{len(validation)}")
        if training & validation:
            raise ValueError(f"{domain} training/validation episodes overlap")
        result.append((training, validation))
    return result[0], result[1]


def fixed_episode_split_indices(dataset, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Apply the fresh-MEM episode split to either mixed action source."""
    del val_ratio, seed
    episodes, _ = _mixed._dataset_episode_and_frame_indices(dataset)  # noqa: SLF001
    episode_count = len(np.unique(episodes))
    if episode_count == _mixed.OLD_EPISODES:
        training, validation = source_episode_splits()[0]
    elif episode_count == _mixed.NEW_EPISODES:
        training, validation = source_episode_splits()[1]
    else:
        raise ValueError(f"Unknown action source with {episode_count} episodes")
    train_indices = np.flatnonzero(np.isin(episodes, list(training))).tolist()
    validation_indices = np.flatnonzero(np.isin(episodes, list(validation))).tolist()
    if not train_indices or not validation_indices:
        raise ValueError("Fresh-MEM episode split produced an empty action split")
    return train_indices, validation_indices


def filter_balanced_training_full_validation(
    dataset,
    indices: list[int],
    classifier_config,
) -> list[int]:
    """Keep frame 241 only; balance training while retaining every validation episode."""
    del classifier_config
    episodes, frames = _mixed._dataset_episode_and_frame_indices(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    selected = selected[frames[selected] == _stage2.CURRENT_START_FRAME]
    episode_count = len(np.unique(episodes))
    if episode_count == _mixed.OLD_EPISODES:
        training, validation = source_episode_splits()[0]
    elif episode_count == _mixed.NEW_EPISODES:
        training, validation = source_episode_splits()[1]
    else:
        raise ValueError(f"Unknown action source with {episode_count} episodes")

    selected_episodes = {int(value) for value in episodes[selected]}
    if selected_episodes <= validation:
        return selected.tolist()
    if selected_episodes <= training:
        return _mixed.filter_balanced_indices(
            dataset,
            selected.tolist(),
            None,
            decision_frame=_stage2.CURRENT_START_FRAME,
        )
    raise ValueError("Action subset does not match either fixed training or validation episodes")


def _source_raw_centroids(
    root: Path,
    labels: tuple[int, ...],
    training: set[int],
) -> np.ndarray:
    grouped: list[list[np.ndarray]] = [[], [], []]
    for episode in sorted(training):
        path = root / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
        table = pq.read_table(path, columns=["frame_index", "actions"])
        frames = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)
        rows = np.flatnonzero(frames == _stage2.CURRENT_START_FRAME)
        if rows.size != 1:
            raise ValueError(f"{path} has {rows.size} frame-241 rows")
        action = np.asarray(table.column("actions")[int(rows[0])].as_py(), dtype=np.float32)
        grouped[labels[episode]].append(action[..., :3])
    return np.stack([np.mean(np.stack(items), axis=0) for items in grouped]).astype(np.float32)


def normalized_xyz_centroids() -> tuple[tuple[tuple[float, ...], ...], ...]:
    labels = _mixed.source_final_cups()
    splits = source_episode_splits()
    roots = (_mixed.OLD_DATASET_ROOT, _mixed.NEW_DATASET_ROOT)
    raw = sum(
        probability * _source_raw_centroids(root, source_labels, training)
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


def _balanced_source_row_counts() -> tuple[int, int]:
    counts = []
    for labels, (training, _) in zip(_mixed.source_final_cups(), source_episode_splits(), strict=True):
        class_counts = [sum(labels[episode] == cup for episode in training) for cup in range(3)]
        counts.append(3 * min(class_counts))
    return counts[0], counts[1]


def sampler_weights() -> list[float]:
    old_rows, new_rows = _balanced_source_row_counts()
    return [
        _mixed.SOURCE_PROBABILITIES[0] / old_rows,
        _mixed.SOURCE_PROBABILITIES[1] / new_rows,
    ]


def build_model_config(condition_mode: str) -> _direction.RealM6DirectionStage1Config:
    if condition_mode not in CONDITION_MODES:
        raise ValueError(f"Unknown condition mode {condition_mode!r}; expected {CONDITION_MODES}")
    fields = {
        field.name: getattr(_stage2.MODEL_CONFIG, field.name) for field in dataclasses.fields(_stage2.MODEL_CONFIG)
    }
    fields["action_memory_injection"] = condition_mode == "prompt_memory"
    return _direction.RealM6DirectionStage1Config(
        **fields,
        final_cups=_mixed.global_final_cups(),
        direction_xyz_centroids=normalized_xyz_centroids(),
        direction_loss_weight=0.0,
        direction_temperature=5e-4,
        direction_frame_start=_stage2.CURRENT_START_FRAME,
        direction_frame_end=_stage2.CURRENT_START_FRAME,
        # Stop only when direction remains ineffective. Reaching a good
        # direction score must not stop optimization before XY error settles.
        direction_early_stop_success_above=2.0,
    )


def make_train_config(
    *,
    condition_mode: str,
    exp_name: str,
    checkpoint: str = str(DEFAULT_MEMORY_CHECKPOINT),
    steps: int = 3_000,
    warmup_steps: int = 150,
    peak_lr: float = 3e-5,
    batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 8,
    fsdp_devices: int = 4,
    eval_interval: int = 100,
    eval_batches: int = 4,
    save_interval: int = 5_000,
    resume: bool = False,
    overwrite: bool = False,
) -> Any:
    from openpi.training import config as _config

    if batch_size <= 0 or batch_size % fsdp_devices:
        raise ValueError("batch_size must be positive and divisible by fsdp_devices")
    model = build_model_config(condition_mode)
    params_path = Path(checkpoint)
    if params_path.name != "params":
        params_path /= "params"
    data_type = _m6.data_config_type(_config)
    reset_modules = (
        r".*(HistorySemanticJointActionReadout|"
        r"HistoryRawMemoryQueryResampler|ActionMemoryCrossAttention).*"
    )
    freeze_filter = (
        model.get_freeze_filter_action_finetune()
        if condition_mode == "prompt_only"
        else model.get_freeze_filter_memory_interface_finetune()
    )
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        freeze_filter=freeze_filter,
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=sampler_weights(),
            norm_weights=list(_mixed.SOURCE_PROBABILITIES),
            datasets=_mixed.make_dataset_factories(
                _config,
                data_type,
                min_frame=_stage2.CURRENT_START_FRAME,
                max_frame=_stage2.CURRENT_START_FRAME,
            ),
        ),
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
