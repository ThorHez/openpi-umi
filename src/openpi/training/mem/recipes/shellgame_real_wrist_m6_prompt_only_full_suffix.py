"""Prompt-only full-suffix continuation after the successful frame-241 probe."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_real_mixed_common as _mixed
from openpi.training.mem.recipes import shellgame_real_wrist_m6 as _m6
from openpi.training.mem.recipes import shellgame_real_wrist_m6_prompt_ablation as _ablation
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2

CONFIG_NAME = "pi0_mem_shellgame_real_m6_prompt_only_full_suffix_mixed"
DEFAULT_CHECKPOINT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_mem_shellgame_real_m6_prompt_action_ablation_mixed/"
    "freshmem1800_prompt_only_vs_memory_seed42_v2_prompt_only/800"
)


def _balanced_full_row_counts() -> tuple[int, int]:
    counts = []
    for labels, (training, _), root in zip(
        _mixed.source_final_cups(),
        _ablation.source_episode_splits(),
        (_mixed.OLD_DATASET_ROOT, _mixed.NEW_DATASET_ROOT),
        strict=True,
    ):
        lengths = {
            int(row["episode_index"]): int(row["length"])
            for row in _mixed._load_jsonl(root / "meta/episodes.jsonl")  # noqa: SLF001
        }
        class_rows = [
            sum(
                max(0, lengths[episode] - _stage2.CURRENT_START_FRAME) for episode in training if labels[episode] == cup
            )
            for cup in range(3)
        ]
        counts.append(3 * min(class_rows))
    return counts[0], counts[1]


def _sampler_weights(anchor_fraction: float) -> list[float]:
    if not 0.0 < anchor_fraction < 1.0:
        raise ValueError("anchor_fraction must be strictly between zero and one")
    anchor_counts = _ablation._balanced_source_row_counts()  # noqa: SLF001
    full_counts = _balanced_full_row_counts()
    anchor = [
        probability * anchor_fraction / rows
        for probability, rows in zip(_mixed.SOURCE_PROBABILITIES, anchor_counts, strict=True)
    ]
    full = [
        probability * (1.0 - anchor_fraction) / rows
        for probability, rows in zip(_mixed.SOURCE_PROBABILITIES, full_counts, strict=True)
    ]
    return [*anchor, *full]


def _single_current_frame(factories: list[Any]) -> list[Any]:
    """Use the current image only for the pure Pi0.5 action-training path."""
    return [
        dataclasses.replace(
            factory,
            num_frames=1,
            video_layout="sliding",
            fixed_prefix_frames=0,
        )
        for factory in factories
    ]


def filter_balanced_training_full_validation(dataset, indices: list[int], classifier_config) -> list[int]:
    """Balance all training rows by cup while retaining every selected validation row."""
    del classifier_config
    episodes, _ = _mixed._dataset_episode_and_frame_indices(dataset)  # noqa: SLF001
    episode_count = len({int(value) for value in episodes})
    if episode_count == _mixed.OLD_EPISODES:
        training, validation = _ablation.source_episode_splits()[0]
    elif episode_count == _mixed.NEW_EPISODES:
        training, validation = _ablation.source_episode_splits()[1]
    else:
        raise ValueError(f"Unknown action source with {episode_count} episodes")
    selected_episodes = {int(episodes[index]) for index in indices}
    if selected_episodes <= validation:
        return list(indices)
    if selected_episodes <= training:
        return _mixed.filter_balanced_indices(dataset, indices, None, decision_frame=None)
    raise ValueError("Action subset does not match either fixed training or validation episodes")


def make_train_config(
    *,
    exp_name: str,
    checkpoint: str = str(DEFAULT_CHECKPOINT),
    steps: int = 20_000,
    warmup_steps: int = 150,
    peak_lr: float = 1e-5,
    batch_size: int = 72,
    eval_batch_size: int = 72,
    num_workers: int = 16,
    fsdp_devices: int = 8,
    eval_interval: int = 250,
    eval_batches: int = 2,
    anchor_fraction: float = 0.30,
    save_interval: int = 5_000,
    resume: bool = False,
    overwrite: bool = False,
    inference_mode: bool = False,
) -> Any:
    from openpi.training import config as _config

    if batch_size <= 0 or batch_size % fsdp_devices:
        raise ValueError("batch_size must be positive and divisible by fsdp_devices")
    params_path = Path(checkpoint)
    if params_path.name != "params":
        params_path /= "params"
    model = dataclasses.replace(
        _ablation.build_model_config("prompt_only", inference_mode=inference_mode),
        direction_loss_weight=0.0,
        direction_frame_start=_stage2.CURRENT_START_FRAME,
        direction_frame_end=_stage2.CURRENT_START_FRAME,
        direction_early_stop_metric="",
    )
    data_type = _m6.data_config_type(_config)
    anchor_datasets = _single_current_frame(
        _mixed.make_dataset_factories(
            _config, data_type, min_frame=_stage2.CURRENT_START_FRAME, max_frame=_stage2.CURRENT_START_FRAME
        )
    )
    full_datasets = _single_current_frame(
        _mixed.make_dataset_factories(_config, data_type, min_frame=_stage2.CURRENT_START_FRAME, max_frame=None)
    )
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_action_finetune(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            # These weights are consumed only by the training loader. Avoid
            # reading labels/manifests when reconstructing a deployed policy.
            weights=[1.0, 1.0, 1.0, 1.0] if inference_mode else _sampler_weights(anchor_fraction),
            norm_weights=[
                *(probability * anchor_fraction for probability in _mixed.SOURCE_PROBABILITIES),
                *(probability * (1.0 - anchor_fraction) for probability in _mixed.SOURCE_PROBABILITIES),
            ],
            datasets=[*anchor_datasets, *full_datasets],
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoader(str(params_path)),
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
