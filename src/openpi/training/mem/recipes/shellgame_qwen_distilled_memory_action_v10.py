"""V10 action training driven by frozen Qwen-distilled visual MEM tokens.

The memory network is evaluated once on frames 0..59 for every episode.  This
recipe then freezes those [128,64] tokens, supplies the live current image and
state to Pi0, and reuses the validated V10 nominal/V6/V9 action sampling mix.
Episodes whose visual memory predicts the wrong final cup are excluded so the
action expert never receives contradictory memory/action supervision.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import pathlib
from typing import Any

import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v10_timing_diag as _v10
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_qwen_event_memory_action as _base


NOMINAL_ROOT = pathlib.Path(_v10.NOMINAL_ROOT).resolve()
V6_ROOT = pathlib.Path(_v10.V6_ROOT).resolve()
V9_ROOT = pathlib.Path(_v10.V9_ROOT).resolve()

ARTIFACT_ROOT = pathlib.Path("/data2/hzl_workspace_for_pi_mem/openpi-umi/artifacts")
DEFAULT_MEMORY_BANKS = {
    NOMINAL_ROOT: ARTIFACT_ROOT / "shellgame_qwen_distilled_direct_visual_memory_step999_all5000_260825.npz",
    V6_ROOT: ARTIFACT_ROOT / "shellgame_qwen_distilled_direct_visual_memory_step999_v6_all1200_260826.npz",
    V9_ROOT: ARTIFACT_ROOT / "shellgame_qwen_distilled_direct_visual_memory_step999_v9_all1200_260826.npz",
}
DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_event_memory_action_eef7_260825/"
    "direct_visual_mem_step999_filtered_action250_6gpu_260825/249/params"
)

SOURCE_FRACTIONS = {NOMINAL_ROOT: 0.60, V6_ROOT: 0.30, V9_ROOT: 0.10}
ROWS_PER_CORRECT_EPISODE = {
    NOMINAL_ROOT: _v10.NOMINAL_ROWS_PER_EPISODE,
    V6_ROOT: _v10.V6_ROWS_PER_EPISODE,
    V9_ROOT: _v10.V9_ROWS_PER_EPISODE,
}


@functools.lru_cache(maxsize=None)
def _bank_correctness(path_string: str) -> np.ndarray:
    path = pathlib.Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen memory bank: {path}")
    with np.load(path, allow_pickle=False) as source:
        episodes = np.asarray(source["episode_index"], dtype=np.int64)
        prediction = np.asarray(source["final_prediction"], dtype=np.int64)
        label = np.asarray(source["final_label"], dtype=np.int64)
    dense = np.zeros(int(np.max(episodes)) + 1, dtype=bool)
    dense[episodes] = prediction == label
    if len(np.unique(episodes)) != len(episodes):
        raise ValueError(f"Duplicate episode indices in {path}")
    return dense


def _correct_episode_count(root: pathlib.Path, banks: dict[pathlib.Path, pathlib.Path]) -> int:
    return int(np.count_nonzero(_bank_correctness(str(banks[root].resolve()))))


def filter_and_sample_indices(
    dataset,
    indices: list[int],
    classifier_config,
    *,
    banks: dict[pathlib.Path, pathlib.Path] | None = None,
) -> list[int]:
    """Drop incorrect-memory episodes, then apply the unmodified V10 sampler."""
    banks = DEFAULT_MEMORY_BANKS if banks is None else banks
    source = _v10._v9._dataset_repo_id(dataset)  # noqa: SLF001
    if source not in banks:
        raise ValueError(f"Unknown frozen-MEM action source: {source}")
    hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)[selected]
    correct = _bank_correctness(str(banks[source].resolve()))
    if np.any(episodes < 0) or np.any(episodes >= len(correct)):
        raise ValueError(f"Episode index exceeds memory bank for {source}")
    kept = selected[correct[episodes]].tolist()
    logging.info(
        "Frozen-MEM correctness filter source=%s rows=%d->%d episodes=%d->%d",
        source.name,
        len(indices),
        len(kept),
        len(np.unique(episodes)),
        len(np.unique(episodes[correct[episodes]])),
    )
    return _v10._indices(dataset, kept, classifier_config)  # noqa: SLF001


def _data_config(
    config_module: Any,
    root: pathlib.Path,
    memory_path: pathlib.Path,
):
    cls = _base.data_config_type(config_module)
    return cls(
        repo_id=str(root),
        memory_path=str(memory_path.resolve()),
        # Retain the nominal EEF normalization contract used by every V10
        # source.  Correction datasets intentionally do not redefine stats.
        assets=config_module.AssetsConfig(asset_id=".", assets_dir=str(NOMINAL_ROOT)),
        base_config=config_module.UmiDataConfig(
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
            robot_type="ARM=1 G=0 H=0",
        ),
        num_frames=1,
        frame_stride=1,
        video_layout="sliding",
        # Keep the dataset's native global row numbering.  The proven V10
        # sampler below performs the frame/phase filtering itself; pre-slicing
        # here would turn its global HF indices into incompatible local ones.
    )


def make_train_config(
    *,
    config_module: Any,
    exp_name: str,
    memory_banks: dict[pathlib.Path, pathlib.Path] | None = None,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    steps: int = 500,
    batch_size: int = 12,
    fsdp_devices: int = 6,
    num_workers: int = 8,
    overwrite: bool = False,
):
    banks = {
        pathlib.Path(root).resolve(): pathlib.Path(path).resolve()
        for root, path in (DEFAULT_MEMORY_BANKS if memory_banks is None else memory_banks).items()
    }
    for root in (NOMINAL_ROOT, V6_ROOT, V9_ROOT):
        if not root.is_dir():
            raise FileNotFoundError(root)
        if root not in banks or not banks[root].is_file():
            raise FileNotFoundError(f"No memory bank for {root}: {banks.get(root)}")
    if not init_checkpoint.is_dir():
        raise FileNotFoundError(init_checkpoint)

    correct_counts = {root: _correct_episode_count(root, banks) for root in banks}
    nominal_rows = correct_counts[NOMINAL_ROOT] * ROWS_PER_CORRECT_EPISODE[NOMINAL_ROOT]
    weights = {}
    for root in (NOMINAL_ROOT, V6_ROOT, V9_ROOT):
        rows = correct_counts[root] * ROWS_PER_CORRECT_EPISODE[root]
        weights[root] = (
            SOURCE_FRACTIONS[root]
            / SOURCE_FRACTIONS[NOMINAL_ROOT]
            * nominal_rows
            / rows
        )
    logging.info(
        "Direct-visual frozen MEM V10 correct episodes nominal/v6/v9=%d/%d/%d; weights=%s",
        correct_counts[NOMINAL_ROOT],
        correct_counts[V6_ROOT],
        correct_counts[V9_ROOT],
        {root.name: weights[root] for root in weights},
    )

    model = _base.make_model_config()
    return config_module.TrainConfig(
        name="pi0_shellgame_qwen_distilled_memory_action_v10_eef7_260826",
        exp_name=exp_name,
        model=model,
        # MEM tokens are external and frozen.  Train the already-validated
        # memory/action conditioner together with Pi0's action expert.
        freeze_filter=model.get_freeze_filter_action_finetune(),
        data=config_module.MultiDataConfigFactory(
            state_pad_dim=96,
            datasets=[
                _data_config(config_module, V9_ROOT, banks[V9_ROOT]),
                _data_config(config_module, V6_ROOT, banks[V6_ROOT]),
                _data_config(config_module, NOMINAL_ROOT, banks[NOMINAL_ROOT]),
            ],
            weights=[weights[V9_ROOT], weights[V6_ROOT], weights[NOMINAL_ROOT]],
            use_merged_norm_stats=False,
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(
            str(init_checkpoint.resolve())
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(50, max(steps - 1, 0)),
            peak_lr=3e-6,
            decay_steps=max(steps, 2),
            decay_lr=3e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        log_interval=10,
        save_interval=250,
        keep_period=250,
        val_ratio=0.1,
        eval_interval=125,
        eval_batches=20,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=config_module.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=config_module.ShellgameCupEvalConfig(enabled=False),
    )
