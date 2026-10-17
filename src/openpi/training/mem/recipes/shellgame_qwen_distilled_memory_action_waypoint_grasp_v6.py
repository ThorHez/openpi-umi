"""Recover full grasping while preserving the validated MEM-to-XY waypoint.

This stage starts from the frame-59 waypoint checkpoint, freezes the complete
semantic-memory conditioner (including the continuous XY decoder), and trains
only Pi0.5's action expert and action/time projections.  It reuses the exact
V6 data contract that reached 67/100 closed-loop grasp success:

* 60% phase-balanced nominal demonstrations;
* 30% gated recenter/descent correction;
*  5% hold-Z recenter;
*  3% aligned close/grasp; and
*  2% early-lift continuity.

The visual recurrent memory remains an external frozen bank.  Episodes whose
bank predicts the wrong final cup are removed before V6 row sampling, so the
action expert never sees contradictory memory/action supervision.
"""

from __future__ import annotations

import functools
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v6 as _v6
from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v9 as _v9
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.shared import nnx_utils
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_frame59_waypoint as _waypoint
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_v10 as _memory_data


NOMINAL_ROOT = pathlib.Path(_v9.NOMINAL_ROOT).resolve()
V6_ROOT = pathlib.Path(_v9.V6_ROOT).resolve()

DEFAULT_MEMORY_BANKS = {
    NOMINAL_ROOT: _memory_data.DEFAULT_MEMORY_BANKS[NOMINAL_ROOT],
    V6_ROOT: _memory_data.DEFAULT_MEMORY_BANKS[V6_ROOT],
}
DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_distilled_memory_action_frame59_waypoint_eef7_260826/"
    "direct_visual_frame59_waypoint500_4gpu_260826/499/params"
)

SOURCE_FRACTIONS = {NOMINAL_ROOT: 0.60, V6_ROOT: 0.40}
ROWS_PER_CORRECT_EPISODE = {
    NOMINAL_ROOT: _v6.NOMINAL_ROWS_PER_EPISODE,
    V6_ROOT: _v6.CORRECTION_ROWS_PER_EPISODE,
}


def _dataset_root(dataset) -> pathlib.Path:
    return _v9._dataset_repo_id(dataset)  # noqa: SLF001


def filter_correct_and_sample_v6(
    dataset,
    indices: list[int],
    classifier_config,
    *,
    banks: dict[pathlib.Path, pathlib.Path] | None = None,
) -> list[int]:
    """Apply the frozen-MEM correctness gate, then the audited V6 sampler."""
    banks = DEFAULT_MEMORY_BANKS if banks is None else banks
    source = _dataset_root(dataset)
    if source not in banks:
        raise ValueError(f"Unknown waypoint-grasp data source: {source}")

    hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)[selected]
    correct = _memory_data._bank_correctness(str(banks[source].resolve()))  # noqa: SLF001
    if np.any(episodes < 0) or np.any(episodes >= len(correct)):
        raise ValueError(f"Episode index exceeds memory bank for {source}")
    kept = selected[correct[episodes]].tolist()
    logging.info(
        "Waypoint V6 correctness gate source=%s rows=%d->%d episodes=%d->%d",
        source.name,
        len(indices),
        len(kept),
        len(np.unique(episodes)),
        len(np.unique(episodes[correct[episodes]])),
    )
    return _v6._v6_indices(dataset, kept, classifier_config)  # noqa: SLF001


def _correct_episode_count(root: pathlib.Path, banks: dict[pathlib.Path, pathlib.Path]) -> int:
    correct = _memory_data._bank_correctness(str(banks[root].resolve()))  # noqa: SLF001
    return int(np.count_nonzero(correct))


def make_train_config(
    *,
    config_module: Any,
    exp_name: str,
    memory_banks: dict[pathlib.Path, pathlib.Path] | None = None,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    steps: int = 3_000,
    batch_size: int = 12,
    fsdp_devices: int = 6,
    num_workers: int = 8,
    overwrite: bool = False,
):
    banks = {
        pathlib.Path(root).resolve(): pathlib.Path(path).expanduser().resolve()
        for root, path in (DEFAULT_MEMORY_BANKS if memory_banks is None else memory_banks).items()
    }
    init_checkpoint = init_checkpoint.expanduser().resolve()
    for root in (NOMINAL_ROOT, V6_ROOT):
        if not root.is_dir():
            raise FileNotFoundError(root)
        if root not in banks or not banks[root].is_file():
            raise FileNotFoundError(f"No frozen memory bank for {root}: {banks.get(root)}")
    if not init_checkpoint.is_dir():
        raise FileNotFoundError(init_checkpoint)
    if steps < 2:
        raise ValueError("steps must be at least 2")

    correct_counts = {root: _correct_episode_count(root, banks) for root in banks}
    nominal_rows = correct_counts[NOMINAL_ROOT] * ROWS_PER_CORRECT_EPISODE[NOMINAL_ROOT]
    weights = {NOMINAL_ROOT: 1.0}
    v6_rows = correct_counts[V6_ROOT] * ROWS_PER_CORRECT_EPISODE[V6_ROOT]
    weights[V6_ROOT] = (
        SOURCE_FRACTIONS[V6_ROOT]
        / SOURCE_FRACTIONS[NOMINAL_ROOT]
        * nominal_rows
        / v6_rows
    )
    logging.info(
        "Waypoint grasp V6 correct episodes nominal/v6=%d/%d weights=(1,%.9f)",
        correct_counts[NOMINAL_ROOT],
        correct_counts[V6_ROOT],
        weights[V6_ROOT],
    )

    model = _waypoint.make_model_config()
    action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm/.*_1.*")
    action_modules = nnx_utils.PathRegex(
        r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"
    )
    return config_module.TrainConfig(
        name="pi0_shellgame_qwen_distilled_memory_waypoint_grasp_v6_eef7_260826",
        exp_name=exp_name,
        model=model,
        # Freeze the proven memory->waypoint bridge.  Only the local motion
        # policy (Z, rotation, gripper and lift continuity) is adapted here.
        freeze_filter=nnx.Not(nnx.Any(action_expert, action_modules)),
        data=config_module.MultiDataConfigFactory(
            state_pad_dim=96,
            datasets=[
                _memory_data._data_config(config_module, V6_ROOT, banks[V6_ROOT]),  # noqa: SLF001
                _memory_data._data_config(config_module, NOMINAL_ROOT, banks[NOMINAL_ROOT]),  # noqa: SLF001
            ],
            weights=[weights[V6_ROOT], weights[NOMINAL_ROOT]],
            use_merged_norm_stats=False,
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(
            str(init_checkpoint)
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(100, max(steps - 1, 0)),
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
        save_interval=500,
        keep_period=500,
        val_ratio=0.1,
        eval_interval=250,
        eval_batches=20,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=config_module.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=config_module.ShellgameCupEvalConfig(enabled=False),
    )


def make_index_filter(memory_banks: dict[pathlib.Path, pathlib.Path] | None = None):
    banks = {
        pathlib.Path(root).resolve(): pathlib.Path(path).expanduser().resolve()
        for root, path in (DEFAULT_MEMORY_BANKS if memory_banks is None else memory_banks).items()
    }
    return functools.partial(filter_correct_and_sample_v6, banks=banks)
