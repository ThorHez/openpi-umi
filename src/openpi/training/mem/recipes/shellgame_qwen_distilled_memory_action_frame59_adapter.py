"""Frame-59-only alignment of frozen visual MEM to a frozen Pi action expert.

This is a causal control experiment for the ShellGame selection failure.  It
keeps the validated absolute-EEF action expert fixed and updates only the
external-memory conditioner on the first post-observation action chunk.  No
cup label or classifier loss is used: supervision remains the ordinary Pi
flow-matching action loss stored at dataset frame 59.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import flax.nnx as nnx
import numpy as np

from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.shared import nnx_utils
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_v10 as _v10
from openpi.training.mem.recipes import shellgame_qwen_event_memory_action as _model_base


NOMINAL_ROOT = _v10.NOMINAL_ROOT
DEFAULT_MEMORY = _v10.DEFAULT_MEMORY_BANKS[NOMINAL_ROOT]
DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_event_memory_action_eef7_260825/"
    "direct_visual_mem_step999_filtered_action250_6gpu_260825/249/params"
)


def filter_frame59_correct_indices(
    dataset,
    indices: list[int],
    classifier_config,
    *,
    memory_path: pathlib.Path = DEFAULT_MEMORY,
) -> list[int]:
    """Keep one frame-59 selection row for every semantically correct episode."""
    del classifier_config
    hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)[selected]
    frames = np.asarray(hf["frame_index"], dtype=np.int64)[selected]
    correct = _v10._bank_correctness(str(memory_path.resolve()))  # noqa: SLF001
    if np.any(episodes < 0) or np.any(episodes >= len(correct)):
        raise ValueError("Episode index exceeds the frozen visual-memory bank")
    keep = (frames == 59) & correct[episodes]
    filtered = selected[keep]
    if len(filtered) == 0:
        raise ValueError("Frame-59 adapter training selected no rows")
    kept_episodes = episodes[keep]
    if len(np.unique(kept_episodes)) != len(kept_episodes):
        raise ValueError("Expected exactly one frame-59 row per selected episode")
    logging.info(
        "Frame59 adapter rows=%d->%d episodes=%d; memory=%s",
        len(indices),
        len(filtered),
        len(kept_episodes),
        memory_path,
    )
    return np.random.default_rng(261201 + len(filtered)).permutation(filtered).tolist()


def make_train_config(
    *,
    config_module,
    exp_name: str,
    memory_path: pathlib.Path = DEFAULT_MEMORY,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    steps: int = 500,
    batch_size: int = 12,
    fsdp_devices: int = 4,
    num_workers: int = 8,
    overwrite: bool = False,
):
    memory_path = memory_path.expanduser().resolve()
    init_checkpoint = init_checkpoint.expanduser().resolve()
    if not memory_path.is_file():
        raise FileNotFoundError(memory_path)
    if not init_checkpoint.is_dir():
        raise FileNotFoundError(init_checkpoint)

    # Preserve the same tensor topology as the existing checkpoint while
    # removing the shortcut that drops the complete memory residual.
    model = dataclasses.replace(
        _model_base.make_model_config(),
        semantic_residual_dropout_rate=0.0,
    )
    memory_adapter = nnx_utils.PathRegex(r".*SemanticMemoryActionConditioner.*")
    return config_module.TrainConfig(
        name="pi0_shellgame_qwen_distilled_memory_action_frame59_adapter_eef7_260826",
        exp_name=exp_name,
        model=model,
        # freeze_filter selects frozen leaves; only the conditioner remains
        # trainable.  The Pi action expert, image encoder and projections are
        # therefore bit-for-bit fixed throughout this alignment stage.
        freeze_filter=nnx.Not(memory_adapter),
        data=config_module.MultiDataConfigFactory(
            state_pad_dim=96,
            datasets=[_v10._data_config(config_module, NOMINAL_ROOT, memory_path)],  # noqa: SLF001
            weights=[1.0],
            use_merged_norm_stats=False,
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderIgnoreGripperHead(
            str(init_checkpoint)
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(50, max(steps - 1, 0)),
            peak_lr=3e-5,
            decay_steps=max(steps, 2),
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        log_interval=10,
        save_interval=100,
        keep_period=100,
        val_ratio=0.1,
        eval_interval=50,
        eval_batches=30,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=config_module.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=config_module.ShellgameCupEvalConfig(enabled=False),
    )

