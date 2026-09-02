"""Train only a bounded current-visual XY correction on top of frozen V6.

The positive source is the audited V6 on-policy correction dataset, restricted
to recenter/descent rows.  Normal nominal-descent rows provide a zero/small
correction reference.  The effective source mixture is 80% correction and 20%
nominal.  Every base-policy and memory parameter remains frozen.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v6 as _v6
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.shared import nnx_utils
from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_visual_correction_action as _correction
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_frame59_waypoint as _waypoint
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_waypoint_grasp_v6 as _grasp
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_v10 as _memory_data


NOMINAL_ROOT = _grasp.NOMINAL_ROOT
V6_ROOT = _grasp.V6_ROOT
DEFAULT_MEMORY_BANKS = _grasp.DEFAULT_MEMORY_BANKS
DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_distilled_memory_waypoint_grasp_v6_eef7_260826/"
    "direct_visual_waypoint_grasp_v6_60_30_5_3_2_3k_6gpu_260826/2000/params"
)

V6_ROWS_PER_EPISODE = 160
NOMINAL_ROWS_PER_EPISODE = 80
SOURCE_FRACTIONS = {V6_ROOT: 0.80, NOMINAL_ROOT: 0.20}


def make_model_config() -> _correction.Pi0QwenEventMemoryWaypointVisualCorrectionActionConfig:
    base = _waypoint.make_model_config()
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    return _correction.Pi0QwenEventMemoryWaypointVisualCorrectionActionConfig(
        **values,
        correction_state_dim=10,
        correction_hidden_width=256,
        correction_normalized_limits=(0.132, 0.082),
        correction_scale=1.0,
    )


def _resize_each_episode(
    indices: np.ndarray,
    episodes: np.ndarray,
    rows_per_episode: int,
    *,
    seed: int,
) -> np.ndarray:
    output = []
    for episode in np.unique(episodes):
        rows = indices[episodes == episode]
        if rows.size == 0:
            raise ValueError(f"No visual-correction rows for episode {int(episode)}")
        output.append(
            _v6._resize_each_episode(  # noqa: SLF001
                rows,
                np.full(rows.shape, episode, dtype=np.int64),
                np.asarray([episode], dtype=np.int64),
                rows_per_episode,
                seed=seed,
            )
        )
    return np.concatenate(output)


def filter_correct_visual_correction_rows(
    dataset,
    indices: list[int],
    classifier_config,
    *,
    banks: dict[pathlib.Path, pathlib.Path] | None = None,
) -> list[int]:
    """Select causal oracle-XY rows after the frozen-memory correctness gate."""
    del classifier_config
    banks = DEFAULT_MEMORY_BANKS if banks is None else banks
    source = _grasp._dataset_root(dataset)  # noqa: SLF001
    if source not in (V6_ROOT, NOMINAL_ROOT):
        raise ValueError(f"Unknown visual-correction source: {source}")

    hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    columns = set(getattr(hf, "column_names", ()) or ())
    required = {"episode_index", "frame_index", "phase_id", "action_mask"}
    if not required.issubset(columns):
        raise ValueError(f"Visual correction requires columns {sorted(required)}")

    selected = np.asarray(indices, dtype=np.int64)
    full_episode = np.asarray(hf["episode_index"], dtype=np.int64)
    full_frame = np.asarray(hf["frame_index"], dtype=np.int64)
    full_phase = np.asarray(hf["phase_id"], dtype=np.int64)
    full_action_mask = np.asarray(hf["action_mask"], dtype=bool)
    correct = _memory_data._bank_correctness(str(banks[source].resolve()))  # noqa: SLF001
    selected = selected[correct[full_episode[selected]]]

    frame = full_frame[selected]
    eligible = selected[full_action_mask[selected] & (frame >= 60) & (frame <= 153)]
    target = eligible + 1
    same_episode = target < len(hf)
    same_episode &= full_episode[target.clip(max=len(hf) - 1)] == full_episode[eligible]
    eligible = eligible[same_episode]
    target = eligible + 1

    if source == V6_ROOT:
        keep = np.isin(full_phase[target], (_v6.PHASE_RECENTER, _v6.PHASE_DESCEND))
        rows_per_episode = V6_ROWS_PER_EPISODE
        seed = 26082710
        phase_name = "recenter+descend"
    else:
        # Nominal phase 5 is the ordinary aligned descent.  It prevents the
        # scratch head from learning an unconditional slot-specific offset.
        keep = full_phase[target] == 5
        rows_per_episode = NOMINAL_ROWS_PER_EPISODE
        seed = 26082720
        phase_name = "nominal_descend"
    eligible = eligible[keep]
    if eligible.size == 0:
        raise ValueError(f"No eligible {phase_name} rows for {source}")
    episodes = full_episode[eligible]
    resized = _resize_each_episode(eligible, episodes, rows_per_episode, seed=seed)
    resized = np.random.default_rng(seed + len(indices)).permutation(resized)
    logging.info(
        "Visual correction sampler source=%s phase=%s input=%d correct_phase_rows=%d "
        "episodes=%d output=%d rows_per_episode=%d",
        source.name,
        phase_name,
        len(indices),
        len(eligible),
        len(np.unique(episodes)),
        len(resized),
        rows_per_episode,
    )
    return resized.tolist()


def make_train_config(
    *,
    config_module: Any,
    exp_name: str,
    memory_banks: dict[pathlib.Path, pathlib.Path] | None = None,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    steps: int = 500,
    batch_size: int = 64,
    fsdp_devices: int = 4,
    num_workers: int = 8,
    overwrite: bool = False,
):
    banks = {
        pathlib.Path(root).resolve(): pathlib.Path(path).expanduser().resolve()
        for root, path in (DEFAULT_MEMORY_BANKS if memory_banks is None else memory_banks).items()
    }
    init_checkpoint = init_checkpoint.expanduser().resolve()
    for root in (V6_ROOT, NOMINAL_ROOT):
        if not root.is_dir() or root not in banks or not banks[root].is_file():
            raise FileNotFoundError(f"Missing dataset or memory bank: {root}, {banks.get(root)}")
    if not init_checkpoint.is_dir():
        raise FileNotFoundError(init_checkpoint)
    if steps < 2:
        raise ValueError("steps must be at least 2")

    correct_counts = {
        root: int(np.count_nonzero(_memory_data._bank_correctness(str(banks[root]))))  # noqa: SLF001
        for root in (V6_ROOT, NOMINAL_ROOT)
    }
    selected_rows = {
        V6_ROOT: correct_counts[V6_ROOT] * V6_ROWS_PER_EPISODE,
        NOMINAL_ROOT: correct_counts[NOMINAL_ROOT] * NOMINAL_ROWS_PER_EPISODE,
    }
    weights = {V6_ROOT: 1.0}
    weights[NOMINAL_ROOT] = (
        SOURCE_FRACTIONS[NOMINAL_ROOT]
        / SOURCE_FRACTIONS[V6_ROOT]
        * selected_rows[V6_ROOT]
        / selected_rows[NOMINAL_ROOT]
    )
    logging.info(
        "Visual correction 80/20 rows=%s weights=(1,%.9f)",
        {root.name: selected_rows[root] for root in selected_rows},
        weights[NOMINAL_ROOT],
    )

    model = make_model_config()
    correction_head = nnx_utils.PathRegex(r".*CurrentVisualXYCorrectionHead.*")
    return config_module.TrainConfig(
        name="pi0_shellgame_qwen_distilled_memory_waypoint_visual_correction_v6_eef7_260827",
        exp_name=exp_name,
        model=model,
        # The correction head is the only trainable module. At initialization
        # its zero output reproduces the validated V6 hard-anchor policy.
        freeze_filter=nnx.Not(correction_head),
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
            warmup_steps=min(50, max(steps - 1, 0)),
            peak_lr=1e-4,
            decay_steps=max(steps, 2),
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        log_interval=10,
        save_interval=100,
        keep_period=500,
        val_ratio=0.1,
        eval_interval=50,
        eval_batches=30,
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
    return functools.partial(filter_correct_visual_correction_rows, banks=banks)
