"""Small-step action-only adaptation on audited current-policy recovery data.

The frozen semantic memory and memory-to-waypoint bridge are inherited from
the validated current-action V6 step-2000 checkpoint.  Only Pi0.5's action
expert and action/time projections are updated.  Effective sample mass is:

* 50% nominal demonstrations (behavior preservation);
* 20% validated V6 gated corrections (recovery preservation); and
* 30% V11 states reached by the current policy, followed by coherent Oracle
  recenter, descent, grasp, and lift suffixes.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import pathlib
from typing import Any

import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v6 as _v6
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import optimizer as _optimizer
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_v10 as _memory_data
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_waypoint_grasp_v6 as _grasp

V11_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_current_action_low_stage_recovery_v11_306ep_260827"
).resolve()
V11_MEMORY = V11_ROOT / "frozen_direct_visual_memory_v11.npz"
V11_AUDIT = V11_ROOT / "current_action_low_stage_recovery_v11_audit.json"
DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_distilled_memory_waypoint_grasp_v6_eef7_260826/"
    "direct_visual_waypoint_grasp_v6_60_30_5_3_2_3k_6gpu_260826/2000/params"
)

NOMINAL_ROOT = _grasp.NOMINAL_ROOT
V6_ROOT = _grasp.V6_ROOT
V11_EPISODES = 306
V11_ROWS_PER_EPISODE = 24
V11_GROUP_ROWS = (16, 4, 4)
SOURCE_FRACTIONS = {NOMINAL_ROOT: 0.50, V6_ROOT: 0.20, V11_ROOT: 0.30}


def _resize_each_episode(
    indices: np.ndarray,
    episodes: np.ndarray,
    expected_episodes: np.ndarray,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    return _v6._resize_each_episode(  # noqa: SLF001
        indices,
        episodes,
        expected_episodes,
        count,
        seed=seed,
    )


def _sample_v11(dataset, indices: list[int]) -> list[int]:
    """Sample coherent target phases using the aligned action at row i+1."""
    hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    required = {"episode_index", "frame_index", "phase_id", "action_mask"}
    if not required.issubset(set(getattr(hf, "column_names", ()) or ())):
        raise ValueError(f"V11 sampling requires columns {sorted(required)}")

    full_episode = np.asarray(hf["episode_index"], dtype=np.int64)
    full_frame = np.asarray(hf["frame_index"], dtype=np.int64)
    full_phase = np.asarray(hf["phase_id"], dtype=np.int64)
    full_mask = np.asarray(hf["action_mask"], dtype=bool)
    selected = np.asarray(indices, dtype=np.int64)
    frame = full_frame[selected]
    eligible = selected[full_mask[selected] & (frame >= 60) & (frame <= 154)]
    if eligible.size == 0:
        raise ValueError("V11 split has no eligible Oracle rows")
    target = eligible + 1
    if np.any(target >= len(hf)) or np.any(full_episode[target] != full_episode[eligible]):
        raise ValueError("V11 +1 action target crosses an episode boundary")
    if np.any(full_frame[target] != full_frame[eligible] + 1):
        raise ValueError("V11 +1 action target is not consecutive")

    episode = full_episode[eligible]
    target_phase = full_phase[target]
    expected = np.unique(episode)
    first_lift = np.full(V11_EPISODES, 1_000, dtype=np.int64)
    lift_rows = full_phase == _v6.PHASE_LIFT
    np.minimum.at(first_lift, full_episode[lift_rows], full_frame[lift_rows])
    lift_step = full_frame[target] - first_lift[episode]
    masks = (
        np.isin(target_phase, (_v6.PHASE_RECENTER, _v6.PHASE_DESCEND)),
        target_phase == _v6.PHASE_GRASP,
        (target_phase == _v6.PHASE_LIFT) & (lift_step >= 0) & (lift_step < 10),
    )
    names = ("sustained_recenter_descent", "aligned_grasp", "early_lift")
    groups = tuple(
        _resize_each_episode(
            eligible[mask],
            episode[mask],
            expected,
            count,
            seed=26082730 + group_index * 1_000,
        )
        for group_index, (mask, count) in enumerate(zip(masks, V11_GROUP_ROWS, strict=True))
    )
    merged = np.concatenate(groups)
    required_rows = len(expected) * V11_ROWS_PER_EPISODE
    if len(merged) != required_rows:
        raise RuntimeError(f"V11 sampler emitted {len(merged)}, expected {required_rows}")
    logging.info(
        "V11 coherent sampler split_episodes=%d groups=%s unique_rows=%d output=%d",
        len(expected),
        dict(zip(names, [len(group) for group in groups], strict=True)),
        len(np.unique(merged)),
        len(merged),
    )
    return np.random.default_rng(26082799 + len(selected)).permutation(merged).tolist()


def filter_and_sample(
    dataset,
    indices: list[int],
    classifier_config,
    *,
    banks: dict[pathlib.Path, pathlib.Path],
) -> list[int]:
    source = _grasp._dataset_root(dataset)  # noqa: SLF001
    if source == V11_ROOT:
        correct = _memory_data._bank_correctness(str(banks[source].resolve()))  # noqa: SLF001
        hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
        selected = np.asarray(indices, dtype=np.int64)
        episode = np.asarray(hf["episode_index"], dtype=np.int64)[selected]
        if np.any(~correct[episode]):
            raise ValueError("V11 audit promised correct memory for every episode")
        return _sample_v11(dataset, indices)
    return _grasp.filter_correct_and_sample_v6(
        dataset,
        indices,
        classifier_config,
        banks=banks,
    )


def _validate_v11() -> None:
    if not V11_ROOT.is_dir() or not V11_MEMORY.is_file() or not V11_AUDIT.is_file():
        raise FileNotFoundError(f"Incomplete converted V11 dataset: {V11_ROOT}")
    audit = json.loads(V11_AUDIT.read_text(encoding="utf-8"))
    if audit.get("ok") is not True:
        raise ValueError("V11 conversion audit did not pass")
    if audit.get("raw_audit", {}).get("episodes") != V11_EPISODES:
        raise ValueError("V11 audit has the wrong episode count")
    if audit.get("raw_audit", {}).get("model_generated_actions_supervised") is not False:
        raise ValueError("V11 audit does not prove Oracle-only supervision")
    if audit.get("memory_bank_audit", {}).get("all_predictions_correct") is not True:
        raise ValueError("V11 remapped memory bank did not pass correctness audit")


def make_train_config(
    *,
    config_module: Any,
    exp_name: str,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    steps: int = 1_000,
    peak_lr: float = 1e-6,
    batch_size: int = 12,
    fsdp_devices: int = 6,
    num_workers: int = 8,
    overwrite: bool = False,
):
    _validate_v11()
    init_checkpoint = init_checkpoint.expanduser().resolve()
    if not init_checkpoint.is_dir():
        raise FileNotFoundError(init_checkpoint)
    if steps < 2 or peak_lr <= 0:
        raise ValueError("steps must be >=2 and peak_lr must be positive")

    banks = dict(_grasp.DEFAULT_MEMORY_BANKS)
    banks[V11_ROOT] = V11_MEMORY
    correct_counts = {
        root: int(np.count_nonzero(_memory_data._bank_correctness(str(bank.resolve()))))  # noqa: SLF001
        for root, bank in banks.items()
    }
    selected_rows = {
        NOMINAL_ROOT: correct_counts[NOMINAL_ROOT] * _grasp.ROWS_PER_CORRECT_EPISODE[NOMINAL_ROOT],
        V6_ROOT: correct_counts[V6_ROOT] * _grasp.ROWS_PER_CORRECT_EPISODE[V6_ROOT],
        V11_ROOT: correct_counts[V11_ROOT] * V11_ROWS_PER_EPISODE,
    }
    weights = {NOMINAL_ROOT: 1.0}
    for root in (V6_ROOT, V11_ROOT):
        weights[root] = (
            SOURCE_FRACTIONS[root]
            / SOURCE_FRACTIONS[NOMINAL_ROOT]
            * selected_rows[NOMINAL_ROOT]
            / selected_rows[root]
        )
    logging.info(
        "V11 action adaptation source fractions=50/20/30 selected_rows=%s weights=%s",
        {root.name: selected_rows[root] for root in selected_rows},
        {root.name: weights[root] for root in weights},
    )

    parent = _grasp.make_train_config(
        config_module=config_module,
        exp_name=exp_name,
        memory_banks={NOMINAL_ROOT: banks[NOMINAL_ROOT], V6_ROOT: banks[V6_ROOT]},
        init_checkpoint=init_checkpoint,
        steps=steps,
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        num_workers=num_workers,
        overwrite=overwrite,
    )
    return dataclasses.replace(
        parent,
        name="pi0_shellgame_qwen_memory_action_recovery_v11_eef7_260827",
        data=config_module.MultiDataConfigFactory(
            state_pad_dim=96,
            datasets=[
                _memory_data._data_config(config_module, V11_ROOT, banks[V11_ROOT]),  # noqa: SLF001
                _memory_data._data_config(config_module, V6_ROOT, banks[V6_ROOT]),  # noqa: SLF001
                _memory_data._data_config(config_module, NOMINAL_ROOT, banks[NOMINAL_ROOT]),  # noqa: SLF001
            ],
            weights=[weights[V11_ROOT], weights[V6_ROOT], weights[NOMINAL_ROOT]],
            use_merged_norm_stats=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(100, max(steps - 1, 0)),
            peak_lr=peak_lr,
            decay_steps=max(steps, 2),
            decay_lr=peak_lr * 0.1,
        ),
        save_interval=250,
        keep_period=250,
        eval_interval=125,
        eval_batches=20,
    )


def make_index_filter():
    banks = dict(_grasp.DEFAULT_MEMORY_BANKS)
    banks[V11_ROOT] = V11_MEMORY
    return functools.partial(filter_and_sample, banks=banks)
