"""Fine-tune EEF7 actions with dynamic phase-aware V6 corrections.

The global sampled-row recipe is exactly:

* 60% phase-balanced nominal demonstrations;
* 30% gated recenter/descent targets, with every V6 episode contributing
  equally so the designed high/mid/late ratio remains 10/20/70;
*  5% additional hold-Z recenter targets;
*  3% aligned grasp/close targets; and
*  2% early-lift continuity targets.

V6 has a variable-length gated descent.  Consequently, fixed frame ranges do
not identify action phases.  Since the raw7 converter aligns observation i to
raw action i+1, this sampler classifies a row using ``phase_id[i+1]`` rather
than the current observation's phase.  Complete horizon=16 targets are kept;
the aligned phase is used only to choose rows.

The validated tracker, memory interface, current-image reader, normalization,
and model structure remain frozen.  Only Pi0.5's action expert plus action/time
projections are optimized.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v2 as _v2
from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v3 as _v3
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import config as _config
from openpi.training import config_pi0_mem as _config_pi0_mem
from scripts.mem import train_pi0_mem_compress as _trainer

CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816"
CORRECTION_ROOT = (
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_onpolicy_eef_low_stage_gated_v6_balanced1200_260816"
)
DEFAULT_INIT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v5_260816/"
    "absolute_eef7_mixed_correction_v5_balanced1200_60_30_5_5_b12_3k_6gpu_260816/"
    "2999/params"
)

NOMINAL_EPISODES = 5_000
CORRECTION_EPISODES = 1_200
EPISODE_FRAMES = 155
NOMINAL_ROWS_PER_EPISODE = 175
CORRECTION_ROWS_PER_EPISODE = 200
CORRECTION_SAMPLE_FRACTION = 0.40

# WeightedRandomSampler operates after source-specific row filtering.  This
# weight makes the 200-row V6 source contribute exactly 40% globally while the
# 175-row nominal source contributes 60%.
CORRECTION_PER_ROW_WEIGHT = (
    CORRECTION_SAMPLE_FRACTION
    / (1.0 - CORRECTION_SAMPLE_FRACTION)
    * (NOMINAL_EPISODES * NOMINAL_ROWS_PER_EPISODE)
    / (CORRECTION_EPISODES * CORRECTION_ROWS_PER_EPISODE)
)

PHASE_RECENTER = 8
PHASE_DESCEND = 9
PHASE_GRASP = 10
PHASE_LIFT = 11
EARLY_LIFT_STEPS = 10

GROUP_NAMES = (
    "gated_recenter_descent",
    "hold_z_recenter_boost",
    "aligned_grasp",
    "early_lift",
)
GROUP_ROWS_PER_EPISODE = (150, 25, 15, 10)
STAGE_NAMES = np.asarray(("high", "mid", "late"), dtype=object)


def _anchor_stage_ids(episode: np.ndarray) -> np.ndarray:
    """Reconstruct the generator's exact balanced anchor-stage schedule."""
    episode = np.asarray(episode, dtype=np.int64)
    if np.any(episode < 0) or np.any(episode >= CORRECTION_EPISODES):
        raise ValueError("V6 episode index lies outside 0..1199")
    spatial_slot = episode % 3
    within_spatial_slot = episode // 3
    stage_index = (within_spatial_slot * 37 + spatial_slot * 23) % 100
    return np.where(stage_index < 10, 0, np.where(stage_index < 30, 1, 2)).astype(np.int64)


def _resize_each_episode(
    indices: np.ndarray,
    episodes: np.ndarray,
    expected_episodes: np.ndarray,
    rows_per_episode: int,
    *,
    seed: int,
) -> np.ndarray:
    """Give each episode equal mass despite its variable gated-descent length."""
    resized = []
    for episode in expected_episodes:
        group = indices[episodes == episode]
        if group.size == 0:
            raise ValueError(f"V6 sampling group has no rows for episode {int(episode)}")
        resized.append(
            _v3._resize_group(  # noqa: SLF001
                group,
                rows_per_episode,
                seed=seed + int(episode),
            )
        )
    return np.concatenate(resized)


def _v6_indices(dataset, indices: list[int], classifier_config) -> list[int]:
    """Apply nominal balancing or the exact dynamic 30/5/3/2 V6 recipe."""
    del classifier_config
    hf_dataset = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    columns = set(getattr(hf_dataset, "column_names", ()) or ())
    required = {"episode_index", "frame_index", "phase_id", "action_mask"}
    if not required.issubset(columns):
        raise ValueError(f"V6 dynamic sampling requires columns {sorted(required)}")

    nominal_rows = NOMINAL_EPISODES * EPISODE_FRAMES
    correction_rows = CORRECTION_EPISODES * EPISODE_FRAMES
    if len(hf_dataset) == nominal_rows:
        balanced = _full_joint._balanced_full_action_indices(  # noqa: SLF001
            dataset, indices, None
        )
        logging.info(
            "V6 nominal sampling: input=%d balanced=%d rows_per_episode=%d global_mass=60%%",
            len(indices),
            len(balanced),
            NOMINAL_ROWS_PER_EPISODE,
        )
        return balanced
    if len(hf_dataset) != correction_rows:
        raise ValueError(
            f"Unknown V6 dataset length {len(hf_dataset)}; expected "
            f"{nominal_rows} nominal or {correction_rows} correction rows"
        )

    full_episode = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    full_frame = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    full_phase = np.asarray(hf_dataset["phase_id"], dtype=np.int64)
    full_action_mask = np.asarray(hf_dataset["action_mask"], dtype=bool)
    selected = np.asarray(indices, dtype=np.int64)
    frame = full_frame[selected]
    eligible_mask = full_action_mask[selected] & (frame >= 60) & (frame <= 153)
    eligible = selected[eligible_mask]
    if eligible.size == 0:
        raise ValueError("V6 action mask selected no trainable rows")

    # The converter stores observation i -> raw action i+1.  Classify by the
    # immediate target action, while retaining its complete horizon=16 chunk.
    target_index = eligible + 1
    if np.any(target_index >= len(hf_dataset)):
        raise ValueError("V6 aligned target index exceeds the dataset")
    episode = full_episode[eligible]
    target_episode = full_episode[target_index]
    target_frame = full_frame[target_index]
    if not np.array_equal(target_episode, episode):
        raise ValueError("V6 +1 target crosses an episode boundary")
    if not np.array_equal(target_frame, full_frame[eligible] + 1):
        raise ValueError("V6 +1 target is not the consecutive raw frame")
    target_phase = full_phase[target_index]
    episodes = np.unique(episode)

    main_mask = np.isin(target_phase, (PHASE_RECENTER, PHASE_DESCEND))
    recenter_mask = target_phase == PHASE_RECENTER
    grasp_mask = target_phase == PHASE_GRASP

    # Early lift is defined relative to each episode's actual dynamic phase
    # boundary, not by an absolute frame number.
    first_lift_frame = np.full(CORRECTION_EPISODES, EPISODE_FRAMES, dtype=np.int64)
    all_lift = full_phase == PHASE_LIFT
    np.minimum.at(first_lift_frame, full_episode[all_lift], full_frame[all_lift])
    lift_step = target_frame - first_lift_frame[episode]
    early_lift_mask = (target_phase == PHASE_LIFT) & (lift_step >= 0) & (lift_step < EARLY_LIFT_STEPS)

    # Main, grasp, and early-lift groups give every held-in episode equal mass.
    # That preserves the generator's intended high/mid/late episode design and
    # prevents longer high descents from dominating merely because they have
    # more raw phase-9 rows.
    main = _resize_each_episode(
        eligible[main_mask],
        episode[main_mask],
        episodes,
        GROUP_ROWS_PER_EPISODE[0],
        seed=260820,
    )
    recenter = _v3._resize_group(  # noqa: SLF001
        eligible[recenter_mask],
        len(episodes) * GROUP_ROWS_PER_EPISODE[1],
        seed=260821 + len(indices),
    )
    grasp = _resize_each_episode(
        eligible[grasp_mask],
        episode[grasp_mask],
        episodes,
        GROUP_ROWS_PER_EPISODE[2],
        seed=260822,
    )
    early_lift = _resize_each_episode(
        eligible[early_lift_mask],
        episode[early_lift_mask],
        episodes,
        GROUP_ROWS_PER_EPISODE[3],
        seed=260823,
    )
    resized = (main, recenter, grasp, early_lift)
    merged = np.concatenate(resized)
    merged = np.random.default_rng(260824 + len(indices)).permutation(merged)

    expected = len(episodes) * CORRECTION_ROWS_PER_EPISODE
    if len(merged) != expected:
        raise RuntimeError(f"V6 sampler emitted {len(merged)} rows; expected {expected}")
    proportions = tuple(len(group) / len(merged) for group in resized)
    if not np.allclose(proportions, (0.75, 0.125, 0.075, 0.05), atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Unexpected V6 correction proportions: {proportions}")

    stage_ids = _anchor_stage_ids(episodes)
    stage_episode_counts = {
        str(STAGE_NAMES[stage]): int(np.count_nonzero(stage_ids == stage))
        for stage in range(len(STAGE_NAMES))
    }
    main_episode = full_episode[main]
    main_stage_ids = _anchor_stage_ids(main_episode)
    main_stage_rows = {
        str(STAGE_NAMES[stage]): int(np.count_nonzero(main_stage_ids == stage))
        for stage in range(len(STAGE_NAMES))
    }
    logging.info(
        "V6 dynamic phase-aware sampling: episodes=%d stage_episodes=%s "
        "raw_target_phase=%s targets=%s output=%d within_correction=%s "
        "main_stage_rows=%s global_mass={'gated_recenter_descent': 0.30, "
        "'hold_z_recenter': 0.05, 'aligned_grasp': 0.03, 'early_lift': 0.02}",
        len(episodes),
        stage_episode_counts,
        {
            "recenter": int(np.count_nonzero(recenter_mask)),
            "descend": int(np.count_nonzero(target_phase == PHASE_DESCEND)),
            "grasp": int(np.count_nonzero(grasp_mask)),
            "early_lift": int(np.count_nonzero(early_lift_mask)),
        },
        dict(zip(GROUP_NAMES, [len(group) for group in resized], strict=True)),
        len(merged),
        dict(zip(GROUP_NAMES, [round(value, 6) for value in proportions], strict=True)),
        main_stage_rows,
    )
    return merged.tolist()


def build_config(args):
    parent = _v2.build_config(args)
    data = _config.MultiDataConfigFactory(
        state_pad_dim=96,
        datasets=[
            _v2._eef_data_config(CORRECTION_ROOT),  # noqa: SLF001
            _v2._eef_data_config(_v2.NOMINAL_ROOT),  # noqa: SLF001
        ],
        weights=[CORRECTION_PER_ROW_WEIGHT, 1.0],
        # Keep the proven nominal normalization; V6 does not redefine action
        # semantics or state coordinates.
        use_merged_norm_stats=False,
    )
    return dataclasses.replace(
        parent,
        name=CONFIG_NAME,
        exp_name=args.exp_name,
        data=data,
        resume=False,
    )


def main() -> None:
    args = _full_joint.parse_args()
    if args.init_checkpoint == str(_full_joint.OLD_QUERY_ACTION_CHECKPOINT):
        args.init_checkpoint = DEFAULT_INIT_CHECKPOINT
    if args.steps < 2:
        raise ValueError("V6 dynamic phase-aware training requires at least two steps")
    logging.info(
        "V6 exact 60/30/5/3/2 sampling: correction_per_row_weight=%.9f "
        "correction_root=%s init=%s frozen=tracker+memory target_phase=phase_id[i+1]",
        CORRECTION_PER_ROW_WEIGHT,
        CORRECTION_ROOT,
        args.init_checkpoint,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _v6_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
