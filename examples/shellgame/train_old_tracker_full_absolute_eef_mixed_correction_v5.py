"""Train EEF7 actions with the balanced V4 continuous-correction dataset.

The global sampled-row recipe is exactly:

* 60% nominal demonstrations;
* 30% complete continuously-centred recenter/descent trajectories;
*  5% additional high-approach recenter states; and
*  5% grasp/lift continuity states.

The proven tracker, memory interface, current-image reader, normalization, and
model structure stay frozen.  Only Pi0.5's action expert plus action/time
projections are optimized.  Frame groups refer to the observation whose label
is the following raw controller command, matching the converter's +1 causal
alignment.
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

CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v5_260816"
CORRECTION_ROOT = (
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_onpolicy_eef_continuous_descent_v4_balanced1200_260815"
)
DEFAULT_INIT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v4_260814/"
    "absolute_eef7_mixed_correction_v4_holdz25pct_b12_2k_6gpu_260814/1999/params"
)

NOMINAL_EPISODES = 5_000
CORRECTION_EPISODES = 1_200
EPISODE_FRAMES = 155
NOMINAL_ROWS_PER_EPISODE = 175
CORRECTION_ROWS_PER_EPISODE = 200
CORRECTION_SAMPLE_FRACTION = 0.40

# The correction sampler emits 200 rows/episode in a 150/25/25 mixture.
# Solve wc*Nc / (wc*Nc + Nn) = 0.40 exactly after source-specific filtering.
CORRECTION_PER_ROW_WEIGHT = (
    CORRECTION_SAMPLE_FRACTION
    / (1.0 - CORRECTION_SAMPLE_FRACTION)
    * (NOMINAL_EPISODES * NOMINAL_ROWS_PER_EPISODE)
    / (CORRECTION_EPISODES * CORRECTION_ROWS_PER_EPISODE)
)

GROUP_NAMES = ("continuous_recenter_descent", "high_recenter_boost", "grasp_lift")
GROUP_ROWS_PER_EPISODE = (150, 25, 25)
PREFIX_STEPS = np.asarray((18, 24, 30, 36, 39, 42), dtype=np.int64)
HIGH_PREFIX_STEPS = (18, 24)


def _prefix_steps_for_episode(episode: np.ndarray) -> np.ndarray:
    """Reconstruct the generator's balanced prefix schedule from episode ID."""
    episode = np.asarray(episode, dtype=np.int64)
    spatial_slot = episode % 3
    within_spatial_slot = episode // 3
    prefix_index = (within_spatial_slot + 2 * spatial_slot) % len(PREFIX_STEPS)
    return PREFIX_STEPS[prefix_index]


def _v5_indices(dataset, indices: list[int], classifier_config) -> list[int]:
    """Apply nominal balancing or the exact 30/5/5 correction-row recipe."""
    del classifier_config
    hf_dataset = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    columns = set(getattr(hf_dataset, "column_names", ()) or ())
    required = {"episode_index", "frame_index", "action_mask"}
    if not required.issubset(columns):
        raise ValueError(f"V5 mixed sampling requires columns {sorted(required)}")

    nominal_rows = NOMINAL_EPISODES * EPISODE_FRAMES
    correction_rows = CORRECTION_EPISODES * EPISODE_FRAMES
    if len(hf_dataset) == nominal_rows:
        balanced = _full_joint._balanced_full_action_indices(  # noqa: SLF001
            dataset, indices, None
        )
        logging.info(
            "V5 nominal sampling: input=%d balanced=%d rows_per_episode=%d global_mass=60%%",
            len(indices),
            len(balanced),
            NOMINAL_ROWS_PER_EPISODE,
        )
        return balanced
    if len(hf_dataset) != correction_rows:
        raise ValueError(
            f"Unknown V5 dataset length {len(hf_dataset)}; expected "
            f"{nominal_rows} nominal or {correction_rows} correction rows"
        )

    selected = np.asarray(indices, dtype=np.int64)
    episode = np.asarray(hf_dataset["episode_index"], dtype=np.int64)[selected]
    frame = np.asarray(hf_dataset["frame_index"], dtype=np.int64)[selected]
    action_mask = np.asarray(hf_dataset["action_mask"], dtype=bool)[selected]
    eligible = action_mask & (frame >= 60) & (frame <= 153)
    prefix_steps = _prefix_steps_for_episode(episode)

    # Frames name the observation; the supervised command is raw frame+1.
    # 60..112 covers recenter commands 61..63 and descent commands 64..113.
    # High recenter is deliberately an overlapping boost on frames 60..62.
    # 113..153 predicts grasp/lift commands 114..154.
    groups = (
        selected[eligible & (frame >= 60) & (frame <= 112)],
        selected[eligible & (frame >= 60) & (frame <= 62) & np.isin(prefix_steps, HIGH_PREFIX_STEPS)],
        selected[eligible & (frame >= 113) & (frame <= 153)],
    )
    if any(len(group) == 0 for group in groups):
        raise ValueError(f"Empty V5 correction group: {[len(group) for group in groups]}")

    num_episodes = len(np.unique(episode[eligible]))
    targets = [num_episodes * rows for rows in GROUP_ROWS_PER_EPISODE]
    resized = [
        _v3._resize_group(group, target, seed=260816 + len(indices) + group_index)  # noqa: SLF001
        for group_index, (group, target) in enumerate(zip(groups, targets, strict=True))
    ]
    merged = np.concatenate(resized)
    merged = np.random.default_rng(260816 + len(indices)).permutation(merged)
    proportions = [target / len(merged) for target in targets]
    if not np.allclose(proportions, (0.75, 0.125, 0.125), atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Unexpected V5 correction proportions: {proportions}")
    logging.info(
        "V5 correction sampling: episodes=%d raw=%s targets=%s output=%d "
        "within_correction=%s global_mass={'descent': 0.30, 'high_recenter': 0.05, "
        "'grasp_lift': 0.05}",
        num_episodes,
        dict(zip(GROUP_NAMES, [len(group) for group in groups], strict=True)),
        dict(zip(GROUP_NAMES, targets, strict=True)),
        len(merged),
        dict(zip(GROUP_NAMES, [round(value, 6) for value in proportions], strict=True)),
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
        raise ValueError("V5 balanced correction training requires at least two steps")
    logging.info(
        "V5 exact 60/30/5/5 sampling: correction_per_row_weight=%.9f correction_root=%s init=%s frozen=tracker+memory",
        CORRECTION_PER_ROW_WEIGHT,
        CORRECTION_ROOT,
        args.init_checkpoint,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _v5_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
