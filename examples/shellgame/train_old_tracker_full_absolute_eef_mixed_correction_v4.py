"""Fine-tune EEF7 actions on strict multi-height hold-Z corrections.

V4 keeps the validated frozen tracker and action architecture.  Relative to
V3, only the correction dataset and sampling distribution change:

* correction source mass increases from 15% to 25%;
* switch observations receive 7.5% of all sampled rows;
* active recenter observations receive another 7.5%; and
* every recenter label comes from a current-V3 on-policy state and forbids Z
  descent until measured XY error is <= 5 mm.

Nominal demonstrations retain 75% of source mass to protect approach, grasp,
and lift behavior.
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

CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v4_260814"
CORRECTION_ROOT = (
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_onpolicy_eef_correction_multiheight_holdz_v3_600ep_260814"
)
DEFAULT_INIT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v3_260814/"
    "absolute_eef7_mixed_correction_v3_switch5pct_b12_2k_6gpu_260814/1999/params"
)

NOMINAL_EPISODES = 5_000
CORRECTION_EPISODES = 600
EPISODE_FRAMES = 155
NOMINAL_ROWS_PER_EPISODE = 175
CORRECTION_ROWS_PER_EPISODE = 200
CORRECTION_SAMPLE_FRACTION = 0.25
CORRECTION_PER_ROW_WEIGHT = (
    CORRECTION_SAMPLE_FRACTION
    / (1.0 - CORRECTION_SAMPLE_FRACTION)
    * (NOMINAL_EPISODES * NOMINAL_ROWS_PER_EPISODE)
    / (CORRECTION_EPISODES * CORRECTION_ROWS_PER_EPISODE)
)

GROUP_NAMES = ("switch60", "recenter", "descend", "grasp", "lift")
GROUP_ROWS_PER_EPISODE = (60, 60, 30, 20, 30)


def _v4_indices(dataset, indices: list[int], classifier_config) -> list[int]:
    """Balance nominal phases and strongly sample strict correction states."""
    del classifier_config
    hf_dataset = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    columns = set(getattr(hf_dataset, "column_names", ()) or ())
    required = {"episode_index", "frame_index", "phase_id", "action_mask"}
    if not required.issubset(columns):
        raise ValueError(f"V4 mixed sampling requires columns {sorted(required)}")

    nominal_rows = NOMINAL_EPISODES * EPISODE_FRAMES
    correction_rows = CORRECTION_EPISODES * EPISODE_FRAMES
    if len(hf_dataset) == nominal_rows:
        balanced = _full_joint._balanced_full_action_indices(dataset, indices, None)  # noqa: SLF001
        logging.info(
            "V4 nominal phase sampling: input=%d balanced=%d rows_per_episode=%d",
            len(indices),
            len(balanced),
            NOMINAL_ROWS_PER_EPISODE,
        )
        return balanced
    if len(hf_dataset) != correction_rows:
        raise ValueError(
            f"Unknown V4 dataset length {len(hf_dataset)}; expected "
            f"{nominal_rows} nominal or {correction_rows} correction rows"
        )

    selected = np.asarray(indices, dtype=np.int64)
    episode = np.asarray(hf_dataset["episode_index"], dtype=np.int64)[selected]
    frame = np.asarray(hf_dataset["frame_index"], dtype=np.int64)[selected]
    phase = np.asarray(hf_dataset["phase_id"], dtype=np.int64)[selected]
    action_mask = np.asarray(hf_dataset["action_mask"], dtype=bool)[selected]
    eligible = action_mask & (frame >= 60) & (frame <= 153)
    groups = (
        selected[eligible & (frame == 60)],
        selected[eligible & (frame > 60) & (phase == 8)],
        selected[eligible & (phase == 9)],
        selected[eligible & (phase == 10)],
        selected[eligible & (phase == 11)],
    )
    num_episodes = len(np.unique(episode[eligible]))
    targets = [num_episodes * rows for rows in GROUP_ROWS_PER_EPISODE]
    resized = [
        _v3._resize_group(group, target, seed=260816 + len(indices) + group_index)  # noqa: SLF001
        for group_index, (group, target) in enumerate(zip(groups, targets, strict=True))
    ]
    merged = np.concatenate(resized)
    merged = np.random.default_rng(260816 + len(indices)).permutation(merged)
    logging.info(
        "V4 correction sampling: episodes=%d raw=%s targets=%s output=%d proportions=%s",
        num_episodes,
        dict(zip(GROUP_NAMES, [len(group) for group in groups], strict=True)),
        dict(zip(GROUP_NAMES, targets, strict=True)),
        len(merged),
        dict(
            zip(
                GROUP_NAMES,
                [round(target / len(merged), 6) for target in targets],
                strict=True,
            )
        ),
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
        raise ValueError("V4 hold-Z correction training requires at least two steps")
    logging.info(
        "V4 75/25 nominal/correction sampling: correction_per_row_weight=%.9f; "
        "switch_mass=7.5%% recenter_mass=7.5%% init=%s",
        CORRECTION_PER_ROW_WEIGHT,
        args.init_checkpoint,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _v4_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
