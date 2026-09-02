"""Fine-tune V6 on V8 sustained recovery without row replication.

The global training mass is 60% phase-balanced nominal demonstrations and 40%
V8 episodes.  Every held-in V8 episode contributes exactly 16 *different*
rows: the first four consecutive correction rows plus eight rows spread over
the remaining descent, two grasp rows, and two early lift rows.  Thus the
effective global split is 60/30/5/5 while every episode,
spatial slot, height band, radius bin, and offset direction keeps its designed
mass.  No error bucket is inflated by copying scarce rows.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v2 as _v2
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import config as _config
from openpi.training import config_pi0_mem as _config_pi0_mem
from scripts.mem import train_pi0_mem_compress as _trainer

CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v8_260818"
CORRECTION_ROOT = (
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_onpolicy_eef_sustained_recovery_v8_balanced1200_260818"
)
DEFAULT_INIT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/"
    "absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/"
    "5999/params"
)

NOMINAL_EPISODES = 5_000
CORRECTION_EPISODES = 1_200
EPISODE_FRAMES = 155
NOMINAL_ROWS_PER_EPISODE = 175
RECOVERY_ROWS_PER_EPISODE = 12
GRASP_ROWS_PER_EPISODE = 2
LIFT_ROWS_PER_EPISODE = 2
CORRECTION_ROWS_PER_EPISODE = (
    RECOVERY_ROWS_PER_EPISODE + GRASP_ROWS_PER_EPISODE + LIFT_ROWS_PER_EPISODE
)
CORRECTION_SAMPLE_FRACTION = 0.40
CORRECTION_PER_ROW_WEIGHT = (
    CORRECTION_SAMPLE_FRACTION
    / (1.0 - CORRECTION_SAMPLE_FRACTION)
    * (NOMINAL_EPISODES * NOMINAL_ROWS_PER_EPISODE)
    / (CORRECTION_EPISODES * CORRECTION_ROWS_PER_EPISODE)
)


def _evenly_spaced_unique(group: np.ndarray, count: int, *, label: str, episode: int) -> np.ndarray:
    group = np.asarray(group, dtype=np.int64)
    if len(group) < count:
        raise ValueError(f"V8 episode {episode} has {len(group)} {label} rows; need {count}")
    positions = np.rint(np.linspace(0, len(group) - 1, count)).astype(np.int64)
    chosen = group[positions]
    if len(np.unique(chosen)) != count:
        raise RuntimeError(f"V8 {label} selector duplicated a row in episode {episode}")
    return chosen


def _recovery_rows(group: np.ndarray, *, episode: int) -> np.ndarray:
    """Keep the hard correction onset and the full later descent without copies."""
    group = np.asarray(group, dtype=np.int64)
    if len(group) < RECOVERY_ROWS_PER_EPISODE:
        raise ValueError(
            f"V8 episode {episode} has {len(group)} recovery rows; "
            f"need {RECOVERY_ROWS_PER_EPISODE}"
        )
    onset_count = 4
    onset = group[:onset_count]
    continuation = _evenly_spaced_unique(
        group[onset_count:],
        RECOVERY_ROWS_PER_EPISODE - onset_count,
        label="recovery-continuation",
        episode=episode,
    )
    chosen = np.concatenate([onset, continuation])
    if len(np.unique(chosen)) != RECOVERY_ROWS_PER_EPISODE:
        raise RuntimeError(f"V8 recovery selector duplicated a row in episode {episode}")
    return chosen


def _v8_indices(dataset, indices: list[int], classifier_config) -> list[int]:
    del classifier_config
    hf_dataset = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    nominal_rows = NOMINAL_EPISODES * EPISODE_FRAMES
    correction_rows = CORRECTION_EPISODES * EPISODE_FRAMES
    if len(hf_dataset) == nominal_rows:
        return _full_joint._balanced_full_action_indices(dataset, indices, None)  # noqa: SLF001
    if len(hf_dataset) != correction_rows:
        raise ValueError(
            f"Unknown V8 dataset length {len(hf_dataset)}; expected "
            f"{nominal_rows} nominal or {correction_rows} correction rows"
        )

    full_episode = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    full_frame = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    full_phase = np.asarray(hf_dataset["phase_id"], dtype=np.int64)
    full_action_mask = np.asarray(hf_dataset["action_mask"], dtype=bool)
    eligible_mask = (
        full_action_mask[selected]
        & (full_frame[selected] >= 60)
        & (full_frame[selected] <= 153)
    )
    eligible = selected[eligible_mask]
    target_index = eligible + 1
    if np.any(full_episode[target_index] != full_episode[eligible]):
        raise ValueError("V8 aligned target crosses an episode boundary")
    if np.any(full_frame[target_index] != full_frame[eligible] + 1):
        raise ValueError("V8 aligned target is not the next raw command")
    episodes = np.unique(full_episode[eligible])
    output = []
    phase_counts = {"recovery": 0, "grasp": 0, "lift": 0}
    for episode in episodes:
        episode_rows = eligible[full_episode[eligible] == episode]
        target_phase = full_phase[episode_rows + 1]
        recovery = episode_rows[np.isin(target_phase, (8, 9))]
        grasp = episode_rows[target_phase == 10]
        lift = episode_rows[target_phase == 11]
        chosen = np.concatenate(
            [
                _recovery_rows(recovery, episode=int(episode)),
                _evenly_spaced_unique(grasp, GRASP_ROWS_PER_EPISODE, label="grasp", episode=int(episode)),
                _evenly_spaced_unique(
                    lift[:10], LIFT_ROWS_PER_EPISODE, label="early-lift", episode=int(episode)
                ),
            ]
        )
        if len(np.unique(chosen)) != CORRECTION_ROWS_PER_EPISODE:
            raise RuntimeError(f"V8 episode {episode} emitted duplicate training rows")
        output.append(chosen)
        phase_counts["recovery"] += RECOVERY_ROWS_PER_EPISODE
        phase_counts["grasp"] += GRASP_ROWS_PER_EPISODE
        phase_counts["lift"] += LIFT_ROWS_PER_EPISODE

    merged = np.concatenate(output)
    if len(np.unique(merged)) != len(merged):
        raise RuntimeError("V8 sampler duplicated rows across episodes")
    expected = len(episodes) * CORRECTION_ROWS_PER_EPISODE
    if len(merged) != expected:
        raise RuntimeError(f"V8 selected {len(merged)} rows, expected {expected}")
    merged = np.random.default_rng(260841 + len(indices)).permutation(merged)
    logging.info(
        "V8 unique-row sampling episodes=%d output=%d unique=%d "
        "per_episode=4_onset+8_continuation+2_grasp+2_lift "
        "global_mass={'nominal':.60,'recovery':.30,'grasp':.05,'lift':.05}",
        len(episodes),
        len(merged),
        len(np.unique(merged)),
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
        raise ValueError("V8 sustained-recovery training requires at least two steps")
    logging.info(
        "V8 exact 60/30/5/5 unique-row sampling correction_weight=%.9f root=%s "
        "init=%s frozen=tracker+memory",
        CORRECTION_PER_ROW_WEIGHT,
        CORRECTION_ROOT,
        args.init_checkpoint,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _v8_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
