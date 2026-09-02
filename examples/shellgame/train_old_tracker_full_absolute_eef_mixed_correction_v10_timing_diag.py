"""Short control-variable fine-tune for diagnosing V9 closed-loop regression.

This recipe deliberately reuses the already-audited nominal, V6, and V9
datasets.  It changes only row sampling:

* 60% phase-balanced nominal demonstrations;
* 30% V6 preservation replay: 21% recovery, 5% grasp, 4% early lift;
* 10% V9 timing replay: 2% hard, 2% low, 1% aligned, 2% front-loaded
  descent, 2% genuine close-within-the-first-3-actions, and 1% early lift.

The model is initialized from V6-5999 and the validated tracker/memory remain
frozen.  No action is edited or synthesized by this sampler.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v2 as _v2
from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v9 as _v9
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import config as _config
from openpi.training import config_pi0_mem as _config_pi0_mem
from scripts.mem import train_pi0_mem_compress as _trainer


CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820"
NOMINAL_ROOT = _v9.NOMINAL_ROOT
V6_ROOT = _v9.V6_ROOT
V9_ROOT = _v9.V9_ROOT
DEFAULT_INIT_CHECKPOINT = _v9.DEFAULT_INIT_CHECKPOINT

EPISODE_FRAMES = _v9.EPISODE_FRAMES
NOMINAL_EPISODES = _v9.NOMINAL_EPISODES
V6_EPISODES = _v9.V6_EPISODES
V9_EPISODES = _v9.V9_EPISODES

SOURCE_FRACTIONS = {"nominal": 0.60, "v6": 0.30, "v9": 0.10}
NOMINAL_ROWS_PER_EPISODE = _v9.NOMINAL_ROWS_PER_EPISODE
V6_ROWS_PER_EPISODE = 30
V9_ROWS_PER_EPISODE = 10
V6_GROUP_ROWS_PER_EPISODE = (21, 5, 4)
V9_GROUP_NAMES = (
    "hard_initial_recovery",
    "low_1_4mm",
    "aligned_continuation",
    "front3_descent",
    "close_within3",
    "early_lift",
)
V9_GROUP_ROWS_PER_EPISODE = (2, 2, 1, 2, 2, 1)

NOMINAL_FILTERED_ROWS = NOMINAL_EPISODES * NOMINAL_ROWS_PER_EPISODE
V6_FILTERED_ROWS = V6_EPISODES * V6_ROWS_PER_EPISODE
V9_FILTERED_ROWS = V9_EPISODES * V9_ROWS_PER_EPISODE


def _source_weight(source_fraction: float, source_rows: int) -> float:
    return (
        source_fraction
        / SOURCE_FRACTIONS["nominal"]
        * NOMINAL_FILTERED_ROWS
        / source_rows
    )


V6_PER_ROW_WEIGHT = _source_weight(SOURCE_FRACTIONS["v6"], V6_FILTERED_ROWS)
V9_PER_ROW_WEIGHT = _source_weight(SOURCE_FRACTIONS["v9"], V9_FILTERED_ROWS)


def _sample_v6(dataset, indices: list[int]) -> list[int]:
    hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    episode = np.asarray(hf["episode_index"], dtype=np.int64)
    frame = np.asarray(hf["frame_index"], dtype=np.int64)
    phase = np.asarray(hf["phase_id"], dtype=np.int64)
    action_mask = np.asarray(hf["action_mask"], dtype=bool)
    eligible = selected[
        action_mask[selected]
        & (frame[selected] >= 60)
        & (frame[selected] <= 153)
    ]
    target = eligible + 1
    if np.any(episode[target] != episode[eligible]):
        raise ValueError("V10 V6 +1 action target crosses an episode boundary")
    target_phase = phase[target]
    eligible_episode = episode[eligible]
    expected_episodes = np.unique(eligible_episode)

    first_lift = np.full(V6_EPISODES, EPISODE_FRAMES, dtype=np.int64)
    lift_rows = phase == 11
    np.minimum.at(first_lift, episode[lift_rows], frame[lift_rows])
    lift_step = frame[target] - first_lift[eligible_episode]

    masks = (
        np.isin(target_phase, (8, 9)),
        target_phase == 10,
        (target_phase == 11) & (lift_step >= 0) & (lift_step < 10),
    )
    labels = ("recovery", "grasp", "early_lift")
    groups = tuple(
        _v9._resize_each_episode(  # noqa: SLF001
            eligible[mask],
            eligible_episode[mask],
            expected_episodes,
            rows_per_episode,
            seed=261000 + group_index,
            label=f"V10 V6 {label}",
        )
        for group_index, (mask, rows_per_episode, label) in enumerate(
            zip(masks, V6_GROUP_ROWS_PER_EPISODE, labels, strict=True)
        )
    )
    merged = np.concatenate(groups)
    expected = len(expected_episodes) * V6_ROWS_PER_EPISODE
    if len(merged) != expected:
        raise RuntimeError(f"V10 V6 emitted {len(merged)} rows; expected {expected}")
    logging.info(
        "V10 V6 preservation: episodes=%d targets=%s output=%d global_mass="
        "{'recovery':.21,'grasp':.05,'lift':.04}",
        len(expected_episodes),
        dict(zip(labels, [len(group) for group in groups], strict=True)),
        len(merged),
    )
    return np.random.default_rng(261010 + len(selected)).permutation(merged).tolist()


def _sample_v9(dataset, indices: list[int]) -> list[int]:
    hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    metrics = _v9._load_v9_metrics()  # noqa: SLF001
    if len(metrics["episode_index"]) != len(hf):
        raise ValueError("V10 V9 metrics and LeRobot rows have different lengths")

    episode = metrics["episode_index"][selected]
    frame = metrics["frame_index"][selected]
    phase = metrics["target_phase"][selected]
    lift_step = metrics["target_lift_step"][selected]
    action_mask = metrics["action_mask"][selected]
    error = metrics["xy_error_m"][selected]
    height = metrics["height_above_grasp_m"][selected]
    initial_error = metrics["initial_xy_error_m"][selected]
    final_slot = metrics["final_slot"][selected]
    correction_sector = metrics["correction_sector"][selected]

    actions = np.asarray(hf["actions"], dtype=np.float32)[selected]
    eligible = action_mask & (frame >= 60) & (frame <= 153)
    recovery = eligible & np.isin(phase, (8, 9))
    hard_mask = recovery & (initial_error > 0.005) & (error > 0.004)
    low_mask = (
        recovery
        & (error >= 0.001)
        & (error <= 0.004)
        & (height >= -0.003)
        & (height <= 0.040)
    )
    aligned_mask = (
        recovery
        & (error < 0.001)
        & (height >= -0.003)
        & (height <= 0.060)
    )
    front3_descent_mask = (
        recovery
        & (actions[:, 0, 6] < 0.0)
        & ((actions[:, 2, 2] - actions[:, 0, 2]) <= -0.006)
    )
    close_within3_mask = (
        eligible
        & (actions[:, 0, 6] < 0.0)
        & np.any(actions[:, :3, 6] > 0.0, axis=1)
    )
    lift_mask = eligible & (phase == 11) & (lift_step >= 0) & (lift_step < 10)
    masks = (
        hard_mask,
        low_mask,
        aligned_mask,
        front3_descent_mask,
        close_within3_mask,
        lift_mask,
    )
    if any(not np.any(mask) for mask in masks):
        raise ValueError(
            "V10 V9 has an empty timing group: "
            f"{dict(zip(V9_GROUP_NAMES, [int(mask.sum()) for mask in masks], strict=True))}"
        )

    expected_episodes = np.unique(episode[eligible])
    hard = _v9._resize_each_episode(  # noqa: SLF001
        selected[hard_mask], episode[hard_mask], expected_episodes, 2,
        seed=261020, label="V10 hard",
    )
    low_strata = final_slot[low_mask].astype(np.int16) * 16 + correction_sector[low_mask]
    low = _v9._balance_strata(  # noqa: SLF001
        selected[low_mask], low_strata, 2 * len(expected_episodes),
        seed=261021, label="V10 low",
    )
    height_band = np.where(
        height[aligned_mask] <= 0.015,
        0,
        np.where(height[aligned_mask] <= 0.030, 1, 2),
    )
    aligned_strata = final_slot[aligned_mask].astype(np.int16) * 3 + height_band
    aligned = _v9._balance_strata(  # noqa: SLF001
        selected[aligned_mask], aligned_strata, len(expected_episodes),
        seed=261022, label="V10 aligned",
    )
    front3_descent = _v9._resize_each_episode(  # noqa: SLF001
        selected[front3_descent_mask], episode[front3_descent_mask], expected_episodes, 2,
        seed=261023, label="V10 front3 descent",
    )
    close_within3 = _v9._resize_each_episode(  # noqa: SLF001
        selected[close_within3_mask], episode[close_within3_mask], expected_episodes, 2,
        seed=261024, label="V10 close within 3",
    )
    early_lift = _v9._resize_each_episode(  # noqa: SLF001
        selected[lift_mask], episode[lift_mask], expected_episodes, 1,
        seed=261025, label="V10 early lift",
    )
    groups = (hard, low, aligned, front3_descent, close_within3, early_lift)
    merged = np.concatenate(groups)
    expected = len(expected_episodes) * V9_ROWS_PER_EPISODE
    if len(merged) != expected:
        raise RuntimeError(f"V10 V9 emitted {len(merged)} rows; expected {expected}")
    logging.info(
        "V10 V9 timing: episodes=%d raw=%s targets=%s unique=%d output=%d "
        "global_mass={'hard':.02,'low':.02,'aligned':.01,'front3_descent':.02,"
        "'close_within3':.02,'lift':.01}",
        len(expected_episodes),
        dict(zip(V9_GROUP_NAMES, [int(mask.sum()) for mask in masks], strict=True)),
        dict(zip(V9_GROUP_NAMES, [len(group) for group in groups], strict=True)),
        len(np.unique(merged)),
        len(merged),
    )
    return np.random.default_rng(261026 + len(selected)).permutation(merged).tolist()


def _indices(dataset, indices: list[int], classifier_config) -> list[int]:
    del classifier_config
    source = _v9._dataset_repo_id(dataset)  # noqa: SLF001
    if source == NOMINAL_ROOT.resolve():
        return _full_joint._balanced_full_action_indices(dataset, indices, None)  # noqa: SLF001
    if source == V6_ROOT.resolve():
        return _sample_v6(dataset, indices)
    if source == V9_ROOT.resolve():
        return _sample_v9(dataset, indices)
    raise ValueError(f"Unknown V10 dataset source: {source}")


def build_config(args):
    parent = _v2.build_config(args)
    data = _config.MultiDataConfigFactory(
        state_pad_dim=96,
        datasets=[
            _v2._eef_data_config(str(V9_ROOT)),  # noqa: SLF001
            _v2._eef_data_config(str(V6_ROOT)),  # noqa: SLF001
            _v2._eef_data_config(str(NOMINAL_ROOT)),  # noqa: SLF001
        ],
        weights=[V9_PER_ROW_WEIGHT, V6_PER_ROW_WEIGHT, 1.0],
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
    _v9.validate_data_contracts()
    logging.info(
        "V10 timing diagnostic source mass nominal/v6/v9=60/30/10 "
        "weights=(1,%.9f,%.9f) init=%s frozen=tracker+memory",
        V6_PER_ROW_WEIGHT,
        V9_PER_ROW_WEIGHT,
        args.init_checkpoint,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
