"""Fine-tune V6 EEF7 actions with error-aware V7 correction sampling.

The paired right-slot oracle intervention raised lift success from 13/31 to
30/31 while leaving memory selection, Z, rotation, and gripper unchanged.
V7 therefore replaces phase-as-a-proxy-for-difficulty with the actual current
EEF-to-oracle-target XY error reconstructed per LeRobot row.

Global training mass is exactly:

* 55% phase-balanced nominal demonstrations;
* 10% positive-Y outer recovery: target_y >= 0.10 m, inward dy >= 5 mm;
* 15% other recovery states with XY error > 5 mm;
* 10% aligned continuation with XY error in [2, 5] mm;
*  5% grasp/close continuity; and
*  5% early-lift continuity.

The two recovery groups are balanced over 5-12 / 12-22 / >22 mm error
strata.  Positive-Y recovery is also balanced by target cup identity; general
recovery and aligned continuation are balanced by final spatial slot.  The
tracker, memory, current-image path, normalization, and model structure remain
frozen exactly as in V6; only the action expert and action/time projections
are optimized.
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
from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v6 as _v6
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import config as _config
from openpi.training import config_pi0_mem as _config_pi0_mem
from scripts.mem import train_pi0_mem_compress as _trainer

CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v7_260816"
CORRECTION_ROOT = _v6.CORRECTION_ROOT
METRICS_PATH = pathlib.Path(CORRECTION_ROOT) / "xy_sampling_metrics_v7.npz"
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
CORRECTION_ROWS_PER_EPISODE = 180
CORRECTION_SAMPLE_FRACTION = 0.45

# Within correction these are 2/9, 3/9, 2/9, 1/9, 1/9.  Combined with the
# 45% correction source mass they become the documented 10/15/10/5/5 global
# percentages exactly.
GROUP_NAMES = (
    "positive_y_outer_recovery",
    "general_recovery",
    "aligned_2_5mm",
    "grasp",
    "early_lift",
)
GROUP_ROWS_PER_EPISODE = (40, 60, 40, 20, 20)
ERROR_BIN_WEIGHTS = np.asarray((0.30, 0.45, 0.25), dtype=np.float64)

CORRECTION_PER_ROW_WEIGHT = (
    CORRECTION_SAMPLE_FRACTION
    / (1.0 - CORRECTION_SAMPLE_FRACTION)
    * (NOMINAL_EPISODES * NOMINAL_ROWS_PER_EPISODE)
    / (CORRECTION_EPISODES * CORRECTION_ROWS_PER_EPISODE)
)


def _allocate(total: int, weights: np.ndarray) -> np.ndarray:
    """Allocate an exact integer total by deterministic largest remainder."""
    weights = np.asarray(weights, dtype=np.float64)
    if total < 0 or weights.ndim != 1 or len(weights) == 0 or np.any(weights < 0):
        raise ValueError(f"Invalid allocation total={total} weights={weights}")
    if not np.any(weights > 0):
        raise ValueError("Allocation weights are all zero")
    raw = weights / weights.sum() * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    if int(counts.sum()) != total:
        raise RuntimeError("Largest-remainder allocation lost mass")
    return counts


def _error_bin(error_m: np.ndarray) -> np.ndarray:
    return np.where(error_m <= 0.012, 0, np.where(error_m <= 0.022, 1, 2)).astype(np.int8)


def _resize_hierarchical(
    indices: np.ndarray,
    primary: np.ndarray,
    error_bins: np.ndarray | None,
    target_size: int,
    *,
    seed: int,
) -> np.ndarray:
    """Balance primary strata, then desired error bins within each stratum."""
    indices = np.asarray(indices, dtype=np.int64)
    primary = np.asarray(primary)
    if len(indices) == 0 or len(primary) != len(indices):
        raise ValueError("Hierarchical resize received empty or mismatched arrays")
    values = np.unique(primary)
    primary_targets = _allocate(target_size, np.ones(len(values), dtype=np.float64))
    resized = []
    for offset, (value, primary_target) in enumerate(zip(values, primary_targets, strict=True)):
        mask = primary == value
        group = indices[mask]
        if error_bins is None:
            resized.append(_v3._resize_group(group, int(primary_target), seed=seed + 101 * offset))  # noqa: SLF001
            continue
        bins = np.asarray(error_bins)[mask]
        present = np.unique(bins)
        weights = ERROR_BIN_WEIGHTS[present]
        bin_targets = _allocate(int(primary_target), weights)
        for bin_offset, (bin_id, bin_target) in enumerate(zip(present, bin_targets, strict=True)):
            bin_group = group[bins == bin_id]
            resized.append(
                _v3._resize_group(  # noqa: SLF001
                    bin_group,
                    int(bin_target),
                    seed=seed + 101 * offset + 17 * bin_offset,
                )
            )
    output = np.concatenate(resized)
    if len(output) != target_size:
        raise RuntimeError(f"Hierarchical resize emitted {len(output)} rows, expected {target_size}")
    return output


def _load_metrics() -> dict[str, np.ndarray]:
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {METRICS_PATH}. Run build_eef_xy_sampling_metrics_v7.py first."
        )
    with np.load(METRICS_PATH, allow_pickle=False) as payload:
        if int(payload["schema_version"]) != 1:
            raise ValueError(f"Unsupported V7 metrics schema: {payload['schema_version']}")
        return {key: np.asarray(payload[key]) for key in payload.files}


def _sample_correction_metrics(metrics: dict[str, np.ndarray], selected: np.ndarray) -> np.ndarray:
    episode = metrics["episode_index"][selected]
    frame = metrics["frame_index"][selected]
    target_phase = metrics["target_phase"][selected]
    lift_step = metrics["target_lift_step"][selected]
    action_mask = metrics["action_mask"][selected]
    error = metrics["xy_error_m"][selected]
    delta_y = metrics["delta_y_m"][selected]
    target_y = metrics["target_y_m"][selected]
    final_slot = metrics["final_slot"][selected]
    target_identity = metrics["target_identity"][selected]

    eligible = action_mask & (frame >= 60) & (frame <= 153) & (target_phase >= 0)
    recovery_phase = np.isin(target_phase, (8, 9))
    hard_mask = (
        eligible
        & recovery_phase
        & (error > 0.005)
        & (target_y >= 0.10)
        & (delta_y >= 0.005)
    )
    general_mask = eligible & recovery_phase & (error > 0.005) & ~hard_mask
    aligned_mask = eligible & recovery_phase & (error >= 0.002) & (error <= 0.005)
    grasp_mask = eligible & (target_phase == 10)
    lift_mask = eligible & (target_phase == 11) & (lift_step >= 0) & (lift_step < 10)
    masks = (hard_mask, general_mask, aligned_mask, grasp_mask, lift_mask)
    if any(not np.any(mask) for mask in masks):
        raise ValueError(f"V7 sampling has an empty group: {[int(mask.sum()) for mask in masks]}")

    episodes = np.unique(episode[eligible])
    targets = np.asarray(GROUP_ROWS_PER_EPISODE, dtype=np.int64) * len(episodes)
    hard_indices = selected[hard_mask]
    general_indices = selected[general_mask]
    hard = _resize_hierarchical(
        hard_indices,
        target_identity[hard_mask],
        _error_bin(error[hard_mask]),
        int(targets[0]),
        seed=260831,
    )
    general = _resize_hierarchical(
        general_indices,
        final_slot[general_mask],
        _error_bin(error[general_mask]),
        int(targets[1]),
        seed=260832,
    )
    aligned = _resize_hierarchical(
        selected[aligned_mask],
        final_slot[aligned_mask],
        None,
        int(targets[2]),
        seed=260833,
    )
    grasp = _v6._resize_each_episode(  # noqa: SLF001
        selected[grasp_mask],
        episode[grasp_mask],
        episodes,
        GROUP_ROWS_PER_EPISODE[3],
        seed=260834,
    )
    lift = _v6._resize_each_episode(  # noqa: SLF001
        selected[lift_mask],
        episode[lift_mask],
        episodes,
        GROUP_ROWS_PER_EPISODE[4],
        seed=260835,
    )
    groups = (hard, general, aligned, grasp, lift)
    merged = np.concatenate(groups)
    merged = np.random.default_rng(260836 + len(selected)).permutation(merged)
    expected = len(episodes) * CORRECTION_ROWS_PER_EPISODE
    if len(merged) != expected:
        raise RuntimeError(f"V7 emitted {len(merged)} correction rows, expected {expected}")
    logging.info(
        "V7 error-aware sampling episodes=%d raw=%s targets=%s output=%d "
        "global_mass={'nominal':.55,'positive_y_outer':.10,'general_recovery':.15,"
        "'aligned_2_5mm':.10,'grasp':.05,'early_lift':.05}",
        len(episodes),
        dict(zip(GROUP_NAMES, [int(mask.sum()) for mask in masks], strict=True)),
        dict(zip(GROUP_NAMES, [len(group) for group in groups], strict=True)),
        len(merged),
    )
    return merged


def _v7_indices(dataset, indices: list[int], classifier_config) -> list[int]:
    del classifier_config
    hf_dataset = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    nominal_rows = NOMINAL_EPISODES * EPISODE_FRAMES
    correction_rows = CORRECTION_EPISODES * EPISODE_FRAMES
    if len(hf_dataset) == nominal_rows:
        return _full_joint._balanced_full_action_indices(dataset, indices, None)  # noqa: SLF001
    if len(hf_dataset) != correction_rows:
        raise ValueError(
            f"Unknown V7 dataset length {len(hf_dataset)}; expected "
            f"{nominal_rows} nominal or {correction_rows} correction rows"
        )

    metrics = _load_metrics()
    if len(metrics["episode_index"]) != len(hf_dataset):
        raise ValueError(
            f"V7 metrics rows={len(metrics['episode_index'])} do not match dataset rows={len(hf_dataset)}"
        )
    # Scalar identity checks catch stale or reordered sidecars without loading
    # the expensive image/action columns.
    hf_episode = np.asarray(hf_dataset["episode_index"], dtype=np.int64)[selected]
    hf_frame = np.asarray(hf_dataset["frame_index"], dtype=np.int64)[selected]
    if not np.array_equal(hf_episode, metrics["episode_index"][selected]):
        raise ValueError("V7 sidecar episode_index does not match LeRobot rows")
    if not np.array_equal(hf_frame, metrics["frame_index"][selected]):
        raise ValueError("V7 sidecar frame_index does not match LeRobot rows")
    return _sample_correction_metrics(metrics, selected).tolist()


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
        raise ValueError("V7 error-aware training requires at least two steps")
    logging.info(
        "V7 55/10/15/10/5/5 error-aware sampling correction_weight=%.9f "
        "metrics=%s init=%s frozen=tracker+memory",
        CORRECTION_PER_ROW_WEIGHT,
        METRICS_PATH,
        args.init_checkpoint,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _v7_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
