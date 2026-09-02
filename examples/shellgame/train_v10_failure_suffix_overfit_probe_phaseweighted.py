"""Phase-weighted exact-state overfit probe for failed V10 suffixes.

This is the controlled successor to ``train_v10_failure_suffix_overfit_probe``.
It changes only the row sampler: every held-in episode contributes 50 recenter,
30 descend, and 20 grasp/lift rows.  Repeated indices are intentional and make
the critical first correction chunk half of all optimizer samples.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v3 as _v3
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from examples.shellgame import train_v10_failure_suffix_overfit_probe as _base
from openpi.training import config_pi0_mem as _config_pi0_mem
from scripts.mem import train_pi0_mem_compress as _trainer


CONFIG_NAME = "pi0_shellgame_v10_failure_suffix_overfit_phaseweighted_260820"
ROWS_PER_EPISODE = {
    "recenter_60_69": 50,
    "descend_70_99": 30,
    "grasp_lift_100_154": 20,
}


def _resize_per_episode(
    eligible: np.ndarray,
    episode: np.ndarray,
    episodes: np.ndarray,
    target_rows: int,
    *,
    seed: int,
) -> np.ndarray:
    resized = []
    for episode_id in episodes:
        group = eligible[episode == episode_id]
        if group.size == 0:
            raise RuntimeError(f"No eligible rows for episode {int(episode_id)}")
        resized.append(
            _v3._resize_group(  # noqa: SLF001
                group,
                target_rows,
                seed=seed + int(episode_id),
            )
        )
    return np.concatenate(resized)


def _phase_weighted_indices(dataset, indices: list[int], classifier_config) -> list[int]:
    del classifier_config
    hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    full_episode = np.asarray(hf["episode_index"], dtype=np.int64)
    full_frame = np.asarray(hf["frame_index"], dtype=np.int64)
    full_action_mask = np.asarray(hf["action_mask"], dtype=bool)

    episode = full_episode[selected]
    frame = full_frame[selected]
    eligible_mask = full_action_mask[selected] & (frame >= 60) & (frame <= 154)
    eligible = selected[eligible_mask]
    eligible_episode = full_episode[eligible]
    eligible_frame = full_frame[eligible]
    episodes = np.unique(eligible_episode)
    if episodes.size == 0:
        raise RuntimeError("Phase-weighted probe found no Oracle-aligned rows")

    masks = {
        "recenter_60_69": (eligible_frame >= 60) & (eligible_frame <= 69),
        "descend_70_99": (eligible_frame >= 70) & (eligible_frame <= 99),
        "grasp_lift_100_154": (eligible_frame >= 100) & (eligible_frame <= 154),
    }
    expected_raw_per_episode = {
        "recenter_60_69": 10,
        "descend_70_99": 30,
        "grasp_lift_100_154": 55,
    }
    resized = []
    raw_counts = {}
    for group_index, (name, mask) in enumerate(masks.items()):
        group = eligible[mask]
        group_episode = eligible_episode[mask]
        counts = {
            int(ep): int(np.count_nonzero(group_episode == ep)) for ep in episodes
        }
        if any(count != expected_raw_per_episode[name] for count in counts.values()):
            raise RuntimeError(f"Unexpected raw {name} rows: {counts}")
        raw_counts[name] = counts
        resized.append(
            _resize_per_episode(
                group,
                group_episode,
                episodes,
                ROWS_PER_EPISODE[name],
                seed=260830 + group_index * 1_000,
            )
        )

    merged = np.concatenate(resized)
    merged = np.random.default_rng(260833 + len(indices)).permutation(merged)
    expected = len(episodes) * sum(ROWS_PER_EPISODE.values())
    if len(merged) != expected:
        raise RuntimeError(f"Sampler emitted {len(merged)} rows, expected {expected}")
    logging.info(
        "Exact-state phase-weighted sampler: episodes=%s input=%d raw=%s "
        "targets_per_episode=%s output=%d proportions={'recenter': 0.50, "
        "'descend': 0.30, 'grasp_lift': 0.20}",
        episodes.tolist(),
        len(indices),
        raw_counts,
        ROWS_PER_EPISODE,
        len(merged),
    )
    return merged.tolist()


def build_config(args):
    return dataclasses.replace(_base.build_config(args), name=CONFIG_NAME)


def main() -> None:
    args = _full_joint.parse_args()
    if args.init_checkpoint == str(_full_joint.OLD_QUERY_ACTION_CHECKPOINT):
        args.init_checkpoint = _base.DEFAULT_INIT_CHECKPOINT
    if not 100 <= args.steps <= 500:
        raise ValueError("Keep the exact-state probe between 100 and 500 steps")
    _base._validate_data()  # noqa: SLF001
    logging.info(
        "Phase-weighted exact-state overfit: recenter=50%% descend=30%% "
        "grasp_lift=20%% init=%s tracker+memory=frozen",
        args.init_checkpoint,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _phase_weighted_indices
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
