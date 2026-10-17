"""Fine-tune EEF7 actions with targeted switch-state correction sampling.

V2 kept correction data at a conservative 15% source mass, but its natural
time distribution made the only state that teaches the deployed three-command
contract (frame 60) roughly 0.16% of all sampled rows.  V3 keeps the same
85/15 nominal/correction source mixture and changes only the within-correction
row distribution:

* post-model switch state, frame 60:  33.33% of correction (5.0% overall);
* recenter continuation, frames 61-62: 16.67% (2.5% overall);
* descend:                              20.00% (3.0% overall);
* grasp:                               10.00% (1.5% overall);
* lift:                                20.00% (3.0% overall).

Memory, model structure, nominal phase balancing, normalization, optimizer,
and the frozen-parameter filter are unchanged from V2.  Frame 60 is therefore
about 31x more likely than under V2 while nominal behavior still owns 85% of
the training mass.
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


CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v3_260814"

# The nominal sampler emits 175 rows / episode.  V3 emits 180 correction rows
# / episode in the exact proportions documented above.  This per-row source
# weight makes correction contribute exactly 15% under WeightedRandomSampler.
NOMINAL_BALANCED_ROWS_PER_EPISODE = 175
CORRECTION_V3_ROWS_PER_EPISODE = 180
CORRECTION_SAMPLE_FRACTION = 0.15
CORRECTION_PER_ROW_WEIGHT = (
    CORRECTION_SAMPLE_FRACTION
    / (1.0 - CORRECTION_SAMPLE_FRACTION)
    * (_v2.NOMINAL_EPISODES * NOMINAL_BALANCED_ROWS_PER_EPISODE)
    / (_v2.CORRECTION_EPISODES * CORRECTION_V3_ROWS_PER_EPISODE)
)

GROUP_NAMES = ("switch60", "recenter61_62", "descend", "grasp", "lift")
GROUP_ROWS_PER_EPISODE = (60, 30, 36, 18, 36)


def _resize_group(group: np.ndarray, target_size: int, *, seed: int) -> np.ndarray:
    if len(group) == 0:
        raise ValueError("Cannot resize an empty correction phase group")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(group)
    if target_size <= len(shuffled):
        return shuffled[:target_size]
    repeats, remainder = divmod(target_size, len(shuffled))
    return np.concatenate([np.tile(shuffled, repeats), shuffled[:remainder]], axis=0)


def _v3_indices(dataset, indices: list[int], classifier_config) -> list[int]:
    """Route nominal rows unchanged and resize five correction groups."""
    del classifier_config
    hf_dataset = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    columns = set(getattr(hf_dataset, "column_names", ()) or ())
    required = {"episode_index", "frame_index", "phase_id", "action_mask"}
    if not required.issubset(columns):
        raise ValueError(f"V3 mixed sampling requires columns {sorted(required)}")

    nominal_rows = _v2.NOMINAL_EPISODES * _v2.EPISODE_FRAMES
    correction_rows = _v2.CORRECTION_EPISODES * _v2.EPISODE_FRAMES
    if len(hf_dataset) == nominal_rows:
        balanced = _full_joint._balanced_full_action_indices(  # noqa: SLF001
            dataset, indices, None
        )
        logging.info(
            "V3 nominal five-phase sampling: input=%d balanced=%d rows_per_episode=%d",
            len(indices),
            len(balanced),
            NOMINAL_BALANCED_ROWS_PER_EPISODE,
        )
        return balanced
    if len(hf_dataset) != correction_rows:
        raise ValueError(
            "Unknown V3 mixed-action dataset length: "
            f"got {len(hf_dataset)}, expected {nominal_rows} nominal or {correction_rows} correction rows"
        )

    selected = np.asarray(indices, dtype=np.int64)
    episode = np.asarray(hf_dataset["episode_index"], dtype=np.int64)[selected]
    frame = np.asarray(hf_dataset["frame_index"], dtype=np.int64)[selected]
    phase = np.asarray(hf_dataset["phase_id"], dtype=np.int64)[selected]
    action_mask = np.asarray(hf_dataset["action_mask"], dtype=bool)[selected]
    eligible = action_mask & (frame >= 60) & (frame <= 153)
    groups = (
        selected[eligible & (frame == 60)],
        selected[eligible & (frame >= 61) & (frame <= 62)],
        selected[eligible & (phase == 9)],
        selected[eligible & (phase == 10)],
        selected[eligible & (phase == 11)],
    )
    num_episodes = len(np.unique(episode[eligible]))
    targets = [num_episodes * rows for rows in GROUP_ROWS_PER_EPISODE]
    resized = [
        _resize_group(group, target, seed=260814 + len(indices) + group_index)
        for group_index, (group, target) in enumerate(zip(groups, targets, strict=True))
    ]
    # A deterministic global permutation prevents sequential validation batches
    # from containing only one phase while preserving the exact proportions.
    merged = np.concatenate(resized)
    merged = np.random.default_rng(260814 + len(indices)).permutation(merged)
    logging.info(
        "V3 correction targeted sampling: episodes=%d input=%d raw=%s targets=%s "
        "output=%d proportions=%s",
        num_episodes,
        len(indices),
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
            _v2._eef_data_config(_v2.CORRECTION_ROOT),  # noqa: SLF001
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
    if args.steps < 2:
        raise ValueError("V3 targeted correction training requires at least two steps")
    logging.info(
        "V3 85/15 source sampling: correction_per_row_weight=%.9f; "
        "frame60_overall_mass=5%%; normalization=%s; temporal_last_frame=%d",
        CORRECTION_PER_ROW_WEIGHT,
        _v2.NOMINAL_ROOT,
        _full_joint.LAST_EPISODE_FRAME,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _v3_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
