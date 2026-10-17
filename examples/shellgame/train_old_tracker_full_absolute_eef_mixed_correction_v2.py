"""Fine-tune EEF7 actions with a conservative amount of V2 correction data.

Differences from the original mixed-correction experiment are intentionally
limited to the two diagnosed data issues:

* correction contributes 15%, rather than 50%, of training samples; and
* its 94 valid rows keep their natural temporal proportions instead of
  oversampling the two-frame recenter phase to 25% of correction samples.

The proven nominal sampler, nominal normalization statistics, frozen memory
tracker, action dimensions, and temporal prefix contract are unchanged.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef as _full_eef
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import config as _config
from openpi.training import config_pi0_mem as _config_pi0_mem
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from scripts.mem import train_pi0_mem_compress as _trainer


NOMINAL_ROOT = _full_eef.LEROBOT_ABSOLUTE_EEF7_ROOT
CORRECTION_ROOT = (
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_onpolicy_eef_correction_raw7_replan_v2_500ep_260813"
)
CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v2_260813"

NOMINAL_EPISODES = 5_000
CORRECTION_EPISODES = 500
EPISODE_FRAMES = 155
NOMINAL_BALANCED_ROWS_PER_EPISODE = 5 * 35
CORRECTION_VALID_ROWS_PER_EPISODE = 94
CORRECTION_SAMPLE_FRACTION = 0.15

# WeightedRandomSampler assigns a weight to every row remaining after the two
# source-specific filters. Solve wc*Nc / (wc*Nc + Nn) = 0.15 exactly.
CORRECTION_PER_ROW_WEIGHT = (
    CORRECTION_SAMPLE_FRACTION
    / (1.0 - CORRECTION_SAMPLE_FRACTION)
    * (NOMINAL_EPISODES * NOMINAL_BALANCED_ROWS_PER_EPISODE)
    / (CORRECTION_EPISODES * CORRECTION_VALID_ROWS_PER_EPISODE)
)


def _mixed_v2_indices(dataset, indices: list[int], classifier_config) -> list[int]:
    """Keep nominal phase balancing but preserve correction time proportions."""
    del classifier_config
    hf_dataset = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    columns = set(getattr(hf_dataset, "column_names", ()) or ())
    required = {"frame_index", "phase_id", "action_mask"}
    if not required.issubset(columns):
        raise ValueError(f"V2 mixed sampling requires columns {sorted(required)}")

    nominal_rows = NOMINAL_EPISODES * EPISODE_FRAMES
    correction_rows = CORRECTION_EPISODES * EPISODE_FRAMES
    if len(hf_dataset) == nominal_rows:
        balanced = _full_joint._balanced_full_action_indices(  # noqa: SLF001
            dataset, indices, None
        )
        logging.info(
            "Nominal five-phase sampling: input=%d balanced=%d",
            len(indices),
            len(balanced),
        )
        return balanced
    if len(hf_dataset) != correction_rows:
        raise ValueError(
            "Unknown V2 mixed-action dataset length: "
            f"got {len(hf_dataset)}, expected {nominal_rows} nominal or "
            f"{correction_rows} correction rows"
        )

    selected = np.asarray(indices, dtype=np.int64)
    frame = np.asarray(hf_dataset["frame_index"], dtype=np.int64)[selected]
    phase = np.asarray(hf_dataset["phase_id"], dtype=np.int64)[selected]
    action_mask = np.asarray(hf_dataset["action_mask"], dtype=bool)[selected]
    eligible = action_mask & (frame >= 60) & (frame <= 153)
    kept = selected[eligible]
    phase_counts = {
        int(phase_id): int(np.count_nonzero(phase[eligible] == phase_id))
        for phase_id in (8, 9, 10, 11)
    }
    logging.info(
        "V2 correction natural-time sampling: input=%d kept=%d phases=%s",
        len(indices),
        len(kept),
        phase_counts,
    )
    return kept.tolist()


def _eef_data_config(repo_id: str) -> _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_AbsoluteEEF7:
    return _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_AbsoluteEEF7(
        repo_id=repo_id,
        assets=_config.AssetsConfig(asset_id=".", assets_dir=NOMINAL_ROOT),
        base_config=_config.UmiDataConfig(
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
            robot_type="ARM=1 G=0 H=0",
        ),
        num_frames=_full_joint.TOTAL_INPUT_FRAMES,
        frame_stride=1,
    )


def build_config(args):
    parent = _full_eef.build_config(args)
    data = _config.MultiDataConfigFactory(
        state_pad_dim=96,
        datasets=[
            _eef_data_config(CORRECTION_ROOT),
            _eef_data_config(NOMINAL_ROOT),
        ],
        weights=[CORRECTION_PER_ROW_WEIGHT, 1.0],
        use_merged_norm_stats=False,
    )
    return dataclasses.replace(
        parent,
        name=CONFIG_NAME,
        exp_name=args.exp_name,
        data=data,
        freeze_filter=parent.model.get_freeze_filter_full_action(),
        weight_loader=weight_loaders.CheckpointWeightLoader(args.init_checkpoint),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 2),
            decay_lr=args.peak_lr * 0.1,
        ),
        num_train_steps=args.steps,
        val_ratio=0.1,
        resume=False,
    )


def main() -> None:
    args = _full_joint.parse_args()
    if args.steps < 2:
        raise ValueError("V2 mixed correction training requires at least two steps")
    logging.info(
        "85/15 nominal/correction sampling: correction_per_row_weight=%.9f; "
        "normalization=%s; temporal_last_frame=%d",
        CORRECTION_PER_ROW_WEIGHT,
        NOMINAL_ROOT,
        _full_joint.LAST_EPISODE_FRAME,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _mixed_v2_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
