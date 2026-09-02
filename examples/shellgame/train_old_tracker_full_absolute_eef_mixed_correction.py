"""Fine-tune absolute EEF7 actions on nominal and on-policy correction data.

This is a controlled extension of ``train_old_tracker_full_absolute_eef``:

* the nominal dataset keeps its proven five-way phase-balanced sampler;
* the correction dataset keeps only Oracle-supervised anchors and balances
  recenter / descend / grasp / lift at 25% each;
* dataset-level sampling mass is 50/50 after phase expansion;
* both datasets use the nominal absolute-EEF normalization statistics; and
* the validated history tracker and memory interface remain frozen.

The correction anchor at frame 154 is intentionally omitted.  This preserves
the validated global ``last_episode_frame=154`` temporal-loss contract shared
with the nominal dataset; the omitted target is only the final repeated lift
hold command.
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
    "shellgame_lerobot_onpolicy_eef_correction_raw7_500ep_260813"
)
CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_260813"

# With a 90/10 episode split, the nominal sampler expands each episode to
# 5 * 35 anchors and the correction sampler to 4 * 38 anchors.  The source
# datasets contain 5000 and 500 episodes respectively, so this per-row weight
# gives both datasets equal total probability under WeightedRandomSampler.
NOMINAL_EPISODES = 5_000
CORRECTION_EPISODES = 500
NOMINAL_BALANCED_ROWS_PER_EPISODE = 5 * 35
CORRECTION_BALANCED_ROWS_PER_EPISODE = 4 * 38
CORRECTION_PER_ROW_WEIGHT = (
    NOMINAL_EPISODES * NOMINAL_BALANCED_ROWS_PER_EPISODE
) / (CORRECTION_EPISODES * CORRECTION_BALANCED_ROWS_PER_EPISODE)


def _expand_and_interleave(groups: list[np.ndarray], *, seed: int) -> list[int]:
    """Oversample groups to equal length and deterministically interleave them."""
    if any(len(group) == 0 for group in groups):
        raise ValueError(f"Empty mixed-action phase group: {[len(group) for group in groups]}")
    target_size = max(len(group) for group in groups)
    balanced = []
    for group_index, group in enumerate(groups):
        rng = np.random.default_rng(seed + group_index)
        shuffled = rng.permutation(group)
        repeats, remainder = divmod(target_size, len(shuffled))
        expanded = np.concatenate(
            [np.tile(shuffled, repeats), shuffled[:remainder]], axis=0
        )
        balanced.append(expanded)
    return np.stack(balanced, axis=1).reshape(-1).tolist()


def _mixed_balanced_indices(dataset, indices: list[int], classifier_config) -> list[int]:
    """Route nominal rows to five phases and correction rows to four phases."""
    del classifier_config
    hf_dataset = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    columns = set(getattr(hf_dataset, "column_names", ()) or ())
    required = {"frame_index", "phase_id", "action_mask"}
    if not required.issubset(columns):
        raise ValueError(f"Mixed correction sampling requires columns {sorted(required)}")

    nominal_rows = NOMINAL_EPISODES * 155
    correction_rows = CORRECTION_EPISODES * 156
    if len(hf_dataset) == nominal_rows:
        balanced = _full_joint._balanced_full_action_indices(  # noqa: SLF001
            dataset, indices, None
        )
        logging.info(
            "Nominal five-phase sampling: input=%d balanced=%d rows_per_episode=%d",
            len(indices),
            len(balanced),
            NOMINAL_BALANCED_ROWS_PER_EPISODE,
        )
        return balanced
    if len(hf_dataset) != correction_rows:
        raise ValueError(
            "Unknown mixed-action dataset length: "
            f"got {len(hf_dataset)}, expected {nominal_rows} nominal or "
            f"{correction_rows} correction rows"
        )

    selected = np.asarray(indices, dtype=np.int64)
    frame = np.asarray(hf_dataset["frame_index"], dtype=np.int64)[selected]
    phase = np.asarray(hf_dataset["phase_id"], dtype=np.int64)[selected]
    action_mask = np.asarray(hf_dataset["action_mask"], dtype=bool)[selected]
    eligible = action_mask & (frame >= 60) & (frame <= 153)
    groups = [selected[eligible & (phase == phase_id)] for phase_id in (8, 9, 10, 11)]
    balanced = _expand_and_interleave(groups, seed=260813 + len(indices))
    logging.info(
        "Correction four-phase sampling: input=%d raw_groups=%s balanced=%d "
        "(frame 154 omitted)",
        len(indices),
        [len(group) for group in groups],
        len(balanced),
    )
    return balanced


def _eef_data_config(repo_id: str) -> _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_AbsoluteEEF7:
    """Build one EEF child while forcing nominal normalization assets."""
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
    """Build the frozen-tracker 50/50 nominal-correction training config."""
    parent = _full_eef.build_config(args)
    data = _config.MultiDataConfigFactory(
        state_pad_dim=96,
        # Correction is first so the standard sequential validation iterator
        # reports held-out correction loss during this targeted experiment.
        datasets=[
            _eef_data_config(CORRECTION_ROOT),
            _eef_data_config(NOMINAL_ROOT),
        ],
        weights=[CORRECTION_PER_ROW_WEIGHT, 1.0],
        # Both children already point at the exact nominal stats used by the
        # checkpoint.  Re-merging would shift the checkpoint's action scale.
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
        raise ValueError("Mixed correction training requires at least two steps")
    logging.info(
        "50/50 source sampling: correction_per_row_weight=%.9f; "
        "normalization=%s; temporal_last_frame=%d",
        CORRECTION_PER_ROW_WEIGHT,
        NOMINAL_ROOT,
        _full_joint.LAST_EPISODE_FRAME,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _mixed_balanced_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
