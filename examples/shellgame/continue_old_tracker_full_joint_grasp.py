"""Resume the full joint + gripper action run from a copied train state.

The continuation keeps the restored AdamW state and global step.  Its learning
rate is continuous with the approximately 3e-6 terminal LR of the original
2,000-step run, then decays smoothly to 1e-6 at global step 6,000.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.shellgame import train_old_tracker_full_joint_grasp as full_action
from openpi.training import config_pi0_mem
from openpi.training import optimizer as optimizer
from scripts.mem import train_pi0_mem_compress as trainer


CONTINUATION_PEAK_LR = 3.5e-6
CONTINUATION_END_LR = 1.0e-6


def main() -> None:
    args = full_action.parse_args()
    if args.steps != 6_000:
        raise ValueError(
            "This continuation is calibrated for --steps 6000; "
            f"got --steps {args.steps}"
        )
    config = full_action.build_config(args)
    config = dataclasses.replace(
        config,
        resume=True,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=CONTINUATION_PEAK_LR,
            decay_steps=6_000,
            decay_lr=CONTINUATION_END_LR,
        ),
    )
    logging.info(
        "Continuation schedule: restored global step, peak_lr=%g decay_lr=%g total_steps=%d",
        CONTINUATION_PEAK_LR,
        CONTINUATION_END_LR,
        config.num_train_steps,
    )
    config_pi0_mem.VideoFrameDataset = full_action.FixedPrefixCurrentVideoDataset
    trainer._filter_memory_classifier_frame_range = (  # noqa: SLF001
        full_action._balanced_full_action_indices  # noqa: SLF001
    )
    trainer.main(config)


if __name__ == "__main__":
    main()
