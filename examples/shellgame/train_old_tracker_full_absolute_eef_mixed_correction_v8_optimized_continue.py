"""Resume optimized-V8 training at its terminal low learning rate.

This entry point preserves the complete step-1999 train state, including Adam
moments and the global step.  The original 2k cosine schedule is retained, so
all resumed steps run at its terminal 1e-6 learning rate without a new warmup.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v8 as _v8
from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v8_optimized as _optimized
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import config_pi0_mem as _config_pi0_mem
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer


def build_config(args):
    _optimized._configure_actual_dataset()  # noqa: SLF001
    config = _v8.build_config(args)
    return dataclasses.replace(
        config,
        resume=True,
        num_train_steps=args.steps,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=1e-5,
            decay_steps=2_000,
            decay_lr=1e-6,
        ),
    )


def main() -> None:
    args = _full_joint.parse_args()
    if args.steps <= 2_000:
        raise ValueError("Optimized-V8 continuation requires --steps greater than 2000")

    logging.info(
        "Resuming optimized-V8 through step %d at terminal lr=1e-6; "
        "optimizer state and sampling recipe are unchanged",
        args.steps,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _v8._v8_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
