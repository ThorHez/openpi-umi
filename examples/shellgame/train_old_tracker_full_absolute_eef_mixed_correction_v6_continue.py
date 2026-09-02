"""Continue the completed V6 action run at its terminal low learning rate.

This entry point resumes the full train state (including Adam moments and the
global step) in the original experiment directory.  It deliberately keeps the
original 3k cosine schedule: Optax clamps that schedule to ``decay_lr`` after
step 3000, so steps 3000--5999 are a true low-LR continuation instead of a new
warmup that could overwrite the action timing learned by V5/V6.
"""

import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v6 as _v6
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import config_pi0_mem as _config_pi0_mem
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer


def build_config(args):
    config = _v6.build_config(args)
    return dataclasses.replace(
        config,
        resume=True,
        num_train_steps=args.steps,
        # Preserve the schedule used by the original 0--2999 run.  For all
        # resumed steps this evaluates to the terminal 3e-6 learning rate.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=3e-5,
            decay_steps=3_000,
            decay_lr=3e-6,
        ),
    )


def main() -> None:
    args = _full_joint.parse_args()
    if args.steps <= 3_000:
        raise ValueError("V6 continuation requires --steps greater than 3000")

    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _v6._v6_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
