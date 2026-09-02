"""Continue the general V10 action run from step 499 to a larger budget.

This resumes the complete V10 train state in the original experiment directory.
The model, frozen tracker/memory, datasets, 60/30/10 source mixture, row sampler,
optimizer, and original 500-step cosine schedule are unchanged.  Since Optax
clamps the schedule after its decay horizon, resumed steps use V10's terminal
learning rate of 3e-7.  No failure-suffix or task-specific diagnostic data is
introduced here.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v10_timing_diag as _v10
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import config_pi0_mem as _config_pi0_mem
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer


def build_config(args):
    config = _v10.build_config(args)
    return dataclasses.replace(
        config,
        resume=True,
        num_train_steps=args.steps,
        # Preserve the schedule used by the original 0--499 V10 run.  The
        # restored optimizer count starts at 500, so all continuation steps
        # evaluate to the terminal learning rate rather than restarting warmup.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=50,
            peak_lr=3e-6,
            decay_steps=500,
            decay_lr=3e-7,
        ),
    )


def main() -> None:
    args = _full_joint.parse_args()
    if args.steps <= 500:
        raise ValueError("V10 continuation requires --steps greater than 500")

    _v10._v9.validate_data_contracts()  # noqa: SLF001
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _v10._indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
