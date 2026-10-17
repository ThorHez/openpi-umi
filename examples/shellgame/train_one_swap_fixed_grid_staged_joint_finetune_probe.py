"""Stage-3 joint fine-tuning for the fixed-grid temporal memory probe.

Stages 1 and 2 have already produced a checkpoint with a pretrained fixed-grid
temporal encoder and a trained 128-token memory compressor.  This script loads
that checkpoint, unfreezes the complete fixed-grid temporal-memory module, and
jointly fine-tunes it at a lower learning rate to test staged-training
stability.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.shellgame.train_one_swap_fixed_grid_temporal_memory_probe import (
    build_config as build_joint_config,
)
from examples.shellgame.train_one_swap_fixed_grid_temporal_memory_probe import (
    parse_args,
)
from examples.shellgame.train_one_swap_history_probe import build_one_swap_labels
from openpi.training import weight_loaders
from scripts.mem import train_pi0_mem_compress as _trainer


STAGE2_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_one_swap_fixed_grid_pretrained_memory_probe_260808/"
    "one_swap_fixed_grid_pretrained_k64_memory128_260808/499/params"
)


def main() -> None:
    args = parse_args()
    config = build_joint_config(args, build_one_swap_labels())
    config = dataclasses.replace(
        config,
        name="pi0_shellgame_one_swap_fixed_grid_staged_joint_probe_260808",
        weight_loader=weight_loaders.CheckpointWeightLoader(STAGE2_CHECKPOINT),
        # ``build_joint_config`` already makes the complete
        # HistoryFixedGridTemporalMemory subtree trainable and freezes Pi0.
        freeze_filter=config.model.get_freeze_filter_fixed_grid_memory(),
    )
    _trainer.main(config)


if __name__ == "__main__":
    main()
