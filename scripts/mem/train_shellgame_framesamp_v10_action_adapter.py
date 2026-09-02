#!/usr/bin/env python3
"""Train the FrameSamp action interface while keeping the exact V10 frozen."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.shellgame.fixed_prefix_current_video_dataset import (  # noqa: E402
    FixedPrefixCurrentVideoDataset,
)
from openpi.training import config as _config  # noqa: E402
import openpi.training.config_pi0_mem as _config_pi0_mem  # noqa: E402
from openpi.training.mem.recipes import (  # noqa: E402
    shellgame_framesamp_v10_action_adapter as _recipe,
)
from scripts.mem import train_pi0_mem_compress as _trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--memory-bank-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-root", type=pathlib.Path, default=_recipe.DEFAULT_DATASET_ROOT)
    parser.add_argument("--init-checkpoint", type=pathlib.Path, default=_recipe.DEFAULT_V10_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _recipe.make_train_config(
        config_module=_config,
        exp_name=args.exp_name,
        memory_bank_dir=args.memory_bank_dir,
        dataset_root=args.dataset_root,
        init_checkpoint=args.init_checkpoint,
        steps=args.steps,
        peak_lr=args.peak_lr,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    _config_pi0_mem.VideoFrameDataset = FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _recipe.make_index_filter(  # noqa: SLF001
        args.memory_bank_dir
    )
    _trainer.main(config)


if __name__ == "__main__":
    main()

