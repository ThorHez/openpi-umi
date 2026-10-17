#!/usr/bin/env python3
"""Train semantic memory as an exact-path replacement for V10 memory."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.shellgame.fixed_prefix_current_video_dataset import (  # noqa: E402, I001
    FixedPrefixCurrentVideoDataset,
)
from openpi.training import config as _config  # noqa: E402
import openpi.training.config_pi0_mem as _config_pi0_mem  # noqa: E402
from openpi.training.mem.recipes import (  # noqa: E402
    shellgame_qwen_distilled_memory_action_v10 as _memory_data,
)
from openpi.training.mem.recipes import (  # noqa: E402
    shellgame_v10_exact_parallel_semantic_adapter as _recipe,
)
from scripts.mem import train_pi0_mem_compress as _trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", type=pathlib.Path, default=_recipe.DEFAULT_V10_CHECKPOINT)
    parser.add_argument("--nominal-memory", type=pathlib.Path, default=_memory_data.DEFAULT_MEMORY_BANKS[_memory_data.NOMINAL_ROOT])
    parser.add_argument("--v6-memory", type=pathlib.Path, default=_memory_data.DEFAULT_MEMORY_BANKS[_memory_data.V6_ROOT])
    parser.add_argument("--v9-memory", type=pathlib.Path, default=_memory_data.DEFAULT_MEMORY_BANKS[_memory_data.V9_ROOT])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banks = {
        _memory_data.NOMINAL_ROOT: args.nominal_memory,
        _memory_data.V6_ROOT: args.v6_memory,
        _memory_data.V9_ROOT: args.v9_memory,
    }
    config = _recipe.make_train_config(
        config_module=_config,
        exp_name=args.exp_name,
        memory_banks=banks,
        init_checkpoint=args.init_checkpoint,
        steps=args.steps,
        peak_lr=args.peak_lr,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    # Reuse V10's exact frames 0..59 + current-frame data contract.
    _config_pi0_mem.VideoFrameDataset = FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _recipe.make_index_filter(banks)  # noqa: SLF001
    _trainer.main(config)


if __name__ == "__main__":
    main()
