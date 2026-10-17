#!/usr/bin/env python3
"""Continue V1 with the memory-statistics waypoint correction branch."""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.training import config as _config
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_frame59_waypoint_stats as _recipe
from train_pi0_mem_compress import main as _train_main
import train_pi0_mem_compress as _trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--memory", type=pathlib.Path, default=_recipe.DEFAULT_MEMORY)
    parser.add_argument("--init-checkpoint", type=pathlib.Path, default=_recipe.DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = _recipe.make_train_config(
        config_module=_config,
        exp_name=args.exp_name,
        memory_path=args.memory,
        init_checkpoint=args.init_checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    _trainer._filter_memory_classifier_frame_range = functools.partial(  # noqa: SLF001
        _recipe.filter_frame59_correct_indices,
        memory_path=args.memory.expanduser().resolve(),
    )
    _train_main(config)


if __name__ == "__main__":
    main()
