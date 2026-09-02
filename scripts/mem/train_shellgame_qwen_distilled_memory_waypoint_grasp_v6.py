#!/usr/bin/env python3
"""Train full ShellGame grasping behind the frozen MEM-to-XY waypoint."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.training import config as _config
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_waypoint_grasp_v6 as _recipe
from train_pi0_mem_compress import main as _train_main
import train_pi0_mem_compress as _trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", type=pathlib.Path, default=_recipe.DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _recipe.make_train_config(
        config_module=_config,
        exp_name=args.exp_name,
        init_checkpoint=args.init_checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    _trainer._filter_memory_classifier_frame_range = _recipe.make_index_filter()  # noqa: SLF001
    _train_main(config)


if __name__ == "__main__":
    main()
