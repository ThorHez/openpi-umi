#!/usr/bin/env python3
# ruff: noqa: E402
"""Low-LR adaptation of V10 action weights to frozen Qwen memory/waypoint."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_pi0_mem_compress as _trainer
from train_pi0_mem_compress import main as _train_main

from openpi.training import config as _config
from openpi.training.mem.recipes import shellgame_qwen_memory_waypoint_v10_action_adapt as _recipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--current-checkpoint", type=pathlib.Path, default=_recipe.DEFAULT_CURRENT_CHECKPOINT)
    parser.add_argument(
        "--v10-action-checkpoint",
        type=pathlib.Path,
        default=_recipe.DEFAULT_V10_ACTION_CHECKPOINT,
    )
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--peak-lr", type=float, default=1e-6)
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
        current_checkpoint=args.current_checkpoint,
        v10_action_checkpoint=args.v10_action_checkpoint,
        steps=args.steps,
        peak_lr=args.peak_lr,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    _trainer._filter_memory_classifier_frame_range = _recipe.make_index_filter()  # noqa: SLF001
    _train_main(config)


if __name__ == "__main__":
    main()
