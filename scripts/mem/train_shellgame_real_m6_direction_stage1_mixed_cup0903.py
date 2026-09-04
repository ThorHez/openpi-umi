#!/usr/bin/env python3
"""Train direction-guarded frame-241 M6 on old306 + cup_0903."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.training.mem.recipes import shellgame_real_mixed_common as _mixed  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_m6_direction_stage1_mixed as _recipe  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2  # noqa: E402
from scripts.mem import train_pi0_mem_compress as _trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--action-checkpoint", default=str(_mixed.OLD_H16_ACTION_CHECKPOINT))
    parser.add_argument("--memory-checkpoint", default=str(_mixed.ADAPTED_MEMORY_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--schedule-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--peak-lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--fsdp-devices", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--direction-loss-weight", type=float, default=0.1)
    parser.add_argument("--direction-temperature", type=float, default=5e-4)
    parser.add_argument("--disable-direction-early-stop", action="store_true")
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _recipe.make_train_config(
        exp_name=args.exp_name,
        action_checkpoint=args.action_checkpoint,
        memory_checkpoint=args.memory_checkpoint,
        steps=args.steps,
        schedule_steps=args.schedule_steps,
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        direction_loss_weight=args.direction_loss_weight,
        direction_temperature=args.direction_temperature,
        enable_direction_early_stop=not args.disable_direction_early_stop,
        save_interval=args.save_interval,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    _trainer._episode_split_indices = _mixed.fixed_episode_split_indices  # noqa: SLF001
    _trainer._filter_memory_classifier_frame_range = functools.partial(  # noqa: SLF001
        _mixed.filter_balanced_indices,
        decision_frame=_stage2.CURRENT_START_FRAME,
    )
    print(
        f"M6 mixed stage1: frame=241, flow+{args.direction_loss_weight:g}*direction loss, "
        f"batch={args.batch_size}, old/new=25%/75%, class-balanced",
        flush=True,
    )
    _trainer.main(config)


if __name__ == "__main__":
    main()
