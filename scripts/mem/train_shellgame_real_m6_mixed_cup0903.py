#!/usr/bin/env python3
"""Train M6 over the complete gradual-action suffix of old306 + cup_0903."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.training.mem.recipes import shellgame_real_mixed_common as _mixed  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_m6_mixed as _recipe  # noqa: E402
from scripts.mem import train_pi0_mem_compress as _trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--peak-lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--fsdp-devices", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _recipe.make_train_config(
        exp_name=args.exp_name,
        checkpoint=args.checkpoint,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        save_interval=args.save_interval,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    _trainer._episode_split_indices = _mixed.fixed_episode_split_indices  # noqa: SLF001
    _trainer._filter_memory_classifier_frame_range = functools.partial(  # noqa: SLF001
        _mixed.filter_balanced_indices,
        decision_frame=None,
    )
    print(
        f"M6 mixed full suffix: frames>=241, batch={args.batch_size}, "
        "old/new=25%/75%, cup classes balanced per source",
        flush=True,
    )
    _trainer.main(config)


if __name__ == "__main__":
    main()
