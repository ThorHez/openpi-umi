#!/usr/bin/env python3
"""Continue prompt-only M6 on all action-time frames with direction-anchor replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.training.mem.recipes import shellgame_real_wrist_m6_prompt_ablation as _split  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_m6_prompt_only_full_suffix as _recipe  # noqa: E402
from scripts.mem import train_pi0_mem_compress as _trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint", default=str(_recipe.DEFAULT_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--peak-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--eval-batch-size", type=int, default=72)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--fsdp-devices", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--anchor-fraction", type=float, default=0.30)
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
        anchor_fraction=args.anchor_fraction,
        save_interval=args.save_interval,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    _trainer._episode_split_indices = _split.fixed_episode_split_indices  # noqa: SLF001
    _trainer._filter_memory_classifier_frame_range = (  # noqa: SLF001
        _recipe.filter_balanced_training_full_validation
    )
    print(
        "M6 prompt-only full suffix: pure Pi0.5 current-frame action input, flow frames>=241, direction loss=0, "
        f"frame241 anchor_fraction={args.anchor_fraction:g}, batch={args.batch_size}, old/new=25%/75%",
        flush=True,
    )
    _trainer.main(config)


if __name__ == "__main__":
    main()
