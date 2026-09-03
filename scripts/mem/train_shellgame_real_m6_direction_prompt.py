#!/usr/bin/env python3
"""Train real-robot ShellGame M6 with explicit left/middle/right prompts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


OPENPI_ROOT = Path(__file__).resolve().parents[2]
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from openpi.training.mem.recipes import shellgame_real_wrist_m6 as _m6  # noqa: E402
from scripts.mem import train_pi0_mem_compress as _trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint", default=_m6.DEFAULT_M5_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=21_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--peak-lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _m6.make_train_config(
        exp_name=args.exp_name,
        checkpoint=args.checkpoint,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        save_interval=args.save_interval,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(
        "M6 contract: history=0..240, current>=241, state=episode-first EEF10, "
        "16 targets=current-frame-relative EEF10",
        flush=True,
    )
    print(
        f"checkpoint={args.checkpoint} exp={args.exp_name} "
        "prompt=GT final_cup during training; deployment prompt=MEM prediction",
        flush=True,
    )
    print(
        "trainable=HistoryRawMemoryQueryResampler + ActionMemoryCrossAttention + "
        "Pi0.5 action expert/projections; visual MEM frozen",
        flush=True,
    )
    _trainer.main(config)


if __name__ == "__main__":
    main()
