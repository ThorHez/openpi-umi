#!/usr/bin/env python3
"""Train the deterministic real-robot ShellGame M5 semantic-action probe.

Examples:

    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python \
      scripts/mem/train_shellgame_real_m5_action_probe.py \
      --semantic-source oracle --exp-name real306_m5_oracle_seed42

    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python \
      scripts/mem/train_shellgame_real_m5_action_probe.py \
      --semantic-source memory --exp-name real306_m5_memory_seed42
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


OPENPI_ROOT = Path(__file__).resolve().parents[2]
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from openpi.training.mem.recipes import shellgame_real_wrist_m5 as _m5  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2  # noqa: E402
from scripts.mem import train_pi0_mem_compress as _trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantic-source",
        choices=("oracle", "memory"),
        default="memory",
        help="Use GT final-cup one-hot or frozen MEM final-cup probabilities.",
    )
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint", default=_stage2.MEMORY_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _m5.make_train_config(
        semantic_source=args.semantic_source,
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
        overwrite=args.overwrite,
    )
    print(
        "M5 contract: history=frames 0..240, current=241, "
        "targets=commands 242..257, action=current-relative EEF10",
        flush=True,
    )
    print(
        f"semantic_source={args.semantic_source} checkpoint={args.checkpoint} "
        f"trainable=HistorySemanticJointActionReadout exp={args.exp_name}",
        flush=True,
    )
    _trainer.main(config)


if __name__ == "__main__":
    main()
