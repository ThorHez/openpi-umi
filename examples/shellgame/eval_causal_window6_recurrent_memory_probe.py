"""Evaluate causal six-frame recurrent updates against fixed six-frame chunks.

No training or future-score ranking is used. ``causal_sliding6`` visits all
six-frame windows in chronological order and triggers only on a low-to-high
event-gate transition. ``fixed6_offset0`` is the literal ten-block partition
of frames 0..59; offsets 1..5 diagnose sensitivity to the chunk boundary.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.shellgame import train_sliding_window6_event_recurrent_memory_probe as _six
from examples.shellgame import train_sliding_window_event_recurrent_memory_probe as _window
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as _recipe
from scripts.mem import train_semantic_memory as _trainer

DEFAULT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "shellgame_sliding_window6_event_recurrent_memory_probe/"
    "sliding_window6_event_gate_500_260821/499/params"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="causal_window6_recurrent_eval_260821")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _six._configure_six_frame_globals()  # noqa: SLF001
    _window.ENABLE_CAUSAL_EVAL_SELECTIONS = True
    _window.CONDITION_NAMES = (
        "aligned",
        "automatic",
        "automatic_no_cross",
        "forced_cross",
        "causal_sliding6",
        "fixed6_offset0",
        "fixed6_offset1",
        "fixed6_offset2",
        "fixed6_offset3",
        "fixed6_offset4",
        "fixed6_offset5",
    )
    _recipe.compute_objective = _window.sliding_window_objective
    _trainer.eval_step = _window.sliding_full_eval_step

    config_args = argparse.Namespace(
        exp_name=args.exp_name,
        init_checkpoint=args.checkpoint,
        steps=0,
        warmup_steps=0,
        peak_lr=1e-4,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        eval_interval=1,
        eval_batches=args.eval_batches,
        save_interval=1,
        overwrite=args.overwrite,
    )
    _trainer.main(_six.build_config(config_args))


if __name__ == "__main__":
    main()
