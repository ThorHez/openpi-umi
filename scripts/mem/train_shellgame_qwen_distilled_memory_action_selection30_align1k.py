#!/usr/bin/env python3
"""Train the 1k-step Selection30 memory/action alignment control."""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.training import config as _config
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_selection30_align1k as _recipe
from train_pi0_mem_compress import main as _train_main
import train_pi0_mem_compress as _trainer


def parse_args() -> argparse.Namespace:
    selection = _recipe._selection  # noqa: SLF001
    base = selection._base  # noqa: SLF001
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--nominal-memory", type=pathlib.Path, default=base.DEFAULT_MEMORY_BANKS[base.NOMINAL_ROOT])
    parser.add_argument("--v6-memory", type=pathlib.Path, default=base.DEFAULT_MEMORY_BANKS[base.V6_ROOT])
    parser.add_argument("--v9-memory", type=pathlib.Path, default=base.DEFAULT_MEMORY_BANKS[base.V9_ROOT])
    parser.add_argument("--init-checkpoint", type=pathlib.Path, default=base.DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = _recipe._selection  # noqa: SLF001
    base = selection._base  # noqa: SLF001
    banks = {
        base.NOMINAL_ROOT: args.nominal_memory,
        base.V6_ROOT: args.v6_memory,
        base.V9_ROOT: args.v9_memory,
    }
    config = _recipe.make_train_config(
        config_module=_config,
        exp_name=args.exp_name,
        memory_banks=banks,
        init_checkpoint=args.init_checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    _trainer._filter_memory_classifier_frame_range = functools.partial(  # noqa: SLF001
        _recipe.filter_and_sample_indices,
        banks={root: path.resolve() for root, path in banks.items()},
    )
    _train_main(config)


if __name__ == "__main__":
    main()
