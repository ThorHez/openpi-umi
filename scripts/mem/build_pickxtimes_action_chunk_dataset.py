#!/usr/bin/env python3
"""Build aligned future EEF7 chunks for the PickXtimes action cache."""

from __future__ import annotations

import argparse
import pathlib

from openpi.training.mem import robomme_pickxtimes_action_chunk_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=pathlib.Path, required=True)
    parser.add_argument("--base-cache", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robomme_pickxtimes_action_chunk_dataset.write_action_chunk_targets(
        args.h5,
        args.base_cache,
        args.output,
        action_horizon=args.action_horizon,
    )
    print(f"Wrote PickXtimes action chunks to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
