#!/usr/bin/env python3
"""Build the train70/dev15 PickXtimes memory-action array cache."""

from __future__ import annotations

import argparse
import pathlib

from openpi.training.mem import robomme_pickxtimes_action_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=pathlib.Path, required=True)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--split", type=pathlib.Path, required=True)
    parser.add_argument("--memories", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robomme_pickxtimes_action_dataset.write_action_cache(
        args.h5,
        args.features,
        args.labels,
        args.split,
        args.memories,
        args.output,
        frame_stride=args.frame_stride,
    )
    print(f"Wrote PickXtimes action cache to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
