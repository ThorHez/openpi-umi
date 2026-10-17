#!/usr/bin/env python3
"""Pool PickXtimes SigLIP patches into a 4x4 action-conditioning grid."""

from __future__ import annotations

import argparse
import pathlib

from openpi.training.mem import robomme_pickxtimes_action_chunk_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--base-cache", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robomme_pickxtimes_action_chunk_dataset.write_spatial_visual_cache(
        args.features,
        args.base_cache,
        args.output,
    )
    print(f"Wrote spatial PickXtimes action features to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
