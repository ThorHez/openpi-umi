#!/usr/bin/env python3
"""Read a local UMI LeRobot dataset and print the first N rows."""

from __future__ import annotations

import argparse
import pprint
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a local LeRobot dataset and print the first rows."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("/root/openpi-umi/data/umi_lerobot_dataset_hitl_test_with_value"),
        help="Local path to the LeRobot dataset.",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=100,
        help="Number of rows to print.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help=(
            "Only print specific columns. Use space-separated names, "
            'e.g. --columns index episode_index task. '
            "If omitted, print the full row."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if args.num_rows <= 0:
        raise ValueError("--num-rows must be > 0")

    dataset = LeRobotDataset(
        repo_id=str(dataset_path),
        root=str(dataset_path),
    )
    hf_dataset = dataset.hf_dataset.with_format(None)

    total_rows = len(hf_dataset)
    rows_to_print = min(args.num_rows, total_rows)
    available_columns = list(hf_dataset.column_names)

    selected_columns = args.columns
    if selected_columns is not None:
        unknown_columns = [col for col in selected_columns if col not in available_columns]
        if unknown_columns:
            raise ValueError(
                "Unknown column(s): "
                f"{unknown_columns}. Available columns: {available_columns}"
            )

    print(f"Dataset path: {dataset_path}")
    print(f"Total rows: {total_rows}")
    print(f"Columns: {available_columns}")
    if selected_columns is not None:
        print(f"Selected columns: {selected_columns}")
    print(f"Printing first {rows_to_print} rows")
    print("-" * 80)

    for idx in range(rows_to_print):
        row = hf_dataset[idx]
        if selected_columns is not None:
            row = {k: row.get(k) for k in selected_columns}
        print(f"[{idx}]")
        pprint.pprint(row, width=120, compact=False)
        print("-" * 80)


if __name__ == "__main__":
    main()
