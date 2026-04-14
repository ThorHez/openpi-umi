#!/usr/bin/env python3
"""Count binned gripper values (0 vs 0.085) in a LeRobot v2.1 parquet dataset.

Run with the project venv, for example:
  /root/openpi-umi/.venv/bin/python scripts/analyze_gripper_bin_dataset.py \\
    /path/to/dataset_root
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


def _count_values(arr: np.ndarray, tol: float) -> tuple[int, int, int]:
    """Return counts for ~0, ~0.085, and other."""
    flat = np.asarray(arr, dtype=np.float64).ravel()
    is0 = np.abs(flat) < tol
    is085 = np.abs(flat - 0.085) < tol
    n0 = int(np.sum(is0))
    n085 = int(np.sum(is085))
    not_bin = ~(is0 | is085)
    n_other = int(np.sum(not_bin))
    return n0, n085, n_other


def analyze_actions(
    dataset_data_dir: Path,
    indices: tuple[int, ...],
    batch_size: int,
    tol: float,
) -> dict:
    table = ds.dataset(dataset_data_dir, format="parquet")
    scanner = table.scanner(columns=["actions"], batch_size=batch_size)
    n0 = n085 = n_other = 0
    for batch in scanner.to_batches():
        col = batch.column(0)
        for row in col.to_pylist():
            a = np.asarray(row, dtype=np.float64)
            parts = [a[:, idx] for idx in indices]
            g = np.concatenate(parts) if parts else np.array([])
            c0, c085, co = _count_values(g, tol)
            n0 += c0
            n085 += c085
            n_other += co
    return _summary("actions", indices, n0, n085, n_other, tol)


def analyze_observation(
    dataset_data_dir: Path,
    batch_size: int,
    tol: float,
) -> dict:
    cols = ["robot0_gripper_width", "robot1_gripper_width"]
    table = ds.dataset(dataset_data_dir, format="parquet")
    scanner = table.scanner(columns=cols, batch_size=batch_size)
    n0 = n085 = n_other = 0
    for batch in scanner.to_batches():
        for i in range(batch.num_columns):
            col = batch.column(i)
            for row in col.to_pylist():
                g = np.asarray(row, dtype=np.float64)
                c0, c085, co = _count_values(g, tol)
                n0 += c0
                n085 += c085
                n_other += co
    return _summary("observation_gripper_width", ("robot0", "robot1"), n0, n085, n_other, tol)


def _summary(
    source: str,
    fields: tuple[int, ...] | tuple[str, ...],
    n0: int,
    n085: int,
    n_other: int,
    tol: float,
) -> dict:
    total = n0 + n085 + n_other
    denom = n0 + n085
    return {
        "source": source,
        "fields": fields,
        "tolerance": tol,
        "count_0": n0,
        "count_0_085": n085,
        "count_other": n_other,
        "total_samples": total,
        "fraction_0_among_bin": (n0 / denom) if denom else None,
        "fraction_0_085_among_bin": (n085 / denom) if denom else None,
        "ratio_0_085_to_0": (n085 / n0) if n0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=Path(
            "/root/openpi-umi/data/"
            "horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin"
        ),
        help="LeRobot dataset root (contains meta/ and data/).",
    )
    parser.add_argument(
        "--source",
        choices=("actions", "observation", "both"),
        default="both",
        help="Where to read gripper: action channels, observation widths, or both.",
    )
    parser.add_argument(
        "--action-indices",
        type=str,
        default="9,19",
        help="Comma-separated action column indices for bimanual gripper (default 9,19).",
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    data_dir = args.dataset_root / "data"
    if not data_dir.is_dir():
        raise SystemExit(f"Missing data directory: {data_dir}")

    indices = tuple(int(x.strip()) for x in args.action_indices.split(",") if x.strip())
    results: list[dict] = []

    if args.source in ("actions", "both"):
        results.append(analyze_actions(data_dir, indices, args.batch_size, args.tolerance))
    if args.source in ("observation", "both"):
        results.append(analyze_observation(data_dir, args.batch_size, args.tolerance))

    meta = args.dataset_root / "meta" / "info.json"
    if meta.is_file():
        info = json.loads(meta.read_text())
        print(
            "dataset:",
            args.dataset_root,
            f"(episodes={info.get('total_episodes')}, frames={info.get('total_frames')})",
        )
    else:
        print("dataset:", args.dataset_root)

    for r in results:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        denom = r["count_0"] + r["count_0_085"]
        if r["count_other"]:
            print(
                f"  WARNING: {r['count_other']} samples are neither 0 nor 0.085 "
                f"(tolerance={r['tolerance']}).",
            )
        if denom:
            print(
                f"  Among bin {{0, 0.085}}: P(0)={r['count_0']/denom:.6f}, "
                f"P(0.085)={r['count_0_085']/denom:.6f}, "
                f"ratio 0.085:0 = {r['count_0_085']/r['count_0']:.6f}"
                if r["count_0"]
                else f"  Among bin {{0, 0.085}}: P(0.085)=1.0 (no zeros)",
            )


if __name__ == "__main__":
    main()
