#!/usr/bin/env python3
"""
Analyze a UMI zarr dataset to find episodes with empty or mismatched columns.

Checks each episode for:
  - Columns with length 0
  - Columns whose length differs from the majority (episode length mismatch)

Usage:
    python analyze_zarr_empty_columns.py --zarr /path/to/dataset.zarr.zip
    python analyze_zarr_empty_columns.py --zarr /path/to/dataset.zarr.zip --keys robot0_eef_pos robot1_eef_pos
"""

import argparse
import os
import tempfile
import zipfile
from collections import defaultdict

import numpy as np
import zarr
from tqdm import tqdm


def analyze_zarr(zarr_path: str, keys: list[str] | None = None):
    is_zip = zarr_path.endswith(".zip")
    if is_zip:
        tmp_dir = tempfile.mkdtemp(prefix="zarr_analyze_")
        print(f"Extracting {zarr_path} -> {tmp_dir}")
        with zipfile.ZipFile(zarr_path, "r") as zf:
            zf.extractall(tmp_dir)
        root = zarr.open(tmp_dir, mode="r")
    else:
        root = zarr.open(zarr_path, mode="r")
        tmp_dir = None

    data = root["data"]
    meta = root["meta"]

    all_keys = sorted(data.keys())
    print(f"\nAll data keys ({len(all_keys)}):")
    for k in all_keys:
        arr = data[k]
        print(f"  {k:45s}  shape={arr.shape}  dtype={arr.dtype}")

    episode_ends = np.array(meta["episode_ends"][:])
    num_episodes = len(episode_ends)
    print(f"\nTotal episodes: {num_episodes}")
    print(f"Total frames (last episode_end): {episode_ends[-1]}")

    check_keys = keys if keys else all_keys
    missing_keys = [k for k in check_keys if k not in data]
    if missing_keys:
        print(f"\nWARNING: keys not found in dataset: {missing_keys}")
        check_keys = [k for k in check_keys if k not in missing_keys]

    print(f"\nChecking {len(check_keys)} keys per episode...")

    # Per-episode analysis
    problems = []  # list of (ep_idx, key, issue_str, ep_len, actual_len)
    empty_episodes = set()
    key_empty_count = defaultdict(int)

    for ep_idx in tqdm(range(num_episodes), desc="Scanning episodes"):
        start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        end = int(episode_ends[ep_idx])
        expected_len = end - start

        for key in check_keys:
            arr = data[key]
            actual_len = end - start  # default
            try:
                sliced = arr[start:end]
                actual_len = len(sliced)
            except Exception as e:
                problems.append((ep_idx, key, f"read error: {e}", expected_len, 0))
                continue

            if actual_len == 0:
                problems.append((ep_idx, key, "EMPTY (length=0)", expected_len, 0))
                empty_episodes.add(ep_idx)
                key_empty_count[key] += 1
            elif actual_len != expected_len:
                problems.append((ep_idx, key, f"length mismatch", expected_len, actual_len))

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)

    if not problems:
        print("\nNo problems found. All episodes have consistent, non-empty data.")
    else:
        print(f"\nTotal issues found: {len(problems)}")
        print(f"Episodes with empty columns: {sorted(empty_episodes)}")
        print(f"Number of affected episodes: {len(empty_episodes)} / {num_episodes}")

        if key_empty_count:
            print(f"\nEmpty count per key:")
            for k, cnt in sorted(key_empty_count.items(), key=lambda x: -x[1]):
                print(f"  {k:45s}  empty in {cnt}/{num_episodes} episodes")

        print(f"\nDetailed issues:")
        for ep_idx, key, issue, expected, actual in problems:
            print(f"  Episode {ep_idx:4d}  |  {key:40s}  |  {issue}  (expected={expected}, actual={actual})")

    if tmp_dir and is_zip:
        print(f"\nTemp dir: {tmp_dir}  (remove manually if not needed)")


def main():
    parser = argparse.ArgumentParser(description="Analyze zarr dataset for empty/mismatched columns")
    parser.add_argument("--zarr", type=str, required=True, help="Path to zarr zip or directory")
    parser.add_argument("--keys", type=str, nargs="+", default=None,
                        help="Specific keys to check (default: all)")
    args = parser.parse_args()
    analyze_zarr(args.zarr, args.keys)


if __name__ == "__main__":
    main()
