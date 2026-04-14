#!/usr/bin/env python3
"""Detect episodes with abnormal timestamp intervals in a zarr.zip dataset.

Usage:
    python scripts/detect_timestamp_anomaly.py <path_to_zarr_zip> [--threshold 0.15]
"""

import argparse
import zarr
import numpy as np
import sys


def detect_anomalies(zarr_path: str, threshold: float = 0.15):
    store = zarr.storage.ZipStore(zarr_path, mode='r')
    root = zarr.open(store, mode='r')

    ep_ends = root['meta/episode_ends'][:]
    ts_all = root['data/timestamp'][:, 0]
    n_episodes = len(ep_ends)

    print(f'Dataset: {zarr_path}')
    print(f'Total episodes: {n_episodes}')
    print(f'Total frames: {len(ts_all)}')
    print(f'Abnormal threshold: dt > {threshold}s (expected ~0.05s at 20Hz)')
    print()

    abnormal_episodes = []
    normal_episodes = []

    ep_start = 0
    for i in range(n_episodes):
        ep_end = int(ep_ends[i])
        ts = ts_all[ep_start:ep_end]
        n_frames = len(ts)
        diffs = np.diff(ts)
        bad_indices = np.where(diffs > threshold)[0]
        n_bad = len(bad_indices)
        max_gap = float(np.max(diffs))
        mean_dt = float(np.mean(diffs))
        median_dt = float(np.median(diffs))
        total_time = float(ts[-1] - ts[0])

        info = {
            'ep': i, 'n_frames': n_frames, 'n_bad': n_bad,
            'max_gap': max_gap, 'mean_dt': mean_dt, 'median_dt': median_dt,
            'total_time': total_time,
        }

        if n_bad >= 1:
            total_bad_time = float(np.sum(diffs[bad_indices]))
            bf = bad_indices.tolist()
            ranges = []
            start = bf[0]; prev = bf[0]
            for f in bf[1:]:
                if f == prev + 1:
                    prev = f
                else:
                    ranges.append(f'{start}-{prev}' if start != prev else str(start))
                    start = f; prev = f
            ranges.append(f'{start}-{prev}' if start != prev else str(start))
            info.update({
                'total_bad_time': total_bad_time,
                'range_str': ', '.join(ranges),
            })
            abnormal_episodes.append(info)
        else:
            normal_episodes.append(info)
            print(f'  Ep {i:3d}: NORMAL | {n_frames:5d} frames | {total_time:.2f}s | mean_dt={mean_dt:.4f}s ({1/mean_dt:.1f}Hz) | max_gap={max_gap:.4f}s')

        ep_start = ep_end

    print()
    print(f'========== RESULT ==========')
    print(f'Normal episodes:   {len(normal_episodes)} / {n_episodes}')
    print(f'Abnormal episodes: {len(abnormal_episodes)} / {n_episodes}')

    if abnormal_episodes:
        print()
        header = f'{"Ep":>4} {"Frames":>7} {"#Bad":>5} {"MaxGap(s)":>10} {"MeanDt(s)":>10} {"MedianDt":>10} {"MeanHz":>8} {"BadTime(s)":>11} {"TotalTime":>10} {"Bad%":>6}  {"Bad Frame Ranges"}'
        print(header)
        print('-' * len(header) + '-' * 30)
        for e in abnormal_episodes:
            bad_pct = e['total_bad_time'] / e['total_time'] * 100 if e['total_time'] > 0 else 0
            mean_hz = 1 / e['mean_dt'] if e['mean_dt'] > 0 else 0
            print(f'{e["ep"]:>4} {e["n_frames"]:>7} {e["n_bad"]:>5} {e["max_gap"]:>10.4f} {e["mean_dt"]:>10.4f} {e["median_dt"]:>10.4f} {mean_hz:>8.1f} {e["total_bad_time"]:>11.3f} {e["total_time"]:>10.2f} {bad_pct:>5.1f}%  [{e["range_str"]}]')

    print()
    severe = [e for e in abnormal_episodes if e['max_gap'] > 0.5 or e['n_bad'] >= 5]
    moderate = [e for e in abnormal_episodes if e not in severe and (e['max_gap'] > 0.3 or e['n_bad'] >= 3)]
    mild = [e for e in abnormal_episodes if e not in severe and e not in moderate]

    print('=== Severity Classification ===')
    print(f'  SEVERE   (max_gap > 0.5s or >= 5 bad frames): {len(severe):3d}  {[e["ep"] for e in severe] if severe else ""}')
    print(f'  MODERATE (max_gap > 0.3s or >= 3 bad frames): {len(moderate):3d}  {[e["ep"] for e in moderate] if moderate else ""}')
    print(f'  MILD     (others):                            {len(mild):3d}  {[e["ep"] for e in mild] if mild else ""}')
    print(f'  NORMAL   (no gaps > {threshold}s):              {len(normal_episodes):3d}')

    store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect timestamp anomalies in zarr.zip dataset')
    parser.add_argument('zarr_path', help='Path to the zarr.zip file')
    parser.add_argument('--threshold', type=float, default=0.15, help='Abnormal dt threshold in seconds (default: 0.15)')
    args = parser.parse_args()
    detect_anomalies(args.zarr_path, args.threshold)
