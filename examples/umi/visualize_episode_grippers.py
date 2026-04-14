#!/usr/bin/env python3
"""
Load one LeRobot v2.1 episode parquet and plot **observation** grippers and **action** grippers
on **separate figures** (not overlaid).

Observations: ``robot{N}_gripper_width`` as nested lists (low-dim horizon x inner width).
Actions: last dim per robot; gripper index = ``robot_id * action_dim_per_robot + (action_dim_per_robot - 1)``.

With ``--save /tmp/gripper_ep0.png`` writes ``/tmp/gripper_ep0_obs.png`` and
``/tmp/gripper_ep0_actions.png`` (suffix preserved).

Usage (project venv at ``/root/openpi-umi/.venv``):

  /root/openpi-umi/.venv/bin/python /root/openpi-umi/examples/umi/visualize_episode_grippers.py \\
    --dataset-root /root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260408_151838_to_20260408_154016_ep11 \\
    --episode-index 0 \\
    --save /tmp/gripper_ep0.png

  # or: source /root/openpi-umi/.venv/bin/activate && python examples/umi/visualize_episode_grippers.py ...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def episode_parquet_path(dataset_root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _scalar_from_obs_gripper_cell(cell: Any, horizon_index: int) -> float:
    """Cell is list[list[float]] e.g. [[w], [w]] for horizon 2."""
    if cell is None:
        return float("nan")
    if not isinstance(cell, (list, tuple)) or len(cell) == 0:
        return float("nan")
    idx = horizon_index if horizon_index >= 0 else len(cell) + horizon_index
    idx = max(0, min(idx, len(cell) - 1))
    step = cell[idx]
    if isinstance(step, (list, tuple)):
        if len(step) == 0:
            return float("nan")
        return float(step[0])
    arr = np.asarray(step, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def _action_grippers_from_cell(cell: Any, action_row: int, gripper_indices: Sequence[int]) -> List[float]:
    """
    Cell is list[list[float]]: (action_horizon, action_dim).
    Return gripper values for each robot index at the chosen action row.
    """
    if cell is None or not isinstance(cell, (list, tuple)) or len(cell) == 0:
        return [float("nan")] * len(gripper_indices)
    rows = cell
    ridx = action_row if action_row >= 0 else len(rows) + action_row
    ridx = max(0, min(ridx, len(rows) - 1))
    row = rows[ridx]
    if not isinstance(row, (list, tuple)):
        return [float("nan")] * len(gripper_indices)
    vals = []
    for gi in gripper_indices:
        if gi < len(row):
            vals.append(float(row[gi]))
        else:
            vals.append(float("nan"))
    return vals


def _action_gripper_values_all_horizons(
    cell: Any, gripper_indices: Sequence[int]
) -> List[List[float]]:
    """For each action timestep row, return one float per robot gripper index."""
    if cell is None or not isinstance(cell, (list, tuple)) or len(cell) == 0:
        return []
    out: List[List[float]] = []
    for row in cell:
        if not isinstance(row, (list, tuple)):
            out.append([float("nan")] * len(gripper_indices))
            continue
        vals = []
        for gi in gripper_indices:
            vals.append(float(row[gi]) if gi < len(row) else float("nan"))
        out.append(vals)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize obs vs action grippers for one episode.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot dataset root (meta/ + data/).")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--action-dim-per-robot",
        type=int,
        default=10,
        help="Gripper flat index = robot_id * D + (D-1). Default 10 (UMI LeRobot export).",
    )
    parser.add_argument(
        "--obs-horizon-index",
        type=int,
        default=-1,
        help="Which low-dim observation timestep to plot (Python index; -1 = current / last in stack).",
    )
    parser.add_argument(
        "--action-row-index",
        type=int,
        default=0,
        help="Which row inside the stored action horizon (0 = first waypoint in chunk).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Base path for PNGs: writes <stem>_obs<suffix> and <stem>_actions<suffix>. Else show interactively.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames for faster debug.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="RNG seed for the random frame used by --print-random-action-gripper.",
    )
    parser.add_argument(
        "--print-random-action-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pick one random parquet row, then print that row ± --action-print-context rows: full action_horizon gripper values each.",
    )
    parser.add_argument(
        "--action-print-context",
        type=int,
        default=10,
        help="Number of parquet rows before/after the random center row to include in the gripper print (default: 10).",
    )
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    chunks_size = int(info.get("chunks_size", 1000))

    ep = args.episode_index
    pq_path = episode_parquet_path(root, ep, chunks_size)
    if not pq_path.is_file():
        raise FileNotFoundError(f"Missing episode parquet: {pq_path}")

    table = pq.read_table(pq_path)
    n = table.num_rows
    if args.max_frames is not None:
        n = min(n, args.max_frames)

    gripper_cols = sorted(
        k for k in (info.get("features") or {}).keys() if k.endswith("_gripper_width")
    )
    if not gripper_cols:
        raise SystemExit("No *_gripper_width in meta/features; nothing to plot for observations.")

    has_actions = "actions" in table.column_names
    if not has_actions:
        print("Warning: no ``actions`` column; only observation grippers will be plotted.")

    # robot ids from column names
    robot_ids: List[int] = []
    for col in gripper_cols:
        m = re.match(r"robot(\d+)_gripper_width", col)
        if m:
            robot_ids.append(int(m.group(1)))
        else:
            robot_ids.append(len(robot_ids))

    action_gripper_indices = [rid * args.action_dim_per_robot + (args.action_dim_per_robot - 1) for rid in robot_ids]

    if has_actions and args.print_random_action_gripper:
        ctx = max(0, int(args.action_print_context))
        rng = np.random.default_rng(args.sample_seed)
        sample_fi = int(rng.integers(0, table.num_rows))
        lo = max(0, sample_fi - ctx)
        hi = min(table.num_rows - 1, sample_fi + ctx)
        print(
            f"\n[sample] episode_index={ep}  center_parquet_row={sample_fi}  "
            f"print_rows=[{lo}, {hi}] (±{ctx} neighbors, {hi - lo + 1} rows)  "
            f"gripper_flat_indices={list(action_gripper_indices)}"
        )
        for fi in range(lo, hi + 1):
            acell = table.column("actions")[fi].as_py()
            per_t = _action_gripper_values_all_horizons(acell, action_gripper_indices)
            tag = "  <<<CENTER>>>" if fi == sample_fi else ""
            print(f"\n--- parquet_row={fi}{tag}  action_horizon={len(per_t)} ---")
            for t, vals in enumerate(per_t):
                parts = [
                    f"robot{rid}@[{gi}]={vals[j]:.6g}"
                    for j, (rid, gi) in enumerate(zip(robot_ids, action_gripper_indices))
                ]
                print(f"  action_t={t:2d}  " + "  ".join(parts))
        print()

    frame_idx = np.arange(n, dtype=np.int64)
    obs_series: dict[str, np.ndarray] = {}
    act_series: dict[str, np.ndarray] = {}

    for col in gripper_cols:
        obs_series[col] = np.full(n, np.nan, dtype=np.float64)
    if has_actions:
        for rid in robot_ids:
            act_series[f"robot{rid}_action_gripper"] = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        for j, col in enumerate(gripper_cols):
            cell = table.column(col)[i].as_py()
            obs_series[col][i] = _scalar_from_obs_gripper_cell(cell, args.obs_horizon_index)
        if has_actions:
            acell = table.column("actions")[i].as_py()
            vals = _action_grippers_from_cell(acell, args.action_row_index, action_gripper_indices)
            for j, rid in enumerate(robot_ids):
                act_series[f"robot{rid}_action_gripper"][i] = vals[j]

    n_plots = len(gripper_cols)
    fps = float(info.get("fps", 1.0))
    t_sec = frame_idx / fps
    meta_title = (
        f"{root.name}  episode_index={ep}  frames={n}  "
        f"obs_horizon_idx={args.obs_horizon_index}  action_row_idx={args.action_row_index}"
    )

    # Figure 1: observation grippers only
    fig_obs, axes_obs = plt.subplots(n_plots, 1, figsize=(11, 2.8 * n_plots), sharex=True)
    if n_plots == 1:
        axes_obs = [axes_obs]
    for ax, col in zip(axes_obs, gripper_cols):
        ax.plot(t_sec, obs_series[col], color="C0", linewidth=1.2, label=col)
        ax.set_ylabel("value")
        ax.set_title(f"Observation: {col}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes_obs[-1].set_xlabel("time (s)")
    fig_obs.suptitle(f"Observation grippers\n{meta_title}", fontsize=10)
    fig_obs.tight_layout()

    fig_act = None
    if has_actions:
        fig_act, axes_act = plt.subplots(n_plots, 1, figsize=(11, 2.8 * n_plots), sharex=True)
        if n_plots == 1:
            axes_act = [axes_act]
        for ax, rid, agi in zip(axes_act, robot_ids, action_gripper_indices):
            akey = f"robot{rid}_action_gripper"
            ax.plot(t_sec, act_series[akey], color="C1", linewidth=1.2, label=f"dim {agi}")
            ax.set_ylabel("value")
            ax.set_title(f"Actions: robot{rid} gripper (flat index {agi})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes_act[-1].set_xlabel("time (s)")
        fig_act.suptitle(f"Action tensor grippers\n{meta_title}", fontsize=10)
        fig_act.tight_layout()

    if args.save:
        base = args.save.expanduser().resolve()
        base.parent.mkdir(parents=True, exist_ok=True)
        stem, suf = base.stem, base.suffix if base.suffix else ".png"
        path_obs = base.parent / f"{stem}_obs{suf}"
        fig_obs.savefig(path_obs, dpi=150)
        print(f"Saved: {path_obs}")
        if fig_act is not None:
            path_act = base.parent / f"{stem}_actions{suf}"
            fig_act.savefig(path_act, dpi=150)
            print(f"Saved: {path_act}")
        plt.close(fig_obs)
        if fig_act is not None:
            plt.close(fig_act)
    else:
        plt.show()


if __name__ == "__main__":
    main()
