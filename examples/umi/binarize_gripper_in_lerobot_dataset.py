#!/usr/bin/env python3
"""
Post-process an existing LeRobot v2.1 dataset: binarize gripper width observations and the
matching gripper dimensions inside ``actions`` (same ``features.*_gripper_width.binarize`` and
``dataset.action_dim`` as ``convert_umi_data_to_lerobot_add_target_value.py``).

Rewrites per-episode parquet files (in place or under ``--output``) and regenerates
``meta/episodes_stats.jsonl``. Does not update ``norm_stats.json``; recompute training stats if needed.

Usage (from repo root):

  uv run python examples/umi/binarize_gripper_in_lerobot_dataset.py \\
    --dataset-root /root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83 \\
    --config examples/umi/config/bimanual_dataset_config_head_view_with_depth_hitl.yaml

  uv run python examples/umi/binarize_gripper_in_lerobot_dataset.py \\
    --dataset-root /path/to/src \\
    --output /path/to/dst \\
    --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

_UMI_DIR = Path(__file__).resolve().parent
if str(_UMI_DIR) not in sys.path:
    sys.path.insert(0, str(_UMI_DIR))

from dataset_config_loader import load_dataset_config, validate_config
from remove_abnormal_data import compute_episode_stats, get_heavy_columns


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_episodes_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_tasks_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def episode_parquet_path(dataset_root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _print_gripper_binarize_config(dataset_config: Dict[str, Any] | None) -> None:
    print("\nGripper width binarize (YAML features.*_gripper_width.binarize):")
    if not dataset_config:
        print("  (empty config)")
        return
    features = dataset_config.get("features") or {}
    names = sorted(n for n in features if str(n).endswith("_gripper_width"))
    if not names:
        print("  (no *_gripper_width entries under features)")
        return
    for feat_name in names:
        feat_def = features[feat_name]
        bin_cfg = feat_def.get("binarize")
        if not isinstance(bin_cfg, dict):
            print(f"  {feat_name}: (no binarize section)")
            continue
        enabled = bool(bin_cfg.get("enabled", False))
        lo_raw, hi_raw = bin_cfg.get("min_value"), bin_cfg.get("max_value")
        min_val = float(lo_raw) if lo_raw is not None else 0.0
        max_val = float(hi_raw) if hi_raw is not None else 1.0
        threshold = float(bin_cfg.get("threshold", 0.05))
        invert = bool(bin_cfg.get("invert", False))
        hi_out, lo_out = (min_val, max_val) if invert else (max_val, min_val)
        print(f"  {feat_name}:")
        print(f"    enabled={enabled}")
        print(f"    min_value={min_val}  max_value={max_val}  threshold={threshold}  invert={invert}")
        print(f"    when enabled: width > threshold -> {hi_out}, width <= threshold -> {lo_out}")
    if names:
        print(
            "  (when enabled, the same rule applies to each arm's gripper dim in ``actions``)"
        )


def _gripper_binarize_plans(dataset_config: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    """
    column_name -> (threshold, value_if_gt, value_if_le).
    Matches convert_umi_data_to_lerobot_add_target_value._maybe_binarize_gripper_features.
    """
    plans: Dict[str, Tuple[float, float, float]] = {}
    for feat_name, feat_def in (dataset_config.get("features") or {}).items():
        if not str(feat_name).endswith("_gripper_width"):
            continue
        bin_cfg = feat_def.get("binarize")
        if not isinstance(bin_cfg, dict) or not bin_cfg.get("enabled", False):
            continue
        threshold = float(bin_cfg.get("threshold", 0.05))
        invert = bool(bin_cfg.get("invert", False))
        lo_raw, hi_raw = bin_cfg.get("min_value"), bin_cfg.get("max_value")
        min_val = float(lo_raw) if lo_raw is not None else 0.0
        max_val = float(hi_raw) if hi_raw is not None else 1.0
        hi_out, lo_out = (min_val, max_val) if invert else (max_val, min_val)
        plans[feat_name] = (threshold, hi_out, lo_out)
    return plans


def _gripper_index_to_plan(
    plans: Dict[str, Tuple[float, float, float]],
    action_dim_per_robot: int,
) -> Dict[int, Tuple[float, float, float]]:
    """Map flat ``actions`` index -> (threshold, hi_if_gt, lo_if_le) for each enabled arm."""
    idx_plans: Dict[int, Tuple[float, float, float]] = {}
    for feat_name, t in plans.items():
        m = re.match(r"robot(\d+)_gripper_width$", feat_name)
        if not m:
            continue
        rid = int(m.group(1))
        gi = rid * action_dim_per_robot + (action_dim_per_robot - 1)
        idx_plans[gi] = t
    return idx_plans


def _binarize_actions_row(
    row: List[float],
    idx_plans: Dict[int, Tuple[float, float, float]],
) -> List[float]:
    out = list(row)
    for gi, (threshold, hi_out, lo_out) in idx_plans.items():
        if gi >= len(out):
            continue
        x = float(out[gi])
        out[gi] = float(hi_out if x > threshold else lo_out)
    return out


def _binarize_actions_nested(val: Any, idx_plans: Dict[int, Tuple[float, float, float]]) -> Any:
    """``actions`` cell: list[ timestep ][ action_dim ] floats."""
    if not isinstance(val, list):
        return val
    if len(val) == 0:
        return val
    if isinstance(val[0], list):
        return [_binarize_actions_row(row, idx_plans) for row in val]
    return val


def _binarize_nested_lists(val: Any, threshold: float, hi_out: float, lo_out: float) -> Any:
    """Parquet cells are nested lists of floats (e.g. list<list<float>>); binarize leaves."""
    if isinstance(val, list):
        if len(val) == 0:
            return val
        if isinstance(val[0], list):
            return [_binarize_nested_lists(v, threshold, hi_out, lo_out) for v in val]
        arr = np.asarray(val, dtype=np.float32)
        out = np.where(arr > threshold, hi_out, lo_out).astype(np.float32)
        return out.tolist()
    # scalar
    x = float(val)
    return float(hi_out if x > threshold else lo_out)


def _transform_parquet_table(
    table: pa.Table,
    plans: Dict[str, Tuple[float, float, float]],
    *,
    action_dim_per_robot: int,
) -> pa.Table:
    idx_plans = _gripper_index_to_plan(plans, action_dim_per_robot)
    arrays = []
    names: List[str] = []
    for name in table.column_names:
        field = table.schema.field(name)
        py_type = field.type
        col = table.column(name)
        if name in plans:
            threshold, hi_out, lo_out = plans[name]
            new_py = [
                _binarize_nested_lists(col[i].as_py(), threshold, hi_out, lo_out)
                for i in range(table.num_rows)
            ]
            arrays.append(pa.array(new_py, type=py_type))
            names.append(name)
        elif name == "actions" and idx_plans:
            new_py = [_binarize_actions_nested(col[i].as_py(), idx_plans) for i in range(table.num_rows)]
            arrays.append(pa.array(new_py, type=py_type))
            names.append(name)
        else:
            arrays.append(col)
            names.append(name)
    return pa.Table.from_arrays(arrays, schema=table.schema)


def _prepare_output_root(
    src: Path, dst: Path, *, force: bool, copy_stale_stats: bool
) -> None:
    if dst.resolve() == src.resolve():
        return
    if dst.exists():
        if not force:
            raise FileExistsError(f"--output exists: {dst} (pass --force to overwrite)")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    for p in ["info.json", "episodes.jsonl", "tasks.jsonl"]:
        sp = src / "meta" / p
        if sp.is_file():
            shutil.copy2(sp, dst / "meta" / p)
    if copy_stale_stats:
        st = src / "meta" / "episodes_stats.jsonl"
        if st.is_file():
            shutil.copy2(st, dst / "meta" / "episodes_stats.jsonl")
            print(
                "Warning: copied episodes_stats.jsonl from source (--skip-stats); "
                "gripper stats inside are stale vs new parquets."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Binarize gripper columns in a LeRobot dataset using YAML config.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root (contains meta/info.json and data/chunk-*/episode_*.parquet).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML with features.*_gripper_width.binarize (same as UMI convert script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write a new dataset here; default: modify --dataset-root in place.",
    )
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing --output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print plans only; do not write files.")
    parser.add_argument("--skip-stats", action="store_true", help="Do not regenerate meta/episodes_stats.jsonl.")
    args = parser.parse_args()

    src = args.dataset_root.expanduser().resolve()
    if not (src / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing meta/info.json under {src}")

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    dataset_config = load_dataset_config(str(cfg_path))
    warnings = validate_config(dataset_config)
    if warnings:
        print("Config warnings:")
        for w in warnings:
            print(f"  - {w}")

    _print_gripper_binarize_config(dataset_config)
    plans = _gripper_binarize_plans(dataset_config)
    if not plans:
        raise SystemExit(
            "No enabled gripper binarize rules found under features.*_gripper_width.binarize "
            "(set enabled: true for at least one arm)."
        )

    dst = args.output.expanduser().resolve() if args.output else src
    if args.output and not args.dry_run:
        _prepare_output_root(
            src, dst, force=args.force, copy_stale_stats=args.skip_stats
        )

    info_src = load_json(src / "meta" / "info.json")
    missing = [c for c in plans if c not in info_src.get("features", {})]
    if missing:
        print(f"Warning: columns in config but not in dataset features meta: {missing}")

    episodes = load_episodes_jsonl(src / "meta" / "episodes.jsonl")
    chunks_size = int(info_src.get("chunks_size", 1000))
    sorted_indices = sorted(int(ep["episode_index"]) for ep in episodes)
    action_dim = int(dataset_config.get("dataset", {}).get("action_dim", 10))
    idx_plans = _gripper_index_to_plan(plans, action_dim)

    print(f"\nDataset: {src}")
    print(f"  Episodes: {len(sorted_indices)}")
    print(f"  Columns to binarize: {list(plans.keys())}")
    if idx_plans:
        print(f"  actions: same rule at flat indices (0-based): {sorted(idx_plans.keys())} (action_dim/robot={action_dim})")
    if dst != src:
        print(f"  Output: {dst}")
    if (src / "norm_stats.json").is_file():
        print("  Note: norm_stats.json present; recompute norm stats after gripper distribution change if training uses them.")

    if args.dry_run:
        print("\nDry run: no files written.")
        return

    if idx_plans:
        acts_meta = (info_src.get("features") or {}).get("actions")
        if acts_meta:
            ad = int(acts_meta.get("shape", [0, 0])[-1])
            need = max(idx_plans.keys()) + 1
            if ad < need:
                raise ValueError(
                    f"meta features.actions last dim is {ad} but binarize needs index up to {need - 1}"
                )

    work_root = dst
    for ep_idx in tqdm(sorted_indices, desc="Binarize episodes"):
        src_pq = episode_parquet_path(src, ep_idx, chunks_size)
        if not src_pq.is_file():
            raise FileNotFoundError(f"Missing parquet: {src_pq}")
        table = pq.read_table(src_pq)
        for col in plans:
            if col not in table.column_names:
                raise ValueError(f"Episode {ep_idx}: column {col!r} missing from parquet")
        if idx_plans and "actions" not in table.column_names:
            raise ValueError(
                f"Episode {ep_idx}: gripper binarize requires an ``actions`` column in parquet"
            )
        new_table = _transform_parquet_table(table, plans, action_dim_per_robot=action_dim)
        out_pq = episode_parquet_path(work_root, ep_idx, chunks_size)
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, out_pq)

    if args.skip_stats:
        print("Skipped episodes_stats.jsonl (--skip-stats).")
        return

    heavy = get_heavy_columns(info_src)
    stats_path = work_root / "meta" / "episodes_stats.jsonl"
    with open(stats_path, "w", encoding="utf-8") as f:
        for ep_idx in tqdm(sorted_indices, desc="Episode stats"):
            out_pq = episode_parquet_path(work_root, ep_idx, chunks_size)
            t = pq.read_table(out_pq)
            st = compute_episode_stats(t, heavy)
            f.write(json.dumps({"episode_index": ep_idx, "stats": st}) + "\n")

    print(f"\nDone. Updated {len(sorted_indices)} episode parquets under {work_root / 'data'}")
    print(f"  episodes_stats.jsonl -> {stats_path}")


if __name__ == "__main__":
    main()
