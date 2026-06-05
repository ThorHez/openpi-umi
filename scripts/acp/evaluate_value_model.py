"""Evaluate value model prediction accuracy on annotated LeRobot datasets.

Reads predicted_value, value_target, advantage, is_positive, success, etc.
from parquet files written by lerobot_value_infer.py and produces:
  1. Global regression metrics (MSE, MAE, Pearson r, R²)
  2. Per-episode regression metrics
  3. Success vs failure distribution analysis
  4. Temporal analysis within episodes
  5. Advantage distribution & is_positive consistency
  6. Per-episode value prediction curves (saved as PNG)

Usage:
    python scripts/evaluate_value_model.py \
        --dataset-root ./data/my_lerobot_dataset \
        [--output-dir ./value_eval_results] \
        [--plot-episodes 5]
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy import stats

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_root: Path) -> dict[str, np.ndarray]:
    """Load all parquet files and extract relevant columns into numpy arrays."""
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    tables = [pq.read_table(f) for f in parquet_files]
    import pyarrow as pa

    table = pa.concat_tables(tables)

    required = ["index", "episode_index", "frame_index", "predicted_value", "value_target"]
    for col in required:
        if col not in table.column_names:
            raise ValueError(f"Missing required column '{col}' in parquet. Run lerobot_value_infer.py first.")

    result = {}
    for col in ["index", "episode_index", "frame_index", "predicted_value",
                 "value_target", "advantage", "is_positive", "task_index"]:
        if col in table.column_names:
            result[col] = table[col].to_numpy()

    if "success" in table.column_names:
        result["success"] = np.array(table["success"].to_pylist(), dtype=bool)

    sort_order = np.argsort(result["index"])
    for k in result:
        result[k] = result[k][sort_order]

    return result


def load_episode_meta(dataset_root: Path) -> dict[int, dict]:
    """Load episode metadata from episodes.jsonl."""
    ep_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = {}
    if ep_path.exists():
        with open(ep_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    episodes[int(entry["episode_index"])] = entry
    return episodes


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    ss_res = float(np.sum((pred - target) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if len(pred) > 2:
        r_val, p_val = stats.pearsonr(pred, target)
    else:
        r_val, p_val = float("nan"), float("nan")
    return {"mse": mse, "mae": mae, "r2": r2, "pearson_r": float(r_val), "pearson_p": float(p_val), "n": len(pred)}


def print_metrics(name: str, m: dict) -> None:
    logger.info(
        "  %-25s | n=%6d | MSE=%.6f | MAE=%.6f | R²=%.4f | r=%.4f (p=%.2e)",
        name, m["n"], m["mse"], m["mae"], m["r2"], m["pearson_r"], m["pearson_p"],
    )


def evaluate(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", dataset_root)
    data = load_dataset(dataset_root)
    ep_meta = load_episode_meta(dataset_root)

    pred = data["predicted_value"].astype(np.float64)
    target = data["value_target"].astype(np.float64)
    ep_ids = data["episode_index"]
    frame_ids = data["frame_index"]
    has_success = "success" in data
    has_advantage = "advantage" in data
    has_positive = "is_positive" in data

    # ---- 0. Print last 10 frames per episode ----
    logger.info("=" * 80)
    logger.info("0. LAST 10 FRAMES PER EPISODE")
    logger.info("=" * 80)
    for ep in np.unique(ep_ids):
        mask = ep_ids == ep
        idx = np.argsort(data["frame_index"][mask])
        t_ep = target[mask][idx]
        p_ep = pred[mask][idx]
        a_ep = data["advantage"][mask][idx] if has_advantage else None
        logger.info("  Episode %d (last 10 frames):", ep)
        logger.info("    value_target:     %s", np.array2string(t_ep[-10:], precision=6, separator=", "))
        logger.info("    predicted_value:  %s", np.array2string(p_ep[-10:], precision=6, separator=", "))
        if a_ep is not None:
            logger.info("    advantage:        %s", np.array2string(a_ep[-10:], precision=6, separator=", "))

    # ---- 1. Global regression metrics ----
    logger.info("=" * 80)
    logger.info("1. GLOBAL REGRESSION METRICS")
    logger.info("=" * 80)
    global_m = regression_metrics(pred, target)
    print_metrics("All frames", global_m)

    if has_success:
        success = data["success"]
        succ_m = regression_metrics(pred[success], target[success])
        print_metrics("Success episodes", succ_m)
        if (~success).sum() > 0:
            fail_m = regression_metrics(pred[~success], target[~success])
            print_metrics("Failure episodes", fail_m)
        else:
            logger.info("  %-25s | (no failure frames in dataset)", "Failure episodes")

    # ---- 2. Per-episode metrics ----
    logger.info("=" * 80)
    logger.info("2. PER-EPISODE REGRESSION METRICS")
    logger.info("=" * 80)
    unique_eps = np.unique(ep_ids)
    ep_metrics = {}
    for ep in unique_eps:
        mask = ep_ids == ep
        m = regression_metrics(pred[mask], target[mask])
        ep_success = bool(data["success"][mask][0]) if has_success else None
        m["success"] = ep_success
        ep_metrics[int(ep)] = m
        label = f"Episode {ep}" + (f" ({'succ' if ep_success else 'FAIL'})" if ep_success is not None else "")
        print_metrics(label, m)

    # ---- 3. Error distribution by episode position ----
    logger.info("=" * 80)
    logger.info("3. PREDICTION ERROR BY EPISODE POSITION (quartiles)")
    logger.info("=" * 80)
    errors = pred - target
    abs_errors = np.abs(errors)

    rel_positions = np.zeros_like(frame_ids, dtype=np.float64)
    for ep in unique_eps:
        mask = ep_ids == ep
        max_frame = frame_ids[mask].max()
        if max_frame > 0:
            rel_positions[mask] = frame_ids[mask].astype(np.float64) / max_frame

    quartile_names = ["Q1 (0-25%: start)", "Q2 (25-50%: early-mid)", "Q3 (50-75%: late-mid)", "Q4 (75-100%: end)"]
    for q, name in enumerate(quartile_names):
        lo, hi = q * 0.25, (q + 1) * 0.25
        mask = (rel_positions >= lo) & (rel_positions < hi) if q < 3 else (rel_positions >= lo)
        if mask.sum() == 0:
            continue
        q_errors = errors[mask]
        q_abs = abs_errors[mask]
        logger.info(
            "  %-30s | n=%5d | mean_err=%+.6f | MAE=%.6f | std=%.6f",
            name, mask.sum(), q_errors.mean(), q_abs.mean(), q_errors.std(),
        )

    # ---- 4. Success vs Failure distribution analysis ----
    if has_success:
        logger.info("=" * 80)
        logger.info("4. SUCCESS vs FAILURE DISTRIBUTION ANALYSIS")
        logger.info("=" * 80)
        for label, mask in [("Success", success), ("Failure", ~success)]:
            if mask.sum() == 0:
                logger.info("  %-10s | (no frames)", label)
                continue
            p, t_ = pred[mask], target[mask]
            logger.info(
                "  %-10s | n=%5d | pred: mean=%.4f std=%.4f | target: mean=%.4f std=%.4f | bias=%+.4f",
                label, mask.sum(), p.mean(), p.std(), t_.mean(), t_.std(), (p - t_).mean(),
            )

        if has_advantage:
            adv = data["advantage"].astype(np.float64)
            logger.info("  Advantage distribution:")
            for label, mask in [("Success", success), ("Failure", ~success)]:
                if mask.sum() == 0:
                    continue
                a = adv[mask]
                logger.info(
                    "    %-10s | mean=%+.6f | std=%.6f | min=%.6f | max=%.6f | %%>0=%.1f%%",
                    label, a.mean(), a.std(), a.min(), a.max(), 100.0 * (a > 0).mean(),
                )

    # ---- 5. Advantage & is_positive consistency ----
    if has_advantage and has_positive:
        logger.info("=" * 80)
        logger.info("5. ADVANTAGE & IS_POSITIVE CONSISTENCY")
        logger.info("=" * 80)
        adv = data["advantage"].astype(np.float64)
        pos = data["is_positive"]

        logger.info("  Total frames: %d", len(adv))
        logger.info("  is_positive=1: %d (%.1f%%)", pos.sum(), 100.0 * pos.mean())
        logger.info("  is_positive=0: %d (%.1f%%)", (1 - pos).sum(), 100.0 * (1 - pos).mean())
        logger.info("  Advantage of positive frames:  mean=%+.6f  std=%.6f", adv[pos == 1].mean(), adv[pos == 1].std())
        logger.info("  Advantage of negative frames:  mean=%+.6f  std=%.6f", adv[pos == 0].mean(), adv[pos == 0].std())

        if has_success:
            logger.info("  Positive frames from success eps: %.1f%%", 100.0 * success[pos == 1].mean())
            logger.info("  Positive frames from failure eps: %.1f%%", 100.0 * (~success[pos == 1]).mean())
            logger.info("  Negative frames from success eps: %.1f%%", 100.0 * success[pos == 0].mean())
            logger.info("  Negative frames from failure eps: %.1f%%", 100.0 * (~success[pos == 0]).mean())

    # ---- 6. Plots ----
    logger.info("=" * 80)
    logger.info("6. GENERATING PLOTS -> %s", output_dir)
    logger.info("=" * 80)

    # 6a. Scatter: predicted vs target
    fig, ax = plt.subplots(figsize=(8, 8))
    if has_success:
        ax.scatter(target[success], pred[success], alpha=0.3, s=4, c="tab:blue", label="Success")
        ax.scatter(target[~success], pred[~success], alpha=0.3, s=4, c="tab:red", label="Failure")
        ax.legend(fontsize=10)
    else:
        ax.scatter(target, pred, alpha=0.3, s=4, c="tab:blue")
    lo = min(target.min(), pred.min()) - 0.05
    hi = max(target.max(), pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
    ax.set_xlabel("Value Target (ground truth)", fontsize=12)
    ax.set_ylabel("Predicted Value", fontsize=12)
    ax.set_title(f"Predicted vs Target  |  R²={global_m['r2']:.4f}  r={global_m['pearson_r']:.4f}", fontsize=13)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_pred_vs_target.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved scatter_pred_vs_target.png")

    # 6b. Error histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(errors, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].axvline(0, color="k", ls="--", lw=1)
    axes[0].set_xlabel("Prediction Error (pred - target)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Error Distribution  |  mean={errors.mean():+.4f}  std={errors.std():.4f}")

    axes[1].hist(abs_errors, bins=100, color="coral", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("Absolute Error")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Abs Error  |  MAE={global_m['mae']:.4f}  MSE={global_m['mse']:.6f}")
    fig.tight_layout()
    fig.savefig(output_dir / "error_histogram.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved error_histogram.png")

    # 6c. Error by episode position
    fig, ax = plt.subplots(figsize=(10, 5))
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_mae, bin_bias = [], []
    for i in range(n_bins):
        mask = (rel_positions >= bin_edges[i]) & (rel_positions < bin_edges[i + 1])
        if i == n_bins - 1:
            mask |= rel_positions == bin_edges[i + 1]
        if mask.sum() > 0:
            bin_mae.append(abs_errors[mask].mean())
            bin_bias.append(errors[mask].mean())
        else:
            bin_mae.append(float("nan"))
            bin_bias.append(float("nan"))
    ax.bar(bin_centers, bin_mae, width=0.045, color="steelblue", alpha=0.7, label="MAE")
    ax.plot(bin_centers, bin_bias, "r-o", ms=4, label="Mean bias")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Relative Episode Position (0=start, 1=end)")
    ax.set_ylabel("Error")
    ax.set_title("Prediction Error by Episode Position")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "error_by_position.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved error_by_position.png")

    # 6d. Advantage distribution
    if has_advantage:
        fig, ax = plt.subplots(figsize=(10, 5))
        adv = data["advantage"].astype(np.float64)
        if has_success:
            ax.hist(adv[success], bins=80, alpha=0.6, color="tab:blue", label="Success", density=True)
            ax.hist(adv[~success], bins=80, alpha=0.6, color="tab:red", label="Failure", density=True)
            ax.legend()
        else:
            ax.hist(adv, bins=80, alpha=0.7, color="steelblue", density=True)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("Advantage")
        ax.set_ylabel("Density")
        ax.set_title("Advantage Distribution (Success vs Failure)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "advantage_distribution.png", dpi=150)
        plt.close(fig)
        logger.info("  Saved advantage_distribution.png")

    # 6e. Per-episode curves
    n_plot = min(args.plot_episodes, len(unique_eps))
    if n_plot > 0:
        plot_eps = unique_eps[:n_plot]
        fig, axes = plt.subplots(n_plot, 1, figsize=(14, 4 * n_plot), squeeze=False)
        for i, ep in enumerate(plot_eps):
            ax = axes[i, 0]
            mask = ep_ids == ep
            fi = frame_ids[mask]
            sort_idx = np.argsort(fi)
            fi = fi[sort_idx]

            ax.plot(fi, target[mask][sort_idx], "b-", lw=1.5, label="value_target (GT)", alpha=0.8)
            ax.plot(fi, pred[mask][sort_idx], "r--", lw=1.5, label="predicted_value", alpha=0.8)

            if has_advantage:
                adv_ep = data["advantage"][mask][sort_idx].astype(np.float64)
                ax2 = ax.twinx()
                ax2.fill_between(fi, adv_ep, 0, alpha=0.15, color="green", label="advantage")
                ax2.set_ylabel("Advantage", color="green", fontsize=10)
                ax2.tick_params(axis="y", labelcolor="green")

            ep_info_str = ""
            if has_success:
                ep_info_str += f"  success={data['success'][mask][0]}"
            m = ep_metrics.get(int(ep), {})
            ep_info_str += f"  R²={m.get('r2', float('nan')):.3f}  r={m.get('pearson_r', float('nan')):.3f}"

            ax.set_title(f"Episode {ep}{ep_info_str}", fontsize=11)
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Value")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "episode_curves.png", dpi=150)
        plt.close(fig)
        logger.info("  Saved episode_curves.png (%d episodes)", n_plot)

    # ---- 7. Save summary JSON ----
    summary = {
        "dataset_root": str(dataset_root),
        "total_frames": len(pred),
        "total_episodes": len(unique_eps),
        "global_metrics": global_m,
        "per_episode_metrics": {str(k): v for k, v in ep_metrics.items()},
    }
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("  Saved eval_summary.json")

    logger.info("=" * 80)
    logger.info("DONE. All outputs saved to %s", output_dir)
    logger.info("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate value model prediction accuracy")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to LeRobot dataset with value annotations")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots and metrics (default: <dataset-root>/value_eval)")
    parser.add_argument("--plot-episodes", type=int, default=5, help="Number of episodes to plot curves for")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.dataset_root).resolve() / "value_eval")
    evaluate(args)
