"""
Episode-wise pairwise ranking accuracy for Pi0Value.

Changes vs the old script:
1) No train/val split at all: evaluate on all frames.
2) Run inference over the whole dataset first, preserving dataset order.
   If the dataset already has ``predicted_value`` or ``value_predict``, skip model load
   and inference and use that column (same row order as the HF dataset).
3) Compute pairwise ranking accuracy *within each episode* instead of within each batch.
4) Report both:
   - micro accuracy: sum(correct pairs) / sum(total pairs)
   - macro accuracy: mean(per-episode accuracy)

Ground-truth targets:
- target_source=auto: use dataset column 'value_target' if present
- otherwise compute from episode_index/frame_index + meta, matching training logic

Optional: save per-episode pred vs target curves (PNG) with ``--plot-dir ./plots``.

Usage:
python scripts/evaluate_value_pairwise_ranking.py \\
    --config-name pi0_value_umi \\
    --checkpoint-dir ./checkpoints/pi0_value_umi/exp/30000 \\
    --dataset-root ./data/my_lerobot_dataset \\
    --batch-size 32 \\
    --plot-dir ./value_eval_plots

If the parquet already contains ``predicted_value`` (or ``value_predict``), inference
is skipped; you may omit ``--checkpoint-dir`` unless you use ``--max-batches``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

if "TMPDIR" not in os.environ:
    _tmp = Path(os.environ.get("HOME", "/root")) / "tmp"
    _tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_tmp)

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import tqdm_loggable.auto as tqdm

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.value_targets as value_targets

# jax/openpi/tqdm_loggable may configure logging first; plain basicConfig() then does nothing.
# force=True reapplies our handler so INFO logs reach stderr.
_bc_kw = dict(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
if sys.version_info >= (3, 8):
    _bc_kw["force"] = True
logging.basicConfig(**_bc_kw)
logger = logging.getLogger(__name__)


def _dataset_precomputed_pred_column(hf_cols: list[str]) -> str | None:
    """Column name to use for precomputed value predictions, or None to run inference."""
    for name in ("predicted_value", "value_predict"):
        if name in hf_cols:
            return name
    return None


def parse_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise ValueError(f"Cannot parse boolean value from: {x}")


def build_episode_info(
    dataset: lerobot_dataset.LeRobotDataset,
    success_field: str,
    default_success: str,
) -> tuple[dict[int, value_targets.EpisodeTargetInfo], dict[int, int]]:
    """Build episode_info and task_max_lengths from dataset metadata."""
    episodes_jsonl_path = Path(dataset.root) / "meta" / "episodes.jsonl"
    if not episodes_jsonl_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found at {episodes_jsonl_path}")

    episodes: list[dict] = []
    with open(episodes_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    tasks_jsonl_path = Path(dataset.root) / "meta" / "tasks.jsonl"
    task_name_to_index: dict[str, int] = {}
    if tasks_jsonl_path.exists():
        with open(tasks_jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    task_name_to_index[entry["task"]] = int(entry["task_index"])

    episode_info: dict[int, value_targets.EpisodeTargetInfo] = {}
    task_max_length: dict[int, int] = {}

    default_success_bool = parse_bool(default_success)

    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        ep_length = int(ep["length"])

        tasks = ep.get("tasks", [])
        task_name = tasks[0] if isinstance(tasks, list) and len(tasks) > 0 else "unknown"
        task_index = task_name_to_index.get(task_name, 0)

        explicit_success = ep.get(success_field, None)
        if explicit_success is not None:
            ep_success = parse_bool(explicit_success)
        else:
            ep_success = default_success_bool

        episode_info[ep_idx] = value_targets.EpisodeTargetInfo(
            task_index=task_index,
            length=ep_length,
            success=ep_success,
        )
        task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)

    return episode_info, task_max_length


def _replace_config_for_inference(
    config: _config.TrainConfig,
    dataset_root: Path,
    batch_size: int,
    checkpoint_dir: str | None = None,
) -> _config.TrainConfig:
    """Replace dataset root and batch size; load norm stats from checkpoint if given."""
    data_factory = config.data
    if hasattr(data_factory, "repo_id"):
        data_factory = dataclasses.replace(data_factory, repo_id=str(dataset_root))

    if checkpoint_dir is not None and hasattr(data_factory, "assets"):
        ckpt_assets_dir = str(Path(checkpoint_dir) / "assets")
        data_factory = dataclasses.replace(
            data_factory,
            assets=dataclasses.replace(data_factory.assets, assets_dir=ckpt_assets_dir),
        )

    return dataclasses.replace(
        config,
        data=data_factory,
        batch_size=batch_size,
        num_workers=4,
    )


def pairwise_ranking_stats(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    eps: float,
) -> tuple[int, int]:
    """
    Return (num_correct, num_discriminative_pairs) for one episode.

    Pair (i, j) is counted only if |target_i - target_j| > eps.
    Correct iff sign(pred_i - pred_j) == sign(target_i - target_j).
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    b = pred.shape[0]
    if b < 2:
        return 0, 0

    i, j = np.triu_indices(b, k=1)
    dt = target[i] - target[j]
    dp = pred[i] - pred[j]

    mask = np.abs(dt) > eps
    if not np.any(mask):
        return 0, 0

    dt = dt[mask]
    dp = dp[mask]

    correct = np.sign(dt) == np.sign(dp)
    return int(np.sum(correct)), int(mask.sum())


DEFAULT_TOLERANCES = (0.01, 0.02, 0.05, 0.1, 0.2)


def _frame_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    tolerances: tuple[float, ...] = DEFAULT_TOLERANCES,
) -> dict:
    """Per-frame hit rate: fraction of frames with |pred - target| <= tol."""
    abs_err = np.abs(pred.astype(np.float64) - target.astype(np.float64))
    n = abs_err.shape[0]
    if n == 0:
        return {f"frame_acc@{t}": float("nan") for t in tolerances}
    return {f"frame_acc@{t}": float(np.mean(abs_err <= t)) for t in tolerances}


def _regression_stats(pred: np.ndarray, target: np.ndarray) -> dict:
    """MAE, RMSE, Pearson r, Spearman rho for one sequence."""
    from scipy.stats import spearmanr

    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    n = pred.shape[0]
    nan_row = {"mae": float("nan"), "rmse": float("nan"), "pearson_r": float("nan"), "spearman_rho": float("nan")}
    if n == 0:
        return nan_row

    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    if n < 2 or np.std(pred) < 1e-12 or np.std(target) < 1e-12:
        return {**nan_row, "mae": mae, "rmse": rmse}

    pearson_r = float(np.corrcoef(pred, target)[0, 1])
    spearman_rho = float(spearmanr(pred, target).correlation)
    return {"mae": mae, "rmse": rmse, "pearson_r": pearson_r, "spearman_rho": spearman_rho}


def compute_episodewise_pairwise_metrics(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    pred_values: np.ndarray,
    target_values: np.ndarray,
    *,
    eps: float,
    worst_k: int = 10,
) -> dict:
    """
    Compute episode-wise pairwise ranking + regression accuracy metrics.

    Returns dict with:
      Pairwise ranking:  micro_acc, macro_acc, total_correct, total_pairs
      Regression global:  global_mae, global_rmse, global_pearson, global_spearman
      Regression macro:   macro_mae, macro_pearson, macro_spearman
      Per-episode detail: per_episode_rows, worst_rows
    """
    episode_indices = np.asarray(episode_indices).reshape(-1)
    frame_indices = np.asarray(frame_indices).reshape(-1)
    pred_values = np.asarray(pred_values).reshape(-1)
    target_values = np.asarray(target_values).reshape(-1)

    if not (
        len(episode_indices)
        == len(frame_indices)
        == len(pred_values)
        == len(target_values)
    ):
        raise ValueError("Input arrays must have the same length.")

    unique_eps = np.unique(episode_indices)

    total_correct = 0
    total_pairs = 0
    per_episode_accs = []
    per_episode_maes = []
    per_episode_pearsons = []
    per_episode_spearmans = []
    per_episode_rows = []

    for ep in unique_eps:
        idx = np.where(episode_indices == ep)[0]
        order = np.argsort(frame_indices[idx], kind="stable")
        idx = idx[order]

        pred_ep = pred_values[idx]
        tgt_ep = target_values[idx]
        reg = _regression_stats(pred_ep, tgt_ep)
        fa = _frame_accuracy(pred_ep, tgt_ep)

        if idx.size < 2:
            per_episode_rows.append({
                "episode_index": int(ep), "num_frames": int(idx.size),
                "num_pairs": 0, "num_correct": 0, "pairwise_acc": float("nan"),
                **reg, **fa,
            })
            if not np.isnan(reg["mae"]):
                per_episode_maes.append(reg["mae"])
            continue

        c, n = pairwise_ranking_stats(pred_ep, tgt_ep, eps=eps)
        acc = (c / n) if n > 0 else float("nan")

        if n > 0:
            total_correct += c
            total_pairs += n
            per_episode_accs.append(acc)
        if not np.isnan(reg["mae"]):
            per_episode_maes.append(reg["mae"])
        if not np.isnan(reg["pearson_r"]):
            per_episode_pearsons.append(reg["pearson_r"])
        if not np.isnan(reg["spearman_rho"]):
            per_episode_spearmans.append(reg["spearman_rho"])

        per_episode_rows.append({
            "episode_index": int(ep), "num_frames": int(idx.size),
            "num_pairs": int(n), "num_correct": int(c), "pairwise_acc": float(acc),
            **reg, **fa,
        })

    micro_acc = (total_correct / total_pairs) if total_pairs > 0 else float("nan")
    macro_acc = float(np.mean(per_episode_accs)) if per_episode_accs else float("nan")

    global_reg = _regression_stats(pred_values, target_values)
    global_frame_acc = _frame_accuracy(pred_values, target_values)
    macro_mae = float(np.mean(per_episode_maes)) if per_episode_maes else float("nan")
    macro_pearson = float(np.mean(per_episode_pearsons)) if per_episode_pearsons else float("nan")
    macro_spearman = float(np.mean(per_episode_spearmans)) if per_episode_spearmans else float("nan")

    sortable_rows = []
    for row in per_episode_rows:
        acc = row["pairwise_acc"]
        sort_key = 999.0 if np.isnan(acc) else acc
        sortable_rows.append((sort_key, row))
    sortable_rows.sort(key=lambda x: x[0])
    worst_rows = [row for _, row in sortable_rows[:worst_k]]

    return {
        "micro_acc": micro_acc,
        "macro_acc": macro_acc,
        "total_correct": total_correct,
        "total_pairs": total_pairs,
        "num_episodes": int(len(unique_eps)),
        "num_scored_episodes": int(sum(row["num_pairs"] > 0 for row in per_episode_rows)),
        "global_mae": global_reg["mae"],
        "global_rmse": global_reg["rmse"],
        "global_pearson": global_reg["pearson_r"],
        "global_spearman": global_reg["spearman_rho"],
        "macro_mae": macro_mae,
        "macro_pearson": macro_pearson,
        "macro_spearman": macro_spearman,
        **global_frame_acc,
        "per_episode_rows": per_episode_rows,
        "worst_rows": worst_rows,
    }


def save_episode_pred_vs_target_curves(
    pred_values: np.ndarray,
    target_values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    plot_dir: Path,
    *,
    max_episodes: int | None = None,
    dpi: int = 150,
) -> int:
    """Save one PNG per episode: value_pred vs value_target vs frame_index (sorted by frame)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_values = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target_values, dtype=np.float64).reshape(-1)
    episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    unique_eps = np.unique(episode_indices)
    unique_eps = np.sort(unique_eps)
    if max_episodes is not None and max_episodes > 0:
        unique_eps = unique_eps[: int(max_episodes)]

    n_saved = 0
    for ep in unique_eps:
        mask = episode_indices == ep
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        order = np.argsort(frame_indices[idx], kind="stable")
        idx = idx[order]
        x = frame_indices[idx].astype(np.float64)
        pred_ep = pred_values[idx]
        tgt_ep = target_values[idx]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, tgt_ep, "b-", lw=1.8, label="value_target (GT)", alpha=0.9)
        ax.plot(x, pred_ep, "r--", lw=1.8, label="value_pred", alpha=0.9)
        ax.set_xlabel("frame_index")
        ax.set_ylabel("normalized value")
        ax.set_title(f"episode_index={int(ep)}  (n_frames={len(x)})")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.35)
        lo = float(min(np.min(pred_ep), np.min(tgt_ep)))
        hi = float(max(np.max(pred_ep), np.max(tgt_ep)))
        pad = 0.05 * (hi - lo + 1e-8)
        ax.set_ylim(lo - pad, hi + pad)

        out_path = plot_dir / f"episode_{int(ep)}_pred_vs_target.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        n_saved += 1

    return n_saved


def run_eval(args: argparse.Namespace) -> None:
    config = _config.get_config(args.config_name)
    dataset_root = Path(args.dataset_root).resolve()

    logger.info("Config: %s", args.config_name)
    logger.info("Checkpoint: %s", args.checkpoint_dir or "(none — precomputed predictions only)")
    logger.info("Dataset root: %s", dataset_root)

    num_devices = jax.device_count()

    # ---- Dataset metadata ----
    raw_dataset = lerobot_dataset.LeRobotDataset(
        repo_id=str(dataset_root),
        root=str(dataset_root),
    )
    raw_frames = raw_dataset.hf_dataset.with_format(None)
    frame_count = len(raw_frames)
    if frame_count == 0:
        raise ValueError("Dataset has no frames.")

    episode_indices_full = np.asarray(raw_frames["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices_full = np.asarray(raw_frames["frame_index"], dtype=np.int64).reshape(-1)
    logger.info("Dataset: %d frames, %d episodes", frame_count, len(np.unique(episode_indices_full)))

    episode_info, task_max_lengths = build_episode_info(
        raw_dataset,
        success_field=args.success_field,
        default_success=args.default_success,
    )
    logger.info("Episode info: %d episodes, %d tasks", len(episode_info), len(task_max_lengths))

    hf_cols = list(raw_frames.column_names)
    use_dataset_targets = args.target_source == "dataset" or (
        args.target_source == "auto" and "value_target" in hf_cols
    )

    if args.target_source == "dataset" and "value_target" not in hf_cols:
        raise ValueError(
            "target_source=dataset but dataset has no 'value_target' column. "
            "Use target_source=computed or add the column."
        )

    if use_dataset_targets:
        raw_vt = raw_frames["value_target"]
        value_targets_full = np.asarray(raw_vt, dtype=np.float32).reshape(frame_count, -1)
        if value_targets_full.shape[1] > 1:
            value_targets_full = value_targets_full.mean(axis=1)
        else:
            value_targets_full = value_targets_full.reshape(-1)

        logger.info(
            "Using dataset value_target for all frames (%d frames); --c-fail-coef is ignored.",
            frame_count,
        )
    else:
        value_targets_full = value_targets.compute_normalized_value_targets(
            episode_indices=episode_indices_full,
            frame_indices=frame_indices_full,
            episode_info=episode_info,
            task_max_lengths=task_max_lengths,
            c_fail_coef=args.c_fail_coef,
            clip_min=config.value_clip_min,
            clip_max=config.value_clip_max,
        )
        logger.info(
            "Using computed normalized targets for all frames "
            "(c_fail_coef=%s, clip=[%s, %s]).",
            args.c_fail_coef,
            config.value_clip_min,
            config.value_clip_max,
        )

    precomputed_pred_col = _dataset_precomputed_pred_column(hf_cols)
    # If targets are forced to be recomputed from meta, also force model inference so
    # predicted_value is recomputed (ignore any stored prediction columns).
    if args.target_source == "computed":
        precomputed_pred_col = None

    if precomputed_pred_col is None and not args.checkpoint_dir:
        raise ValueError(
            "Missing --checkpoint-dir (required for inference). "
            "Or add predicted_value / value_predict to the dataset to skip inference."
        )
    if args.max_batches is not None and not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required when --max-batches is set (data loader prefix alignment).")

    # ---- Batch size (inference or max_batches prefix alignment) ----
    effective_bs = args.batch_size
    if effective_bs % num_devices != 0:
        effective_bs = ((effective_bs + num_devices - 1) // num_devices) * num_devices
        logger.info(
            "Adjusted batch_size %d -> %d (divisible by %d devices)",
            args.batch_size,
            effective_bs,
            num_devices,
        )

    batch_count = (frame_count + effective_bs - 1) // effective_bs
    if args.max_batches is not None:
        batch_count = min(batch_count, args.max_batches)
        logger.warning(
            "max_batches=%d is enabled. This may truncate episodes and make episode-wise metrics less reliable.",
            args.max_batches,
        )

    if precomputed_pred_col is not None:
        logger.info(
            "Using dataset column %r; skipping model load and GPU inference.",
            precomputed_pred_col,
        )
        raw_pv = raw_frames[precomputed_pred_col]
        pred_values_stored = np.asarray(raw_pv, dtype=np.float32).reshape(frame_count, -1)
        if pred_values_stored.shape[1] > 1:
            pred_values_stored = pred_values_stored.mean(axis=1)
        else:
            pred_values_stored = pred_values_stored.reshape(-1)

        if args.max_batches is not None:
            mesh = jax.sharding.Mesh(jax.devices(), ("batch",))
            data_parallel = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
            infer_config = _replace_config_for_inference(
                config,
                dataset_root,
                effective_bs,
                checkpoint_dir=args.checkpoint_dir,
            )
            data_loader = _data_loader.create_data_loader(
                infer_config,
                sharding=data_parallel,
                shuffle=False,
            )
            sample_cursor = 0
            batches_done = 0
            for obs, _actions in data_loader:
                if batches_done >= batch_count or sample_cursor >= frame_count:
                    break
                n_take = min(int(obs.state.shape[0]), frame_count - sample_cursor)
                if n_take <= 0:
                    break
                sample_cursor += n_take
                batches_done += 1
            pred_values_full = pred_values_stored[:sample_cursor].copy()
        else:
            sample_cursor = frame_count
            pred_values_full = pred_values_stored

        if pred_values_full.shape[0] != sample_cursor:
            raise RuntimeError(
                f"Prediction length mismatch: got {pred_values_full.shape[0]} vs cursor {sample_cursor}"
            )
    else:
        mesh = jax.sharding.Mesh(jax.devices(), ("batch",))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_parallel = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
        logger.info("Inference mesh: %d device(s) %s", num_devices, jax.devices())

        # ---- Model ----
        logger.info("Creating model from config...")
        model = config.model.create(jax.random.key(0))

        params_path = Path(args.checkpoint_dir) / "params"
        if not params_path.exists():
            raise FileNotFoundError(f"Missing params directory: {params_path}")

        logger.info("Restoring params from %s", params_path)
        params = _model.restore_params(params_path, dtype=jnp.bfloat16, sharding=replicated)

        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(params)
        model = nnx.merge(graphdef, state)
        model.eval()
        logger.info("Model loaded.")

        infer_config = _replace_config_for_inference(
            config, dataset_root, effective_bs, checkpoint_dir=args.checkpoint_dir
        )
        data_loader = _data_loader.create_data_loader(
            infer_config,
            sharding=data_parallel,
            shuffle=False,
        )

        def _infer_value(obs):
            logits = model.forward_value_logits(obs)
            return model.expected_value_from_logits(logits)

        infer_value = jax.jit(_infer_value, out_shardings=replicated)

        preds: list[np.ndarray] = []
        sample_cursor = 0
        batches_done = 0

        logger.info(
            "Running full inference first (%d batches, batch_size=%d)...",
            batch_count,
            effective_bs,
        )
        pbar = tqdm.tqdm(total=batch_count, desc="Inference")

        for obs, _actions in data_loader:
            if batches_done >= batch_count or sample_cursor >= frame_count:
                break

            values = infer_value(obs)
            val_np = np.asarray(values).astype(np.float32).reshape(-1)

            n_take = min(len(val_np), int(obs.state.shape[0]), frame_count - sample_cursor)
            if n_take <= 0:
                break

            preds.append(val_np[:n_take])
            sample_cursor += n_take
            batches_done += 1
            pbar.update(1)

        pbar.close()

        if len(preds) == 0:
            raise RuntimeError("No predictions were produced.")

        pred_values_full = np.concatenate(preds, axis=0)
        if pred_values_full.shape[0] != sample_cursor:
            raise RuntimeError(
                f"Prediction length mismatch: got {pred_values_full.shape[0]} vs cursor {sample_cursor}"
            )

    # If max_batches is used, only evaluate the prefix we actually inferred.
    episode_indices_eval = episode_indices_full[:sample_cursor]
    frame_indices_eval = frame_indices_full[:sample_cursor]
    target_values_eval = value_targets_full[:sample_cursor]
    pred_values_eval = pred_values_full[:sample_cursor]

    logger.info("Collected predictions for %d frames.", sample_cursor)

    metrics = compute_episodewise_pairwise_metrics(
        episode_indices=episode_indices_eval,
        frame_indices=frame_indices_eval,
        pred_values=pred_values_eval,
        target_values=target_values_eval,
        eps=args.target_eps,
        worst_k=args.worst_k,
    )

    logger.info("=" * 70)
    logger.info("Pairwise ranking accuracy")
    logger.info(
        "  micro_acc=%.6f  macro_acc=%.6f  correct=%d  pairs=%d  scored_eps=%d/%d",
        metrics["micro_acc"],
        metrics["macro_acc"],
        metrics["total_correct"],
        metrics["total_pairs"],
        metrics["num_scored_episodes"],
        metrics["num_episodes"],
    )
    logger.info("Regression accuracy (global)")
    logger.info(
        "  MAE=%.6f  RMSE=%.6f  Pearson=%.6f  Spearman=%.6f",
        metrics["global_mae"],
        metrics["global_rmse"],
        metrics["global_pearson"],
        metrics["global_spearman"],
    )
    logger.info("Regression accuracy (macro = mean of per-episode)")
    logger.info(
        "  MAE=%.6f  Pearson=%.6f  Spearman=%.6f",
        metrics["macro_mae"],
        metrics["macro_pearson"],
        metrics["macro_spearman"],
    )
    logger.info("Per-frame accuracy (|pred - target| <= tolerance)")
    tol_parts = []
    for t in DEFAULT_TOLERANCES:
        key = f"frame_acc@{t}"
        tol_parts.append(f"@{t}={metrics[key]:.4f}")
    logger.info("  %s", "  ".join(tol_parts))
    logger.info("=" * 70)

    if metrics["worst_rows"]:
        logger.info("Worst %d episodes by pairwise accuracy:", len(metrics["worst_rows"]))
        for row in metrics["worst_rows"]:
            logger.info(
                "  ep=%d  frames=%d  pairs=%d  correct=%d  pw_acc=%s  mae=%.6f  pearson=%s",
                row["episode_index"],
                row["num_frames"],
                row["num_pairs"],
                row["num_correct"],
                "nan" if np.isnan(row["pairwise_acc"]) else f"{row['pairwise_acc']:.6f}",
                row["mae"],
                "nan" if np.isnan(row["pearson_r"]) else f"{row['pearson_r']:.6f}",
            )

    if args.output_json:
        out = {
            "config_name": args.config_name,
            "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
            "pred_source": ("dataset:" + precomputed_pred_col)
            if precomputed_pred_col is not None
            else "inference",
            "dataset_root": str(dataset_root),
            "target_source": args.target_source,
            "c_fail_coef": args.c_fail_coef,
            "target_eps": args.target_eps,
            "frames_evaluated": int(sample_cursor),
            "micro_acc": float(metrics["micro_acc"]),
            "macro_acc": float(metrics["macro_acc"]),
            "total_correct": int(metrics["total_correct"]),
            "total_pairs": int(metrics["total_pairs"]),
            "num_episodes": int(metrics["num_episodes"]),
            "num_scored_episodes": int(metrics["num_scored_episodes"]),
            "global_mae": float(metrics["global_mae"]),
            "global_rmse": float(metrics["global_rmse"]),
            "global_pearson": float(metrics["global_pearson"]),
            "global_spearman": float(metrics["global_spearman"]),
            "macro_mae": float(metrics["macro_mae"]),
            "macro_pearson": float(metrics["macro_pearson"]),
            "macro_spearman": float(metrics["macro_spearman"]),
            **{f"frame_acc@{t}": float(metrics[f"frame_acc@{t}"]) for t in DEFAULT_TOLERANCES},
            "worst_rows": metrics["worst_rows"],
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.plot_dir:
            out["plot_dir"] = str(Path(args.plot_dir).resolve())
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Saved summary JSON to %s", out_path)

    if args.plot_dir:
        plot_path = Path(args.plot_dir).resolve()
        n_plots = save_episode_pred_vs_target_curves(
            pred_values_eval,
            target_values_eval,
            episode_indices_eval,
            frame_indices_eval,
            plot_path,
            max_episodes=args.plot_max_episodes,
            dpi=args.plot_dpi,
        )
        logger.info("Saved %d per-episode pred vs target curves to %s", n_plots, plot_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Episode-wise pairwise ranking accuracy for Pi0Value on all frames."
    )
    p.add_argument("--config-name", type=str, required=True, help="Training config name")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint step dir with params/. Required for inference. "
        "Optional if the dataset already has predicted_value or value_predict and --max-batches is not set.",
    )
    p.add_argument("--dataset-root", type=str, required=True, help="LeRobot dataset root")
    p.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional quick test only. Will truncate dataset prefix and may cut episodes.",
    )
    p.add_argument(
        "--target-source",
        type=str,
        choices=("auto", "dataset", "computed"),
        default="auto",
        help="auto = use dataset value_target if present else compute from meta; "
        "dataset = require dataset value_target; computed = always recompute",
    )
    p.add_argument(
        "--c-fail-coef",
        type=float,
        default=1.0,
        help="Used only when targets are computed from meta. Must match training.",
    )
    p.add_argument("--success-field", type=str, default="success", help="Episode success field in episodes.jsonl")
    p.add_argument("--default-success", type=str, default="true", help="Default success if field missing")
    p.add_argument(
        "--target-eps",
        type=float,
        default=1e-6,
        help="Skip pairs with |target_i - target_j| <= eps",
    )
    p.add_argument(
        "--worst-k",
        type=int,
        default=10,
        help="How many worst episodes to print",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save summary JSON",
    )
    p.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="If set, save one PNG per episode: value_pred vs value_target vs frame_index (sorted).",
    )
    p.add_argument(
        "--plot-max-episodes",
        type=int,
        default=None,
        help="If set, only plot the first N episodes (by ascending episode_index). Default: all.",
    )
    p.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
        help="DPI for saved PNGs (default 150).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())