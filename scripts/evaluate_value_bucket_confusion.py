"""Evaluate Pi0Value bucket classification on one dataset.

This script runs value-head inference, converts scalar ``value_target`` to the
nearest discrete bucket, and reports:

1. Top1 metrics: exact bucket match.
2. TopK metrics: relaxed match within a centered bucket window.
   Example: ``--topk 3`` accepts ``gt-1, gt, gt+1``.

Outputs per dataset (printed to stdout):
- Full confusion matrix
- Per-bucket metrics for Top1 / TopK
- Summary metrics

Example:
    python scripts/evaluate_value_bucket_confusion.py \
        --config-name pi0_value_umi_bimanual_headview_depth_multi_dataset \
        --checkpoint-dir ./checkpoints/pi0_value_umi_bimanual_headview_depth_multi_dataset/my_experiment/79999 \
        --val-dataset-root /root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260320 \
        --topk 3 \
        --batch-size 32
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

if "TMPDIR" not in os.environ:
    _tmp = Path(os.environ.get("HOME", "/root")) / "tmp"
    _tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_tmp)

nnx = None
jax = None
jnp = None
lerobot_dataset = None
np = None
tqdm = None
_model = None
_config = None
_data_loader = None
value_targets = None

_bc_kw = dict(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
if sys.version_info >= (3, 8):
    _bc_kw["force"] = True
logging.basicConfig(**_bc_kw)
logger = logging.getLogger(__name__)


def ensure_runtime_deps() -> None:
    global nnx, jax, jnp, lerobot_dataset, np, tqdm, _model, _config, _data_loader, value_targets
    if np is not None:
        return

    try:
        nnx = importlib.import_module("flax.nnx")
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        lerobot_dataset = importlib.import_module("lerobot.common.datasets.lerobot_dataset")
        np = importlib.import_module("numpy")
        tqdm = importlib.import_module("tqdm_loggable.auto")
        _model = importlib.import_module("openpi.models.model")
        _config = importlib.import_module("openpi.training.config")
        _data_loader = importlib.import_module("openpi.training.data_loader")
        value_targets = importlib.import_module("openpi.training.value_targets")
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise SystemExit(
            "\n".join(
                [
                    f"当前 Python 环境缺少依赖: {missing}",
                    f"当前解释器: {sys.executable}",
                    "这个脚本会直接加载 openpi value checkpoint，因此必须和能跑 "
                    "`scripts/lerobot_value_infer.py` 的解释器一致。",
                    "请使用真正可运行 openpi 的 Python 来执行，例如：",
                    "  /path/to/python scripts/evaluate_value_bucket_confusion.py ...",
                    "如果你已经执行了 `source .venv/bin/activate`，请先确认：",
                    "  which python",
                    "  python -c \"import flax, jax, openpi\"",
                ]
            )
        ) from exc


def parse_bool(x: Any) -> bool | None:
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    if isinstance(x, int):
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

    default_success_bool = parse_bool(default_success)
    episode_info: dict[int, value_targets.EpisodeTargetInfo] = {}
    task_max_length: dict[int, int] = {}

    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        ep_length = int(ep["length"])
        tasks = ep.get("tasks", [])
        task_name = tasks[0] if isinstance(tasks, list) and tasks else "unknown"
        task_index = task_name_to_index.get(task_name, 0)

        explicit_success = ep.get(success_field)
        ep_success = parse_bool(explicit_success) if explicit_success is not None else default_success_bool
        episode_info[ep_idx] = value_targets.EpisodeTargetInfo(
            task_index=task_index,
            length=ep_length,
            success=bool(ep_success),
        )
        task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)

    return episode_info, task_max_length


def resolve_single_dataset_factory(
    train_config: _config.TrainConfig,
    dataset_root: Path,
    checkpoint_dir: Path | None,
) -> _config.TrainConfig:
    data_factory = train_config.data
    if hasattr(data_factory, "datasets"):
        if not data_factory.datasets:
            raise ValueError("MultiDataConfigFactory.datasets is empty.")
        data_factory = data_factory.datasets[0]

    if not hasattr(data_factory, "repo_id"):
        raise TypeError(f"Unsupported data config factory: {type(data_factory)}")

    data_factory = dataclasses.replace(data_factory, repo_id=str(dataset_root))
    if checkpoint_dir is not None and hasattr(data_factory, "assets"):
        data_factory = dataclasses.replace(
            data_factory,
            assets=dataclasses.replace(data_factory.assets, assets_dir=str(checkpoint_dir / "assets")),
        )

    return dataclasses.replace(
        train_config,
        data=data_factory,
        batch_size=train_config.batch_size,
        num_workers=4,
    )


def load_targets(
    config: _config.TrainConfig,
    dataset_root: Path,
    raw_dataset: lerobot_dataset.LeRobotDataset,
    raw_frames,
    args: argparse.Namespace,
) -> np.ndarray:
    hf_cols = list(raw_frames.column_names)
    use_dataset_targets = args.target_source == "dataset" or (
        args.target_source == "auto" and "value_target" in hf_cols
    )

    if args.target_source == "dataset" and "value_target" not in hf_cols:
        raise ValueError("target_source=dataset but dataset has no 'value_target' column.")

    if use_dataset_targets:
        raw_vt = raw_frames["value_target"]
        targets = np.asarray(raw_vt, dtype=np.float32).reshape(len(raw_frames), -1)
        if targets.shape[1] > 1:
            targets = targets.mean(axis=1)
        else:
            targets = targets.reshape(-1)
        logger.info("Using dataset value_target from %s", dataset_root)
        return targets

    episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64).reshape(-1)
    episode_info, task_max_lengths = build_episode_info(
        raw_dataset,
        success_field=args.success_field,
        default_success=args.default_success,
    )
    targets = value_targets.compute_normalized_value_targets(
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        task_max_lengths=task_max_lengths,
        c_fail_coef=args.c_fail_coef,
        clip_min=config.value_clip_min,
        clip_max=config.value_clip_max,
    )
    logger.info("Computed normalized value_target for %s", dataset_root)
    return np.asarray(targets, dtype=np.float32).reshape(-1)


def scalar_values_to_bins(
    values: np.ndarray,
    *,
    value_min: float,
    value_max: float,
    num_bins: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = np.clip(values, value_min, value_max)
    step = (value_max - value_min) / max(num_bins - 1, 1)
    scaled = (values - value_min) / (step + 1e-8)
    bins = np.rint(scaled).astype(np.int64)
    return np.clip(bins, 0, num_bins - 1)


def infer_pred_bins(
    config: _config.TrainConfig,
    checkpoint_dir: Path,
    dataset_root: Path,
    batch_size: int,
) -> np.ndarray:
    num_devices = jax.device_count()
    effective_bs = batch_size
    if effective_bs % num_devices != 0:
        effective_bs = ((effective_bs + num_devices - 1) // num_devices) * num_devices
        logger.info(
            "Adjusted batch_size %d -> %d for %s",
            batch_size,
            effective_bs,
            dataset_root,
        )

    config = dataclasses.replace(config, batch_size=effective_bs)
    config = resolve_single_dataset_factory(config, dataset_root, checkpoint_dir)

    mesh = jax.sharding.Mesh(jax.devices(), ("batch",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_parallel = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))

    logger.info("Creating model for %s", dataset_root)
    model = config.model.create(jax.random.key(0))
    params_path = checkpoint_dir / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params directory: {params_path}")

    params = _model.restore_params(params_path, dtype=jnp.bfloat16, sharding=replicated)
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)
    model.eval()

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_parallel,
        shuffle=False,
    )

    raw_dataset = lerobot_dataset.LeRobotDataset(repo_id=str(dataset_root), root=str(dataset_root))
    frame_count = len(raw_dataset.hf_dataset)

    def _infer_logits(obs):
        return model.forward_value_logits(obs)

    infer_logits = jax.jit(_infer_logits, out_shardings=replicated)

    preds: list[np.ndarray] = []
    sample_cursor = 0
    batch_count = (frame_count + effective_bs - 1) // effective_bs
    pbar = tqdm.tqdm(total=batch_count, desc=f"Infer {dataset_root.name}")

    for obs, _actions in data_loader:
        if sample_cursor >= frame_count:
            break
        logits = infer_logits(obs)
        logits_np = np.asarray(logits).astype(np.float32)
        pred_bins = np.argmax(logits_np, axis=-1).reshape(-1)

        n_take = min(pred_bins.shape[0], int(obs.state.shape[0]), frame_count - sample_cursor)
        if n_take <= 0:
            break
        preds.append(pred_bins[:n_take].astype(np.int64, copy=False))
        sample_cursor += n_take
        pbar.update(1)

    pbar.close()

    if sample_cursor != frame_count:
        raise RuntimeError(f"Inference count mismatch for {dataset_root}: {sample_cursor} vs {frame_count}")

    return np.concatenate(preds, axis=0)


def build_confusion_matrix(true_bins: np.ndarray, pred_bins: np.ndarray, num_bins: int) -> np.ndarray:
    confusion = np.zeros((num_bins, num_bins), dtype=np.int64)
    np.add.at(confusion, (true_bins, pred_bins), 1)
    return confusion


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def compute_top1_per_bin_metrics(confusion: np.ndarray, centers: np.ndarray) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    true_support = confusion.sum(axis=1)
    pred_support = confusion.sum(axis=0)
    total = int(confusion.sum())

    for i in range(confusion.shape[0]):
        tp = int(confusion[i, i])
        fp = int(pred_support[i] - tp)
        fn = int(true_support[i] - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall) if not (np.isnan(precision) or np.isnan(recall)) else float("nan")
        rows.append(
            {
                "bin_index": i,
                "bin_center": float(centers[i]),
                "support_true": int(true_support[i]),
                "support_pred": int(pred_support[i]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy_contrib": _safe_div(tp, total),
            }
        )
    return rows


def compute_topk_per_bin_metrics(
    confusion: np.ndarray,
    centers: np.ndarray,
    radius: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    true_support = confusion.sum(axis=1)
    pred_support = confusion.sum(axis=0)

    for i in range(confusion.shape[0]):
        lo = max(0, i - radius)
        hi = min(confusion.shape[0] - 1, i + radius)

        relaxed_recall_hits = int(confusion[i, lo : hi + 1].sum())
        relaxed_precision_hits = int(confusion[lo : hi + 1, i].sum())
        relaxed_recall = _safe_div(relaxed_recall_hits, int(true_support[i]))
        relaxed_precision = _safe_div(relaxed_precision_hits, int(pred_support[i]))
        relaxed_f1 = (
            _safe_div(2.0 * relaxed_precision * relaxed_recall, relaxed_precision + relaxed_recall)
            if not (np.isnan(relaxed_precision) or np.isnan(relaxed_recall))
            else float("nan")
        )

        rows.append(
            {
                "bin_index": i,
                "bin_center": float(centers[i]),
                "support_true": int(true_support[i]),
                "support_pred": int(pred_support[i]),
                "window_left": lo,
                "window_right": hi,
                "relaxed_precision_hits": relaxed_precision_hits,
                "relaxed_recall_hits": relaxed_recall_hits,
                "precision": relaxed_precision,
                "recall": relaxed_recall,
                "f1": relaxed_f1,
            }
        )
    return rows


def summarize_metrics(
    confusion: np.ndarray,
    top1_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    topk_correct: np.ndarray,
    topk: int,
) -> dict[str, Any]:
    total = int(confusion.sum())
    top1_acc = _safe_div(float(np.trace(confusion)), float(total))

    true_support = confusion.sum(axis=1).astype(np.float64)
    weights = true_support / max(float(np.sum(true_support)), 1.0)

    def _macro(rows: list[dict[str, Any]], key: str) -> float:
        vals = [float(r[key]) for r in rows if not np.isnan(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    def _weighted(rows: list[dict[str, Any]], key: str) -> float:
        vals = np.array(
            [0.0 if np.isnan(float(r[key])) else float(r[key]) for r in rows],
            dtype=np.float64,
        )
        return float(np.sum(vals * weights))

    return {
        "num_samples": total,
        "num_bins": int(confusion.shape[0]),
        "top1_accuracy": top1_acc,
        "top1_macro_precision": _macro(top1_rows, "precision"),
        "top1_macro_recall": _macro(top1_rows, "recall"),
        "top1_macro_f1": _macro(top1_rows, "f1"),
        "top1_weighted_precision": _weighted(top1_rows, "precision"),
        "top1_weighted_recall": _weighted(top1_rows, "recall"),
        "top1_weighted_f1": _weighted(top1_rows, "f1"),
        "topk": int(topk),
        "topk_radius": int((topk - 1) // 2),
        "topk_accuracy": float(np.mean(topk_correct.astype(np.float32))) if total > 0 else float("nan"),
        "topk_macro_precision": _macro(topk_rows, "precision"),
        "topk_macro_recall": _macro(topk_rows, "recall"),
        "topk_macro_f1": _macro(topk_rows, "f1"),
        "topk_weighted_precision": _weighted(topk_rows, "precision"),
        "topk_weighted_recall": _weighted(topk_rows, "recall"),
        "topk_weighted_f1": _weighted(topk_rows, "f1"),
    }


def _fmt_float(x: Any) -> str:
    x = float(x)
    return "nan" if np.isnan(x) else f"{x:.6f}"


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 100)
    print(f"Dataset      : {summary['dataset_name']}")
    print(f"Dataset root : {summary['dataset_root']}")
    print(f"Checkpoint   : {summary['checkpoint_dir']}")
    print(f"Config       : {summary['config_name']}")
    print(f"Samples      : {summary['num_samples']}")
    print(f"Bins         : {summary['num_bins']}")
    print("-" * 100)
    print(
        "Top1  | "
        f"acc={_fmt_float(summary['top1_accuracy'])}  "
        f"macro_p={_fmt_float(summary['top1_macro_precision'])}  "
        f"macro_r={_fmt_float(summary['top1_macro_recall'])}  "
        f"macro_f1={_fmt_float(summary['top1_macro_f1'])}  "
        f"weighted_f1={_fmt_float(summary['top1_weighted_f1'])}"
    )
    print(
        f"Top{summary['topk']} | "
        f"acc={_fmt_float(summary['topk_accuracy'])}  "
        f"macro_p={_fmt_float(summary['topk_macro_precision'])}  "
        f"macro_r={_fmt_float(summary['topk_macro_recall'])}  "
        f"macro_f1={_fmt_float(summary['topk_macro_f1'])}  "
        f"weighted_f1={_fmt_float(summary['topk_weighted_f1'])}"
    )


def print_confusion_matrix(confusion: np.ndarray, centers: np.ndarray) -> None:
    print("-" * 100)
    print("Confusion matrix (rows=true bucket, cols=pred bucket)")
    print("Columns:", " ".join(f"{i}:{centers[i]:.3f}" for i in range(confusion.shape[1])))
    for i in range(confusion.shape[0]):
        row_label = f"{i}:{centers[i]:.3f}"
        row_vals = " ".join(str(int(v)) for v in confusion[i])
        print(f"{row_label} | {row_vals}")


def print_metric_rows(title: str, rows: list[dict[str, Any]]) -> None:
    print("-" * 100)
    print(title)
    print(
        "bin  center      support_true  support_pred  "
        "extra1      extra2      precision  recall     f1"
    )
    for row in rows:
        extra1 = row.get("tp", row.get("window_left", ""))
        extra2 = row.get("fp", row.get("window_right", ""))
        if "relaxed_precision_hits" in row:
            extra1 = row["relaxed_precision_hits"]
            extra2 = row["relaxed_recall_hits"]
        print(
            f"{int(row['bin_index']):3d}  "
            f"{float(row['bin_center']):10.6f}  "
            f"{int(row['support_true']):12d}  "
            f"{int(row['support_pred']):12d}  "
            f"{str(extra1):10}  "
            f"{str(extra2):10}  "
            f"{_fmt_float(row['precision']):9}  "
            f"{_fmt_float(row['recall']):9}  "
            f"{_fmt_float(row['f1'])}"
        )


def evaluate_one_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    dataset_root: Path,
    base_config: _config.TrainConfig,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    logger.info("=" * 80)
    logger.info("Evaluating %s dataset: %s", dataset_name, dataset_root)

    raw_dataset = lerobot_dataset.LeRobotDataset(repo_id=str(dataset_root), root=str(dataset_root))
    raw_frames = raw_dataset.hf_dataset.with_format(None)
    if len(raw_frames) == 0:
        raise ValueError(f"Dataset has no frames: {dataset_root}")

    targets = load_targets(base_config, dataset_root, raw_dataset, raw_frames, args)
    pred_bins = infer_pred_bins(base_config, checkpoint_dir, dataset_root, args.batch_size)

    model_cfg = base_config.model
    num_bins = int(getattr(model_cfg, "num_value_bins"))
    value_min = float(getattr(model_cfg, "value_min"))
    value_max = float(getattr(model_cfg, "value_max"))
    centers = np.linspace(value_min, value_max, num_bins, dtype=np.float32)

    true_bins = scalar_values_to_bins(
        targets,
        value_min=value_min,
        value_max=value_max,
        num_bins=num_bins,
    )
    confusion = build_confusion_matrix(true_bins, pred_bins, num_bins)

    radius = (args.topk - 1) // 2
    topk_correct = np.abs(pred_bins - true_bins) <= radius
    top1_rows = compute_top1_per_bin_metrics(confusion, centers)
    topk_rows = compute_topk_per_bin_metrics(confusion, centers, radius)
    summary = summarize_metrics(confusion, top1_rows, topk_rows, topk_correct, args.topk)
    summary.update(
        {
            "dataset_name": dataset_name,
            "dataset_root": str(dataset_root),
            "checkpoint_dir": str(checkpoint_dir),
            "config_name": args.config_name,
        }
    )

    logger.info(
        "%s | top1_acc=%.6f | top%d_acc=%.6f | top1_macro_f1=%.6f | top%d_macro_f1=%.6f",
        dataset_name,
        summary["top1_accuracy"],
        args.topk,
        summary["topk_accuracy"],
        summary["top1_macro_f1"],
        args.topk,
        summary["topk_macro_f1"],
    )
    print_summary(summary)
    print_confusion_matrix(confusion, centers)
    print_metric_rows("Per-bin Top1 metrics (extra1=tp, extra2=fp)", top1_rows)
    print_metric_rows(
        f"Per-bin Top{args.topk} metrics (extra1=relaxed_precision_hits, extra2=relaxed_recall_hits)",
        topk_rows,
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Pi0Value bucket confusion matrix on one dataset.")
    p.add_argument(
        "--config-name",
        type=str,
        default="pi0_value_umi_bimanual_headview_depth_multi_dataset",
        help="Training config name used to build the model and data pipeline.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint step directory containing params/ and assets/.",
    )
    p.add_argument("--val-dataset-root", type=str, required=True, help="Validation dataset root.")
    p.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    p.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Odd bucket window size centered on GT bucket. Top1=exact, Top3=gt-1/gt/gt+1.",
    )
    p.add_argument(
        "--target-source",
        type=str,
        choices=("auto", "dataset", "computed"),
        default="auto",
        help="auto=use dataset value_target if present else compute from meta.",
    )
    p.add_argument(
        "--c-fail-coef",
        type=float,
        default=1.0,
        help="Used only when targets are computed from metadata.",
    )
    p.add_argument("--success-field", type=str, default="success", help="Success field in episodes.jsonl.")
    p.add_argument("--default-success", type=str, default="true", help="Default success if field missing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.topk <= 0 or args.topk % 2 == 0:
        raise ValueError("--topk must be a positive odd integer. Use 1, 3, 5, ...")

    ensure_runtime_deps()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    config = _config.get_config(args.config_name)
    dataset_root = Path(args.val_dataset_root).resolve()
    evaluate_one_dataset(
        args=args,
        dataset_name="val",
        dataset_root=dataset_root,
        base_config=config,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
