"""
Value inference + advantage annotation for LeRobot datasets.

This version:
- reads precomputed value_target from the dataset
- does NOT regenerate value_target from success/failure metadata
- runs value inference to get predicted_value
- computes rewards and n-step advantages from the existing value_target
- writes predicted_value / advantage / is_positive back to parquet
- provides strong debug logs and optional CSV dump

Example:
python scripts/lerobot_value_infer_use_existing_target.py \
    --config-name pi0_value_umi_bimanual_headview_depth_infer \
    --checkpoint-dir /root/openpi-umi/checkpoints/pi0_value_umi_bimanual_headview_depth/my_experiment_v2/59999 \
    --dataset-root /root/openpi-umi/data/horizon_cloth_folding_advantage_eval_failure_20260404_161149_to_20260404_161519_ep4 \
    --value-target-field value_target \
    --n-step 50 \
    --batch-size 32 \
    --debug-episode 1 \
    --debug-tail-k 80 \
    --debug-dump-csv
"""

import argparse
import csv
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

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm_loggable.auto as tqdm

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("===== logger initialized successfully =====")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_numpy_1d(x, dtype):
    arr = np.asarray(x, dtype=dtype)
    return arr.reshape(-1)


def load_existing_value_targets(
    raw_frames,
    field_name: str,
) -> np.ndarray:
    """
    Load existing value_target from hf dataset.

    Supports common storage layouts:
    - scalar column: shape [N]
    - vector column with one element: shape [N, 1]
    """
    column_names = getattr(raw_frames, "column_names", None)
    if column_names is None:
        try:
            column_names = list(raw_frames.features.keys())
        except Exception:
            column_names = []

    if field_name not in column_names:
        raise KeyError(
            f"Dataset does not contain precomputed '{field_name}' column. "
            f"Available columns: {column_names}"
        )

    arr = np.asarray(raw_frames[field_name], dtype=np.float32)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].astype(np.float32, copy=False)

    # Fallback: flatten if user stored shape (N, 1, 1) etc.
    if arr.shape[0] > 0:
        reshaped = arr.reshape(arr.shape[0], -1)
        if reshaped.shape[1] == 1:
            return reshaped[:, 0].astype(np.float32, copy=False)

    raise ValueError(
        f"Unsupported shape for '{field_name}': {arr.shape}. "
        "Expected [N] or [N,1]."
    )


def log_frame_order_check(
    logger: logging.Logger,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    name: str,
) -> None:
    """Check whether adjacent rows inside the same episode are frame-contiguous."""
    bad = 0
    for i in range(len(frame_indices) - 1):
        same_ep = episode_indices[i] == episode_indices[i + 1]
        if same_ep and frame_indices[i + 1] != frame_indices[i] + 1:
            if bad < 20:
                logger.warning(
                    "[%s] Non-contiguous adjacent rows at row %d: ep=%d frame=%d -> next frame=%d",
                    name,
                    i,
                    int(episode_indices[i]),
                    int(frame_indices[i]),
                    int(frame_indices[i + 1]),
                )
            bad += 1
    logger.info("[%s] Non-contiguous adjacent pairs inside same episode: %d", name, bad)


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def compute_dense_rewards_from_targets(
    targets: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Compute per-frame rewards from existing value targets.

    reward[i] = target[i] - target[i+1]  (if i+1 is next frame in same episode)
    reward[i] = target[i]                (at episode boundary / last frame)
    """
    n = targets.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    same_ep = episode_indices[:-1] == episode_indices[1:]
    contiguous = frame_indices[1:] == frame_indices[:-1] + 1
    is_next = same_ep & contiguous

    rewards = targets.copy().astype(np.float32)
    rewards[:-1] = np.where(is_next, targets[:-1] - targets[1:], targets[:-1])
    return rewards


def compute_n_step_advantages_debug(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    n_step: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """N-step advantage with debug tensors.

    A(i) = sum_{k=0}^{n-1} r(i+k) + V(i+n) - V(i)

    Bootstrap with V(i+n) only if i+n is in the same episode and contiguous.
    """
    if n_step <= 0:
        raise ValueError("n_step must be > 0.")

    n = rewards.shape[0]
    if n == 0:
        empty_f = np.zeros(0, dtype=np.float32)
        empty_i = np.zeros(0, dtype=np.int64)
        empty_b = np.zeros(0, dtype=np.bool_)
        return empty_f, {
            "effective_n": empty_i,
            "max_reach": empty_i,
            "reward_sum": empty_f,
            "bootstrap": empty_f,
            "can_bootstrap": empty_b,
        }

    same_ep = np.empty(n, dtype=np.bool_)
    same_ep[:-1] = (
        (episode_indices[:-1] == episode_indices[1:])
        & (frame_indices[1:] == frame_indices[:-1] + 1)
    )
    same_ep[-1] = False

    max_reach = np.zeros(n, dtype=np.int64)
    max_reach[-1] = 1
    for i in range(n - 2, -1, -1):
        if same_ep[i]:
            max_reach[i] = max_reach[i + 1] + 1
        else:
            max_reach[i] = 1

    cum_rewards = np.zeros(n + 1, dtype=np.float64)
    cum_rewards[1:] = np.cumsum(rewards.astype(np.float64))

    effective_n = np.minimum(n_step, max_reach).astype(np.int64)
    idx = np.arange(n, dtype=np.int64)

    reward_sum = (cum_rewards[idx + effective_n] - cum_rewards[idx]).astype(np.float32)

    can_bootstrap = (effective_n == n_step) & ((idx + n_step) < n)
    idx_n = np.minimum(idx + n_step, n - 1)
    can_bootstrap &= (episode_indices[idx_n] == episode_indices[idx])
    can_bootstrap &= (frame_indices[idx_n] == frame_indices[idx] + n_step)

    bootstrap = np.where(
        can_bootstrap,
        values[idx_n],
        0.0,
    ).astype(np.float32)

    advantages = reward_sum + bootstrap - values.astype(np.float32)

    debug = {
        "effective_n": effective_n,
        "max_reach": max_reach,
        "reward_sum": reward_sum,
        "bootstrap": bootstrap,
        "can_bootstrap": can_bootstrap,
    }
    return advantages, debug


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def log_advantage_debug(
    logger: logging.Logger,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    value_tgts: np.ndarray,
    predicted_values: np.ndarray,
    advantages: np.ndarray,
    adv_debug: dict[str, np.ndarray],
    debug_episode: int | None = None,
    tail_k: int = 60,
    tol: float = 1e-5,
) -> None:
    """Check internal consistency and optionally dump one episode tail to logs."""
    effective_n = adv_debug["effective_n"]
    reward_sum = adv_debug["reward_sum"]
    bootstrap = adv_debug["bootstrap"]
    can_bootstrap = adv_debug["can_bootstrap"]

    formula = reward_sum + bootstrap - predicted_values
    formula_err = float(np.max(np.abs(advantages - formula))) if advantages.size > 0 else 0.0
    logger.info(
        "Adv debug | max|adv - (reward_sum + bootstrap - value)| = %.8f",
        formula_err,
    )

    no_boot = ~can_bootstrap
    if np.any(no_boot):
        no_boot_expected = value_tgts[no_boot] - predicted_values[no_boot]
        no_boot_err = float(np.max(np.abs(advantages[no_boot] - no_boot_expected)))
        logger.info(
            "Adv debug | no-bootstrap frames=%d, max|adv - (target - value)| = %.8f",
            int(np.sum(no_boot)),
            no_boot_err,
        )

        bad = np.where(no_boot & (np.abs(advantages - (value_tgts - predicted_values)) > tol))[0]
        if bad.size > 0:
            logger.warning("Found %d no-bootstrap frames violating adv = target - value", bad.size)
            for j in bad[:20]:
                logger.warning(
                    "bad row=%d ep=%d frame=%d eff_n=%d can_boot=%d "
                    "target=%.6f pred=%.6f reward_sum=%.6f boot=%.6f adv=%.6f target-pred=%.6f",
                    int(j),
                    int(episode_indices[j]),
                    int(frame_indices[j]),
                    int(effective_n[j]),
                    int(can_bootstrap[j]),
                    float(value_tgts[j]),
                    float(predicted_values[j]),
                    float(reward_sum[j]),
                    float(bootstrap[j]),
                    float(advantages[j]),
                    float(value_tgts[j] - predicted_values[j]),
                )

    if debug_episode is not None:
        ep_mask = episode_indices == debug_episode
        ep_idx = np.where(ep_mask)[0]
        if ep_idx.size == 0:
            logger.warning("debug_episode=%d not found", debug_episode)
            return

        tail_idx = ep_idx[-tail_k:]
        logger.info("===== Debug tail for episode %d (last %d frames) =====", debug_episode, len(tail_idx))
        for j in tail_idx:
            logger.info(
                "row=%d frame=%d eff_n=%d can_boot=%d "
                "target=%.6f pred=%.6f reward_sum=%.6f boot=%.6f adv=%.6f target-pred=%.6f",
                int(j),
                int(frame_indices[j]),
                int(effective_n[j]),
                int(can_bootstrap[j]),
                float(value_tgts[j]),
                float(predicted_values[j]),
                float(reward_sum[j]),
                float(bootstrap[j]),
                float(advantages[j]),
                float(value_tgts[j] - predicted_values[j]),
            )


def dump_episode_debug_csv(
    out_path: Path,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    value_tgts: np.ndarray,
    predicted_values: np.ndarray,
    advantages: np.ndarray,
    indicators: np.ndarray,
    adv_debug: dict[str, np.ndarray],
    episode_id: int,
) -> None:
    """Dump one episode's detailed debug rows to CSV."""
    mask = episode_indices == episode_id
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        logger.warning("Cannot dump CSV: episode %d not found", episode_id)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row",
            "episode_index",
            "frame_index",
            "effective_n",
            "max_reach",
            "can_bootstrap",
            "value_target",
            "predicted_value",
            "reward_sum",
            "bootstrap",
            "advantage",
            "target_minus_pred",
            "is_positive",
        ])
        for j in idxs:
            writer.writerow([
                int(j),
                int(episode_indices[j]),
                int(frame_indices[j]),
                int(adv_debug["effective_n"][j]),
                int(adv_debug["max_reach"][j]),
                int(adv_debug["can_bootstrap"][j]),
                float(value_tgts[j]),
                float(predicted_values[j]),
                float(adv_debug["reward_sum"][j]),
                float(adv_debug["bootstrap"][j]),
                float(advantages[j]),
                float(value_tgts[j] - predicted_values[j]),
                int(indicators[j]),
            ])
    logger.info("Wrote debug CSV: %s", out_path)


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def compute_task_thresholds(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    positive_ratio: float,
) -> dict[int, float]:
    """Per-task quantile threshold: frames above threshold are 'positive'."""
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("positive_ratio must be in [0, 1].")

    quantile = 1.0 - positive_ratio
    thresholds: dict[int, float] = {}
    for task_idx in np.unique(task_indices):
        task_adv = advantages[task_indices == task_idx]
        if task_adv.size == 0:
            thresholds[int(task_idx)] = float("inf")
        else:
            thresholds[int(task_idx)] = float(np.quantile(task_adv, quantile))
    return thresholds


def binarize_advantages(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    thresholds: dict[int, float],
) -> np.ndarray:
    """Assign indicator=1 if advantage >= task threshold, else 0."""
    threshold_arr = np.full(advantages.shape[0], np.inf, dtype=np.float32)
    for task_idx, thr in thresholds.items():
        mask = task_indices == task_idx
        threshold_arr[mask] = thr
    return (advantages >= threshold_arr).astype(np.int64)


# ---------------------------------------------------------------------------
# Parquet write-back
# ---------------------------------------------------------------------------

def write_columns_to_parquet(
    dataset_root: Path,
    absolute_indices: np.ndarray,
    columns: dict[str, np.ndarray],
    feature_infos: dict[str, dict[str, Any]],
) -> None:
    """Write annotation columns back into dataset parquet files in-place."""
    max_index = int(np.max(absolute_indices))
    selected = np.zeros(max_index + 1, dtype=np.bool_)
    selected[absolute_indices] = True

    lookups: dict[str, np.ndarray] = {}
    for field, vals in columns.items():
        lookup_dtype = np.float32 if feature_infos[field]["dtype"] == "float32" else np.int64
        lookup = np.zeros(max_index + 1, dtype=lookup_dtype)
        lookup[absolute_indices] = vals.astype(lookup_dtype, copy=False)
        lookups[field] = lookup

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    for pq_path in tqdm.tqdm(parquet_files, desc="Writing annotations", leave=False):
        table = pq.read_table(pq_path)
        idx_np = table["index"].to_numpy().astype(np.int64, copy=False)

        in_range = (idx_np >= 0) & (idx_np <= max_index)
        in_subset = np.zeros_like(in_range)
        in_subset[in_range] = selected[idx_np[in_range]]

        new_table = table
        for field, lookup in lookups.items():
            ftype = feature_infos[field]["dtype"]
            if ftype == "float32":
                default_value = np.nan
                target_dtype = np.float32
                pa_type = pa.float32()
            else:
                default_value = 0
                target_dtype = np.int64
                pa_type = pa.int64()

            if field in new_table.schema.names:
                current = new_table[field].to_numpy().astype(target_dtype, copy=True)
            else:
                current = np.full(idx_np.shape[0], default_value, dtype=target_dtype)

            if np.any(in_subset):
                current[in_subset] = lookup[idx_np[in_subset]]

            array = pa.array(current, type=pa_type)
            if field in new_table.schema.names:
                col_idx = new_table.schema.names.index(field)
                new_table = new_table.set_column(col_idx, field, array)
            else:
                new_table = new_table.append_column(field, array)

        pq.write_table(new_table, pq_path, compression="snappy")

    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    for feature_name, feature_info in feature_infos.items():
        info["features"][feature_name] = {
            "dtype": feature_info["dtype"],
            "shape": list(feature_info["shape"]),
            "names": feature_info.get("names"),
        }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    logger.info("Updated info.json with new feature metadata.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_value_inference(args: argparse.Namespace) -> None:
    config = _config.get_config(args.config_name)
    dataset_root = Path(args.dataset_root).resolve()

    logger.info("Config: %s", args.config_name)
    logger.info("Checkpoint: %s", args.checkpoint_dir)
    logger.info("Dataset root: %s", dataset_root)
    logger.info("Using existing value target field: %s", args.value_target_field)

    # ---- 1. Load model ----
    num_devices = jax.device_count()
    mesh = jax.sharding.Mesh(jax.devices(), ("batch",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_parallel = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    logger.info("Inference mesh: %d device(s) %s", num_devices, jax.devices())

    logger.info("Creating model from config...")
    model = config.model.create(jax.random.key(0))

    params_path = Path(args.checkpoint_dir) / "params"
    logger.info("Restoring params from %s", params_path)
    params = _model.restore_params(params_path, dtype=jnp.bfloat16, sharding=replicated)

    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)
    model.eval()
    logger.info("Model loaded and replicated across %d device(s).", num_devices)

    # ---- 2. Load raw dataset metadata ----
    raw_dataset = lerobot_dataset.LeRobotDataset(
        repo_id=str(dataset_root),
        root=str(dataset_root),
    )
    raw_frames = raw_dataset.hf_dataset.with_format(None)
    frame_count = len(raw_frames)
    if frame_count == 0:
        raise ValueError("Dataset has no frames.")

    absolute_indices = _to_numpy_1d(raw_frames["index"], np.int64)
    episode_indices = _to_numpy_1d(raw_frames["episode_index"], np.int64)
    frame_indices = _to_numpy_1d(raw_frames["frame_index"], np.int64)
    task_indices = _to_numpy_1d(raw_frames["task_index"], np.int64)
    existing_value_targets = load_existing_value_targets(raw_frames, args.value_target_field)

    logger.info("Dataset: %d frames, %d episodes", frame_count, len(np.unique(episode_indices)))
    logger.info(
        "Existing value_target stats | min=%.6f max=%.6f mean=%.6f std=%.6f",
        float(np.min(existing_value_targets)),
        float(np.max(existing_value_targets)),
        float(np.mean(existing_value_targets)),
        float(np.std(existing_value_targets)),
    )

    log_frame_order_check(logger, episode_indices, frame_indices, name="raw_order")

    # Explicit sort for all adjacency-based computations.
    sort_order = np.lexsort((frame_indices, episode_indices))
    inv_sort = np.empty_like(sort_order)
    inv_sort[sort_order] = np.arange(sort_order.shape[0])

    absolute_indices_sorted = absolute_indices[sort_order]
    episode_indices_sorted = episode_indices[sort_order]
    frame_indices_sorted = frame_indices[sort_order]
    task_indices_sorted = task_indices[sort_order]
    value_tgts_sorted = existing_value_targets[sort_order]

    log_frame_order_check(logger, episode_indices_sorted, frame_indices_sorted, name="sorted_order")

    # ---- 3. Batched value inference ----
    effective_bs = args.batch_size
    if effective_bs % num_devices != 0:
        effective_bs = ((effective_bs + num_devices - 1) // num_devices) * num_devices
        logger.info(
            "Adjusted batch_size %d -> %d (divisible by %d devices)",
            args.batch_size,
            effective_bs,
            num_devices,
        )

    infer_config = _replace_config_for_inference(
        config, dataset_root, effective_bs, checkpoint_dir=args.checkpoint_dir
    )
    data_loader = _data_loader.create_data_loader(
        infer_config,
        sharding=data_parallel,
        shuffle=False,
    )

    max_abs_index = int(np.max(absolute_indices))
    prediction_lookup = np.zeros(max_abs_index + 1, dtype=np.float32)
    prediction_seen = np.zeros(max_abs_index + 1, dtype=np.bool_)

    def _infer_value(obs):
        logits = model.forward_value_logits(obs)
        return model.expected_value_from_logits(logits)

    infer_value = jax.jit(_infer_value, out_shardings=replicated)

    batch_count = (frame_count + effective_bs - 1) // effective_bs
    sample_cursor = 0

    logger.info(
        "Starting value inference (%d batches, batch_size=%d, %d device(s))...",
        batch_count,
        effective_bs,
        num_devices,
    )
    pbar = tqdm.tqdm(total=batch_count, desc="Value inference")
    batches_done = 0

    # NOTE:
    # This still assumes the dataloader emits samples in the same order as raw_frames when shuffle=False.
    # Earlier checks suggested math consistency was fine, but if predicted_value ever looks misaligned,
    # this is still the first place to suspect.
    prev_values = None
    prev_bs = 0
    prev_cursor = 0

    for obs, _actions in data_loader:
        bs = obs.state.shape[0]
        cur_values = infer_value(obs)

        if prev_values is not None:
            val_np = np.asarray(prev_values).astype(np.float32).reshape(-1)
            batch_abs_indices = absolute_indices[prev_cursor: prev_cursor + prev_bs]
            prediction_lookup[batch_abs_indices] = val_np[: len(batch_abs_indices)]
            prediction_seen[batch_abs_indices] = True
            pbar.update(1)

        prev_values = cur_values
        prev_bs = bs
        prev_cursor = sample_cursor
        sample_cursor += bs

        batches_done += 1
        if batches_done >= batch_count:
            break

    if prev_values is not None:
        val_np = np.asarray(prev_values).astype(np.float32).reshape(-1)
        batch_abs_indices = absolute_indices[prev_cursor: prev_cursor + prev_bs]
        prediction_lookup[batch_abs_indices] = val_np[: len(batch_abs_indices)]
        prediction_seen[batch_abs_indices] = True
        pbar.update(1)

    pbar.close()

    missing = ~prediction_seen[absolute_indices]
    if np.any(missing):
        missing_count = int(np.sum(missing))
        logger.warning("Missing predictions for %d frames, filling with 0.", missing_count)

    predicted_values = prediction_lookup[absolute_indices]
    predicted_values_sorted = prediction_lookup[absolute_indices_sorted]

    logger.info(
        "Predicted value stats | min=%.6f max=%.6f mean=%.6f std=%.6f",
        float(np.min(predicted_values)),
        float(np.max(predicted_values)),
        float(np.mean(predicted_values)),
        float(np.std(predicted_values)),
    )

    # ---- 4. Compute rewards and advantages from existing value_target ----
    rewards_sorted = compute_dense_rewards_from_targets(
        value_tgts_sorted,
        episode_indices_sorted,
        frame_indices_sorted,
    )

    advantages_sorted, adv_debug = compute_n_step_advantages_debug(
        rewards=rewards_sorted,
        values=predicted_values_sorted,
        episode_indices=episode_indices_sorted,
        frame_indices=frame_indices_sorted,
        n_step=args.n_step,
    )

    thresholds = compute_task_thresholds(
        task_indices_sorted,
        advantages_sorted,
        args.positive_ratio,
    )
    indicators_sorted = binarize_advantages(
        task_indices_sorted,
        advantages_sorted,
        thresholds,
    )

    log_advantage_debug(
        logger=logger,
        episode_indices=episode_indices_sorted,
        frame_indices=frame_indices_sorted,
        value_tgts=value_tgts_sorted,
        predicted_values=predicted_values_sorted,
        advantages=advantages_sorted,
        adv_debug=adv_debug,
        debug_episode=args.debug_episode,
        tail_k=args.debug_tail_k,
    )

    if args.debug_dump_csv and args.debug_episode is not None:
        dump_path = dataset_root / "debug_advantage" / f"episode_{args.debug_episode:06d}_debug.csv"
        dump_episode_debug_csv(
            out_path=dump_path,
            episode_indices=episode_indices_sorted,
            frame_indices=frame_indices_sorted,
            value_tgts=value_tgts_sorted,
            predicted_values=predicted_values_sorted,
            advantages=advantages_sorted,
            indicators=indicators_sorted,
            adv_debug=adv_debug,
            episode_id=args.debug_episode,
        )

    positive_ratio_observed = float(np.mean(indicators_sorted.astype(np.float32)))
    logger.info(
        "ACP stats | n_step=%d positive_ratio_target=%.4f positive_ratio_observed=%.4f",
        args.n_step,
        args.positive_ratio,
        positive_ratio_observed,
    )
    for task_idx, threshold in sorted(thresholds.items()):
        logger.info("  task %d threshold=%.6f", task_idx, threshold)

    # ---- 5. Unsort back to original row order before write-back ----
    advantages = advantages_sorted[inv_sort]
    indicators = indicators_sorted[inv_sort]

    # ---- 6. Write back to parquet ----
    columns = {
        "predicted_value": predicted_values.astype(np.float32),
        "advantage": advantages.astype(np.float32),
        "is_positive": indicators.astype(np.int64),
    }
    feature_infos = {
        "predicted_value": {"dtype": "float32", "shape": (1,), "names": None},
        "advantage": {"dtype": "float32", "shape": (1,), "names": None},
        "is_positive": {"dtype": "int64", "shape": (1,), "names": None},
    }

    write_columns_to_parquet(
        dataset_root=dataset_root,
        absolute_indices=absolute_indices,
        columns=columns,
        feature_infos=feature_infos,
    )
    logger.info("Done. Wrote predicted_value, advantage, is_positive to %s", dataset_root)


def _replace_config_for_inference(
    config: _config.TrainConfig,
    dataset_root: Path,
    batch_size: int,
    checkpoint_dir: str | None = None,
) -> _config.TrainConfig:
    """Create a modified config that points to the real dataset for inference."""
    import dataclasses

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Value inference + advantage annotation using existing value_target from dataset"
    )
    parser.add_argument("--config-name", type=str, required=True, help="Name of the training config")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to checkpoint step dir (contains params/)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to LeRobot dataset root")
    parser.add_argument("--value-target-field", type=str, default="value_target", help="Existing dataset field to use as value target")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--n-step", type=int, default=50, help="N-step for advantage computation")
    parser.add_argument("--positive-ratio", type=float, default=0.4, help="Target ratio of positive indicators per task")
    parser.add_argument("--debug-episode", type=int, default=None, help="Episode index to dump tail debug logs for")
    parser.add_argument("--debug-tail-k", type=int, default=60, help="How many tail frames to dump for debug episode")
    parser.add_argument("--debug-dump-csv", action="store_true", help="Dump one episode debug CSV under dataset_root/debug_advantage/")
    return parser.parse_args()


if __name__ == "__main__":
    run_value_inference(parse_args())