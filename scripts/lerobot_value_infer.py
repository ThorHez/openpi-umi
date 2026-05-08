"""Value inference + advantage annotation for LeRobot datasets.

Loads a trained Pi0Value checkpoint, runs batched value inference over every
frame, computes n-step advantages from value targets, binarizes them into
positive/negative indicators, and writes all annotations back into the
dataset's parquet files.

<<<<<<< Updated upstream
Usage:
    python scripts/lerobot_value_infer.py \
        --config-name pi0_value_umi \
        --checkpoint-dir ./checkpoints/pi0_value_umi/my_exp/10000 \
        --dataset-root ./data/my_lerobot_dataset \
        --n-step 1 \
        --positive-ratio 0.5 \
        --c-fail-coef 1.0 \
        --success-field success \
        --default-success true \
=======
Two modes are supported via ``--dataset-root``:

* **Single-dataset mode** -- if the path is itself a LeRobot dataset root
  (i.e. it contains ``meta/info.json`` and ``data/chunk-*/episode_*.parquet``),
  only that dataset is processed.
* **Batch mode** -- otherwise the path is treated as a *parent directory* and
  every immediate subdirectory that looks like a LeRobot dataset is processed
  in turn. The model is loaded **once** and reused across all datasets, so the
  per-dataset overhead is just data loading + inference + writeback.

Usage::

    # Single dataset
    python scripts/lerobot_value_infer.py \\
        --config-name pi0_value_umi \\
        --checkpoint-dir ./checkpoints/pi0_value_umi/my_exp/10000 \\
        --dataset-root ./data/my_lerobot_dataset \\
        --batch-size 32

    # All LeRobot datasets directly under ./data
    python scripts/lerobot_value_infer.py \\
        --config-name pi0_value_umi \\
        --checkpoint-dir ./checkpoints/pi0_value_umi/my_exp/10000 \\
        --dataset-root ./data \\
        --skip-existing \\
        --continue-on-error \\
>>>>>>> Stashed changes
        --batch-size 32
"""

import argparse
<<<<<<< Updated upstream
import json
import logging
import os
from pathlib import Path
from typing import Any
=======
import dataclasses
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, NamedTuple
>>>>>>> Stashed changes

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
import openpi.training.sharding as sharding
import openpi.training.value_targets as value_targets

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
<<<<<<< Updated upstream
=======
    force=True,  # openpi imports install a WARNING-level handler; override it
>>>>>>> Stashed changes
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode metadata extraction
# ---------------------------------------------------------------------------

def build_episode_info(
    dataset: lerobot_dataset.LeRobotDataset,
    success_field: str,
    default_success: str,
) -> tuple[dict[int, value_targets.EpisodeTargetInfo], dict[int, int]]:
    """Build episode_info and task_max_lengths from the dataset's metadata."""
    meta = dataset.meta
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

    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        ep_length = int(ep["length"])
        tasks = ep.get("tasks", [])
        task_name = tasks[0] if isinstance(tasks, list) and tasks else "unknown"
        task_index = task_name_to_index.get(task_name, 0)

        explicit_success = ep.get(success_field)
        if explicit_success is not None:
            ep_success = bool(explicit_success)
        else:
            ep_success = default_success.lower() in ("true", "1", "yes")

        episode_info[ep_idx] = value_targets.EpisodeTargetInfo(
            task_index=task_index,
            length=ep_length,
            success=ep_success,
        )
        task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)

    return episode_info, task_max_length


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def compute_dense_rewards_from_targets(
    targets: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Compute per-frame rewards as the difference in consecutive value targets.

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


def compute_n_step_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    n_step: int,
) -> np.ndarray:
    """N-step advantage: A(i) = sum_{k=0}^{n-1} r(i+k) + V(i+n) - V(i).

    Bootstrap with V(i+n) only if i+n is in the same episode and contiguous.
    """
    if n_step <= 0:
        raise ValueError("n_step must be > 0.")

    n = rewards.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # Precompute contiguity mask between consecutive frames.
    same_ep = np.empty(n, dtype=np.bool_)
    same_ep[:-1] = (episode_indices[:-1] == episode_indices[1:]) & (frame_indices[1:] == frame_indices[:-1] + 1)
    same_ep[-1] = False

    # Build cumulative contiguity: how many contiguous frames follow frame i within its episode.
    # max_reach[i] = max k such that frames i..i+k-1 are all contiguous and same episode.
    max_reach = np.zeros(n, dtype=np.int64)
    max_reach[-1] = 1
    for i in range(n - 2, -1, -1):
        if same_ep[i]:
            max_reach[i] = max_reach[i + 1] + 1
        else:
            max_reach[i] = 1

    # Prefix sum of rewards for sliding-window sums.
    cum_rewards = np.zeros(n + 1, dtype=np.float64)
    cum_rewards[1:] = np.cumsum(rewards.astype(np.float64))

    # Clamp effective step count to available contiguous frames.
    effective_n = np.minimum(n_step, max_reach).astype(np.int64)
    idx = np.arange(n, dtype=np.int64)

    reward_sum = (cum_rewards[idx + effective_n] - cum_rewards[idx]).astype(np.float32)

    # Bootstrap: use V(i+n_step) only if we had the full n_step contiguous frames AND i+n_step is still contiguous.
    can_bootstrap = (effective_n == n_step) & ((idx + n_step) < n)
    can_bootstrap &= (episode_indices[np.minimum(idx + n_step, n - 1)] == episode_indices[idx])
    can_bootstrap &= (frame_indices[np.minimum(idx + n_step, n - 1)] == frame_indices[idx] + n_step)

    bootstrap = np.where(can_bootstrap, values[np.minimum(idx + n_step, n - 1)], 0.0).astype(np.float32)

    return reward_sum + bootstrap - values.astype(np.float32)


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
    """Assign indicator=1 (positive) if advantage >= threshold, else 0."""
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
    """Write annotation columns back into the dataset's parquet files in-place."""
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


# ---------------------------------------------------------------------------
<<<<<<< Updated upstream
# Main pipeline
# ---------------------------------------------------------------------------

def run_value_inference(args: argparse.Namespace) -> None:
    config = _config.get_config(args.config_name)
    dataset_root = Path(args.dataset_root).resolve()

    logger.info("Config: %s", args.config_name)
    logger.info("Checkpoint: %s", args.checkpoint_dir)
    logger.info("Dataset root: %s", dataset_root)

    # ---- 1. Load model (data-parallel across all visible devices) ----
=======
# Dataset discovery
# ---------------------------------------------------------------------------

ANNOTATION_COLUMNS = ("predicted_value", "advantage", "is_positive")


def is_lerobot_dataset_root(path: Path) -> bool:
    """A directory is a LeRobot dataset iff it has ``meta/info.json`` and at
    least one ``data/chunk-*/episode_*.parquet``.
    """
    if not path.is_dir():
        return False
    if not (path / "meta" / "info.json").is_file():
        return False
    return any(path.glob("data/chunk-*/episode_*.parquet"))


def discover_dataset_roots(
    target: Path,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Path]:
    """Discover LeRobot dataset roots at or under ``target``.

    * If ``target`` itself is a dataset root, returns ``[target]``.
    * Otherwise scans direct children of ``target`` (one level deep) and
      returns any that look like dataset roots, sorted by path.

    ``include``/``exclude`` are lists of regex patterns matched against the
    directory *name* (not full path).
    """
    target = target.resolve()
    if not target.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {target}")

    if is_lerobot_dataset_root(target):
        return [target]

    roots: list[Path] = []
    for child in sorted(target.iterdir()):
        if child.is_dir() and is_lerobot_dataset_root(child):
            roots.append(child)

    if include:
        compiled = [re.compile(p) for p in include]
        roots = [r for r in roots if any(c.search(r.name) for c in compiled)]
    if exclude:
        compiled = [re.compile(p) for p in exclude]
        roots = [r for r in roots if not any(c.search(r.name) for c in compiled)]

    return roots


def already_annotated(dataset_root: Path) -> bool:
    """Return True if ``info.json`` already lists every annotation column."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return False
    try:
        with info_path.open() as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    features = info.get("features") or {}
    return all(col in features for col in ANNOTATION_COLUMNS)


# ---------------------------------------------------------------------------
# Model setup (loaded once, reused across all datasets)
# ---------------------------------------------------------------------------


class ModelHandle(NamedTuple):
    model: Any
    infer_value: Any  # jit'd function obs -> jax.Array
    mesh: jax.sharding.Mesh
    replicated: jax.sharding.NamedSharding
    data_parallel: jax.sharding.NamedSharding
    num_devices: int
    effective_bs: int


def setup_model(args: argparse.Namespace, config: _config.TrainConfig) -> ModelHandle:
    """Load checkpoint, build the JIT'd value-inference function, return a
    handle that can be reused across many datasets.
    """
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
    # ---- 2. Load raw dataset metadata ----
=======
    def _infer_value(obs):
        logits = model.forward_value_logits(obs)
        return model.expected_value_from_logits(logits)

    infer_value = jax.jit(_infer_value, out_shardings=replicated)

    effective_bs = args.batch_size
    if effective_bs % num_devices != 0:
        effective_bs = ((effective_bs + num_devices - 1) // num_devices) * num_devices
        logger.info(
            "Adjusted batch_size %d -> %d (divisible by %d devices)",
            args.batch_size,
            effective_bs,
            num_devices,
        )

    return ModelHandle(
        model=model,
        infer_value=infer_value,
        mesh=mesh,
        replicated=replicated,
        data_parallel=data_parallel,
        num_devices=num_devices,
        effective_bs=effective_bs,
    )


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------


def process_dataset(
    args: argparse.Namespace,
    config: _config.TrainConfig,
    dataset_root: Path,
    handle: ModelHandle,
    *,
    progress_prefix: str = "",
) -> None:
    """Run inference + advantage annotation for a single LeRobot dataset.

    Reuses the pre-loaded model from ``handle``. ``progress_prefix`` is shown
    in the tqdm bar (useful when iterating over many datasets).
    """
    logger.info("Dataset root: %s", dataset_root)

    # ---- 1. Load raw dataset metadata ----
>>>>>>> Stashed changes
    raw_dataset = lerobot_dataset.LeRobotDataset(
        repo_id=str(dataset_root),
        root=str(dataset_root),
    )
    raw_frames = raw_dataset.hf_dataset.with_format(None)
    frame_count = len(raw_frames)
    if frame_count == 0:
<<<<<<< Updated upstream
        raise ValueError("Dataset has no frames.")
=======
        raise ValueError(f"Dataset has no frames: {dataset_root}")
>>>>>>> Stashed changes

    absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64).reshape(-1)
    episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64).reshape(-1)
    task_indices = np.asarray(raw_frames["task_index"], dtype=np.int64).reshape(-1)

<<<<<<< Updated upstream
    logger.info("Dataset: %d frames, %d episodes", frame_count, len(np.unique(episode_indices)))
=======
    logger.info(
        "Dataset: %d frames, %d episodes",
        frame_count,
        len(np.unique(episode_indices)),
    )
>>>>>>> Stashed changes

    episode_info, task_max_lengths = build_episode_info(
        raw_dataset,
        success_field=args.success_field,
        default_success=args.default_success,
    )
<<<<<<< Updated upstream
    logger.info("Episode info: %d episodes, %d tasks", len(episode_info), len(task_max_lengths))

    # ---- 3. Batched value inference via data loader ----
    # batch_size must be divisible by num_devices for even data-parallel splitting.
    effective_bs = args.batch_size
    if effective_bs % num_devices != 0:
        effective_bs = ((effective_bs + num_devices - 1) // num_devices) * num_devices
        logger.info("Adjusted batch_size %d -> %d (divisible by %d devices)", args.batch_size, effective_bs, num_devices)

    infer_config = _replace_config_for_inference(config, dataset_root, effective_bs, checkpoint_dir=args.checkpoint_dir)
    data_loader = _data_loader.create_data_loader(
        infer_config,
        sharding=data_parallel,
=======
    logger.info(
        "Episode info: %d episodes, %d tasks",
        len(episode_info),
        len(task_max_lengths),
    )

    # ---- 2. Batched value inference via data loader ----
    infer_config = _replace_config_for_inference(
        config,
        dataset_root,
        handle.effective_bs,
        checkpoint_dir=args.checkpoint_dir,
    )
    data_loader = _data_loader.create_data_loader(
        infer_config,
        sharding=handle.data_parallel,
>>>>>>> Stashed changes
        shuffle=False,
    )

    max_abs_index = int(np.max(absolute_indices))
    prediction_lookup = np.zeros(max_abs_index + 1, dtype=np.float32)
    prediction_seen = np.zeros(max_abs_index + 1, dtype=np.bool_)

<<<<<<< Updated upstream
    def _infer_value(obs):
        logits = model.forward_value_logits(obs)
        return model.expected_value_from_logits(logits)

    infer_value = jax.jit(_infer_value, out_shardings=replicated)

    batch_count = (frame_count + effective_bs - 1) // effective_bs
    sample_cursor = 0

    logger.info("Starting value inference (%d batches, batch_size=%d, %d device(s))...", batch_count, effective_bs, num_devices)
    pbar = tqdm.tqdm(total=batch_count, desc="Value inference")
=======
    batch_count = (frame_count + handle.effective_bs - 1) // handle.effective_bs
    sample_cursor = 0

    desc = f"{progress_prefix}Value inference" if progress_prefix else "Value inference"
    logger.info(
        "Starting value inference (%d batches, batch_size=%d, %d device(s))...",
        batch_count,
        handle.effective_bs,
        handle.num_devices,
    )
    pbar = tqdm.tqdm(total=batch_count, desc=desc)
>>>>>>> Stashed changes
    batches_done = 0

    # Async pipeline: submit batch N to GPU while post-processing batch N-1 on CPU.
    prev_values = None
    prev_bs = 0
    prev_cursor = 0

    for obs, _actions in data_loader:
        bs = obs.state.shape[0]
<<<<<<< Updated upstream
        cur_values = infer_value(obs)  # non-blocking: returns a future/lazy JAX array

        # While GPU works on cur_values, collect results from previous batch
=======
        cur_values = handle.infer_value(obs)

>>>>>>> Stashed changes
        if prev_values is not None:
            val_np = np.asarray(prev_values).astype(np.float32).reshape(-1)
            batch_abs_indices = absolute_indices[prev_cursor : prev_cursor + prev_bs]
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

<<<<<<< Updated upstream
    # Collect the last batch
=======
>>>>>>> Stashed changes
    if prev_values is not None:
        val_np = np.asarray(prev_values).astype(np.float32).reshape(-1)
        batch_abs_indices = absolute_indices[prev_cursor : prev_cursor + prev_bs]
        prediction_lookup[batch_abs_indices] = val_np[: len(batch_abs_indices)]
        prediction_seen[batch_abs_indices] = True
        pbar.update(1)

    pbar.close()

    missing = ~prediction_seen[absolute_indices]
    if np.any(missing):
        missing_count = int(np.sum(missing))
        logger.warning("Missing predictions for %d frames, filling with 0.", missing_count)

    predicted_values = prediction_lookup[absolute_indices]
    logger.info(
        "Value stats | min=%.6f max=%.6f mean=%.6f std=%.6f",
        float(np.min(predicted_values)),
        float(np.max(predicted_values)),
        float(np.mean(predicted_values)),
        float(np.std(predicted_values)),
    )

<<<<<<< Updated upstream
    # ---- 4. Compute value targets, rewards, advantages ----
    # value_tgts = value_targets.compute_normalized_value_targets(
    #     episode_indices=episode_indices,
    #     frame_indices=frame_indices,
    #     episode_info=episode_info,
    #     task_max_lengths=task_max_lengths,
    #     c_fail_coef=args.c_fail_coef,
    #     clip_min=config.value_clip_min,
    #     clip_max=config.value_clip_max,
    # )
=======
    # ---- 3. Compute value targets, rewards, advantages ----
>>>>>>> Stashed changes
    value_tgts = load_existing_value_targets(raw_frames, "value_target")

    rewards = compute_dense_rewards_from_targets(value_tgts, episode_indices, frame_indices)
    advantages = compute_n_step_advantages(
        rewards=rewards,
        values=predicted_values,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        n_step=args.n_step,
    )

    thresholds = compute_task_thresholds(task_indices, advantages, args.positive_ratio)
    indicators = binarize_advantages(task_indices, advantages, thresholds)

    positive_ratio_observed = float(np.mean(indicators.astype(np.float32)))
    logger.info(
        "ACP stats | n_step=%d positive_ratio_target=%.4f positive_ratio_observed=%.4f",
        args.n_step,
        args.positive_ratio,
        positive_ratio_observed,
    )
    for task_idx, threshold in sorted(thresholds.items()):
        logger.info("  task %d threshold=%.6f", task_idx, threshold)

<<<<<<< Updated upstream
    # ---- 5. Write back to parquet ----
=======
    # ---- 4. Write back to parquet ----
>>>>>>> Stashed changes
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


<<<<<<< Updated upstream
=======
# ---------------------------------------------------------------------------
# Orchestrator (single dataset OR batch)
# ---------------------------------------------------------------------------


def run_value_inference(args: argparse.Namespace) -> None:
    config = _config.get_config(args.config_name)
    target_path = Path(args.dataset_root).resolve()

    logger.info("Config: %s", args.config_name)
    logger.info("Checkpoint: %s", args.checkpoint_dir)
    logger.info("Target path: %s", target_path)

    dataset_roots = discover_dataset_roots(
        target_path,
        include=args.include,
        exclude=args.exclude,
    )
    if not dataset_roots:
        raise RuntimeError(
            f"No LeRobot datasets found at or under {target_path} "
            "(expected meta/info.json + data/chunk-*/episode_*.parquet)."
        )

    if args.skip_existing:
        kept: list[Path] = []
        for root in dataset_roots:
            if already_annotated(root):
                logger.info("Skipping (already annotated): %s", root)
            else:
                kept.append(root)
        dataset_roots = kept
        if not dataset_roots:
            logger.info("All discovered datasets are already annotated; nothing to do.")
            return

    logger.info("Will process %d dataset(s):", len(dataset_roots))
    for r in dataset_roots:
        logger.info("  - %s", r)

    if args.list_only:
        return

    handle = setup_model(args, config)

    successes: list[Path] = []
    failures: list[tuple[Path, BaseException]] = []
    overall_t0 = time.monotonic()

    for i, dataset_root in enumerate(dataset_roots, 1):
        prefix = f"[{i}/{len(dataset_roots)}] "
        logger.info("%s===== Processing %s =====", prefix, dataset_root)
        t0 = time.monotonic()
        try:
            process_dataset(
                args,
                config,
                dataset_root,
                handle,
                progress_prefix=prefix,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.exception(
                "%sFailed after %.1fs: %s -> %r",
                prefix,
                elapsed,
                dataset_root,
                exc,
            )
            failures.append((dataset_root, exc))
            if not args.continue_on_error:
                raise
        else:
            elapsed = time.monotonic() - t0
            logger.info("%sFinished %s in %.1fs", prefix, dataset_root, elapsed)
            successes.append(dataset_root)

    total_elapsed = time.monotonic() - overall_t0
    logger.info(
        "Summary: %d/%d succeeded in %.1fs total",
        len(successes),
        len(dataset_roots),
        total_elapsed,
    )
    if failures:
        logger.error("Failed datasets:")
        for r, e in failures:
            logger.error("  - %s: %r", r, e)
        raise SystemExit(1)


>>>>>>> Stashed changes
def _replace_config_for_inference(
    config: _config.TrainConfig,
    dataset_root: Path,
    batch_size: int,
    checkpoint_dir: str | None = None,
) -> _config.TrainConfig:
    """Create a modified config that points to the real dataset for inference.

    When checkpoint_dir is given, norm stats are loaded from the checkpoint's
    ``assets/`` directory instead of from the (possibly different) eval dataset.
    """
<<<<<<< Updated upstream
    import dataclasses

=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    parser = argparse.ArgumentParser(description="Value inference + advantage annotation for LeRobot datasets")
    parser.add_argument("--config-name", type=str, required=True, help="Name of the training config (e.g. pi0_value_umi)")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to checkpoint step dir (contains params/)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to LeRobot dataset root")
=======
    parser = argparse.ArgumentParser(
        description="Value inference + advantage annotation for LeRobot datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="Name of the training config (e.g. pi0_value_umi)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoint step dir (contains params/)",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help=(
            "Either (a) the path to a single LeRobot dataset root or (b) a "
            "parent directory whose immediate subdirectories are LeRobot "
            "datasets to process in batch."
        ),
    )
>>>>>>> Stashed changes
    parser.add_argument("--batch-size", type=int, default=72, help="Inference batch size")
    parser.add_argument("--n-step", type=int, default=50, help="N-step for advantage computation")
    parser.add_argument("--positive-ratio", type=float, default=0.4, help="Target ratio of positive indicators per task")
    parser.add_argument("--c-fail-coef", type=float, default=1.0, help="Failure penalty coefficient for value targets")
    parser.add_argument("--success-field", type=str, default="success", help="Field name for success label in episodes.jsonl")
    parser.add_argument("--default-success", type=str, default="true", help="Default success label if field is missing (true/false)")
<<<<<<< Updated upstream
=======

    # Batch-mode controls
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="REGEX",
        help=(
            "Regex applied to dataset directory NAME; only matching datasets "
            "are processed. Repeatable; ORed across patterns."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="REGEX",
        help="Regex applied to dataset directory NAME; matching datasets are dropped. Repeatable.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets whose meta/info.json already declares predicted_value/advantage/is_positive.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If a single dataset fails, log and proceed to the next (instead of aborting).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print which datasets would be processed, then exit.",
    )
>>>>>>> Stashed changes
    return parser.parse_args()


if __name__ == "__main__":
    run_value_inference(parse_args())
