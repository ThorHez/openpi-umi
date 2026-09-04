#!/usr/bin/env python3
"""Adapt the old real-ShellGame MEM relation classifier to cup_0903.

The run has four deliberate constraints:

* initialize the complete model from ``/4999/params``;
* cache frozen SigLIP features once and optimize only
  ``swap_relation_classifier``;
* draw every batch as 24 new-domain episodes (8 per final cup) plus 8 old;
* select/early-stop on free-running new-domain cup tracking while enforcing
  an old-domain regression gate.

The final held-out new-domain test split is never consulted during checkpoint
selection.  One full OpenPI checkpoint is written at the selected step.
"""

# Environment variables must be set before importing JAX.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import functools
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

_WORKSPACE = Path("/data2/hzl_workspace_for_pi_mem")
_CACHE_HOME = _WORKSPACE / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_HOME))
os.environ.setdefault("OPENPI_DATA_HOME", str(_CACHE_HOME / "openpi"))
os.environ.setdefault("HF_HOME", str(_CACHE_HOME / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_HOME / "huggingface" / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE_HOME / "huggingface" / "transformers"))
os.environ.setdefault("TMPDIR", str(_CACHE_HOME / "tmp"))

import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
import pyarrow.parquet as pq
import zarr

from openpi.models import siglip_mem_semantic as memory_core
from openpi.shared import normalize as _normalize
import openpi.training.checkpoints as _checkpoints
from openpi.training.mem.recipes import shellgame_real_memory_adapt as _recipe
import openpi.training.sharding as sharding

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from mem.train_semantic_memory import init_train_state
from train import init_logging

OLD_DATASET = _WORKSPACE / "openpi-umi/data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10"
OLD_LABELS = _WORKSPACE / "labels_merged_306_degap.jsonl"
NEW_ZARR = _WORKSPACE / "cup_0903/replay_buffer.zarr"
NEW_LABELS = _WORKSPACE / "cup_0903/labels.jsonl"
HISTORY_FRAMES = 241
SWAP_FRAME_INDICES = tuple(
    tuple(range(1 + stage * 80, 1 + (stage + 1) * 80)) for stage in range(3)
)
IMAGE_KEY = "observation.left_wrist_0_rgb_0"
NUM_CLASSES = 3
SOURCE_NAMES = ("old306", "cup_0903")
DEFAULT_EXP_NAME = "cup0903_new75_old25_relation_only_lr1e5_b32_seed42_v1"
DEFAULT_CACHE_DIR = _WORKSPACE / "openpi-umi/artifacts/shellgame_real_memory_adapt_4999_features_fp16"


@dataclasses.dataclass(frozen=True)
class EpisodeLabels:
    initial: int
    relations: tuple[int, int, int]
    stages: tuple[int, int, int]

    @property
    def final(self) -> int:
        return self.stages[-1]


@dataclasses.dataclass(frozen=True)
class EpisodeRef:
    source: int
    episode_id: int
    labels: EpisodeLabels


@dataclasses.dataclass(frozen=True)
class _CheckpointDataConfig:
    norm_stats: dict[str, _normalize.NormStats]
    asset_id: str = "."


class _CheckpointAssetLoader:
    def __init__(self):
        self._data_config = _CheckpointDataConfig(_normalize.load(OLD_DATASET))

    def data_config(self):
        return self._data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--num-train-steps", type=int, default=1_500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-step", type=int, default=500)
    parser.add_argument("--old-final-accuracy-gate", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.feature_batch_size <= 0:
        parser.error("--feature-batch-size must be positive")
    if args.num_train_steps <= 0 or args.eval_interval <= 0:
        parser.error("training steps and eval interval must be positive")
    if args.early_stop_patience <= 0:
        parser.error("--early-stop-patience must be positive")
    return args


def _parse_labels(path: Path) -> list[EpisodeLabels]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [int(row["episode_id"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: episode_id must be contiguous and ordered")
    parsed = []
    canonical_pairs = ((0, 1), (0, 2), (1, 2))
    for row in rows:
        initial = int(row["initial_cup"])
        cup = initial
        relations: list[int] = []
        stages: list[int] = []
        moves = row["moves"]
        if len(moves) != 3 or int(row.get("n_observe_frames", HISTORY_FRAMES)) != HISTORY_FRAMES:
            raise ValueError(f"{path}: episode {row['episode_id']} has an invalid 241-frame event contract")
        for move in moves:
            pair = tuple(sorted(int(value) for value in move))
            relations.append(canonical_pairs.index(pair))
            if cup == pair[0]:
                cup = pair[1]
            elif cup == pair[1]:
                cup = pair[0]
            stages.append(cup)
        if cup != int(row["final_cup"]):
            raise ValueError(f"{path}: episode {row['episode_id']} does not roll out to final_cup")
        parsed.append(EpisodeLabels(initial, tuple(relations), tuple(stages)))
    return parsed


def _stratified_new_split(labels: list[EpisodeLabels], seed: int) -> dict[str, list[int]]:
    """Create an exact 70/15/15 class-stratified episode split."""
    rng = np.random.default_rng(seed)
    split = {"train": [], "validation": [], "test": []}
    for cup in range(NUM_CLASSES):
        ids = np.asarray([index for index, label in enumerate(labels) if label.final == cup], dtype=np.int64)
        ids = rng.permutation(ids)
        num_validation = round(len(ids) * 0.15)
        num_test = round(len(ids) * 0.15)
        split["validation"].extend(ids[:num_validation].tolist())
        split["test"].extend(ids[num_validation : num_validation + num_test].tolist())
        split["train"].extend(ids[num_validation + num_test :].tolist())
    split = {key: sorted(value) for key, value in split.items()}
    if tuple(len(split[key]) for key in ("train", "validation", "test")) != (70, 15, 15):
        raise AssertionError(f"Unexpected new-domain split sizes: { {key: len(value) for key, value in split.items()} }")
    return split


def _old_split(labels: list[EpisodeLabels]) -> dict[str, list[int]]:
    audit_path = OLD_DATASET / "conversion_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validation = sorted(int(value) for value in audit["validation_episode_ids"])
    validation_set = set(validation)
    train = [index for index in range(len(labels)) if index not in validation_set]
    if len(train) != 275 or len(validation) != 31:
        raise ValueError(f"Unexpected old split sizes: train={len(train)}, validation={len(validation)}")
    return {"train": train, "validation": validation}


def _split_summary(ids: list[int], labels: list[EpisodeLabels]) -> dict[str, Any]:
    finals = np.asarray([labels[index].final for index in ids], dtype=np.int32)
    relations = np.asarray([labels[index].relations for index in ids], dtype=np.int32).reshape(-1)
    return {
        "episodes": len(ids),
        "final_counts": np.bincount(finals, minlength=NUM_CLASSES).tolist(),
        "relation_counts": np.bincount(relations, minlength=NUM_CLASSES).tolist(),
    }


def build_manifest(seed: int) -> tuple[list[EpisodeRef], dict[str, Any]]:
    old_labels = _parse_labels(OLD_LABELS)
    new_labels = _parse_labels(NEW_LABELS)
    if len(old_labels) != 306 or len(new_labels) != 100:
        raise ValueError(f"Expected 306 old and 100 new episodes, got {len(old_labels)} and {len(new_labels)}")
    old_split = _old_split(old_labels)
    new_split = _stratified_new_split(new_labels, seed)
    refs = [EpisodeRef(0, index, label) for index, label in enumerate(old_labels)]
    refs.extend(EpisodeRef(1, index, label) for index, label in enumerate(new_labels))
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "global_episode_contract": "old ids 0..305; new ids 306..405",
        "old": {
            "labels": str(OLD_LABELS),
            "dataset": str(OLD_DATASET),
            "train_ids": old_split["train"],
            "validation_ids": old_split["validation"],
            "train_summary": _split_summary(old_split["train"], old_labels),
            "validation_summary": _split_summary(old_split["validation"], old_labels),
        },
        "new": {
            "labels": str(NEW_LABELS),
            "zarr": str(NEW_ZARR),
            "train_ids": new_split["train"],
            "validation_ids": new_split["validation"],
            "test_ids": new_split["test"],
            "train_summary": _split_summary(new_split["train"], new_labels),
            "validation_summary": _split_summary(new_split["validation"], new_labels),
            "test_summary": _split_summary(new_split["test"], new_labels),
        },
    }
    return refs, manifest


def _old_episode_path(episode_id: int) -> Path:
    candidates = sorted(OLD_DATASET.glob(f"data/chunk-*/episode_{episode_id:06d}.parquet"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one old parquet for episode {episode_id}, got {candidates}")
    return candidates[0]


def _decode_old_history(episode_id: int) -> np.ndarray:
    table = pq.read_table(_old_episode_path(episode_id), columns=[IMAGE_KEY]).slice(0, HISTORY_FRAMES)
    cells = table.column(IMAGE_KEY).to_pylist()
    if len(cells) != HISTORY_FRAMES:
        raise ValueError(f"Old episode {episode_id} has only {len(cells)} history frames")
    images = [np.asarray(Image.open(BytesIO(cell["bytes"])).convert("RGB"), dtype=np.uint8) for cell in cells]
    result = np.stack(images)
    if result.shape != (HISTORY_FRAMES, 224, 224, 3):
        raise ValueError(f"Unexpected old image shape {result.shape}")
    return result


class HistoryReader:
    def __init__(self):
        replay = zarr.open(NEW_ZARR, mode="r")
        self.camera = replay["data"]["camera0_rgb"]
        ends = np.asarray(replay["meta"]["episode_ends"][:], dtype=np.int64)
        self.starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
        self.ends = ends
        if len(ends) != 100 or self.camera.shape[0] != int(ends[-1]):
            raise ValueError("New Zarr episode metadata is inconsistent")
        if np.any(ends - self.starts < HISTORY_FRAMES):
            raise ValueError("New Zarr contains an episode shorter than 241 frames")

    def read(self, ref: EpisodeRef) -> np.ndarray:
        if ref.source == 0:
            return _decode_old_history(ref.episode_id)
        start = int(self.starts[ref.episode_id])
        images = np.asarray(self.camera[start : start + HISTORY_FRAMES])
        if np.issubdtype(images.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(images)) <= 1.5 else 1.0
            images = np.clip(np.rint(images * scale), 0, 255).astype(np.uint8)
        elif images.dtype != np.uint8:
            images = np.clip(images, 0, 255).astype(np.uint8)
        if images.shape != (HISTORY_FRAMES, 224, 224, 3):
            raise ValueError(f"Unexpected new image shape {images.shape}")
        return np.ascontiguousarray(images)


def _feature_cache_contract(refs: list[EpisodeRef], config) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_checkpoint": _recipe.SOURCE_MEMORY_CHECKPOINT,
        "model_config": _recipe.CONFIG_NAME,
        "episodes": len(refs),
        "history_frames": HISTORY_FRAMES,
        "swap_frame_indices": [list(stage) for stage in SWAP_FRAME_INDICES],
        "feature_shape": [len(refs), 3, 80, 64, 1152],
        "feature_dtype": "float16",
        "old_labels_mtime_ns": OLD_LABELS.stat().st_mtime_ns,
        "new_labels_mtime_ns": NEW_LABELS.stat().st_mtime_ns,
        "new_zarr_group_mtime_ns": (NEW_ZARR / ".zgroup").stat().st_mtime_ns,
        "model_history_frames": config.model.history_frames,
    }


def _extract_step(state, images):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    normalized = images.astype(jnp.float32) / 127.5 - 1.0
    _, initial_encoder_out = model.PaliGemma.img(normalized[:, :1], train=False)
    initial_logits = model.HistoryFrame0InitialCupClassifier(initial_encoder_out["encoded"])
    _, history_encoder_out = model.PaliGemma.img(normalized, train=False)
    patches = history_encoder_out["with_posemb"][:, :HISTORY_FRAMES]
    pooled = memory_core.pool_fixed_grid(patches, pool_factor=2)
    clips = jnp.stack(
        [pooled[:, jnp.asarray(indices)] for indices in SWAP_FRAME_INDICES],
        axis=1,
    )
    return clips, initial_logits


def prepare_feature_cache(
    args: argparse.Namespace,
    refs: list[EpisodeRef],
    config,
    state,
    state_sharding,
    mesh,
    data_sharding,
) -> tuple[np.memmap, np.ndarray]:
    cache_dir = args.feature_cache_dir.resolve()
    feature_path = cache_dir / "swap_clips_fp16.npy"
    initial_path = cache_dir / "initial_logits.npy"
    metadata_path = cache_dir / "metadata.json"
    contract = _feature_cache_contract(refs, config)
    if not args.rebuild_feature_cache and feature_path.is_file() and initial_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata == contract:
            features = np.load(feature_path, mmap_mode="r")
            initial_logits = np.load(initial_path)
            if list(features.shape) == contract["feature_shape"] and initial_logits.shape == (len(refs), 3):
                logging.info("Reusing frozen feature cache: %s", feature_path)
                return features, initial_logits
        logging.warning("Feature cache contract changed; rebuilding %s", cache_dir)

    if cache_dir.exists() and args.rebuild_feature_cache:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_dir).free
    required_bytes = int(np.prod(contract["feature_shape"]) * np.dtype(np.float16).itemsize * 1.1)
    if free_bytes < required_bytes:
        raise OSError(
            f"Feature cache needs about {required_bytes / 1024**3:.1f} GiB, "
            f"only {free_bytes / 1024**3:.1f} GiB free"
        )

    partial_feature = cache_dir / "swap_clips_fp16.partial.npy"
    partial_initial = cache_dir / "initial_logits.partial.npy"
    features = np.lib.format.open_memmap(
        partial_feature,
        mode="w+",
        dtype=np.float16,
        shape=tuple(contract["feature_shape"]),
    )
    initial_logits = np.lib.format.open_memmap(
        partial_initial,
        mode="w+",
        dtype=np.float32,
        shape=(len(refs), 3),
    )
    reader = HistoryReader()
    pextract = jax.jit(
        _extract_step,
        in_shardings=(state_sharding, data_sharding),
        out_shardings=(data_sharding, data_sharding),
    )
    batch_size = args.feature_batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"feature batch {batch_size} must be divisible by visible device count {jax.device_count()}"
        )
    started = time.time()
    for start in range(0, len(refs), batch_size):
        stop = min(start + batch_size, len(refs))
        batch_refs = refs[start:stop]
        images = [reader.read(ref) for ref in batch_refs]
        valid = len(images)
        while len(images) < batch_size:
            images.append(images[-1])
        image_batch = jax.make_array_from_process_local_data(data_sharding, np.stack(images))
        with sharding.set_mesh(mesh):
            clip_batch, initial_batch = pextract(state, image_batch)
        clip_batch, initial_batch = jax.device_get((clip_batch, initial_batch))
        features[start:stop] = np.asarray(clip_batch[:valid], dtype=np.float16)
        initial_logits[start:stop] = np.asarray(initial_batch[:valid], dtype=np.float32)
        features.flush()
        initial_logits.flush()
        elapsed = time.time() - started
        logging.info(
            "Feature cache %d/%d episodes (%.1f min elapsed, %.1f min ETA)",
            stop,
            len(refs),
            elapsed / 60,
            (elapsed / stop * (len(refs) - stop)) / 60,
        )
    del features, initial_logits
    partial_feature.replace(feature_path)
    partial_initial.replace(initial_path)
    metadata_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return np.load(feature_path, mmap_mode="r"), np.load(initial_path)


def _labels_arrays(refs: list[EpisodeRef], initial_logits: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "initial": np.asarray([ref.labels.initial for ref in refs], dtype=np.int32),
        "initial_pred": np.argmax(initial_logits, axis=-1).astype(np.int32),
        "relations": np.asarray([ref.labels.relations for ref in refs], dtype=np.int32),
        "stages": np.asarray([ref.labels.stages for ref in refs], dtype=np.int32),
        "final": np.asarray([ref.labels.final for ref in refs], dtype=np.int32),
        "source": np.asarray([ref.source for ref in refs], dtype=np.int32),
    }


def _global_ids(source: int, episode_ids: list[int]) -> list[int]:
    offset = 0 if source == 0 else 306
    return [offset + episode_id for episode_id in episode_ids]


def sample_train_indices(manifest: dict[str, Any], labels: dict[str, np.ndarray], rng) -> np.ndarray:
    old_ids = np.asarray(_global_ids(0, manifest["old"]["train_ids"]), dtype=np.int64)
    new_ids = np.asarray(_global_ids(1, manifest["new"]["train_ids"]), dtype=np.int64)
    chosen = [rng.choice(old_ids, size=8, replace=True)]
    for cup in range(NUM_CLASSES):
        candidates = new_ids[labels["final"][new_ids] == cup]
        chosen.append(rng.choice(candidates, size=8, replace=True))
    result = np.concatenate(chosen)
    rng.shuffle(result)
    return result


def _batch_from_indices(
    features: np.memmap,
    labels: dict[str, np.ndarray],
    indices: np.ndarray,
    data_sharding,
    *,
    valid_count: int | None = None,
) -> dict[str, jax.Array]:
    if valid_count is None:
        valid_count = len(indices)
    local = {
        "clips": np.asarray(features[indices], dtype=np.float16),
        "initial": labels["initial"][indices],
        "initial_pred": labels["initial_pred"][indices],
        "relations": labels["relations"][indices],
        "stages": labels["stages"][indices],
        "valid": (np.arange(len(indices)) < valid_count).astype(np.float32),
    }
    return jax.tree.map(lambda value: jax.make_array_from_process_local_data(data_sharding, value), local)


def train_step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(memory_model):
        logits = memory_model.HistoryThreeSwapVisualRelationMemoryTracker(
            batch["clips"].astype(jnp.bfloat16),
            batch["initial"],
            batch["relations"],
            preselected_pooled_clips=True,
            return_relation_logits_only=True,
        )
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits.astype(jnp.float32), batch["relations"]
            )
        )
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch["relations"])
        return loss, {"loss": loss, "relation_accuracy": accuracy}

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
    trainable_params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, trainable_params)
    updated_params = optax.apply_updates(trainable_params, updates)
    nnx.update(model, updated_params)
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=nnx.state(model),
        opt_state=new_opt_state,
    )
    return new_state, {
        **info,
        "objective_loss": loss,
        "grad_norm": optax.global_norm(grads),
        "learning_rate": config.lr_schedule.create()(state.step),
    }


def eval_step(state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    _, stage_logits, _, relation_logits, _ = model.HistoryThreeSwapVisualRelationMemoryTracker(
        batch["clips"].astype(jnp.bfloat16),
        batch["initial_pred"],
        None,
        preselected_pooled_clips=True,
    )
    relation_pred = jnp.argmax(relation_logits, axis=-1)
    stage_pred = jnp.argmax(stage_logits, axis=-1)
    final_pred = stage_pred[:, -1]
    valid = batch["valid"].astype(jnp.int32)
    relation_valid = jnp.repeat(valid[:, None], 3, axis=1)
    relation_confusion = jnp.bincount(
        (batch["relations"] * 3 + relation_pred).reshape(-1),
        weights=relation_valid.reshape(-1),
        length=9,
    ).reshape(3, 3)
    final_confusion = jnp.bincount(
        batch["stages"][:, -1] * 3 + final_pred,
        weights=valid,
        length=9,
    ).reshape(3, 3)
    stage_correct = jnp.sum((stage_pred == batch["stages"]) * valid[:, None], axis=0)
    initial_confusion = jnp.bincount(
        batch["initial"] * 3 + batch["initial_pred"],
        weights=valid,
        length=9,
    ).reshape(3, 3)
    return {
        "relation_confusion": relation_confusion,
        "final_confusion": final_confusion,
        "initial_confusion": initial_confusion,
        "stage_correct": stage_correct,
        "episodes": jnp.sum(valid),
    }


def _metrics_from_counts(counts: dict[str, np.ndarray]) -> dict[str, Any]:
    final_confusion = np.asarray(counts["final_confusion"], dtype=np.int64)
    relation_confusion = np.asarray(counts["relation_confusion"], dtype=np.int64)
    initial_confusion = np.asarray(counts["initial_confusion"], dtype=np.int64)
    episodes = int(np.asarray(counts["episodes"]))
    stage_correct = np.asarray(counts["stage_correct"], dtype=np.int64)

    def recalls(confusion):
        return [
            float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None
            for i in range(NUM_CLASSES)
        ]

    return {
        "episodes": episodes,
        "final_accuracy": float(np.trace(final_confusion) / episodes),
        "final_confusion_gt_rows_pred_cols": final_confusion.tolist(),
        "final_per_cup_accuracy": recalls(final_confusion),
        "relation_accuracy": float(np.trace(relation_confusion) / relation_confusion.sum()),
        "relation_confusion_gt_rows_pred_cols": relation_confusion.tolist(),
        "relation_per_type_accuracy": recalls(relation_confusion),
        "stage_accuracy": (stage_correct / episodes).astype(float).tolist(),
        "initial_accuracy": float(np.trace(initial_confusion) / episodes),
    }


def evaluate_indices(peval, state, features, labels, indices, data_sharding, batch_size):
    totals = None
    for start in range(0, len(indices), batch_size):
        current = list(indices[start : start + batch_size])
        valid_count = len(current)
        while len(current) < batch_size:
            current.append(current[-1])
        batch = _batch_from_indices(
            features,
            labels,
            np.asarray(current, dtype=np.int64),
            data_sharding,
            valid_count=valid_count,
        )
        counts = jax.device_get(peval(state, batch))
        totals = counts if totals is None else jax.tree.map(np.add, totals, counts)
    return _metrics_from_counts(totals)


def _selection_key(new_metrics: dict[str, Any], old_metrics: dict[str, Any], old_gate: float):
    eligible = old_metrics["final_accuracy"] >= old_gate
    per_cup = [value for value in new_metrics["final_per_cup_accuracy"] if value is not None]
    return (
        int(eligible),
        new_metrics["final_accuracy"],
        new_metrics["relation_accuracy"],
        min(per_cup),
        old_metrics["final_accuracy"],
    )


def _replace_trainable(state, trainable_params, trainable_sharding):
    trainable_params = jax.device_put(trainable_params, trainable_sharding)
    model = nnx.merge(state.model_def, state.params)
    nnx.update(model, trainable_params)
    return dataclasses.replace(state, params=nnx.state(model))


def main() -> None:
    args = parse_args()
    init_logging()
    refs, manifest = build_manifest(args.seed)
    logging.info("Split contract:\n%s", json.dumps({
        "old_train": manifest["old"]["train_summary"],
        "old_validation": manifest["old"]["validation_summary"],
        "new_train": manifest["new"]["train_summary"],
        "new_validation": manifest["new"]["validation_summary"],
        "new_test": manifest["new"]["test_summary"],
    }, indent=2))
    if args.validate_only:
        print(json.dumps(manifest, indent=2))
        return

    config = dataclasses.replace(
        _recipe.make_train_config(),
        exp_name=args.exp_name,
        seed=args.seed,
        num_train_steps=args.num_train_steps,
        eval_interval=args.eval_interval,
        overwrite=args.overwrite,
    )
    if config.batch_size != 32:
        raise ValueError("This experiment requires global batch_size=32")
    if jax.device_count() != 8:
        raise ValueError(f"Expected exactly 8 visible GPUs, got {jax.device_count()}")

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state, state_sharding = init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    trainable_count = 0
    for leaf in jax.tree.leaves(state.params.filter(config.trainable_filter)):
        value = leaf.value if hasattr(leaf, "value") else leaf
        if hasattr(value, "shape"):
            trainable_count += int(np.prod(value.shape))
    if trainable_count <= 0:
        raise ValueError("Freeze filter selected zero trainable parameters")
    logging.info("Verified relation-classifier-only trainable parameters: %d (%.2fM)", trainable_count, trainable_count / 1e6)

    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=False,
    )
    features, initial_logits = prepare_feature_cache(
        args, refs, config, state, state_sharding, mesh, data_sharding
    )
    labels = _labels_arrays(refs, initial_logits)
    output_dir = config.checkpoint_dir
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    ptrain = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    peval = jax.jit(
        eval_step,
        in_shardings=(state_sharding, data_sharding),
        out_shardings=replicated,
    )
    old_val_ids = _global_ids(0, manifest["old"]["validation_ids"])
    new_val_ids = _global_ids(1, manifest["new"]["validation_ids"])
    new_test_ids = _global_ids(1, manifest["new"]["test_ids"])
    all_new_ids = list(range(306, 406))
    eval_batch_size = config.batch_size

    history: list[dict[str, Any]] = []

    def evaluate(step: int):
        new_metrics = evaluate_indices(
            peval, state, features, labels, new_val_ids, data_sharding, eval_batch_size
        )
        old_metrics = evaluate_indices(
            peval, state, features, labels, old_val_ids, data_sharding, eval_batch_size
        )
        record = {
            "step": step,
            "new_validation": new_metrics,
            "old_validation": old_metrics,
            "selection_key": list(_selection_key(new_metrics, old_metrics, args.old_final_accuracy_gate)),
        }
        history.append(record)
        (output_dir / "metrics_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        logging.info(
            "Step %d [free-run eval] new final=%.4f relation=%.4f per_cup=%s; "
            "old final=%.4f relation=%.4f",
            step,
            new_metrics["final_accuracy"],
            new_metrics["relation_accuracy"],
            new_metrics["final_per_cup_accuracy"],
            old_metrics["final_accuracy"],
            old_metrics["relation_accuracy"],
        )
        return record

    baseline = evaluate(0)
    best_key = tuple(baseline["selection_key"])
    best_step = 0
    best_trainable = jax.device_get(state.params.filter(config.trainable_filter))
    evaluations_without_improvement = 0
    np_rng = np.random.default_rng(config.seed)
    infos = []
    stopped_early = False
    started = time.time()
    for step in range(config.num_train_steps):
        indices = sample_train_indices(manifest, labels, np_rng)
        batch = _batch_from_indices(features, labels, indices, data_sharding)
        with sharding.set_mesh(mesh):
            state, info = ptrain(train_rng, state, batch)
        infos.append(info)
        completed_step = step + 1
        if completed_step % config.log_interval == 0:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            logging.info(
                "Step %d [train] %s",
                completed_step,
                ", ".join(f"{key}={float(value):.6f}" for key, value in reduced.items()),
            )
            infos = []
        if completed_step % config.eval_interval == 0 or completed_step == config.num_train_steps:
            record = evaluate(completed_step)
            key = tuple(record["selection_key"])
            if key > best_key:
                best_key = key
                best_step = completed_step
                best_trainable = jax.device_get(state.params.filter(config.trainable_filter))
                evaluations_without_improvement = 0
                logging.info("New best relation head at step %d: key=%s", best_step, best_key)
            else:
                evaluations_without_improvement += 1
            if (
                completed_step >= args.early_stop_min_step
                and evaluations_without_improvement >= args.early_stop_patience
            ):
                logging.info(
                    "Early stopping at step %d after %d evaluations without free-run improvement",
                    completed_step,
                    evaluations_without_improvement,
                )
                stopped_early = True
                break

    state = _replace_trainable(
        state,
        best_trainable,
        state_sharding.params.filter(config.trainable_filter),
    )
    best_new_val = evaluate_indices(peval, state, features, labels, new_val_ids, data_sharding, eval_batch_size)
    best_old_val = evaluate_indices(peval, state, features, labels, old_val_ids, data_sharding, eval_batch_size)
    best_new_test = evaluate_indices(peval, state, features, labels, new_test_ids, data_sharding, eval_batch_size)
    best_new_all = evaluate_indices(peval, state, features, labels, all_new_ids, data_sharding, eval_batch_size)

    _checkpoints.save_state(checkpoint_manager, state, _CheckpointAssetLoader(), best_step)
    checkpoint_manager.wait_until_finished()
    summary = {
        "config": config.name,
        "exp_name": config.exp_name,
        "source_checkpoint": _recipe.SOURCE_MEMORY_CHECKPOINT,
        "selected_step": best_step,
        "stopped_early": stopped_early,
        "elapsed_hours": (time.time() - started) / 3600,
        "batch_contract": "32 total = 24 new (8 left, 8 middle, 8 right) + 8 old",
        "trainable_contract": "swap_relation_classifier only",
        "old_final_accuracy_gate": args.old_final_accuracy_gate,
        "new_validation": best_new_val,
        "old_validation": best_old_val,
        "new_heldout_test": best_new_test,
        "new_all_100_diagnostic": best_new_all,
        "checkpoint": str(output_dir / str(best_step)),
        "feature_cache": str(args.feature_cache_dir.resolve()),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("TRAINING_COMPLETE %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
