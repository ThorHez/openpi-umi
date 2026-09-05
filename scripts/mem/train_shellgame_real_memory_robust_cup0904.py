#!/usr/bin/env python3
"""Robustly adapt the real ShellGame relation head to cup_0904.

This deliberately keeps the full 224x224 image.  It reuses the frozen feature
cache for old306/cup_0903, caches deterministic temporally-consistent views of
cup_0904, and trains only ``swap_relation_classifier`` with supervised
relation CE plus augmented-view consistency.
"""

# Environment variables must be set before importing JAX.
# ruff: noqa: E402, SLF001

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

_WORKSPACE = Path("/data2/hzl_workspace_for_pi_mem")
_REPO = _WORKSPACE / "openpi-umi"
_CACHE_HOME = _WORKSPACE / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_HOME))
os.environ.setdefault("OPENPI_DATA_HOME", str(_CACHE_HOME / "openpi"))
os.environ.setdefault("HF_HOME", str(_CACHE_HOME / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_HOME / "huggingface" / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE_HOME / "huggingface" / "transformers"))
os.environ.setdefault("TMPDIR", str(_CACHE_HOME / "tmp"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import cv2
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import zarr

from openpi.shared import normalize as _normalize
import openpi.training.checkpoints as _checkpoints
from openpi.training.mem.recipes import shellgame_real_memory_robust_cup0904 as _recipe
import openpi.training.sharding as sharding
from scripts.mem import train_shellgame_real_memory_adapt as _base
from scripts.mem.train_semantic_memory import init_train_state
from scripts.train import init_logging

OLD306_COUNT = 306
CUP0903_COUNT = 100
CUP0904_COUNT = 21
CUP0903_OFFSET = OLD306_COUNT
CUP0904_OFFSET = OLD306_COUNT + CUP0903_COUNT
TOTAL_EPISODES = OLD306_COUNT + CUP0903_COUNT + CUP0904_COUNT
NUM_CLASSES = 3
NUM_AUGMENTATIONS = 8
TRAIN_AUGMENTATIONS = tuple(range(6))
VALIDATION_AUGMENTATIONS = (6, 7)
CUP0904_ZARR = _WORKSPACE / "cup_0904/replay_buffer.zarr"
CUP0904_LABELS = _WORKSPACE / "cup_0904/labels.jsonl"
BASE_CACHE_DIR = _base.DEFAULT_CACHE_DIR
DEFAULT_CACHE_DIR = _REPO / "artifacts/shellgame_real_memory_robust_cup0904_features_fp16"
DEFAULT_EXP_NAME = "cup0904_fullframe_augconsistency_fold0_seed42_v1"


@dataclasses.dataclass(frozen=True)
class _CheckpointDataConfig:
    norm_stats: dict[str, _normalize.NormStats]
    asset_id: str = "."


class _CheckpointAssetLoader:
    def __init__(self):
        self._data_config = _CheckpointDataConfig(_normalize.load(_base.OLD_DATASET))

    def data_config(self):
        return self._data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--fold", choices=("0", "1", "2", "all"), default="0")
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--num-train-steps", type=int, default=1_200)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-step", type=int, default=400)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.5)
    parser.add_argument("--cup0903-final-gate", type=float, default=0.90)
    parser.add_argument("--old-final-gate", type=float, default=0.65)
    parser.add_argument("--select-last", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.feature_batch_size <= 0 or args.feature_batch_size % 2:
        parser.error("--feature-batch-size must be a positive multiple of 2")
    if args.num_train_steps <= 0 or args.eval_interval <= 0:
        parser.error("training steps and eval interval must be positive")
    if args.early_stop_patience <= 0:
        parser.error("--early-stop-patience must be positive")
    if args.consistency_loss_weight < 0:
        parser.error("--consistency-loss-weight must be non-negative")
    if args.fold == "all" and not args.select_last:
        parser.error("--fold all requires --select-last because it has no held-out cup_0904 split")
    return args


def _cup0904_three_folds(labels: list[_base.EpisodeLabels], seed: int) -> list[list[int]]:
    """Return three deterministic class-stratified folds of exactly seven episodes."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[], [], []]
    for cup in range(NUM_CLASSES):
        ids = np.asarray([i for i, label in enumerate(labels) if label.final == cup], dtype=np.int64)
        ids = rng.permutation(ids)
        chunks = [chunk.tolist() for chunk in np.array_split(ids, 3)]
        # cup_0904 has counts [6,4,11].  Rotate the middle-cup chunks so its
        # two-item chunk complements the three-item right-cup chunk.
        if cup == 1:
            chunks = chunks[1:] + chunks[:1]
        for fold, chunk in zip(folds, chunks, strict=True):
            fold.extend(chunk)
    folds = [sorted(fold) for fold in folds]
    if [len(fold) for fold in folds] != [7, 7, 7]:
        raise ValueError(f"Expected three seven-episode folds, got {[len(fold) for fold in folds]}")
    if sorted(index for fold in folds for index in fold) != list(range(CUP0904_COUNT)):
        raise ValueError("cup_0904 folds are not a partition")
    return folds


def _summary(ids: list[int], labels: list[_base.EpisodeLabels]) -> dict[str, Any]:
    finals = np.asarray([labels[index].final for index in ids], dtype=np.int32)
    relations = np.asarray([labels[index].relations for index in ids], dtype=np.int32).reshape(-1)
    return {
        "episodes": len(ids),
        "final_counts": np.bincount(finals, minlength=NUM_CLASSES).tolist(),
        "relation_counts": np.bincount(relations, minlength=NUM_CLASSES).tolist(),
    }


def build_manifest(seed: int, fold_arg: str) -> tuple[list[_base.EpisodeRef], dict[str, Any]]:
    old_labels = _base._parse_labels(_base.OLD_LABELS)
    cup0903_labels = _base._parse_labels(_base.NEW_LABELS)
    cup0904_labels = _base._parse_labels(CUP0904_LABELS)
    if [len(old_labels), len(cup0903_labels), len(cup0904_labels)] != [306, 100, 21]:
        raise ValueError("Expected old306/cup0903/cup0904 episode counts 306/100/21")
    old_split = _base._old_split(old_labels)
    cup0903_split = _base._stratified_new_split(cup0903_labels, seed)
    folds = _cup0904_three_folds(cup0904_labels, seed)
    if fold_arg == "all":
        cup0904_train = list(range(CUP0904_COUNT))
        cup0904_validation = list(range(CUP0904_COUNT))
    else:
        fold = int(fold_arg)
        cup0904_validation = folds[fold]
        validation_set = set(cup0904_validation)
        cup0904_train = [i for i in range(CUP0904_COUNT) if i not in validation_set]

    refs = [_base.EpisodeRef(0, i, label) for i, label in enumerate(old_labels)]
    refs.extend(_base.EpisodeRef(1, i, label) for i, label in enumerate(cup0903_labels))
    refs.extend(_base.EpisodeRef(2, i, label) for i, label in enumerate(cup0904_labels))
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "fold": fold_arg,
        "source_offsets": {"old306": 0, "cup0903": CUP0903_OFFSET, "cup0904": CUP0904_OFFSET},
        "old306": {
            "train_ids": old_split["train"],
            "validation_ids": old_split["validation"],
            "validation_summary": _summary(old_split["validation"], old_labels),
        },
        "cup0903": {
            "train_ids": cup0903_split["train"],
            "validation_ids": cup0903_split["validation"],
            "test_ids": cup0903_split["test"],
            "validation_summary": _summary(cup0903_split["validation"], cup0903_labels),
        },
        "cup0904": {
            "zarr": str(CUP0904_ZARR),
            "labels": str(CUP0904_LABELS),
            "folds": folds,
            "train_ids": cup0904_train,
            "validation_ids": cup0904_validation,
            "train_summary": _summary(cup0904_train, cup0904_labels),
            "validation_summary": _summary(cup0904_validation, cup0904_labels),
            "validation_is_held_out": fold_arg != "all",
        },
    }
    return refs, manifest


class Cup0904Reader:
    def __init__(self):
        replay = zarr.open(CUP0904_ZARR, mode="r")
        self.camera = replay["data"]["camera0_rgb"]
        ends = np.asarray(replay["meta"]["episode_ends"][:], dtype=np.int64)
        self.starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
        self.ends = ends
        if len(ends) != CUP0904_COUNT or self.camera.shape[0] != int(ends[-1]):
            raise ValueError("cup_0904 Zarr episode metadata is inconsistent")
        if np.any(ends - self.starts < _base.HISTORY_FRAMES):
            raise ValueError("cup_0904 contains an episode shorter than 241 frames")
        self._cached_episode = -1
        self._cached_images: np.ndarray | None = None

    def read(self, episode_id: int) -> np.ndarray:
        if episode_id == self._cached_episode and self._cached_images is not None:
            return self._cached_images
        start = int(self.starts[episode_id])
        images = np.asarray(self.camera[start : start + _base.HISTORY_FRAMES])
        if np.issubdtype(images.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(images)) <= 1.5 else 1.0
            images = np.clip(np.rint(images * scale), 0, 255).astype(np.uint8)
        elif images.dtype != np.uint8:
            images = np.clip(images, 0, 255).astype(np.uint8)
        images = np.ascontiguousarray(images)
        if images.shape != (_base.HISTORY_FRAMES, 224, 224, 3):
            raise ValueError(f"Unexpected cup_0904 image shape {images.shape}")
        self._cached_episode = episode_id
        self._cached_images = images
        return images


def augment_history(images: np.ndarray, *, seed: int) -> np.ndarray:
    """Apply one full-frame transform consistently to every history frame."""
    rng = np.random.default_rng(seed)
    gamma = float(rng.uniform(0.72, 1.38))
    contrast = float(rng.uniform(0.68, 1.34))
    brightness = float(rng.uniform(-0.10, 0.10))
    channel_gain = rng.uniform(0.82, 1.18, size=3)
    lookup = np.empty((3, 256), dtype=np.uint8)
    x = np.arange(256, dtype=np.float32) / 255.0
    for channel in range(3):
        y = ((np.power(x, gamma) - 0.5) * contrast + 0.5 + brightness) * channel_gain[channel]
        lookup[channel] = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
    result = np.empty_like(images)
    for channel in range(3):
        result[..., channel] = lookup[channel][images[..., channel]]

    scale = float(rng.uniform(0.95, 1.05))
    shift_x = float(rng.uniform(-5.0, 5.0))
    shift_y = float(rng.uniform(-5.0, 5.0))
    matrix = np.asarray(
        [[scale, 0.0, (1.0 - scale) * 111.5 + shift_x],
         [0.0, scale, (1.0 - scale) * 111.5 + shift_y]],
        dtype=np.float32,
    )
    border = tuple(int(value) for value in np.median(result[:, (0, -1), :, :], axis=(0, 1, 2)))
    warped = np.empty_like(result)
    for frame_index in range(result.shape[0]):
        warped[frame_index] = cv2.warpAffine(
            result[frame_index], matrix, (224, 224), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=border,
        )

    if rng.random() < 0.65:
        sigma = float(rng.uniform(0.25, 1.10))
        kernel = int(max(3, 2 * round(3 * sigma) + 1))
        for frame_index in range(warped.shape[0]):
            warped[frame_index] = cv2.GaussianBlur(warped[frame_index], (kernel, kernel), sigma)

    # A fixed low-frequency sensor/illumination field is shared by all frames;
    # it changes appearance without introducing fake temporal motion.
    coarse = rng.normal(0.0, rng.uniform(2.0, 8.0), size=(8, 8, 3)).astype(np.float32)
    field = cv2.resize(coarse, (224, 224), interpolation=cv2.INTER_CUBIC)
    warped = np.clip(warped.astype(np.float32) + field[None], 0, 255).astype(np.uint8)
    return np.ascontiguousarray(warped)


def _cache_contract(seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_checkpoint": _recipe.SOURCE_MEMORY_CHECKPOINT,
        "feature_backbone_contract": "same frozen PaliGemma/SigLIP as old /4999 cache",
        "cup0904_labels_mtime_ns": CUP0904_LABELS.stat().st_mtime_ns,
        "cup0904_zarr_mtime_ns": (CUP0904_ZARR / ".zgroup").stat().st_mtime_ns,
        "episodes": CUP0904_COUNT,
        "augmentations": NUM_AUGMENTATIONS,
        "augmentation_seed": seed,
        "augmentation_contract": "full-frame temporal-consistent photometric+affine+blur+lowfreq-field-v1",
        "base_shape": [CUP0904_COUNT, 3, 80, 64, 1152],
        "augmented_shape": [CUP0904_COUNT, NUM_AUGMENTATIONS, 3, 80, 64, 1152],
        "dtype": "float16",
    }


def prepare_cup0904_feature_cache(args, state, state_sharding, mesh, data_sharding):
    cache_dir = args.feature_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_path = cache_dir / "cup0904_base_fp16.npy"
    aug_path = cache_dir / "cup0904_aug8_fp16.npy"
    initial_path = cache_dir / "cup0904_initial_logits.npy"
    metadata_path = cache_dir / "metadata.json"
    contract = _cache_contract(args.seed)
    if all(path.is_file() for path in (base_path, aug_path, initial_path, metadata_path)):
        if json.loads(metadata_path.read_text(encoding="utf-8")) != contract:
            raise ValueError(f"Feature-cache contract mismatch; use a new --feature-cache-dir: {cache_dir}")
        logging.info("Reusing cup_0904 augmented feature cache: %s", cache_dir)
        return np.load(base_path, mmap_mode="r"), np.load(aug_path, mmap_mode="r"), np.load(initial_path)

    required = int((CUP0904_COUNT * (1 + NUM_AUGMENTATIONS) * 3 * 80 * 64 * 1152 * 2) * 1.08)
    if required > os.statvfs(cache_dir).f_bavail * os.statvfs(cache_dir).f_frsize:
        raise OSError(f"cup_0904 feature cache needs about {required / 1024**3:.1f} GiB")
    partial_base = cache_dir / "cup0904_base_fp16.partial.npy"
    partial_aug = cache_dir / "cup0904_aug8_fp16.partial.npy"
    partial_initial = cache_dir / "cup0904_initial_logits.partial.npy"
    progress_path = cache_dir / "progress.json"
    items = [(episode, -1) for episode in range(CUP0904_COUNT)]
    items += [(episode, variant) for episode in range(CUP0904_COUNT) for variant in range(NUM_AUGMENTATIONS)]

    resume_index = 0
    if all(path.is_file() for path in (partial_base, partial_aug, partial_initial, progress_path)):
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("contract") != contract:
            raise ValueError(f"Partial feature-cache contract mismatch: {cache_dir}")
        base = np.lib.format.open_memmap(partial_base, mode="r+")
        augmented = np.lib.format.open_memmap(partial_aug, mode="r+")
        initial = np.lib.format.open_memmap(partial_initial, mode="r+")
        resume_index = int(progress["next_item"])
        logging.info("Resuming cup_0904 feature cache at item %d/%d", resume_index, len(items))
    else:
        base = np.lib.format.open_memmap(partial_base, mode="w+", dtype=np.float16, shape=tuple(contract["base_shape"]))
        augmented = np.lib.format.open_memmap(partial_aug, mode="w+", dtype=np.float16, shape=tuple(contract["augmented_shape"]))
        initial = np.lib.format.open_memmap(partial_initial, mode="w+", dtype=np.float32, shape=(CUP0904_COUNT, 3))
        progress_path.write_text(json.dumps({"contract": contract, "next_item": 0}, indent=2) + "\n", encoding="utf-8")

    reader = Cup0904Reader()
    pextract = jax.jit(
        _base._extract_step,
        in_shardings=(state_sharding, data_sharding),
        out_shardings=(data_sharding, data_sharding),
    )
    started = time.time()
    batch_size = args.feature_batch_size
    for start in range(resume_index, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        images = []
        for episode, variant in batch_items:
            original = reader.read(episode)
            images.append(
                original if variant < 0 else augment_history(original, seed=args.seed * 1_000_003 + episode * 101 + variant)
            )
        valid = len(images)
        while len(images) < batch_size:
            images.append(images[-1])
        image_batch = jax.make_array_from_process_local_data(data_sharding, np.stack(images))
        with sharding.set_mesh(mesh):
            clip_batch, initial_batch = pextract(state, image_batch)
        clip_batch, initial_batch = jax.device_get((clip_batch, initial_batch))
        for local_index, (episode, variant) in enumerate(batch_items):
            if variant < 0:
                base[episode] = np.asarray(clip_batch[local_index], dtype=np.float16)
                initial[episode] = np.asarray(initial_batch[local_index], dtype=np.float32)
            else:
                augmented[episode, variant] = np.asarray(clip_batch[local_index], dtype=np.float16)
        base.flush()
        augmented.flush()
        initial.flush()
        next_item = start + valid
        progress_path.write_text(json.dumps({"contract": contract, "next_item": next_item}, indent=2) + "\n", encoding="utf-8")
        elapsed = time.time() - started
        logging.info(
            "cup_0904 feature cache %d/%d (%.1f min elapsed, %.1f min ETA)",
            next_item, len(items), elapsed / 60,
            (elapsed / max(next_item - resume_index, 1) * (len(items) - next_item)) / 60,
        )
    del base, augmented, initial
    partial_base.replace(base_path)
    partial_aug.replace(aug_path)
    partial_initial.replace(initial_path)
    metadata_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    if progress_path.exists():
        progress_path.unlink()
    return np.load(base_path, mmap_mode="r"), np.load(aug_path, mmap_mode="r"), np.load(initial_path)


class FeatureStore:
    def __init__(self, cup0904_base, cup0904_augmented):
        base_path = BASE_CACHE_DIR / "swap_clips_fp16.npy"
        self.base406 = np.load(base_path, mmap_mode="r")
        self.cup0904_base = cup0904_base
        self.cup0904_augmented = cup0904_augmented
        if self.base406.shape != (406, 3, 80, 64, 1152):
            raise ValueError(f"Unexpected reused base cache shape {self.base406.shape}")

    def base(self, indices: np.ndarray) -> np.ndarray:
        result = np.empty((len(indices), 3, 80, 64, 1152), dtype=np.float16)
        old_mask = indices < CUP0904_OFFSET
        if np.any(old_mask):
            result[old_mask] = self.base406[indices[old_mask]]
        if np.any(~old_mask):
            result[~old_mask] = self.cup0904_base[indices[~old_mask] - CUP0904_OFFSET]
        return result

    def augmented_cup0904(self, indices: np.ndarray, variants: np.ndarray) -> np.ndarray:
        local = indices - CUP0904_OFFSET
        if np.any(local < 0) or np.any(local >= CUP0904_COUNT):
            raise ValueError("Augmented lookup accepts only cup_0904 global ids")
        return np.asarray(self.cup0904_augmented[local, variants], dtype=np.float16)


def _labels_arrays(refs: list[_base.EpisodeRef], initial_logits: np.ndarray) -> dict[str, np.ndarray]:
    if initial_logits.shape != (TOTAL_EPISODES, 3):
        raise ValueError(f"Unexpected initial-logit shape {initial_logits.shape}")
    return {
        "initial": np.asarray([ref.labels.initial for ref in refs], dtype=np.int32),
        "initial_pred": np.argmax(initial_logits, axis=-1).astype(np.int32),
        "relations": np.asarray([ref.labels.relations for ref in refs], dtype=np.int32),
        "stages": np.asarray([ref.labels.stages for ref in refs], dtype=np.int32),
        "final": np.asarray([ref.labels.final for ref in refs], dtype=np.int32),
    }


def _global_ids(offset: int, ids: list[int]) -> np.ndarray:
    return np.asarray([offset + value for value in ids], dtype=np.int64)


def _balanced_choice(ids: np.ndarray, finals: np.ndarray, counts: tuple[int, int, int], rng) -> np.ndarray:
    chosen = []
    for cup, count in enumerate(counts):
        candidates = ids[finals[ids] == cup]
        if not len(candidates):
            raise ValueError(f"No training episodes for final cup {cup}")
        chosen.append(rng.choice(candidates, size=count, replace=True))
    result = np.concatenate(chosen)
    rng.shuffle(result)
    return result


def sample_train_indices(manifest, labels, rng):
    old_ids = _global_ids(0, manifest["old306"]["train_ids"])
    cup0903_ids = _global_ids(CUP0903_OFFSET, manifest["cup0903"]["train_ids"])
    cup0904_ids = _global_ids(CUP0904_OFFSET, manifest["cup0904"]["train_ids"])
    cup0904 = _balanced_choice(cup0904_ids, labels["final"], (5, 5, 6), rng)
    cup0903 = _balanced_choice(cup0903_ids, labels["final"], (3, 3, 4), rng)
    old306 = _balanced_choice(old_ids, labels["final"], (2, 2, 2), rng)
    base = np.concatenate((cup0904, cup0903, old306))
    rng.shuffle(base)
    variants = rng.choice(np.asarray(TRAIN_AUGMENTATIONS), size=len(cup0904), replace=True)
    return base, cup0904, variants


def _class_weights(train_ids: np.ndarray, labels: dict[str, np.ndarray]) -> np.ndarray:
    counts = np.bincount(labels["relations"][train_ids].reshape(-1), minlength=NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (NUM_CLASSES * counts)
    weights /= weights.mean()
    return weights.astype(np.float32)


def _train_batch(store, labels, base_ids, aug_ids, aug_variants, class_weights, data_sharding):
    del class_weights
    local = {
        "clips": store.base(base_ids),
        "initial": labels["initial"][base_ids],
        "relations": labels["relations"][base_ids],
        "aug_clips": store.augmented_cup0904(aug_ids, aug_variants),
        "aug_initial": labels["initial"][aug_ids],
        "aug_relations": labels["relations"][aug_ids],
    }
    return jax.tree.map(lambda value: jax.make_array_from_process_local_data(data_sharding, value), local)


def _eval_batch(store, labels, indices, data_sharding, *, variants=None, valid_count=None):
    if valid_count is None:
        valid_count = len(indices)
    clips = store.base(indices) if variants is None else store.augmented_cup0904(indices, variants)
    local = {
        "clips": clips,
        "initial": labels["initial"][indices],
        "initial_pred": labels["initial_pred"][indices],
        "relations": labels["relations"][indices],
        "stages": labels["stages"][indices],
        "valid": (np.arange(len(indices)) < valid_count).astype(np.float32),
    }
    return jax.tree.map(lambda value: jax.make_array_from_process_local_data(data_sharding, value), local)


def train_step(config, consistency_weight, class_weights, rng, state, batch):
    del rng
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def predict(memory_model, clips, initial):
        return memory_model.HistoryThreeSwapVisualRelationMemoryTracker(
            clips.astype(jnp.bfloat16), initial, None,
            preselected_pooled_clips=True, return_relation_logits_only=True,
        ).astype(jnp.float32)

    def weighted_ce(logits, targets):
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        weights = jnp.asarray(class_weights)[targets]
        return jnp.sum(losses * weights) / jnp.sum(weights)

    # The paired original clips are passed explicitly so consistency compares
    # the same cup_0904 episode, not the shuffled mixed-domain batch.
    def actual_loss_fn(memory_model):
        base_logits = predict(memory_model, batch["clips"], batch["initial"])
        pair_logits = predict(memory_model, batch["pair_clips"], batch["aug_initial"])
        aug_logits = predict(memory_model, batch["aug_clips"], batch["aug_initial"])
        base_ce = weighted_ce(base_logits, batch["relations"])
        aug_ce = weighted_ce(aug_logits, batch["aug_relations"])
        p = jax.nn.softmax(pair_logits, axis=-1)
        q = jax.nn.softmax(aug_logits, axis=-1)
        midpoint = jnp.maximum(0.5 * (p + q), 1e-7)
        js = 0.5 * jnp.sum(p * (jnp.log(jnp.maximum(p, 1e-7)) - jnp.log(midpoint)), axis=-1)
        js += 0.5 * jnp.sum(q * (jnp.log(jnp.maximum(q, 1e-7)) - jnp.log(midpoint)), axis=-1)
        consistency = jnp.mean(js)
        supervised = 0.5 * (base_ce + aug_ce)
        loss = supervised + consistency_weight * consistency
        return loss, {
            "loss": loss,
            "supervised_ce": supervised,
            "base_ce": base_ce,
            "aug_ce": aug_ce,
            "consistency_js": consistency,
            "base_relation_accuracy": jnp.mean(jnp.argmax(base_logits, axis=-1) == batch["relations"]),
            "aug_relation_accuracy": jnp.mean(jnp.argmax(aug_logits, axis=-1) == batch["aug_relations"]),
        }

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(actual_loss_fn, argnums=diff_state, has_aux=True)(model)
    trainable_params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, trainable_params)
    updated_params = optax.apply_updates(trainable_params, updates)
    nnx.update(model, updated_params)
    new_state = dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=new_opt_state)
    return new_state, {
        **info,
        "objective_loss": loss,
        "grad_norm": optax.global_norm(grads),
        "learning_rate": config.lr_schedule.create()(state.step),
    }


def eval_step(state, batch):
    return _base.eval_step(state, batch)


def evaluate_indices(peval, state, store, labels, indices, data_sharding, batch_size, *, augmented=False):
    expanded_ids = []
    expanded_variants = []
    if augmented:
        for index in indices:
            for variant in VALIDATION_AUGMENTATIONS:
                expanded_ids.append(index)
                expanded_variants.append(variant)
    else:
        expanded_ids = list(indices)
    totals = None
    for start in range(0, len(expanded_ids), batch_size):
        current = expanded_ids[start : start + batch_size]
        current_variants = expanded_variants[start : start + batch_size] if augmented else None
        valid_count = len(current)
        while len(current) < batch_size:
            current.append(current[-1])
            if augmented:
                current_variants.append(current_variants[-1])
        batch = _eval_batch(
            store, labels, np.asarray(current, dtype=np.int64), data_sharding,
            variants=None if current_variants is None else np.asarray(current_variants, dtype=np.int64),
            valid_count=valid_count,
        )
        counts = jax.device_get(peval(state, batch))
        totals = counts if totals is None else jax.tree.map(np.add, totals, counts)
    return _base._metrics_from_counts(totals)


def _selection_key(cup0904, cup0904_aug, cup0903, old306, args):
    eligible = cup0903["final_accuracy"] >= args.cup0903_final_gate and old306["final_accuracy"] >= args.old_final_gate
    relation_recalls = [value for value in cup0904["relation_per_type_accuracy"] if value is not None]
    return (
        int(eligible),
        min(cup0904["final_accuracy"], cup0904_aug["final_accuracy"]),
        cup0904["relation_accuracy"],
        min(relation_recalls),
        cup0903["final_accuracy"],
        old306["final_accuracy"],
    )


def _replace_trainable(state, trainable_params, trainable_sharding):
    trainable_params = jax.device_put(trainable_params, trainable_sharding)
    model = nnx.merge(state.model_def, state.params)
    nnx.update(model, trainable_params)
    return dataclasses.replace(state, params=nnx.state(model))


def main() -> None:
    args = parse_args()
    init_logging()
    refs, manifest = build_manifest(args.seed, args.fold)
    logging.info("Split contract:\n%s", json.dumps(manifest, indent=2))
    if args.validate_only:
        print(json.dumps(manifest, indent=2))
        return

    config = dataclasses.replace(
        _recipe.make_train_config(), exp_name=args.exp_name, seed=args.seed,
        num_train_steps=args.num_train_steps, eval_interval=args.eval_interval,
        overwrite=args.overwrite,
    )
    if config.batch_size != 32 or config.fsdp_devices != 2 or jax.device_count() != 2:
        raise ValueError(
            f"This run requires batch_size=32, fsdp_devices=2, and exactly two visible GPUs; "
            f"got {config.batch_size}, {config.fsdp_devices}, {jax.device_count()}"
        )
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state, state_sharding = init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    trainable_count = sum(
        int(np.prod((leaf.value if hasattr(leaf, "value") else leaf).shape))
        for leaf in jax.tree.leaves(state.params.filter(config.trainable_filter))
        if hasattr(leaf.value if hasattr(leaf, "value") else leaf, "shape")
    )
    if trainable_count <= 0:
        raise ValueError("Freeze filter selected zero trainable parameters")
    logging.info("Verified relation-only trainable parameters: %d (%.2fM)", trainable_count, trainable_count / 1e6)

    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period,
        overwrite=config.overwrite, resume=False,
    )
    cup0904_base, cup0904_aug, cup0904_initial = prepare_cup0904_feature_cache(
        args, state, state_sharding, mesh, data_sharding
    )
    base_initial = np.load(BASE_CACHE_DIR / "initial_logits.npy")
    initial_logits = np.concatenate((base_initial, cup0904_initial), axis=0)
    store = FeatureStore(cup0904_base, cup0904_aug)
    labels = _labels_arrays(refs, initial_logits)
    output_dir = config.checkpoint_dir
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    all_train_ids = np.concatenate((
        _global_ids(0, manifest["old306"]["train_ids"]),
        _global_ids(CUP0903_OFFSET, manifest["cup0903"]["train_ids"]),
        _global_ids(CUP0904_OFFSET, manifest["cup0904"]["train_ids"]),
    ))
    class_weights = _class_weights(all_train_ids, labels)
    logging.info("Relation class weights: %s", class_weights.tolist())
    ptrain = jax.jit(
        functools.partial(train_step, config, args.consistency_loss_weight, class_weights),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,),
    )
    peval = jax.jit(
        eval_step, in_shardings=(state_sharding, data_sharding), out_shardings=replicated
    )
    old_val = _global_ids(0, manifest["old306"]["validation_ids"]).tolist()
    cup0903_val = _global_ids(CUP0903_OFFSET, manifest["cup0903"]["validation_ids"]).tolist()
    cup0904_val = _global_ids(CUP0904_OFFSET, manifest["cup0904"]["validation_ids"]).tolist()
    history = []

    def evaluate(step):
        metrics0904 = evaluate_indices(peval, state, store, labels, cup0904_val, data_sharding, 32)
        metrics0904_aug = evaluate_indices(peval, state, store, labels, cup0904_val, data_sharding, 32, augmented=True)
        metrics0903 = evaluate_indices(peval, state, store, labels, cup0903_val, data_sharding, 32)
        metrics_old = evaluate_indices(peval, state, store, labels, old_val, data_sharding, 32)
        record = {
            "step": step,
            "cup0904_validation": metrics0904,
            "cup0904_augmented_validation": metrics0904_aug,
            "cup0903_validation": metrics0903,
            "old306_validation": metrics_old,
        }
        record["selection_key"] = list(_selection_key(metrics0904, metrics0904_aug, metrics0903, metrics_old, args))
        history.append(record)
        (output_dir / "metrics_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        logging.info(
            "Step %d [eval] 0904 final=%.4f relation=%.4f aug_final=%.4f; "
            "0903 final=%.4f relation=%.4f; old final=%.4f relation=%.4f",
            step, metrics0904["final_accuracy"], metrics0904["relation_accuracy"],
            metrics0904_aug["final_accuracy"], metrics0903["final_accuracy"],
            metrics0903["relation_accuracy"], metrics_old["final_accuracy"], metrics_old["relation_accuracy"],
        )
        return record

    baseline = evaluate(0)
    best_key = tuple(baseline["selection_key"])
    best_step = 0
    best_trainable = jax.device_get(state.params.filter(config.trainable_filter))
    no_improvement = 0
    np_rng = np.random.default_rng(config.seed + (3 if args.fold == "all" else int(args.fold)))
    infos = []
    stopped_early = False
    started = time.time()
    for step in range(config.num_train_steps):
        base_ids, aug_ids, aug_variants = sample_train_indices(manifest, labels, np_rng)
        batch = _train_batch(store, labels, base_ids, aug_ids, aug_variants, class_weights, data_sharding)
        batch["pair_clips"] = jax.make_array_from_process_local_data(data_sharding, store.base(aug_ids))
        with sharding.set_mesh(mesh):
            state, info = ptrain(train_rng, state, batch)
        infos.append(info)
        completed = step + 1
        if completed % config.log_interval == 0:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            logging.info("Step %d [train] %s", completed, ", ".join(f"{k}={float(v):.6f}" for k, v in reduced.items()))
            infos = []
        if completed % config.eval_interval == 0 or completed == config.num_train_steps:
            record = evaluate(completed)
            key = tuple(record["selection_key"])
            improved = key > best_key
            if args.select_last or improved:
                best_key = key
                best_step = completed
                best_trainable = jax.device_get(state.params.filter(config.trainable_filter))
            if improved:
                no_improvement = 0
                logging.info("New best robust relation head at step %d: key=%s", best_step, best_key)
            else:
                no_improvement += 1
            if not args.select_last and completed >= args.early_stop_min_step and no_improvement >= args.early_stop_patience:
                logging.info("Early stopping at step %d after %d evals without improvement", completed, no_improvement)
                stopped_early = True
                break

    state = _replace_trainable(state, best_trainable, state_sharding.params.filter(config.trainable_filter))
    selected0904 = evaluate_indices(peval, state, store, labels, cup0904_val, data_sharding, 32)
    selected0904_aug = evaluate_indices(peval, state, store, labels, cup0904_val, data_sharding, 32, augmented=True)
    selected0903 = evaluate_indices(peval, state, store, labels, cup0903_val, data_sharding, 32)
    selected_old = evaluate_indices(peval, state, store, labels, old_val, data_sharding, 32)
    _checkpoints.save_state(checkpoint_manager, state, _CheckpointAssetLoader(), best_step)
    checkpoint_manager.wait_until_finished()
    summary = {
        "config": config.name,
        "exp_name": config.exp_name,
        "source_checkpoint": _recipe.SOURCE_MEMORY_CHECKPOINT,
        "fold": args.fold,
        "cup0904_validation_is_held_out": args.fold != "all",
        "selected_step": best_step,
        "stopped_early": stopped_early,
        "elapsed_hours_excluding_feature_cache": (time.time() - started) / 3600,
        "batch_contract": "32 base = 16 cup0904 + 10 cup0903 + 6 old306; plus 16 paired augmented cup0904",
        "augmentation_contract": "full-frame temporally consistent; train variants 0..5; eval-only variants 6..7",
        "loss_contract": f"balanced relation CE + {args.consistency_loss_weight} * paired Jensen-Shannon",
        "trainable_contract": "swap_relation_classifier only",
        "cup0904_validation": selected0904,
        "cup0904_augmented_validation": selected0904_aug,
        "cup0903_validation": selected0903,
        "old306_validation": selected_old,
        "checkpoint": str(output_dir / str(best_step)),
        "feature_cache": str(args.feature_cache_dir.resolve()),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("TRAINING_COMPLETE %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
