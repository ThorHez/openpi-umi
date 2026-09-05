#!/usr/bin/env python3
"""Train a fresh ShellGame MEM on real episodes with mild augmentation.

The generic Pi0.5 backbone is retained, while every ShellGame-specific memory
parameter starts from a new random initialization.  The default legacy mode
uses all 427 episodes.  ``--heldout-validation`` instead creates a fixed,
per-domain and per-final-cup stratified episode split before optimization.
Augmentation variant 2 is never sampled by the optimizer and supplies an
unseen mild view of either the training set or the held-out validation set.
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

WORKSPACE = Path("/data2/hzl_workspace_for_pi_mem")
REPO = WORKSPACE / "openpi-umi"
CACHE_HOME = WORKSPACE / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_HOME))
os.environ.setdefault("OPENPI_DATA_HOME", str(CACHE_HOME / "openpi"))
os.environ.setdefault("HF_HOME", str(CACHE_HOME / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_HOME / "huggingface" / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_HOME / "huggingface" / "transformers"))
os.environ.setdefault("TMPDIR", str(CACHE_HOME / "tmp"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import zarr

from openpi.models import siglip_mem_semantic as memory_core
from openpi.shared import normalize as _normalize
import openpi.training.checkpoints as _checkpoints
from openpi.training.mem.recipes import shellgame_real_memory_mild_all as _recipe
import openpi.training.sharding as sharding
from scripts.mem import train_shellgame_real_memory_adapt as _old_adapt
from scripts.mem.train_semantic_memory import init_train_state
from scripts.train import init_logging

OLD_COUNT = 306
CUP0903_COUNT = 100
CUP0904_COUNT = 21
CUP0903_OFFSET = OLD_COUNT
CUP0904_OFFSET = OLD_COUNT + CUP0903_COUNT
TOTAL_EPISODES = OLD_COUNT + CUP0903_COUNT + CUP0904_COUNT
HISTORY_FRAMES = 241
NUM_CLASSES = 3
TRAIN_AUGMENTATIONS = (0, 1)
EVAL_AUGMENTATION = 2
NUM_AUGMENTATIONS = 3
CUP0904_ZARR = WORKSPACE / "cup_0904/replay_buffer.zarr"
CUP0904_LABELS = WORKSPACE / "cup_0904/labels.jsonl"
DEFAULT_CACHE_DIR = REPO / "artifacts/shellgame_real_fresh_memory_mild_all_features_fp16"
DEFAULT_EXP_NAME = "freshmem_all427_mildaug_b32_seed42_v1"


@dataclasses.dataclass(frozen=True)
class _CheckpointDataConfig:
    norm_stats: dict[str, _normalize.NormStats]
    asset_id: str = "."


class _CheckpointAssetLoader:
    def __init__(self):
        self._data_config = _CheckpointDataConfig(_normalize.load(_old_adapt.OLD_DATASET))

    def data_config(self):
        return self._data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--num-train-steps", type=int, default=3_000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--early-stop-min-step", type=int, default=1_000)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-validation", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--validation-min-per-class", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=20260904)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.feature_batch_size <= 0 or args.feature_batch_size % 2:
        parser.error("--feature-batch-size must be a positive multiple of two")
    if args.num_train_steps <= 0 or args.eval_interval <= 0:
        parser.error("training steps and eval interval must be positive")
    if args.early_stop_patience <= 0 or args.consistency_loss_weight < 0:
        parser.error("invalid early-stop or consistency-loss value")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    if args.validation_min_per_class <= 0:
        parser.error("--validation-min-per-class must be positive")
    return args


def _load_manifest() -> tuple[list[_old_adapt.EpisodeRef], dict[str, Any]]:
    old_labels = _old_adapt._parse_labels(_old_adapt.OLD_LABELS)
    cup0903_labels = _old_adapt._parse_labels(_old_adapt.NEW_LABELS)
    cup0904_labels = _old_adapt._parse_labels(CUP0904_LABELS)
    counts = [len(old_labels), len(cup0903_labels), len(cup0904_labels)]
    if counts != [OLD_COUNT, CUP0903_COUNT, CUP0904_COUNT]:
        raise ValueError(f"Expected episode counts [306,100,21], got {counts}")
    refs = [_old_adapt.EpisodeRef(0, index, label) for index, label in enumerate(old_labels)]
    refs.extend(_old_adapt.EpisodeRef(1, index, label) for index, label in enumerate(cup0903_labels))
    refs.extend(_old_adapt.EpisodeRef(2, index, label) for index, label in enumerate(cup0904_labels))

    def summarize(labels):
        finals = np.asarray([label.final for label in labels], dtype=np.int32)
        relations = np.asarray([label.relations for label in labels], dtype=np.int32).reshape(-1)
        initials = np.asarray([label.initial for label in labels], dtype=np.int32)
        return {
            "episodes": len(labels),
            "initial_counts": np.bincount(initials, minlength=3).tolist(),
            "relation_counts": np.bincount(relations, minlength=3).tolist(),
            "final_counts": np.bincount(finals, minlength=3).tolist(),
        }

    manifest = {
        "schema_version": 1,
        "training_uses_every_episode": True,
        "episode_heldout_validation": False,
        "source_offsets": {"old306": 0, "cup0903": CUP0903_OFFSET, "cup0904": CUP0904_OFFSET},
        "old306": summarize(old_labels),
        "cup0903": summarize(cup0903_labels),
        "cup0904": summarize(cup0904_labels),
        "train_augmentations": list(TRAIN_AUGMENTATIONS),
        "perturbation_check_augmentation": EVAL_AUGMENTATION,
    }
    return refs, manifest


class _ZarrReader:
    def __init__(self, path: Path, count: int):
        replay = zarr.open(path, mode="r")
        self.camera = replay["data"]["camera0_rgb"]
        ends = np.asarray(replay["meta"]["episode_ends"][:], dtype=np.int64)
        self.starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
        if len(ends) != count or self.camera.shape[0] != int(ends[-1]):
            raise ValueError(f"Inconsistent Zarr metadata in {path}")
        if np.any(ends - self.starts < HISTORY_FRAMES):
            raise ValueError(f"An episode in {path} is shorter than {HISTORY_FRAMES}")

    def read(self, episode_id: int) -> np.ndarray:
        start = int(self.starts[episode_id])
        images = np.asarray(self.camera[start : start + HISTORY_FRAMES])
        if np.issubdtype(images.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(images)) <= 1.5 else 1.0
            images = np.clip(np.rint(images * scale), 0, 255).astype(np.uint8)
        elif images.dtype != np.uint8:
            images = np.clip(images, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(images)


class HistoryReader:
    def __init__(self):
        self.cup0903 = _ZarrReader(_old_adapt.NEW_ZARR, CUP0903_COUNT)
        self.cup0904 = _ZarrReader(CUP0904_ZARR, CUP0904_COUNT)
        self.cached_global_id = -1
        self.cached_images: np.ndarray | None = None

    def read(self, global_id: int) -> np.ndarray:
        if global_id == self.cached_global_id and self.cached_images is not None:
            return self.cached_images
        if global_id < CUP0903_OFFSET:
            images = _old_adapt._decode_old_history(global_id)
        elif global_id < CUP0904_OFFSET:
            images = self.cup0903.read(global_id - CUP0903_OFFSET)
        else:
            images = self.cup0904.read(global_id - CUP0904_OFFSET)
        if images.shape != (HISTORY_FRAMES, 224, 224, 3):
            raise ValueError(f"Unexpected history shape {images.shape} for global episode {global_id}")
        self.cached_global_id = global_id
        self.cached_images = images
        return images


def mild_augment_history(images: np.ndarray, *, seed: int) -> np.ndarray:
    """Apply a small, temporally fixed appearance/geometry perturbation."""
    rng = np.random.default_rng(seed)
    gamma = float(rng.uniform(0.92, 1.08))
    contrast = float(rng.uniform(0.90, 1.10))
    brightness = float(rng.uniform(-0.025, 0.025))
    channel_gain = rng.uniform(0.95, 1.05, size=3)
    x = np.arange(256, dtype=np.float32) / 255.0
    lookup = np.empty((3, 256), dtype=np.uint8)
    for channel in range(3):
        y = ((np.power(x, gamma) - 0.5) * contrast + 0.5 + brightness) * channel_gain[channel]
        lookup[channel] = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
    result = np.empty_like(images)
    for channel in range(3):
        result[..., channel] = lookup[channel][images[..., channel]]

    scale = float(rng.uniform(0.985, 1.015))
    shift_x = float(rng.uniform(-2.0, 2.0))
    shift_y = float(rng.uniform(-2.0, 2.0))
    matrix = np.asarray(
        [[scale, 0.0, (1.0 - scale) * 111.5 + shift_x], [0.0, scale, (1.0 - scale) * 111.5 + shift_y]],
        dtype=np.float32,
    )
    border = tuple(int(value) for value in np.median(result[:, (0, -1), :, :], axis=(0, 1, 2)))
    warped = np.empty_like(result)
    for frame_index in range(HISTORY_FRAMES):
        warped[frame_index] = cv2.warpAffine(
            result[frame_index],
            matrix,
            (224, 224),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
    if rng.random() < 0.30:
        sigma = float(rng.uniform(0.20, 0.45))
        for frame_index in range(HISTORY_FRAMES):
            warped[frame_index] = cv2.GaussianBlur(warped[frame_index], (3, 3), sigma)
    coarse = rng.normal(0.0, rng.uniform(0.4, 1.5), size=(8, 8, 3)).astype(np.float32)
    field = cv2.resize(coarse, (224, 224), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(np.clip(warped.astype(np.float32) + field[None], 0, 255).astype(np.uint8))


def _extract_features(state, images):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    normalized = images.astype(jnp.float32) / 127.5 - 1.0
    _, initial_out = model.PaliGemma.img(normalized[:, :1], train=False)
    initial_patches = initial_out["encoded"]
    _, history_out = model.PaliGemma.img(normalized, train=False)
    patches = history_out["with_posemb"][:, :HISTORY_FRAMES]
    pooled = memory_core.pool_fixed_grid(patches, pool_factor=2)
    clips = jnp.stack([pooled[:, jnp.asarray(indices)] for indices in _old_adapt.SWAP_FRAME_INDICES], axis=1)
    return clips, initial_patches


def _cache_contract(seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generic_backbone_checkpoint": _recipe.GENERIC_PI05_CHECKPOINT,
        "explicitly_excluded_memory_checkpoints": ["/data2/hzl_workspace_for_pi_mem/4999", "cup0903 step500"],
        "episodes": TOTAL_EPISODES,
        "augmentations": NUM_AUGMENTATIONS,
        "seed": seed,
        "augmentation": "mild-temporally-consistent-v1",
        "raw_clip_shape": [TOTAL_EPISODES, 3, 80, 64, 1152],
        "raw_initial_shape": [TOTAL_EPISODES, 256, 1152],
        "aug_clip_shape": [TOTAL_EPISODES, NUM_AUGMENTATIONS, 3, 80, 64, 1152],
        "aug_initial_shape": [TOTAL_EPISODES, NUM_AUGMENTATIONS, 256, 1152],
        "old_labels_mtime_ns": _old_adapt.OLD_LABELS.stat().st_mtime_ns,
        "cup0903_labels_mtime_ns": _old_adapt.NEW_LABELS.stat().st_mtime_ns,
        "cup0904_labels_mtime_ns": CUP0904_LABELS.stat().st_mtime_ns,
    }


def prepare_feature_cache(args, state, state_sharding, mesh, data_sharding):
    cache_dir = args.feature_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "raw_clips": "raw_clips_fp16.npy",
        "raw_initial": "raw_initial_patches_fp16.npy",
        "aug_clips": "mild_aug3_clips_fp16.npy",
        "aug_initial": "mild_aug3_initial_patches_fp16.npy",
    }
    paths = {key: cache_dir / value for key, value in names.items()}
    metadata_path = cache_dir / "metadata.json"
    contract = _cache_contract(args.seed)
    if metadata_path.is_file() and all(path.is_file() for path in paths.values()):
        if json.loads(metadata_path.read_text(encoding="utf-8")) != contract:
            raise ValueError(f"Feature cache contract mismatch; choose a new cache dir: {cache_dir}")
        logging.info("Reusing fresh-backbone all-domain feature cache: %s", cache_dir)
        return {key: np.load(path, mmap_mode="r") for key, path in paths.items()}

    shapes = {
        "raw_clips": tuple(contract["raw_clip_shape"]),
        "raw_initial": tuple(contract["raw_initial_shape"]),
        "aug_clips": tuple(contract["aug_clip_shape"]),
        "aug_initial": tuple(contract["aug_initial_shape"]),
    }
    required = int(sum(np.prod(shape) * 2 for shape in shapes.values()) * 1.08)
    available = os.statvfs(cache_dir).f_bavail * os.statvfs(cache_dir).f_frsize
    if required > available:
        raise OSError(f"Feature cache needs {required / 1024**3:.1f} GiB, only {available / 1024**3:.1f} GiB free")
    partial_paths = {key: cache_dir / value.replace(".npy", ".partial.npy") for key, value in names.items()}
    progress_path = cache_dir / "progress.json"
    items = [(episode, variant) for episode in range(TOTAL_EPISODES) for variant in (-1, 0, 1, 2)]
    if progress_path.is_file() and all(path.is_file() for path in partial_paths.values()):
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("contract") != contract:
            raise ValueError(f"Partial feature cache contract mismatch: {cache_dir}")
        arrays = {key: np.lib.format.open_memmap(path, mode="r+") for key, path in partial_paths.items()}
        resume_index = int(progress["next_item"])
        logging.info("Resuming feature cache at %d/%d", resume_index, len(items))
    else:
        arrays = {
            key: np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shapes[key])
            for key, path in partial_paths.items()
        }
        resume_index = 0
        progress_path.write_text(json.dumps({"contract": contract, "next_item": 0}, indent=2) + "\n", encoding="utf-8")

    reader = HistoryReader()
    pextract = jax.jit(
        _extract_features,
        in_shardings=(state_sharding, data_sharding),
        out_shardings=(data_sharding, data_sharding),
    )
    started = time.time()
    for start in range(resume_index, len(items), args.feature_batch_size):
        batch_items = items[start : start + args.feature_batch_size]
        images = []
        for episode, variant in batch_items:
            raw = reader.read(episode)
            aug_seed = args.seed * 1_000_003 + episode * 101 + variant
            images.append(raw if variant < 0 else mild_augment_history(raw, seed=aug_seed))
        valid = len(images)
        while len(images) < args.feature_batch_size:
            images.append(images[-1])
        image_batch = jax.make_array_from_process_local_data(data_sharding, np.stack(images))
        with sharding.set_mesh(mesh):
            clips, initial_patches = pextract(state, image_batch)
        clips, initial_patches = jax.device_get((clips, initial_patches))
        for local_index, (episode, variant) in enumerate(batch_items):
            if variant < 0:
                arrays["raw_clips"][episode] = np.asarray(clips[local_index], dtype=np.float16)
                arrays["raw_initial"][episode] = np.asarray(initial_patches[local_index], dtype=np.float16)
            else:
                arrays["aug_clips"][episode, variant] = np.asarray(clips[local_index], dtype=np.float16)
                arrays["aug_initial"][episode, variant] = np.asarray(initial_patches[local_index], dtype=np.float16)
        for array in arrays.values():
            array.flush()
        next_item = start + valid
        progress_path.write_text(
            json.dumps({"contract": contract, "next_item": next_item}, indent=2) + "\n", encoding="utf-8"
        )
        if next_item % 20 == 0 or next_item == len(items):
            elapsed = time.time() - started
            eta = elapsed / max(next_item - resume_index, 1) * (len(items) - next_item)
            logging.info(
                "Feature cache %d/%d (%.1f min elapsed, %.1f min ETA)", next_item, len(items), elapsed / 60, eta / 60
            )
    del arrays
    for key in names:
        partial_paths[key].replace(paths[key])
    metadata_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    progress_path.unlink()
    return {key: np.load(path, mmap_mode="r") for key, path in paths.items()}


def _labels(refs):
    return {
        "initial": np.asarray([ref.labels.initial for ref in refs], dtype=np.int32),
        "relations": np.asarray([ref.labels.relations for ref in refs], dtype=np.int32),
        "stages": np.asarray([ref.labels.stages for ref in refs], dtype=np.int32),
        "final": np.asarray([ref.labels.final for ref in refs], dtype=np.int32),
    }


def _domain_ids() -> dict[str, np.ndarray]:
    return {
        "old306": np.arange(0, CUP0903_OFFSET, dtype=np.int64),
        "cup0903": np.arange(CUP0903_OFFSET, CUP0904_OFFSET, dtype=np.int64),
        "cup0904": np.arange(CUP0904_OFFSET, TOTAL_EPISODES, dtype=np.int64),
    }


def _summarize_split_ids(ids: np.ndarray, labels: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "episodes": len(ids),
        "global_episode_ids": ids.tolist(),
        "initial_counts": np.bincount(labels["initial"][ids], minlength=NUM_CLASSES).tolist(),
        "relation_counts": np.bincount(labels["relations"][ids].reshape(-1), minlength=NUM_CLASSES).tolist(),
        "final_counts": np.bincount(labels["final"][ids], minlength=NUM_CLASSES).tolist(),
    }


def make_episode_split(
    labels: dict[str, np.ndarray],
    *,
    heldout_validation: bool,
    validation_fraction: float,
    validation_min_per_class: int,
    split_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    all_domain_ids = _domain_ids()
    if not heldout_validation:
        empty = {domain: np.empty(0, dtype=np.int64) for domain in all_domain_ids}
        return (
            all_domain_ids,
            empty,
            {
                "heldout_validation": False,
                "split_seed": None,
                "training": {domain: _summarize_split_ids(ids, labels) for domain, ids in all_domain_ids.items()},
                "validation": {domain: _summarize_split_ids(ids, labels) for domain, ids in empty.items()},
            },
        )

    rng = np.random.default_rng(split_seed)
    training: dict[str, np.ndarray] = {}
    validation: dict[str, np.ndarray] = {}
    for domain, ids in all_domain_ids.items():
        class_candidates = [ids[labels["final"][ids] == cup] for cup in range(NUM_CLASSES)]
        class_counts = np.asarray([len(candidates) for candidates in class_candidates])
        if np.any(class_counts <= validation_min_per_class):
            raise ValueError(
                f"{domain} final counts {class_counts.tolist()} cannot hold out "
                f"{validation_min_per_class} per class and retain training data"
            )
        validation_target = max(
            NUM_CLASSES * validation_min_per_class,
            round(len(ids) * validation_fraction),
        )
        validation_target = min(validation_target, int(np.sum(class_counts - 1)))
        desired_counts = class_counts * (validation_target / len(ids))
        validation_counts = np.full(NUM_CLASSES, validation_min_per_class, dtype=np.int64)
        while int(validation_counts.sum()) < validation_target:
            has_capacity = validation_counts < class_counts - 1
            deficits = np.where(has_capacity, desired_counts - validation_counts, -np.inf)
            validation_counts[int(np.argmax(deficits))] += 1

        train_parts = []
        validation_parts = []
        for cup, candidates in enumerate(class_candidates):
            shuffled = rng.permutation(candidates)
            validation_count = int(validation_counts[cup])
            validation_parts.append(shuffled[:validation_count])
            train_parts.append(shuffled[validation_count:])
        training[domain] = np.sort(np.concatenate(train_parts)).astype(np.int64)
        validation[domain] = np.sort(np.concatenate(validation_parts)).astype(np.int64)

    split_manifest = {
        "heldout_validation": True,
        "split_seed": split_seed,
        "validation_fraction": validation_fraction,
        "validation_min_per_class": validation_min_per_class,
        "training": {domain: _summarize_split_ids(ids, labels) for domain, ids in training.items()},
        "validation": {domain: _summarize_split_ids(ids, labels) for domain, ids in validation.items()},
    }
    return training, validation, split_manifest


def _balanced_choice(ids, finals, counts, rng):
    selected = []
    for cup, count in enumerate(counts):
        candidates = ids[finals[ids] == cup]
        selected.append(rng.choice(candidates, size=count, replace=True))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


def sample_batch_ids(labels, domain_ids, rng):
    old = _balanced_choice(domain_ids["old306"], labels["final"], (2, 2, 2), rng)
    cup0903 = _balanced_choice(domain_ids["cup0903"], labels["final"], (3, 3, 4), rng)
    cup0904 = _balanced_choice(domain_ids["cup0904"], labels["final"], (5, 5, 6), rng)
    ids = np.concatenate((old, cup0903, cup0904))
    rng.shuffle(ids)
    variants = rng.choice(np.asarray(TRAIN_AUGMENTATIONS), size=len(ids), replace=True)
    return ids, variants


def make_batch(features, labels, ids, variants, data_sharding):
    local = {
        "raw_clips": np.asarray(features["raw_clips"][ids], dtype=np.float16),
        "raw_initial": np.asarray(features["raw_initial"][ids], dtype=np.float16),
        "aug_clips": np.asarray(features["aug_clips"][ids, variants], dtype=np.float16),
        "aug_initial": np.asarray(features["aug_initial"][ids, variants], dtype=np.float16),
        "initial_labels": labels["initial"][ids],
        "relation_labels": labels["relations"][ids],
        "stage_labels": labels["stages"][ids],
    }
    return jax.tree.map(lambda value: jax.make_array_from_process_local_data(data_sharding, value), local)


def _js_divergence(logits_a, logits_b):
    p = jax.nn.softmax(logits_a.astype(jnp.float32), axis=-1)
    q = jax.nn.softmax(logits_b.astype(jnp.float32), axis=-1)
    midpoint = jnp.maximum(0.5 * (p + q), 1e-7)
    kl_p = jnp.sum(p * (jnp.log(jnp.maximum(p, 1e-7)) - jnp.log(midpoint)), axis=-1)
    kl_q = jnp.sum(q * (jnp.log(jnp.maximum(q, 1e-7)) - jnp.log(midpoint)), axis=-1)
    return jnp.mean(0.5 * (kl_p + kl_q))


def train_step(config, class_weights, consistency_weight, rng, state, batch):
    del rng
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def forward(memory_model, initial_patches, clips):
        initial_logits = memory_model.HistoryFrame0InitialCupClassifier(initial_patches.astype(jnp.bfloat16))
        _, stage_logits, _, relation_logits, _ = memory_model.HistoryThreeSwapVisualRelationMemoryTracker(
            clips.astype(jnp.bfloat16),
            batch["initial_labels"],
            batch["relation_labels"],
            preselected_pooled_clips=True,
        )
        return initial_logits.astype(jnp.float32), relation_logits.astype(jnp.float32), stage_logits.astype(jnp.float32)

    def relation_ce(logits):
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, batch["relation_labels"])
        weights = jnp.asarray(class_weights)[batch["relation_labels"]]
        return jnp.sum(losses * weights) / jnp.sum(weights)

    def supervised(outputs):
        initial_logits, relation_logits, stage_logits = outputs
        initial = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(initial_logits, batch["initial_labels"]))
        relation = relation_ce(relation_logits)
        stage = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(stage_logits, batch["stage_labels"]))
        return initial + relation + stage, (initial, relation, stage)

    def loss_fn(memory_model):
        raw = forward(memory_model, batch["raw_initial"], batch["raw_clips"])
        aug = forward(memory_model, batch["aug_initial"], batch["aug_clips"])
        raw_loss, raw_parts = supervised(raw)
        aug_loss, aug_parts = supervised(aug)
        consistency = (
            _js_divergence(raw[0], aug[0]) + _js_divergence(raw[1], aug[1]) + _js_divergence(raw[2], aug[2])
        ) / 3.0
        loss = 0.5 * (raw_loss + aug_loss) + consistency_weight * consistency
        return loss, {
            "loss": loss,
            "raw_initial_ce": raw_parts[0],
            "raw_relation_ce": raw_parts[1],
            "raw_stage_ce": raw_parts[2],
            "aug_initial_ce": aug_parts[0],
            "aug_relation_ce": aug_parts[1],
            "aug_stage_ce": aug_parts[2],
            "consistency_js": consistency,
            "raw_initial_accuracy": jnp.mean(jnp.argmax(raw[0], axis=-1) == batch["initial_labels"]),
            "raw_relation_accuracy": jnp.mean(jnp.argmax(raw[1], axis=-1) == batch["relation_labels"]),
            "raw_stage_accuracy": jnp.mean(jnp.argmax(raw[2], axis=-1) == batch["stage_labels"]),
            "aug_relation_accuracy": jnp.mean(jnp.argmax(aug[1], axis=-1) == batch["relation_labels"]),
        }

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
    trainable = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, trainable)
    nnx.update(model, optax.apply_updates(trainable, updates))
    new_state = dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=new_opt_state)
    return new_state, {
        **info,
        "objective_loss": loss,
        "grad_norm": optax.global_norm(grads),
        "learning_rate": config.lr_schedule.create()(state.step),
    }


def eval_step(state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    initial_logits = model.HistoryFrame0InitialCupClassifier(batch["initial_patches"].astype(jnp.bfloat16))
    initial_pred = jnp.argmax(initial_logits, axis=-1)
    _, stage_logits, _, relation_logits, _ = model.HistoryThreeSwapVisualRelationMemoryTracker(
        batch["clips"].astype(jnp.bfloat16),
        initial_pred,
        None,
        preselected_pooled_clips=True,
    )
    relation_pred = jnp.argmax(relation_logits, axis=-1)
    stage_pred = jnp.argmax(stage_logits, axis=-1)
    valid = batch["valid"].astype(jnp.int32)
    relation_valid = jnp.repeat(valid[:, None], 3, axis=1)
    return {
        "relation_confusion": jnp.bincount(
            (batch["relations"] * 3 + relation_pred).reshape(-1),
            weights=relation_valid.reshape(-1),
            length=9,
        ).reshape(3, 3),
        "final_confusion": jnp.bincount(
            batch["stages"][:, -1] * 3 + stage_pred[:, -1],
            weights=valid,
            length=9,
        ).reshape(3, 3),
        "initial_confusion": jnp.bincount(
            batch["initial"] * 3 + initial_pred,
            weights=valid,
            length=9,
        ).reshape(3, 3),
        "stage_correct": jnp.sum((stage_pred == batch["stages"]) * valid[:, None], axis=0),
        "episodes": jnp.sum(valid),
    }


def evaluate(peval, state, features, labels, ids, data_sharding, *, augmented):
    totals = None
    for start in range(0, len(ids), 32):
        current = list(ids[start : start + 32])
        valid_count = len(current)
        while len(current) < 32:
            current.append(current[-1])
        current = np.asarray(current, dtype=np.int64)
        if augmented:
            clips = np.asarray(features["aug_clips"][current, EVAL_AUGMENTATION], dtype=np.float16)
            initial_patches = np.asarray(features["aug_initial"][current, EVAL_AUGMENTATION], dtype=np.float16)
        else:
            clips = np.asarray(features["raw_clips"][current], dtype=np.float16)
            initial_patches = np.asarray(features["raw_initial"][current], dtype=np.float16)
        local = {
            "clips": clips,
            "initial_patches": initial_patches,
            "initial": labels["initial"][current],
            "relations": labels["relations"][current],
            "stages": labels["stages"][current],
            "valid": (np.arange(32) < valid_count).astype(np.float32),
        }
        batch = jax.tree.map(lambda value: jax.make_array_from_process_local_data(data_sharding, value), local)
        counts = jax.device_get(peval(state, batch))
        totals = counts if totals is None else jax.tree.map(np.add, totals, counts)
    return _old_adapt._metrics_from_counts(totals)


def _selection_key(metrics):
    raw = metrics["raw"]
    aug = metrics["mild_aug_unseen"]
    return (
        min(aug[domain]["final_accuracy"] for domain in ("cup0904", "cup0903", "old306")),
        min(aug[domain]["relation_accuracy"] for domain in ("cup0904", "cup0903", "old306")),
        aug["cup0904"]["final_accuracy"],
        raw["cup0904"]["final_accuracy"],
        raw["cup0903"]["final_accuracy"],
        raw["old306"]["final_accuracy"],
    )


def _replace_trainable(state, params, param_sharding):
    params = jax.device_put(params, param_sharding)
    model = nnx.merge(state.model_def, state.params)
    nnx.update(model, params)
    return dataclasses.replace(state, params=nnx.state(model))


def main() -> None:
    args = parse_args()
    init_logging()
    refs, manifest = _load_manifest()
    labels = _labels(refs)
    train_domain_ids, validation_domain_ids, split_manifest = make_episode_split(
        labels,
        heldout_validation=args.heldout_validation,
        validation_fraction=args.validation_fraction,
        validation_min_per_class=args.validation_min_per_class,
        split_seed=args.split_seed,
    )
    manifest["training_uses_every_episode"] = not args.heldout_validation
    manifest["episode_heldout_validation"] = args.heldout_validation
    manifest["episode_split"] = split_manifest
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
    if config.batch_size != 32 or config.fsdp_devices != 2 or jax.device_count() != 2:
        raise ValueError("This experiment requires batch_size=32 and exactly two visible GPUs")
    mesh = sharding.make_mesh(2)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)
    state, state_sharding = init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    trainable_count = sum(
        int(np.prod((leaf.value if hasattr(leaf, "value") else leaf).shape))
        for leaf in jax.tree.leaves(state.params.filter(config.trainable_filter))
    )
    logging.info("Fresh ShellGame MEM trainable parameters: %d (%.2fM)", trainable_count, trainable_count / 1e6)
    logging.info("Generic backbone source (not a MEM checkpoint): %s", _recipe.GENERIC_PI05_CHECKPOINT)
    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=False
    )
    features = prepare_feature_cache(args, state, state_sharding, mesh, data_sharding)
    output_dir = config.checkpoint_dir
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    train_ids = np.concatenate(list(train_domain_ids.values()))
    relation_counts = np.bincount(labels["relations"][train_ids].reshape(-1), minlength=3).astype(np.float64)
    class_weights = (relation_counts.sum() / (3 * relation_counts)).astype(np.float32)
    class_weights /= class_weights.mean()
    ptrain = jax.jit(
        functools.partial(train_step, config, class_weights, args.consistency_loss_weight),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    peval = jax.jit(eval_step, in_shardings=(state_sharding, data_sharding), out_shardings=replicated)
    evaluation_domain_ids = validation_domain_ids if args.heldout_validation else train_domain_ids
    logging.info(
        "Episode split: training=%d validation=%d; validation final counts old/0903/0904=%s/%s/%s",
        sum(len(ids) for ids in train_domain_ids.values()),
        sum(len(ids) for ids in validation_domain_ids.values()),
        split_manifest["validation"]["old306"]["final_counts"],
        split_manifest["validation"]["cup0903"]["final_counts"],
        split_manifest["validation"]["cup0904"]["final_counts"],
    )
    history = []

    def run_eval(step):
        metrics = {"raw": {}, "mild_aug_unseen": {}}
        for domain, ids in evaluation_domain_ids.items():
            metrics["raw"][domain] = evaluate(peval, state, features, labels, ids, data_sharding, augmented=False)
            metrics["mild_aug_unseen"][domain] = evaluate(
                peval, state, features, labels, ids, data_sharding, augmented=True
            )
        record = {"step": step, **metrics, "selection_key": list(_selection_key(metrics))}
        history.append(record)
        (output_dir / "metrics_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        logging.info(
            "Step %d [eval] raw final old/0903/0904=%.4f/%.4f/%.4f; "
            "unseen-mild final=%.4f/%.4f/%.4f; unseen-mild relation=%.4f/%.4f/%.4f",
            step,
            metrics["raw"]["old306"]["final_accuracy"],
            metrics["raw"]["cup0903"]["final_accuracy"],
            metrics["raw"]["cup0904"]["final_accuracy"],
            metrics["mild_aug_unseen"]["old306"]["final_accuracy"],
            metrics["mild_aug_unseen"]["cup0903"]["final_accuracy"],
            metrics["mild_aug_unseen"]["cup0904"]["final_accuracy"],
            metrics["mild_aug_unseen"]["old306"]["relation_accuracy"],
            metrics["mild_aug_unseen"]["cup0903"]["relation_accuracy"],
            metrics["mild_aug_unseen"]["cup0904"]["relation_accuracy"],
        )
        return record

    baseline = run_eval(0)
    best_key = tuple(baseline["selection_key"])
    best_step = 0
    best_params = jax.device_get(state.params.filter(config.trainable_filter))
    no_improvement = 0
    np_rng = np.random.default_rng(args.seed)
    infos = []
    stopped_early = False
    started = time.time()
    for step in range(args.num_train_steps):
        ids, variants = sample_batch_ids(labels, train_domain_ids, np_rng)
        batch = make_batch(features, labels, ids, variants, data_sharding)
        with sharding.set_mesh(mesh):
            state, info = ptrain(train_rng, state, batch)
        infos.append(info)
        completed = step + 1
        if completed % config.log_interval == 0:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            logging.info(
                "Step %d [train] %s",
                completed,
                ", ".join(f"{key}={float(value):.6f}" for key, value in reduced.items()),
            )
            infos = []
        if completed % args.eval_interval == 0 or completed == args.num_train_steps:
            record = run_eval(completed)
            key = tuple(record["selection_key"])
            if key > best_key:
                best_key = key
                best_step = completed
                best_params = jax.device_get(state.params.filter(config.trainable_filter))
                no_improvement = 0
                logging.info("New best fresh MEM at step %d: key=%s", best_step, best_key)
            else:
                no_improvement += 1
            if completed >= args.early_stop_min_step and no_improvement >= args.early_stop_patience:
                logging.info(
                    "Early stopping at step %d after %d evaluations without improvement", completed, no_improvement
                )
                stopped_early = True
                break
    state = _replace_trainable(state, best_params, state_sharding.params.filter(config.trainable_filter))
    final_record = run_eval(best_step)
    _checkpoints.save_state(checkpoint_manager, state, _CheckpointAssetLoader(), best_step)
    checkpoint_manager.wait_until_finished()
    summary = {
        "config": config.name,
        "exp_name": config.exp_name,
        "initialization_contract": "generic Pi0.5 backbone; every ShellGame-specific MEM parameter random",
        "generic_backbone_checkpoint": _recipe.GENERIC_PI05_CHECKPOINT,
        "uses_4999_or_adapted_mem": False,
        "all_427_episodes_used_for_training": not args.heldout_validation,
        "episode_heldout_validation": args.heldout_validation,
        "split_seed": args.split_seed if args.heldout_validation else None,
        "training_episodes": int(sum(len(ids) for ids in train_domain_ids.values())),
        "validation_episodes": int(sum(len(ids) for ids in validation_domain_ids.values())),
        "selected_step": best_step,
        "stopped_early": stopped_early,
        "elapsed_hours_excluding_feature_cache": (time.time() - started) / 3600,
        "batch_contract": "32 = 16 cup0904 + 10 cup0903 + 6 old306, final-cup balanced within each domain",
        "loss_contract": f"initial CE + balanced relation CE + stage CE + {args.consistency_loss_weight} * mild-view JS",
        "selected_validation_metrics": {key: final_record[key] for key in ("raw", "mild_aug_unseen")},
        "selected_metrics": {key: final_record[key] for key in ("raw", "mild_aug_unseen")},
        "checkpoint": str(output_dir / str(best_step)),
        "feature_cache": str(args.feature_cache_dir.resolve()),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("TRAINING_COMPLETE %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
