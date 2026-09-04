"""Shared old306 + cup_0903 contracts for real ShellGame M5/M6 training."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

import openpi.transforms as _transforms

ROOT = Path("/data2/hzl_workspace_for_pi_mem/openpi-umi")
OLD_DATASET_ROOT = ROOT / "data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10"
NEW_DATASET_ROOT = ROOT / "data/shellgame_real_cup0903_state_epfirst_action_currentrel_eef10"
OLD_LABELS_PATH = ROOT.parent / "labels_merged_306_degap.jsonl"
NEW_LABELS_PATH = ROOT.parent / "cup_0903/labels.jsonl"
NEW_AUDIT_PATH = ROOT / "artifacts/shellgame_real_cup0903_stage2_conversion_audit.json"
OLD_EPISODES = 306
NEW_EPISODES = 100
NEW_EPISODE_OFFSET = OLD_EPISODES
SOURCE_PROBABILITIES = (0.25, 0.75)

ADAPTED_MEMORY_CHECKPOINT = (
    ROOT
    / "checkpoints/pi0_mem_shellgame_real_relation_adapt_new75_old25/"
    "cup0903_new75_old25_relation_only_lr1e5_b32_seed42_v1/500/params"
)
OLD_H16_ACTION_CHECKPOINT = (
    ROOT
    / "checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_stage1/"
    "real306_m6_direction_stage1_frame241_dirloss010_b32_seed42_v1_best_direction/1199/params"
)


def _load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    episode_ids = [int(row["episode_id"]) for row in rows]
    if episode_ids != list(range(len(rows))):
        raise ValueError(f"Expected contiguous episode ids in {path}")
    return rows


def source_final_cups() -> tuple[tuple[int, ...], tuple[int, ...]]:
    old = tuple(int(row["final_cup"]) for row in _load_rows(OLD_LABELS_PATH))
    new = tuple(int(row["final_cup"]) for row in _load_rows(NEW_LABELS_PATH))
    if len(old) != OLD_EPISODES or len(new) != NEW_EPISODES:
        raise ValueError(f"Expected {OLD_EPISODES}/{NEW_EPISODES} labels, got {len(old)}/{len(new)}")
    if any(value not in (0, 1, 2) for value in (*old, *new)):
        raise ValueError("final_cup labels must be 0, 1, or 2")
    return old, new


def global_final_cups() -> tuple[int, ...]:
    old, new = source_final_cups()
    return old + new


def source_splits() -> tuple[tuple[set[int], set[int]], tuple[set[int], set[int]]]:
    old_audit = json.loads((OLD_DATASET_ROOT / "conversion_audit.json").read_text(encoding="utf-8"))
    old_validation = {int(value) for value in old_audit["validation_episode_ids"]}
    old_training = set(range(OLD_EPISODES)) - old_validation

    new_audit = json.loads(NEW_AUDIT_PATH.read_text(encoding="utf-8"))
    new_training = {int(value) for value in new_audit["training_episode_ids"]}
    new_validation = {int(value) for value in new_audit["validation_episode_ids"]}
    new_test = {int(value) for value in new_audit["test_episode_ids"]}
    if new_training & new_validation or new_training & new_test or new_validation & new_test:
        raise ValueError("cup_0903 train/validation/test episode splits overlap")
    if new_training | new_validation | new_test != set(range(NEW_EPISODES)):
        raise ValueError("cup_0903 episode split is incomplete")
    return (old_training, old_validation), (new_training, new_validation)


def _dataset_episode_and_frame_indices(dataset) -> tuple[np.ndarray, np.ndarray]:
    current = dataset
    hf_dataset = None
    sample_indices = None
    while current is not None:
        if sample_indices is None:
            sample_indices = getattr(current, "sample_indices", None)
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            break
        current = getattr(current, "_dataset", None)
    if hf_dataset is None:
        raise ValueError("Could not find the underlying HuggingFace dataset")
    episodes = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    frames = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    if sample_indices is not None:
        mapped = np.asarray(sample_indices, dtype=np.int64)
        episodes, frames = episodes[mapped], frames[mapped]
    if episodes.shape != (len(dataset),) or frames.shape != (len(dataset),):
        raise ValueError("Underlying episode/frame indices do not match the video dataset")
    return episodes, frames


def fixed_episode_split_indices(dataset, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Use audited episode splits and permanently exclude the new final-test set."""
    del val_ratio, seed
    episodes, _ = _dataset_episode_and_frame_indices(dataset)
    episode_count = len(np.unique(episodes))
    splits = source_splits()
    if episode_count == OLD_EPISODES:
        training, validation = splits[0]
    elif episode_count == NEW_EPISODES:
        training, validation = splits[1]
    else:
        raise ValueError(f"Unknown mixed-training source with {episode_count} episodes")
    train_indices = np.flatnonzero(np.isin(episodes, list(training))).tolist()
    val_indices = np.flatnonzero(np.isin(episodes, list(validation))).tolist()
    if not train_indices or not val_indices:
        raise ValueError("Audited episode split produced an empty train or validation subset")
    return train_indices, val_indices


def filter_balanced_indices(
    dataset,
    indices: list[int],
    classifier_config,
    *,
    decision_frame: int | None,
) -> list[int]:
    """Balance left/middle/right independently inside each source and split."""
    del classifier_config
    episodes, frames = _dataset_episode_and_frame_indices(dataset)
    selected = np.asarray(indices, dtype=np.int64)
    if decision_frame is not None:
        selected = selected[frames[selected] == decision_frame]
    old_labels, new_labels = source_final_cups()
    episode_count = len(np.unique(episodes))
    labels = old_labels if episode_count == OLD_EPISODES else new_labels if episode_count == NEW_EPISODES else None
    if labels is None:
        raise ValueError(f"Unknown mixed-training source with {episode_count} episodes")
    selected_labels = np.asarray(labels, dtype=np.int64)[episodes[selected]]
    groups = [selected[selected_labels == cup] for cup in range(3)]
    per_class = min(map(len, groups))
    if per_class <= 0:
        raise ValueError("Balanced mixed sampler found an empty cup class")
    rng = np.random.default_rng(42 + episode_count + (decision_frame or 0))
    balanced = np.concatenate([rng.permutation(group)[:per_class] for group in groups])
    return rng.permutation(balanced).tolist()


def balanced_source_row_counts(*, decision_frame: int | None) -> tuple[int, int]:
    """Return post-balance train-row counts without loading image datasets."""
    labels_by_source = source_final_cups()
    splits = source_splits()
    roots = (OLD_DATASET_ROOT, NEW_DATASET_ROOT)
    counts = []
    for labels, (training, _), root in zip(labels_by_source, splits, roots, strict=True):
        if decision_frame is not None:
            class_counts = [sum(labels[episode] == cup for episode in training) for cup in range(3)]
        else:
            episode_rows = {
                int(row["episode_index"]): max(0, int(row["length"]) - 241)
                for row in _load_jsonl(root / "meta/episodes.jsonl")
            }
            class_counts = [
                sum(episode_rows[episode] for episode in training if labels[episode] == cup)
                for cup in range(3)
            ]
        counts.append(3 * min(class_counts))
    return int(counts[0]), int(counts[1])


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sampler_weights(*, decision_frame: int | None) -> list[float]:
    """Per-row weights that realize the requested 25% old / 75% new mix."""
    old_rows, new_rows = balanced_source_row_counts(decision_frame=decision_frame)
    return [SOURCE_PROBABILITIES[0] / old_rows, SOURCE_PROBABILITIES[1] / new_rows]


@dataclasses.dataclass(frozen=True)
class OffsetEpisodeIndex(_transforms.DataTransformFn):
    offset: int

    def __call__(self, data: dict) -> dict:
        if "episode_index" not in data:
            raise KeyError("Mixed ShellGame training requires episode_index")
        data["episode_index"] = np.asarray(data["episode_index"]) + self.offset
        return data


_DATA_CONFIG_TYPES: dict[type, type] = {}


def mixed_data_config_type(base_type: type) -> type:
    """Extend a Stage2/M6 data config with a globally unique episode id."""
    if concrete := _DATA_CONFIG_TYPES.get(base_type):
        return concrete

    @dataclasses.dataclass(frozen=True)
    class MixedEpisodeOffsetDataConfig(base_type):
        episode_index_offset: int = 0

        def create(self, assets_dirs, model_config):
            config = super().create(assets_dirs, model_config)
            return dataclasses.replace(
                config,
                data_transforms=_transforms.Group(
                    inputs=(OffsetEpisodeIndex(self.episode_index_offset), *config.data_transforms.inputs),
                    outputs=config.data_transforms.outputs,
                ),
            )

    MixedEpisodeOffsetDataConfig.__module__ = __name__
    MixedEpisodeOffsetDataConfig.__qualname__ = "MixedEpisodeOffsetDataConfig"
    _DATA_CONFIG_TYPES[base_type] = MixedEpisodeOffsetDataConfig
    return MixedEpisodeOffsetDataConfig


def make_dataset_factories(config_module: Any, base_type: type, *, min_frame: int, max_frame: int | None):
    data_cls = mixed_data_config_type(base_type)
    action_mask = (1.0,) * 10 + (0.0,) * 22
    return [
        data_cls(
            repo_id=str(root),
            assets=config_module.AssetsConfig(asset_id=".", assets_dir=str(root)),
            base_config=config_module.UmiDataConfig(
                action_loss_mask=action_mask,
                robot_type="ARM=1 G=0 H=0",
            ),
            min_frame_index=min_frame,
            max_frame_index=max_frame,
            episode_index_offset=offset,
        )
        for root, offset in ((OLD_DATASET_ROOT, 0), (NEW_DATASET_ROOT, NEW_EPISODE_OFFSET))
    ]
