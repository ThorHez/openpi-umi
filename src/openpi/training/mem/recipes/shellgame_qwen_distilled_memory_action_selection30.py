"""Selection-heavy control variant of the frozen direct-MEM V10 recipe.

This changes only nominal phase sampling.  The original nominal sampler emits
35 rows for each of five action phases (20% each).  This variant retains the
same 175 rows per episode but emits 87 selection rows and 22 rows for each
later phase.  With the unchanged 60% nominal source mass, frame-59 selection
supervision therefore increases from 12% to approximately 30% globally.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_v10 as _base


NOMINAL_GROUP_NAMES = ("selection", "approach", "descend", "grasp", "lift")
NOMINAL_GROUP_ROWS_PER_EPISODE = (87, 22, 22, 22, 22)

if sum(NOMINAL_GROUP_ROWS_PER_EPISODE) != _base.ROWS_PER_CORRECT_EPISODE[_base.NOMINAL_ROOT]:
    raise RuntimeError("Selection-heavy nominal sampler must preserve the original row count")


def _selection_heavy_nominal_indices(dataset, indices: list[int]) -> list[int]:
    hf = _base._full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    episode = np.asarray(hf["episode_index"], dtype=np.int64)[selected]
    frame = np.asarray(hf["frame_index"], dtype=np.int64)[selected]
    expected_episodes = np.unique(episode)
    masks = (
        frame == 59,
        (frame >= 60) & (frame <= 88),
        (frame >= 89) & (frame <= 108),
        (frame >= 109) & (frame <= 118),
        (frame >= 119) & (frame <= 153),
    )
    groups = tuple(
        _base._v10._v9._resize_each_episode(  # noqa: SLF001
            selected[mask],
            episode[mask],
            expected_episodes,
            rows_per_episode,
            seed=261100 + group_index,
            label=f"selection30 nominal {name}",
        )
        for group_index, (name, rows_per_episode, mask) in enumerate(
            zip(NOMINAL_GROUP_NAMES, NOMINAL_GROUP_ROWS_PER_EPISODE, masks, strict=True)
        )
    )
    merged = np.concatenate(groups)
    expected = len(expected_episodes) * sum(NOMINAL_GROUP_ROWS_PER_EPISODE)
    if len(merged) != expected:
        raise RuntimeError(f"Selection-heavy nominal sampler emitted {len(merged)} rows; expected {expected}")
    logging.info(
        "Selection-heavy nominal sampling: episodes=%d per_episode=%s output=%d "
        "nominal_selection_fraction=%.6f global_selection_mass=%.6f",
        len(expected_episodes),
        dict(zip(NOMINAL_GROUP_NAMES, NOMINAL_GROUP_ROWS_PER_EPISODE, strict=True)),
        len(merged),
        NOMINAL_GROUP_ROWS_PER_EPISODE[0] / sum(NOMINAL_GROUP_ROWS_PER_EPISODE),
        _base.SOURCE_FRACTIONS[_base.NOMINAL_ROOT]
        * NOMINAL_GROUP_ROWS_PER_EPISODE[0]
        / sum(NOMINAL_GROUP_ROWS_PER_EPISODE),
    )
    return np.random.default_rng(261105 + len(selected)).permutation(merged).tolist()


def filter_and_sample_indices(
    dataset,
    indices: list[int],
    classifier_config,
    *,
    banks=None,
) -> list[int]:
    """Apply the correctness filter, then change only nominal phase mass."""
    banks = _base.DEFAULT_MEMORY_BANKS if banks is None else banks
    source = _base._v10._v9._dataset_repo_id(dataset)  # noqa: SLF001
    if source not in banks:
        raise ValueError(f"Unknown frozen-MEM action source: {source}")
    hf = _base._full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)[selected]
    correct = _base._bank_correctness(str(banks[source].resolve()))  # noqa: SLF001
    if np.any(episodes < 0) or np.any(episodes >= len(correct)):
        raise ValueError(f"Episode index exceeds memory bank for {source}")
    kept = selected[correct[episodes]].tolist()
    logging.info(
        "Frozen-MEM correctness filter source=%s rows=%d->%d episodes=%d->%d",
        source.name,
        len(indices),
        len(kept),
        len(np.unique(episodes)),
        len(np.unique(episodes[correct[episodes]])),
    )
    if source == _base.NOMINAL_ROOT:
        return _selection_heavy_nominal_indices(dataset, kept)
    return _base._v10._indices(dataset, kept, classifier_config)  # noqa: SLF001


def make_train_config(**kwargs):
    config = _base.make_train_config(**kwargs)
    return dataclasses.replace(
        config,
        name="pi0_shellgame_qwen_distilled_memory_action_v10_selection30_eef7_260826",
    )

