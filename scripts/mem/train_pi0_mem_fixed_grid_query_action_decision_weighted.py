"""Train query-action MEM with decision-focused post-swap sampling.

The episode split remains unchanged.  Within the training episodes, rows from
frames 59..70 are repeated 28 times and rows from 71..154 are kept once:

    12 * 28 / (12 * 28 + 84) = 0.8

This makes 80% of training samples come from the phase where the policy must
turn visual memory into a target-cup decision.  Validation remains an
unweighted view of all post-swap rows (59..154).
"""

from __future__ import annotations

import logging

import numpy as np
import train_pi0_mem_compress as _trainer

from openpi.training import config as _config

FIRST_ACTION_FRAME = 59
LAST_DECISION_FRAME = 70
EARLY_REPEAT = 28


def _frame_indices(dataset) -> np.ndarray:
    current = dataset
    while current is not None:
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            if "frame_index" not in getattr(hf_dataset, "column_names", ()):
                break
            return np.asarray(hf_dataset["frame_index"], dtype=np.int64)
        current = getattr(current, "_dataset", None)
    raise ValueError("Decision-weighted action training requires a frame_index column")


def _post_swap_rows(indices: list[int], frame_indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    return selected[frame_indices[selected] >= FIRST_ACTION_FRAME]


def _decision_weighted_episode_split(dataset, val_ratio: float, seed: int):
    train_indices, val_indices = _ORIGINAL_EPISODE_SPLIT(dataset, val_ratio, seed)
    frame_indices = _frame_indices(dataset)
    train_post_swap = _post_swap_rows(train_indices, frame_indices)
    val_post_swap = _post_swap_rows(val_indices, frame_indices)

    decision_mask = frame_indices[train_post_swap] <= LAST_DECISION_FRAME
    decision = train_post_swap[decision_mask]
    continuation = train_post_swap[~decision_mask]
    weighted_train = np.concatenate(
        (np.repeat(decision, EARLY_REPEAT), continuation),
    )

    decision_fraction = len(decision) * EARLY_REPEAT / len(weighted_train)
    logging.info(
        "Decision-weighted split: train=%d (decision=%d x%d, continuation=%d, "
        "effective_decision_fraction=%.3f), val=%d unweighted post-swap rows",
        len(weighted_train),
        len(decision),
        EARLY_REPEAT,
        len(continuation),
        decision_fraction,
        len(val_post_swap),
    )
    return weighted_train.tolist(), val_post_swap.tolist()


def _keep_preselected_rows(_dataset, indices: list[int], _classifier_config) -> list[int]:
    return indices


_ORIGINAL_EPISODE_SPLIT = _trainer._episode_split_indices  # noqa: SLF001


def main() -> None:
    _trainer._episode_split_indices = _decision_weighted_episode_split  # noqa: SLF001
    _trainer._filter_memory_classifier_frame_range = _keep_preselected_rows  # noqa: SLF001
    _trainer.main(_config.cli())


if __name__ == "__main__":
    main()
