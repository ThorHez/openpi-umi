"""Force action supervision through the query-action memory path.

Only frames 59..70 are retained in both the train and validation episode
splits.  The matching config freezes the Pi0 action expert and every action
projection, leaving only these modules trainable:

* FixedGridTemporalHistory
* HistoryActionQueryResampler
* ActionMemoryCrossAttention

This entry point also extends the trainer's gradient diagnostics without
modifying the shared trainer.
"""

from __future__ import annotations

import logging

import numpy as np
import train_pi0_mem_compress as _trainer

from openpi.training import config as _config

FIRST_DECISION_FRAME = 59
LAST_DECISION_FRAME = 70


def _frame_indices(dataset) -> np.ndarray:
    current = dataset
    while current is not None:
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            if "frame_index" not in getattr(hf_dataset, "column_names", ()):
                break
            return np.asarray(hf_dataset["frame_index"], dtype=np.int64)
        current = getattr(current, "_dataset", None)
    raise ValueError("Memory-only action training requires a frame_index column")


def _decision_rows(indices: list[int], frame_indices: np.ndarray) -> list[int]:
    selected = np.asarray(indices, dtype=np.int64)
    keep = (frame_indices[selected] >= FIRST_DECISION_FRAME) & (
        frame_indices[selected] <= LAST_DECISION_FRAME
    )
    return selected[keep].tolist()


def _decision_only_episode_split(dataset, val_ratio: float, seed: int):
    train_indices, val_indices = _ORIGINAL_EPISODE_SPLIT(dataset, val_ratio, seed)
    frame_indices = _frame_indices(dataset)
    train_decision = _decision_rows(train_indices, frame_indices)
    val_decision = _decision_rows(val_indices, frame_indices)
    logging.info(
        "Memory-only decision split: train=%d, val=%d, frames=%d..%d",
        len(train_decision),
        len(val_decision),
        FIRST_DECISION_FRAME,
        LAST_DECISION_FRAME,
    )
    return train_decision, val_decision


def _keep_preselected_rows(_dataset, indices: list[int], _classifier_config) -> list[int]:
    return indices


def _query_action_memory_grad_metrics(grads):
    metrics = _ORIGINAL_GRAD_METRICS(grads)
    path_leaves = list(_trainer._tree_leaves_with_paths(grads))  # noqa: SLF001

    def select(needle: str):
        return [leaf for path, leaf in path_leaves if needle in path]

    fixed_grid = select("FixedGridTemporalHistory_0")
    query_resampler = select("HistoryActionQueryResampler")
    action_cross_attention = select("ActionMemoryCrossAttention")
    metrics.update(
        {
            "grad/query_memory_path_total_l2": _trainer._global_l2_norm(  # noqa: SLF001
                fixed_grid + query_resampler + action_cross_attention
            ),
            "grad/fixed_grid_temporal_l2": _trainer._global_l2_norm(fixed_grid),  # noqa: SLF001
            "grad/action_query_resampler_l2": _trainer._global_l2_norm(  # noqa: SLF001
                query_resampler
            ),
            "grad/action_memory_cross_attention_l2": _trainer._global_l2_norm(  # noqa: SLF001
                action_cross_attention
            ),
        }
    )
    return metrics


_ORIGINAL_EPISODE_SPLIT = _trainer._episode_split_indices  # noqa: SLF001
_ORIGINAL_GRAD_METRICS = _trainer.new_memory_param_grad_metrics


def main() -> None:
    _trainer._episode_split_indices = _decision_only_episode_split  # noqa: SLF001
    _trainer._filter_memory_classifier_frame_range = _keep_preselected_rows  # noqa: SLF001
    _trainer.new_memory_param_grad_metrics = _query_action_memory_grad_metrics
    _trainer.main(_config.cli())


if __name__ == "__main__":
    main()
