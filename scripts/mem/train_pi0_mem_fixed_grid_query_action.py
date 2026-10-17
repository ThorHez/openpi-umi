"""Train the fixed-grid MEM query-action model on post-swap joint actions.

The generic MEM trainer performs an episode-level split.  This entry point
additionally keeps only rows at or after frame 59, so the supervised chunks
cover the complete grasp phase while excluding scripted-observation no-op
actions.  Model and data configuration still come from the normal config CLI.
"""

from __future__ import annotations

import numpy as np
import train_pi0_mem_compress as _trainer

from openpi.training import config as _config

FIRST_ACTION_QUERY_FRAME = 59


def _post_swap_action_rows(dataset, indices: list[int], _classifier_config) -> list[int]:
    current = dataset
    hf_dataset = None
    while current is not None:
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            break
        current = getattr(current, "_dataset", None)
    if hf_dataset is None or "frame_index" not in getattr(hf_dataset, "column_names", ()):
        raise ValueError("Post-swap action training requires a frame_index column")
    selected = np.asarray(indices, dtype=np.int64)
    frame_indices = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    filtered = selected[frame_indices[selected] >= FIRST_ACTION_QUERY_FRAME].tolist()
    if not filtered:
        raise ValueError(f"frame_index >= {FIRST_ACTION_QUERY_FRAME} selected no rows")
    return filtered


def main() -> None:
    _trainer._filter_memory_classifier_frame_range = _post_swap_action_rows  # noqa: SLF001
    _trainer.main(_config.cli())


if __name__ == "__main__":
    main()
