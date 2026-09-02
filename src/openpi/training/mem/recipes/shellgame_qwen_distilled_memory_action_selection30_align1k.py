"""Selection-heavy memory/action alignment run with a moderately larger LR."""

from __future__ import annotations

import dataclasses

from openpi.training import optimizer as _optimizer
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_selection30 as _selection


filter_and_sample_indices = _selection.filter_and_sample_indices


def make_train_config(**kwargs):
    """Keep Selection30 data/model settings and change only the alignment schedule."""
    steps = int(kwargs.get("steps", 1_000))
    config = _selection.make_train_config(**kwargs)
    return dataclasses.replace(
        config,
        name="pi0_shellgame_qwen_distilled_memory_action_selection30_align1k_eef7_260826",
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(50, max(steps - 1, 0)),
            peak_lr=1e-5,
            decay_steps=max(steps, 2),
            decay_lr=1e-6,
        ),
    )
