"""Standalone trainer for causal sliding-window semantic event memory.

This entry point uses the same sharded optimizer, episode-held-out validation,
checkpoint, and logging loop as ``train_semantic_memory.py`` while selecting
the event-memory model and objective from its own recipe.

Example::

    uv run python scripts/mem/train_event_semantic_memory.py \
        shellgame_event_semantic_memory_pretrain \
        --exp-name=event_memory_v1
"""

from __future__ import annotations

import train_semantic_memory as _shared_trainer

from openpi.training.mem.recipes import shellgame_event_semantic_memory_pretrain as _recipe


def main(config: _recipe.ShellGameSemanticMemoryPretrainConfig):
    if config.event_loss_weight < 0:
        raise ValueError("event_loss_weight must be nonnegative")
    # The shared loop resolves its recipe module dynamically in train/eval
    # steps. Select the new recipe without modifying the legacy entry point.
    _shared_trainer._recipe = _recipe  # noqa: SLF001
    _shared_trainer.main(config)


if __name__ == "__main__":
    main(_recipe.cli())
