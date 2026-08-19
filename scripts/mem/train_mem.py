"""Unified training entry point for compressed and semantic Pi0 memory models.

This stable entry point delegates to the mature Pi0MemCompress trainer, which
already provides the video-aware data loader, validation, checkpointing and
memory diagnostics required by :class:`Pi0MemSemanticActionConfig`.

Example::

    uv run python scripts/mem/train_mem.py pi0_mem_semantic_action_shellgame_eef7 \
        --exp-name=my_run
"""

from train_pi0_mem_compress import main

from openpi.training import config as _config

if __name__ == "__main__":
    main(_config.cli())
