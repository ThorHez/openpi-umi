"""Validate compact three-swap visual memory with 60 stride-1 history frames.

At dataset frame 60, the loader returns frames 0..60. Frames 0..59 are the
only tracker history and frame 60 is excluded as the current observation. The
tracker consumes reveal frames 0..19, three contiguous ten-frame swap clips
20..29, 30..39, and 40..49, and deliberately ignores post-swap frames 50..59.

This is an isolated stride-1 variant of the single-history-read experiment.
It restores the proven ten-frame swap encoder without temporal subsampling and
keeps action loss disabled, so held-out accuracy directly tests whether the
complete visual-to-compact-memory path generalizes before action training.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.shellgame import train_three_swap_visual_single_history_read_adapter_probe as _probe

_probe.TOTAL_INPUT_FRAMES = 61
_probe.FRAME_STRIDE = 1
_probe.HISTORY_FRAMES = 60
_probe.REVEAL_END = 20
_probe.SWAP_SLICES = ((20, 30), (30, 40), (40, 50))
_probe.POST_START = 50
_probe.SWAP_SEGMENT_SIZE = 10


if __name__ == "__main__":
    _probe.main()
