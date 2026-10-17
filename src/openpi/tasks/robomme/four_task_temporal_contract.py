"""Shared causal frame contract for the four-task RoboMME teacher/student pipeline.

The Qwen teacher consumes twelve ordered keyframes for every local-event
example. The direct-visual recurrent student also consumes twelve frames per
update, but advances by six frames so event boundaries remain covered by
overlapping causal clips. A short final clip is padded by the data loader and
identified by ``frame_mask``; it must not be dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

TEACHER_FRAME_COUNT = 12
STUDENT_CLIP_FRAME_COUNT = 12
STUDENT_CLIP_STRIDE = 6


@dataclass(frozen=True)
class StudentClipSpec:
    """One chronological student update before tensor padding."""

    start: int
    frame_indices: tuple[int, ...]
    frame_mask: tuple[bool, ...]

    @property
    def end(self) -> int:
        """Exclusive end of the real (unpadded) frames."""

        return self.start + sum(self.frame_mask)


def causal_student_clips(num_frames: int) -> tuple[StudentClipSpec, ...]:
    """Split a causal prefix into overlapping twelve-frame student updates."""

    if num_frames < 0:
        raise ValueError(f"num_frames must be nonnegative, got {num_frames}")
    clips = []
    for start in range(0, num_frames, STUDENT_CLIP_STRIDE):
        stop = min(start + STUDENT_CLIP_FRAME_COUNT, num_frames)
        indices = tuple(range(start, stop))
        valid = len(indices)
        clips.append(
            StudentClipSpec(
                start=start,
                frame_indices=indices,
                frame_mask=(True,) * valid
                + (False,) * (STUDENT_CLIP_FRAME_COUNT - valid),
            )
        )
    return tuple(clips)


def validate_teacher_frame_indices(frame_indices: list[int] | tuple[int, ...]) -> None:
    """Reject manifests that violate the unified twelve-frame teacher input."""

    if len(frame_indices) != TEACHER_FRAME_COUNT:
        raise ValueError(
            f"Teacher examples require {TEACHER_FRAME_COUNT} frames, got {len(frame_indices)}"
        )
    if any(after < before for before, after in pairwise(frame_indices)):
        raise ValueError(f"Teacher frame indices must be chronological, got {frame_indices}")
