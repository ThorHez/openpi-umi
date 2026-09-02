"""Observable VideoPlaceOrder event extraction for training-time supervision.

The extractor consumes only demonstration RGB, robot joints/gripper, and
episode-local target anchors.  It never reads simulator subgoals, object poses,
contacts, success flags, or the queried final region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservablePlaceEvents:
    placement_regions: tuple[int, ...]
    placement_frames: tuple[int, ...]
    relocation_pair: tuple[int, int] | None
    relocation_frame: int | None
    relocation_motion_scores: tuple[float, ...]
    relocation_peak_scores: tuple[float, ...]
    joint_motion_end: int


def color_center(image: np.ndarray, color_id: int) -> np.ndarray | None:
    """Return the simulator-native cube center for 1-based RGB color id."""

    image = np.asarray(image, dtype=np.uint8)
    red, green, blue = (image[..., index] for index in range(3))
    masks = (
        (red > 160) & (green < 110) & (blue < 110),
        (green > 160) & (red < 110) & (blue < 110),
        (blue > 160) & (red < 110) & (green < 110),
    )
    if not 1 <= int(color_id) <= len(masks):
        raise ValueError(f"Expected 1-based RGB color id, got {color_id}")
    y, x = np.nonzero(masks[int(color_id) - 1])
    if len(y) < 20:
        return None
    return np.asarray([np.median(y), np.median(x)], dtype=np.float32)


def _patch_motion(
    frames: np.ndarray,
    anchors_yx: np.ndarray,
    start: int,
    radius: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    sample_frames = np.linspace(
        min(start, len(frames) - 2), len(frames) - 1, 12
    ).round().astype(int)
    images = np.asarray(frames[sample_frames], dtype=np.float32)
    means = []
    peaks = []
    for point in np.asarray(anchors_yx):
        y, x = (int(round(float(value))) for value in point)
        y0, y1 = max(0, y - radius), min(images.shape[1], y + radius + 1)
        x0, x1 = max(0, x - radius), min(images.shape[2], x + radius + 1)
        patches = images[:, y0:y1, x0:x1]
        difference = np.abs(patches - patches[0]).mean(axis=(1, 2, 3))
        means.append(float(difference.mean()))
        peaks.append(float(difference.max()))
    return np.asarray(means, dtype=np.float32), np.asarray(peaks, dtype=np.float32)


def extract_observable_place_events(
    frames: np.ndarray,
    joint_states: np.ndarray,
    gripper_states: np.ndarray,
    anchors_yx: np.ndarray,
    *,
    target_color_id: int,
    close_threshold: float = 0.03,
    open_threshold: float = 0.035,
    confirmation_frames: int = 3,
    region_vote_frames: int = 31,
    region_distance_threshold: float = 30.0,
    joint_motion_threshold: float = 2e-3,
    relocation_peak_threshold: float = 19.0,
) -> ObservablePlaceEvents:
    """Extract completed placement commits and an optional target relocation."""

    frames = np.asarray(frames, dtype=np.uint8)
    joints = np.asarray(joint_states, dtype=np.float32)
    gripper = np.asarray(gripper_states, dtype=np.float32).reshape(-1)
    anchors = np.asarray(anchors_yx, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames [T,H,W,3], got {frames.shape}")
    if joints.ndim != 2 or len(joints) != len(frames):
        raise ValueError(f"Joint/frame mismatch: {joints.shape}, {frames.shape}")
    if len(gripper) != len(frames):
        raise ValueError(f"Gripper/frame mismatch: {gripper.shape}, {frames.shape}")
    if anchors.ndim != 2 or anchors.shape[1] != 2 or not 2 <= len(anchors) <= 4:
        raise ValueError(f"Expected 2-4 target anchors, got {anchors.shape}")

    placement_regions: list[int] = []
    placement_frames: list[int] = []
    closed_run = 0
    open_run = 0
    release_armed = False
    for frame_index, value in enumerate(gripper):
        if value < close_threshold:
            closed_run += 1
            open_run = 0
            if closed_run >= confirmation_frames:
                release_armed = True
            continue
        closed_run = 0
        if not release_armed or value <= open_threshold:
            open_run = 0
            continue
        open_run += 1
        if open_run < confirmation_frames:
            continue
        release_frame = frame_index - open_run + 1
        votes = []
        for probe_index in range(
            release_frame, min(release_frame + region_vote_frames, len(frames))
        ):
            cube = color_center(frames[probe_index], target_color_id)
            if cube is None:
                continue
            distances = np.linalg.norm(anchors - cube, axis=-1)
            region = int(np.argmin(distances))
            if float(distances[region]) < region_distance_threshold:
                votes.append(region)
        if votes:
            counts = np.bincount(votes, minlength=len(anchors))
            region = int(np.argmax(counts))
            if not placement_regions or placement_regions[-1] != region:
                placement_regions.append(region)
                placement_frames.append(release_frame)
        release_armed = False
        open_run = 0

    joint_motion = np.linalg.norm(np.diff(joints, axis=0), axis=-1)
    moving = np.flatnonzero(joint_motion > joint_motion_threshold)
    joint_motion_end = int(moving[-1] + 1) if len(moving) else 0
    relocation_start = max(
        (placement_frames[-1] + 30) if placement_frames else 0,
        joint_motion_end,
    )
    motion_scores, peak_scores = _patch_motion(
        frames, anchors, min(relocation_start, len(frames) - 2)
    )
    moving_regions = np.argsort(motion_scores)[-2:]
    relocation_pair = None
    relocation_frame = None
    if (
        len(moving_regions) == 2
        and float(min(peak_scores[moving_regions])) > relocation_peak_threshold
    ):
        relocation_pair = tuple(sorted(int(value) for value in moving_regions))
        relocation_frame = int(min(relocation_start, len(frames) - 1))

    return ObservablePlaceEvents(
        placement_regions=tuple(placement_regions),
        placement_frames=tuple(placement_frames),
        relocation_pair=relocation_pair,
        relocation_frame=relocation_frame,
        relocation_motion_scores=tuple(float(value) for value in motion_scores),
        relocation_peak_scores=tuple(float(value) for value in peak_scores),
        joint_motion_end=joint_motion_end,
    )
