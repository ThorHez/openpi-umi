"""Generate balanced low-height sustained absolute-EEF recovery episodes.

V8 deliberately reuses the validated V6 environment, rendering, fixed-history,
and Oracle suffix implementation, but changes the *state distribution*:

* all perturbations occur at low descent heights (15--75 mm above grasp);
* radii are balanced as 25/45/30% over 5--12 / 12--22 / 22--35 mm;
* 16 angular sectors cover both signs of X/Y and all diagonals;
* final spatial slots are exactly balanced; and
* every accepted episode has at least 12 consecutive open-gripper Oracle
  recovery commands, so training can select unique rows without replication.

Only the post-perturbation observation is serialized.  Anchor, learned-policy,
and perturbation commands remain absent from supervision.  From the following
command onward the trajectory is consecutive Oracle control through grasp and
lift, preserving every horizon=16 target window.
"""

# The private V6 entry points are reused intentionally to preserve its audited
# serialization and visual-integrity contracts.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import math

import generate_onpolicy_eef_low_stage_gated_dataset_v6 as _v6
import numpy as np

_ORIGINAL_PARSE_ARGS = _v6.parse_args
_ORIGINAL_VALIDATE_ARGS = _v6._validate_args
_ORIGINAL_ATTEMPT = _v6._attempt

DATASET_KIND = "onpolicy_eef_sustained_recovery_v8"
EXPECTED_EPISODE_FRAMES = _v6.EXPECTED_EPISODE_FRAMES
CORRECTION_COMMANDS = _v6.CORRECTION_COMMANDS
DEFAULT_PREFIX_STEPS = _v6.DEFAULT_PREFIX_STEPS
FINAL_SPATIAL_SLOTS = _v6.FINAL_SPATIAL_SLOTS

# These are deliberately lower than V6's 30--160 mm range.  The proportions
# are 15/35/50%, emphasizing the mid/late descent states seen in failures.
ANCHOR_BANDS_MM = {
    "high": (55.0, 75.0),
    "mid": (35.0, 55.0),
    "late": (15.0, 35.0),
}
ANCHOR_STAGE_CYCLE = ("high",) * 15 + ("mid",) * 35 + ("late",) * 50

OFFSET_BINS_MM = {
    "small": (5.0, 12.0),
    "medium": (12.0, 22.0),
    "large": (22.0, 35.0),
}
OFFSET_BIN_CYCLE = ("small",) * 25 + ("medium",) * 45 + ("large",) * 30
NUM_OFFSET_SECTORS = 16
MIN_UNIQUE_RECOVERY_ROWS = 12

legacy = _v6.legacy
VisualCorruptionError = _v6.VisualCorruptionError
_VisualGuard = _v6._VisualGuard


def parse_args() -> argparse.Namespace:
    """Use V6's stable CLI; V8-specific defaults are enforced after parsing."""
    args = _ORIGINAL_PARSE_ARGS()
    if args.dataset_seed == 260819:
        args.dataset_seed = 260821
    if args.offset_bin_tolerance_mm == 3.0:
        args.offset_bin_tolerance_mm = 2.0
    return args


def _prefix_steps(args: argparse.Namespace) -> tuple[int, ...]:
    return _v6._prefix_steps(args)


def _balanced_design(args: argparse.Namespace, accepted_index: int) -> dict:
    """Assign deterministic slot/stage/radius/direction quotas."""
    slot_index = accepted_index % len(FINAL_SPATIAL_SLOTS)
    within_slot = accepted_index // len(FINAL_SPATIAL_SLOTS)
    prefixes = _prefix_steps(args)
    # Multipliers are coprime to 100 / 16 and decorrelate the axes.
    stage_index = (within_slot * 37 + slot_index * 23) % 100
    offset_index = (within_slot * 29 + slot_index * 17) % 100
    return {
        "final_spatial_slot": FINAL_SPATIAL_SLOTS[slot_index],
        "within_spatial_slot_index": within_slot,
        "prefix_steps": prefixes[(within_slot + slot_index) % len(prefixes)],
        "anchor_stage": ANCHOR_STAGE_CYCLE[stage_index],
        "offset_bin": OFFSET_BIN_CYCLE[offset_index],
        "offset_sector": (within_slot * 5 + 3 * slot_index) % NUM_OFFSET_SECTORS,
    }


def _episode_design(
    args: argparse.Namespace,
    *,
    attempt_index: int,
    accepted_index: int,
) -> tuple[dict, float, np.ndarray, np.ndarray]:
    design = _balanced_design(args, accepted_index)
    rng = np.random.default_rng(
        np.random.SeedSequence([args.dataset_seed, attempt_index, accepted_index, 823])
    )
    anchor_low, anchor_high = ANCHOR_BANDS_MM[design["anchor_stage"]]
    anchor_height_m = float(rng.uniform(anchor_low, anchor_high) / 1_000.0)

    offset_low, offset_high = OFFSET_BINS_MM[design["offset_bin"]]
    angle = 2.0 * math.pi * design["offset_sector"] / NUM_OFFSET_SECTORS
    angle += float(rng.uniform(-math.pi / 32.0, math.pi / 32.0))
    radius_m = float(rng.uniform(offset_low, offset_high) / 1_000.0)
    perturb_xy = radius_m * np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)

    jitter_radius = float(rng.uniform(0.0, args.descent_jitter_mm / 1_000.0))
    jitter_angle = float(rng.uniform(-math.pi, math.pi))
    descent_jitter_xy = jitter_radius * np.asarray(
        [math.cos(jitter_angle), math.sin(jitter_angle)], dtype=np.float64
    )
    return design, anchor_height_m, perturb_xy, descent_jitter_xy


def _configure_v6_backend() -> None:
    """Install V8 design hooks in this process without changing the V6 file."""
    _v6.DATASET_KIND = DATASET_KIND
    _v6.ANCHOR_BANDS_MM = ANCHOR_BANDS_MM
    _v6.ANCHOR_STAGE_CYCLE = ANCHOR_STAGE_CYCLE
    _v6.OFFSET_BINS_MM = OFFSET_BINS_MM
    _v6.OFFSET_BIN_CYCLE = OFFSET_BIN_CYCLE
    _v6._balanced_design = _balanced_design
    _v6._episode_design = _episode_design


def _attempt(*args, **kwargs):
    """Reject short or weak suffixes rather than duplicating their few rows."""
    _configure_v6_backend()
    audit, payload = _ORIGINAL_ATTEMPT(*args, **kwargs)
    if payload is None:
        return audit, None
    metadata = payload[5]
    open_steps = int(metadata["oracle"]["open_steps"])
    measured_offset_m = float(metadata["perturbation"]["measured_offset_m"])
    if open_steps < MIN_UNIQUE_RECOVERY_ROWS:
        audit = dict(audit)
        audit["reason"] = "insufficient_unique_recovery_rows"
        audit["open_steps"] = open_steps
        return audit, None
    if measured_offset_m < OFFSET_BINS_MM["small"][0] / 1_000.0:
        audit = dict(audit)
        audit["reason"] = "measured_offset_below_5mm"
        return audit, None
    metadata["dataset_kind"] = DATASET_KIND
    metadata["v8_distribution"] = {
        "num_offset_sectors": NUM_OFFSET_SECTORS,
        "minimum_unique_recovery_rows": MIN_UNIQUE_RECOVERY_ROWS,
        "low_height_only": True,
        "row_replication_allowed": False,
    }
    return audit, payload


def _validate_args(args: argparse.Namespace) -> None:
    _configure_v6_backend()
    _ORIGINAL_VALIDATE_ARGS(args)
    if args.num_episodes == 1200 and args.num_episodes % NUM_OFFSET_SECTORS:
        raise ValueError("A full V8 dataset must balance all 16 offset sectors")
    if args.max_open_steps < MIN_UNIQUE_RECOVERY_ROWS:
        raise ValueError(
            f"max_open_steps must be >= {MIN_UNIQUE_RECOVERY_ROWS} for unique-row sampling"
        )


def main() -> None:
    _configure_v6_backend()
    # The serial path remains useful for debugging.  Full generation should use
    # the V8 parallel driver.
    _v6.parse_args = parse_args
    _v6._validate_args = _validate_args
    _v6._attempt = _attempt
    _v6.main()


if __name__ == "__main__":
    main()
