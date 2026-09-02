"""Train on the audited optimized-V8 dataset using its actual episode count.

This is intentionally a thin configuration layer over the validated V8
unique-row recipe.  It keeps the 60/30/5/5 global sampling target but derives
the correction episode count and per-row weight from the converted dataset,
instead of assuming that generation reached exactly 1200 episodes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v8 as _v8

CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v8_optimized_260818"
CORRECTION_ROOT = Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_onpolicy_eef_sustained_recovery_v8_optimized_balanced1200_260818"
)
AUDIT_NAME = "sustained_recovery_v8_optimized_oracle_supervision_audit.json"


def _actual_correction_episodes() -> int:
    audit_path = CORRECTION_ROOT / AUDIT_NAME
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing optimized-V8 conversion audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("ok") is not True:
        raise ValueError(f"Optimized-V8 conversion audit did not pass: {audit_path}")
    audited = int(audit["converted_audit"]["converted_episodes"])
    parquet = list(CORRECTION_ROOT.glob("data/chunk-*/episode_*.parquet"))
    if len(parquet) != audited:
        raise ValueError(
            f"Converted episode count mismatch: audit={audited}, parquet={len(parquet)}"
        )
    if audited <= 0:
        raise ValueError("Optimized-V8 correction dataset is empty")
    return audited


def _configure_actual_dataset() -> tuple[int, float]:
    episodes = _actual_correction_episodes()
    weight = (
        _v8.CORRECTION_SAMPLE_FRACTION
        / (1.0 - _v8.CORRECTION_SAMPLE_FRACTION)
        * (_v8.NOMINAL_EPISODES * _v8.NOMINAL_ROWS_PER_EPISODE)
        / (episodes * _v8.CORRECTION_ROWS_PER_EPISODE)
    )
    _v8.CONFIG_NAME = CONFIG_NAME
    _v8.CORRECTION_ROOT = str(CORRECTION_ROOT)
    _v8.CORRECTION_EPISODES = episodes
    _v8.CORRECTION_PER_ROW_WEIGHT = weight
    return episodes, weight


def main() -> None:
    episodes, weight = _configure_actual_dataset()
    logging.info(
        "Optimized-V8 dynamic recipe: episodes=%d selected_rows=%d "
        "correction_per_row_weight=%.9f target_mass=40%%",
        episodes,
        episodes * _v8.CORRECTION_ROWS_PER_EPISODE,
        weight,
    )
    _v8.main()


if __name__ == "__main__":
    main()
