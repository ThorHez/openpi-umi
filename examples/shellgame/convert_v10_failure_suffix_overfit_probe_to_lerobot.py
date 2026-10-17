"""Audit and convert the six-episode exact-state V10 failure probe."""

# ruff: noqa: SLF001

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROBOSUITE_SCRIPTS = Path(__file__).resolve().parents[3] / "robosuite/robosuite/scripts"
sys.path.insert(0, str(ROBOSUITE_SCRIPTS))

import convert_v10_onpolicy_oracle_correction_to_lerobot as validated
import generate_v10_failure_suffix_overfit_probe_dataset as generator


EXPECTED_EPISODES = 6
EXPECTED_SOURCE_COUNTS = {0: 2, 1: 2, 17: 2}


def _audit_raw(paths: list[Path]) -> dict:
    if len(paths) != EXPECTED_EPISODES:
        raise RuntimeError(f"Expected {EXPECTED_EPISODES} raw episodes, found {len(paths)}")
    source_counts = Counter()
    switch_contacts = Counter()
    offsets = []
    heights = []
    for path in paths:
        episode_dir = path.parent
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("dataset_kind") != generator.DATASET_KIND:
            raise RuntimeError(f"{episode_dir}: wrong dataset_kind")
        source = metadata["source_paired_evaluation"]
        source_counts[int(source["source_episode_index"])] += 1
        if source.get("known_v10_outcome") != "failure":
            raise RuntimeError(f"{episode_dir}: source was not a known V10 failure")
        if metadata["model_prefix"].get("executed_steps") != 80:
            raise RuntimeError(f"{episode_dir}: wrong V10 prefix length")
        if metadata["model_prefix"].get("state_source") != "replayed_recorded_failed_v10_commands_0_79":
            raise RuntimeError(f"{episode_dir}: wrong switch-state source")
        contract = metadata["supervision_contract"]
        if contract.get("model_generated_actions_supervised") is not False:
            raise RuntimeError(f"{episode_dir}: model action entered supervision")
        if metadata["switch"]["selected_cup"] != metadata["switch"]["target_cup"]:
            raise RuntimeError(f"{episode_dir}: V10 selected the wrong cup")
        if metadata["final_ball_cup"] != "right":
            raise RuntimeError(f"{episode_dir}: expected the diagnosed right spatial slot")
        offsets.append(float(metadata["switch"]["offset_m"]))
        heights.append(float(metadata["switch"]["safe_height_m"]))
        switch_contacts[int(metadata["switch"]["actual_contacts_before_oracle"])] += 1
        with np.load(path, allow_pickle=False) as episode:
            if len(episode["actions"]) != validated.EXPECTED_FRAMES:
                raise RuntimeError(f"{path}: wrong frame count")
            mask = np.asarray(episode["action_mask"], dtype=bool)
            source_mask = np.asarray(episode["supervision_source"], dtype=np.uint8) == 1
            expected = np.arange(validated.EXPECTED_FRAMES) >= 61
            if not np.array_equal(mask, expected) or not np.array_equal(source_mask, expected):
                raise RuntimeError(f"{path}: Oracle-only mask contract failed")
    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"Exact-state repetitions mismatch: {dict(source_counts)}")
    return {
        "episodes": len(paths),
        "source_episode_counts": dict(source_counts),
        "frames_per_episode": validated.EXPECTED_FRAMES,
        "offset_m": {
            "min": min(offsets),
            "mean": float(np.mean(offsets)),
            "max": max(offsets),
        },
        "safe_height_m": {
            "min": min(heights),
            "mean": float(np.mean(heights)),
            "max": max(heights),
        },
        "actual_switch_contact_counts": dict(switch_contacts),
        "model_generated_actions_supervised": False,
        "oracle_only_rows": True,
    }


def main() -> None:
    validated.EXPECTED_EPISODES = EXPECTED_EPISODES
    validated.generator.DATASET_KIND = generator.DATASET_KIND
    validated._audit_raw = _audit_raw
    validated.main()


if __name__ == "__main__":
    main()
