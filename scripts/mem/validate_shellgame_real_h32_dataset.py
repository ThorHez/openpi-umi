#!/usr/bin/env python3
"""Validate the H32 dataset and prove its first 16 targets match H16."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
H16 = ROOT / "data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10"
H32 = ROOT / "data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10_h32"


def episode_path(root: Path, episode: int) -> Path:
    return root / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"


def main() -> None:
    audit = json.loads((H32 / "conversion_audit.json").read_text(encoding="utf-8"))
    info = json.loads((H32 / "meta/info.json").read_text(encoding="utf-8"))
    norm = json.loads((H32 / "norm_stats.json").read_text(encoding="utf-8"))["norm_stats"]
    assert audit["episodes"] == 306
    assert audit["frames"] == 118_807
    assert audit["history_frames"] == 241
    assert audit["action_horizon"] == 32
    assert info["features"]["actions"]["shape"] == [32, 10]
    assert len(norm["actions"]["mean"]) == 10
    assert (
        audit["validation_episode_ids"]
        == json.loads((H16 / "conversion_audit.json").read_text(encoding="utf-8"))["validation_episode_ids"]
    )

    max_prefix_error = 0.0
    checked = []
    for episode in (0, 19, 95, 150, 243, 305):
        columns = ["frame_index", "actions"]
        h16 = pq.read_table(episode_path(H16, episode), columns=columns)
        h32 = pq.read_table(episode_path(H32, episode), columns=columns)
        assert h16.num_rows == h32.num_rows
        assert h16.column("frame_index").to_pylist() == h32.column("frame_index").to_pylist()
        for frame in (0, 241, h16.num_rows - 1):
            a16 = np.asarray(h16.column("actions")[frame].as_py(), dtype=np.float32)
            a32 = np.asarray(h32.column("actions")[frame].as_py(), dtype=np.float32)
            assert a16.shape == (16, 10)
            assert a32.shape == (32, 10)
            max_prefix_error = max(max_prefix_error, float(np.max(np.abs(a16 - a32[:16]))))
        checked.append(episode)
    if max_prefix_error > 1e-7:
        raise AssertionError(f"H16/H32 shared prefix mismatch: {max_prefix_error}")
    print(
        json.dumps(
            {
                "status": "pass",
                "episodes": audit["episodes"],
                "frames": audit["frames"],
                "action_shape": info["features"]["actions"]["shape"],
                "checked_prefix_episodes": checked,
                "max_h16_prefix_abs_error": max_prefix_error,
                "roundtrip_max_position_error_m": audit["roundtrip"]["max_position_error_m"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
