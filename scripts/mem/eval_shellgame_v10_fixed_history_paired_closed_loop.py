#!/usr/bin/env python3
"""Run the frozen-MEM paired evaluator against the original V10 input contract.

The simulator, episode reconstruction, diffusion-noise seeds, rollout length,
scoring, and videos are intentionally inherited from
``eval_shellgame_frozen_mem_action_paired_closed_loop``.  Only the remote
observation is adapted from the semantic-memory policy contract to V10's
fixed frames ``[0..59, current]`` contract.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mem import eval_shellgame_frozen_mem_action_paired_closed_loop as paired


def _argument_value(name: str, default: str) -> str:
    for index, value in enumerate(sys.argv[:-1]):
        if value == name:
            return sys.argv[index + 1]
        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
    return default


class OldV10FixedHistoryRemotePolicy(paired.FixedNoiseRemotePolicy):
    """Convert the common live observation into V10's exact 61-frame input."""

    raw_root: Path = paired.base.DEFAULT_RAW_ROOT

    def __init__(self, host: str, port: int, *, salt: int):
        super().__init__(host, port, salt=salt)
        self._base_prefix: np.ndarray | None = None
        self._wrist_prefix: np.ndarray | None = None

    def start_episode(self, episode: int) -> None:
        super().start_episode(episode)
        trajectory = self.raw_root / f"episode_{episode:06d}" / "vla_trajectory.npz"
        with np.load(trajectory, allow_pickle=False) as source:
            self._base_prefix = np.asarray(source["third_person_images"][:60], dtype=np.uint8)
            self._wrist_prefix = np.asarray(source["wrist_images"][:60], dtype=np.uint8)
        expected = (60, paired.base.POLICY_IMAGE_SIZE, paired.base.POLICY_IMAGE_SIZE, 3)
        if self._base_prefix.shape != expected or self._wrist_prefix.shape != expected:
            raise ValueError(
                f"V10 fixed prefix must be {expected}, got "
                f"{self._base_prefix.shape} and {self._wrist_prefix.shape}"
            )

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._base_prefix is None or self._wrist_prefix is None:
            raise RuntimeError("start_episode must load the V10 prefix before infer")
        state = np.asarray(observation["state_raw"], dtype=np.float32).reshape(-1)
        if state.shape != (10,):
            raise ValueError(f"Expected absolute EEF state10, got {state.shape}")
        width = np.asarray([state[9]], dtype=np.float32)
        request: dict[str, Any] = {
            "robot0_eef_pos": np.stack((state[:3], np.zeros(3, dtype=np.float32))),
            "robot0_eef_rot_axis_angle": np.stack(
                (state[3:9], np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32))
            ),
            "robot0_gripper_width": np.stack((width, width)),
            "actions": np.zeros((16, 7), dtype=np.float32),
            "prompt": str(observation["prompt"]),
        }
        current_base = np.asarray(observation["base_rgb"], dtype=np.uint8)
        current_wrist = np.asarray(observation["wrist_rgb"], dtype=np.uint8)
        for frame in range(60):
            request[f"left_wrist_0_rgb_0_{frame}"] = self._wrist_prefix[frame]
            request[f"left_wrist_0_rgb_1_{frame}"] = self._base_prefix[frame]
        request["left_wrist_0_rgb_0_60"] = current_wrist
        request["left_wrist_0_rgb_1_60"] = current_base
        return super().infer(request)


def main() -> None:
    raw_root = Path(
        _argument_value("--raw-root", str(paired.base.DEFAULT_RAW_ROOT))
    ).expanduser().resolve()
    output = Path(
        _argument_value("--output", str(paired.DEFAULT_OUTPUT))
    ).expanduser().resolve()
    OldV10FixedHistoryRemotePolicy.raw_root = raw_root
    paired.FixedNoiseRemotePolicy = OldV10FixedHistoryRemotePolicy
    paired.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["policy_input_contract"] = "original_v10_fixed_frames_0_59_plus_live_current"
    payload["semantic_memory_used_by_policy"] = False
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
