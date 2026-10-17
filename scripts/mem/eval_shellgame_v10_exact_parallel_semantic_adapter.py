#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate exact V10 while forwarding external semantic memory to a zero adapter."""

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


class V10ExactParallelRemotePolicy(paired.FixedNoiseRemotePolicy):
    """Build V10's exact image/state input and retain semantic_memory."""

    raw_root: Path = paired.base.DEFAULT_RAW_ROOT
    semantic_memory_shape: tuple[int, int] = (128, 64)

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
        semantic_memory = np.asarray(observation["semantic_memory_raw"], dtype=np.float32)
        if state.shape != (10,):
            raise ValueError(f"Expected absolute EEF state10, got {state.shape}")
        if semantic_memory.shape != self.semantic_memory_shape:
            raise ValueError(
                f"Expected semantic memory {self.semantic_memory_shape}, got {semantic_memory.shape}"
            )
        width = np.asarray([state[9]], dtype=np.float32)
        request: dict[str, Any] = {
            "robot0_eef_pos": np.stack((state[:3], np.zeros(3, dtype=np.float32))),
            "robot0_eef_rot_axis_angle": np.stack(
                (state[3:9], np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32))
            ),
            "robot0_gripper_width": np.stack((width, width)),
            "actions": np.zeros((16, 7), dtype=np.float32),
            "semantic_memory": semantic_memory,
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
    V10ExactParallelRemotePolicy.raw_root = raw_root
    paired.FixedNoiseRemotePolicy = V10ExactParallelRemotePolicy
    paired.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["policy_input_contract"] = "exact_v10_fixed_history_plus_parallel_semantic"
    payload["semantic_memory_used_by_policy"] = True
    payload["parallel_semantic_adapter_initial_gate"] = 0.0
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
