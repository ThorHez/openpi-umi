from __future__ import annotations

import numpy as np
import pytest

from openpi.training.mem.recipes import shellgame_real_wrist_m6 as m6
from scripts.mem.serve_shellgame_real_m6_cached import M6CachedHistoryPolicy


class _FailingMemoryPolicy:
    def infer_memory(self, _obs):
        raise AssertionError("MEM must not run during a forced-direction action test")


class _RecordingActionPolicy:
    def __init__(self) -> None:
        self.observation = None
        self.noise = None

    def infer(self, obs, *, noise):
        self.observation = obs
        self.noise = noise
        return {"actions": np.zeros((16, 10), dtype=np.float32)}


def _make_policy(*, enabled: bool = True, current_only: bool = True):
    action = _RecordingActionPolicy()
    wrapper = M6CachedHistoryPolicy(
        _FailingMemoryPolicy(),
        action,
        action_horizon=16,
        current_only_action=current_only,
        enable_forced_direction_test=enabled,
    )
    return wrapper, action


@pytest.mark.parametrize(("direction", "cup"), [("left", 0), ("middle", 1), ("right", 2)])
def test_forced_direction_bypasses_history_and_mem(direction: str, cup: int) -> None:
    wrapper, action = _make_policy()

    ready = wrapper.infer({"mode": "set_forced_direction", "direction": direction})
    assert ready["cache_ready"] is True
    assert ready["direct_action_ready"] is True
    assert ready["memory"]["predicted_final_cup"] == cup
    assert ready["memory"]["predicted_final_cup_probabilities"][cup] == 1.0

    result = wrapper.infer(
        {
            "mode": "infer_step",
            "left_wrist_0_rgb_0": np.zeros((224, 224, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_rot_axis_angle": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
            "robot0_gripper_width": np.array([0.1], dtype=np.float32),
            "prompt": "this client prompt must be ignored",
        }
    )

    assert result["actions"].shape == (16, 10)
    assert result["memory"]["direction_source"] == "forced_direction_test"
    assert action.observation["prompt"] == m6.direction_prompt(direction)
    assert "left_wrist_0_rgb_0_0" in action.observation
    assert not any(
        key.startswith("left_wrist_0_rgb_0_") and key != "left_wrist_0_rgb_0_0"
        for key in action.observation
    )
    assert action.noise.shape == (16, 32)


def test_forced_direction_requires_explicit_server_opt_in() -> None:
    wrapper, _ = _make_policy(enabled=False)
    with pytest.raises(RuntimeError, match="disabled"):
        wrapper.infer({"mode": "set_forced_direction", "direction": "left"})


def test_forced_direction_rejects_history_conditioned_action_model() -> None:
    wrapper, _ = _make_policy(current_only=False)
    with pytest.raises(RuntimeError, match="prompt_only"):
        wrapper.infer({"mode": "set_forced_direction", "direction": "right"})
