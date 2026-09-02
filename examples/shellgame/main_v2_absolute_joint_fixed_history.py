"""Absolute-joint ShellGame evaluation with frames 0..59 plus current frame.

This is the online counterpart of ``FixedPrefixCurrentVideoDataset`` used by
``train_old_tracker_full_joint_grasp.py``.  It keeps the proven tracker's
60-frame history fixed while allowing the action expert to observe the latest
robot state and image during closed-loop replanning.
"""

from __future__ import annotations

import logging

import numpy as np
import tyro

import main as base
import main_v2_absolute_joint as joint


HISTORY_FRAMES = 60
TOTAL_FRAMES = 61


def _fixed_history_policy_input(
    history: list[dict],
    start_eef_pos: np.ndarray,
    *,
    args: joint.Args,
    prompt: str | None = None,
) -> dict:
    del start_eef_pos
    if args.policy_input_mode != "history":
        raise ValueError("Fixed-history evaluation requires --policy-input-mode history")
    if args.num_frames != TOTAL_FRAMES or args.frame_stride != 1:
        raise ValueError(
            "Fixed-history evaluation requires --num-frames 61 --frame-stride 1; "
            f"got {args.num_frames} and {args.frame_stride}"
        )
    if len(history) < HISTORY_FRAMES:
        raise ValueError(
            f"Need the complete scripted 60-frame history, got only {len(history)} frames"
        )

    current = history[-1]
    frames = [*history[:HISTORY_FRAMES], current]
    if len(frames) != TOTAL_FRAMES:
        raise AssertionError(f"Expected {TOTAL_FRAMES} frames, got {len(frames)}")

    element = {
        "robot0_joint_pos": np.asarray(current["joint_pos"], dtype=np.float32).reshape(1, joint.JOINT_DIM),
        "robot0_gripper_width": np.asarray([[current["gripper_width"]]], dtype=np.float32),
        "actions": np.zeros((args.action_horizon, joint.ACTION_DIM), dtype=np.float32),
        "prompt": args.task if prompt is None else prompt,
    }
    for index, frame in enumerate(frames):
        element[f"left_wrist_0_rgb_0_{index}"] = frame["wrist"]
        element[f"left_wrist_0_rgb_1_{index}"] = frame["base"]
    return element


def main() -> None:
    base._episode_namespace = joint._episode_namespace  # noqa: SLF001
    base._append_observation = joint._append_observation  # noqa: SLF001
    base._policy_action_dim = joint._policy_action_dim  # noqa: SLF001
    base._policy_input = _fixed_history_policy_input  # noqa: SLF001
    base._zero_env_action = joint._zero_env_action  # noqa: SLF001
    base._target_action_to_env_action = joint._absolute_joint_action_to_env_action  # noqa: SLF001

    logging.basicConfig(level=logging.INFO, force=True)
    base.eval_shellgame(tyro.cli(joint.Args))


if __name__ == "__main__":
    main()
