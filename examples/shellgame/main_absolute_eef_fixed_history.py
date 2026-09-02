"""Absolute-EEF7 ShellGame evaluation with frames 0..59 plus current frame.

This is the online counterpart of ``FixedPrefixCurrentVideoDataset`` used by
``train_old_tracker_full_absolute_eef.py``.  The first 60 frames are never
shifted during closed-loop replanning; only frame 60 is replaced by the latest
observation.
"""

from __future__ import annotations

import logging

import main as base
import numpy as np
import tyro

HISTORY_FRAMES = 60
TOTAL_FRAMES = 61


def _fixed_history_policy_input(
    history: list[dict],
    start_eef_pos: np.ndarray,
    *,
    args: base.Args,
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
    if args.action_mode != "raw7" or args.action_dim != 7:
        raise ValueError(
            "Absolute-EEF evaluation requires --action-mode raw7 --action-dim 7"
        )
    if args.observation_position_frame != "absolute":
        raise ValueError(
            "Absolute-EEF evaluation requires --observation-position-frame absolute"
        )
    if args.osc_input_type != "absolute":
        raise ValueError("Absolute-EEF raw7 evaluation requires --osc-input-type absolute")
    if len(history) < HISTORY_FRAMES:
        raise ValueError(
            f"Need the complete scripted 60-frame history, got {len(history)} frames"
        )

    current = history[-1]
    frames = [*history[:HISTORY_FRAMES], current]
    rot6d = base._quat_to_rot6d(current["eef_quat"], args.rot6d_convention)  # noqa: SLF001
    identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    width = np.array([current["gripper_width"]], dtype=np.float32)
    element = {
        "robot0_eef_pos": np.stack(
            [np.asarray(current["eef_pos"], dtype=np.float32), np.zeros(3, dtype=np.float32)],
            axis=0,
        ),
        "robot0_eef_rot_axis_angle": np.stack([rot6d, identity_rot6d], axis=0),
        "robot0_gripper_width": np.stack([width, width], axis=0),
        "actions": np.zeros((args.action_horizon, 7), dtype=np.float32),
        "prompt": args.task if prompt is None else prompt,
    }
    for index, frame in enumerate(frames):
        element[f"left_wrist_0_rgb_0_{index}"] = frame["wrist"]
        element[f"left_wrist_0_rgb_1_{index}"] = frame["base"]
    return element


def _current_only_compat_policy_input(
    history: list[dict],
    start_eef_pos: np.ndarray,
    *,
    args: base.Args,
    prompt: str | None = None,
) -> dict:
    """Populate V10's legacy frame keys without exposing temporal history.

    The checkpoint transform requires 61 indexed image keys.  Repeating the
    live observation in every slot satisfies that shape contract while making
    it impossible for the model input to recover the scripted ShellGame
    sequence from those slots.
    """
    element = _fixed_history_policy_input(
        history, start_eef_pos, args=args, prompt=prompt
    )
    current = history[-1]
    for index in range(TOTAL_FRAMES):
        element[f"left_wrist_0_rgb_0_{index}"] = current["wrist"]
        element[f"left_wrist_0_rgb_1_{index}"] = current["base"]
    return element


def main() -> None:
    base._policy_input = _fixed_history_policy_input  # noqa: SLF001
    logging.basicConfig(level=logging.INFO, force=True)
    base.eval_shellgame(tyro.cli(base.Args))


if __name__ == "__main__":
    main()
