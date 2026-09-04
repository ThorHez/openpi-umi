#!/usr/bin/env python3
"""Convert cup_0903 Zarr v2 data to the deployed H16 EEF10 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mem import convert_real_shellgame_stage2_epfirst as _base  # noqa: E402

WORKSPACE = Path("/data2/hzl_workspace_for_pi_mem")
DEFAULT_INPUT = WORKSPACE / "cup_0903/replay_buffer.zarr"
DEFAULT_LABELS = WORKSPACE / "cup_0903/labels.jsonl"
DEFAULT_OUTPUT = WORKSPACE / "openpi-umi/data/shellgame_real_cup0903_state_epfirst_action_currentrel_eef10"
DEFAULT_AUDIT = WORKSPACE / "openpi-umi/artifacts/shellgame_real_cup0903_stage2_conversion_audit.json"


def _stratified_split(labels: list[dict], seed: int = 42) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    result = {"train": [], "validation": [], "test": []}
    for cup in range(3):
        ids = np.asarray(
            [index for index, row in enumerate(labels) if int(row["final_cup"]) == cup],
            dtype=np.int64,
        )
        ids = rng.permutation(ids)
        num_validation = round(len(ids) * 0.15)
        num_test = round(len(ids) * 0.15)
        result["validation"].extend(ids[:num_validation].tolist())
        result["test"].extend(ids[num_validation : num_validation + num_test].tolist())
        result["train"].extend(ids[num_validation + num_test :].tolist())
    return {name: sorted(ids) for name, ids in result.items()}


def _to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.5 else 1.0
        image = np.clip(np.rint(image * scale), 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def convert(args: argparse.Namespace) -> dict:
    labels = _base._load_labels(args.labels)  # noqa: SLF001
    replay = zarr.open(args.input, mode="r")
    data = replay["data"]
    required = (
        "action",
        "camera0_rgb",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing required Zarr arrays: {missing}")
    episode_ends = np.asarray(replay["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), episode_ends[:-1]))
    if len(labels) != len(episode_ends):
        raise ValueError(f"Label/replay episode mismatch: {len(labels)} vs {len(episode_ends)}")
    total_frames = int(episode_ends[-1])
    for key in required:
        if data[key].shape[0] != total_frames:
            raise ValueError(f"Array {key} has {data[key].shape[0]} rows, expected {total_frames}")
    if data["action"].shape[1:] != (7,):
        raise ValueError(f"Expected raw 7-D action, got {data['action'].shape}")
    if data["camera0_rgb"].shape[1:] != (224, 224, 3):
        raise ValueError(f"Unexpected camera shape {data['camera0_rgb'].shape}")

    contracts = []
    state_values = []
    action_values = []
    stale_counts = []
    length_mismatches = []
    # Reading these compact numeric arrays once is much faster than issuing
    # hundreds of small compressed-Zarr reads.  Images remain streamed below.
    eef_positions = np.asarray(data["robot0_eef_pos"][:])
    eef_rotations = np.asarray(data["robot0_eef_rot_axis_angle"][:])
    gripper_widths = np.asarray(data["robot0_gripper_width"][:]).reshape(-1)
    raw_actions = np.asarray(data["action"][:])
    for episode, (raw_start, raw_end) in enumerate(zip(starts, episode_ends, strict=True)):
        start, end = int(raw_start), int(raw_end)
        length = end - start
        if length <= _base.HISTORY_FRAMES:
            raise ValueError(f"Episode {episode} has no action suffix: length={length}")
        label_length = int(labels[episode].get("n_frames", -1))
        if label_length != length:
            length_mismatches.append(
                {"episode_id": episode, "label_n_frames": label_length, "replay_n_frames": length}
            )
        contract = _base.build_episode_contract(
            eef_positions[start:end],
            eef_rotations[start:end],
            gripper_widths[start:end],
            raw_actions[start:end],
            action_horizon=args.action_horizon,
            max_command_position_error_m=args.max_command_position_error_m,
        )
        contracts.append(contract)
        suffix = slice(_base.HISTORY_FRAMES, length)
        state_values.append(contract.state[suffix])
        action_values.append(contract.actions[suffix].reshape(-1, _base.ACTION_DIM))
        stale_counts.append(int(np.sum(~contract.command_valid[suffix])))

    split = _stratified_split(labels, seed=42)
    lengths = episode_ends - starts
    audit = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "labels": str(args.labels.resolve()),
        "episodes": len(labels),
        "frames": total_frames,
        "episode_length": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "max": int(lengths.max()),
        },
        "history_frames": _base.HISTORY_FRAMES,
        "action_horizon": args.action_horizon,
        "state_contract": "episode_first_relative_link6_pose10_plus_direct_gripper",
        "action_contract": "current_frame_same_anchor_relative_link6_pose10_plus_direct_gripper",
        "label_length_mismatches": {
            "count": len(length_mismatches),
            "episodes": length_mismatches,
        },
        "raw_command_fallback": {
            "max_position_error_m": args.max_command_position_error_m,
            "count_after_history": int(sum(stale_counts)),
            "affected_episodes": int(sum(value > 0 for value in stale_counts)),
            "fraction_after_history": float(
                sum(stale_counts) / sum(int(length) - _base.HISTORY_FRAMES for length in lengths)
            ),
        },
        "roundtrip": {
            "max_position_error_m": max(item.max_roundtrip_position_error_m for item in contracts),
            "max_rotation_matrix_error": max(item.max_roundtrip_rotation_matrix_error for item in contracts),
        },
        "class_counts": {
            "initial_cup": np.bincount([int(row["initial_cup"]) for row in labels], minlength=3).tolist(),
            "final_cup": np.bincount([int(row["final_cup"]) for row in labels], minlength=3).tolist(),
        },
        "episode_split_seed": 42,
        "training_episode_ids": split["train"],
        "validation_episode_ids": split["validation"],
        "test_episode_ids": split["test"],
        "norm_stats": {
            "state": _base._norm_stats(np.concatenate(state_values)),  # noqa: SLF001
            "actions": _base._norm_stats(np.concatenate(action_values)),  # noqa: SLF001
        },
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "norm_stats"}, indent=2))
    if args.audit_only:
        return audit

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {args.output}")
    available = shutil.disk_usage(args.output.parent).free
    required_bytes = int(args.min_free_gib * 1024**3)
    if available < required_bytes:
        raise OSError(
            f"Need at least {args.min_free_gib:.1f} GiB free; only {available / 1024**3:.1f} GiB available"
        )
    dataset = _base._create_lerobot_dataset(  # noqa: SLF001
        args.output,
        workers=args.image_workers,
        action_horizon=args.action_horizon,
        repo_id=args.repo_id,
    )
    camera = data["camera0_rgb"]
    for episode, contract in enumerate(contracts):
        start, end = int(starts[episode]), int(episode_ends[episode])
        label = labels[episode]
        length = end - start
        for local_frame, global_frame in enumerate(range(start, end)):
            dataset.add_frame(
                {
                    "observation.robot0_eef_pos": contract.state[local_frame, :3],
                    "observation.robot0_eef_rot_axis_angle": contract.state[local_frame, 3:9],
                    "observation.robot0_gripper_width": contract.state[local_frame, 9:10],
                    "observation.left_wrist_0_rgb_0": _to_uint8(camera[global_frame]),
                    "actions": contract.actions[local_frame],
                    "episode_length": np.asarray([length], dtype=np.int64),
                    "initial_cup": np.asarray([label["initial_cup"]], dtype=np.int64),
                    "final_cup": np.asarray([label["final_cup"]], dtype=np.int64),
                    "swap_pairs": np.asarray(label["moves"], dtype=np.int64),
                    "task": _base.PROMPT,
                }
            )
        dataset.save_episode()
    (args.output / "norm_stats.json").write_text(
        json.dumps({"norm_stats": audit["norm_stats"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "conversion_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default="local/shellgame_real_cup0903_currentrel_eef10")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--min-free-gib", type=float, default=25.0)
    parser.add_argument("--max-command-position-error-m", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        convert(parse_args())
    except Exception as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        raise
