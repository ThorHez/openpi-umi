"""Convert single-target VideoUnmask HDF5 episodes to Pi EEF action chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training.mem import robomme_videounmask_action_dataset as _action_data

ROBOSUITE_SCRIPTS = Path(__file__).resolve().parents[3] / "robosuite/robosuite/scripts"
sys.path.insert(0, str(ROBOSUITE_SCRIPTS))
import convert_shellgame_to_openpi_umi_v2_openpi_action as _lerobot  # noqa: E402

ACTION_HORIZON = 16
ACTION_DIM = 7
BASE_STATE_DIM = 9
DEFAULT_H5 = Path("data/robomme_extracted/record_dataset_VideoUnmask.h5")
DEFAULT_LABELS = Path("data/robomme_extracted/videounmask_semantic_labels_seed260823.json")
DEFAULT_OUTPUT = Path("data/robomme_videounmask_lerobot_pi_action_train")
PROMPT = "Pick up and lift the container identified by the target point."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "internal_val", "all_single"), default="train")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--png-compress-level", type=int, choices=range(10), default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--target-mode", choices=("absolute", "phase_waypoint_delta"), default="absolute"
    )
    parser.add_argument(
        "--target-conditioning",
        choices=("image_yx", "world_xy"),
        default="image_yx",
        help=(
            "Two-value action conditioner. world_xy uses the demonstrated contact waypoint "
            "and removes the pixel-to-robot calibration burden from the action expert."
        ),
    )
    parser.add_argument("--phase-conditioned", action="store_true")
    parser.add_argument("--recovery-augmentations", type=int, default=0)
    return parser.parse_args()


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return str(value)


def _episode_indices(payload: dict, split: str) -> list[int]:
    by_index = {int(item["episode_index"]): item for item in payload["episodes"]}
    if split == "train":
        candidates = payload["train_episode_indices"]
    elif split == "internal_val":
        candidates = payload["val_episode_indices"]
    else:
        candidates = [item["episode_index"] for item in payload["episodes"]]
    return [int(index) for index in candidates if int(by_index[int(index)]["num_targets"]) == 1]


def _phase_id(action: np.ndarray) -> np.ndarray:
    """Coarse phase labels used only for balanced row sampling."""
    z = action[:, 2]
    closed = action[:, 6] < 0.0
    phase = np.zeros(len(action), dtype=np.int64)
    phase[(~closed) & (z <= 0.22)] = 1
    phase[(~closed) & (z <= 0.10)] = 2
    phase[closed & (z <= 0.10)] = 3
    phase[closed & (z > 0.10)] = 4
    return phase


def _feature_info(image_size: int, state_dim: int) -> dict:
    image_shape = [image_size, image_size, 3]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [1, state_dim],
            "names": ["state"],
            "_type": "Array2D",
        },
        "actions": {
            "dtype": "float32",
            "shape": [ACTION_HORIZON, ACTION_DIM],
            "names": ["actions"],
            "_type": "Array2D",
        },
        "observation.front_rgb": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
            "_type": "Image",
        },
        "observation.wrist_rgb": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
            "_type": "Image",
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "_type": "Value"},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "_type": "Value"},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "_type": "Value"},
        "index": {"dtype": "int64", "shape": [1], "names": None, "_type": "Value"},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "_type": "Value"},
        "action_mask": {"dtype": "bool", "shape": [1], "names": None, "_type": "Value"},
        "phase_id": {"dtype": "int64", "shape": [1], "names": None, "_type": "Value"},
        "episode_T": {"dtype": "int64", "shape": [1], "names": None, "_type": "Value"},
    }


def _episode_arrays(
    episode,
    metadata: dict,
    *,
    target_mode: str,
    target_conditioning: str,
    phase_conditioned: bool,
    recovery_augmentations: int,
) -> dict[str, np.ndarray]:
    start = int(metadata["execution_start"])
    end = int(metadata["num_steps"])
    if end - start <= ACTION_HORIZON:
        raise ValueError(f"Episode {metadata['episode_index']} is too short for horizon {ACTION_HORIZON}")
    observation_steps = np.arange(start, end - 1, dtype=np.int64)
    raw_actions = np.stack(
        [np.asarray(episode[f"timestep_{step}/action/eef_action"][()], dtype=np.float32) for step in range(start, end)]
    )
    observation_poses = np.stack(
        [
            np.asarray(episode[f"timestep_{step}/obs/eef_state"][()], dtype=np.float32)
            for step in observation_steps
        ]
    )
    close_targets = raw_actions[1 : len(observation_steps) + 1, 6] < 0.0
    phase_id = _action_data._episode_phases(observation_poses[:, 2], close_targets)  # noqa: SLF001
    if raw_actions.shape[1:] != (ACTION_DIM,) or not np.all(np.isfinite(raw_actions)):
        raise ValueError(f"Invalid EEF actions for episode {metadata['episode_index']}: {raw_actions.shape}")

    row = np.arange(len(observation_steps), dtype=np.int64)
    source = row[:, None] + 1 + np.arange(ACTION_HORIZON, dtype=np.int64)[None, :]
    valid = source < len(raw_actions)
    source = np.minimum(source, len(raw_actions) - 1)
    action_chunks = raw_actions[source]
    action_mask = np.all(valid, axis=1)
    if target_mode == "phase_waypoint_delta":
        first_descend = np.flatnonzero(phase_id == 1)
        first_close = np.flatnonzero(phase_id == 2)
        descend_index = int(first_descend[0]) if len(first_descend) else len(phase_id) - 1
        contact_index = int(first_close[0]) if len(first_close) else len(phase_id) - 1
        endpoints = np.stack(
            (
                observation_poses[descend_index],
                observation_poses[contact_index],
                observation_poses[contact_index],
                observation_poses[-1],
            )
        )
        residual = endpoints[phase_id] - observation_poses
        residual[:, 3:] = (residual[:, 3:] + np.pi) % (2.0 * np.pi) - np.pi
        gripper = np.where(phase_id[:, None] < 2, 1.0, -1.0).astype(np.float32)
        command = np.concatenate((residual, gripper), axis=1)
        action_chunks = np.repeat(command[:, None, :], ACTION_HORIZON, axis=1)
        action_mask = np.ones(len(action_chunks), dtype=np.bool_)

    if target_conditioning == "image_yx":
        target = np.asarray(metadata["target_point_normalized_yx"], dtype=np.float32)
    elif target_conditioning == "world_xy":
        if target_mode != "phase_waypoint_delta":
            raise ValueError("world_xy conditioning requires phase_waypoint_delta targets")
        # The first close waypoint is a demonstration-derived, non-simulator-
        # privileged proxy for the stationary target object's world XY.
        target = observation_poses[contact_index, :2].astype(np.float32)
    else:
        raise ValueError(f"Unknown target conditioning: {target_conditioning}")
    states = []
    front = []
    wrist = []
    for step in observation_steps:
        obs = episode[f"timestep_{int(step)}/obs"]
        eef = np.asarray(obs["eef_state"][()], dtype=np.float32).reshape(6)
        gripper_width = np.asarray(obs["gripper_state"][()], dtype=np.float32).sum(keepdims=True)
        state = np.concatenate((eef, gripper_width, target), axis=0)
        if phase_conditioned:
            state = np.concatenate(
                (state, np.eye(len(_action_data.PHASES), dtype=np.float32)[phase_id[len(states)]])
            )
        states.append(state)
        front.append(np.asarray(obs["front_rgb"][()], dtype=np.uint8))
        wrist.append(np.asarray(obs["wrist_rgb"][()], dtype=np.uint8))
    states = np.stack(states)
    front = np.stack(front)
    wrist = np.stack(wrist)
    state_dim = BASE_STATE_DIM + (len(_action_data.PHASES) if phase_conditioned else 0)
    if states.shape != (len(observation_steps), state_dim) or not np.all(np.isfinite(states)):
        raise ValueError(f"Invalid state array for episode {metadata['episode_index']}: {states.shape}")
    if recovery_augmentations > 0:
        if target_mode != "phase_waypoint_delta":
            raise ValueError("Recovery augmentation requires phase_waypoint_delta targets")
        rng = np.random.default_rng(260827 + int(metadata["episode_index"]))
        state_rows = [states]
        action_rows = [action_chunks]
        front_rows = [front]
        wrist_rows = [wrist]
        phase_rows = [phase_id]
        mask_rows = [action_mask]
        for _ in range(recovery_augmentations):
            noisy = states.copy()
            pose_noise = np.zeros((len(states), 6), dtype=np.float32)
            pose_noise[:, :2] = rng.uniform(-0.04, 0.04, size=(len(states), 2))
            pose_noise[:, 2] = rng.uniform(-0.06, 0.06, size=len(states))
            pose_noise[:, 3:5] = rng.uniform(-0.03, 0.03, size=(len(states), 2))
            pose_noise[:, 5] = rng.uniform(-0.20, 0.20, size=len(states))
            noisy[:, :6] += pose_noise
            noisy[:, :3] = np.clip(noisy[:, :3], [-0.42, -0.42, 0.05], [0.42, 0.42, 0.62])
            actual_noise = noisy[:, :6] - states[:, :6]
            corrected = action_chunks.copy()
            corrected[:, :, :6] -= actual_noise[:, None, :]
            corrected[:, :, 3:6] = (
                corrected[:, :, 3:6] + np.pi
            ) % (2.0 * np.pi) - np.pi
            state_rows.append(noisy)
            action_rows.append(corrected)
            front_rows.append(front)
            wrist_rows.append(wrist)
            phase_rows.append(phase_id)
            mask_rows.append(action_mask)
        states = np.concatenate(state_rows)
        action_chunks = np.concatenate(action_rows)
        front = np.concatenate(front_rows)
        wrist = np.concatenate(wrist_rows)
        phase_id = np.concatenate(phase_rows)
        action_mask = np.concatenate(mask_rows)
    return {
        "state": states,
        "actions": action_chunks,
        "front_rgb": front,
        "wrist_rgb": wrist,
        "action_mask": action_mask,
        "phase_id": phase_id,
    }


def _write_episode(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    episode_index: int,
    global_start: int,
    fps: int,
    image_size: int,
    png_compress_level: int,
    features: dict,
    state_dim: int,
) -> None:
    length = len(arrays["state"])
    columns = [
        _lerobot.nested_float_array(arrays["state"][:, None, :]),
        _lerobot.nested_float_array(arrays["actions"]),
        _lerobot.encoded_image_column(arrays["front_rgb"], image_size, png_compress_level),
        _lerobot.encoded_image_column(arrays["wrist_rgb"], image_size, png_compress_level),
        pa.array(np.arange(length, dtype=np.float32) / fps, type=pa.float32()),
        pa.array(np.arange(length, dtype=np.int64), type=pa.int64()),
        pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
        pa.array(np.arange(global_start, global_start + length, dtype=np.int64), type=pa.int64()),
        pa.array(np.zeros(length, dtype=np.int64), type=pa.int64()),
        pa.array(arrays["action_mask"], type=pa.bool_()),
        pa.array(arrays["phase_id"], type=pa.int64()),
        pa.array(np.full(length, length, dtype=np.int64), type=pa.int64()),
    ]
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema(
        [
            _lerobot.extension_field("observation.state", (1, state_dim)),
            _lerobot.extension_field("actions", (ACTION_HORIZON, ACTION_DIM)),
            pa.field("observation.front_rgb", image_type),
            pa.field("observation.wrist_rgb", image_type),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("action_mask", pa.bool_()),
            pa.field("phase_id", pa.int64()),
            pa.field("episode_T", pa.int64()),
        ],
        metadata={b"huggingface": _lerobot.huggingface_metadata(features)},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(columns, schema=schema), path)


def main() -> None:
    args = parse_args()
    h5_path = args.h5.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    if min(args.image_size, args.fps) <= 0:
        raise ValueError("image-size and fps must be positive")

    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    metadata = {int(item["episode_index"]): item for item in payload["episodes"]}
    source_indices = _episode_indices(payload, args.split)
    state_dim = BASE_STATE_DIM + (len(_action_data.PHASES) if args.phase_conditioned else 0)
    features = _feature_info(args.image_size, state_dim)
    results = []
    norm: dict[str, list[np.ndarray]] = {}
    global_start = 0
    with h5py.File(h5_path, "r") as source:
        for converted_index, source_index in enumerate(source_indices):
            item = metadata[source_index]
            episode = source[item["episode_name"]]
            arrays = _episode_arrays(
                episode,
                item,
                target_mode=args.target_mode,
                target_conditioning=args.target_conditioning,
                phase_conditioned=args.phase_conditioned,
                recovery_augmentations=args.recovery_augmentations,
            )
            parquet = output / "data/chunk-000" / f"episode_{converted_index:06d}.parquet"
            _write_episode(
                parquet,
                arrays,
                episode_index=converted_index,
                global_start=global_start,
                fps=args.fps,
                image_size=args.image_size,
                png_compress_level=args.png_compress_level,
                features=features,
                state_dim=state_dim,
            )
            length = len(arrays["state"])
            stats = {
                "observation.state": _lerobot.stats_for_array(arrays["state"][:, None, :]),
                "actions": _lerobot.stats_for_array(arrays["actions"]),
            }
            results.append(
                {
                    "episode_index": converted_index,
                    "source_episode_index": source_index,
                    "tasks": [PROMPT],
                    "length": length,
                    "target_point_yx": item["target_point_yx"],
                    "full_horizon_rows": int(arrays["action_mask"].sum()),
                    "phase_counts": {
                        str(phase): int(np.count_nonzero(arrays["phase_id"] == phase))
                        for phase in range(5)
                    },
                    "stats": stats,
                }
            )
            _lerobot.update_norm(norm, "state", arrays["state"])
            _lerobot.update_norm(norm, "actions", arrays["actions"])
            global_start += length
            print(
                f"[{converted_index + 1}/{len(source_indices)}] source={source_index} "
                f"rows={length} full_horizon={int(arrays['action_mask'].sum())}",
                flush=True,
            )

    meta = output / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": len(results),
        "total_frames": global_start,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": max(len(results), 1),
        "fps": args.fps,
        "splits": {"train": f"0:{len(results)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "robomme_videounmask": {
            "source_h5": str(h5_path),
            "source_labels": str(labels_path),
            "source_split": args.split,
            "single_target_only": True,
            "state_layout": [
                "eef_x", "eef_y", "eef_z", "eef_roll", "eef_pitch", "eef_yaw",
                "gripper_width",
                *(
                    ["target_y_normalized", "target_x_normalized"]
                    if args.target_conditioning == "image_yx"
                    else ["target_world_x", "target_world_y"]
                ),
            ],
            "action_layout": [
                "world_x", "world_y", "world_z", "roll", "pitch", "yaw", "gripper_command"
            ],
            "action_alignment": "post_step_observation[t] -> eef_action[t+1:t+17]",
            "terminal_padding": "repeat_final_absolute_eef7_command",
            "action_target_mode": args.target_mode,
            "target_conditioning": args.target_conditioning,
            "phase_conditioned": args.phase_conditioned,
            "recovery_augmentations": args.recovery_augmentations,
            "action_mask": "true only when all 16 future commands are real",
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _lerobot.write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": PROMPT}])
    _lerobot.write_jsonl(
        meta / "episodes.jsonl",
        [{key: value for key, value in result.items() if key != "stats"} for result in results],
    )
    episode_stats = [
        {
            "episode_index": result["episode_index"],
            "stats": result["stats"],
        }
        for result in results
    ]
    _lerobot.write_jsonl(meta / "episodes_stats.jsonl", episode_stats)
    (output / "norm_stats.json").write_text(
        json.dumps(_lerobot.finalize_norm(norm), indent=2), encoding="utf-8"
    )
    summary = {
        "ok": True,
        "output": str(output),
        "source_split": args.split,
        "episodes": len(results),
        "rows": global_start,
        "full_horizon_rows": sum(item["full_horizon_rows"] for item in results),
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "state_dim": state_dim,
        "action_target_mode": args.target_mode,
        "target_conditioning": args.target_conditioning,
        "phase_conditioned": args.phase_conditioned,
        "recovery_augmentations": args.recovery_augmentations,
        "phase_counts": {
            str(phase): sum(item["phase_counts"][str(phase)] for item in results)
            for phase in range(5)
        },
    }
    (output / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
