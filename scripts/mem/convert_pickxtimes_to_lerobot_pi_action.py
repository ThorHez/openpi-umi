#!/usr/bin/env python3
"""Convert the frozen PickXtimes train/dev split to Pi0.5 EEF7 chunks."""

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

ROBOSUITE_SCRIPTS = Path(__file__).resolve().parents[3] / "robosuite/robosuite/scripts"
sys.path.insert(0, str(ROBOSUITE_SCRIPTS))
import convert_shellgame_to_openpi_umi_v2_openpi_action as _lerobot  # noqa: E402

ACTION_HORIZON = 16
ACTION_DIM = 7
STATE_DIM = 11
DEFAULT_H5 = Path("data/robomme_extracted/record_dataset_PickXtimes.h5")
DEFAULT_LABELS = Path("data/robomme_extracted/pickxtimes_event_labels_w10_v3_press5_gripper.json")
DEFAULT_SPLIT = Path("data/robomme_extracted/pickxtimes_split_seed260827_train70_dev15_test15.json")
DEFAULT_OUTPUT = Path("data/robomme_pickxtimes_lerobot_pi_action_train70_stride2")
COLOR_TO_ID = {"red": 0, "green": 1, "blue": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--row-stride", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--png-compress-level", type=int, choices=range(10), default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _feature_info(image_size: int) -> dict:
    image_shape = [image_size, image_size, 3]
    return {
        "observation.state": {
            "dtype": "float32", "shape": [1, STATE_DIM], "names": ["state"], "_type": "Array2D"
        },
        "actions": {
            "dtype": "float32", "shape": [ACTION_HORIZON, ACTION_DIM],
            "names": ["actions"], "_type": "Array2D",
        },
        "observation.front_rgb": {
            "dtype": "image", "shape": image_shape,
            "names": ["height", "width", "channel"], "_type": "Image",
        },
        "observation.wrist_rgb": {
            "dtype": "image", "shape": image_shape,
            "names": ["height", "width", "channel"], "_type": "Image",
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


def _phase_id(item: dict, timesteps: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Five manipulation phases plus a dedicated button phase."""
    z = actions[:, 0, 2]
    closed = actions[:, 0, 6] < 0.0
    phase = np.zeros(len(actions), dtype=np.int64)
    phase[(~closed) & (z <= 0.22)] = 1
    phase[(~closed) & (z <= 0.10)] = 2
    phase[closed & (z <= 0.10)] = 3
    phase[closed & (z > 0.10)] = 4
    press_start = int(item["events"][-1]["start"])
    phase[timesteps >= press_start] = 5
    return phase


def _episode_arrays(episode, item: dict, row_stride: int) -> dict[str, np.ndarray]:
    episode_t = int(item["num_steps"])
    timesteps = np.arange(0, episode_t - 1, row_stride, dtype=np.int64)
    source = timesteps[:, None] + 1 + np.arange(ACTION_HORIZON, dtype=np.int64)[None]
    valid = source < episode_t
    source = np.minimum(source, episode_t - 1)
    actions = np.stack(
        [
            np.stack(
                [np.asarray(episode[f"timestep_{int(step)}/action/eef_action"][()], dtype=np.float32)
                 for step in row]
            )
            for row in source
        ]
    )
    color = np.eye(3, dtype=np.float32)[COLOR_TO_ID[item["target_color"]]]
    count = np.asarray([int(item["required_count"]) / 5.0], dtype=np.float32)
    states, front, wrist = [], [], []
    for step in timesteps:
        obs = episode[f"timestep_{int(step)}/obs"]
        eef = np.asarray(obs["eef_state"][()], dtype=np.float32).reshape(6)
        width = np.asarray(obs["gripper_state"][()], dtype=np.float32).sum(keepdims=True)
        states.append(np.concatenate((eef, width, color, count)))
        front.append(np.asarray(obs["front_rgb"][()], dtype=np.uint8))
        wrist.append(np.asarray(obs["wrist_rgb"][()], dtype=np.uint8))
    result = {
        "timesteps": timesteps,
        "state": np.stack(states),
        "actions": actions,
        "front_rgb": np.stack(front),
        "wrist_rgb": np.stack(wrist),
        "action_mask": np.all(valid, axis=1),
        "episode_T": episode_t,
    }
    result["phase_id"] = _phase_id(item, timesteps, actions)
    return result


def _write_episode(path: Path, arrays: dict, *, episode_index: int, global_start: int,
                   task_index: int, fps: int, image_size: int, png_level: int, features: dict) -> None:
    length = len(arrays["state"])
    columns = [
        _lerobot.nested_float_array(arrays["state"][:, None, :]),
        _lerobot.nested_float_array(arrays["actions"]),
        _lerobot.encoded_image_column(arrays["front_rgb"], image_size, png_level),
        _lerobot.encoded_image_column(arrays["wrist_rgb"], image_size, png_level),
        # LeRobot requires consecutive timestamps at the declared dataset FPS.
        # ``frame_index`` below intentionally remains the original simulator
        # timestep because it drives causal MEM lookup and suffix masking.
        pa.array(np.arange(length, dtype=np.float32) / fps, type=pa.float32()),
        pa.array(arrays["timesteps"], type=pa.int64()),
        pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
        pa.array(np.arange(global_start, global_start + length, dtype=np.int64), type=pa.int64()),
        pa.array(np.full(length, task_index, dtype=np.int64), type=pa.int64()),
        pa.array(arrays["action_mask"], type=pa.bool_()),
        pa.array(arrays["phase_id"], type=pa.int64()),
        pa.array(np.full(length, arrays["episode_T"], dtype=np.int64), type=pa.int64()),
    ]
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema(
        [
            _lerobot.extension_field("observation.state", (1, STATE_DIM)),
            _lerobot.extension_field("actions", (ACTION_HORIZON, ACTION_DIM)),
            pa.field("observation.front_rgb", image_type),
            pa.field("observation.wrist_rgb", image_type),
            pa.field("timestamp", pa.float32()), pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()), pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()), pa.field("action_mask", pa.bool_()),
            pa.field("phase_id", pa.int64()), pa.field("episode_T", pa.int64()),
        ],
        metadata={b"huggingface": _lerobot.huggingface_metadata(features)},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(columns, schema=schema), path)


def main() -> None:
    args = parse_args()
    if min(args.row_stride, args.image_size, args.fps) <= 0:
        raise ValueError("row-stride, image-size and fps must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    metadata = {int(item["episode_index"]): item for item in labels["episodes"]}
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    split_key = "train_episode_indices" if args.split == "train" else "val_episode_indices"
    source_indices = [int(value) for value in split[split_key]]
    if args.max_episodes is not None:
        source_indices = source_indices[: args.max_episodes]
    if set(source_indices) & {int(value) for value in split.get("test_episode_indices", [])}:
        raise ValueError("Frozen-test leakage")
    prompts = sorted({metadata[index]["prompts"][index % len(metadata[index]["prompts"])] for index in source_indices})
    prompt_to_index = {prompt: index for index, prompt in enumerate(prompts)}
    features = _feature_info(args.image_size)
    results, norm = [], {}
    global_start = 0
    with h5py.File(args.h5.expanduser().resolve(), "r") as source:
        for converted_index, source_index in enumerate(source_indices):
            item = metadata[source_index]
            arrays = _episode_arrays(source[item["episode_name"]], item, args.row_stride)
            prompt = item["prompts"][source_index % len(item["prompts"])]
            _write_episode(
                output / "data/chunk-000" / f"episode_{converted_index:06d}.parquet",
                arrays,
                episode_index=converted_index,
                global_start=global_start,
                task_index=prompt_to_index[prompt],
                fps=args.fps,
                image_size=args.image_size,
                png_level=args.png_compress_level,
                features=features,
            )
            stats = {
                "observation.state": _lerobot.stats_for_array(arrays["state"][:, None, :]),
                "actions": _lerobot.stats_for_array(arrays["actions"]),
            }
            phase_counts = {str(i): int(np.sum(arrays["phase_id"] == i)) for i in range(6)}
            results.append({
                "episode_index": converted_index, "source_episode_index": source_index,
                "source_episode_name": item["episode_name"], "tasks": [prompt],
                "length": len(arrays["state"]), "full_horizon_rows": int(arrays["action_mask"].sum()),
                "phase_counts": phase_counts, "stats": stats,
            })
            _lerobot.update_norm(norm, "state", arrays["state"])
            _lerobot.update_norm(norm, "actions", arrays["actions"])
            global_start += len(arrays["state"])
            print(f"[{converted_index + 1}/{len(source_indices)}] source={source_index} rows={len(arrays['state'])}", flush=True)

    meta = output / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1", "robot_type": "panda", "total_episodes": len(results),
        "total_frames": global_start, "total_tasks": len(prompts), "total_videos": 0,
        "total_chunks": 1, "chunks_size": max(len(results), 1), "fps": args.fps,
        "splits": {"train": f"0:{len(results)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "robomme_pickxtimes": {
            "source_h5": str(args.h5.resolve()), "source_labels": str(args.labels.resolve()),
            "source_split_file": str(args.split_file.resolve()), "source_split": args.split,
            "row_stride": args.row_stride, "frozen_test_accessed": False,
            "state_layout": ["eef_x", "eef_y", "eef_z", "roll", "pitch", "yaw", "gripper_width",
                             "goal_red", "goal_green", "goal_blue", "required_count_div_5"],
            "memory": "loaded causally from external frozen Round-9 token bank",
            "action_alignment": "observation[t] -> eef_action[t+1:t+17]",
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _lerobot.write_jsonl(meta / "tasks.jsonl", [
        {"task_index": index, "task": prompt} for index, prompt in enumerate(prompts)
    ])
    _lerobot.write_jsonl(meta / "episodes.jsonl", [
        {key: value for key, value in result.items() if key != "stats"} for result in results
    ])
    _lerobot.write_jsonl(meta / "episodes_stats.jsonl", [
        {"episode_index": result["episode_index"], "stats": result["stats"]} for result in results
    ])
    (output / "norm_stats.json").write_text(json.dumps(_lerobot.finalize_norm(norm), indent=2), encoding="utf-8")
    summary = {
        "ok": True, "output": str(output), "source_split": args.split,
        "episodes": len(results), "rows": global_start, "row_stride": args.row_stride,
        "full_horizon_rows": sum(item["full_horizon_rows"] for item in results),
        "phase_counts": {str(i): sum(item["phase_counts"][str(i)] for item in results) for i in range(6)},
    }
    (output / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
