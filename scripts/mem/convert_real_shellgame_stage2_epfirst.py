#!/usr/bin/env python3
"""Convert the 306-episode real ShellGame replay to the deployment contract.

The raw replay stores measured world-frame EEF poses and 7-D commanded EEF
targets. For a dataset row at frame ``t`` this converter emits:

* state: ``inv(T_episode0) @ T_measured[t]`` as xyz + rot6d + gripper;
* action[h]: ``inv(T_measured[t]) @ T_command[t+h+1]`` for the requested
  action horizon (16 by default);
* one wrist RGB image;
* episode length and semantic labels for masking/evaluation.

Every future waypoint in a chunk has the same *current-frame* anchor. This is
the exact output contract decoded by ``eval_arx5_pi_hzl.py`` via
``relative_chunk_to_world(actions, anchor_link6)``. Only the policy input state
is episode-first-relative.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any
import zipfile

from numcodecs import blosc
import numpy as np

from openpi.utils.pose_utils import mat_to_pose10d
from openpi.utils.pose_utils import pose_to_mat

DEFAULT_INPUT = Path("/data2/hzl_workspace_for_pi_mem/replay_buffer_merged_306_degap.zarr.zip")
DEFAULT_LABELS = Path("/data2/hzl_workspace_for_pi_mem/labels_merged_306_degap.jsonl")
DEFAULT_OUTPUT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10"
)
# Must match DEFAULT_PROMPT in umi-arx-kian/scripts/eval_arx5_pi_hzl.py.
PROMPT = "The shell game has ended. Grasp and lift the cup containing the ball."
HISTORY_FRAMES = 241
ACTION_HORIZON = 16
ACTION_DIM = 10


class _Store:
    def read(self, key: str) -> bytes:
        raise NotImplementedError

    def keys(self) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _DirectoryStore(_Store):
    def __init__(self, root: Path):
        self.root = root

    def read(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def keys(self) -> list[str]:
        return [str(path.relative_to(self.root)) for path in self.root.rglob("*") if path.is_file()]


class _ZipStore(_Store):
    def __init__(self, path: Path):
        self.archive = zipfile.ZipFile(path, "r")
        names = self.archive.namelist()
        suffix = "/data/action/zarr.json"
        matches = [name[: -len(suffix)] for name in names if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one replay root in {path}, found {len(matches)}: {matches[:3]}")
        self.prefix = f"{matches[0]}/"
        self._keys = [name[len(self.prefix) :] for name in names if name.startswith(self.prefix)]

    def read(self, key: str) -> bytes:
        return self.archive.read(f"{self.prefix}{key}")

    def keys(self) -> list[str]:
        return self._keys

    def close(self) -> None:
        self.archive.close()


def _open_store(path: Path) -> _Store:
    if path.is_dir():
        root = path
        if not (root / "data/action/zarr.json").is_file():
            candidates = list(root.glob("*/data/action/zarr.json"))
            if len(candidates) != 1:
                raise ValueError(f"Cannot locate replay Zarr root under {path}")
            root = candidates[0].parents[2]
        return _DirectoryStore(root)
    if path.suffix == ".zip":
        return _ZipStore(path)
    raise ValueError(f"Input must be a Zarr directory or .zip archive: {path}")


@dataclass
class ZarrV3Array:
    store: _Store
    key: str

    def __post_init__(self) -> None:
        metadata = json.loads(self.store.read(f"{self.key}/zarr.json"))
        if metadata.get("zarr_format") != 3:
            raise ValueError(f"{self.key} is not a Zarr v3 array")
        self.shape = tuple(int(value) for value in metadata["shape"])
        self.chunk_shape = tuple(int(value) for value in metadata["chunk_grid"]["configuration"]["chunk_shape"])
        self.dtype = np.dtype(metadata["data_type"]).newbyteorder("<")
        self.codecs = tuple(codec["name"] for codec in metadata.get("codecs", ()))
        unsupported = set(self.codecs) - {"bytes", "blosc"}
        if unsupported:
            raise ValueError(f"Unsupported codecs for {self.key}: {sorted(unsupported)}")
        self._cached_chunk_index: int | None = None
        self._cached_chunk: np.ndarray | None = None

    def _chunk_key(self, first_axis_chunk: int) -> str:
        coordinates = (first_axis_chunk,) + (0,) * (len(self.shape) - 1)
        return f"{self.key}/c/" + "/".join(str(value) for value in coordinates)

    def _read_chunk(self, first_axis_chunk: int) -> np.ndarray:
        if self._cached_chunk_index == first_axis_chunk:
            assert self._cached_chunk is not None
            return self._cached_chunk
        payload = self.store.read(self._chunk_key(first_axis_chunk))
        if "blosc" in self.codecs:
            payload = blosc.decompress(payload)
        chunk = np.frombuffer(payload, dtype=self.dtype).reshape(self.chunk_shape)
        self._cached_chunk_index = first_axis_chunk
        self._cached_chunk = chunk
        return chunk

    def load(self) -> np.ndarray:
        output = np.empty(self.shape, dtype=self.dtype)
        num_chunks = (self.shape[0] + self.chunk_shape[0] - 1) // self.chunk_shape[0]
        for chunk_index in range(num_chunks):
            chunk = self._read_chunk(chunk_index)
            start = chunk_index * self.chunk_shape[0]
            rows = min(self.chunk_shape[0], self.shape[0] - start)
            output[start : start + rows] = chunk[:rows]
        return output

    def row(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.shape[0]:
            raise IndexError(index)
        chunk_index, offset = divmod(index, self.chunk_shape[0])
        return self._read_chunk(chunk_index)[offset]


def _load_labels(path: Path) -> list[dict[str, Any]]:
    labels = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [int(row["episode_id"]) for row in labels] != list(range(len(labels))):
        raise ValueError("labels must be ordered by contiguous episode_id")
    for row in labels:
        moves = row["moves"]
        slot = int(row["initial_cup"])
        if len(moves) != 3 or int(row["n_observe_frames"]) != HISTORY_FRAMES:
            raise ValueError(f"Episode {row['episode_id']} has an invalid event timeline")
        for left, right in moves:
            if slot == int(left):
                slot = int(right)
            elif slot == int(right):
                slot = int(left)
        if slot != int(row["final_cup"]):
            raise ValueError(f"Episode {row['episode_id']} labels do not roll out to final_cup")
    return labels


def _relative_pose10(world_poses: np.ndarray, episode_first: np.ndarray) -> np.ndarray:
    inverse_first = np.linalg.inv(episode_first)
    relative = inverse_first[None] @ world_poses
    return np.asarray(mat_to_pose10d(relative), dtype=np.float32)


@dataclass
class EpisodeContract:
    state: np.ndarray
    actions: np.ndarray
    command_valid: np.ndarray
    max_roundtrip_position_error_m: float
    max_roundtrip_rotation_matrix_error: float


def build_episode_contract(
    eef_pos: np.ndarray,
    eef_rotvec: np.ndarray,
    gripper: np.ndarray,
    raw_action: np.ndarray,
    *,
    action_horizon: int = ACTION_HORIZON,
    max_command_position_error_m: float = 0.05,
) -> EpisodeContract:
    """Construct ep-first state and current-anchored future action chunks."""
    length = len(eef_pos)
    if not (
        eef_pos.shape == (length, 3)
        and eef_rotvec.shape == (length, 3)
        and gripper.shape == (length,)
        and raw_action.shape == (length, 7)
    ):
        raise ValueError("Unexpected real ShellGame low-dimensional array shapes")

    measured_pose6 = np.concatenate((eef_pos, eef_rotvec), axis=-1)
    measured_world = np.asarray(pose_to_mat(measured_pose6), dtype=np.float64)
    command_world = np.asarray(pose_to_mat(raw_action[:, :6]), dtype=np.float64)
    command_position_error = np.linalg.norm(raw_action[:, :3] - eef_pos, axis=-1)
    command_valid = (
        np.all(np.isfinite(raw_action), axis=-1)
        & (np.linalg.norm(raw_action[:, :6], axis=-1) > 1e-9)
        & (command_position_error <= max_command_position_error_m)
    )
    sanitized_world = np.where(command_valid[:, None, None], command_world, measured_world)
    sanitized_gripper = np.where(command_valid, raw_action[:, 6], gripper)

    episode_first = measured_world[0]
    state_pose10 = _relative_pose10(measured_world, episode_first)
    state = np.concatenate((state_pose10, gripper[:, None]), axis=-1).astype(np.float32)

    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    future_indices = np.minimum(
        np.arange(length)[:, None] + 1 + np.arange(action_horizon)[None, :],
        length - 1,
    )
    future_world = sanitized_world[future_indices]
    current_inverse = np.linalg.inv(measured_world)
    relative_world = current_inverse[:, None] @ future_world
    target_pose10 = np.asarray(mat_to_pose10d(relative_world), dtype=np.float32)
    actions = np.concatenate(
        (target_pose10, sanitized_gripper[future_indices, None]),
        axis=-1,
    ).astype(np.float32)

    # Verify the same multiplication used in deployment:
    # T_world_target = T_world_current @ T_model_output.
    reconstructed = measured_world[:, None] @ relative_world
    position_error = np.linalg.norm(
        reconstructed[..., :3, 3] - future_world[..., :3, 3],
        axis=-1,
    )
    rotation_error = np.max(
        np.abs(reconstructed[..., :3, :3] - future_world[..., :3, :3]),
        axis=(-2, -1),
    )
    return EpisodeContract(
        state=state,
        actions=actions,
        command_valid=command_valid,
        max_roundtrip_position_error_m=float(np.max(position_error)),
        max_roundtrip_rotation_matrix_error=float(np.max(rotation_error)),
    )


def _norm_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": np.maximum(values.std(axis=0), 1e-8).tolist(),
        "max": values.max(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _free_bytes(path: Path) -> int:
    probe = path if path.exists() else path.parent
    while not probe.exists():
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def _create_lerobot_dataset(output: Path, *, workers: int, action_horizon: int, repo_id: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.robot0_eef_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        },
        "observation.robot0_eef_rot_axis_angle": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["r1x", "r1y", "r1z", "r2x", "r2y", "r2z"],
        },
        "observation.robot0_gripper_width": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_width"],
        },
        "observation.left_wrist_0_rgb_0": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_horizon, ACTION_DIM),
            "names": ["horizon", "eef10"],
        },
        "episode_length": {"dtype": "int64", "shape": (1,), "names": ["frames"]},
        "initial_cup": {"dtype": "int64", "shape": (1,), "names": ["slot"]},
        "final_cup": {"dtype": "int64", "shape": (1,), "names": ["slot"]},
        "swap_pairs": {"dtype": "int64", "shape": (3, 2), "names": ["stage", "pair"]},
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=20,
        root=output,
        robot_type="umi",
        features=features,
        use_videos=True,
        image_writer_processes=1,
        image_writer_threads=max(1, workers),
    )


def convert(args: argparse.Namespace) -> dict[str, Any]:
    labels = _load_labels(args.labels)
    store = _open_store(args.input)
    try:
        episode_ends = ZarrV3Array(store, "meta/episode_ends").load().astype(np.int64)
        starts = np.concatenate((np.array([0], dtype=np.int64), episode_ends[:-1]))
        if len(episode_ends) != len(labels):
            raise ValueError(f"Label/replay episode mismatch: {len(labels)} vs {len(episode_ends)}")
        total_frames = int(episode_ends[-1])
        arrays = {
            key: ZarrV3Array(store, f"data/{key}").load()
            for key in (
                "action",
                "robot0_eef_pos",
                "robot0_eef_rot_axis_angle",
                "robot0_gripper_width",
            )
        }
        if any(value.shape[0] != total_frames for value in arrays.values()):
            raise ValueError("Low-dimensional replay arrays have inconsistent lengths")

        num_episodes = len(labels)
        if args.max_episodes is not None:
            num_episodes = min(num_episodes, args.max_episodes)
        contracts: list[EpisodeContract] = []
        state_for_stats: list[np.ndarray] = []
        actions_for_stats: list[np.ndarray] = []
        stale_by_episode: list[int] = []
        length_mismatches: list[dict[str, int]] = []
        for episode in range(num_episodes):
            start, end = int(starts[episode]), int(episode_ends[episode])
            expected_length = int(labels[episode]["n_frames"])
            actual_length = end - start
            if actual_length <= HISTORY_FRAMES:
                raise ValueError(f"Episode {episode}: replay length {actual_length} has no action suffix")
            if actual_length != expected_length:
                # The merged degap replay trims a few terminal frames from
                # episodes 251+, while their event labels still describe the
                # same fixed 0..240 observation prefix. Use authoritative Zarr
                # boundaries for action masking and retain the discrepancy in
                # the audit instead of shifting any event label.
                length_mismatches.append(
                    {
                        "episode_id": episode,
                        "label_n_frames": expected_length,
                        "replay_n_frames": actual_length,
                        "trimmed_terminal_frames": expected_length - actual_length,
                    }
                )
            contract = build_episode_contract(
                arrays["robot0_eef_pos"][start:end],
                arrays["robot0_eef_rot_axis_angle"][start:end],
                np.asarray(arrays["robot0_gripper_width"][start:end]).reshape(-1),
                arrays["action"][start:end],
                action_horizon=args.action_horizon,
                max_command_position_error_m=args.max_command_position_error_m,
            )
            contracts.append(contract)
            eligible = slice(HISTORY_FRAMES, end - start)
            state_for_stats.append(contract.state[eligible])
            actions_for_stats.append(contract.actions[eligible].reshape(-1, ACTION_DIM))
            stale_by_episode.append(int(np.sum(~contract.command_valid[HISTORY_FRAMES:])))

        state_values = np.concatenate(state_for_stats, axis=0)
        action_values = np.concatenate(actions_for_stats, axis=0)
        shuffled = np.random.default_rng(42).permutation(num_episodes)
        num_val = min(max(1, round(num_episodes * 0.1)), num_episodes - 1)
        audit = {
            "schema_version": 1,
            "input": str(args.input.resolve()),
            "labels": str(args.labels.resolve()),
            "episodes": num_episodes,
            "frames": int(sum(int(episode_ends[i] - starts[i]) for i in range(num_episodes))),
            "episode_length": {
                "min": int(min(int(episode_ends[i] - starts[i]) for i in range(num_episodes))),
                "median": float(np.median([int(episode_ends[i] - starts[i]) for i in range(num_episodes)])),
                "max": int(max(int(episode_ends[i] - starts[i]) for i in range(num_episodes))),
            },
            "history_frames": HISTORY_FRAMES,
            "label_length_mismatches": {
                "count": len(length_mismatches),
                "max_abs_difference": max(
                    (abs(row["trimmed_terminal_frames"]) for row in length_mismatches),
                    default=0,
                ),
                "episodes": length_mismatches,
            },
            "action_horizon": args.action_horizon,
            "state_contract": "episode_first_relative_link6_pose10_plus_direct_gripper",
            "action_contract": "current_frame_same_anchor_relative_link6_pose10_plus_direct_gripper",
            "raw_command_fallback": {
                "max_position_error_m": args.max_command_position_error_m,
                "count_after_history": int(sum(stale_by_episode)),
                "fraction_after_history": float(
                    sum(stale_by_episode) / sum(len(contract.state) - HISTORY_FRAMES for contract in contracts)
                ),
                "affected_episodes": int(sum(count > 0 for count in stale_by_episode)),
            },
            "roundtrip": {
                "max_position_error_m": max(contract.max_roundtrip_position_error_m for contract in contracts),
                "max_rotation_matrix_error": max(
                    contract.max_roundtrip_rotation_matrix_error for contract in contracts
                ),
            },
            "class_counts": {
                "initial_cup": np.bincount(
                    [int(labels[i]["initial_cup"]) for i in range(num_episodes)], minlength=3
                ).tolist(),
                "final_cup": np.bincount(
                    [int(labels[i]["final_cup"]) for i in range(num_episodes)], minlength=3
                ).tolist(),
            },
            "episode_split_seed": 42,
            "validation_episode_ids": sorted(int(value) for value in shuffled[:num_val]),
            "norm_stats": {
                "state": _norm_stats(state_values),
                "actions": _norm_stats(action_values),
            },
        }
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in audit.items() if key != "norm_stats"}, indent=2))
        if args.audit_only:
            return audit

        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing dataset: {args.output}")
        required_bytes = int(args.min_free_gib * 1024**3)
        available = _free_bytes(args.output)
        if available < required_bytes:
            raise OSError(
                f"Need at least {args.min_free_gib:.1f} GiB free before conversion; "
                f"only {available / 1024**3:.2f} GiB is available"
            )

        camera = ZarrV3Array(store, "data/camera0_rgb")
        if camera.shape != (total_frames, 224, 224, 3):
            raise ValueError(f"Unexpected camera shape: {camera.shape}")
        dataset = _create_lerobot_dataset(
            args.output,
            workers=args.image_workers,
            action_horizon=args.action_horizon,
            repo_id=args.repo_id,
        )
        for episode, contract in enumerate(contracts):
            start, end = int(starts[episode]), int(episode_ends[episode])
            label = labels[episode]
            length = end - start
            for local_frame, global_frame in enumerate(range(start, end)):
                image = np.asarray(camera.row(global_frame))
                if np.issubdtype(image.dtype, np.floating):
                    scale = 255.0 if float(np.nanmax(image)) <= 1.5 else 1.0
                    image = np.clip(np.rint(image * scale), 0, 255).astype(np.uint8)
                dataset.add_frame(
                    {
                        "observation.robot0_eef_pos": contract.state[local_frame, :3],
                        "observation.robot0_eef_rot_axis_angle": contract.state[local_frame, 3:9],
                        "observation.robot0_gripper_width": contract.state[local_frame, 9:10],
                        "observation.left_wrist_0_rgb_0": image,
                        "actions": contract.actions[local_frame],
                        "episode_length": np.asarray([length], dtype=np.int64),
                        "initial_cup": np.asarray([label["initial_cup"]], dtype=np.int64),
                        "final_cup": np.asarray([label["final_cup"]], dtype=np.int64),
                        "swap_pairs": np.asarray(label["moves"], dtype=np.int64),
                        "task": PROMPT,
                    }
                )
            dataset.save_episode()

        norm_path = args.output / "norm_stats.json"
        norm_path.write_text(
            json.dumps({"norm_stats": audit["norm_stats"]}, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output / "conversion_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        return audit
    finally:
        store.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--repo-id",
        default="local/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10",
    )
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("artifacts/shellgame_real_306_stage2_conversion_audit.json"),
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--min-free-gib", type=float, default=35.0)
    parser.add_argument("--max-command-position-error-m", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    if args.action_horizon <= 0:
        parser.error("--action-horizon must be positive")
    return args


if __name__ == "__main__":
    try:
        convert(parse_args())
    except Exception as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        raise
