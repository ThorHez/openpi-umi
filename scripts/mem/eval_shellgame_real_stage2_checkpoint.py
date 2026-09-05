#!/usr/bin/env python3
"""Offline held-out evaluation for the real ShellGame Stage-2 checkpoint.

The evaluator reads the exact converted LeRobot rows used by training, sends
241 immutable history frames plus the current frame, and compares the returned
16x10 current-anchored action chunk with the stored ground truth.  History can
be kept intact, zeroed, or replaced by a different final-cup episode for causal
memory checks.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from PIL import Image
import pyarrow.parquet as pq

from openpi.utils.pose_utils import pose10d_to_mat

DATASET = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10"
)
LABELS = Path("/data2/hzl_workspace_for_pi_mem/labels_merged_306_degap.jsonl")
PROMPT = "The shell game has ended. Grasp and lift the cup containing the ball."
HISTORY_FRAMES = 241
ACTION_HORIZON = 16
ACTION_DIM = 10
IMAGE_KEY = "observation.left_wrist_0_rgb_0"
VIDEO_FRAME_KEY_PREFIX = "left_wrist_0_rgb_0_"
VIDEO_CURRENT_STEP_KEY = "left_wrist_0_rgb_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--history-mode", choices=("normal", "zero", "wrong_episode"), default="normal")
    parser.add_argument("--episodes-per-class", type=int, default=3)
    parser.add_argument("--samples-per-frame", type=int, default=2)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split-domain", choices=("old306", "cup0903"))
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def episode_path(dataset: Path, episode_id: int) -> Path:
    candidates = sorted(dataset.glob(f"data/chunk-*/episode_{episode_id:06d}.parquet"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one parquet for episode {episode_id}, got {candidates}")
    return candidates[0]


def decode_image(cell: dict) -> np.ndarray:
    image = Image.open(BytesIO(cell["bytes"])).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (224, 224, 3):
        raise ValueError(f"Unexpected image shape {array.shape}")
    return np.ascontiguousarray(array)


def load_labels(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [int(row["episode_id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Labels are not ordered by contiguous episode_id")
    return rows


def choose_balanced_validation_episodes(
    dataset: Path,
    labels: list[dict],
    episodes_per_class: int,
    *,
    split_manifest: Path | None = None,
    split_domain: str | None = None,
    episode_offset: int = 0,
) -> list[int]:
    if split_manifest is not None:
        if split_domain is None:
            raise ValueError("--split-domain is required with --split-manifest")
        manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
        episode_ids = [
            int(value) - episode_offset
            for value in manifest["episode_split"]["validation"][split_domain]["global_episode_ids"]
        ]
    else:
        audit = json.loads((dataset / "conversion_audit.json").read_text(encoding="utf-8"))
        episode_ids = audit["validation_episode_ids"]
    by_class: dict[int, list[int]] = defaultdict(list)
    for episode_id in episode_ids:
        by_class[int(labels[int(episode_id)]["final_cup"])].append(int(episode_id))
    chosen: list[int] = []
    for final_cup in range(3):
        candidates = sorted(by_class[final_cup])
        if len(candidates) < episodes_per_class:
            raise ValueError(f"Validation class {final_cup} has {len(candidates)} episodes, need {episodes_per_class}")
        chosen.extend(candidates[:episodes_per_class])
    return sorted(chosen)


def load_history(dataset: Path, episode_id: int) -> list[np.ndarray]:
    table = pq.read_table(episode_path(dataset, episode_id), columns=[IMAGE_KEY]).slice(0, HISTORY_FRAMES)
    cells = table.column(IMAGE_KEY).to_pylist()
    if len(cells) != HISTORY_FRAMES:
        raise ValueError(f"Episode {episode_id} has only {len(cells)} history frames")
    return [decode_image(cell) for cell in cells]


def load_eval_rows(dataset: Path, episode_id: int) -> list[dict]:
    columns = [
        "observation.robot0_eef_pos",
        "observation.robot0_eef_rot_axis_angle",
        "observation.robot0_gripper_width",
        IMAGE_KEY,
        "actions",
        "frame_index",
        "final_cup",
    ]
    rows = pq.read_table(episode_path(dataset, episode_id), columns=columns).to_pylist()
    rows.sort(key=lambda row: int(row["frame_index"]))
    return rows


def eval_frame_indices(length: int) -> list[int]:
    latest = max(HISTORY_FRAMES, length - ACTION_HORIZON - 1)
    candidates = [HISTORY_FRAMES, HISTORY_FRAMES + 1, HISTORY_FRAMES + 5, 271, 321]
    return sorted({min(index, latest) for index in candidates if min(index, latest) < length})


def cosine(a: np.ndarray, b: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=-1)
    b_norm = np.linalg.norm(b, axis=-1)
    valid = (a_norm > threshold) & (b_norm > threshold)
    result = np.full(a_norm.shape, np.nan, dtype=np.float64)
    result[valid] = np.sum(a[valid] * b[valid], axis=-1) / (a_norm[valid] * b_norm[valid])
    return result


def finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def rotation_error_rad(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # pose_utils.normalize uses a 2-D transpose idiom, so flatten all leading
    # dimensions before converting the 6-D rotation representation.
    output_shape = pred.shape[:-1]
    pred_rotation = pose10d_to_mat(pred[..., :9].reshape(-1, 9))[..., :3, :3]
    gt_rotation = pose10d_to_mat(gt[..., :9].reshape(-1, 9))[..., :3, :3]
    relative = np.swapaxes(gt_rotation, -1, -2) @ pred_rotation
    cosine_angle = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine_angle).reshape(output_shape)


def pairwise_centroid_distance(centroids: dict[int, np.ndarray]) -> float:
    distances = [
        float(np.linalg.norm(centroids[left] - centroids[right])) for left in range(3) for right in range(left + 1, 3)
    ]
    return float(np.mean(distances))


def summarize(rows: list[dict]) -> dict:
    pred = np.asarray([row["pred_actions"] for row in rows], dtype=np.float64)
    gt = np.asarray([row["gt_actions"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["final_cup"] for row in rows], dtype=np.int64)
    frames = np.asarray([row["frame_index"] for row in rows], dtype=np.int64)
    error = pred - gt
    xyz_rmse = float(np.sqrt(np.mean(np.square(error[..., :3]))))
    zero_xyz_rmse = float(np.sqrt(np.mean(np.square(gt[..., :3]))))
    xyz_cos = cosine(pred[..., :3].reshape(-1, 3), gt[..., :3].reshape(-1, 3))
    early = frames <= HISTORY_FRAMES + 50
    early_cos = cosine(pred[early, ..., :3].reshape(-1, 3), gt[early, ..., :3].reshape(-1, 3))

    pred_features = pred[..., :3].reshape(len(rows), -1)
    gt_features = gt[..., :3].reshape(len(rows), -1)
    pred_centroids = {cup: pred_features[labels == cup].mean(axis=0) for cup in range(3)}
    gt_centroids = {cup: gt_features[labels == cup].mean(axis=0) for cup in range(3)}
    distances = np.stack(
        [np.linalg.norm(pred_features - gt_centroids[cup][None], axis=-1) for cup in range(3)],
        axis=-1,
    )
    predicted_class = np.argmin(distances, axis=-1)
    predicted_counts = np.bincount(predicted_class, minlength=3)
    pred_separation = pairwise_centroid_distance(pred_centroids)
    gt_separation = pairwise_centroid_distance(gt_centroids)

    by_class = {}
    for cup in range(3):
        mask = labels == cup
        class_error = error[mask, ..., :3]
        by_class[str(cup)] = {
            "rows": int(mask.sum()),
            "xyz_rmse_mm": float(np.sqrt(np.mean(np.square(class_error))) * 1000),
            "xyz_direction_cosine_mean": finite_mean(
                cosine(pred[mask, ..., :3].reshape(-1, 3), gt[mask, ..., :3].reshape(-1, 3))
            ),
            "pred_step0_xyz_mean_mm": (pred[mask, 0, :3].mean(axis=0) * 1000).tolist(),
            "gt_step0_xyz_mean_mm": (gt[mask, 0, :3].mean(axis=0) * 1000).tolist(),
        }

    return {
        "n_episodes": len({row["episode_id"] for row in rows}),
        "n_eval_rows": len(rows),
        "xyz_rmse_mm": xyz_rmse * 1000,
        "xyz_zero_baseline_rmse_mm": zero_xyz_rmse * 1000,
        "xyz_rmse_vs_zero_ratio": xyz_rmse / max(zero_xyz_rmse, 1e-12),
        "rotation_geodesic_rmse_deg": float(np.sqrt(np.mean(np.square(rotation_error_rad(pred, gt)))) * 180 / np.pi),
        "gripper_rmse": float(np.sqrt(np.mean(np.square(error[..., 9])))),
        "xyz_direction_cosine_mean": finite_mean(xyz_cos),
        "early_xyz_direction_cosine_mean": finite_mean(early_cos),
        "xyz_direction_opposite_fraction": float(np.mean(xyz_cos[np.isfinite(xyz_cos)] < 0)),
        "xyz_sign_agreement_fraction_by_dim": np.mean(
            np.sign(pred[..., :3]) == np.sign(gt[..., :3]), axis=(0, 1)
        ).tolist(),
        "class_nearest_gt_centroid_accuracy": float(np.mean(predicted_class == labels)),
        "class_predicted_counts": predicted_counts.tolist(),
        "class_max_predicted_fraction": float(predicted_counts.max() / predicted_counts.sum()),
        "pred_class_centroid_separation": pred_separation,
        "gt_class_centroid_separation": gt_separation,
        "class_centroid_separation_ratio": pred_separation / max(gt_separation, 1e-12),
        "by_final_cup": by_class,
    }


def main() -> None:
    args = parse_args()
    if args.episodes_per_class <= 0 or args.samples_per_frame <= 0:
        raise ValueError("episodes-per-class and samples-per-frame must be positive")
    dataset = args.dataset.resolve()
    labels = load_labels(args.labels.resolve())
    episodes = choose_balanced_validation_episodes(
        dataset,
        labels,
        args.episodes_per_class,
        split_manifest=args.split_manifest,
        split_domain=args.split_domain,
        episode_offset=args.episode_offset,
    )
    final_cups = {episode: int(labels[episode]["final_cup"]) for episode in episodes}
    donor = {
        episode: next(candidate for candidate in episodes if final_cups[candidate] != final_cups[episode])
        for episode in episodes
    }
    history_cache: dict[int, list[np.ndarray]] = {}

    def history_for(episode: int) -> list[np.ndarray]:
        source = donor[episode] if args.history_mode == "wrong_episode" else episode
        if source not in history_cache:
            history_cache[source] = load_history(dataset, source)
        if args.history_mode == "zero":
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(HISTORY_FRAMES)]
        return history_cache[source]

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    if not metadata.get("supports_cached_infer"):
        raise RuntimeError(f"Server must support cached history, metadata={metadata}")
    print(f"server metadata: {metadata}", flush=True)
    print(f"history_mode={args.history_mode} held_out_episodes={episodes}", flush=True)

    result_rows: list[dict] = []
    for episode in episodes:
        history = history_for(episode)
        upload = {"mode": "reset_history", "prompt": PROMPT}
        upload.update({f"{VIDEO_FRAME_KEY_PREFIX}{index}": frame for index, frame in enumerate(history)})
        cache_result = client.infer(upload)
        if not cache_result.get("cache_ready"):
            raise RuntimeError(f"Server rejected episode {episode} history: {cache_result}")

        rows = load_eval_rows(dataset, episode)
        for frame_index in eval_frame_indices(len(rows)):
            row = rows[frame_index]
            if int(row["frame_index"]) != frame_index:
                raise ValueError(f"Episode {episode} frame ordering mismatch at {frame_index}")
            observation = {
                "mode": "infer_step",
                "prompt": PROMPT,
                VIDEO_CURRENT_STEP_KEY: decode_image(row[IMAGE_KEY]),
                "robot0_eef_pos": np.asarray(row["observation.robot0_eef_pos"], dtype=np.float32),
                "robot0_eef_rot_axis_angle": np.asarray(row["observation.robot0_eef_rot_axis_angle"], dtype=np.float32),
                "robot0_gripper_width": np.asarray([row["observation.robot0_gripper_width"]], dtype=np.float32),
            }
            samples = []
            timings = []
            for sample_index in range(args.samples_per_frame):
                # Identical episode/frame/sample seeds across history modes
                # isolate the history intervention from diffusion noise.
                observation["noise_seed"] = episode * 1_000_003 + frame_index * 101 + sample_index
                started = time.monotonic()
                prediction = client.infer(observation)
                timings.append(time.monotonic() - started)
                actions = np.asarray(prediction["actions"], dtype=np.float64)
                if actions.shape != (ACTION_HORIZON, ACTION_DIM):
                    raise ValueError(f"Policy returned {actions.shape}, expected {(ACTION_HORIZON, ACTION_DIM)}")
                samples.append(actions)
            pred_samples = np.stack(samples)
            pred = pred_samples.mean(axis=0)
            gt = np.asarray(row["actions"], dtype=np.float64)
            result = {
                "episode_id": episode,
                "history_episode_id": donor[episode] if args.history_mode == "wrong_episode" else episode,
                "frame_index": frame_index,
                "final_cup": int(row["final_cup"]),
                "pred_actions": pred.tolist(),
                "gt_actions": gt.tolist(),
                "sample_action_std_mean": float(np.mean(np.std(pred_samples, axis=0))),
                "latency_s_mean": float(np.mean(timings)),
            }
            result_rows.append(result)
            rmse_mm = np.sqrt(np.mean(np.square(pred[..., :3] - gt[..., :3]))) * 1000
            print(
                f"ep={episode:03d} cup={result['final_cup']} frame={frame_index:03d} "
                f"history_ep={result['history_episode_id']:03d} xyz_rmse={rmse_mm:.2f}mm",
                flush=True,
            )

    payload = {
        "dataset": str(dataset),
        "history_mode": args.history_mode,
        "episodes": episodes,
        "server_metadata": metadata,
        "summary": summarize(result_rows),
        "rows": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(payload["summary"], indent=2), flush=True)
    print(f"saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
