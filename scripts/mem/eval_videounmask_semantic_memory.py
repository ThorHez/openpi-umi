#!/usr/bin/env python3
"""Evaluate VideoUnmask memory, including nearest-container causal controls."""

from __future__ import annotations

import argparse
import json
import pathlib

import flax
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.tasks.robomme.videounmask import semantic_memory_target_event
from openpi.training.mem import robomme_videounmask_dataset
from openpi.training.mem.recipes import robomme_videounmask_target_event_pretrain as loss_recipe

COLOR_NAMES = ("red", "green", "blue")
ABLATIONS = ("full", "late_only", "zero_video", "wrong_video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--h5", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _jax_batch(batch, keys):
    return {key: jnp.asarray(batch[key].numpy() if isinstance(batch[key], torch.Tensor) else batch[key]) for key in keys}


def _transform(inputs, ablation):
    result = dict(inputs)
    video = inputs["demo_patch_tokens"]
    if ablation == "late_only":
        result["demo_patch_tokens"] = jnp.repeat(video[:, -1:], video.shape[1], axis=1)
    elif ablation == "zero_video":
        result["demo_patch_tokens"] = jnp.zeros_like(video)
    elif ablation == "wrong_video":
        result["demo_patch_tokens"] = jnp.roll(video, 1, axis=0)
    elif ablation != "full":
        raise ValueError(ablation)
    return result


def _cube_centers(image: np.ndarray) -> dict[str, list[float]]:
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    # Rendered cube faces are nearly pure RGB (roughly 231 in their dominant
    # channel); strict masks avoid the reddish wood table and robot pixels.
    masks = {
        "red": (red > 200) & (green < 50) & (blue < 50),
        "green": (green > 200) & (red < 50) & (blue < 50),
        "blue": (blue > 200) & (red < 50) & (green < 50),
    }
    centers = {}
    for color, mask in masks.items():
        y, x = np.nonzero(mask)
        if len(y) < 4:
            raise ValueError(f"Could not segment {color} cube; pixels={len(y)}")
        centers[color] = [float(np.mean(y)), float(np.mean(x))]
    return centers


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or args.run_dir / "checkpoints/step_1000.msgpack"
    payload = json.loads(args.labels.read_text(encoding="utf-8"))
    val_indices = [int(value) for value in payload["val_episode_indices"]]
    dataset = robomme_videounmask_dataset.VideoUnmaskFeatureDataset(
        args.features, args.labels, episode_indices=val_indices
    )
    raw = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    inputs = _jax_batch(raw, ("demo_patch_tokens", "prompt_tokens", "prompt_mask", "frame_mask"))
    targets = _jax_batch(raw, ("target_point", "target_cell", "target_color"))
    model = semantic_memory_target_event.VideoUnmaskTargetEventMemory(
        encoder_width=int(config["encoder_width"]),
        encoder_depth=int(config["encoder_depth"]),
        memory_width=int(config["memory_width"]),
        memory_depth=int(config["memory_depth"]),
        num_memory_tokens=int(config["memory_tokens"]),
    )
    variables = model.init(jax.random.key(0), **inputs, train=False)
    params = flax.serialization.from_bytes(variables["params"], checkpoint.read_bytes())

    metadata = {int(item["episode_index"]): item for item in payload["episodes"]}
    with h5py.File(args.h5, "r") as source:
        centers = {
            index: _cube_centers(source[f"episode_{index}/timestep_0/obs/front_rgb"][()])
            for index in val_indices
        }

    result = {
        "checkpoint": str(checkpoint.resolve()),
        "val_episode_indices": val_indices,
        "ablations": {},
    }
    for ablation in ABLATIONS:
        outputs = model.apply({"params": params}, **_transform(inputs, ablation), train=False)
        _, metrics = loss_recipe.compute_losses(outputs, targets)
        host_metrics = {key: float(value) for key, value in jax.device_get(metrics).items()}
        predicted_points = np.asarray(outputs["target_point"]) * 255.0
        records = []
        correct = 0
        for row, episode_index in enumerate(val_indices):
            candidate_centers = centers[episode_index]
            point = predicted_points[row]
            selected_color = min(
                COLOR_NAMES,
                key=lambda color: float(np.linalg.norm(point - np.asarray(candidate_centers[color]))),
            )
            target_color = metadata[episode_index]["target_color"]
            correct += int(selected_color == target_color)
            records.append(
                {
                    "episode_index": episode_index,
                    "target_color": target_color,
                    "selected_color": selected_color,
                    "selection_correct": selected_color == target_color,
                    "predicted_point_yx": point.tolist(),
                    "target_point_yx": metadata[episode_index]["target_point_yx"],
                    "candidate_centers_yx": candidate_centers,
                }
            )
        result["ablations"][ablation] = {
            "metrics": host_metrics,
            "nearest_container_accuracy": correct / len(val_indices),
            "episodes": records,
        }
        print(
            f"{ablation}: distance={host_metrics['point_distance_px']:.2f}px "
            f"within20={host_metrics['within_20px']:.3f} nearest_container={correct}/{len(val_indices)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
