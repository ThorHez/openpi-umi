#!/usr/bin/env python3
"""Precompute recurrent-MEM GroundSG strings for official RoboMME rollouts."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


_OPENPI_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _OPENPI_ROOT.parent
_POLICY_ROOT = _WORKSPACE / "robomme_policy_learning"
sys.path.insert(0, str(_POLICY_ROOT / "third_party/robomme_benchmark/src"))
sys.path.append(str(_WORKSPACE / "robomme/.venv/lib/python3.11/site-packages"))
sys.path.insert(0, str(_OPENPI_ROOT))

from robomme.env_record_wrapper import BenchmarkEnvBuilder  # noqa: E402
from scripts.mem import eval_videounmask_memory_multichoice as memory_eval  # noqa: E402
from scripts.mem import robomme_fixed_chunk_inference as fixed_memory  # noqa: E402


DEFAULT_TRAINING_DIR = _OPENPI_ROOT / (
    "checkpoints/robomme_single_task_unmask_equal_exposure_seed260827_260827"
)
DEFAULT_OUTPUT = _POLICY_ROOT / (
    "runs/evaluation/recurrent-mem-groundsg-videounmask-val10/manifest.json"
)
COLOR_PATTERN = re.compile(r"container hiding the (red|green|blue) cube", re.IGNORECASE)
POINT_PATTERN = re.compile(r"<(\d+),\s*(\d+)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("train", "val", "test"), default="val")
    parser.add_argument("--episodes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    return parser.parse_args()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _prompt(info: dict[str, Any]) -> str:
    value = info["task_goal"]
    while isinstance(value, list | tuple):
        value = value[0]
    return str(value).strip()


def _point(text: str) -> list[int] | None:
    match = POINT_PATTERN.search(text)
    return [int(match.group(1)), int(match.group(2))] if match else None


def _predicted_cells(output: dict[str, np.ndarray], ordered: list[tuple[str, list[float]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    predictions = output["all_predictions"][-1]
    for color in ("red", "green", "blue"):
        class_id = int(predictions[fixed_memory.field_index(f"{color}_cell")])
        region = class_id - 1
        if 0 <= region < len(ordered):
            selected_color, point_yx = ordered[region]
            result[color] = {
                "class_id": class_id,
                "region": region,
                "valid": True,
                "selected_candidate_color": selected_color,
                "point_yx": [float(value) for value in point_yx],
            }
        else:
            result[color] = {
                "class_id": class_id,
                "region": region,
                "valid": False,
                "selected_candidate_color": None,
                "point_yx": None,
            }
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OPENPI_DATA_HOME", str(_WORKSPACE / ".cache/openpi"))
    episodes = [int(item) for item in args.episodes.split(",") if item.strip()]
    predictor = fixed_memory.FixedChunkMemoryPredictor(args.training_dir)
    backbone = fixed_memory.load_backbone()
    builder = BenchmarkEnvBuilder(
        env_id="VideoUnmask",
        dataset=args.dataset,
        action_space="joint_angle",
        gui_render=False,
        max_steps=10,
    )
    records = []
    for episode_id in episodes:
        env = None
        try:
            env = builder.make_env_for_episode(episode_id)
            observation, info = env.reset()
            prompt = _prompt(info)
            goal_colors = [value.lower() for value in COLOR_PATTERN.findall(prompt)]
            if not goal_colors:
                raise ValueError(f"Could not parse target colors from: {prompt}")
            frames_like = observation["front_rgb_list"]
            demo_frames = np.stack(
                [_as_numpy(frame).astype(np.uint8) for frame in frames_like[:-1]]
            )
            centers = memory_eval._cube_centers(demo_frames[0])  # noqa: SLF001
            ordered = sorted(centers.items(), key=lambda item: (item[1][0], item[1][1]))
            tokens = fixed_memory.encode_frames(
                backbone, demo_frames, batch_size=args.encode_batch_size
            )
            chunks = fixed_memory.tokens_to_chunks(tokens)
            padded_colors = [fixed_memory.COLOR_IDS[color] for color in goal_colors[:2]]
            padded_colors += [0] * (2 - len(padded_colors))
            output = predictor.predict_encoded(
                chunks,
                task_id=0,
                goal_color_ids=tuple(padded_colors),
                required_count=0,
                queried_ordinal=0,
                num_regions=len(ordered),
            )
            predicted_cells = _predicted_cells(output, ordered)
            first_color = goal_colors[0]
            first_prediction = predicted_cells[first_color]
            subgoal = None
            if first_prediction["valid"]:
                y, x = np.rint(first_prediction["point_yx"]).astype(int)
                subgoal = (
                    f"pick up the container at <{y}, {x}> that hides the "
                    f"{first_color} cube"
                )
            oracle_subgoal = str(info["grounded_subgoal_online"])
            oracle_region = next(
                index for index, (color, _) in enumerate(ordered) if color == first_color
            )
            records.append(
                {
                    "task": "VideoUnmask",
                    "dataset": args.dataset,
                    "episode_id": episode_id,
                    "difficulty": str(env.unwrapped.difficulty),
                    "task_goal": prompt,
                    "goal_colors": goal_colors,
                    "single_target_supported": len(goal_colors) == 1,
                    "demo_frames": len(demo_frames),
                    "memory_chunks": len(chunks),
                    "candidate_centers_yx": centers,
                    "ordered_candidates": [
                        {"region": index, "color": color, "point_yx": point}
                        for index, (color, point) in enumerate(ordered)
                    ],
                    "predicted_cells": predicted_cells,
                    "predicted_first_region": first_prediction["region"],
                    "oracle_first_region": oracle_region,
                    "first_region_exact": bool(
                        first_prediction["valid"]
                        and first_prediction["region"] == oracle_region
                    ),
                    "predicted_grounded_subgoal": subgoal,
                    "oracle_grounded_subgoal": oracle_subgoal,
                    "oracle_point_yx": _point(oracle_subgoal),
                    "final_write_gate": (
                        float(output["write_gates"][-1])
                        if len(output["write_gates"])
                        else None
                    ),
                }
            )
            print(
                f"episode={episode_id} difficulty={env.unwrapped.difficulty} "
                f"colors={goal_colors} pred={first_prediction['region']} "
                f"oracle={oracle_region} exact={records[-1]['first_region_exact']}",
                flush=True,
            )
        finally:
            if env is not None:
                with contextlib.suppress(Exception):
                    env.close()

    supported = [record for record in records if record["single_target_supported"]]
    payload = {
        "schema_version": 1,
        "memory_training_dir": str(args.training_dir.resolve()),
        "memory_checkpoint": str(predictor.checkpoint),
        "dataset": args.dataset,
        "episodes": records,
        "summary": {
            "episodes": len(records),
            "single_target_supported": len(supported),
            "unsupported_multi_target": len(records) - len(supported),
            "single_target_first_region_exact": sum(
                record["first_region_exact"] for record in supported
            ),
            "single_target_first_region_exact_rate": (
                float(np.mean([record["first_region_exact"] for record in supported]))
                if supported
                else None
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
