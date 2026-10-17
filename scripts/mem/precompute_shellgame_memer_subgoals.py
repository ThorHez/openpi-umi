#!/usr/bin/env python3
"""Cache causal MemER subgoals for ShellGame episodes.

Only the recorded 60-frame observation prefix and its final frame are passed
to MemER.  Dataset metadata is read after inference and is used exclusively
for scoring the predicted screen/world slot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


DEFAULT_RAW_ROOT = Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_absolute_eef_phase_instruction_dataset"
)
DEFAULT_MEMER_ROOT = Path(
    "/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/"
    "examples/robomme/subgoal_prediction/qwenvl"
)
DEFAULT_ADAPTER = Path(
    "/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/ckpts/"
    "vlm_subgoal_predictor/memer/grounded_subgoal/checkpoint-1300"
)
TASK_GOAL = "Watch the video carefully, then grasp and lift the cup containing the ball."
COORDINATE = re.compile(r"<\s*(\d+)\s*,\s*(\d+)\s*>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--memer-api-root", type=Path, default=DEFAULT_MEMER_ROOT)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-goal", default=TASK_GOAL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _predicted_slots(subgoal: str) -> tuple[list[list[int]], str | None, str | None]:
    coordinates = [[int(x), int(y)] for x, y in COORDINATE.findall(subgoal)]
    if not coordinates:
        return coordinates, None, None
    x = coordinates[-1][0]
    screen = "left" if x < 256 / 3 else "right" if x >= 2 * 256 / 3 else "middle"
    # The ShellGame front-view camera is horizontally mirrored relative to
    # world cup identities.
    world = {"left": "right", "middle": "middle", "right": "left"}[screen]
    return coordinates, screen, world


def _last_raw_response(log_path: Path) -> str | None:
    if not log_path.is_file():
        return None
    response = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "response" in row:
            response = str(row["response"])
    return response


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.adapter = args.adapter.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.memer_api_root = args.memer_api_root.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    requested = [int(value.strip()) for value in args.episodes.split(",") if value.strip()]
    episodes = list(dict.fromkeys(requested))
    if not episodes:
        raise ValueError("--episodes must contain integer episode IDs")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    if not args.adapter.is_dir():
        raise FileNotFoundError(args.adapter)

    sys.path.insert(0, str(args.memer_api_root))
    from api_memer import Qwen3VLModelMemER  # noqa: PLC0415

    model = Qwen3VLModelMemER(adapter_path=str(args.adapter))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "MemER zero-shot ShellGame grounded-subgoal cache",
        "adapter": str(args.adapter),
        "task_goal": args.task_goal,
        "causal_inputs": "recorded third_person_images[0:60] plus frame 59",
        "metadata_used_for_inference": False,
        "episodes": episodes,
        "records": [],
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for ordinal, episode in enumerate(episodes, start=1):
        episode_dir = args.raw_root / f"episode_{episode:06d}"
        with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
            prefix = np.asarray(source["third_person_images"][:60], dtype=np.uint8)
        if prefix.shape != (60, 224, 224, 3):
            raise ValueError(f"episode {episode}: unexpected prefix shape {prefix.shape}")

        save_dir = args.work_dir / f"episode_{episode:06d}"
        model.start_new_episode(str(save_dir), prefix, args.task_goal)
        model.add_execution_frame(prefix[-1])
        subgoal = model.call()
        raw_response = _last_raw_response(
            args.work_dir / f"episode_{episode:06d}_MemER_log.jsonl"
        )
        coordinates, predicted_screen_slot, predicted_world_slot = _predicted_slots(subgoal)

        # Scoring-only fields are added after MemER has returned.
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        target = str(metadata["target_cup_identity"])
        record = {
            "episode": episode,
            "subgoal": subgoal,
            "raw_response": raw_response,
            "coordinates_256": coordinates,
            "predicted_screen_slot": predicted_screen_slot,
            "predicted_world_slot": predicted_world_slot,
            "target_world_slot_scoring_only": target,
            "grounding_parseable": predicted_world_slot is not None,
            "grounding_correct": predicted_world_slot == target,
            "log": str(
                (args.work_dir / f"episode_{episode:06d}_MemER_log.jsonl").resolve()
            ),
        }
        payload["records"].append(record)
        payload["summary"] = {
            "completed": len(payload["records"]),
            "requested": len(episodes),
            "parseable": sum(row["grounding_parseable"] for row in payload["records"]),
            "grounding_correct": sum(row["grounding_correct"] for row in payload["records"]),
        }
        _write_manifest(args.output, payload)
        print(
            f"[{ordinal}/{len(episodes)}] ep={episode} subgoal={subgoal!r} "
            f"pred={predicted_world_slot} target={target} "
            f"correct={record['grounding_correct']}",
            flush=True,
        )

    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
