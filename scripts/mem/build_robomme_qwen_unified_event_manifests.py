#!/usr/bin/env python3
"""Convert four RoboMME local-event datasets to one task-neutral contract."""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
from typing import Any

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme import four_task_temporal_contract as temporal_contract  # noqa: E402
from openpi.tasks.robomme.qwen3vl_unified_event_contract import compact_response  # noqa: E402
from openpi.tasks.robomme.qwen3vl_unified_event_contract import entity_for_color  # noqa: E402
from scripts.mem.build_videoplaceorder_qwen3vl_sft_manifest import _episode_metadata  # noqa: E402

DEFAULT_OUTPUT = _ROOT / "artifacts/robomme_qwen_unified_events_seed260826"
DEFAULT_INPUTS = {
    "videounmask_variable_demo": _ROOT / "artifacts/videounmask_qwen3vl_variable_demo_seed260826",
    "videounmaskswap_local_event": _ROOT / "artifacts/videounmaskswap_qwen3vl_local_events_seed260826",
    "videoplaceorder_local_event": _ROOT / "artifacts/videoplaceorder_qwen3vl_local_events_seed260826",
    "pickxtimes_local_event": _ROOT / "artifacts/pickxtimes_qwen3vl_local_events_seed260826",
}
_CELL_RE = re.compile(r"r(\d+)_c(\d+)\Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pickxtimes-dir",
        type=Path,
        default=DEFAULT_INPUTS["pickxtimes_local_event"],
    )
    parser.add_argument(
        "--shellgame-train",
        type=Path,
        default=_ROOT / "artifacts/shellgame_qwen3vl_gt_event_sft_v1/train.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists: {path}; pass --overwrite")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _fixed(
    row: dict[str, Any],
    *,
    target: str,
    goal: str,
    focus_entity: str | None,
    candidate_region_count: int | None = None,
) -> dict[str, Any]:
    temporal_contract.validate_teacher_frame_indices(row["frame_indices"])
    return {
        **row,
        "schema_version": 3,
        "contract": "unified_causal_event_v1",
        "teacher_frame_count": temporal_contract.TEACHER_FRAME_COUNT,
        "goal": goal,
        "focus_entity": focus_entity,
        "candidate_region_count": candidate_region_count,
        "target": target,
    }


def _resample_teacher_frames(frame_indices: list[int]) -> list[int]:
    """Preserve a causal span while normalizing legacy replay to twelve frames."""

    if not frame_indices:
        raise ValueError("Cannot resample an empty teacher clip")
    positions = np.linspace(
        0,
        len(frame_indices) - 1,
        temporal_contract.TEACHER_FRAME_COUNT,
    ).round().astype(int)
    result = [int(frame_indices[index]) for index in positions]
    temporal_contract.validate_teacher_frame_indices(result)
    return result


def _goal_for_focus(original_goal: str, focus_entity: str) -> tuple[str, str]:
    color = focus_entity.removesuffix("_cube")
    if color in original_goal.lower():
        return original_goal, "original_instruction"
    return (
        f"Track the {color} cube through the observed motion and report its latest local event.",
        "focus_consistent_auxiliary",
    )


def _cell_key(cell: str) -> tuple[int, int]:
    match = _CELL_RE.fullmatch(cell)
    if match is None:
        raise ValueError(f"Invalid legacy cell: {cell!r}")
    return int(match.group(1)), int(match.group(2))


def _unmask(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells_by_episode: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        cell = row.get("target_cell")
        if cell is not None and cell not in cells_by_episode[int(row["episode_index"])]:
            cells_by_episode[int(row["episode_index"])].append(str(cell))
    regions = {
        episode: {cell: f"region_{index}" for index, cell in enumerate(sorted(cells, key=_cell_key))}
        for episode, cells in cells_by_episode.items()
    }
    result = []
    for row in rows:
        old = json.loads(row["target"])
        color = str(row["target_color"])
        entity = entity_for_color(color)
        if old["event"] in ("target_visible", "target_covered"):
            target = compact_response(
                old["event"],
                entity=entity,
                region_a=regions[int(row["episode_index"])][str(row["target_cell"])],
            )
        else:
            target = compact_response(old["event"])
        goal, goal_source = _goal_for_focus(str(row["goal"]), entity)
        result.append(
            _fixed(
                {**row, "original_goal": str(row["goal"]), "goal_source": goal_source},
                target=target,
                goal=goal,
                focus_entity=entity,
                candidate_region_count=len(regions[int(row["episode_index"])]),
            )
        )
    return result


def _unmask_swap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        old = json.loads(row["target"])
        event = str(old["event"])
        focus = None
        if event in ("target_visible", "target_covered"):
            focus = entity_for_color(str(old["target_color"]))
            target = compact_response(
                event,
                entity=focus,
                region_a=str(old["container_id"]).replace("slot_", "region_"),
            )
        elif event == "swap_complete":
            pair = [str(value).replace("slot_", "region_") for value in old["container_ids"]]
            target = compact_response(event, region_a=pair[0], region_b=pair[1])
        else:
            target = compact_response(event)
        if focus is None:
            goal, goal_source = str(row["goal"]), "original_instruction"
        else:
            goal, goal_source = _goal_for_focus(str(row["goal"]), focus)
        result.append(
            _fixed(
                {**row, "original_goal": str(row["goal"]), "goal_source": goal_source},
                target=target,
                goal=goal,
                focus_entity=focus,
                candidate_region_count=int(row["num_containers"]),
            )
        )
    return result


@lru_cache(maxsize=1)
def _vpo_maps(h5_path: str) -> dict[int, dict[str, Any]]:
    result = {}
    with h5py.File(h5_path, "r") as h5:
        for episode_index in range(100):
            item = _episode_metadata(h5[f"episode_{episode_index}"], episode_index)
            points = [tuple(float(value) for value in drop["target_xy"]) for drop in item["drops"]]
            final = np.asarray(item["target_xy"], dtype=np.float64)
            nearest_distance = min(
                float(np.linalg.norm(final - np.asarray(point))) for point in points
            )
            # In several hard episodes a target is moved to a new empty spatial
            # slot instead of another demonstrated drop slot. Include that new
            # slot in the same row-major region vocabulary.
            anchors = [*points]
            final_anchor = min(
                range(len(points)),
                key=lambda index: float(np.linalg.norm(final - np.asarray(points[index]))),
            )
            if nearest_distance > 16.0:
                final_anchor = len(anchors)
                anchors.append(tuple(float(value) for value in final))
            order = sorted(range(len(anchors)), key=lambda index: anchors[index])
            anchor_regions = {
                anchor_index: f"region_{order.index(anchor_index)}"
                for anchor_index in range(len(anchors))
            }
            event_regions = {
                event_index: anchor_regions[event_index] for event_index in range(len(points))
            }
            result[episode_index] = {
                "goal": str(item["goal"]),
                "event_regions": event_regions,
                "queried_region": event_regions[int(item["ordinal"]) - 1],
                "final_region": anchor_regions[final_anchor],
                "candidate_region_count": len(anchors),
            }
    return result


def _place_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    maps = _vpo_maps(str(rows[0]["h5_path"]))
    result = []
    for row in rows:
        old = json.loads(row["target"])
        event = str(old["event"])
        mapping = maps[int(row["episode_index"])]
        if event == "place_complete":
            target = compact_response(
                event, region_a=mapping["event_regions"][int(row["event_index"])]
            )
        elif event == "swap_complete":
            target = compact_response(
                event,
                region_a=mapping["queried_region"],
                region_b=mapping["final_region"],
            )
        else:
            target = compact_response(event)
        result.append(
            _fixed(
                row,
                target=target,
                goal=mapping["goal"],
                focus_entity=None,
                candidate_region_count=int(mapping["candidate_region_count"]),
            )
        )
    return result


def _pick(rows: list[dict[str, Any]], prompts: dict[int, str]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        old = json.loads(row["target"])
        event = str(old["event"])
        focus = entity_for_color(str(row["target_color"]))
        entity = focus if event in ("pick_complete", "place_complete") else None
        target = compact_response(event, entity=entity)
        result.append(
            _fixed(
                row,
                target=target,
                goal=prompts[int(row["episode_index"])],
                focus_entity=focus,
                candidate_region_count=None,
            )
        )
    return result


def _shellgame(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    region = {
        "screen_left_cup": "region_0",
        "screen_middle_cup": "region_1",
        "screen_right_cup": "region_2",
    }
    result = []
    for row in rows:
        source_frames = [int(value) for value in row["frame_indices"]]
        teacher_frames = _resample_teacher_frames(source_frames)
        old = json.loads(row["target"])
        sample_type = str(row["sample_type"])
        if sample_type == "reveal":
            target = compact_response(
                "target_visible", entity="tracked_object", region_a=region[old["screen_cup"]]
            )
        elif sample_type == "swap":
            target = compact_response(
                "swap_complete",
                region_a=region[old["screen_pair"][0]],
                region_b=region[old["screen_pair"][1]],
            )
        elif sample_type == "no_event":
            target = compact_response("no_completed_event")
        else:
            target = compact_response("incomplete_event")
        result.append(
            _fixed(
                {
                    **row,
                    "source": "shellgame_unified_replay",
                    "frame_indices": teacher_frames,
                    "source_frame_count": len(source_frames),
                    "temporal_resample": "linspace_preserve_causal_span",
                },
                target=target,
                goal="Track the hidden object under the cups through the observed motion.",
                focus_entity="tracked_object",
                candidate_region_count=3,
            )
        )
    return result


def main() -> None:
    args = parse_args()
    labels = json.loads(
        (_ROOT / "data/robomme_extracted/pickxtimes_event_labels_w10_v3_press5_gripper.json").read_text(
            encoding="utf-8"
        )
    )
    prompts = {int(item["episode_index"]): str(item["prompts"][0]) for item in labels["episodes"]}
    inputs = {**DEFAULT_INPUTS, "pickxtimes_local_event": args.pickxtimes_dir}
    converters = {
        "videounmask_variable_demo": _unmask,
        "videounmaskswap_local_event": _unmask_swap,
        "videoplaceorder_local_event": _place_order,
        "pickxtimes_local_event": lambda rows: _pick(rows, prompts),
    }
    summary = {
        "schema_version": 3,
        "contract": "unified_causal_event_v1",
        "teacher_frame_count": temporal_contract.TEACHER_FRAME_COUNT,
        "tasks": {},
    }
    for source, input_dir in inputs.items():
        task_summary = {}
        for split in ("train", "dev", "test"):
            converted = converters[source](_read(input_dir / f"{split}.jsonl"))
            _write(args.output_dir / source / f"{split}.jsonl", converted, args.overwrite)
            task_summary[split] = {
                "samples": len(converted),
                "events": dict(sorted(Counter(json.loads(row["target"])["event"] for row in converted).items())),
            }
        summary["tasks"][source] = task_summary
    shell = _shellgame(_read(args.shellgame_train))
    _write(args.output_dir / "shellgame_unified_replay/train.jsonl", shell, args.overwrite)
    summary["tasks"]["shellgame_unified_replay"] = {"train": {"samples": len(shell)}}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
