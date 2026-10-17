#!/usr/bin/env python3
"""Summarize boundary and event errors from a recurrent-rollout JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme.four_task_state import OrderedTargetState  # noqa: E402
from openpi.tasks.robomme.four_task_state import PickCountState  # noqa: E402
from openpi.tasks.robomme.four_task_state import TargetIdentityState  # noqa: E402


STATE_CHANGING = {
    "target_visible",
    "target_covered",
    "swap_complete",
    "pick_complete",
    "place_complete",
    "press_complete",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _exact(clip: dict[str, Any]) -> bool:
    return clip["prediction"] == clip["expected"]


def _event(clip: dict[str, Any] | None) -> str:
    if clip is None or clip["prediction"] is None:
        return "<invalid>"
    return str(clip["prediction"]["event"])


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _snapshot(state: Any) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in asdict(state).items()
    }


def _oracle_boundary_final_state(episode: dict[str, Any]) -> dict[str, Any]:
    """Offline diagnostic: discard predictions outside GT completion windows."""
    source = str(episode["source"])
    final = episode["oracle"]["final_state"]
    if source in ("videounmask_variable_demo", "videounmaskswap_local_event"):
        state: Any = TargetIdentityState.empty(tuple(final["target_colors"]))
    elif source == "videoplaceorder_local_event":
        state = OrderedTargetState(final["target_color"], int(final["queried_ordinal"]))
    elif source == "pickxtimes_local_event":
        state = PickCountState(final["target_color"], int(final["required_count"]))
    else:
        raise ValueError(source)

    for clip in episode["clips"]:
        if clip["expected"]["event"] not in STATE_CHANGING or clip["prediction"] is None:
            continue
        event = clip["prediction"]
        event_type = str(event["event"])
        try:
            if source in ("videounmask_variable_demo", "videounmaskswap_local_event"):
                if event_type in ("target_visible", "target_covered"):
                    entity, region = event["entity"], event["region_a"]
                    if entity is not None and region is not None:
                        state = state.observe_target(
                            str(entity).removesuffix("_cube"),
                            str(region),
                            covered=event_type == "target_covered",
                        )
                elif event_type == "swap_complete" and event["region_a"] and event["region_b"]:
                    state = state.apply_swap(str(event["region_a"]), str(event["region_b"]))
            elif source == "videoplaceorder_local_event":
                if event_type == "place_complete" and event["region_a"]:
                    state = state.place_complete(state.written_count + 1, str(event["region_a"]))
                elif event_type == "swap_complete" and event["region_a"] and event["region_b"]:
                    state = state.swap_complete(str(event["region_a"]), str(event["region_b"]))
            elif event_type in ("pick_complete", "place_complete", "press_complete"):
                state = state.apply(event_type)
        except ValueError:
            pass
    return _snapshot(state)


def _answer_from_state(source: str, state: dict[str, Any]) -> Any:
    if source in ("videounmask_variable_demo", "videounmaskswap_local_event"):
        return state["target_cells"]
    if source == "videoplaceorder_local_event":
        return state["target_cells"][int(state["queried_ordinal"]) - 1]
    return {
        "completed_count": state["completed_count"],
        "holding": state["holding"],
        "done": state["done"],
    }


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for source in sorted({str(episode["source"]) for episode in episodes}):
        subset = [episode for episode in episodes if episode["source"] == source]
        clips = [clip for episode in subset for clip in episode["clips"]]
        by_sample_type: dict[str, Any] = {}
        for sample_type in sorted({str(clip["sample_type"]) for clip in clips}):
            selected = [clip for clip in clips if clip["sample_type"] == sample_type]
            false_commits = sum(
                _event(clip) in STATE_CHANGING
                and clip["expected"]["event"] not in STATE_CHANGING
                for clip in selected
            )
            by_sample_type[sample_type] = {
                "clips": len(selected),
                "event_accuracy": _safe_ratio(
                    sum(clip["event_correct"] for clip in selected), len(selected)
                ),
                "exact_accuracy": _safe_ratio(sum(_exact(clip) for clip in selected), len(selected)),
                "false_commit_rate": _safe_ratio(false_commits, len(selected)),
            }

        confusion = Counter(
            (str(clip["expected"]["event"]), _event(clip)) for clip in clips
        )
        boundary_pairs = Counter()
        for episode in subset:
            positive_by_index = {
                int(clip["event_index"]): clip
                for clip in episode["clips"]
                if clip["expected"]["event"] in STATE_CHANGING
            }
            for incomplete in episode["clips"]:
                if incomplete["sample_type"] not in {
                    "incomplete_event",
                    "incomplete_place",
                    "incomplete_placement",
                    "incomplete_swap",
                }:
                    continue
                positive = positive_by_index.get(int(incomplete["event_index"]))
                if positive is None:
                    continue
                predicted_early = _event(incomplete) == positive["expected"]["event"]
                positive_correct = bool(positive["event_correct"])
                if not predicted_early and positive_correct:
                    label = "clean_boundary"
                elif predicted_early and positive_correct:
                    label = "early_duplicate"
                elif predicted_early:
                    label = "early_then_positive_wrong"
                else:
                    label = "positive_missed_or_wrong"
                boundary_pairs[label] += 1

        sources[source] = {
            "clips": len(clips),
            "episodes": len(subset),
            "event_accuracy": _safe_ratio(sum(clip["event_correct"] for clip in clips), len(clips)),
            "exact_accuracy": _safe_ratio(sum(_exact(clip) for clip in clips), len(clips)),
            "field_accuracy": {
                field: _safe_ratio(
                    sum(
                        clip["prediction"] is not None
                        and clip["prediction"].get(field) == clip["expected"].get(field)
                        for clip in clips
                    ),
                    len(clips),
                )
                for field in ("event", "entity", "region_a", "region_b")
            },
            "by_sample_type": by_sample_type,
            "boundary_pairs": dict(sorted(boundary_pairs.items())),
            "event_confusion": [
                {"expected": expected, "predicted": predicted, "count": count}
                for (expected, predicted), count in sorted(confusion.items())
            ],
            "dedup_final_state_accuracy": _safe_ratio(
                sum(
                    episode["gated_dedup"]["final_state"] == episode["oracle"]["final_state"]
                    for episode in subset
                ),
                len(subset),
            ),
            "dedup_final_answer_accuracy": _safe_ratio(
                sum(
                    episode["gated_dedup"]["final_answer"] == episode["oracle"]["final_answer"]
                    for episode in subset
                ),
                len(subset),
            ),
            "oracle_boundary_final_state_accuracy": _safe_ratio(
                sum(
                    _oracle_boundary_final_state(episode) == episode["oracle"]["final_state"]
                    for episode in subset
                ),
                len(subset),
            ),
            "oracle_boundary_final_answer_accuracy": _safe_ratio(
                sum(
                    _answer_from_state(source, _oracle_boundary_final_state(episode))
                    == episode["oracle"]["final_answer"]
                    for episode in subset
                ),
                len(subset),
            ),
        }
    return {"schema_version": 1, "sources": sources}


def main() -> None:
    args = parse_args()
    episodes = [
        json.loads(line)
        for line in args.rollout.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize(episodes)
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
