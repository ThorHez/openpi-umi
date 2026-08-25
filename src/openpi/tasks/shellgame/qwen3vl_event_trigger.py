"""Task-agnostic clustering for low-frequency Qwen visual event proposals."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventTrigger:
    first_window_start: int
    last_window_start: int
    representative_start: int
    pair: tuple[str, str]
    supporting_windows: int


def _pair_from_prediction(prediction: Any) -> tuple[str, str] | None:
    if not isinstance(prediction, dict):
        return None
    raw_pair = prediction.get("screen_pair")
    if not isinstance(raw_pair, list) or len(raw_pair) != 2:
        return None
    return str(raw_pair[0]), str(raw_pair[1])


def cluster_event_windows(
    windows: Iterable[dict[str, Any]],
    *,
    max_positive_gap: int = 3,
) -> list[EventTrigger]:
    """Cluster positive sliding windows without consulting phase/GT labels.

    The cluster is finalized after positive proposals are separated by more
    than ``max_positive_gap`` window-start frames.  A modal pair suppresses
    repeated proposals from overlapping windows around one physical event.
    """
    positives = []
    for window in sorted(windows, key=lambda value: int(value["window_start"])):
        pair = _pair_from_prediction(window.get("prediction"))
        if pair is not None:
            positives.append((int(window["window_start"]), pair))
    if not positives:
        return []
    groups: list[list[tuple[int, tuple[str, str]]]] = [[positives[0]]]
    for item in positives[1:]:
        if item[0] - groups[-1][-1][0] <= max_positive_gap:
            groups[-1].append(item)
        else:
            groups.append([item])
    result = []
    for group in groups:
        counts = Counter(pair for _, pair in group)
        # Counter preserves first-seen order for deterministic tie breaking.
        modal_pair = counts.most_common(1)[0][0]
        starts = [start for start, _ in group]
        result.append(
            EventTrigger(
                first_window_start=starts[0],
                last_window_start=starts[-1],
                representative_start=starts[len(starts) // 2],
                pair=modal_pair,
                supporting_windows=len(group),
            )
        )
    return result


def score_event_triggers(
    triggers: Iterable[EventTrigger],
    *,
    gt_starts: list[int],
    gt_pairs: list[tuple[str, str]],
    tolerance: int = 5,
) -> dict[str, Any]:
    """Match predicted clusters to GT only for scoring, never for triggering."""
    triggers = list(triggers)
    unmatched = set(range(len(gt_starts)))
    matches = []
    for trigger_index, trigger in enumerate(triggers):
        candidates = [index for index in unmatched if abs(trigger.representative_start - gt_starts[index]) <= tolerance]
        if not candidates:
            continue
        gt_index = min(candidates, key=lambda index: abs(trigger.representative_start - gt_starts[index]))
        unmatched.remove(gt_index)
        matches.append(
            {
                "trigger_index": trigger_index,
                "gt_index": gt_index,
                "pair_correct": trigger.pair == gt_pairs[gt_index],
                "start_error": trigger.representative_start - gt_starts[gt_index],
            }
        )
    correct_pairs = sum(bool(match["pair_correct"]) for match in matches)
    ordered = sorted(matches, key=lambda match: int(match["gt_index"]))
    exact_sequence = (
        len(triggers) == len(gt_starts)
        and len(matches) == len(gt_starts)
        and all(bool(match["pair_correct"]) for match in ordered)
    )
    return {
        "num_triggers": len(triggers),
        "num_gt_events": len(gt_starts),
        "matched_events": len(matches),
        "correct_pair_events": correct_pairs,
        "false_positive_or_duplicate_triggers": len(triggers) - len(matches),
        "missed_events": len(gt_starts) - len(matches),
        "event_precision": correct_pairs / max(len(triggers), 1),
        "event_recall": correct_pairs / max(len(gt_starts), 1),
        "exact_relation_sequence": exact_sequence,
        "matches": matches,
    }
