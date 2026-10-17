#!/usr/bin/env python3
"""Episode-level recurrent-state rollout from unified Qwen local events."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import is_dataclass
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

from peft import PeftModel
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme.four_task_state import OrderedTargetState  # noqa: E402
from openpi.tasks.robomme.four_task_state import PickCountState  # noqa: E402
from openpi.tasks.robomme.four_task_state import TargetIdentityState  # noqa: E402
from scripts.mem.eval_robomme_four_task_qwen3vl_local_events import FrameStore  # noqa: E402
from scripts.mem.eval_robomme_four_task_qwen3vl_local_events import _contract  # noqa: E402
from scripts.mem.eval_robomme_four_task_qwen3vl_local_events import _inputs  # noqa: E402
from scripts.mem.eval_robomme_four_task_qwen3vl_local_events import (  # noqa: E402
    _install_non_tp_peft_load_compat,
)

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_MANIFEST = (
    _ROOT / "artifacts/robomme_four_task_qwen_unified_optimized_v2_mixture_seed260826/test.jsonl"
)
SOURCES = (
    "videounmask_variable_demo",
    "videounmaskswap_local_event",
    "videoplaceorder_local_event",
    "pickxtimes_local_event",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One causal positive per event plus its preceding incomplete/hold windows."""
    selected = []
    for row in rows:
        source = str(row.get("source"))
        if source not in SOURCES or row.get("contract") != "unified_causal_event_v1":
            continue
        sample_type = str(row["sample_type"])
        variant = int(row.get("variant", 0))
        if source == "videounmask_variable_demo":
            keep = variant == 0 and row.get("goal_source") == "original_instruction"
        elif source == "videounmaskswap_local_event":
            if sample_type in ("target_visible", "target_covered"):
                focus_color = str(row.get("focus_entity", "")).removesuffix("_cube")
                keep = variant == 0 and focus_color in row["target_colors"]
            else:
                keep = variant == 0
        elif source == "videoplaceorder_local_event":
            keep = variant == 0
        else:
            keep = sample_type == "no_completed_event" or variant == 0
        if keep:
            selected.append(row)
    return selected


def _temporal_key(row: dict[str, Any]) -> tuple[int, int, str, int]:
    frames = [int(value) for value in row["frame_indices"]]
    return max(frames), min(frames), str(row["sample_type"]), int(row.get("event_index", -1))


def _initial_state(source: str, row: dict[str, Any]) -> Any:
    if source == "videounmask_variable_demo":
        return TargetIdentityState.empty((str(row["target_color"]),))
    if source == "videounmaskswap_local_event":
        return TargetIdentityState.empty(tuple(str(value) for value in row["target_colors"]))
    if source == "videoplaceorder_local_event":
        return OrderedTargetState(str(row["target_color"]), int(row["queried_ordinal"]))
    if source == "pickxtimes_local_event":
        return PickCountState(str(row["target_color"]), int(row["required_count"]))
    raise ValueError(f"Unsupported source: {source}")


def _snapshot(state: Any) -> dict[str, Any]:
    if not is_dataclass(state):
        raise TypeError(type(state))
    value = asdict(state)
    return {key: list(item) if isinstance(item, tuple) else item for key, item in value.items()}


def _signature(event: dict[str, Any]) -> tuple[Any, ...]:
    return event["event"], event["entity"], event["region_a"], event["region_b"]


def _is_state_changing(event: dict[str, Any]) -> bool:
    return event["event"] in {
        "target_visible",
        "target_covered",
        "swap_complete",
        "pick_complete",
        "place_complete",
        "press_complete",
    }


def _apply(source: str, state: Any, event: dict[str, Any] | None) -> tuple[Any, bool, bool]:
    """Return state, committed, rejected-by-gating."""
    if event is None or not _is_state_changing(event):
        return state, False, False
    event_type = str(event["event"])
    try:
        if source in ("videounmask_variable_demo", "videounmaskswap_local_event"):
            if event_type in ("target_visible", "target_covered"):
                entity = event.get("entity")
                region = event.get("region_a")
                if not isinstance(entity, str) or not entity.endswith("_cube") or region is None:
                    return state, False, True
                color = entity.removesuffix("_cube")
                if color not in state.target_colors:
                    return state, False, True
                return state.observe_target(color, str(region), covered=event_type == "target_covered"), True, False
            if event_type == "swap_complete" and event.get("region_a") and event.get("region_b"):
                return state.apply_swap(str(event["region_a"]), str(event["region_b"])), True, False
            return state, False, True
        if source == "videoplaceorder_local_event":
            if event_type == "place_complete" and event.get("region_a"):
                ordinal = state.written_count + 1
                if ordinal > 4:
                    return state, False, True
                return state.place_complete(ordinal, str(event["region_a"])), True, False
            if event_type == "swap_complete" and event.get("region_a") and event.get("region_b"):
                return state.swap_complete(str(event["region_a"]), str(event["region_b"])), True, False
            return state, False, True
        if source == "pickxtimes_local_event":
            if event_type in ("pick_complete", "place_complete", "press_complete"):
                return state.apply(event_type), True, False
            return state, False, True
    except ValueError:
        return state, False, True
    raise ValueError(source)


def _final_answer(source: str, state: Any) -> Any:
    if source in ("videounmask_variable_demo", "videounmaskswap_local_event"):
        return list(state.target_cells)
    if source == "videoplaceorder_local_event":
        return state.queried_cell
    if source == "pickxtimes_local_event":
        return {
            "completed_count": state.completed_count,
            "holding": state.holding,
            "done": state.done,
        }
    raise ValueError(source)


def _rollout(
    source: str,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any] | None],
    *,
    deduplicate: bool,
) -> dict[str, Any]:
    state = _initial_state(source, rows[0])
    recent_end_by_signature: dict[tuple[Any, ...], int] = {}
    states, committed, rejected, suppressed = [], 0, 0, 0
    for row, event in zip(rows, events, strict=True):
        clip_start = min(int(value) for value in row["frame_indices"])
        clip_end = max(int(value) for value in row["frame_indices"])
        if deduplicate and event is not None and _is_state_changing(event):
            signature = _signature(event)
            if clip_start <= recent_end_by_signature.get(signature, -1):
                suppressed += 1
                states.append(_snapshot(state))
                continue
        next_state, did_commit, was_rejected = _apply(source, state, event)
        state = next_state
        committed += int(did_commit)
        rejected += int(was_rejected)
        if deduplicate and did_commit and event is not None:
            recent_end_by_signature[_signature(event)] = clip_end
        states.append(_snapshot(state))
    return {
        "states": states,
        "final_state": _snapshot(state),
        "final_answer": _final_answer(source, state),
        "committed": committed,
        "rejected": rejected,
        "suppressed": suppressed,
    }


def _aggregate(episodes: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    result = {}
    for source in SOURCES:
        subset = [episode for episode in episodes if episode["source"] == source]
        steps = sum(len(episode["rows"]) for episode in subset)
        state_matches = sum(
            predicted == expected
            for episode in subset
            for predicted, expected in zip(
                episode[mode]["states"], episode["oracle"]["states"], strict=True
            )
        )
        clips = [clip for episode in subset for clip in episode["clips"]]
        result[source] = {
            "episodes": len(subset),
            "clips": len(clips),
            "qwen_valid_rate": sum(clip["valid"] for clip in clips) / max(len(clips), 1),
            "qwen_event_accuracy": sum(clip["event_correct"] for clip in clips) / max(len(clips), 1),
            "full_state_sequence_accuracy": state_matches / max(steps, 1),
            "final_state_accuracy": sum(
                episode[mode]["final_state"] == episode["oracle"]["final_state"]
                for episode in subset
            )
            / max(len(subset), 1),
            "final_answer_accuracy": sum(
                episode[mode]["final_answer"] == episode["oracle"]["final_answer"]
                for episode in subset
            )
            / max(len(subset), 1),
            "committed": sum(episode[mode]["committed"] for episode in subset),
            "rejected": sum(episode[mode]["rejected"] for episode in subset),
            "suppressed": sum(episode[mode]["suppressed"] for episode in subset),
        }
    all_episodes = episodes
    total_steps = sum(len(episode["rows"]) for episode in all_episodes)
    result["overall_macro"] = {
        key: sum(result[source][key] for source in SOURCES) / len(SOURCES)
        for key in (
            "qwen_valid_rate",
            "qwen_event_accuracy",
            "full_state_sequence_accuracy",
            "final_state_accuracy",
            "final_answer_accuracy",
        )
    }
    result["overall_micro_state_sequence_accuracy"] = sum(
        predicted == expected
        for episode in all_episodes
        for predicted, expected in zip(episode[mode]["states"], episode["oracle"]["states"], strict=True)
    ) / max(total_steps, 1)
    return result


def _generate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    if args.adapter_path is None:
        raise ValueError("--adapter-path is required unless --oracle-only is used")
    processor = AutoProcessor.from_pretrained(
        args.adapter_path, local_files_only=True, min_pixels=224**2, max_pixels=224**2
    )
    processor.video_processor.fps = None
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    )
    _install_non_tp_peft_load_compat()
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    predictions: list[dict[str, Any] | None] = []
    store = FrameStore()
    try:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            inputs = _inputs(processor, store, batch_rows, args.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            trimmed = [
                output[len(source) :]
                for source, output in zip(inputs.input_ids, generated, strict=True)
            ]
            texts = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for row, text in zip(batch_rows, texts, strict=True):
                validator = _contract(row)[2]
                try:
                    prediction = validator(text)
                except Exception:
                    prediction = None
                predictions.append(prediction)
                row["raw_text"] = text
            print(f"generated {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)
    finally:
        store.close()
    return predictions


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    all_rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = canonical_rows(all_rows)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in selected:
        groups.setdefault((str(row["source"]), int(row["episode_index"])), []).append(row)
    for rows in groups.values():
        rows.sort(key=_temporal_key)
    ordered_rows = [row for key in sorted(groups) for row in groups[key]]
    expected = [_contract(row)[2](str(row["target"])) for row in ordered_rows]
    predictions = expected if args.oracle_only else _generate(args, ordered_rows)
    prediction_by_id = {id(row): prediction for row, prediction in zip(ordered_rows, predictions, strict=True)}

    episodes = []
    for (source, episode_index), rows in sorted(groups.items()):
        oracle_events = [_contract(row)[2](str(row["target"])) for row in rows]
        predicted_events = [prediction_by_id[id(row)] for row in rows]
        oracle = _rollout(source, rows, oracle_events, deduplicate=False)
        raw = _rollout(source, rows, predicted_events, deduplicate=False)
        dedup = _rollout(source, rows, predicted_events, deduplicate=True)
        clips = []
        for row, target, prediction in zip(rows, oracle_events, predicted_events, strict=True):
            clips.append(
                {
                    "sample_type": row["sample_type"],
                    "event_index": int(row.get("event_index", -1)),
                    "frame_indices": row["frame_indices"],
                    "expected": target,
                    "prediction": prediction,
                    "raw_text": row.get("raw_text"),
                    "valid": prediction is not None,
                    "event_correct": prediction is not None and prediction["event"] == target["event"],
                }
            )
        episodes.append(
            {
                "source": source,
                "episode_index": episode_index,
                "rows": [{"sample_type": row["sample_type"], "frame_indices": row["frame_indices"]} for row in rows],
                "clips": clips,
                "oracle": oracle,
                "gated_raw": raw,
                "gated_dedup": dedup,
            }
        )

    # Clean metadata must produce complete states before Qwen is assessed.
    for episode in episodes:
        source = episode["source"]
        answer = episode["oracle"]["final_answer"]
        if source in ("videounmask_variable_demo", "videounmaskswap_local_event"):
            assert all(value is not None for value in answer)
        elif source == "videoplaceorder_local_event":
            assert answer is not None
        else:
            assert answer["done"] and answer["completed_count"] > 0 and not answer["holding"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "adapter_path": None if args.adapter_path is None else str(args.adapter_path.resolve()),
        "oracle_only": args.oracle_only,
        "canonical_clips": len(ordered_rows),
        "episodes": len(episodes),
        "overlap_deduplication": "suppress identical state-changing signatures whose causal clips overlap",
        "gated_raw": _aggregate(episodes, "gated_raw"),
        "gated_dedup": _aggregate(episodes, "gated_dedup"),
    }
    path = args.output.with_suffix(".summary.json")
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
