"""Build causal ShellGame event pseudo-labels with a frozen local Qwen3-VL.

Run this script in the isolated Qwen environment, not OpenPI's uv environment::

    CUDA_VISIBLE_DEVICES=0 /data1/conda_envs/qwen3vl_shellgame/bin/python \
      scripts/mem/build_shellgame_qwenvl_event_cache.py \
      --split semantic-val --num-episodes 10 --overwrite

Each episode is processed sequentially.  The reveal proposal is committed to
a deterministic ledger before the three swap windows are interpreted.  Only
past state and the current short clip are included in each prompt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.planning.qwenvl_event_schema import PlannerPatch
from openpi.planning.qwenvl_subgoal_planner import LocalQwen3VLPlanner
from openpi.planning.qwenvl_subgoal_planner import QwenVLPlannerConfig
from openpi.tasks.shellgame.qwenvl_event_adapter import SLOTS
from openpi.tasks.shellgame.qwenvl_event_adapter import SWAP_PAIRS
from openpi.tasks.shellgame.qwenvl_event_adapter import ShellGameTaskLedger
from openpi.tasks.shellgame.qwenvl_event_adapter import apply_swap
from openpi.tasks.shellgame.qwenvl_event_adapter import initial_slot_from_patch
from openpi.tasks.shellgame.qwenvl_event_adapter import swap_pair_from_patch


DEFAULT_RAW_ROOT = Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_absolute_eef_phase_instruction_dataset"
)
DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_OUTPUT = _ROOT / "evaluation/shellgame/qwenvl_event_cache/qwen3vl_4b_semantic_val.jsonl"


CAMERA_SLOT_CONTRACT = """Use only camera-relative visual entity names: screen_left_cup,
screen_middle_cup, and screen_right_cup, based on horizontal position in the first frame. Do not
convert them to robot/world coordinates; a deterministic downstream calibration does that."""


REVEAL_REQUEST = f"""Task: identify which cup will contain the visible yellow ball. One cup is
lifted so the ball is visible below it. Identify that cup. Do not predict later swaps or the final cup.
{CAMERA_SLOT_CONTRACT}

Return exactly one JSON object with this shape, replacing SELECTED_CUP with exactly one of
screen_left_cup, screen_middle_cup, screen_right_cup and filling CONFIDENCE/EVIDENCE:
{{
  "request_id": "COPY_REQUEST_ID_EXACTLY",
  "event": {{
    "event_id": "COPY_EVENT_ID_EXACTLY",
    "type": "object_hidden_by_container",
    "entities": ["ball", "SELECTED_CUP"],
    "state_delta": {{"operation":"set_relation", "subject":"ball",
                    "predicate":"contained_by", "object":"SELECTED_CUP"}},
    "confidence": CONFIDENCE,
    "evidence": ["EVIDENCE"]
  }},
  "subgoal_updates": [],
  "next_subgoal": null,
  "decision": "propose_update",
  "request_reobservation": false
}}
Output JSON only."""


SWAP_REQUEST = f"""Task: determine which two of three cups exchange spatial slots during this
short video. Follow cup trajectories from the first frame to the last. Exactly one pair exchanges;
the third cup is not part of the exchange. Report one pair only. Do not report the hidden ball's
location and do not infer a final target. {CAMERA_SLOT_CONTRACT}

Return exactly one JSON object with this shape. Replace CUP_A and CUP_B with two distinct values
from screen_left_cup, screen_middle_cup, screen_right_cup. Fill CONFIDENCE/EVIDENCE:
{{
  "request_id": "COPY_REQUEST_ID_EXACTLY",
  "event": {{
    "event_id": "COPY_EVENT_ID_EXACTLY",
    "type": "container_exchange",
    "entities": ["CUP_A", "CUP_B"],
    "state_delta": {{"operation":"exchange_entity_states", "subjects":["CUP_A", "CUP_B"]}},
    "confidence": CONFIDENCE,
    "evidence": ["EVIDENCE"]
  }},
  "subgoal_updates": [],
  "next_subgoal": null,
  "decision": "propose_update",
  "request_reobservation": false
}}
Output JSON only."""


NO_EVENT_REQUEST = """Task: determine whether these chronological frames contain a completed
state-changing interaction among the three cups. The cups remain stationary in this negative
window. Do not turn ordinary visual persistence into an event.

Return exactly this JSON object after copying the request/event ids and filling CONFIDENCE:
{
  "request_id": "COPY_REQUEST_ID_EXACTLY",
  "event": {
    "event_id": "COPY_EVENT_ID_EXACTLY",
    "type": "no_state_change",
    "entities": [],
    "state_delta": {"operation":"no_state_change"},
    "confidence": CONFIDENCE,
    "evidence": ["cups remain stationary"]
  },
  "subgoal_updates": [],
  "next_subgoal": null,
  "decision": "keep_state",
  "request_reobservation": false
}
Output JSON only."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("head", "semantic-val", "explicit"), default="semantic-val")
    parser.add_argument("--episode-list", type=Path, help="Newline-delimited ids for --split explicit")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--include-negative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection/prompts without loading Qwen")
    return parser.parse_args()


def _available_episode_ids(root: Path) -> list[int]:
    result = []
    for path in root.glob("episode_*"):
        try:
            episode = int(path.name.removeprefix("episode_"))
        except ValueError:
            continue
        if (path / "metadata.json").is_file() and (path / "vla_trajectory.npz").is_file():
            result.append(episode)
    if not result:
        raise FileNotFoundError(f"No complete ShellGame episodes found under {root}")
    return sorted(result)


def _select_episode_ids(args: argparse.Namespace) -> list[int]:
    available = _available_episode_ids(args.raw_root)
    if args.split == "head":
        ordered = available
    elif args.split == "semantic-val":
        shuffled = np.random.default_rng(args.split_seed).permutation(np.asarray(available, dtype=np.int64))
        num_val = min(max(1, round(len(available) * args.val_ratio)), len(available) - 1)
        ordered = sorted(int(value) for value in shuffled[:num_val])
    else:
        if args.episode_list is None:
            raise ValueError("--split explicit requires --episode-list")
        ordered = [int(line.strip()) for line in args.episode_list.read_text().splitlines() if line.strip()]
        missing = sorted(set(ordered) - set(available))
        if missing:
            raise ValueError(f"Requested episodes are missing from raw root: {missing[:10]}")
    if args.episode_offset < 0 or args.num_episodes < 1:
        raise ValueError("episode-offset must be nonnegative and num-episodes must be positive")
    selected = ordered[args.episode_offset : args.episode_offset + args.num_episodes]
    if len(selected) != args.num_episodes:
        raise ValueError(f"Requested {args.num_episodes} episodes but selected only {len(selected)}")
    return selected


def _canonical_pair(raw_pair: Any) -> tuple[str, str]:
    if not isinstance(raw_pair, list) or len(raw_pair) != 2:
        raise ValueError(f"Invalid metadata swap: {raw_pair!r}")
    pair = tuple(sorted((str(raw_pair[0]), str(raw_pair[1])), key=SLOTS.index))
    if pair not in SWAP_PAIRS:
        raise ValueError(f"Invalid metadata swap: {raw_pair!r}")
    return pair


def _frame_indices(query_key: str) -> list[int]:
    if query_key == "reveal":
        return (0, 2, 4, 6, 8, 9)
    if query_key.startswith("swap_"):
        stage = int(query_key.split("_")[1])
        start = 20 + 10 * stage
        return list(range(start, start + 10))
    if query_key == "negative_settle":
        return (50, 51, 52, 53, 54, 55)
    raise ValueError(f"Unknown query key: {query_key}")


def _request_for(query_key: str) -> str:
    if query_key == "reveal":
        return REVEAL_REQUEST
    if query_key.startswith("swap_"):
        return SWAP_REQUEST
    if query_key == "negative_settle":
        return NO_EVENT_REQUEST
    raise ValueError(query_key)


def _validate_adapter(query_key: str, patch: PlannerPatch) -> Any:
    if query_key == "reveal":
        return initial_slot_from_patch(patch)
    if query_key.startswith("swap_"):
        return list(swap_pair_from_patch(patch))
    if patch.event.state_delta.operation != "no_state_change" or patch.decision != "keep_state":
        raise ValueError("Negative window must produce no_state_change + keep_state")
    return "no_state_change"


def _expected_for(query_key: str, metadata: dict[str, Any]) -> Any:
    if query_key == "reveal":
        return str(metadata["initial_ball_cup"])
    if query_key.startswith("swap_"):
        return list(_canonical_pair(metadata["swaps"][int(query_key.split("_")[1])]))
    return "no_state_change"


def _load_completed(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            result.add((int(record["episode_index"]), str(record["query_key"])))
    return result


def _write_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _summarize(output: Path, selected: list[int], elapsed: float) -> dict[str, Any]:
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [record for record in records if int(record["episode_index"]) in set(selected)]
    by_query: dict[str, dict[str, Any]] = {}
    for query in ("reveal", "swap_0", "swap_1", "swap_2", "negative_settle"):
        subset = [record for record in records if record["query_key"] == query]
        if not subset:
            continue
        by_query[query] = {
            "count": len(subset),
            "schema_valid_rate": sum(bool(record["schema_valid"]) for record in subset) / len(subset),
            "adapter_valid_rate": sum(bool(record["adapter_valid"]) for record in subset) / len(subset),
            "accuracy": sum(bool(record["correct"]) for record in subset) / len(subset),
            "mean_latency_seconds": sum(float(record["latency_seconds"]) for record in subset) / len(subset),
            "prediction_counts": dict(Counter(str(record.get("prediction")) for record in subset)),
        }
    complete = []
    final_conditions = {
        "gt_initial_gt_swaps": [],
        "gt_initial_qwen_swaps": [],
        "qwen_initial_gt_swaps": [],
        "qwen_initial_qwen_swaps": [],
    }
    for episode in selected:
        subset = {record["query_key"]: record for record in records if int(record["episode_index"]) == episode}
        required = ("reveal", "swap_0", "swap_1", "swap_2")
        complete.append(all(key in subset and subset[key]["correct"] for key in required))
        if not all(key in subset for key in required):
            for values in final_conditions.values():
                values.append(False)
            continue
        gt_initial = str(subset["reveal"]["expected"])
        qwen_initial = subset["reveal"].get("prediction")
        gt_swaps = [tuple(subset[f"swap_{stage}"]["expected"]) for stage in range(3)]
        qwen_swaps = [subset[f"swap_{stage}"].get("prediction") for stage in range(3)]
        gt_final = gt_initial
        for pair in gt_swaps:
            gt_final = apply_swap(gt_final, pair)

        def predicted_final(initial, pairs):
            if initial not in SLOTS or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs):
                return None
            slot = initial
            for pair in pairs:
                slot = apply_swap(slot, tuple(pair))
            return slot

        final_conditions["gt_initial_gt_swaps"].append(
            predicted_final(gt_initial, [list(pair) for pair in gt_swaps]) == gt_final
        )
        final_conditions["gt_initial_qwen_swaps"].append(predicted_final(gt_initial, qwen_swaps) == gt_final)
        final_conditions["qwen_initial_gt_swaps"].append(
            predicted_final(qwen_initial, [list(pair) for pair in gt_swaps]) == gt_final
        )
        final_conditions["qwen_initial_qwen_swaps"].append(predicted_final(qwen_initial, qwen_swaps) == gt_final)
    summary = {
        "episodes": len(selected),
        "episode_ids": selected,
        "complete_event_sequence_accuracy": sum(complete) / len(complete),
        "final_slot_accuracy": {
            key: sum(values) / len(values) for key, values in final_conditions.items()
        },
        "by_query": by_query,
        "wall_time_seconds": elapsed,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    selected = _select_episode_ids(args)
    query_keys = ["reveal", "swap_0", "swap_1", "swap_2"]
    if args.include_negative:
        query_keys.append("negative_settle")
    print(f"selected episodes ({len(selected)}): {selected}")
    print(f"queries per episode: {query_keys}")
    if args.dry_run:
        return
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed = _load_completed(args.output)
    planner = LocalQwen3VLPlanner(
        QwenVLPlannerConfig(
            model_path=str(args.model_path),
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )
    )
    started = time.perf_counter()
    with args.output.open("a", encoding="utf-8") as output_handle:
        for episode_order, episode in enumerate(selected):
            episode_dir = args.raw_root / f"episode_{episode:06d}"
            metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
            trajectory = np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False)
            frames = trajectory["third_person_images"]
            ledger = ShellGameTaskLedger(task_id=f"shellgame-{episode:06d}")
            for query_key in query_keys:
                key = (episode, query_key)
                if key in completed:
                    continue
                indices = _frame_indices(query_key)
                images = [Image.fromarray(frames[index]).convert("RGB") for index in indices]
                request_id = f"ep{episode:06d}-{query_key}"
                event_id = request_id
                request = _request_for(query_key).replace("COPY_REQUEST_ID_EXACTLY", request_id).replace(
                    "COPY_EVENT_ID_EXACTLY", event_id
                )
                memory_before = ledger.snapshot()
                attempts = []
                patch = None
                prediction = None
                schema_valid = adapter_valid = False
                error = None
                total_latency = 0.0
                for attempt_index in range(args.max_retries + 1):
                    retry_request = request
                    if error is not None:
                        retry_request += (
                            "\nYour previous response was rejected by the deterministic validator: "
                            f"{error}. Return a corrected JSON object."
                        )
                    try:
                        generation = planner.generate(
                            images,
                            request=retry_request,
                            previous_task_memory=memory_before,
                            request_id=request_id,
                        )
                        total_latency += generation.latency_seconds
                        patch = generation.patch
                        schema_valid = True
                        prediction = _validate_adapter(query_key, patch)
                        adapter_valid = True
                        error = None
                        attempts.append({"raw_text": generation.raw_text, "error": None})
                        break
                    except Exception as exc:  # Preserve invalid generations for audit and retry.
                        error = f"{type(exc).__name__}: {exc}"
                        raw_text = getattr(exc, "raw_text", None)
                        total_latency += float(getattr(exc, "latency_seconds", 0.0))
                        attempts.append({"raw_text": raw_text, "error": error})
                expected = _expected_for(query_key, metadata)
                correct = adapter_valid and prediction == expected
                committed = False
                if adapter_valid and patch is not None:
                    if query_key == "reveal":
                        committed = ledger.commit_reveal(patch)
                    elif query_key.startswith("swap_") and ledger.target_slot is not None:
                        committed = ledger.commit_swap(patch)
                if not adapter_valid:
                    ledger.uncertain = True
                record = {
                    "schema_version": 1,
                    "prompt_version": "shellgame_screen_slots_v1",
                    "model_path": str(args.model_path.resolve()),
                    "episode_index": episode,
                    "episode_order": episode_order,
                    "query_key": query_key,
                    "window_start": min(indices),
                    "window_end": max(indices),
                    "frame_indices": indices,
                    "request_id": request_id,
                    "event_id": event_id,
                    "task_memory_before": memory_before,
                    "task_memory_after": ledger.snapshot(),
                    "patch": None if patch is None else patch.to_dict(),
                    "prediction": prediction,
                    "expected": expected,
                    "schema_valid": schema_valid,
                    "adapter_valid": adapter_valid,
                    "correct": bool(correct),
                    "committed": committed,
                    "latency_seconds": total_latency,
                    "attempts": attempts,
                    "error": error,
                }
                _write_record(output_handle, record)
                print(
                    f"[{episode_order + 1}/{len(selected)}] ep={episode:06d} {query_key}: "
                    f"pred={prediction!r} gt={expected!r} valid={adapter_valid} correct={correct} "
                    f"latency={total_latency:.2f}s"
                )
            trajectory.close()
    summary = _summarize(args.output, selected, time.perf_counter() - started)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
