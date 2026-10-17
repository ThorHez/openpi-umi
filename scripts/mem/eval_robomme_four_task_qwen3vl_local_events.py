#!/usr/bin/env python3
"""Balanced greedy evaluation for the four-task RoboMME Qwen local-event pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable

import h5py
from peft import PeftModel
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.video_utils import VideoMetadata

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme.pickxtimes.qwen3vl_local_event_contract import (  # noqa: E402
    SYSTEM_PROMPT as PICK_SYSTEM_PROMPT,
    prompt_for_task as pick_prompt,
    validate_compact_response as validate_pick,
)
from openpi.tasks.robomme.videoplaceorder.qwen3vl_local_event_contract import (  # noqa: E402
    SYSTEM_PROMPT as ORDER_SYSTEM_PROMPT,
    prompt_for_local_event as order_prompt,
    validate_compact_response as validate_order,
)
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import (  # noqa: E402
    SYSTEM_PROMPT as UNMASK_SYSTEM_PROMPT,
    prompt_for_target as unmask_prompt,
    validate_compact_response as validate_unmask,
)
from openpi.tasks.robomme.videounmaskswap.qwen3vl_local_event_contract import (  # noqa: E402
    SYSTEM_PROMPT as SWAP_SYSTEM_PROMPT,
    prompt_for_local_event as swap_prompt,
    validate_compact_response as validate_swap,
)
from openpi.tasks.robomme.qwen3vl_unified_event_contract import (  # noqa: E402
    SYSTEM_PROMPT as UNIFIED_SYSTEM_PROMPT,
    prompt_for_goal as unified_prompt,
    validate_compact_response as validate_unified,
)

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_MANIFEST = _ROOT / "artifacts/robomme_four_task_qwen_mixture_seed260826/dev.jsonl"
SOURCES = (
    "videounmask_variable_demo",
    "videounmaskswap_local_event",
    "videoplaceorder_local_event",
    "pickxtimes_local_event",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _contract(row: dict[str, Any]) -> tuple[str, str, Callable[[str], dict[str, Any]]]:
    source = row["source"]
    if row.get("contract") == "unified_causal_event_v1":
        return (
            UNIFIED_SYSTEM_PROMPT,
            unified_prompt(
                str(row["goal"]),
                focus_entity=row.get("focus_entity"),
                candidate_region_count=row.get("candidate_region_count"),
            ),
            validate_unified,
        )
    if source == "videounmask_variable_demo":
        return UNMASK_SYSTEM_PROMPT, unmask_prompt(str(row["target_color"])), validate_unmask
    if source == "videounmaskswap_local_event":
        return SWAP_SYSTEM_PROMPT, swap_prompt(int(row["num_containers"])), validate_swap
    if source == "videoplaceorder_local_event":
        return ORDER_SYSTEM_PROMPT, order_prompt(), validate_order
    if source == "pickxtimes_local_event":
        return PICK_SYSTEM_PROMPT, pick_prompt(str(row["target_color"])), validate_pick
    raise ValueError(f"Unsupported source: {source}")


def _balanced_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin over event labels so frequent negatives cannot dominate."""
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(json.loads(row["target"])["event"])].append(row)
    for group in groups.values():
        rng.shuffle(group)
    events = sorted(groups)
    selected = []
    while len(selected) < min(count, len(rows)):
        progressed = False
        for event in events:
            if groups[event] and len(selected) < count:
                selected.append(groups[event].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def _install_non_tp_peft_load_compat() -> None:
    import peft.utils.save_and_load as peft_save_and_load
    import transformers.integrations.tensor_parallel as tensor_parallel

    if hasattr(tensor_parallel, "EmbeddingParallel"):
        return
    original = peft_save_and_load._maybe_shard_state_dict_for_tp

    def compatible(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], adapter_name: str) -> None:
        has_tp = any(
            getattr(module, "_hf_tp_plan", None) is not None
            and getattr(module, "_hf_device_mesh", None) is not None
            for module in model.modules()
        )
        if has_tp:
            original(model, state_dict, adapter_name)

    peft_save_and_load._maybe_shard_state_dict_for_tp = compatible


class FrameStore:
    def __init__(self) -> None:
        self.files: dict[str, h5py.File] = {}

    def frames(self, row: dict[str, Any]) -> list[Image.Image]:
        path = str(row["h5_path"])
        if path not in self.files:
            self.files[path] = h5py.File(path, "r")
        episode = self.files[path][str(row["episode_name"])]
        indices = [int(value) for value in row["frame_indices"]]
        if len(indices) != 12:
            raise ValueError(f"Expected 12 causal frames, got {indices}")
        if "demo_end" in row and max(indices) >= int(row["demo_end"]):
            raise ValueError(f"Execution leakage: {indices=}, demo_end={row['demo_end']}")
        return [
            Image.fromarray(episode[f"timestep_{index}/obs/front_rgb"][()]).convert("RGB")
            for index in indices
        ]

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()


def _inputs(
    processor: Any, store: FrameStore, rows: list[dict[str, Any]], device: str
) -> dict[str, torch.Tensor]:
    conversations, metadata = [], []
    for row in rows:
        system, prompt, _ = _contract(row)
        frames = store.frames(row)
        conversations.append(
            [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        )
        metadata.append(
            VideoMetadata(
                total_num_frames=12,
                fps=10.0,
                width=frames[0].width,
                height=frames[0].height,
                duration=1.2,
                frames_indices=list(range(12)),
            )
        )
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        num_frames=12,
        video_metadata=metadata,
    )
    return inputs.to(device)


def _rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(records)
    fields_correct = sum(record["fields_correct"] for record in records)
    fields_total = sum(record["fields_total"] for record in records)
    return {
        "samples": samples,
        "valid_rate": sum(record["valid"] for record in records) / max(samples, 1),
        "exact_accuracy": sum(record["exact"] for record in records) / max(samples, 1),
        "event_accuracy": sum(record["event_correct"] for record in records) / max(samples, 1),
        "field_accuracy": fields_correct / max(fields_total, 1),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = {}
    for source in SOURCES:
        subset = [record for record in records if record["source"] == source]
        events = sorted({record["expected"]["event"] for record in subset})
        by_source[source] = {
            **_rates(subset),
            "by_event": {
                event: _rates([record for record in subset if record["expected"]["event"] == event])
                for event in events
            },
        }
    return {"overall": _rates(records), "by_source": by_source}


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    all_rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for index, source in enumerate(SOURCES):
        candidates = [row for row in all_rows if row.get("source") == source]
        rows.extend(_balanced_rows(candidates, args.samples_per_source, args.seed + index))
    random.Random(args.seed).shuffle(rows)

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

    records = []
    store = FrameStore()
    started = time.perf_counter()
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
                expected = validator(str(row["target"]))
                try:
                    prediction = validator(text)
                    valid, error = True, None
                except Exception as exc:
                    prediction, valid, error = None, False, f"{type(exc).__name__}: {exc}"
                field_keys = set(expected) - {"event"}
                records.append(
                    {
                        "source": row["source"],
                        "episode_index": int(row["episode_index"]),
                        "sample_type": row["sample_type"],
                        "frame_indices": row["frame_indices"],
                        "expected": expected,
                        "prediction": prediction,
                        "raw_text": text,
                        "valid": valid,
                        "exact": prediction == expected,
                        "event_correct": prediction is not None
                        and prediction.get("event") == expected.get("event"),
                        "fields_correct": sum(
                            prediction is not None and prediction.get(key) == expected[key]
                            for key in field_keys
                        ),
                        "fields_total": len(field_keys),
                        "error": error,
                    }
                )
            print(f"evaluated {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)
    finally:
        store.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        **_summary(records),
        "adapter_path": str(args.adapter_path.resolve()),
        "manifest": str(args.manifest.resolve()),
        "selection": "event-balanced held-out episode clips",
        "samples_per_source": args.samples_per_source,
        "seed": args.seed,
        "wall_time_seconds": time.perf_counter() - started,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
