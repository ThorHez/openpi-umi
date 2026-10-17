#!/usr/bin/env python3
"""Greedy held-out progress-memory evaluation for SwingXtimes Qwen3-VL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
from peft import PeftModel
from PIL import Image
import torch
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from transformers.video_utils import VideoMetadata

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme.swingxtimes.qwen3vl_sft_contract import SYSTEM_PROMPT  # noqa: E402
from openpi.tasks.robomme.swingxtimes.qwen3vl_sft_contract import prompt_for_task  # noqa: E402
from openpi.tasks.robomme.swingxtimes.qwen3vl_sft_contract import validate_compact_response  # noqa: E402

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_MANIFEST = _ROOT / "artifacts/swingxtimes_qwen3vl_sft_seed260825/val.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _frames(h5: h5py.File, row: dict[str, Any]) -> list[Image.Image]:
    episode = h5[str(row["episode_name"])]
    return [
        Image.fromarray(episode[f"timestep_{int(index)}/obs/front_rgb"][()]).convert("RGB")
        for index in row["frame_indices"]
    ]


def _inputs(
    processor: Any, h5: h5py.File, rows: list[dict[str, Any]], device: str
) -> dict[str, torch.Tensor]:
    conversations, metadata = [], []
    for row in rows:
        frames = _frames(h5, row)
        conversations.append(
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {
                            "type": "text",
                            "text": prompt_for_task(
                                str(row["target_color"]), int(row["target_round_trips"])
                            ),
                        },
                    ],
                },
            ]
        )
        metadata.append(
            VideoMetadata(
                total_num_frames=12,
                fps=10.0,
                width=256,
                height=256,
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


def _rate(records: list[dict[str, Any]], key: str) -> float:
    return sum(bool(record[key]) for record in records) / max(len(records), 1)


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = {}
    for sample_type in ("causal_prefix", "no_event", "local_only"):
        subset = [record for record in records if record["sample_type"] == sample_type]
        by_type[sample_type] = {
            "samples": len(subset),
            "valid_rate": _rate(subset, "valid"),
            "exact_accuracy": _rate(subset, "correct"),
            "event_accuracy": _rate(subset, "event_correct"),
            "count_accuracy": _rate(subset, "count_correct")
            if sample_type == "causal_prefix"
            else None,
            "prediction_counts": dict(
                sorted(Counter(str(record["prediction"]) for record in subset).items())
            ),
        }
    progress = [record for record in records if record["sample_type"] == "causal_prefix"]
    by_target = {}
    for target in (1, 2, 3):
        subset = [record for record in progress if record["target_round_trips"] == target]
        by_target[str(target)] = {
            "samples": len(subset),
            "exact_accuracy": _rate(subset, "correct"),
            "count_accuracy": _rate(subset, "count_correct"),
        }
    episodes = sorted({int(record["episode_index"]) for record in records})
    sequence_correct, final_correct = [], []
    for episode in episodes:
        subset = [
            record
            for record in progress
            if record["episode_index"] == episode and record["variant"] == 0
        ]
        sequence_correct.append(bool(subset) and all(record["correct"] for record in subset))
        if subset:
            final = max(subset, key=lambda record: int(record["event_index"]))
            final_correct.append(
                bool(final["count_correct"])
                and final["prediction"] is not None
                and final["prediction"].get("ready_to_stop") is True
            )
    return {
        "episodes": len(episodes),
        "samples": len(records),
        "overall_valid_rate": _rate(records, "valid"),
        "overall_exact_accuracy": _rate(records, "correct"),
        "by_type": by_type,
        "progress_by_target_round_trips": by_target,
        "full_progress_sequence_accuracy": sum(sequence_correct) / max(len(sequence_correct), 1),
        "final_count_and_ready_accuracy": sum(final_correct) / max(len(final_correct), 1),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not args.all_variants:
        rows = [row for row in rows if int(row["variant"]) == 0]
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
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    h5_paths = {str(row["h5_path"]) for row in rows}
    if len(h5_paths) != 1:
        raise ValueError(f"Expected one HDF5 source, got {h5_paths}")
    records = []
    started = time.perf_counter()
    with h5py.File(next(iter(h5_paths)), "r") as h5:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            inputs = _inputs(processor, h5, batch_rows, args.device)
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
                target_round_trips = int(row["target_round_trips"])
                expected = validate_compact_response(
                    str(row["target"]), target_round_trips=target_round_trips
                )
                try:
                    prediction = validate_compact_response(
                        text, target_round_trips=target_round_trips
                    )
                    valid, error = True, None
                except Exception as exc:
                    prediction, valid, error = None, False, f"{type(exc).__name__}: {exc}"
                records.append(
                    {
                        "episode_index": int(row["episode_index"]),
                        "sample_type": str(row["sample_type"]),
                        "event_index": int(row["event_index"]),
                        "variant": int(row["variant"]),
                        "target_color": str(row["target_color"]),
                        "target_round_trips": target_round_trips,
                        "frame_indices": row["frame_indices"],
                        "expected": expected,
                        "prediction": prediction,
                        "raw_text": text,
                        "valid": valid,
                        "correct": prediction == expected,
                        "event_correct": prediction is not None
                        and prediction.get("event") == expected.get("event"),
                        "count_correct": prediction is not None
                        and prediction.get("completed_round_trips")
                        == expected.get("completed_round_trips"),
                        "error": error,
                    }
                )
            print(f"evaluated {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = _summary(records)
    summary.update(
        {
            "model_path": str(args.model_path.resolve()),
            "adapter_path": str(args.adapter_path.resolve()),
            "manifest": str(args.manifest.resolve()),
            "all_variants": args.all_variants,
            "wall_time_seconds": time.perf_counter() - started,
        }
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
