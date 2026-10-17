# ruff: noqa: E402
"""Greedy held-out evaluation for compact ShellGame Qwen3-VL event outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
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

from openpi.tasks.shellgame.qwen3vl_sft_contract import SYSTEM_PROMPT
from openpi.tasks.shellgame.qwen3vl_sft_contract import prompt_for_sample_type
from openpi.tasks.shellgame.qwen3vl_sft_contract import validate_compact_response
from openpi.tasks.shellgame.qwenvl_event_adapter import apply_swap
from openpi.tasks.shellgame.qwenvl_event_adapter import normalize_cup_entity

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_MANIFEST = _ROOT / "artifacts/shellgame_qwen3vl_gt_event_sft_v1/val.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _frames(row: dict[str, Any]) -> list[Image.Image]:
    indices = np.asarray(row["frame_indices"], dtype=np.int64)
    with np.load(row["trajectory_path"], allow_pickle=False) as trajectory:
        array = np.asarray(trajectory["third_person_images"][indices], dtype=np.uint8)
    return [Image.fromarray(frame).convert("RGB") for frame in array]


def _batch_inputs(processor: Any, rows: list[dict[str, Any]], device: str) -> dict[str, torch.Tensor]:
    conversations = []
    metadata = []
    for row in rows:
        frames = _frames(row)
        conversations.append(
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": prompt_for_sample_type(str(row["sample_type"]))},
                    ],
                },
            ]
        )
        metadata.append(
            VideoMetadata(
                total_num_frames=10,
                fps=10.0,
                width=224,
                height=224,
                duration=1.0,
                frames_indices=list(range(10)),
            )
        )
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        num_frames=10,
        video_metadata=metadata,
    )
    return inputs.to(device)


def _summary(records: list[dict[str, Any]], episode_ids: list[int]) -> dict[str, Any]:
    by_type = {}
    for sample_type in ("reveal", "swap", "no_event", "incomplete_event"):
        subset = [record for record in records if record["sample_type"] == sample_type]
        by_type[sample_type] = {
            "count": len(subset),
            "valid_rate": sum(bool(record["valid"]) for record in subset) / max(len(subset), 1),
            "accuracy": sum(bool(record["correct"]) for record in subset) / max(len(subset), 1),
            "prediction_counts": dict(sorted(Counter(str(record["prediction"]) for record in subset).items())),
        }
    sequences = []
    final_slots = []
    for episode in episode_ids:
        subset = [record for record in records if int(record["episode_index"]) == episode]
        reveal = [record for record in subset if record["sample_type"] == "reveal"]
        swaps = sorted(
            (record for record in subset if record["sample_type"] == "swap"),
            key=lambda record: int(record["event_index"]),
        )
        if len(reveal) != 1 or len(swaps) != 3:
            continue
        sequences.append(bool(reveal[0]["correct"] and all(record["correct"] for record in swaps)))
        try:
            predicted = normalize_cup_entity(reveal[0]["prediction"]["screen_cup"])
            expected = normalize_cup_entity(reveal[0]["expected"]["screen_cup"])
            for row in swaps:
                predicted_pair = tuple(normalize_cup_entity(value) for value in row["prediction"]["screen_pair"])
                expected_pair = tuple(normalize_cup_entity(value) for value in row["expected"]["screen_pair"])
                predicted = apply_swap(predicted, tuple(sorted(predicted_pair, key=("left", "middle", "right").index)))
                expected = apply_swap(expected, tuple(sorted(expected_pair, key=("left", "middle", "right").index)))
            final_slots.append(predicted == expected)
        except Exception:
            final_slots.append(False)
    return {
        "episodes": len(episode_ids),
        "samples": len(records),
        "overall_accuracy": sum(bool(record["correct"]) for record in records) / max(len(records), 1),
        "by_type": by_type,
        "complete_reveal_plus_three_swap_accuracy": sum(sequences) / max(len(sequences), 1),
        "symbolic_final_slot_accuracy": sum(final_slots) / max(len(final_slots), 1),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    all_rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    all_episode_ids = sorted({int(row["episode_index"]) for row in all_rows})
    episode_ids = all_episode_ids[args.episode_offset : args.episode_offset + args.num_episodes]
    if len(episode_ids) != args.num_episodes:
        raise ValueError(f"Selected only {len(episode_ids)} of {args.num_episodes} requested episodes")
    selected = set(episode_ids)
    rows = [row for row in all_rows if int(row["episode_index"]) in selected]

    processor_path = args.adapter_path if args.adapter_path is not None else args.model_path
    processor = AutoProcessor.from_pretrained(
        processor_path,
        local_files_only=True,
        min_pixels=224 * 224,
        max_pixels=224 * 224,
    )
    processor.video_processor.fps = None
    # Decoder-only batched generation requires left padding; training keeps the
    # processor default because labels explicitly mask padding positions.
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    )
    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    records = []
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        inputs = _batch_inputs(processor, batch_rows, args.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)]
        texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for row, text in zip(batch_rows, texts, strict=True):
            expected = validate_compact_response(str(row["target"]))
            try:
                prediction = validate_compact_response(text)
                valid = True
                error = None
            except Exception as exc:
                prediction = None
                valid = False
                error = f"{type(exc).__name__}: {exc}"
            records.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "sample_type": str(row["sample_type"]),
                    "event_index": row.get("event_index"),
                    "frame_indices": row["frame_indices"],
                    "expected": expected,
                    "prediction": prediction,
                    "raw_text": text,
                    "valid": valid,
                    "correct": prediction == expected,
                    "error": error,
                }
            )
        print(f"evaluated {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = _summary(records, episode_ids)
    summary.update(
        {
            "model_path": str(args.model_path.resolve()),
            "adapter_path": None if args.adapter_path is None else str(args.adapter_path.resolve()),
            "episode_ids": episode_ids,
            "wall_time_seconds": time.perf_counter() - started,
        }
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
