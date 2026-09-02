#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate and score independent facts from 36-frame real-cup contexts."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openpi.tasks.shellgame.real_cup_qwen3vl_sft_contract import SYSTEM_PROMPT
from openpi.tasks.shellgame.real_cup_qwen3vl_sft_contract import prompt_for_sample_type
from openpi.tasks.shellgame.real_cup_qwen3vl_sft_contract import validate_response

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_ADAPTER = ROOT / "checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375"
DEFAULT_MANIFEST = ROOT / "artifacts/real_cup_qwen3vl_full_context36_sft_v1_260826/val.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation/shellgame/real_cup_qwen3vl_full_context_v1/baseline.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _frames(row: dict[str, Any]) -> list[Image.Image]:
    with np.load(row["clip_path"], allow_pickle=False) as clip:
        frames = np.asarray(clip["frames"], dtype=np.uint8)
    if frames.shape != (36, 224, 224, 3):
        raise ValueError(f"Expected [36,224,224,3], got {frames.shape}")
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def _generate(processor: Any, model: Any, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    conversations, metadata = [], []
    for row in rows:
        frames = _frames(row)
        conversations.append(
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {
                            "type": "text",
                            "text": prompt_for_sample_type(str(row["sample_type"]), row.get("event_index")),
                        },
                    ],
                },
            ]
        )
        metadata.append(
            VideoMetadata(
                total_num_frames=36,
                fps=10.0,
                width=224,
                height=224,
                duration=3.6,
                frames_indices=list(range(36)),
            )
        )
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        num_frames=36,
        video_metadata=metadata,
    ).to(args.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
    trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _rollout(initial: str, moves: list[list[str]]) -> str:
    tracked = initial
    for left, right in moves:
        if tracked == left:
            tracked = right
        elif tracked == right:
            tracked = left
    return tracked


def summarize(records: list[dict[str, Any]], args: argparse.Namespace, wall_time: float) -> dict[str, Any]:
    typed = {name: [row for row in records if row["sample_type"] == name] for name in ("full_initial", "full_swap", "full_final")}
    correct = {name: sum(row["prediction"] == row["target_value"] for row in rows) for name, rows in typed.items()}
    swap_by_ordinal = {
        ordinal: [row for row in typed["full_swap"] if int(row["event_index"]) == ordinal]
        for ordinal in range(3)
    }
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        episodes[int(row["episode_id"])].append(row)
    episode_scores = []
    for episode, rows in episodes.items():
        initial = next(row for row in rows if row["sample_type"] == "full_initial")
        final = next(row for row in rows if row["sample_type"] == "full_final")
        swaps = sorted(
            (row for row in rows if row["sample_type"] == "full_swap"),
            key=lambda row: int(row["event_index"]),
        )
        all_valid = initial["valid"] and final["valid"] and len(swaps) == 3 and all(row["valid"] for row in swaps)
        swaps_exact = all(row["prediction"] == row["target_value"] for row in swaps)
        all_facts_exact = all_valid and initial["prediction"] == initial["target_value"] and swaps_exact and final["prediction"] == final["target_value"]
        recurrent_final = False
        if all_valid:
            recurrent_final = _rollout(
                initial["prediction"]["initial_cup"],
                [row["prediction"]["screen_pair"] for row in swaps],
            ) == final["target_value"]["final_cup"]
        episode_scores.append(
            {
                "episode_id": episode,
                "all_valid": all_valid,
                "all_swaps_exact": swaps_exact,
                "all_facts_exact": all_facts_exact,
                "recurrent_final_correct": recurrent_final,
            }
        )
    n_episodes = len(episode_scores)
    return {
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()),
        "manifest": str(args.manifest.resolve()),
        "samples": len(records),
        "episodes": n_episodes,
        "valid_json_rate": sum(row["valid"] for row in records) / max(len(records), 1),
        "initial_accuracy": correct["full_initial"] / max(len(typed["full_initial"]), 1),
        "swap_accuracy": correct["full_swap"] / max(len(typed["full_swap"]), 1),
        "swap_accuracy_by_ordinal": {
            str(ordinal + 1): sum(row["prediction"] == row["target_value"] for row in rows) / max(len(rows), 1)
            for ordinal, rows in swap_by_ordinal.items()
        },
        "final_accuracy": correct["full_final"] / max(len(typed["full_final"]), 1),
        "all_three_swaps_exact_accuracy": sum(row["all_swaps_exact"] for row in episode_scores) / max(n_episodes, 1),
        "recurrent_final_accuracy": sum(row["recurrent_final_correct"] for row in episode_scores) / max(n_episodes, 1),
        "all_five_facts_exact_accuracy": sum(row["all_facts_exact"] for row in episode_scores) / max(n_episodes, 1),
        "episode_all_valid_rate": sum(row["all_valid"] for row in episode_scores) / max(n_episodes, 1),
        "wall_time_seconds": wall_time,
        "per_episode": episode_scores,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    processor = AutoProcessor.from_pretrained(args.adapter_path, local_files_only=True, min_pixels=224**2, max_pixels=224**2)
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

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = _generate(processor, model, batch, args)
        for row, text in zip(batch, texts, strict=True):
            sample_type = str(row["sample_type"])
            target_value = validate_response(str(row["target"]), sample_type)
            try:
                prediction = validate_response(text, sample_type)
                valid, error = True, None
            except Exception as exc:
                prediction = None
                valid, error = False, f"{type(exc).__name__}: {exc}"
            records.append(
                {
                    "episode_id": int(row["episode_id"]),
                    "sample_type": sample_type,
                    "event_index": row.get("event_index"),
                    "target_value": target_value,
                    "prediction": prediction,
                    "raw_text": text,
                    "valid": valid,
                    "error": error,
                }
            )
        if start == 0 or start + len(batch) == len(rows) or (start + len(batch)) % 20 == 0:
            print(f"evaluated {start + len(batch)}/{len(rows)}", flush=True)
    summary = summarize(records, args, time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
