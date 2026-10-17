#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate Qwen3-VL on episode-held-out labeled real-cup clips."""

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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openpi.tasks.shellgame.real_cup_qwen3vl_sft_contract import SYSTEM_PROMPT
from openpi.tasks.shellgame.real_cup_qwen3vl_sft_contract import prompt_for_sample_type
from openpi.tasks.shellgame.real_cup_qwen3vl_sft_contract import validate_response

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_ADAPTER = ROOT / "checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375"
DEFAULT_MANIFEST = ROOT / "artifacts/real_cup_qwen3vl_gt_sft_v1_260826/val.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation/shellgame/real_cup_qwen3vl_gt_lora_v1/baseline.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--reuse-records", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _frames(row: dict[str, Any]) -> list[Image.Image]:
    with np.load(row["clip_path"], allow_pickle=False) as clip:
        frames = np.asarray(clip["frames"], dtype=np.uint8)
    if frames.shape != (12, 224, 224, 3):
        raise ValueError(f"Expected [12,224,224,3], got {frames.shape} from {row['clip_path']}")
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
                        {"type": "text", "text": prompt_for_sample_type(str(row["sample_type"]))},
                    ],
                },
            ]
        )
        metadata.append(
            VideoMetadata(
                total_num_frames=12,
                fps=10.0,
                width=224,
                height=224,
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
    ).to(args.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
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


def _summary(records: list[dict[str, Any]], args: argparse.Namespace, wall_time: float) -> dict[str, Any]:
    valid = [row for row in records if row["valid"]]
    local = [row for row in records if row["sample_type"] == "local_swap"]
    sequence = [row for row in records if row["sample_type"] == "sequence"]
    local_correct = sum(row["prediction"] == row["target_value"] for row in local)
    sequence_valid = [row for row in sequence if row["valid"]]
    initial_correct = sum(row["prediction"]["initial_cup"] == row["target_value"]["initial_cup"] for row in sequence_valid)
    final_correct = sum(row["prediction"]["final_cup"] == row["target_value"]["final_cup"] for row in sequence_valid)
    moves_exact = sum(row["prediction"]["moves"] == row["target_value"]["moves"] for row in sequence_valid)
    sequence_exact = sum(row["prediction"] == row["target_value"] for row in sequence_valid)
    move_slots_correct = sum(
        predicted == target
        for row in sequence_valid
        for predicted, target in zip(row["prediction"]["moves"], row["target_value"]["moves"], strict=True)
    )
    rollout_correct = sum(
        _rollout(row["prediction"]["initial_cup"], row["prediction"]["moves"])
        == row["target_value"]["final_cup"]
        for row in sequence_valid
    )
    internally_consistent = sum(
        _rollout(row["prediction"]["initial_cup"], row["prediction"]["moves"])
        == row["prediction"]["final_cup"]
        for row in sequence_valid
    )
    local_by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in local:
        local_by_episode.setdefault(int(row["episode_id"]), []).append(row)
    recurrent_rows = []
    for sequence_row in sequence:
        episode = int(sequence_row["episode_id"])
        event_rows = sorted(local_by_episode.get(episode, []), key=lambda row: int(row["event_index"]))
        if len(event_rows) != 3 or any(not row["valid"] for row in event_rows):
            continue
        predicted_moves = [row["prediction"]["screen_pair"] for row in event_rows]
        recurrent_rows.append(
            {
                "three_moves_exact": predicted_moves == sequence_row["target_value"]["moves"],
                "final_correct": _rollout(sequence_row["target_value"]["initial_cup"], predicted_moves)
                == sequence_row["target_value"]["final_cup"],
            }
        )
    return {
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()),
        "manifest": str(args.manifest.resolve()),
        "samples": len(records),
        "episodes": len({int(row["episode_id"]) for row in records}),
        "valid_json_rate": len(valid) / max(len(records), 1),
        "raw_prediction_types": dict(sorted(Counter(str(row["sample_type"]) for row in records).items())),
        "local_swap_samples": len(local),
        "local_swap_pair_accuracy": local_correct / max(len(local), 1),
        "local_recurrent_coverage": len(recurrent_rows) / max(len(sequence), 1),
        "local_three_move_exact_accuracy": sum(row["three_moves_exact"] for row in recurrent_rows)
        / max(len(sequence), 1),
        "local_recurrent_final_accuracy": sum(row["final_correct"] for row in recurrent_rows)
        / max(len(sequence), 1),
        "sequence_samples": len(sequence),
        "sequence_valid_rate": len(sequence_valid) / max(len(sequence), 1),
        "sequence_initial_accuracy": initial_correct / max(len(sequence), 1),
        "sequence_move_slot_accuracy": move_slots_correct / max(3 * len(sequence), 1),
        "sequence_moves_exact_accuracy": moves_exact / max(len(sequence), 1),
        "sequence_final_accuracy": final_correct / max(len(sequence), 1),
        "sequence_rollout_final_accuracy": rollout_correct / max(len(sequence), 1),
        "sequence_internal_consistency_rate": internally_consistent / max(len(sequence), 1),
        "sequence_full_exact_accuracy": sequence_exact / max(len(sequence), 1),
        "wall_time_seconds": wall_time,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite and not args.reuse_records:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if args.reuse_records:
        records = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
        for record in records:
            sample_type = str(record["sample_type"])
            try:
                record["prediction"] = validate_response(str(record["raw_text"]), sample_type)
                record["valid"], record["error"] = True, None
            except Exception as exc:
                record["prediction"] = None
                record["valid"], record["error"] = False, f"{type(exc).__name__}: {exc}"
        summary = _summary(records, args, 0.0)
        with args.output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        args.output.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        return
    processor = AutoProcessor.from_pretrained(
        args.adapter_path,
        local_files_only=True,
        min_pixels=224 * 224,
        max_pixels=224 * 224,
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

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = _generate(processor, model, batch, args)
        for row, text in zip(batch, texts, strict=True):
            sample_type = str(row["sample_type"])
            target_value = validate_response(str(row["target"]), sample_type, require_consistent=True)
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
                    "target": str(row["target"]),
                    "target_value": target_value,
                    "raw_text": text,
                    "prediction": prediction,
                    "valid": valid,
                    "error": error,
                }
            )
        if start == 0 or start + len(batch) == len(rows) or (start + len(batch)) % 20 == 0:
            print(f"evaluated {start + len(batch)}/{len(rows)}", flush=True)
    summary = _summary(records, args, time.perf_counter() - started)
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
