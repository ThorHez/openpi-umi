#!/usr/bin/env python3
"""Greedy held-out demonstration-memory evaluation for VideoPlaceOrder Qwen3-VL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
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

from openpi.tasks.robomme.videoplaceorder.qwen3vl_sft_contract import SYSTEM_PROMPT  # noqa: E402
from openpi.tasks.robomme.videoplaceorder.qwen3vl_sft_contract import prompt_for_task  # noqa: E402
from openpi.tasks.robomme.videoplaceorder.qwen3vl_sft_contract import validate_compact_response  # noqa: E402

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_MANIFEST = _ROOT / "artifacts/videoplaceorder_qwen3vl_sft_seed260825/val.jsonl"
_CELL_RE = re.compile(r"r(\d+)_c(\d+)\Z")


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
    indices = [int(value) for value in row["frame_indices"]]
    if max(indices) >= int(row["demo_end"]):
        raise ValueError(f"Execution frame leakage: {indices}")
    return [
        Image.fromarray(episode[f"timestep_{index}/obs/front_rgb"][()]).convert("RGB")
        for index in indices
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
                                str(row["target_color"]), int(row["ordinal"])
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


def _cell_distance(prediction: dict[str, Any] | None, expected: dict[str, Any]) -> int | None:
    if prediction is None or "target_cell" not in prediction or "target_cell" not in expected:
        return None
    predicted_match = _CELL_RE.fullmatch(str(prediction["target_cell"]))
    expected_match = _CELL_RE.fullmatch(str(expected["target_cell"]))
    if predicted_match is None or expected_match is None:
        return None
    return abs(int(predicted_match.group(1)) - int(expected_match.group(1))) + abs(
        int(predicted_match.group(2)) - int(expected_match.group(2))
    )


def _nearest_candidate_correct(
    prediction: dict[str, Any] | None,
    candidates_xy: list[list[int]],
    target_xy: list[int],
) -> bool:
    if prediction is None or "target_cell" not in prediction or not candidates_xy:
        return False
    predicted_match = _CELL_RE.fullmatch(str(prediction["target_cell"]))
    if predicted_match is None:
        return False
    predicted_xy = (
        int(predicted_match.group(2)) * 32 + 16,
        int(predicted_match.group(1)) * 32 + 16,
    )
    predicted_distances = [
        (predicted_xy[0] - xy[0]) ** 2 + (predicted_xy[1] - xy[1]) ** 2
        for xy in candidates_xy
    ]
    target_distances = [
        (target_xy[0] - xy[0]) ** 2 + (target_xy[1] - xy[1]) ** 2
        for xy in candidates_xy
    ]
    return predicted_distances.index(min(predicted_distances)) == target_distances.index(
        min(target_distances)
    )


def _rate(records: list[dict[str, Any]], key: str) -> float:
    return sum(bool(record[key]) for record in records) / max(len(records), 1)


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(records),
        "valid_rate": _rate(records, "valid"),
        "exact_accuracy": _rate(records, "correct"),
        "event_accuracy": _rate(records, "event_correct"),
        "target_cell_accuracy": _rate(records, "cell_correct"),
        "within_one_cell_accuracy": _rate(records, "within_one_cell"),
        "nearest_candidate_accuracy": _rate(records, "nearest_candidate_correct"),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = {}
    for sample_type in ("full_demo", "truncated_demo", "local_only"):
        subset = [record for record in records if record["sample_type"] == sample_type]
        by_type[sample_type] = {
            **_metrics(subset),
            "prediction_counts": dict(
                sorted(Counter(str(record["prediction"]) for record in subset).items())
            ),
        }
    full = [record for record in records if record["sample_type"] == "full_demo"]
    by_difficulty = {
        difficulty: _metrics([record for record in full if record["difficulty"] == difficulty])
        for difficulty in ("easy", "medium", "hard")
    }
    by_ordinal = {
        str(ordinal): _metrics([record for record in full if record["ordinal"] == ordinal])
        for ordinal in range(1, 5)
    }
    moved = [record for record in full if record["target_cell_moved"]]
    stale_predictions = sum(
        record["prediction"] is not None
        and record["prediction"].get("target_cell") == record["demonstrated_cell"]
        and not record["cell_correct"]
        for record in moved
    )
    variant_zero = [record for record in full if record["variant"] == 0]
    return {
        "episodes": len({record["episode_index"] for record in records}),
        "samples": len(records),
        "overall_valid_rate": _rate(records, "valid"),
        "overall_exact_accuracy": _rate(records, "correct"),
        "by_type": by_type,
        "full_demo_by_difficulty": by_difficulty,
        "full_demo_by_ordinal": by_ordinal,
        "moved_target": {
            **_metrics(moved),
            "stale_pre_swap_cell_rate": stale_predictions / max(len(moved), 1),
        },
        "episode_variant0_exact_accuracy": _rate(variant_zero, "correct"),
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
                expected = validate_compact_response(str(row["target"]))
                try:
                    prediction = validate_compact_response(text)
                    valid, error = True, None
                except Exception as exc:
                    prediction, valid, error = None, False, f"{type(exc).__name__}: {exc}"
                distance = _cell_distance(prediction, expected)
                candidate_cells = [str(value) for value in row["candidate_target_cells"]]
                candidate_xy = [[int(coord) for coord in value] for value in row["candidate_target_xy"]]
                target_xy = [int(value) for value in row["target_xy"]]
                records.append(
                    {
                        "episode_index": int(row["episode_index"]),
                        "sample_type": str(row["sample_type"]),
                        "variant": int(row["variant"]),
                        "difficulty": str(row["difficulty"]),
                        "target_color": str(row["target_color"]),
                        "ordinal": int(row["ordinal"]),
                        "target_cell_moved": bool(row["target_cell_moved"]),
                        "demonstrated_cell": row["demonstrated_cell"],
                        "candidate_target_cells": candidate_cells,
                        "candidate_target_xy": candidate_xy,
                        "target_xy": target_xy,
                        "frame_indices": row["frame_indices"],
                        "expected": expected,
                        "prediction": prediction,
                        "raw_text": text,
                        "valid": valid,
                        "correct": prediction == expected,
                        "event_correct": prediction is not None
                        and prediction.get("event") == expected.get("event"),
                        "cell_correct": distance == 0,
                        "within_one_cell": distance is not None and distance <= 1,
                        "nearest_candidate_correct": _nearest_candidate_correct(
                            prediction, candidate_xy, target_xy
                        ),
                        "cell_distance": distance,
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
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
