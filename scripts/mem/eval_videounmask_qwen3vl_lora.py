#!/usr/bin/env python3
"""Greedy held-out evaluation for VideoUnmask Qwen3-VL compact memory outputs."""

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

from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import SYSTEM_PROMPT  # noqa: E402
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import prompt_for_target  # noqa: E402
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import validate_compact_response  # noqa: E402

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_MANIFEST = _ROOT / "artifacts/videounmask_qwen3vl_sft_seed260823/val.jsonl"
_CELL_RE = re.compile(r"r([0-7])_c([0-7])\Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _frames(h5: h5py.File, row: dict[str, Any]) -> list[Image.Image]:
    episode = h5[str(row["episode_name"])]
    return [
        Image.fromarray(episode[f"timestep_{int(index)}/obs/front_rgb"][()]).convert("RGB")
        for index in row["frame_indices"]
    ]


def _inputs(processor: Any, h5: h5py.File, rows: list[dict[str, Any]], device: str) -> dict[str, torch.Tensor]:
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
                        {"type": "text", "text": prompt_for_target(str(row["target_color"]))},
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


def _color_centers(image: Any) -> dict[str, tuple[float, float]]:
    import numpy as np

    masks = {
        "red": (image[..., 0] > 180) & (image[..., 1] < 70) & (image[..., 2] < 70),
        "green": (image[..., 1] > 180) & (image[..., 0] < 70) & (image[..., 2] < 70),
        "blue": (image[..., 2] > 180) & (image[..., 0] < 70) & (image[..., 1] < 70),
    }
    result = {}
    for color, mask in masks.items():
        y, x = np.where(mask)
        result[color] = (float(np.median(y)), float(np.median(x)))
    return result


def _nearest_container_correct(
    prediction: dict[str, Any] | None,
    target_color: str,
    centers: dict[str, tuple[float, float]],
) -> bool:
    import numpy as np

    if prediction is None or not isinstance(prediction.get("target_cell"), str):
        return False
    match = _CELL_RE.fullmatch(prediction["target_cell"])
    if match is None:
        return False
    row, column = (int(value) for value in match.groups())
    predicted_yx = np.asarray([(row + 0.5) * 32, (column + 0.5) * 32])
    nearest = min(
        centers,
        key=lambda color: float(np.linalg.norm(predicted_yx - np.asarray(centers[color]))),
    )
    return nearest == target_color


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = {}
    for sample_type in ("paired_memory", "visible_grounding", "masked_only"):
        subset = [record for record in records if record["sample_type"] == sample_type]
        grounded = [record for record in subset if record["expected"].get("target_cell") is not None]
        by_type[sample_type] = {
            "samples": len(subset),
            "valid_rate": sum(record["valid"] for record in subset) / max(len(subset), 1),
            "exact_accuracy": sum(record["correct"] for record in subset) / max(len(subset), 1),
            "event_accuracy": sum(record["event_correct"] for record in subset) / max(len(subset), 1),
            "target_cell_accuracy": (
                sum(record["cell_correct"] for record in grounded) / len(grounded) if grounded else None
            ),
            "nearest_container_accuracy": (
                sum(record["nearest_container_correct"] for record in grounded) / len(grounded)
                if grounded
                else None
            ),
            "prediction_counts": dict(sorted(Counter(str(record["prediction"]) for record in subset).items())),
        }
    task_single = [
        record
        for record in records
        if record["target_color"] == record["goal_target_color"] and record["num_targets"] == 1
    ]
    task_single_by_type = {}
    for sample_type in ("paired_memory", "visible_grounding", "masked_only"):
        subset = [record for record in task_single if record["sample_type"] == sample_type]
        task_single_by_type[sample_type] = {
            "samples": len(subset),
            "exact_accuracy": sum(record["correct"] for record in subset) / max(len(subset), 1),
            "nearest_container_accuracy": (
                sum(record["nearest_container_correct"] for record in subset) / len(subset)
                if sample_type != "masked_only"
                else None
            ),
        }
    return {
        "episodes": len({int(record["episode_index"]) for record in records}),
        "samples": len(records),
        "overall_exact_accuracy": sum(record["correct"] for record in records) / max(len(records), 1),
        "overall_valid_rate": sum(record["valid"] for record in records) / max(len(records), 1),
        "by_type": by_type,
        "task_prompt_single_target": task_single_by_type,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("source") == "videounmask"]
    if not args.all_variants:
        rows = [row for row in rows if int(row["variant"]) == 0]
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
    records = []
    started = time.perf_counter()
    h5_paths = {str(row["h5_path"]) for row in rows}
    if len(h5_paths) != 1:
        raise ValueError(f"Expected one HDF5 source, got {h5_paths}")
    with h5py.File(next(iter(h5_paths)), "r") as h5:
        centers_by_episode = {
            int(row["episode_index"]): _color_centers(
                h5[f'{row["episode_name"]}/timestep_0/obs/front_rgb'][()]
            )
            for row in rows
        }
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
                output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)
            ]
            texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for row, text in zip(batch_rows, texts, strict=True):
                expected = validate_compact_response(str(row["target"]))
                try:
                    prediction = validate_compact_response(text)
                    valid, error = True, None
                except Exception as exc:
                    prediction, valid, error = None, False, f"{type(exc).__name__}: {exc}"
                records.append(
                    {
                        "episode_index": int(row["episode_index"]),
                        "target_color": str(row["target_color"]),
                        "goal_target_color": str(row["goal_target_color"]),
                        "difficulty": str(row["difficulty"]),
                        "num_targets": int(row["num_targets"]),
                        "sample_type": str(row["sample_type"]),
                        "variant": int(row["variant"]),
                        "frame_indices": row["frame_indices"],
                        "expected": expected,
                        "prediction": prediction,
                        "raw_text": text,
                        "valid": valid,
                        "correct": prediction == expected,
                        "event_correct": prediction is not None and prediction.get("event") == expected.get("event"),
                        "cell_correct": prediction is not None and prediction.get("target_cell") == expected.get("target_cell"),
                        "nearest_container_correct": _nearest_container_correct(
                            prediction,
                            str(row["target_color"]),
                            centers_by_episode[int(row["episode_index"])],
                        ),
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
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
