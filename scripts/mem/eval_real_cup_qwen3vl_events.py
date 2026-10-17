#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the ShellGame Qwen3-VL LoRA on unlabeled real-cup episodes.

Only chronological sliding video windows enter Qwen.  A simple red-ball
detector reads the visible initial/final screen slot for episode-level
consistency scoring; it never supplies a slot or relation to Qwen.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
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

from openpi.tasks.shellgame.qwen3vl_event_trigger import cluster_event_windows
from openpi.tasks.shellgame.qwen3vl_sft_contract import SYSTEM_PROMPT
from openpi.tasks.shellgame.qwen3vl_sft_contract import prompt_for_sample_type
from openpi.tasks.shellgame.qwen3vl_sft_contract import validate_compact_response
from openpi.tasks.shellgame.qwenvl_event_adapter import apply_swap

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_ADAPTER = ROOT / "checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375"
DEFAULT_EPISODE_DIR = ROOT / "artifacts/cup_replay_real_qwen_probe"
DEFAULT_OUTPUT = ROOT / "evaluation/shellgame/real_cup_qwen3vl_step375_probe/stride2.jsonl"
SCREEN_SLOTS = ("screen_left_cup", "screen_middle_cup", "screen_right_cup")
FORCED_SWAP_PROMPT = """This real-world clip is manually trimmed to contain exactly one completed
exchange of two cups. Track their horizontal screen positions from the first frame to the last.
Return exactly {\"screen_pair\":[\"LEFT_LABEL\",\"RIGHT_LABEL\"]}, ordered left-to-right on screen.
You must select one of the three possible cup pairs and must not return no_event or incomplete_event."""


@dataclass(frozen=True)
class WindowSpec:
    episode_index: int
    window_start: int
    frame_indices: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--episode-ids", default="0,1,2")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--source-frame-stride", type=int, default=2)
    parser.add_argument("--window-start-stride", type=int, default=5)
    parser.add_argument("--cluster-gap", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-windows-per-episode", type=int)
    parser.add_argument(
        "--explicit-ranges",
        help="Optional semicolon-separated EPISODE:START:END ranges; frames are uniformly sampled inclusive.",
    )
    parser.add_argument("--force-completed-swap", action="store_true")
    parser.add_argument("--reuse-records", action="store_true", help="Recompute summary without loading Qwen.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _episode_path(root: Path, episode: int) -> Path:
    path = root / f"episode_{episode:03d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _window_specs(episode: int, num_source_frames: int, args: argparse.Namespace) -> list[WindowSpec]:
    if args.explicit_ranges:
        result = []
        for raw in args.explicit_ranges.split(";"):
            range_episode, start, end = (int(value) for value in raw.split(":"))
            if range_episode != episode:
                continue
            if not 0 <= start < end < num_source_frames:
                raise ValueError(f"Invalid explicit range {raw!r} for episode length {num_source_frames}")
            indices = tuple(int(value) for value in np.linspace(start, end, args.num_frames).round())
            result.append(WindowSpec(episode_index=episode, window_start=start, frame_indices=indices))
        return result
    span = (args.num_frames - 1) * args.source_frame_stride + 1
    starts = list(range(0, num_source_frames - span + 1, args.window_start_stride))
    if args.max_windows_per_episode is not None and len(starts) > args.max_windows_per_episode:
        selected = np.linspace(0, len(starts) - 1, args.max_windows_per_episode, dtype=np.int64)
        starts = [starts[int(index)] for index in selected]
    return [
        WindowSpec(
            episode_index=episode,
            window_start=start,
            frame_indices=tuple(start + offset * args.source_frame_stride for offset in range(args.num_frames)),
        )
        for start in starts
    ]


def _red_ball_slot(frames: np.ndarray, *, at_end: bool) -> dict[str, Any]:
    """Detect the visible red ball in the first/last 80 source frames."""

    endpoint_span = min(80, len(frames))
    sample_start = len(frames) - endpoint_span if at_end else 0
    sample = frames[sample_start:] if at_end else frames[:endpoint_span]
    yy, xx = np.indices(sample.shape[1:3])
    best = None
    # Use the frame with the strongest red-ball component.  This remains valid
    # when the demonstrator reveals the endpoint before the very last frame.
    for offset, image in enumerate(sample.astype(np.float32)):
        red, green, blue = image[..., 0], image[..., 1], image[..., 2]
        mask = (yy >= 65) & (yy < 135) & (red > 145) & (red - green > 55) & (red - blue > 55)
        count = int(mask.sum())
        if best is None or count > best[0]:
            best = (count, offset, mask)
    assert best is not None
    count, offset, mask = best
    if count < 3:
        return {"slot": None, "x": None, "y": None, "pixels": count, "frame": None, "valid": False}
    x = float(np.median(xx[mask]))
    y = float(np.median(yy[mask]))
    # One-time screen calibration from the fixed real camera; no per-episode
    # intermediate annotation is used.
    anchors = np.asarray([84.0, 112.0, 140.0])
    slot_index = int(np.argmin(np.abs(anchors - x)))
    return {
        "slot": SCREEN_SLOTS[slot_index],
        "x": x,
        "y": y,
        "pixels": count,
        "frame": sample_start + offset,
        "valid": True,
    }


def _generate_batch(
    processor: Any,
    model: Any,
    frames_by_episode: dict[int, np.ndarray],
    specs: list[WindowSpec],
    args: argparse.Namespace,
) -> list[str]:
    conversations = []
    metadata = []
    effective_fps = args.source_fps / args.source_frame_stride
    for spec in specs:
        frames = frames_by_episode[spec.episode_index][np.asarray(spec.frame_indices, dtype=np.int64)]
        conversations.append(
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": [Image.fromarray(frame).convert("RGB") for frame in frames]},
                        {
                            "type": "text",
                            "text": FORCED_SWAP_PROMPT if args.force_completed_swap else prompt_for_sample_type("swap"),
                        },
                    ],
                },
            ]
        )
        metadata.append(
            VideoMetadata(
                total_num_frames=args.num_frames,
                fps=effective_fps,
                width=224,
                height=224,
                duration=args.num_frames / effective_fps,
                frames_indices=list(range(args.num_frames)),
            )
        )
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        num_frames=args.num_frames,
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


def _rollout(initial_slot: str | None, triggers: list[Any]) -> str | None:
    if initial_slot is None:
        return None
    # Stay in camera coordinates.  normalize_cup_entity intentionally mirrors
    # screen labels into simulator/world coordinates and must not be used for
    # this real-camera endpoint-consistency check.
    slot = ("left", "middle", "right")[SCREEN_SLOTS.index(initial_slot)]
    for trigger in triggers:
        pair = tuple(("left", "middle", "right")[SCREEN_SLOTS.index(value)] for value in trigger.pair)
        pair = tuple(sorted(pair, key=("left", "middle", "right").index))
        slot = apply_swap(slot, pair)
    return SCREEN_SLOTS[("left", "middle", "right").index(slot)]


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite and not args.reuse_records:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    episodes = [int(value.strip()) for value in args.episode_ids.split(",") if value.strip()]
    frames_by_episode = {}
    specs = []
    endpoint_slots = {}
    for episode in episodes:
        with np.load(_episode_path(args.episode_dir, episode), allow_pickle=False) as payload:
            frames = np.asarray(payload["frames"], dtype=np.uint8)
        frames_by_episode[episode] = frames
        specs.extend(_window_specs(episode, len(frames), args))
        endpoint_slots[episode] = {
            "initial": _red_ball_slot(frames, at_end=False),
            "final": _red_ball_slot(frames, at_end=True),
        }

    started = time.perf_counter()
    if args.reuse_records:
        records = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
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

        records = []
        for start in range(0, len(specs), args.batch_size):
            batch_specs = specs[start : start + args.batch_size]
            texts = _generate_batch(processor, model, frames_by_episode, batch_specs, args)
            for spec, text in zip(batch_specs, texts, strict=True):
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
                        **asdict(spec),
                        "frame_indices": list(spec.frame_indices),
                        "prediction": prediction,
                        "raw_text": text,
                        "valid": valid,
                        "error": error,
                    }
                )
            if start == 0 or (start + len(batch_specs)) % 100 < len(batch_specs) or start + len(batch_specs) == len(specs):
                print(f"evaluated {start + len(batch_specs)}/{len(specs)} windows", flush=True)

    per_episode = []
    for episode in episodes:
        windows = [record for record in records if int(record["episode_index"]) == episode]
        triggers = cluster_event_windows(windows, max_positive_gap=args.cluster_gap)
        initial = endpoint_slots[episode]["initial"]["slot"]
        final = endpoint_slots[episode]["final"]["slot"]
        rollout = _rollout(initial, triggers)
        prediction_counts = Counter(
            "swap" if isinstance(record["prediction"], dict) and "screen_pair" in record["prediction"]
            else str(None if record["prediction"] is None else record["prediction"].get("event"))
            for record in windows
        )
        per_episode.append(
            {
                "episode_index": episode,
                "frames": len(frames_by_episode[episode]),
                "windows": len(windows),
                "prediction_counts": dict(sorted(prediction_counts.items())),
                "initial_ball": endpoint_slots[episode]["initial"],
                "final_ball": endpoint_slots[episode]["final"],
                "triggers": [asdict(trigger) for trigger in triggers],
                "rollout_final_slot": rollout,
                "final_consistent": None if rollout is None or final is None else rollout == final,
            }
        )

    valid_endpoint_rows = [row for row in per_episode if row["final_consistent"] is not None]
    trigger_count = sum(len(row["triggers"]) for row in per_episode)
    swap_window_count = sum(int(row["prediction_counts"].get("swap", 0)) for row in per_episode)
    summary = {
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()),
        "episodes": len(episodes),
        "episode_ids": episodes,
        "windows": len(records),
        "valid_json_rate": sum(bool(record["valid"]) for record in records) / max(len(records), 1),
        "source_fps": args.source_fps,
        "source_frame_stride": args.source_frame_stride,
        "effective_qwen_fps": args.source_fps / args.source_frame_stride,
        "window_start_stride": args.window_start_stride,
        "explicit_ranges": args.explicit_ranges,
        "force_completed_swap": args.force_completed_swap,
        "swap_window_count": swap_window_count,
        "swap_window_rate": swap_window_count / max(len(records), 1),
        "trigger_count": trigger_count,
        "valid_endpoint_episodes": len(valid_endpoint_rows),
        "final_consistency_rate": sum(bool(row["final_consistent"]) for row in valid_endpoint_rows)
        / max(len(valid_endpoint_rows), 1),
        "no_change_endpoint_baseline_rate": sum(
            row["initial_ball"]["slot"] == row["final_ball"]["slot"] for row in valid_endpoint_rows
        )
        / max(len(valid_endpoint_rows), 1),
        "uses_intermediate_gt": False,
        "uses_endpoint_slot_as_qwen_input": False,
        "endpoint_source": "fixed-ROI red-ball detector over visible first/last frames",
        "important_limitation": "final consistency cannot establish intermediate event correctness; manual event annotation is required on a small audit subset",
        "wall_time_seconds": 0.0 if args.reuse_records else time.perf_counter() - started,
        "per_episode": per_episode,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
