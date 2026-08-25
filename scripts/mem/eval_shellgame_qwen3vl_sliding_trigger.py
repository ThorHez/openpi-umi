# ruff: noqa: E402
"""Evaluate Qwen3-VL event discovery on continuous held-out ShellGame video.

The model receives only chronological sliding windows and one generic exchange
prompt.  Phase boundaries and GT pairs are used after generation solely for
scoring.  Overlapping positive windows are clustered without GT to measure
misses, duplicates, and relation-sequence accuracy.
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

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.shellgame.qwen3vl_event_trigger import cluster_event_windows
from openpi.tasks.shellgame.qwen3vl_event_trigger import score_event_triggers
from openpi.tasks.shellgame.qwen3vl_sft_contract import SYSTEM_PROMPT
from openpi.tasks.shellgame.qwen3vl_sft_contract import compact_response
from openpi.tasks.shellgame.qwen3vl_sft_contract import prompt_for_sample_type
from openpi.tasks.shellgame.qwen3vl_sft_contract import validate_compact_response
from openpi.tasks.shellgame.qwenvl_event_adapter import screen_pair_from_world_pair

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_ADAPTER = _ROOT / "checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375"
DEFAULT_MANIFEST = _ROOT / "artifacts/shellgame_qwen3vl_gt_event_sft_v1/val.jsonl"
DEFAULT_OUTPUT = _ROOT / "evaluation/shellgame/qwen3vl_gt_event_lora_v1_step375_sliding_trigger_val20.jsonl"


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    episode_index: int
    trajectory_path: str
    suite: str
    condition: str
    frame_indices: tuple[int, ...]
    video_fps: float
    expected_text: str
    window_start: int | None = None
    expected_kind: str = "swap"
    gt_stage: int | None = None
    relative_offset: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        help="Override selected held-out episode paths with ROOT/episode_NNNNNN/vla_trajectory.npz.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument(
        "--episode-ids",
        help="Optional comma-separated episode ids; requires --trajectory-root and bypasses manifest selection.",
    )
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--cluster-gap", type=int, default=3)
    parser.add_argument("--trigger-tolerance", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temporal-stress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _episode_sources(
    manifest: Path,
    num_episodes: int,
    offset: int,
    trajectory_root: Path | None = None,
    explicit_episode_ids: str | None = None,
) -> dict[int, str]:
    if explicit_episode_ids is not None:
        if trajectory_root is None:
            raise ValueError("--episode-ids requires --trajectory-root")
        selected_ids = [int(value.strip()) for value in explicit_episode_ids.split(",") if value.strip()]
        if len(selected_ids) != num_episodes:
            raise ValueError(f"--episode-ids contains {len(selected_ids)} ids but --num-episodes={num_episodes}")
        result = {
            episode: str((trajectory_root / f"episode_{episode:06d}" / "vla_trajectory.npz").resolve())
            for episode in selected_ids
        }
        missing = [path for path in result.values() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing overridden trajectories: {missing[:5]}")
        return result
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_by_episode = {}
    for row in rows:
        source_by_episode.setdefault(int(row["episode_index"]), str(row["trajectory_path"]))
    selected_ids = sorted(source_by_episode)[offset : offset + num_episodes]
    if len(selected_ids) != num_episodes:
        raise ValueError(f"Selected only {len(selected_ids)} of {num_episodes} requested episodes")
    if trajectory_root is None:
        return {episode: source_by_episode[episode] for episode in selected_ids}
    result = {
        episode: str((trajectory_root / f"episode_{episode:06d}" / "vla_trajectory.npz").resolve())
        for episode in selected_ids
    }
    missing = [path for path in result.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing overridden trajectories: {missing[:5]}")
    return result


def _metadata(trajectory_path: str) -> dict[str, Any]:
    return json.loads(Path(trajectory_path).with_name("metadata.json").read_text(encoding="utf-8"))


def _swap_contract(metadata: dict[str, Any]) -> tuple[list[tuple[int, int]], list[tuple[str, str]]]:
    ranges = []
    pairs = []
    for stage, world_pair in enumerate(metadata["swaps"]):
        start, end = metadata["phase_ranges"][f"swap_{stage}"]
        ranges.append((int(start), int(end)))
        pairs.append(screen_pair_from_world_pair(world_pair))
    return ranges, pairs


def _dense_specs(
    episode: int,
    trajectory_path: str,
    metadata: dict[str, Any],
    *,
    window_size: int,
    stride: int,
) -> list[ClipSpec]:
    swap_ranges, swap_pairs = _swap_contract(metadata)
    scan_start = int(metadata["phase_ranges"]["cover"][0])
    scan_end = int(metadata["phase_ranges"]["settle"][1]) - window_size + 1
    specs = []
    for start in range(scan_start, scan_end + 1, stride):
        end = start + window_size - 1
        overlapping = [stage for stage, (left, right) in enumerate(swap_ranges) if start <= right and end >= left]
        contained = [stage for stage, (left, right) in enumerate(swap_ranges) if start <= left and end >= right]
        if len(contained) == 1:
            stage = contained[0]
            expected_kind = "swap"
            expected_text = compact_response("swap", swap_pairs[stage])
            gt_stage = stage
        elif overlapping:
            expected_kind = "incomplete_event"
            expected_text = compact_response("incomplete_event")
            gt_stage = min(overlapping, key=lambda value: abs(start - swap_ranges[value][0]))
        else:
            expected_kind = "no_event"
            expected_text = compact_response("no_event")
            gt_stage = None
        relative_offset = None if gt_stage is None else start - swap_ranges[gt_stage][0]
        specs.append(
            ClipSpec(
                clip_id=f"dense-ep{episode:06d}-s{start:03d}",
                episode_index=episode,
                trajectory_path=trajectory_path,
                suite="dense_sliding",
                condition="window10_stride1",
                frame_indices=tuple(range(start, end + 1)),
                video_fps=10.0,
                expected_text=expected_text,
                window_start=start,
                expected_kind=expected_kind,
                gt_stage=gt_stage,
                relative_offset=relative_offset,
            )
        )
    return specs


def _resample_range(start: int, end: int, positions: list[float]) -> tuple[int, ...]:
    return tuple(round(start + value * (end - start)) for value in positions)


def _stress_specs(
    episode: int,
    trajectory_path: str,
    metadata: dict[str, Any],
) -> list[ClipSpec]:
    swap_ranges, swap_pairs = _swap_contract(metadata)
    specs = []
    for stage, ((start, end), pair) in enumerate(zip(swap_ranges, swap_pairs, strict=True)):
        condition_frames = {
            "clean_10f_10fps": (tuple(range(start, end + 1)), 10.0),
            "context_left2": (tuple(range(start - 2, end + 1)), 10.0),
            "context_right2": (tuple(range(start, end + 3)), 10.0),
            "context_both2": (tuple(range(start - 2, end + 3)), 10.0),
            "drop_to_6f": (_resample_range(start, end, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), 6.0),
            "fps_metadata_5": (tuple(range(start, end + 1)), 5.0),
            "fps_metadata_15": (tuple(range(start, end + 1)), 15.0),
            "motion_early": (
                _resample_range(start, end, [0.0, 0.22, 0.44, 0.67, 0.89, 1.0, 1.0, 1.0, 1.0, 1.0]),
                10.0,
            ),
            "motion_late": (
                _resample_range(start, end, [0.0, 0.0, 0.0, 0.0, 0.0, 0.11, 0.33, 0.56, 0.78, 1.0]),
                10.0,
            ),
        }
        for condition, (indices, fps) in condition_frames.items():
            specs.append(
                ClipSpec(
                    clip_id=f"stress-ep{episode:06d}-swap{stage}-{condition}",
                    episode_index=episode,
                    trajectory_path=trajectory_path,
                    suite="temporal_stress",
                    condition=condition,
                    frame_indices=indices,
                    video_fps=fps,
                    expected_text=compact_response("swap", pair),
                    expected_kind="swap",
                    gt_stage=stage,
                )
            )
    return specs


class FrameCache:
    def __init__(self):
        self._values: dict[str, np.ndarray] = {}

    def frames(self, spec: ClipSpec) -> list[Image.Image]:
        if spec.trajectory_path not in self._values:
            with np.load(spec.trajectory_path, allow_pickle=False) as trajectory:
                self._values[spec.trajectory_path] = np.asarray(trajectory["third_person_images"], dtype=np.uint8)
        array = self._values[spec.trajectory_path][np.asarray(spec.frame_indices, dtype=np.int64)]
        return [Image.fromarray(frame).convert("RGB") for frame in array]


def _generate_batch(processor, model, cache, specs, device, max_new_tokens):
    num_frames = len(specs[0].frame_indices)
    if any(len(spec.frame_indices) != num_frames for spec in specs):
        raise ValueError("A generation batch must use one frame count")
    conversations = []
    metadata = []
    for spec in specs:
        frames = cache.frames(spec)
        conversations.append(
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": prompt_for_sample_type("swap")},
                    ],
                },
            ]
        )
        metadata.append(
            VideoMetadata(
                total_num_frames=num_frames,
                fps=spec.video_fps,
                width=224,
                height=224,
                duration=num_frames / spec.video_fps,
                frames_indices=list(range(num_frames)),
            )
        )
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        num_frames=num_frames,
        video_metadata=metadata,
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _mean(values):
    return sum(values) / max(len(values), 1)


def _summarize(records, sources, args):
    dense = [record for record in records if record["suite"] == "dense_sliding"]
    stress = [record for record in records if record["suite"] == "temporal_stress"]
    dense_by_expected = {}
    for kind in ("swap", "incomplete_event", "no_event"):
        subset = [record for record in dense if record["expected_kind"] == kind]
        dense_by_expected[kind] = {
            "count": len(subset),
            "exact_accuracy": _mean([bool(record["correct"]) for record in subset]),
            "non_trigger_rate": _mean(
                [
                    not isinstance(record["prediction"], dict) or "screen_pair" not in record["prediction"]
                    for record in subset
                ]
            ),
            "prediction_counts": dict(sorted(Counter(str(record["prediction"]) for record in subset).items())),
        }
    offsets = {}
    for offset in sorted({record["relative_offset"] for record in dense if record["relative_offset"] is not None}):
        subset = [record for record in dense if record["relative_offset"] == offset]
        offsets[str(offset)] = {
            "count": len(subset),
            "trigger_rate": _mean(
                [isinstance(record["prediction"], dict) and "screen_pair" in record["prediction"] for record in subset]
            ),
            "exact_accuracy": _mean([bool(record["correct"]) for record in subset]),
        }

    per_episode = []
    for episode, trajectory_path in sources.items():
        metadata = _metadata(trajectory_path)
        swap_ranges, gt_pairs = _swap_contract(metadata)
        episode_windows = [record for record in dense if int(record["episode_index"]) == episode]
        triggers = cluster_event_windows(episode_windows, max_positive_gap=args.cluster_gap)
        score = score_event_triggers(
            triggers,
            gt_starts=[left for left, _ in swap_ranges],
            gt_pairs=gt_pairs,
            tolerance=args.trigger_tolerance,
        )
        per_episode.append(
            {
                "episode_index": episode,
                "triggers": [asdict(value) for value in triggers],
                **score,
            }
        )
    total_triggers = sum(int(value["num_triggers"]) for value in per_episode)
    total_gt = sum(int(value["num_gt_events"]) for value in per_episode)
    correct_events = sum(int(value["correct_pair_events"]) for value in per_episode)
    trigger_summary = {
        "uses_gt_during_triggering": False,
        "episodes": len(per_episode),
        "num_triggers": total_triggers,
        "num_gt_events": total_gt,
        "correct_pair_events": correct_events,
        "false_positive_or_duplicate_triggers": sum(
            int(value["false_positive_or_duplicate_triggers"]) for value in per_episode
        ),
        "missed_events": sum(int(value["missed_events"]) for value in per_episode),
        "event_precision": correct_events / max(total_triggers, 1),
        "event_recall": correct_events / max(total_gt, 1),
        "exact_three_relation_sequence_accuracy": _mean(
            [bool(value["exact_relation_sequence"]) for value in per_episode]
        ),
        "per_episode": per_episode,
    }

    stress_by_condition = {}
    for condition in sorted({record["condition"] for record in stress}):
        subset = [record for record in stress if record["condition"] == condition]
        stress_by_condition[condition] = {
            "count": len(subset),
            "accuracy": _mean([bool(record["correct"]) for record in subset]),
            "valid_rate": _mean([bool(record["valid"]) for record in subset]),
            "prediction_counts": dict(sorted(Counter(str(record["prediction"]) for record in subset).items())),
        }
    return {
        "episodes": len(sources),
        "dense_windows": len(dense),
        "dense_by_expected": dense_by_expected,
        "dense_by_relative_offset": offsets,
        "event_trigger": trigger_summary,
        "temporal_stress": stress_by_condition,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    sources = _episode_sources(
        args.manifest,
        args.num_episodes,
        args.episode_offset,
        args.trajectory_root,
        args.episode_ids,
    )
    specs = []
    for episode, trajectory_path in sources.items():
        metadata = _metadata(trajectory_path)
        specs.extend(
            _dense_specs(
                episode,
                trajectory_path,
                metadata,
                window_size=args.window_size,
                stride=args.window_stride,
            )
        )
        if args.temporal_stress:
            specs.extend(_stress_specs(episode, trajectory_path, metadata))

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

    # Batch only clips with equal frame counts; their FPS may differ because
    # VideoMetadata remains per sample.
    groups: dict[int, list[ClipSpec]] = {}
    for spec in specs:
        groups.setdefault(len(spec.frame_indices), []).append(spec)
    cache = FrameCache()
    records_by_id = {}
    started = time.perf_counter()
    processed = 0
    for _num_frames, group in sorted(groups.items()):
        for start in range(0, len(group), args.batch_size):
            batch_specs = group[start : start + args.batch_size]
            texts = _generate_batch(processor, model, cache, batch_specs, args.device, args.max_new_tokens)
            for spec, text in zip(batch_specs, texts, strict=True):
                expected = validate_compact_response(spec.expected_text)
                try:
                    prediction = validate_compact_response(text)
                    valid = True
                    error = None
                except Exception as exc:
                    prediction = None
                    valid = False
                    error = f"{type(exc).__name__}: {exc}"
                records_by_id[spec.clip_id] = {
                    **asdict(spec),
                    "frame_indices": list(spec.frame_indices),
                    "expected": expected,
                    "prediction": prediction,
                    "raw_text": text,
                    "valid": valid,
                    "correct": prediction == expected,
                    "error": error,
                }
            processed += len(batch_specs)
            if processed % 100 < len(batch_specs) or processed == len(specs):
                print(f"evaluated {processed}/{len(specs)} clips", flush=True)
    records = [records_by_id[spec.clip_id] for spec in specs]
    summary = _summarize(records, sources, args)
    summary.update(
        {
            "model_path": str(args.model_path.resolve()),
            "adapter_path": str(args.adapter_path.resolve()),
            "episode_ids": list(sources),
            "wall_time_seconds": time.perf_counter() - started,
            "synthetic_speed_scope": (
                "motion_early/motion_late preserve event endpoints but are temporal resampling; "
                "true simulator swap_frames variants must be evaluated separately"
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
