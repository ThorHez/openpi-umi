#!/usr/bin/env python3
"""Evaluate PickXtimes memory while reusing a frozen detector's records."""

from __future__ import annotations

import argparse
import json
import pathlib

import eval_pickxtimes_causal_event_memory as causal_eval
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--detection-records", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--record-section", choices=("calibrated", "default"), default="calibrated")
    parser.add_argument("--use-decoder-event-types", action="store_true")
    return parser.parse_args()


def aggregate_memory(memory_records):
    flat_memory = [prompt for record in memory_records for prompt in record["prompts"]]
    metrics = {
        key + "_accuracy": float(np.mean([prompt[key] for prompt in flat_memory]))
        for key in (
            "goal_color_correct",
            "goal_count_correct",
            "final_count_correct",
            "final_remaining_correct",
            "final_done_correct",
            "state_sequence_e2e_correct",
            "strict_state_sequence_e2e_correct",
        )
    }
    metrics["trigger_overflow_rate"] = float(np.mean([prompt["trigger_overflow"] for prompt in flat_memory]))
    return metrics


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["episodes"]
    metadata_by_index = {int(episode["episode_index"]): episode for episode in labels}
    detection_payload = json.loads(args.detection_records.read_text(encoding="utf-8"))
    if args.record_section == "calibrated":
        detection_records = detection_payload["calibrated_threshold"]["episodes"]
    else:
        detection_records = detection_payload["default_threshold"]["episodes"]

    tracker = causal_eval.make_tracker(config)
    params = causal_eval.initialize_and_restore(tracker, args.checkpoint)
    memory_records = []
    with h5py.File(args.features, "r") as features:
        for ordinal, detection_record in enumerate(detection_records, start=1):
            episode_index = int(detection_record["episode_index"])
            metadata = metadata_by_index[episode_index]
            feature_episode = features[metadata["episode_name"]]
            prompt_records = [
                causal_eval.memory_record(
                    tracker,
                    params,
                    feature_episode,
                    metadata,
                    detection_record,
                    prompt_index=prompt_index,
                    use_decoder_event_types=args.use_decoder_event_types,
                )
                for prompt_index in range(len(metadata["prompts"]))
            ]
            memory_records.append({"episode_index": episode_index, "prompts": prompt_records})
            print(f"memory {ordinal}/{len(detection_records)} episode={episode_index}", flush=True)

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "detection_records": str(args.detection_records.resolve()),
        "record_section": args.record_section,
        "use_decoder_event_types": args.use_decoder_event_types,
        "evaluation_episode_indices": [int(record["episode_index"]) for record in detection_records],
        "memory_metrics_two_prompts_per_episode": aggregate_memory(memory_records),
        "memory_episodes": memory_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["memory_metrics_two_prompts_per_episode"], indent=2), flush=True)


if __name__ == "__main__":
    main()
