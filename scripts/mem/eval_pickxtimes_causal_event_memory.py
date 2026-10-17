#!/usr/bin/env python3
"""Evaluate PickXtimes memory with a full stride-1 causal window scan."""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import time
from typing import Any

import flax
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip_mem_semantic as memory_core
from openpi.tasks.robomme.pickxtimes import semantic_memory_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--window-classifier-checkpoint", type=pathlib.Path)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--split", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--max-train-episodes", type=int)
    parser.add_argument("--max-val-episodes", type=int)
    parser.add_argument("--calibration-split", choices=("train", "val"), default="train")
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="val")
    parser.add_argument("--decoder", choices=("hysteresis", "transition_grammar"), default="hysteresis")
    parser.add_argument("--transition-pick-place-probability", type=float, default=0.5)
    parser.add_argument("--transition-press-probability", type=float, default=0.6)
    parser.add_argument("--transition-press-type-probability", type=float, default=0.5)
    parser.add_argument("--transition-press-min-gap", type=int, default=20)
    parser.add_argument("--transition-disable-press-type-gate", action="store_true")
    parser.add_argument("--calibrate-transition-press", action="store_true")
    parser.add_argument("--save-window-logits", action="store_true")
    return parser.parse_args()


def probability_to_logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def causal_hysteresis(event_logits: np.ndarray, *, high_probability: float, low_probability: float) -> list[int]:
    if not 0.0 < low_probability < high_probability < 1.0:
        raise ValueError((low_probability, high_probability))
    high = probability_to_logit(high_probability)
    low = probability_to_logit(low_probability)
    active = False
    triggers = []
    for index, score in enumerate(event_logits):
        if not active and score > high:
            triggers.append(index)
            active = True
        elif active and score <= low:
            active = False
    return triggers


def strict_matches(metadata: dict[str, Any], triggers: list[int]) -> list[tuple[int, int]]:
    """One-to-one matches where a trigger lies in an annotated positive-start set."""
    trigger_by_start: dict[int, list[int]] = collections.defaultdict(list)
    for trigger_index, start in enumerate(triggers):
        trigger_by_start[start].append(trigger_index)
    matches = []
    used_triggers = set()
    for event_index, event in enumerate(metadata["events"]):
        candidates = [
            trigger_index
            for start in event["positive_starts"]
            for trigger_index in trigger_by_start.get(int(start), [])
            if trigger_index not in used_triggers
        ]
        if candidates:
            trigger_index = min(candidates)
            used_triggers.add(trigger_index)
            matches.append((event_index, trigger_index))
    return matches


def episode_detection_record(
    metadata: dict[str, Any],
    event_logits: np.ndarray,
    event_type_logits: np.ndarray,
    press_event_logits: np.ndarray | None = None,
    *,
    decoder: str = "hysteresis",
    high_probability: float = 0.7,
    low_probability: float = 0.3,
    transition_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if decoder == "hysteresis":
        triggers = causal_hysteresis(
            event_logits,
            high_probability=high_probability,
            low_probability=low_probability,
        )
        predicted_types = np.argmax(event_type_logits[triggers], axis=-1).astype(int).tolist() if triggers else []
    elif decoder == "transition_grammar":
        transition_config = transition_config or {}
        triggers, predicted_types = semantic_memory_event.transition_grammar_events(
            metadata["gripper_closed"],
            event_logits,
            event_type_logits,
            press_event_logits,
            required_count=int(metadata["required_count"]),
            **transition_config,
        )
    else:
        raise ValueError(f"Unknown decoder: {decoder}")
    matches = strict_matches(metadata, triggers)
    type_correct = 0
    timing_errors = []
    matched_by_class = collections.Counter()
    typed_by_class = collections.Counter()
    for event_index, trigger_index in matches:
        event = metadata["events"][event_index]
        event_type = int(event["event_type_id"])
        matched_by_class[event_type] += 1
        type_correct += int(predicted_types[trigger_index] == event_type)
        typed_by_class[event_type] += int(predicted_types[trigger_index] == event_type)
        nominal_start = float(np.median(event["positive_starts"]))
        timing_errors.append(abs(triggers[trigger_index] - nominal_start))

    expected_types = [int(event["event_type_id"]) for event in metadata["events"]]
    matched_event_indices = [event_index for event_index, _ in matches]
    strict_sequence = len(triggers) == len(expected_types) and matched_event_indices == list(range(len(expected_types)))
    typed_sequence = strict_sequence and predicted_types == expected_types
    return {
        "episode_index": int(metadata["episode_index"]),
        "required_count": int(metadata["required_count"]),
        "target_color": metadata["target_color"],
        "num_windows": int(event_logits.shape[0]),
        "expected_trigger_count": len(expected_types),
        "trigger_count": len(triggers),
        "trigger_starts": triggers,
        "predicted_types": predicted_types,
        "true_positive_count": len(matches),
        "false_trigger_count": len(triggers) - len(matches),
        "missed_event_count": len(expected_types) - len(matches),
        "type_correct_count": type_correct,
        "matched_by_class": dict(matched_by_class),
        "typed_by_class": dict(typed_by_class),
        "timing_absolute_errors": timing_errors,
        "exact_trigger_count": len(triggers) == len(expected_types),
        "strict_event_sequence": strict_sequence,
        "typed_event_sequence": typed_sequence,
    }


def aggregate_detection(records: list[dict[str, Any]], metadata_by_index: dict[int, dict[str, Any]]) -> dict[str, Any]:
    true_positives = sum(record["true_positive_count"] for record in records)
    predicted = sum(record["trigger_count"] for record in records)
    expected = sum(record["expected_trigger_count"] for record in records)
    precision = true_positives / max(predicted, 1)
    recall = true_positives / max(expected, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    timing_errors = [value for record in records for value in record["timing_absolute_errors"]]
    class_names = ("PICK", "PLACE", "PRESS")
    typed_class_recall = {}
    matched_class_recall = {}
    for class_id, class_name in enumerate(class_names):
        class_total = sum(
            int(event["event_type_id"] == class_id)
            for record in records
            for event in metadata_by_index[record["episode_index"]]["events"]
        )
        matched = sum(int(record["matched_by_class"].get(class_id, 0)) for record in records)
        typed = sum(int(record["typed_by_class"].get(class_id, 0)) for record in records)
        matched_class_recall[class_name] = matched / max(class_total, 1)
        typed_class_recall[class_name] = typed / max(class_total, 1)
    return {
        "episodes": len(records),
        "ground_truth_events": expected,
        "predicted_triggers": predicted,
        "mean_trigger_count": predicted / max(len(records), 1),
        "trigger_precision": precision,
        "trigger_recall": recall,
        "trigger_f1": f1,
        "exact_trigger_count_accuracy": float(np.mean([record["exact_trigger_count"] for record in records])),
        "strict_event_sequence_accuracy": float(np.mean([record["strict_event_sequence"] for record in records])),
        "typed_event_sequence_accuracy": float(np.mean([record["typed_event_sequence"] for record in records])),
        "matched_event_type_accuracy": sum(record["type_correct_count"] for record in records) / max(true_positives, 1),
        "matched_class_recall": matched_class_recall,
        "typed_class_recall": typed_class_recall,
        "matched_start_mae_frames": float(np.mean(timing_errors)) if timing_errors else None,
        "false_triggers": predicted - true_positives,
        "missed_events": expected - true_positives,
    }


def calibration_score(metrics: dict[str, Any]) -> float:
    typed_recall = float(np.mean(list(metrics["typed_class_recall"].values())))
    return (metrics["trigger_f1"] + metrics["exact_trigger_count_accuracy"] + typed_recall) / 3.0


def make_tracker(config: dict[str, Any]):
    return semantic_memory_event.PickXtimesSlidingWindowEventMemoryTracker(
        encoder_width=int(config["encoder_width"]),
        encoder_depth=int(config["encoder_depth"]),
        memory_width=int(config["memory_width"]),
        memory_depth=int(config["memory_depth"]),
        num_memory_tokens=int(config["memory_tokens"]),
        # Repeated identical event-code tokens are attention-equivalent to one
        # code token; using one avoids needless causal-evaluation memory use.
        evidence_tokens_per_event=1,
    )


def restore_compatible_params(target_params, checkpoint: pathlib.Path):
    """Restore matching leaves while retaining newly initialized parameters."""
    was_frozen = isinstance(target_params, flax.core.FrozenDict)
    target = flax.traverse_util.flatten_dict(flax.core.unfreeze(target_params))
    source = flax.traverse_util.flatten_dict(flax.serialization.msgpack_restore(checkpoint.read_bytes()))
    restored = 0
    for key, reference in target.items():
        candidate = source.get(key)
        if candidate is not None and np.shape(candidate) == np.shape(reference):
            target[key] = jnp.asarray(candidate, dtype=reference.dtype)
            restored += 1
    result = flax.traverse_util.unflatten_dict(target)
    return (flax.core.freeze(result) if was_frozen else result), restored, len(target) - restored


def initialize_and_restore(
    tracker,
    checkpoint: pathlib.Path,
    window_classifier_checkpoint: pathlib.Path | None = None,
):
    variables = tracker.init(
        jax.random.key(0),
        jnp.zeros((1, 1, semantic_memory_event.WINDOW_SIZE, 256, 1152), dtype=jnp.float16),
        jnp.zeros((1, 1, semantic_memory_event.WINDOW_SIZE), dtype=jnp.bool_),
        jnp.zeros((1, 256, 2048), dtype=jnp.float16),
        jnp.ones((1, 256), dtype=jnp.bool_),
        jnp.zeros((1, semantic_memory_event.MAX_EVENTS), dtype=jnp.int32),
        jnp.zeros((1, semantic_memory_event.MAX_EVENTS), dtype=jnp.bool_),
        causal_selection=False,
        train=False,
    )
    params, _, _ = restore_compatible_params(variables["params"], checkpoint)
    if window_classifier_checkpoint is not None:
        classifier_source, _, _ = restore_compatible_params(params, window_classifier_checkpoint)
        was_frozen = isinstance(params, flax.core.FrozenDict)
        mutable = flax.core.unfreeze(params)
        source_mutable = flax.core.unfreeze(classifier_source)
        mutable["window_classifier"] = source_mutable["window_classifier"]
        params = flax.core.freeze(mutable) if was_frozen else mutable
    return params


def scan_episode_classifier(
    patch_dataset: h5py.Dataset,
    gripper_closed: np.ndarray,
    *,
    batch_size: int,
    classifier_apply,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_windows = patch_dataset.shape[0] - semantic_memory_event.WINDOW_SIZE + 1
    all_event_logits = []
    all_type_logits = []
    all_press_logits = []
    for start in range(0, num_windows, batch_size):
        stop = min(start + batch_size, num_windows)
        timeline = np.asarray(patch_dataset[start : stop + semantic_memory_event.WINDOW_SIZE - 1])
        windows = np.stack(
            [timeline[offset : offset + semantic_memory_event.WINDOW_SIZE] for offset in range(stop - start)]
        )
        real_count = windows.shape[0]
        if real_count < batch_size:
            windows = np.concatenate((windows, np.repeat(windows[-1:], batch_size - real_count, axis=0)), axis=0)
        gripper_windows = np.stack(
            [gripper_closed[index : index + semantic_memory_event.WINDOW_SIZE] for index in range(start, stop)]
        )
        if real_count < batch_size:
            gripper_windows = np.concatenate(
                (gripper_windows, np.repeat(gripper_windows[-1:], batch_size - real_count, axis=0)),
                axis=0,
            )
        event_logits, type_logits, press_logits = classifier_apply(jnp.asarray(windows), jnp.asarray(gripper_windows))
        all_event_logits.append(np.asarray(event_logits)[:real_count])
        all_type_logits.append(np.asarray(type_logits)[:real_count])
        all_press_logits.append(np.asarray(press_logits)[:real_count])
    return np.concatenate(all_event_logits), np.concatenate(all_type_logits), np.concatenate(all_press_logits)


def memory_record(
    tracker,
    params,
    feature_episode: h5py.Group,
    metadata: dict[str, Any],
    detection_record: dict[str, Any],
    *,
    prompt_index: int,
    use_decoder_event_types: bool = False,
) -> dict[str, Any]:
    triggers = detection_record["trigger_starts"]
    selected = triggers[: semantic_memory_event.MAX_EVENTS]
    if not selected:
        selected = [0]
    padded = selected + [selected[0]] * (semantic_memory_event.MAX_EVENTS - len(selected))
    windows = np.stack(
        [feature_episode["patch_tokens"][start : start + semantic_memory_event.WINDOW_SIZE] for start in padded]
    )[None]
    gripper_timeline = np.asarray(metadata["gripper_closed"], dtype=np.bool_)
    gripper_windows = np.stack(
        [gripper_timeline[start : start + semantic_memory_event.WINDOW_SIZE] for start in padded]
    )[None]
    sequence_mask = np.arange(semantic_memory_event.MAX_EVENTS)[None] < min(
        len(triggers), semantic_memory_event.MAX_EVENTS
    )
    sequence_event_types = None
    if use_decoder_event_types:
        selected_types = detection_record["predicted_types"][: semantic_memory_event.MAX_EVENTS]
        if not selected_types:
            selected_types = [semantic_memory_event.PICK_COMPLETE]
        padded_types = selected_types + [selected_types[0]] * (semantic_memory_event.MAX_EVENTS - len(selected_types))
        sequence_event_types = jnp.asarray(padded_types, dtype=jnp.int32)[None]
    outputs = tracker.apply(
        {"params": params},
        jnp.asarray(windows),
        jnp.asarray(gripper_windows),
        jnp.asarray(feature_episode["prompt_tokens"][prompt_index])[None],
        jnp.asarray(feature_episode["prompt_mask"][prompt_index])[None],
        jnp.arange(semantic_memory_event.MAX_EVENTS, dtype=jnp.int32)[None],
        jnp.asarray(sequence_mask),
        causal_selection=False,
        candidate_valid_mask=jnp.asarray(sequence_mask),
        sequence_event_types=sequence_event_types,
        train=False,
    )
    outputs = jax.device_get(outputs)
    last_index = max(min(len(triggers), semantic_memory_event.MAX_EVENTS) - 1, 0)
    completed = np.argmax(outputs["completed_count_logits"][0], axis=-1)
    remaining = np.argmax(outputs["remaining_count_logits"][0], axis=-1)
    should_press = outputs["should_press_logits"][0] > 0
    done = outputs["done_logits"][0] > 0
    expected_states = [event["state_after"] for event in metadata["events"]]
    exact_count = len(triggers) == len(expected_states)
    compared = min(len(expected_states), semantic_memory_event.MAX_EVENTS)
    stage_state_correct = [
        int(completed[index]) == int(expected_states[index]["completed_count"])
        and int(remaining[index]) == int(expected_states[index]["remaining_count"])
        and bool(should_press[index]) == bool(expected_states[index]["should_press"])
        and bool(done[index]) == bool(expected_states[index]["done"])
        for index in range(compared)
    ]
    stage_states = [
        {
            "completed_count": int(completed[index]),
            "remaining_count": int(remaining[index]),
            "should_press": bool(should_press[index]),
            "done": bool(done[index]),
            "correct": bool(stage_state_correct[index]),
        }
        for index in range(compared)
    ]
    return {
        "prompt_index": prompt_index,
        "goal_color_correct": int(np.argmax(outputs["goal_color_logits"][0]))
        == {"red": 0, "green": 1, "blue": 2}[metadata["target_color"]],
        "goal_count_correct": int(np.argmax(outputs["goal_required_count_logits"][0]))
        == int(metadata["required_count"]) - 1,
        "final_completed_count": int(completed[last_index]),
        "final_remaining_count": int(remaining[last_index]),
        "final_should_press": bool(should_press[last_index]),
        "final_done": bool(done[last_index]),
        "final_count_correct": int(completed[last_index]) == int(metadata["required_count"]),
        "final_remaining_correct": int(remaining[last_index]) == 0,
        "final_done_correct": bool(done[last_index]),
        "stage_states": stage_states,
        "state_sequence_e2e_correct": exact_count and all(stage_state_correct),
        "strict_state_sequence_e2e_correct": detection_record["typed_event_sequence"] and all(stage_state_correct),
        "trigger_overflow": len(triggers) > semantic_memory_event.MAX_EVENTS,
    }


def main() -> None:
    args = parse_args()
    if args.window_batch_size < 1:
        raise ValueError("--window-batch-size must be positive")
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or args.run_dir / "checkpoints/step_2000.msgpack"
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["episodes"]
    metadata_by_index = {int(episode["episode_index"]): episode for episode in labels}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_indices = [int(value) for value in split["train_episode_indices"]]
    val_indices = [int(value) for value in split["val_episode_indices"]]
    test_indices = [int(value) for value in split.get("test_episode_indices", [])]
    calibration_indices = train_indices if args.calibration_split == "train" else val_indices
    evaluation_indices = val_indices if args.evaluation_split == "val" else test_indices
    if not evaluation_indices:
        raise ValueError(f"Split has no episodes for --evaluation-split={args.evaluation_split}")
    overlap = set(calibration_indices) & set(evaluation_indices)
    if overlap:
        raise ValueError(f"Calibration/evaluation episode leakage: {sorted(overlap)}")
    if args.max_train_episodes is not None:
        calibration_indices = calibration_indices[: args.max_train_episodes]
    if args.max_val_episodes is not None:
        evaluation_indices = evaluation_indices[: args.max_val_episodes]

    tracker = make_tracker(config)
    params = initialize_and_restore(tracker, checkpoint, args.window_classifier_checkpoint)
    classifier = semantic_memory_event.PickXtimesSlidingWindowEventClassifier(
        input_width=tracker.input_width,
        width=tracker.encoder_width,
        depth=tracker.encoder_depth,
        num_heads=tracker.encoder_heads,
        dtype_mm=tracker.dtype_mm,
    )
    classifier_params = params["window_classifier"]
    fusion = semantic_memory_event.PickXtimesGripperTypeFusion()
    fusion_params = params["gripper_type_fusion"]
    gate_fusion = semantic_memory_event.PickXtimesGripperGateFusion()
    gate_fusion_params = params["gripper_gate_fusion"]
    press_fusion = semantic_memory_event.PickXtimesPressGateFusion()
    press_fusion_params = params["press_gate_fusion"]

    @jax.jit
    def classifier_apply(windows, gripper_windows):
        pooled = memory_core.pool_fixed_grid(windows, pool_factor=2)
        visual_event_logits, visual_type_logits, semantic_features = classifier.apply(
            {"params": classifier_params}, pooled, train=False, return_features=True
        )
        event_logits = gate_fusion.apply({"params": gate_fusion_params}, visual_event_logits, gripper_windows)
        type_logits = fusion.apply({"params": fusion_params}, visual_type_logits, gripper_windows)
        press_logits = press_fusion.apply(
            {"params": press_fusion_params},
            event_logits,
            semantic_features,
        )
        return event_logits, type_logits, press_logits

    predictions = {}
    scan_indices = calibration_indices + evaluation_indices
    started = time.monotonic()
    with h5py.File(args.features, "r") as features:
        for ordinal, episode_index in enumerate(scan_indices, start=1):
            metadata = metadata_by_index[episode_index]
            logits, type_logits, press_logits = scan_episode_classifier(
                features[f"{metadata['episode_name']}/patch_tokens"],
                np.asarray(metadata["gripper_closed"], dtype=np.bool_),
                batch_size=args.window_batch_size,
                classifier_apply=classifier_apply,
            )
            predictions[episode_index] = (logits, type_logits, press_logits)
            print(
                f"scan {ordinal}/{len(scan_indices)} episode={episode_index} windows={len(logits)} "
                f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                flush=True,
            )

        transition_config = {
            "pick_place_probability": args.transition_pick_place_probability,
            "press_probability": args.transition_press_probability,
            "press_type_probability": args.transition_press_type_probability,
            "press_min_gap": args.transition_press_min_gap,
            "require_press_type": not args.transition_disable_press_type_gate,
        }

        def detection_record(
            index: int,
            high_probability: float = 0.7,
            low_probability: float = 0.3,
            transition_override: dict[str, Any] | None = None,
        ):
            return episode_detection_record(
                metadata_by_index[index],
                *predictions[index],
                decoder=args.decoder,
                high_probability=high_probability,
                low_probability=low_probability,
                transition_config=transition_override or transition_config,
            )

        if args.decoder == "hysteresis":
            high_probabilities = np.round(np.arange(0.30, 0.951, 0.05), 2).tolist()
            low_probabilities = np.round(np.arange(0.05, 0.751, 0.05), 2).tolist()
            sweep = []
            for high_probability in high_probabilities:
                for low_probability in low_probabilities:
                    if low_probability >= high_probability:
                        continue
                    records = [
                        detection_record(index, high_probability, low_probability) for index in calibration_indices
                    ]
                    metrics = aggregate_detection(records, metadata_by_index)
                    sweep.append(
                        {
                            "high_probability": high_probability,
                            "low_probability": low_probability,
                            "score": calibration_score(metrics),
                            **metrics,
                        }
                    )
            sweep.sort(
                key=lambda item: (
                    item["score"],
                    item["typed_event_sequence_accuracy"],
                    item["trigger_f1"],
                ),
                reverse=True,
            )
            calibrated = sweep[0]
            calibrated_train_records = [
                detection_record(index, calibrated["high_probability"], calibrated["low_probability"])
                for index in calibration_indices
            ]
        elif args.calibrate_transition_press:
            sweep = []
            for press_min_gap in range(20, 91, 5):
                for press_probability in np.round(np.arange(0.20, 0.901, 0.05), 2).tolist():
                    candidate_config = {
                        **transition_config,
                        "press_probability": press_probability,
                        "press_min_gap": press_min_gap,
                    }
                    records = [
                        detection_record(index, transition_override=candidate_config) for index in calibration_indices
                    ]
                    metrics = aggregate_detection(records, metadata_by_index)
                    sweep.append(
                        {
                            "high_probability": None,
                            "low_probability": None,
                            "transition_config": candidate_config,
                            "score": calibration_score(metrics),
                            **metrics,
                        }
                    )
            sweep.sort(
                key=lambda item: (
                    item["typed_event_sequence_accuracy"],
                    item["trigger_f1"],
                    item["exact_trigger_count_accuracy"],
                    -item["transition_config"]["press_min_gap"],
                ),
                reverse=True,
            )
            calibrated = sweep[0]
            calibrated_train_records = [
                detection_record(index, transition_override=calibrated["transition_config"])
                for index in calibration_indices
            ]
        else:
            calibrated_train_records = [detection_record(index) for index in calibration_indices]
            metrics = aggregate_detection(calibrated_train_records, metadata_by_index)
            calibrated = {
                "high_probability": None,
                "low_probability": None,
                "score": calibration_score(metrics),
                **metrics,
            }
            sweep = [calibrated]

        def evaluate_threshold(
            high_probability: float = 0.7,
            low_probability: float = 0.3,
            transition_override: dict[str, Any] | None = None,
        ):
            records = [
                detection_record(index, high_probability, low_probability, transition_override)
                for index in evaluation_indices
            ]
            return aggregate_detection(records, metadata_by_index), records

        default_metrics, default_records = evaluate_threshold()
        if args.decoder == "hysteresis":
            calibrated_metrics, calibrated_records = evaluate_threshold(
                calibrated["high_probability"], calibrated["low_probability"]
            )
        elif args.calibrate_transition_press:
            calibrated_metrics, calibrated_records = evaluate_threshold(
                transition_override=calibrated["transition_config"]
            )
        else:
            calibrated_metrics, calibrated_records = default_metrics, default_records

        def evaluate_memory(detection_records):
            records = []
            for detection_record in detection_records:
                episode_index = detection_record["episode_index"]
                metadata = metadata_by_index[episode_index]
                feature_episode = features[metadata["episode_name"]]
                prompt_records = [
                    memory_record(
                        tracker,
                        params,
                        feature_episode,
                        metadata,
                        detection_record,
                        prompt_index=prompt_index,
                        use_decoder_event_types=args.decoder == "transition_grammar",
                    )
                    for prompt_index in range(len(metadata["prompts"]))
                ]
                records.append({"episode_index": episode_index, "prompts": prompt_records})
            return records

        calibrated_memory_records = evaluate_memory(calibrated_records)
        default_memory_records = evaluate_memory(default_records)

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

    memory_metrics = aggregate_memory(calibrated_memory_records)
    default_memory_metrics = aggregate_memory(default_memory_records)
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "window_classifier_checkpoint": (
            str(args.window_classifier_checkpoint.resolve()) if args.window_classifier_checkpoint is not None else None
        ),
        "feature_h5": str(args.features.resolve()),
        "calibration_split": args.calibration_split,
        "evaluation_split": args.evaluation_split,
        "decoder": args.decoder,
        "decoder_config": transition_config if args.decoder == "transition_grammar" else None,
        "calibration_episode_indices": calibration_indices,
        "evaluation_episode_indices": evaluation_indices,
        # Backward-compatible name used by the earlier two-way split reports.
        "validation_episode_indices": evaluation_indices,
        "strict_match_definition": "trigger start must be in event.positive_starts",
        "default_threshold": {
            "high_probability": 0.70 if args.decoder == "hysteresis" else None,
            "low_probability": 0.30 if args.decoder == "hysteresis" else None,
            "metrics": default_metrics,
            "episodes": default_records,
            "memory_metrics_two_prompts_per_episode": default_memory_metrics,
            "memory_episodes": default_memory_records,
        },
        "calibrated_threshold": {
            "high_probability": calibrated["high_probability"],
            "low_probability": calibrated["low_probability"],
            "transition_config": calibrated.get("transition_config"),
            "calibration_metrics": {key: value for key, value in calibrated.items() if key != "episodes"},
            "calibration_episodes": calibrated_train_records,
            "validation_metrics": calibrated_metrics,
            "episodes": calibrated_records,
        },
        "memory_metrics_two_prompts_per_episode": memory_metrics,
        "memory_episodes": calibrated_memory_records,
        "calibration_top10": sweep[:10],
    }
    if args.save_window_logits:
        result["window_predictions"] = [
            {
                "episode_index": int(index),
                "event_logits": np.asarray(predictions[index][0], dtype=np.float32).tolist(),
                "event_type_logits": np.asarray(predictions[index][1], dtype=np.float32).tolist(),
                "press_event_logits": np.asarray(predictions[index][2], dtype=np.float32).tolist(),
            }
            for index in scan_indices
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "calibrated_high_probability": calibrated["high_probability"],
                "calibrated_low_probability": calibrated["low_probability"],
                "calibrated_validation_metrics": calibrated_metrics,
                "default_validation_metrics": default_metrics,
                "memory_metrics": memory_metrics,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
