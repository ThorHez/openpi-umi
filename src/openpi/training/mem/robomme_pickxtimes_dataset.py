"""HDF5 window dataset for PickXtimes event-memory pretraining."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import h5py
import numpy as np
import torch

from openpi.tasks.robomme.pickxtimes import semantic_memory_event

COLOR_TO_ID = {"red": 0, "green": 1, "blue": 2}


class PickXtimesWindowDataset(torch.utils.data.Dataset):
    """Return sampled classifier windows plus a chronological event sequence.

    The first ``MAX_EVENTS`` candidate slots contain one positive window for
    each real event in chronological order and padding thereafter.  Six
    negative candidates follow: three transition-adjacent hard negatives and
    three ordinary negatives.  ``sequence_positions`` therefore has a stable
    fixed shape while ``sequence_mask`` prevents padded recurrent updates.
    """

    def __init__(
        self,
        h5_path: str | pathlib.Path,
        labels_path: str | pathlib.Path,
        *,
        feature_h5_path: str | pathlib.Path | None = None,
        causal_records_path: str | pathlib.Path | None = None,
        episode_indices: list[int] | None = None,
        random_seed: int = 0,
        randomize: bool = True,
        gate_neighborhood_sampling: bool = False,
        gate_logit_records_path: str | pathlib.Path | None = None,
        distillation_records_path: str | pathlib.Path | None = None,
        use_decoded_sequence_event_types: bool = False,
        memory_only: bool = False,
        press_gate_sampling: bool = False,
    ):
        self.h5_path = pathlib.Path(h5_path).expanduser().resolve()
        self.labels_path = pathlib.Path(labels_path).expanduser().resolve()
        if not self.h5_path.is_file():
            raise FileNotFoundError(self.h5_path)
        if not self.labels_path.is_file():
            raise FileNotFoundError(self.labels_path)
        self.feature_h5_path = (
            pathlib.Path(feature_h5_path).expanduser().resolve() if feature_h5_path is not None else None
        )
        if self.feature_h5_path is not None and not self.feature_h5_path.is_file():
            raise FileNotFoundError(self.feature_h5_path)
        self.causal_records_path = (
            pathlib.Path(causal_records_path).expanduser().resolve() if causal_records_path is not None else None
        )
        self.causal_records: dict[int, dict[str, Any]] | None = None
        if self.causal_records_path is not None:
            causal_payload = json.loads(self.causal_records_path.read_text(encoding="utf-8"))
            threshold_payload = causal_payload["calibrated_threshold"]
            records = threshold_payload.get("calibration_episodes", []) + threshold_payload["episodes"]
            self.causal_records = {int(record["episode_index"]): record for record in records}
        self.gate_logits: dict[int, np.ndarray] | None = None
        if gate_logit_records_path is not None:
            logit_path = pathlib.Path(gate_logit_records_path).expanduser().resolve()
            if not logit_path.is_file():
                raise FileNotFoundError(logit_path)
            logit_payload = json.loads(logit_path.read_text(encoding="utf-8"))
            if "window_predictions" not in logit_payload:
                raise ValueError(f"No window_predictions in {logit_path}")
            self.gate_logits = {
                int(record["episode_index"]): np.asarray(record["event_logits"], dtype=np.float32)
                for record in logit_payload["window_predictions"]
            }
        self.distillation_logits: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        if distillation_records_path is not None:
            distill_path = pathlib.Path(distillation_records_path).expanduser().resolve()
            if not distill_path.is_file():
                raise FileNotFoundError(distill_path)
            distill_payload = json.loads(distill_path.read_text(encoding="utf-8"))
            records = distill_payload.get("window_predictions")
            if records is None or any("event_type_logits" not in record for record in records):
                raise ValueError(f"Distillation records require event and event-type logits: {distill_path}")
            self.distillation_logits = {
                int(record["episode_index"]): (
                    np.asarray(record["event_logits"], dtype=np.float32),
                    np.asarray(record["event_type_logits"], dtype=np.float32),
                )
                for record in records
            }
        payload = json.loads(self.labels_path.read_text(encoding="utf-8"))
        if int(payload["window_size"]) != semantic_memory_event.WINDOW_SIZE:
            raise ValueError(
                f"Label window_size={payload['window_size']} does not match model "
                f"window_size={semantic_memory_event.WINDOW_SIZE}"
            )
        by_index = {int(episode["episode_index"]): episode for episode in payload["episodes"]}
        selected = sorted(by_index) if episode_indices is None else list(episode_indices)
        missing = [index for index in selected if index not in by_index]
        if missing:
            raise ValueError(f"Unknown episode indices: {missing[:10]}")
        if self.causal_records is not None:
            missing_records = [index for index in selected if index not in self.causal_records]
            if missing_records:
                raise ValueError(f"Missing causal records for episode indices: {missing_records[:10]}")
        if self.gate_logits is not None:
            missing_logits = [index for index in selected if index not in self.gate_logits]
            if missing_logits:
                raise ValueError(f"Missing gate logits for episode indices: {missing_logits[:10]}")
        if self.distillation_logits is not None:
            missing_distillation = [index for index in selected if index not in self.distillation_logits]
            if missing_distillation:
                raise ValueError(f"Missing distillation records for episode indices: {missing_distillation[:10]}")
        self.episodes = [by_index[index] for index in selected]
        self.random_seed = int(random_seed)
        self.randomize = bool(randomize)
        self.gate_neighborhood_sampling = bool(gate_neighborhood_sampling)
        self.use_decoded_sequence_event_types = bool(use_decoded_sequence_event_types)
        self.memory_only = bool(memory_only)
        self.press_gate_sampling = bool(press_gate_sampling)
        self._h5: h5py.File | None = None
        self._feature_h5: h5py.File | None = None
        self._calls = np.zeros((len(self.episodes),), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.episodes)

    def _handle(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _feature_handle(self) -> h5py.File:
        if self.feature_h5_path is None:
            raise RuntimeError("feature_h5_path was not configured")
        if self._feature_h5 is None:
            self._feature_h5 = h5py.File(self.feature_h5_path, "r")
        return self._feature_h5

    def _rng(self, item: int) -> np.random.Generator:
        call = int(self._calls[item]) if self.randomize else 0
        self._calls[item] += int(self.randomize)
        episode_index = int(self.episodes[item]["episode_index"])
        return np.random.default_rng(np.random.SeedSequence([self.random_seed, episode_index, call]))

    @staticmethod
    def _sample_pool(rng: np.random.Generator, values: list[int], count: int) -> list[int]:
        if not values:
            raise ValueError("Cannot sample an empty window pool")
        return rng.choice(values, size=count, replace=len(values) < count).astype(int).tolist()

    def _read_windows(self, episode_name: str, starts: list[int]) -> np.ndarray:
        if self.feature_h5_path is not None:
            features = self._feature_handle()[f"{episode_name}/patch_tokens"]
            return np.stack(
                [features[start : start + semantic_memory_event.WINDOW_SIZE] for start in starts],
                axis=0,
            )
        episode = self._handle()[episode_name]
        frames: dict[int, np.ndarray] = {}
        for start in starts:
            for index in range(start, start + semantic_memory_event.WINDOW_SIZE):
                if index not in frames:
                    frames[index] = episode[f"timestep_{index}/obs/front_rgb"][()]
        return np.stack(
            [
                np.stack(
                    [frames[index] for index in range(start, start + semantic_memory_event.WINDOW_SIZE)],
                    axis=0,
                )
                for start in starts
            ],
            axis=0,
        )

    @staticmethod
    def _read_gripper_windows(episode: dict[str, Any], starts: list[int]) -> np.ndarray:
        if "gripper_closed" not in episode:
            raise ValueError("Labels do not contain gripper_closed; rebuild them with the schema-v3 label builder")
        timeline = np.asarray(episode["gripper_closed"], dtype=np.bool_)
        return np.stack(
            [timeline[start : start + semantic_memory_event.WINDOW_SIZE] for start in starts],
            axis=0,
        )

    def _add_distillation_targets(
        self,
        sample: dict[str, Any],
        episode_index: int,
        starts: list[int],
        valid_mask: np.ndarray,
    ) -> None:
        if self.distillation_logits is None:
            sample["teacher_event_logits"] = np.zeros((len(starts),), dtype=np.float32)
            sample["teacher_event_type_logits"] = np.zeros(
                (len(starts), semantic_memory_event.NUM_EVENT_CLASSES), dtype=np.float32
            )
            sample["teacher_distillation_mask"] = np.zeros((len(starts),), dtype=np.bool_)
            return
        event_logits, event_type_logits = self.distillation_logits[episode_index]
        start_array = np.asarray(starts, dtype=np.int32)
        if np.any(start_array < 0) or np.any(start_array >= len(event_logits)):
            raise ValueError(f"Out-of-range distillation start for episode_index={episode_index}")
        sample["teacher_event_logits"] = event_logits[start_array]
        sample["teacher_event_type_logits"] = event_type_logits[start_array]
        sample["teacher_distillation_mask"] = np.asarray(valid_mask, dtype=np.bool_)

    def __getitem__(self, item: int) -> dict[str, Any]:
        episode = self.episodes[item]
        rng = self._rng(item)
        events = episode["events"]
        if len(events) > semantic_memory_event.MAX_EVENTS:
            raise ValueError(
                f"episode_index={episode['episode_index']} has {len(events)} events, "
                f"exceeding MAX_EVENTS={semantic_memory_event.MAX_EVENTS}"
            )

        if self.press_gate_sampling:
            return self._press_gate_item(episode, events, rng)

        if self.causal_records is not None:
            return self._causal_item(episode, events, rng)

        positive_starts = [int(rng.choice(event["positive_starts"])) for event in events]
        num_events = len(events)
        padded_positive_starts = positive_starts + [positive_starts[0]] * (
            semantic_memory_event.MAX_EVENTS - num_events
        )
        hard_starts = self._sample_pool(rng, episode["hard_negative_starts"], 3)
        ordinary_starts = self._sample_pool(rng, episode["ordinary_negative_starts"], 3)
        candidate_starts = padded_positive_starts + hard_starts + ordinary_starts

        candidate_valid_mask = np.ones((len(candidate_starts),), dtype=np.bool_)
        candidate_valid_mask[num_events : semantic_memory_event.MAX_EVENTS] = False
        event_targets = np.zeros((len(candidate_starts),), dtype=np.float32)
        event_targets[:num_events] = 1.0
        event_type_targets = np.zeros((len(candidate_starts),), dtype=np.int32)
        event_type_targets[:num_events] = [int(event["event_type_id"]) for event in events]
        event_type_mask = event_targets.astype(np.bool_)

        sequence_mask = np.arange(semantic_memory_event.MAX_EVENTS) < num_events
        state_fields = (
            "completed_count",
            "holding",
            "remaining_count",
            "should_press",
            "done",
        )
        state_targets: dict[str, np.ndarray] = {}
        for field in state_fields:
            dtype = np.bool_ if field in {"holding", "should_press", "done"} else np.int32
            values = [event["state_after"][field] for event in events]
            state_targets[field] = np.asarray(
                values + [0] * (semantic_memory_event.MAX_EVENTS - num_events), dtype=dtype
            )
        next_event = np.full((semantic_memory_event.MAX_EVENTS,), semantic_memory_event.PRESS_COMPLETE, dtype=np.int32)
        next_event_mask = sequence_mask & ~state_targets["done"]
        for index in range(num_events):
            if state_targets["holding"][index]:
                next_event[index] = semantic_memory_event.PLACE_COMPLETE
            elif state_targets["remaining_count"][index] > 0:
                next_event[index] = semantic_memory_event.PICK_COMPLETE

        prompt_index = int(rng.integers(len(episode["prompts"]))) if self.randomize else 0
        window_key = "window_patch_tokens" if self.feature_h5_path is not None else "front_windows"
        sample = {
            "episode_index": np.int32(episode["episode_index"]),
            "prompt": episode["prompts"][prompt_index],
            "goal_color": np.int32(COLOR_TO_ID[episode["target_color"]]),
            "goal_required_count": np.int32(int(episode["required_count"]) - 1),
            window_key: self._read_windows(episode["episode_name"], candidate_starts),
            "window_gripper_closed": self._read_gripper_windows(episode, candidate_starts),
            "candidate_starts": np.asarray(candidate_starts, dtype=np.int32),
            "candidate_valid_mask": candidate_valid_mask,
            "event_targets": event_targets,
            "event_type_targets": event_type_targets,
            "event_type_mask": event_type_mask,
            "sequence_positions": np.arange(semantic_memory_event.MAX_EVENTS, dtype=np.int32),
            "sequence_event_types": event_type_targets[: semantic_memory_event.MAX_EVENTS],
            "sequence_mask": sequence_mask,
            "completed_count_targets": state_targets["completed_count"],
            "holding_targets": state_targets["holding"],
            "remaining_count_targets": state_targets["remaining_count"],
            "should_press_targets": state_targets["should_press"],
            "done_targets": state_targets["done"],
            "next_event_targets": next_event,
            "next_event_mask": next_event_mask,
        }
        if self.feature_h5_path is not None:
            feature_episode = self._feature_handle()[episode["episode_name"]]
            sample["prompt_tokens"] = feature_episode["prompt_tokens"][prompt_index]
            sample["prompt_mask"] = feature_episode["prompt_mask"][prompt_index]
        self._add_distillation_targets(sample, int(episode["episode_index"]), candidate_starts, candidate_valid_mask)
        return sample

    def _press_gate_item(
        self,
        episode: dict[str, Any],
        events: list[dict[str, Any]],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Sample PRESS positives and post-goal hard negatives."""
        press_event = events[-1]
        if int(press_event["event_type_id"]) != semantic_memory_event.PRESS_COMPLETE:
            raise ValueError(f"episode_index={episode['episode_index']} does not end with PRESS")
        final_place = events[-2]
        positive_pool = [int(value) for value in press_event["positive_starts"]]
        positive_starts = self._sample_pool(rng, positive_pool, 1)
        max_start = int(episode["num_steps"]) - semantic_memory_event.WINDOW_SIZE
        positive_set = set(positive_pool)

        boundary_pool = [max(min(positive_pool) - 1, 0), min(max(positive_pool) + 1, max_start)]
        approach_pool = [
            start for start in range(int(final_place["anchor"]) + 1, min(positive_pool)) if start not in positive_set
        ]
        post_press_pool = [start for start in range(max(positive_pool) + 1, max_start + 1) if start not in positive_set]
        if not approach_pool:
            approach_pool = [boundary_pool[0]]
        if not post_press_pool:
            post_press_pool = [boundary_pool[-1]]

        predicted_false_pool: list[int] = []
        if self.causal_records is not None:
            record = self.causal_records[int(episode["episode_index"])]
            predicted_false_pool = [
                int(start)
                for start, event_type in zip(record["trigger_starts"], record["predicted_types"], strict=True)
                if int(event_type) == semantic_memory_event.PRESS_COMPLETE and int(start) not in positive_set
            ]
        if not predicted_false_pool:
            predicted_false_pool = approach_pool

        negative_starts = (
            boundary_pool
            + self._sample_pool(rng, approach_pool, 2)
            + self._sample_pool(rng, post_press_pool, 1)
            + self._sample_pool(rng, predicted_false_pool, 1)
            + self._sample_pool(rng, episode["ordinary_negative_starts"], 1)
        )
        candidate_starts = positive_starts + negative_starts
        num_candidates = len(candidate_starts)
        event_targets = np.zeros((num_candidates,), dtype=np.float32)
        event_targets[: len(positive_starts)] = 1.0
        event_type_targets = np.full(
            (num_candidates,),
            semantic_memory_event.PRESS_COMPLETE,
            dtype=np.int32,
        )
        sequence_mask = np.zeros((semantic_memory_event.MAX_EVENTS,), dtype=np.bool_)
        prompt_index = int(rng.integers(len(episode["prompts"]))) if self.randomize else 0
        window_key = "window_patch_tokens" if self.feature_h5_path is not None else "front_windows"
        sample = {
            "episode_index": np.int32(episode["episode_index"]),
            "prompt": episode["prompts"][prompt_index],
            "goal_color": np.int32(COLOR_TO_ID[episode["target_color"]]),
            "goal_required_count": np.int32(int(episode["required_count"]) - 1),
            window_key: self._read_windows(episode["episode_name"], candidate_starts),
            "window_gripper_closed": self._read_gripper_windows(episode, candidate_starts),
            "candidate_starts": np.asarray(candidate_starts, dtype=np.int32),
            "candidate_valid_mask": np.ones((num_candidates,), dtype=np.bool_),
            "event_targets": event_targets,
            "event_type_targets": event_type_targets,
            "event_type_mask": event_targets.astype(np.bool_),
            "sequence_positions": np.zeros((semantic_memory_event.MAX_EVENTS,), dtype=np.int32),
            "sequence_event_types": np.zeros((semantic_memory_event.MAX_EVENTS,), dtype=np.int32),
            "sequence_mask": sequence_mask,
            "completed_count_targets": np.zeros((semantic_memory_event.MAX_EVENTS,), dtype=np.int32),
            "holding_targets": sequence_mask.copy(),
            "remaining_count_targets": np.zeros((semantic_memory_event.MAX_EVENTS,), dtype=np.int32),
            "should_press_targets": sequence_mask.copy(),
            "done_targets": sequence_mask.copy(),
            "next_event_targets": np.full(
                (semantic_memory_event.MAX_EVENTS,),
                semantic_memory_event.PRESS_COMPLETE,
                dtype=np.int32,
            ),
            "next_event_mask": sequence_mask.copy(),
        }
        if self.feature_h5_path is not None:
            feature_episode = self._feature_handle()[episode["episode_name"]]
            sample["prompt_tokens"] = feature_episode["prompt_tokens"][prompt_index]
            sample["prompt_mask"] = feature_episode["prompt_mask"][prompt_index]
        self._add_distillation_targets(
            sample,
            int(episode["episode_index"]),
            candidate_starts,
            sample["candidate_valid_mask"],
        )
        return sample

    @staticmethod
    def _state_before_events(episode: dict[str, Any]) -> dict[str, Any]:
        return {
            "completed_count": 0,
            "holding": False,
            "remaining_count": int(episode["required_count"]),
            "should_press": False,
            "done": False,
        }

    @classmethod
    def _state_at_window_end(
        cls,
        episode: dict[str, Any],
        events: list[dict[str, Any]],
        window_start: int,
    ) -> dict[str, Any]:
        state = cls._state_before_events(episode)
        window_end = window_start + semantic_memory_event.WINDOW_SIZE - 1
        for event in events:
            if int(event["anchor"]) <= window_end:
                state = event["state_after"]
        return state

    @classmethod
    def states_from_event_types(
        cls,
        episode: dict[str, Any],
        event_types: list[int],
    ) -> list[dict[str, Any]]:
        """Apply the task state machine to decoder-accepted semantic events."""
        state = cls._state_before_events(episode)
        states = []
        for event_type in event_types:
            state = dict(state)
            if event_type == semantic_memory_event.PICK_COMPLETE:
                state["holding"] = True
            elif event_type == semantic_memory_event.PLACE_COMPLETE:
                state["holding"] = False
                state["completed_count"] = min(
                    int(state["completed_count"]) + 1,
                    int(episode["required_count"]),
                )
                state["remaining_count"] = int(episode["required_count"]) - int(state["completed_count"])
                state["should_press"] = state["remaining_count"] == 0
            elif event_type == semantic_memory_event.PRESS_COMPLETE:
                state["should_press"] = False
                state["done"] = True
            else:
                raise ValueError(f"Unknown PickXtimes event type: {event_type}")
            states.append(state)
        return states

    def _causal_item(
        self,
        episode: dict[str, Any],
        events: list[dict[str, Any]],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Train memory on the previous model's own causal trigger sequence.

        Candidate slots contain predicted triggers first, all teacher positive
        events second, and ordinary negatives last. Predicted false triggers
        are consequently explicit event-gate negatives while still updating
        recurrent memory with the state target at their causal timestamp.
        """
        assert self.causal_records is not None
        record = self.causal_records[int(episode["episode_index"])]
        predicted_starts = [int(value) for value in record["trigger_starts"]][: semantic_memory_event.MAX_EVENTS]
        num_predicted = len(predicted_starts)
        predicted_types = [int(value) for value in record["predicted_types"]][: semantic_memory_event.MAX_EVENTS]
        if len(predicted_types) != num_predicted:
            raise ValueError(
                f"episode_index={episode['episode_index']} has {num_predicted} trigger starts "
                f"but {len(predicted_types)} predicted types"
            )
        padding_start = predicted_starts[0] if predicted_starts else 0
        padded_predicted = predicted_starts + [padding_start] * (semantic_memory_event.MAX_EVENTS - num_predicted)
        padding_type = predicted_types[0] if predicted_types else semantic_memory_event.PICK_COMPLETE
        padded_predicted_types = predicted_types + [padding_type] * (semantic_memory_event.MAX_EVENTS - num_predicted)

        positive_starts = [int(rng.choice(event["positive_starts"])) for event in events]
        num_events = len(positive_starts)
        padded_positive = positive_starts + [positive_starts[0]] * (semantic_memory_event.MAX_EVENTS - num_events)
        predicted_valid = np.arange(semantic_memory_event.MAX_EVENTS) < num_predicted
        positive_valid = np.arange(semantic_memory_event.MAX_EVENTS) < num_events
        if self.memory_only:
            candidate_starts = padded_predicted
            candidate_valid_mask = predicted_valid
        elif self.gate_logits is not None:
            logits = self.gate_logits[int(episode["episode_index"])]
            expected_windows = int(episode["num_steps"]) - semantic_memory_event.WINDOW_SIZE + 1
            if logits.shape != (expected_windows,):
                raise ValueError(
                    f"episode_index={episode['episode_index']} logits shape={logits.shape}, "
                    f"expected {(expected_windows,)}"
                )
            positive_lookup = {int(start) for event in events for start in event["positive_starts"]}
            hard_positives = [
                min((int(start) for start in event["positive_starts"]), key=lambda start: logits[start])
                for event in events
            ]
            padded_hard_positives = hard_positives + [hard_positives[0]] * (
                semantic_memory_event.MAX_EVENTS - num_events
            )
            teacher_extras = []
            boundary_negatives = []
            for index in range(semantic_memory_event.MAX_EVENTS):
                if index < num_events:
                    pool = [int(value) for value in events[index]["positive_starts"]]
                    teacher_extras.extend(self._sample_pool(rng, pool, 2))
                    boundary_negatives.extend((max(min(pool) - 1, 0), min(max(pool) + 1, expected_windows - 1)))
                else:
                    teacher_extras.extend((positive_starts[0], positive_starts[0]))
                    boundary_negatives.extend((positive_starts[0], positive_starts[0]))

            strict_distances = np.full((expected_windows,), expected_windows, dtype=np.int32)
            for start in positive_lookup:
                strict_distances = np.minimum(strict_distances, np.abs(np.arange(expected_windows) - start))
            ranked_background = np.argsort(-logits)
            top_background = []
            for start in ranked_background.astype(int).tolist():
                if strict_distances[start] <= 3:
                    continue
                if any(abs(start - selected_start) <= 2 for selected_start in top_background):
                    continue
                top_background.append(start)
                if len(top_background) == 12:
                    break
            if len(top_background) < 12:
                raise ValueError(
                    f"episode_index={episode['episode_index']} has only {len(top_background)} mined negatives"
                )
            ordinary_starts = self._sample_pool(rng, episode["ordinary_negative_starts"], 3)
            candidate_starts = (
                padded_predicted
                + padded_hard_positives
                + teacher_extras
                + top_background
                + boundary_negatives
                + ordinary_starts
            )
            candidate_valid_mask = np.concatenate(
                (
                    predicted_valid,
                    positive_valid,
                    np.repeat(positive_valid, 2),
                    np.ones((12,), dtype=np.bool_),
                    np.repeat(positive_valid, 2),
                    np.ones((3,), dtype=np.bool_),
                )
            )
        elif self.gate_neighborhood_sampling:
            max_start = int(episode["num_steps"]) - semantic_memory_event.WINDOW_SIZE
            neighbor_offsets = np.asarray((-3, -2, -1, 1, 2, 3), dtype=np.int32)
            predicted_neighbors = []
            for index, start in enumerate(padded_predicted):
                if index < num_predicted:
                    offsets = rng.choice(neighbor_offsets, size=2, replace=False)
                    predicted_neighbors.extend(int(np.clip(start + int(offset), 0, max_start)) for offset in offsets)
                else:
                    predicted_neighbors.extend((padding_start, padding_start))

            teacher_extras = []
            for index in range(semantic_memory_event.MAX_EVENTS):
                if index < num_events:
                    pool = [int(value) for value in events[index]["positive_starts"]]
                    teacher_extras.extend(self._sample_pool(rng, pool, 2))
                else:
                    teacher_extras.extend((positive_starts[0], positive_starts[0]))

            hard_starts = self._sample_pool(rng, episode["hard_negative_starts"], 6)
            ordinary_starts = self._sample_pool(rng, episode["ordinary_negative_starts"], 3)
            candidate_starts = (
                padded_predicted
                + padded_positive
                + predicted_neighbors
                + teacher_extras
                + hard_starts
                + ordinary_starts
            )
            candidate_valid_mask = np.concatenate(
                (
                    predicted_valid,
                    positive_valid,
                    np.repeat(predicted_valid, 2),
                    np.repeat(positive_valid, 2),
                    np.ones((9,), dtype=np.bool_),
                )
            )
        else:
            ordinary_starts = self._sample_pool(rng, episode["ordinary_negative_starts"], 3)
            candidate_starts = padded_predicted + padded_positive + ordinary_starts
            candidate_valid_mask = np.concatenate((predicted_valid, positive_valid, np.ones((3,), dtype=np.bool_)))

        positive_lookup = {
            int(start): int(event["event_type_id"]) for event in events for start in event["positive_starts"]
        }
        event_targets = np.asarray([float(start in positive_lookup) for start in candidate_starts], dtype=np.float32)
        event_type_targets = np.asarray(
            [positive_lookup.get(start, semantic_memory_event.PICK_COMPLETE) for start in candidate_starts],
            dtype=np.int32,
        )
        event_type_mask = (event_targets > 0) & candidate_valid_mask

        sequence_mask = np.arange(semantic_memory_event.MAX_EVENTS) < num_predicted
        if self.use_decoded_sequence_event_types:
            predicted_states = self.states_from_event_types(episode, predicted_types)
        else:
            predicted_states = [self._state_at_window_end(episode, events, start) for start in predicted_starts]
        padded_states = predicted_states + [self._state_before_events(episode)] * (
            semantic_memory_event.MAX_EVENTS - num_predicted
        )
        state_targets = {
            field: np.asarray(
                [state[field] for state in padded_states],
                dtype=np.bool_ if field in {"holding", "should_press", "done"} else np.int32,
            )
            for field in ("completed_count", "holding", "remaining_count", "should_press", "done")
        }
        next_event = np.full(
            (semantic_memory_event.MAX_EVENTS,),
            semantic_memory_event.PRESS_COMPLETE,
            dtype=np.int32,
        )
        next_event_mask = sequence_mask & ~state_targets["done"]
        for index in range(num_predicted):
            if state_targets["holding"][index]:
                next_event[index] = semantic_memory_event.PLACE_COMPLETE
            elif state_targets["remaining_count"][index] > 0:
                next_event[index] = semantic_memory_event.PICK_COMPLETE

        prompt_index = int(rng.integers(len(episode["prompts"]))) if self.randomize else 0
        window_key = "window_patch_tokens" if self.feature_h5_path is not None else "front_windows"
        sample = {
            "episode_index": np.int32(episode["episode_index"]),
            "prompt": episode["prompts"][prompt_index],
            "goal_color": np.int32(COLOR_TO_ID[episode["target_color"]]),
            "goal_required_count": np.int32(int(episode["required_count"]) - 1),
            window_key: self._read_windows(episode["episode_name"], candidate_starts),
            "window_gripper_closed": self._read_gripper_windows(episode, candidate_starts),
            "candidate_starts": np.asarray(candidate_starts, dtype=np.int32),
            "candidate_valid_mask": candidate_valid_mask,
            "event_targets": event_targets,
            "event_type_targets": event_type_targets,
            "event_type_mask": event_type_mask,
            "sequence_positions": np.arange(semantic_memory_event.MAX_EVENTS, dtype=np.int32),
            "sequence_event_types": np.asarray(padded_predicted_types, dtype=np.int32),
            "sequence_mask": sequence_mask,
            "completed_count_targets": state_targets["completed_count"],
            "holding_targets": state_targets["holding"],
            "remaining_count_targets": state_targets["remaining_count"],
            "should_press_targets": state_targets["should_press"],
            "done_targets": state_targets["done"],
            "next_event_targets": next_event,
            "next_event_mask": next_event_mask,
        }
        if self.feature_h5_path is not None:
            feature_episode = self._feature_handle()[episode["episode_name"]]
            sample["prompt_tokens"] = feature_episode["prompt_tokens"][prompt_index]
            sample["prompt_mask"] = feature_episode["prompt_mask"][prompt_index]
        self._add_distillation_targets(sample, int(episode["episode_index"]), candidate_starts, candidate_valid_mask)
        return sample

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()
        if self._feature_h5 is not None:
            self._feature_h5.close()
