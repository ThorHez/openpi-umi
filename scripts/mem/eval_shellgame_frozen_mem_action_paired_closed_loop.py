#!/usr/bin/env python3
"""Paired closed-loop test of frozen teacher/direct/wrong/zero MEM tokens.

Each condition reconstructs the same recorded ShellGame episode.  A private
seed sent with every policy query fixes the diffusion noise by
``(episode, query_index)``; therefore memory is the only policy input changed
at a matched query.  Selection/approach metrics are reported separately from
lift success so low-level grasp failures do not hide memory-conditioned target
choice.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "shellgame"))

import numpy as np
from openpi_client import websocket_client_policy

from examples.shellgame.serve_old_tracker_full_absolute_eef_deterministic import NOISE_SEED_KEY
from openpi.tasks.shellgame import qwenvl_event_adapter
from scripts.mem import eval_shellgame_qwen_event_pi_action_closed_loop as base


DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi0_shellgame_qwen_event_memory_action_eef7_260825/"
    "direct_visual_mem_step999_filtered_action250_6gpu_260825/249"
)
DEFAULT_DIRECT_MEMORY = Path(
    "artifacts/shellgame_qwen_distilled_direct_visual_memory_step999_all5000_260825.npz"
)
DEFAULT_TEACHER_MEMORY = Path("artifacts/shellgame_qwen_event_final_memory_v1_260825.npz")
DEFAULT_OUTPUT = Path(
    "evaluation/shellgame/frozen_mem_action_paired_closed_loop6_260826/result.json"
)
DEFAULT_VIDEO_DIR = Path(
    "evaluation/shellgame/frozen_mem_action_paired_closed_loop6_260826/videos"
)
DEFAULT_EPISODES = "8,31,16,47,80,195"
CONDITIONS = ("teacher", "direct_visual", "wrong_visual", "zero")


class FixedNoiseRemotePolicy:
    """Attach a reproducible per-query diffusion seed to remote observations."""

    def __init__(self, host: str, port: int, *, salt: int):
        self._policy = websocket_client_policy.WebsocketClientPolicy(host, port)
        self._salt = int(salt)
        self._episode: int | None = None
        self._query_index = 0
        metadata = self._policy.get_server_metadata()
        if metadata.get("deterministic_noise_seed_key") != NOISE_SEED_KEY:
            raise RuntimeError("Remote server does not advertise deterministic-noise support")

    def start_episode(self, episode: int) -> None:
        self._episode = int(episode)
        self._query_index = 0

    @property
    def query_count(self) -> int:
        return self._query_index

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._episode is None:
            raise RuntimeError("start_episode must be called before infer")
        seed = int(
            np.random.SeedSequence([self._salt, self._episode, self._query_index])
            .generate_state(1, dtype=np.uint32)[0]
        )
        self._query_index += 1
        request = dict(observation)
        request[NOISE_SEED_KEY] = seed
        return self._policy.infer(request)

    def close(self) -> None:
        self._policy._ws.close()  # noqa: SLF001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--raw-root", type=Path, default=base.DEFAULT_RAW_ROOT)
    parser.add_argument("--direct-memory", type=Path, default=DEFAULT_DIRECT_MEMORY)
    parser.add_argument("--teacher-memory", type=Path, default=DEFAULT_TEACHER_MEMORY)
    parser.add_argument("--episodes", default=DEFAULT_EPISODES)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--prompt", default=base.PROMPT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8027)
    parser.add_argument("--noise-salt", type=int, default=260826)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--max-policy-steps", type=int, default=95)
    parser.add_argument("--selection-skip", type=int, default=10)
    parser.add_argument("--selection-window", type=int, default=30)
    parser.add_argument("--selection-radius", type=float, default=0.06)
    parser.add_argument("--precision-radius", type=float, default=0.03)
    parser.add_argument("--lift-success-height", type=float, default=0.08)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument(
        "--allow-incorrect-direct-memory",
        action="store_true",
        help="Evaluate the full direct-memory population instead of restricting the interface probe to semantically correct memories.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_direct(path: Path) -> dict[str, np.ndarray | dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        episode = np.asarray(source["episode_index"], dtype=np.int32)
        memory = np.asarray(source["final_memory"], dtype=np.float32)
        label = np.asarray(source["final_label"], dtype=np.int32)
        prediction = np.asarray(source["final_prediction"], dtype=np.int32)
        metadata = json.loads(str(np.asarray(source["metadata_json"]).reshape(())))
    if memory.shape != (len(episode), 128, 64):
        raise ValueError(f"Invalid direct memory shape: {memory.shape}")
    if len(np.unique(episode)) != len(episode) or np.any(episode < 0):
        raise ValueError("Direct-memory episode indices must be unique and non-negative")
    if not np.array_equal(episode, np.arange(len(episode), dtype=np.int32)):
        size = int(np.max(episode)) + 1
        dense_memory = np.zeros((size, 128, 64), dtype=np.float32)
        dense_label = np.full((size,), -1, dtype=np.int32)
        dense_prediction = np.full((size,), -1, dtype=np.int32)
        dense_memory[episode] = memory
        dense_label[episode] = label
        dense_prediction[episode] = prediction
        memory, label, prediction = dense_memory, dense_label, dense_prediction
    return {"memory": memory, "label": label, "prediction": prediction, "metadata": metadata}


def _load_teacher(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        templates = np.asarray(source["memory_templates"], dtype=np.float32)
        episode_to_template = np.asarray(source["episode_template_index"], dtype=np.int32)
    if templates.ndim != 3 or templates.shape[1:] != (128, 64):
        raise ValueError(f"Invalid teacher memory shape: {templates.shape}")
    return templates[episode_to_template]


def _metadata(raw_root: Path, episode: int) -> dict[str, Any]:
    path = raw_root / f"episode_{episode:06d}" / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_sequence(metadata: dict[str, Any]) -> dict[str, Any]:
    initial = str(metadata["initial_ball_cup"])
    pairs = []
    relation_ids = []
    for raw_pair in metadata["swaps"]:
        pair = tuple(sorted((str(raw_pair[0]), str(raw_pair[1])), key=base.SLOTS.index))
        pairs.append(list(pair))
        relation_ids.append(qwenvl_event_adapter.SWAP_PAIRS.index(pair))
    return {
        "initial_slot": initial,
        "pairs": pairs,
        "relation_ids": relation_ids,
        "event_sequence": [base.SLOTS.index(initial), *relation_ids],
        "triggers": [],
    }


def _choose_wrong_donors(
    episodes: list[int],
    metadata: dict[int, dict[str, Any]],
    prediction: np.ndarray,
) -> dict[int, int]:
    donors = {}
    for episode in episodes:
        target = str(metadata[episode]["target_cup_identity"])
        candidates = [
            candidate
            for candidate in episodes
            if str(metadata[candidate]["target_cup_identity"]) != target
            and int(prediction[candidate]) != int(prediction[episode])
        ]
        if not candidates:
            candidates = [
                candidate
                for candidate in episodes
                if str(metadata[candidate]["target_cup_identity"]) != target
            ]
        if not candidates:
            raise ValueError(f"No wrong-memory donor for episode {episode}")
        donors[episode] = candidates[episode % len(candidates)]
    return donors


def _condition_summary(records: list[dict[str, Any]], *, precision_radius: float) -> dict[str, Any]:
    count = len(records)
    min_xy = np.asarray([row["min_target_xy_m"] for row in records], dtype=np.float64)
    return {
        "episodes": count,
        "cup_selection_correct": sum(bool(row["cup_selection_correct"]) for row in records),
        "cup_selection_accuracy": (
            float(np.mean([row["cup_selection_correct"] for row in records])) if count else None
        ),
        "lift_successes": sum(bool(row["success"]) for row in records),
        "lift_success_rate": float(np.mean([row["success"] for row in records])) if count else None,
        "correct_selection_and_contacts": sum(
            bool(row["correct_selection_and_contact"]) for row in records
        ),
        "correct_selection_and_contact_rate": (
            float(np.mean([row["correct_selection_and_contact"] for row in records]))
            if count
            else None
        ),
        "target_cup_contacts": sum(bool(row["target_cup_contact"]) for row in records),
        "target_cup_contact_rate": (
            float(np.mean([row["target_cup_contact"] for row in records])) if count else None
        ),
        "any_cup_contacts": sum(bool(row["any_cup_contact"]) for row in records),
        "any_cup_contact_rate": (
            float(np.mean([row["any_cup_contact"] for row in records])) if count else None
        ),
        "mean_min_target_xy_m": float(np.mean(min_xy)) if count else None,
        "mean_max_target_lift_m": (
            float(np.mean([row["max_target_lift_m"] for row in records])) if count else None
        ),
        "mean_policy_inference_ms": (
            float(np.mean([row["mean_inference_ms"] for row in records])) if count else None
        ),
        "target_approach_within_60mm": int(np.sum(min_xy <= 0.06)),
        "target_approach_within_60mm_rate": float(np.mean(min_xy <= 0.06)) if count else None,
        "target_precision_count": int(np.sum(min_xy <= precision_radius)),
        "target_precision_rate": float(np.mean(min_xy <= precision_radius)) if count else None,
        "median_min_target_xy_m": float(np.median(min_xy)) if count else None,
    }


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.direct_memory = args.direct_memory.expanduser().resolve()
    args.teacher_memory = args.teacher_memory.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.video_dir = args.video_dir.expanduser().resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    episodes = [int(value.strip()) for value in args.episodes.split(",") if value.strip()]
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    if not episodes or not conditions or any(value not in CONDITIONS for value in conditions):
        raise ValueError(f"Conditions must be a non-empty subset of {CONDITIONS}")
    if len(set(episodes)) != len(episodes) or len(set(conditions)) != len(conditions):
        raise ValueError("Episodes and conditions must not contain duplicates")
    if args.precision_radius <= 0.0 or args.selection_radius <= 0.0:
        raise ValueError("Approach radii must be positive")

    direct = _load_direct(args.direct_memory)
    teacher = _load_teacher(args.teacher_memory)
    direct_memory = direct["memory"]
    label = direct["label"]
    prediction = direct["prediction"]
    assert isinstance(direct_memory, np.ndarray)
    assert isinstance(label, np.ndarray)
    assert isinstance(prediction, np.ndarray)
    if max(episodes) >= len(direct_memory) or max(episodes) >= len(teacher):
        raise ValueError("Requested episode is absent from a memory bank")
    metadata = {episode: _metadata(args.raw_root, episode) for episode in episodes}
    donors = _choose_wrong_donors(episodes, metadata, prediction)
    if (
        not args.allow_incorrect_direct_memory
        and any(int(prediction[episode]) != int(label[episode]) for episode in episodes)
    ):
        raise ValueError("This interface test requires semantically correct direct memories")

    policy = FixedNoiseRemotePolicy(args.host, args.port, salt=args.noise_salt)
    shell = base.shell_main._import_shellgame_tools(args.robosuite_root)  # noqa: SLF001
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "frozen external MEM -> Pi action paired closed loop",
        "checkpoint": str(args.checkpoint),
        "direct_memory": str(args.direct_memory),
        "teacher_memory": str(args.teacher_memory),
        "action_parameters_updated": False,
        "memory_parameters_updated": False,
        "same_episode_and_diffusion_noise_per_condition": True,
        "noise_seed_contract": "SeedSequence([noise_salt, episode, policy_query_index])",
        "noise_salt": args.noise_salt,
        "episode_split": "action held-out seed42 validation episodes; two per target identity",
        "episodes": episodes,
        "target_distribution": {
            slot: sum(str(metadata[episode]["target_cup_identity"]) == slot for episode in episodes)
            for slot in base.SLOTS
        },
        "wrong_memory_donor": {str(key): value for key, value in donors.items()},
        "conditions": conditions,
        "control": {
            "action_mode": "absolute_eef7 raw controller command",
            "replan_steps": args.replan_steps,
            "max_policy_steps": args.max_policy_steps,
            "selection_skip": args.selection_skip,
            "selection_window": args.selection_window,
            "selection_radius_m": args.selection_radius,
            "precision_radius_m": args.precision_radius,
        },
        "records": [],
    }
    started = time.monotonic()
    try:
        for episode_ordinal, episode in enumerate(episodes):
            # Rotate order to avoid coupling a condition to renderer/server warm-up.
            offset = episode_ordinal % len(conditions)
            episode_conditions = conditions[offset:] + conditions[:offset]
            reference = _reference_sequence(metadata[episode])
            memories = {
                "teacher": teacher[episode],
                "direct_visual": direct_memory[episode],
                "wrong_visual": direct_memory[donors[episode]],
                "zero": np.zeros((128, 64), dtype=np.float32),
            }
            for condition in episode_conditions:
                policy.start_episode(episode)
                run_args = copy.copy(args)
                run_args.video_dir = args.video_dir / condition
                record = base._run_episode(  # noqa: SLF001
                    episode,
                    policy,
                    np.asarray(memories[condition], dtype=np.float32),
                    reference,
                    shell,
                    run_args,
                )
                for key in (
                    "qwen_initial_slot",
                    "qwen_pairs",
                    "qwen_event_sequence",
                    "qwen_predicted_final_slot",
                    "qwen_final_slot_correct",
                ):
                    record.pop(key, None)
                record.update(
                    {
                        "condition": condition,
                        "policy_queries": policy.query_count,
                        "memory_source_episode": (
                            donors[episode] if condition == "wrong_visual" else episode if condition != "zero" else None
                        ),
                        "direct_memory_final_label": int(label[episode]),
                        "direct_memory_final_prediction": int(prediction[episode]),
                        "direct_memory_semantic_correct": bool(prediction[episode] == label[episode]),
                        "target_approach_within_60mm": bool(record["min_target_xy_m"] <= 0.06),
                        "target_precision_reached": bool(record["min_target_xy_m"] <= args.precision_radius),
                    }
                )
                payload["records"].append(record)
                payload["summary"] = {
                    current: _condition_summary(
                        [row for row in payload["records"] if row["condition"] == current],
                        precision_radius=args.precision_radius,
                    )
                    for current in conditions
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                print(
                    f"ep={episode} condition={condition} select={record['cup_selection_correct']} "
                    f"min_xy={record['min_target_xy_m'] * 1000:.1f}mm "
                    f"precision={record['target_precision_reached']} lift={record['success']} "
                    f"queries={record['policy_queries']} elapsed={(time.monotonic() - started) / 60:.1f}m",
                    flush=True,
                )
    finally:
        policy.close()

    payload["summary"] = {
        condition: _condition_summary(
            [row for row in payload["records"] if row["condition"] == condition],
            precision_radius=args.precision_radius,
        )
        for condition in conditions
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
