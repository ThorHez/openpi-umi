#!/usr/bin/env python3
"""Audit whether GroundSG provides an action-relevant distillation signal on PickXTimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from openpi_client import websocket_client_policy


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H5 = ROOT / "data/robomme_extracted/record_dataset_PickXtimes.h5"
DEFAULT_OUTPUT = ROOT / "evaluation/robomme/pickxtimes_groundsg_distillation_signal_260827.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18011)
    parser.add_argument("--boundary-offsets", default="-16,0,16")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _reset(client) -> None:
    response = client.reset()
    while not response.get("reset_finished", False):
        time.sleep(0.05)


def _infer(client, timestep: h5py.Group, task_goal: str, subgoal: str) -> np.ndarray:
    obs = timestep["obs"]
    state = np.concatenate(
        (
            np.asarray(obs["joint_state"], dtype=np.float32),
            np.asarray(obs["gripper_state"], dtype=np.float32)[:1],
        )
    )
    _reset(client)
    result = client.infer(
        {
            "observation/image": np.asarray(obs["front_rgb"]),
            "observation/wrist_image": np.asarray(obs["wrist_rgb"]),
            "observation/state": state,
            "prompt": task_goal,
            "simple_subgoal": subgoal,
            "grounded_subgoal": subgoal,
        }
    )
    return np.asarray(result["actions"], dtype=np.float32)


def _sorted_timestep_ids(episode: h5py.Group) -> list[int]:
    return sorted(int(key.removeprefix("timestep_")) for key in episode if key.startswith("timestep_"))


def _future_actions(episode: h5py.Group, step: int, horizon: int) -> np.ndarray:
    available = set(_sorted_timestep_ids(episode))
    rows = []
    for index in range(step, step + horizon):
        if index not in available:
            break
        rows.append(np.asarray(episode[f"timestep_{index}/action/joint_action"], dtype=np.float32))
    return np.asarray(rows)


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left.astype(np.float32) - right.astype(np.float32)))))


def main() -> None:
    args = parse_args()
    offsets = [int(value) for value in args.boundary_offsets.split(",") if value.strip()]
    client = websocket_client_policy.MMEVLAWebsocketClientPolicy(args.host, args.port)

    with h5py.File(args.h5, "r") as source:
        episode = source[f"episode_{args.episode}"]
        timestep_ids = _sorted_timestep_ids(episode)
        first_step, last_step = timestep_ids[0], timestep_ids[-1]
        task_goal_raw = episode["setup/task_goal"][()]
        task_goal = _decode(task_goal_raw[0] if np.ndim(task_goal_raw) else task_goal_raw)

        phase_prototypes: dict[str, str] = {}
        phase_changes = []
        previous_simple = None
        for step in timestep_ids:
            info = episode[f"timestep_{step}/info"]
            simple = _decode(info["simple_subgoal_online"][()])
            grounded = _decode(info["grounded_subgoal_online"][()])
            phase_prototypes.setdefault(simple, grounded)
            if previous_simple is not None and simple != previous_simple:
                phase_changes.append(step)
            previous_simple = simple

        sample_steps = sorted(
            {
                min(max(boundary + offset, first_step), last_step)
                for boundary in phase_changes
                for offset in offsets
            }
        )
        rows = []
        for step in sample_steps:
            timestep = episode[f"timestep_{step}"]
            correct_simple = _decode(timestep["info/simple_subgoal_online"][()])
            correct_grounded = _decode(timestep["info/grounded_subgoal_online"][()])
            candidates = {"correct_grounded": correct_grounded, "correct_simple": correct_simple}
            for phase_index, (simple, grounded) in enumerate(phase_prototypes.items()):
                if simple != correct_simple:
                    candidates[f"wrong_phase_{phase_index}"] = grounded

            predictions = {
                name: _infer(client, timestep, task_goal, subgoal)
                for name, subgoal in candidates.items()
            }
            # Reset and repeat the correct query to measure the numerical/noise floor.
            predictions["correct_repeat"] = _infer(client, timestep, task_goal, correct_grounded)
            teacher = predictions["correct_grounded"]
            demo = _future_actions(episode, step, len(teacher))
            candidate_metrics = {}
            for name, actions in predictions.items():
                common = min(len(actions), len(demo))
                candidate_metrics[name] = {
                    "subgoal": correct_grounded if name == "correct_repeat" else candidates.get(name),
                    "action_rmse_to_correct": _rmse(actions, teacher),
                    "first_action_rmse_to_correct": _rmse(actions[0], teacher[0]),
                    "gripper_mae_to_correct": float(np.mean(np.abs(actions[:, -1] - teacher[:, -1]))),
                    "demo_action_rmse": _rmse(actions[:common], demo[:common]) if common else None,
                }
            wrong_names = [name for name in candidates if name.startswith("wrong_phase_")]
            correct_demo = candidate_metrics["correct_grounded"]["demo_action_rmse"]
            rows.append(
                {
                    "step": step,
                    "correct_simple_subgoal": correct_simple,
                    "correct_grounded_subgoal": correct_grounded,
                    "candidate_metrics": candidate_metrics,
                    "wrong_phase_mean_rmse_to_correct": float(
                        np.mean([candidate_metrics[name]["action_rmse_to_correct"] for name in wrong_names])
                    ),
                    "wrong_phase_min_rmse_to_correct": float(
                        np.min([candidate_metrics[name]["action_rmse_to_correct"] for name in wrong_names])
                    ),
                    "correct_beats_wrong_demo_fraction": float(
                        np.mean(
                            [
                                correct_demo < candidate_metrics[name]["demo_action_rmse"]
                                for name in wrong_names
                            ]
                        )
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "phase": correct_simple,
                        "wrong_mean": rows[-1]["wrong_phase_mean_rmse_to_correct"],
                        "wrong_min": rows[-1]["wrong_phase_min_rmse_to_correct"],
                        "simple_rmse": candidate_metrics["correct_simple"]["action_rmse_to_correct"],
                        "repeat_floor": candidate_metrics["correct_repeat"]["action_rmse_to_correct"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    wrong_mean = np.asarray([row["wrong_phase_mean_rmse_to_correct"] for row in rows])
    wrong_min = np.asarray([row["wrong_phase_min_rmse_to_correct"] for row in rows])
    simple = np.asarray(
        [row["candidate_metrics"]["correct_simple"]["action_rmse_to_correct"] for row in rows]
    )
    repeat = np.asarray(
        [row["candidate_metrics"]["correct_repeat"]["action_rmse_to_correct"] for row in rows]
    )
    result = {
        "h5": str(args.h5.resolve()),
        "episode": args.episode,
        "task_goal": task_goal,
        "phase_changes": phase_changes,
        "sample_steps": sample_steps,
        "candidate_phase_count": len(phase_prototypes),
        "rows": rows,
        "summary": {
            "samples": len(rows),
            "wrong_phase_mean_action_rmse": float(wrong_mean.mean()),
            "wrong_phase_min_action_rmse": float(wrong_min.mean()),
            "correct_simple_vs_grounded_action_rmse": float(simple.mean()),
            "repeat_noise_floor_action_rmse": float(repeat.mean()),
            "correct_beats_wrong_demo_fraction": float(
                np.mean([row["correct_beats_wrong_demo_fraction"] for row in rows])
            ),
            "distillation_signal_over_repeat_floor": float(
                wrong_mean.mean() / max(repeat.mean(), 1e-8)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
