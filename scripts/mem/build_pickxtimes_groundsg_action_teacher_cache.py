#!/usr/bin/env python3
"""Build phase-level GroundSG action-distillation targets for PickXTimes.

For a fixed observation, the frozen official policy is queried with three
counterfactual simple subgoals (Pick, Place, Press).  Their action chunks are
compared with the action chunk produced by the oracle grounded subgoal.  A
temperature softmax over the negative RMSEs becomes a three-way functional
teacher target aligned to the corresponding recurrent MEM state.
"""

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
DEFAULT_SEQUENCE = ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_OUTPUT = ROOT / "artifacts/robomme_pickxtimes_groundsg_action_teacher_v1_260827"
PHASES = ("pick", "place", "press")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument("--episodes-per-split", type=int, default=6)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18011)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--chunk-offsets", default="-1,0,1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _reset(client) -> None:
    response = client.reset()
    while not response.get("reset_finished", False):
        time.sleep(0.02)


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


def _phase(text: str) -> int:
    lower = text.lower().strip()
    for index, name in enumerate(PHASES):
        if lower.startswith(name):
            return index
    raise ValueError(f"Unknown PickXTimes subgoal phase: {text!r}")


def _phase_prototypes(episode: h5py.Group) -> dict[int, list[tuple[int, str]]]:
    result = {index: [] for index in range(len(PHASES))}
    previous = None
    for step in sorted(int(key.removeprefix("timestep_")) for key in episode if key.startswith("timestep_")):
        simple = _decode(episode[f"timestep_{step}/info/simple_subgoal_online"][()])
        if simple != previous:
            result[_phase(simple)].append((step, simple))
            previous = simple
    if any(not rows for rows in result.values()):
        raise ValueError("Episode does not contain all Pick/Place/Press phases")
    return result


def _nearest(rows: list[tuple[int, str]], step: int) -> str:
    return min(rows, key=lambda row: abs(row[0] - step))[1]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def build_split(args: argparse.Namespace, split: str, client, source: h5py.File) -> dict:
    with np.load(args.sequence_dir / f"{split}.npz", allow_pickle=False) as payload:
        sequence = {key: np.asarray(payload[key]) for key in payload.files}
    metadata = [
        json.loads(line)
        for line in (args.sequence_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    # Do not assume a numeric task id here: the unified contract currently
    # orders PickXTimes last, whereas older experiment notes used another order.
    pick_rows = np.asarray(
        [
            index
            for index, row in enumerate(metadata)
            if row["source"] == "pickxtimes_local_event"
        ][: args.episodes_per_split],
        dtype=np.int64,
    )
    max_states = sequence["step_mask"].shape[1] + 1
    probs = np.zeros((len(sequence["task_ids"]), max_states, len(PHASES)), dtype=np.float32)
    distances = np.zeros_like(probs)
    mask = np.zeros((len(sequence["task_ids"]), max_states), dtype=np.bool_)
    raw_steps = np.full((len(sequence["task_ids"]), max_states), -1, dtype=np.int32)
    offsets = [int(value) for value in args.chunk_offsets.split(",") if value.strip()]
    rows = []

    for row_index in pick_rows:
        episode_index = int(sequence["episode_index"][row_index])
        episode = source[f"episode_{episode_index}"]
        goal_raw = episode["setup/task_goal"][()]
        task_goal = _decode(goal_raw[0] if np.ndim(goal_raw) else goal_raw)
        prototypes = _phase_prototypes(episode)
        length = int(sequence["step_mask"][row_index].sum())
        changes = np.flatnonzero(sequence["state_change_mask"][row_index, :length])
        selected_chunks = sorted(
            {
                int(change + offset)
                for change in changes
                for offset in offsets
                if 0 <= change + offset < length
            }
        )
        for chunk_index in selected_chunks:
            step = int(sequence["frame_indices"][row_index, chunk_index, -1])
            timestep = episode[f"timestep_{step}"]
            correct_simple = _decode(timestep["info/simple_subgoal_online"][()])
            correct_grounded = _decode(timestep["info/grounded_subgoal_online"][()])
            candidates = [_nearest(prototypes[index], step) for index in range(len(PHASES))]
            reference = _infer(client, timestep, task_goal, correct_grounded)
            actions = [_infer(client, timestep, task_goal, candidate) for candidate in candidates]
            rmses = np.asarray(
                [np.sqrt(np.mean(np.square(action - reference))) for action in actions],
                dtype=np.float32,
            )
            target = _softmax(-rmses / args.temperature).astype(np.float32)
            state_index = chunk_index + 1
            probs[row_index, state_index] = target
            distances[row_index, state_index] = rmses
            mask[row_index, state_index] = True
            raw_steps[row_index, state_index] = step
            correct_phase = _phase(correct_simple)
            rows.append(
                {
                    "split": split,
                    "row_index": int(row_index),
                    "episode_index": episode_index,
                    "chunk_index": chunk_index,
                    "state_index": state_index,
                    "raw_step": step,
                    "correct_phase": PHASES[correct_phase],
                    "correct_simple_subgoal": correct_simple,
                    "correct_grounded_subgoal": correct_grounded,
                    "candidate_subgoals": dict(zip(PHASES, candidates, strict=True)),
                    "action_rmse": dict(zip(PHASES, rmses.tolist(), strict=True)),
                    "phase_probabilities": dict(zip(PHASES, target.tolist(), strict=True)),
                }
            )
            print(
                json.dumps(
                    {
                        "split": split,
                        "episode": episode_index,
                        "state": state_index,
                        "step": step,
                        "correct": PHASES[correct_phase],
                        "p_correct": float(target[correct_phase]),
                        "rmse": rmses.tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    np.savez_compressed(
        args.output_dir / f"{split}.npz",
        action_phase_probs=probs,
        action_phase_distances=distances,
        action_phase_mask=mask,
        raw_steps=raw_steps,
        episode_index=sequence["episode_index"],
        selected_row_indices=pick_rows,
        temperature=np.asarray(args.temperature, dtype=np.float32),
    )
    (args.output_dir / f"{split}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    correct = np.asarray([row["phase_probabilities"][row["correct_phase"]] for row in rows])
    top1 = np.asarray(
        [max(row["phase_probabilities"], key=row["phase_probabilities"].get) == row["correct_phase"] for row in rows]
    )
    return {
        "episodes": len(pick_rows),
        "states": len(rows),
        "mean_correct_phase_probability": float(correct.mean()),
        "teacher_top1_correct_phase_accuracy": float(top1.mean()),
        "selected_episode_indices": sequence["episode_index"][pick_rows].tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    outputs = [args.output_dir / f"{split}.npz" for split in splits]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Output already exists in {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.MMEVLAWebsocketClientPolicy(args.host, args.port)
    summaries = {}
    with h5py.File(args.h5, "r") as source:
        for split in splits:
            summaries[split] = build_split(args, split, client, source)
    summary = {
        "schema_version": 1,
        "teacher": "official GroundSG action policy",
        "phases": PHASES,
        "target": "softmax(-RMSE(candidate_simple_action, oracle_grounded_action) / temperature)",
        "temperature": args.temperature,
        "chunk_offsets_from_gt_transition": [
            int(value) for value in args.chunk_offsets.split(",") if value.strip()
        ],
        "splits": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
