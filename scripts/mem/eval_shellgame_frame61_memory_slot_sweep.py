#!/usr/bin/env python3
"""Sweep left/middle/right memories under one fixed frame-59 observation.

Diffusion noise, current image, state, and prompt stay fixed for a base
episode.  Only the frozen [128,64] memory is replaced by a semantically correct
donor for each final spatial slot.  A useful memory-to-action interface should
move predicted absolute EEF Y monotonically from left to middle to right.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.mem import eval_shellgame_frame61_first_chunk as frame61
from scripts.mem import eval_shellgame_frozen_mem_action_paired_closed_loop as paired
from scripts.mem import eval_shellgame_qwen_event_pi_action_closed_loop as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--direct-memory", type=Path, default=paired.DEFAULT_DIRECT_MEMORY)
    parser.add_argument("--raw-root", type=Path, default=base.DEFAULT_RAW_ROOT)
    parser.add_argument("--episodes", default="31,16,80")
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--prompt", default=base.PROMPT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8039)
    parser.add_argument("--noise-salt", type=int, default=260826)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.direct_memory = args.direct_memory.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.execute_steps = 1
    args.overhead_radius = 0.06
    args.precision_radius = 0.03
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")

    direct = paired._load_direct(args.direct_memory)  # noqa: SLF001
    memory = np.asarray(direct["memory"], dtype=np.float32)
    labels = np.asarray(direct["label"], dtype=np.int32)
    predictions = np.asarray(direct["prediction"], dtype=np.int32)
    donors = {}
    for slot in range(3):
        candidates = np.flatnonzero((labels == slot) & (predictions == labels))
        if len(candidates) == 0:
            raise ValueError(f"No semantically correct donor for slot {slot}")
        donors[slot] = int(candidates[0])

    episodes = [int(value.strip()) for value in args.episodes.split(",") if value.strip()]
    policy = paired.FixedNoiseRemotePolicy(args.host, args.port, salt=args.noise_salt)
    shell = base.shell_main._import_shellgame_tools(args.robosuite_root)  # noqa: SLF001
    payload = {
        "schema_version": 1,
        "experiment": "fixed observation/noise; sweep final-slot visual memory",
        "checkpoint": str(args.checkpoint),
        "episodes": episodes,
        "slot_donor_episode": {str(slot): donor for slot, donor in donors.items()},
        "records": [],
    }
    try:
        for episode in episodes:
            for slot in range(3):
                policy.start_episode(episode)
                row = frame61._run_episode(  # noqa: SLF001
                    episode,
                    policy,
                    memory[donors[slot]],
                    shell,
                    args,
                )
                command = row["command_metrics"]["0"]["command"]
                payload["records"].append(
                    {
                        "base_episode": episode,
                        "base_true_final_slot": int(labels[episode]),
                        "memory_slot": slot,
                        "memory_donor_episode": donors[slot],
                        "command0_xy": command[:2],
                        "command0_y": command[1],
                    }
                )
                print(
                    f"ep={episode} memory_slot={slot} command_xy="
                    f"({command[0]:.4f},{command[1]:.4f})",
                    flush=True,
                )
    finally:
        policy.close()

    per_episode = {}
    ranges = []
    monotonic = []
    for episode in episodes:
        rows = [row for row in payload["records"] if row["base_episode"] == episode]
        rows.sort(key=lambda row: row["memory_slot"])
        ys = np.asarray([row["command0_y"] for row in rows], dtype=np.float64)
        y_range = float(np.max(ys) - np.min(ys))
        ranges.append(y_range)
        monotonic.append(bool(ys[0] < ys[1] < ys[2]))
        per_episode[str(episode)] = {
            "command0_y_by_memory_slot": ys.tolist(),
            "command0_y_range_m": y_range,
            "left_middle_right_monotonic": monotonic[-1],
        }
    payload["summary"] = {
        "per_episode": per_episode,
        "mean_command0_y_range_m": float(np.mean(ranges)),
        "monotonic_episodes": int(np.sum(monotonic)),
        "expected_slot_separation_m": 0.1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
