#!/usr/bin/env python3
"""Evaluate cached MemER subgoals with the V10 action-only policy."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mem import eval_shellgame_frozen_mem_action_paired_closed_loop as paired
from scripts.mem import eval_shellgame_v10_exact_parallel_semantic_adapter as exact_eval


def _consume_argument(name: str) -> str:
    for index, value in enumerate(sys.argv):
        if value == name and index + 1 < len(sys.argv):
            result = sys.argv[index + 1]
            del sys.argv[index : index + 2]
            return result
        if value.startswith(f"{name}="):
            result = value.split("=", 1)[1]
            del sys.argv[index]
            return result
    raise ValueError(f"Missing required argument {name}")


def main() -> None:
    manifest_path = Path(_consume_argument("--memer-manifest")).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_episode = {int(row["episode"]): row for row in manifest["records"]}
    if len(by_episode) != len(manifest["records"]):
        raise ValueError("MemER manifest contains duplicate episodes")

    original_run_episode = paired.base._run_episode  # noqa: SLF001

    def run_episode_with_memer_prompt(
        episode: int,
        policy: Any,
        memory: np.ndarray,
        qwen: dict[str, Any],
        shell: Any,
        args: Any,
    ) -> dict[str, Any]:
        if episode not in by_episode:
            raise KeyError(f"Episode {episode} is absent from {manifest_path}")
        memer = by_episode[episode]
        run_args = copy.copy(args)
        run_args.prompt = str(memer["subgoal"])
        record = original_run_episode(episode, policy, memory, qwen, shell, run_args)
        record.update(
            {
                "memer_subgoal": memer["subgoal"],
                "memer_raw_response": memer.get("raw_response"),
                "memer_coordinates_256": memer.get("coordinates_256", []),
                "memer_predicted_world_slot": memer.get("predicted_world_slot"),
                "memer_grounding_parseable": bool(memer.get("grounding_parseable")),
                "memer_grounding_correct": bool(memer.get("grounding_correct")),
            }
        )
        return record

    # The paired evaluator expects semantic banks.  This experiment passes an
    # all-zero tensor and the server's v10_action_no_memory mode disables both
    # the semantic branch and V10's native tracker-memory branch.
    size = max(by_episode) + 1
    zeros = np.zeros((size, 128, 64), dtype=np.float32)
    labels = np.zeros((size,), dtype=np.int32)
    paired._load_direct = lambda _path: {  # noqa: SLF001
        "memory": zeros,
        "label": labels,
        "prediction": labels.copy(),
        "metadata": {"source": "synthetic zero memory for MemER language interface"},
    }
    paired._load_teacher = lambda _path: zeros  # noqa: SLF001
    paired._choose_wrong_donors = (  # noqa: SLF001
        lambda episodes, _metadata, _prediction: {episode: episode for episode in episodes}
    )
    paired.base._run_episode = run_episode_with_memer_prompt  # noqa: SLF001
    exact_eval.V10ExactParallelRemotePolicy.semantic_memory_shape = (128, 64)
    exact_eval.main()

    output = Path(exact_eval._argument_value("--output", str(paired.DEFAULT_OUTPUT))).resolve()
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload.update(
        {
            "experiment": "MemER zero-shot subgoal -> V10 action-only/no-memory ShellGame",
            "memer_manifest": str(manifest_path),
            "memer_adapter": manifest.get("adapter"),
            "memer_shellgame_training": False,
            "memer_called_before_control": True,
            "memer_call_frequency": "once per episode after the 60-frame observation prefix",
            "policy_input_contract": "MemER current_subtask + current RGB/state; native V10 memory disabled",
            "semantic_memory_used_by_policy": False,
            "native_v10_tracker_memory_used_by_policy": False,
        }
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
