#!/usr/bin/env python3
"""Closed-loop evaluation of the MME FrameSamp -> frozen V10 action adapter."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.training.mem.recipes import shellgame_framesamp_v10_action_adapter as recipe
from scripts.mem import eval_shellgame_frozen_mem_action_paired_closed_loop as paired
from scripts.mem import eval_shellgame_v10_exact_parallel_semantic_adapter as exact_eval


DEFAULT_BANK = Path("/dev/shm/framesamp_v10_nominal5000_step9999_260828")
DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi0_shellgame_framesamp_v10_action_adapter_eef7_v1/"
    "framesamp_modul_step9999_v10_adapter_nominal5000_b8_s500_260828/499"
)
_MEMORIES: np.ndarray | None = None


def _has_argument(name: str) -> bool:
    return any(value == name or value.startswith(f"{name}=") for value in sys.argv[1:])


def _argument_value(name: str, default: str) -> str:
    for index, value in enumerate(sys.argv[:-1]):
        if value == name:
            return sys.argv[index + 1]
        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
    return default


def _load_framesamp(path: Path) -> dict[str, np.ndarray | dict[str, Any]]:
    global _MEMORIES
    lookup = recipe.FrameSampMemoryLookup.load(path)
    # Keep the 5.2 GB bank memory-mapped. Each rollout materializes only its
    # selected episode as float32 when constructing the policy request.
    _MEMORIES = lookup.memories
    labels = np.asarray(lookup.labels, dtype=np.int32)
    return {
        "memory": lookup.memories,
        "label": labels,
        "prediction": labels.copy(),
        "metadata": lookup.metadata,
    }


def _load_same_memory(_path: Path) -> np.ndarray:
    if _MEMORIES is None:
        raise RuntimeError("FrameSamp bank must be loaded before the compatibility teacher")
    return _MEMORIES


def main() -> None:
    if not _has_argument("--direct-memory"):
        sys.argv.append(f"--direct-memory={DEFAULT_BANK}")
    if not _has_argument("--checkpoint"):
        sys.argv.append(f"--checkpoint={DEFAULT_CHECKPOINT}")
    if not _has_argument("--conditions"):
        sys.argv.append("--conditions=direct_visual")

    paired._load_direct = _load_framesamp  # noqa: SLF001
    paired._load_teacher = _load_same_memory  # noqa: SLF001
    exact_eval.V10ExactParallelRemotePolicy.semantic_memory_shape = (512, 1024)
    exact_eval.main()

    output = Path(
        _argument_value("--output", str(paired.DEFAULT_OUTPUT))
    ).expanduser().resolve()
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload.update(
        {
            "experiment": "MME FrameSamp memory -> trained interface -> frozen V10 action closed loop",
            "framesamp_memory_shape": [512, 1024],
            "framesamp_bank": _argument_value("--direct-memory", str(DEFAULT_BANK)),
            "v10_action_parameters_updated": False,
            "parallel_action_interface_updated": True,
            "old_v10_memory_condition_strength": 0.0,
        }
    )
    payload.pop("parallel_semantic_adapter_initial_gate", None)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
