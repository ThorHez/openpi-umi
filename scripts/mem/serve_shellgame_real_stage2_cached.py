#!/usr/bin/env python3
"""Serve the real ShellGame Stage-2 policy with a 241-frame server cache."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config
from openpi.training.mem.recipes.shellgame_real_wrist_m6 import direction_prompt

HISTORY_FRAMES = 241
CURRENT_FRAME = HISTORY_FRAMES
ACTION_HORIZON = 16
MODEL_ACTION_DIM = 32
VIDEO_FRAME_KEY_PREFIX = "left_wrist_0_rgb_0_"
VIDEO_CURRENT_STEP_KEY = "left_wrist_0_rgb_0"
CUP_NAMES = ("left", "middle", "right")
SWAP_PAIR_NAMES = ("left-middle", "left-right", "middle-right")


def _probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def summarize_memory_classification(outputs: dict) -> dict:
    """Convert model-native ShellGame memory logits to transport-safe diagnostics."""
    stage_probabilities = _probabilities(outputs["stage_logits"])
    initial_probabilities = _probabilities(outputs["initial_logits"])
    relation_probabilities = _probabilities(outputs["relation_logits"])
    stage_predictions = np.argmax(stage_probabilities, axis=-1)
    initial_prediction = int(np.argmax(initial_probabilities))
    relation_predictions = np.argmax(relation_probabilities, axis=-1)
    final_prediction = int(stage_predictions[-1])
    timing = outputs.get("policy_timing", {})
    return {
        "cup_order": list(CUP_NAMES),
        "predicted_final_cup": final_prediction,
        "predicted_final_cup_name": CUP_NAMES[final_prediction],
        "predicted_final_cup_probabilities": stage_probabilities[-1].tolist(),
        "initial_cup": initial_prediction,
        "initial_cup_name": CUP_NAMES[initial_prediction],
        "initial_cup_probabilities": initial_probabilities.tolist(),
        "stage_cups": stage_predictions.astype(int).tolist(),
        "stage_cup_names": [CUP_NAMES[int(index)] for index in stage_predictions],
        "stage_cup_probabilities": stage_probabilities.tolist(),
        "swap_pair_order": list(SWAP_PAIR_NAMES),
        "swap_pairs": relation_predictions.astype(int).tolist(),
        "swap_pair_names": [SWAP_PAIR_NAMES[int(index)] for index in relation_predictions],
        "swap_pair_probabilities": relation_probabilities.tolist(),
        "memory_infer_ms": float(timing.get("memory_infer_ms", 0.0)),
    }


class CachedHistoryPolicy:
    """Cache immutable episode history while preserving the exact full input."""

    def __init__(self, policy, *, prompt_from_memory: bool = False):
        self._policy = policy
        self._prompt_from_memory = prompt_from_memory
        self._history: dict[str, object] | None = None
        self._memory: dict[str, object] | None = None

    def infer(self, obs: dict) -> dict:
        mode = obs.get("mode")
        if mode == "reset":
            self._history = None
            self._memory = None
            return {"cache_ready": False}
        if mode == "reset_history":
            expected = {f"{VIDEO_FRAME_KEY_PREFIX}{index}" for index in range(HISTORY_FRAMES)}
            present = {key for key in obs if key.startswith(VIDEO_FRAME_KEY_PREFIX)}
            if present != expected:
                raise ValueError(f"Expected history image keys 0..{HISTORY_FRAMES - 1}, got {len(present)} keys")
            self._history = {key: obs[key] for key in expected}
            # The classifier reads only the fixed 241-frame history, while the
            # shared training transform requires a 242nd image and a 10-D
            # state. Duplicate the final history frame and supply an identity
            # state; neither is consumed by the memory tracker.
            classifier_obs = dict(self._history)
            classifier_obs[f"{VIDEO_FRAME_KEY_PREFIX}{CURRENT_FRAME}"] = self._history[
                f"{VIDEO_FRAME_KEY_PREFIX}{HISTORY_FRAMES - 1}"
            ]
            classifier_obs.update(
                {
                    "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                    "robot0_eef_rot_axis_angle": np.asarray(
                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        dtype=np.float32,
                    ),
                    "robot0_gripper_width": np.zeros(1, dtype=np.float32),
                    "prompt": obs.get("prompt", ""),
                }
            )
            self._memory = summarize_memory_classification(
                self._policy.infer_memory(classifier_obs)
            )
            logging.info(
                "MEM final cup=%s probabilities=%s stages=%s swaps=%s",
                self._memory["predicted_final_cup_name"],
                np.round(self._memory["predicted_final_cup_probabilities"], 4).tolist(),
                self._memory["stage_cup_names"],
                self._memory["swap_pair_names"],
            )
            return {"cache_ready": True, "memory": self._memory}
        if mode == "infer_step":
            if self._history is None:
                raise RuntimeError("History is not cached; upload reset_history first")
            noise_seed = int(obs.get("noise_seed", 0))
            full = dict(self._history)
            full.update(
                {key: value for key, value in obs.items() if key not in {"mode", VIDEO_CURRENT_STEP_KEY, "noise_seed"}}
            )
            if self._prompt_from_memory and self._memory is not None:
                full["prompt"] = direction_prompt(int(self._memory["predicted_final_cup"]))
            full[f"{VIDEO_FRAME_KEY_PREFIX}{CURRENT_FRAME}"] = obs[VIDEO_CURRENT_STEP_KEY]
            noise = np.random.default_rng(noise_seed).standard_normal(
                (ACTION_HORIZON, MODEL_ACTION_DIM), dtype=np.float32
            )
            result = self._policy.infer(full, noise=noise)
            if self._memory is not None:
                result["memory"] = self._memory
            return result
        return self._policy.infer(obs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_stage2",
    )
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument(
        "--prompt-from-memory",
        action="store_true",
        help="Use the cached MEM final-cup prediction to build the M6 direction prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = training_config.get_config(args.config)
    policy = policy_config.create_trained_policy(config, args.checkpoint)
    metadata = dict(policy.metadata)
    metadata.update(
        {
            "supports_cached_infer": True,
            "history_frames": HISTORY_FRAMES,
            "total_model_frames": HISTORY_FRAMES + 1,
            "action_contract": "current_frame_same_anchor_relative_link6_eef10",
            "state_contract": "episode_first_relative_link6_eef10",
        }
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=CachedHistoryPolicy(policy, prompt_from_memory=args.prompt_from_memory),
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
