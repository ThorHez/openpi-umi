#!/usr/bin/env python3
"""Serve M6 and derive its direction prompt from cached MEM classification."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training.mem.recipes import shellgame_real_wrist_m6 as _m6
from scripts.mem.serve_shellgame_real_stage2_cached import (
    ACTION_HORIZON,
    CUP_NAMES,
    CURRENT_FRAME,
    HISTORY_FRAMES,
    MODEL_ACTION_DIM,
    VIDEO_CURRENT_STEP_KEY,
    VIDEO_FRAME_KEY_PREFIX,
    summarize_memory_classification,
)


class M6CachedHistoryPolicy:
    def __init__(self, policy):
        self._policy = policy
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
                raise ValueError(
                    f"Expected history image keys 0..{HISTORY_FRAMES - 1}, got {len(present)} keys"
                )
            self._history = {key: obs[key] for key in expected}
            classifier_obs = dict(self._history)
            classifier_obs[f"{VIDEO_FRAME_KEY_PREFIX}{CURRENT_FRAME}"] = self._history[
                f"{VIDEO_FRAME_KEY_PREFIX}{HISTORY_FRAMES - 1}"
            ]
            classifier_obs.update(
                {
                    "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                    "robot0_eef_rot_axis_angle": np.asarray(
                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32
                    ),
                    "robot0_gripper_width": np.zeros(1, dtype=np.float32),
                    "prompt": _m6.direction_prompt(1),
                }
            )
            self._memory = summarize_memory_classification(
                self._policy.infer_memory(classifier_obs)
            )
            predicted_cup = int(self._memory["predicted_final_cup"])
            self._memory["direction_prompt"] = _m6.direction_prompt(predicted_cup)
            logging.info(
                "M6 MEM final cup=%s probabilities=%s prompt=%r",
                CUP_NAMES[predicted_cup],
                np.round(self._memory["predicted_final_cup_probabilities"], 4).tolist(),
                self._memory["direction_prompt"],
            )
            return {"cache_ready": True, "memory": self._memory}
        if mode == "infer_step":
            if self._history is None or self._memory is None:
                raise RuntimeError("History is not cached; upload reset_history first")
            full = dict(self._history)
            full.update(
                {
                    key: value
                    for key, value in obs.items()
                    if key not in {"mode", VIDEO_CURRENT_STEP_KEY, "noise_seed", "prompt"}
                }
            )
            full[f"{VIDEO_FRAME_KEY_PREFIX}{CURRENT_FRAME}"] = obs[VIDEO_CURRENT_STEP_KEY]
            # Never trust a client-supplied direction at deployment. The
            # prompt and raw memory are derived from the same frozen history.
            full["prompt"] = self._memory["direction_prompt"]
            noise_seed = int(obs.get("noise_seed", 0))
            noise = np.random.default_rng(noise_seed).standard_normal(
                (ACTION_HORIZON, MODEL_ACTION_DIM), dtype=np.float32
            )
            result = self._policy.infer(full, noise=noise)
            result["memory"] = self._memory
            return result
        return self._policy.infer(obs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8017)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _m6.make_train_config(
        exp_name="serve_only",
        checkpoint=args.checkpoint,
        steps=1,
        batch_size=1,
        fsdp_devices=1,
        num_workers=0,
        eval_interval=1,
        eval_batches=1,
        save_interval=1,
    )
    policy = policy_config.create_trained_policy(config, args.checkpoint)
    metadata = dict(policy.metadata)
    metadata.update(
        {
            "supports_cached_infer": True,
            "history_frames": HISTORY_FRAMES,
            "total_model_frames": HISTORY_FRAMES + 1,
            "direction_prompt_source": "frozen_mem_final_cup",
            "direction_prompt_template": _m6.PROMPT_TEMPLATE,
            "action_contract": "current_frame_same_anchor_relative_link6_eef10",
            "state_contract": "episode_first_relative_link6_eef10",
        }
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=M6CachedHistoryPolicy(policy),
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
