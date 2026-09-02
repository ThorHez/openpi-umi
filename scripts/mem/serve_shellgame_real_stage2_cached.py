#!/usr/bin/env python3
"""Serve the real ShellGame Stage-2 policy with a 241-frame server cache."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config

HISTORY_FRAMES = 241
CURRENT_FRAME = HISTORY_FRAMES
ACTION_HORIZON = 16
MODEL_ACTION_DIM = 32
VIDEO_FRAME_KEY_PREFIX = "left_wrist_0_rgb_0_"
VIDEO_CURRENT_STEP_KEY = "left_wrist_0_rgb_0"


class CachedHistoryPolicy:
    """Cache immutable episode history while preserving the exact full input."""

    def __init__(self, policy):
        self._policy = policy
        self._history: dict[str, object] | None = None

    def infer(self, obs: dict) -> dict:
        mode = obs.get("mode")
        if mode == "reset":
            self._history = None
            return {"cache_ready": False}
        if mode == "reset_history":
            expected = {f"{VIDEO_FRAME_KEY_PREFIX}{index}" for index in range(HISTORY_FRAMES)}
            present = {key for key in obs if key.startswith(VIDEO_FRAME_KEY_PREFIX)}
            if present != expected:
                raise ValueError(f"Expected history image keys 0..{HISTORY_FRAMES - 1}, got {len(present)} keys")
            self._history = {key: obs[key] for key in expected}
            return {"cache_ready": True}
        if mode == "infer_step":
            if self._history is None:
                raise RuntimeError("History is not cached; upload reset_history first")
            noise_seed = int(obs.get("noise_seed", 0))
            full = dict(self._history)
            full.update(
                {key: value for key, value in obs.items() if key not in {"mode", VIDEO_CURRENT_STEP_KEY, "noise_seed"}}
            )
            full[f"{VIDEO_FRAME_KEY_PREFIX}{CURRENT_FRAME}"] = obs[VIDEO_CURRENT_STEP_KEY]
            noise = np.random.default_rng(noise_seed).standard_normal(
                (ACTION_HORIZON, MODEL_ACTION_DIM), dtype=np.float32
            )
            return self._policy.infer(full, noise=noise)
        return self._policy.infer(obs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_stage2",
    )
    parser.add_argument("--port", type=int, default=8017)
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
        policy=CachedHistoryPolicy(policy),
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
