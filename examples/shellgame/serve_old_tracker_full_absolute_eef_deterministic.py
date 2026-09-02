"""Serve absolute EEF7 with client-specified deterministic diffusion noise.

This server is only for controlled ablations.  The client supplies a private
integer seed in each observation; the wrapper removes it before preprocessing
and passes a reproducible standard-normal action-noise tensor to Policy.infer.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import socket

import numpy as np

import serve_old_tracker_full_absolute_eef as standard_server
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server

NOISE_SEED_KEY = "__openpi_deterministic_noise_seed__"


class DeterministicNoisePolicy:
    def __init__(self, policy):
        self._policy = policy
        model = policy._model  # noqa: SLF001
        self._noise_shape = (int(model.action_horizon), int(model.action_dim))

    @property
    def metadata(self):
        return {**self._policy.metadata, "deterministic_noise_seed_key": NOISE_SEED_KEY}

    def infer(self, observation):
        observation = dict(observation)
        if NOISE_SEED_KEY not in observation:
            raise KeyError(f"Controlled ablation requires observation key {NOISE_SEED_KEY!r}")
        seed = int(observation.pop(NOISE_SEED_KEY))
        noise = np.random.default_rng(seed).standard_normal(self._noise_shape).astype(np.float32)
        return self._policy.infer(observation, noise=noise)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=standard_server.DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    config = standard_server._build_config(sampling_steps=args.num_sampling_steps)  # noqa: SLF001
    trained = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=standard_server.PROMPT,
        sample_kwargs={"num_steps": args.num_sampling_steps},
    )
    policy = DeterministicNoisePolicy(trained)
    logging.info(
        "Loaded deterministic-noise EEF7 policy from %s noise_shape=%s",
        args.checkpoint_dir,
        policy._noise_shape,  # noqa: SLF001
    )
    logging.info("Serving on host=%s port=%d", socket.gethostname(), args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
