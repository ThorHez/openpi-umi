"""Serve batched stochastic action candidates from the frozen old tracker."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from examples.shellgame.eval_old_tracker_query_action_closed_loop_gate import PROMPT
from examples.shellgame.serve_old_tracker_full_joint_grasp import DEFAULT_CHECKPOINT
from examples.shellgame.serve_old_tracker_full_joint_grasp import _build_config
from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server


class BatchedCandidatePolicy:
    """Run one transformed observation with multiple independent diffusion noises."""

    def __init__(self, policy, *, candidate_count: int):
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if policy._is_pytorch_model:  # noqa: SLF001
            raise TypeError("BatchedCandidatePolicy currently requires a JAX policy")
        self._policy = policy
        self._candidate_count = int(candidate_count)

    @property
    def metadata(self) -> dict:
        return {
            **self._policy.metadata,
            "candidate_count": self._candidate_count,
            "candidate_batching": True,
        }

    def infer(self, obs: dict) -> dict:
        inputs = jax.tree.map(lambda value: value, obs)
        inputs = self._policy._input_transform(inputs)  # noqa: SLF001
        inputs = jax.tree.map(  # Add and replicate the model batch dimension.
            lambda value: jnp.broadcast_to(
                jnp.asarray(value)[None, ...],
                (self._candidate_count, *np.shape(value)),
            ),
            inputs,
        )
        self._policy._rng, sample_rng = jax.random.split(self._policy._rng)  # noqa: SLF001
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._policy._sample_actions(  # noqa: SLF001
            sample_rng,
            observation,
            **self._policy._sample_kwargs,  # noqa: SLF001
        )
        model_time = time.monotonic() - start_time

        actions = np.asarray(actions)
        state = np.asarray(inputs["state"])
        candidates = []
        for index in range(self._candidate_count):
            transformed = self._policy._output_transform(  # noqa: SLF001
                {"state": state[index], "actions": actions[index]}
            )
            candidates.append(np.asarray(transformed["actions"], dtype=np.float32))
        candidates = np.stack(candidates, axis=0)
        return {
            "actions": candidates[0],
            "actions_candidates": candidates,
            "policy_timing": {"infer_ms": model_time * 1000},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    config = _build_config(sampling_steps=args.num_sampling_steps)
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=PROMPT,
        sample_kwargs={"num_steps": args.num_sampling_steps},
    )
    candidate_policy = BatchedCandidatePolicy(policy, candidate_count=args.candidate_count)
    logging.info(
        "Loaded batched candidate policy from %s (candidates=%d fixed_history=60 total_frames=61)",
        args.checkpoint_dir,
        args.candidate_count,
    )
    logging.info("Serving on host=%s port=%d", socket.gethostname(), args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=candidate_policy,
        host="0.0.0.0",
        port=args.port,
        metadata=candidate_policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
