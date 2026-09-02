"""Serve the proven 60-frame visual tracker + query-action checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.shellgame.eval_old_tracker_query_action_closed_loop_gate import DEFAULT_CHECKPOINT
from examples.shellgame.eval_old_tracker_query_action_closed_loop_gate import PROMPT
from examples.shellgame.eval_old_tracker_query_action_closed_loop_gate import _original_args
from examples.shellgame import train_three_swap_query_crossattn_pi_joint_action_probe as old_model
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    dynamic_args = _original_args(
        episodes=60,
        batch_size=6,
        sampling_steps=args.num_sampling_steps,
    )
    config = dataclasses.replace(
        old_model.build_config(dynamic_args),
        exp_name="serve_old_tracker_query_action",
        fsdp_devices=1,
    )
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=PROMPT,
        sample_kwargs={"num_steps": args.num_sampling_steps},
    )
    logging.info(
        "Loaded %s from %s (frames=60 stride=1 prompt=%r)",
        type(policy._model).__name__,  # noqa: SLF001
        args.checkpoint_dir,
        PROMPT,
    )
    hostname = socket.gethostname()
    logging.info("Serving on host=%s port=%d", hostname, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
