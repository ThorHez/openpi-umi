#!/usr/bin/env python3
"""Serve the frozen-Qwen-event recurrent-MEM absolute-EEF7 action policy."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "mem"))

from openpi.serving import websocket_policy_server

from eval_shellgame_qwen_event_pi_action_closed_loop import DEFAULT_CHECKPOINT
from eval_shellgame_qwen_event_pi_action_closed_loop import _load_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=8025)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    logging.basicConfig(level=logging.INFO, force=True)
    policy = _load_policy(checkpoint, args.num_sampling_steps)
    logging.info(
        "Loaded Qwen-event recurrent-MEM action policy from %s (absolute_eef7)",
        checkpoint,
    )
    logging.info("server listening on host=%s port=%d", socket.gethostname(), args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
