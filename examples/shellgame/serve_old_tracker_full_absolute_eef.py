"""Serve the validated old tracker with the absolute EEF7 action expert."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.shellgame import train_old_tracker_full_absolute_eef as full_eef
from examples.shellgame import train_old_tracker_full_joint_grasp as full_joint
from examples.shellgame.eval_old_tracker_query_action_closed_loop_gate import PROMPT
from examples.shellgame.eval_old_tracker_query_action_closed_loop_gate import _original_args
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server

DEFAULT_CHECKPOINT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_260812/"
    "absolute_eef7_old_tracker_phase_balanced_b12_2k_6gpu_260812/1999"
)


def _build_config(*, sampling_steps: int):
    args = _original_args(episodes=20, batch_size=6, sampling_steps=sampling_steps)
    args.exp_name = "serve_old_tracker_full_absolute_eef"
    args.init_checkpoint = full_joint.OLD_QUERY_ACTION_CHECKPOINT
    args.gripper_loss_weight = 4.0
    args.save_interval = 500
    args.keep_period = 1_000
    config = full_eef.build_config(args)
    return dataclasses.replace(config, exp_name=args.exp_name, fsdp_devices=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    config = _build_config(sampling_steps=args.num_sampling_steps)
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=PROMPT,
        sample_kwargs={"num_steps": args.num_sampling_steps},
    )
    if not isinstance(policy._model, full_joint.OldTrackerFullJointGraspModel):  # noqa: SLF001
        raise TypeError(f"Unexpected restored model: {type(policy._model).__name__}")  # noqa: SLF001
    logging.info(
        "Loaded %s from %s (fixed_history=60 total_frames=61 action=absolute_eef7)",
        type(policy._model).__name__,  # noqa: SLF001
        args.checkpoint_dir,
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
