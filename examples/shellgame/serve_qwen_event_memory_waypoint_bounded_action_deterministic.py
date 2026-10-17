#!/usr/bin/env python3
"""Serve inference-only bounded waypoint XY ablations with fixed noise."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "mem"))

import jax.numpy as jnp

from examples.shellgame.serve_old_tracker_full_absolute_eef_deterministic import DeterministicNoisePolicy
from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies import policy as policy_lib
from openpi.serving import websocket_policy_server
from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_bounded_action as _bounded
from openpi.training import checkpoints
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_frame59_waypoint as recipe
from eval_shellgame_qwen_event_pi_action_closed_loop import OnlineShellGameInputs


def _model_config(limit_mm: float):
    base = recipe.make_model_config()
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    scale = float(limit_mm) / 30.0
    return _bounded.Pi0QwenEventMemoryWaypointBoundedActionConfig(
        **values,
        waypoint_bounded_normalized_limits=(0.395 * scale, 0.246 * scale),
    )


def _load_policy(checkpoint: Path, num_sampling_steps: int, limit_mm: float) -> policy_lib.Policy:
    model_config = _model_config(limit_mm)
    model = model_config.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    model.eval()
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", ".")
    masks = {"actions": transforms.make_bool_mask(7), "state": transforms.make_bool_mask(10)}
    return policy_lib.Policy(
        model,
        transforms=[
            OnlineShellGameInputs(),
            transforms.Normalize(norm_stats, use_quantiles=True, key_masks=masks),
            transforms.TokenizePrompt(
                tokenizer_lib.PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=True,
                robot_type="ARM=1 G=0 H=0",
            ),
            transforms.PadActionsOnly(model_config.action_dim),
            transforms.FlattenState(),
            transforms.KeepModelKeys(),
        ],
        output_transforms=[
            transforms.ChunkActions(target_dim=7),
            transforms.DropKeys(keys=("state",)),
            transforms.Unnormalize(norm_stats, use_quantiles=True, key_masks=masks),
        ],
        sample_kwargs={"num_steps": num_sampling_steps},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--residual-limit-mm", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    args = parser.parse_args()
    if args.residual_limit_mm <= 0:
        raise ValueError("residual-limit-mm must be positive")

    checkpoint = args.checkpoint.expanduser().resolve()
    logging.basicConfig(level=logging.INFO, force=True)
    policy = DeterministicNoisePolicy(
        _load_policy(checkpoint, args.num_sampling_steps, args.residual_limit_mm)
    )
    logging.info(
        "Loaded bounded waypoint checkpoint=%s limit_mm=%.1f noise_shape=%s",
        checkpoint,
        args.residual_limit_mm,
        policy._noise_shape,  # noqa: SLF001
    )
    logging.info("server listening on host=%s port=%d", socket.gethostname(), args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
