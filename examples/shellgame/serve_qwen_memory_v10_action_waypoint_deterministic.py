#!/usr/bin/env python3
# ruff: noqa: E402
"""Serve Qwen-distilled memory/waypoint with a transplanted V10 action branch."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "mem"))

from eval_shellgame_qwen_event_pi_action_closed_loop import OnlineShellGameInputs
import jax.numpy as jnp
import numpy as np

from examples.shellgame.serve_old_tracker_full_absolute_eef_deterministic import DeterministicNoisePolicy
from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies import policy as policy_lib
from openpi.serving import websocket_policy_server
from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_action_no_token_injection
from openpi.tasks.shellgame import v10_action_weight_transplant
from openpi.training import checkpoints
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_frame59_waypoint as recipe

DEFAULT_CURRENT_CHECKPOINT = Path(
    "checkpoints/pi0_shellgame_qwen_distilled_memory_waypoint_grasp_v6_eef7_260826/"
    "direct_visual_waypoint_grasp_v6_60_30_5_3_2_3k_6gpu_260826/2000"
)
DEFAULT_V10_CHECKPOINT = Path(
    "checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/"
    "absolute_eef7_v10_repro_nom60_v6preserve30_v9timing10_b12_step1000_6gpu_noprealloc_260827/1000"
)


def _load_policy(
    current_checkpoint: Path,
    v10_checkpoint: Path,
    num_sampling_steps: int,
    *,
    disable_token_injection: bool,
) -> policy_lib.Policy:
    model_config = recipe.make_model_config()
    if disable_token_injection:
        model_config = pi0_qwen_event_memory_waypoint_action_no_token_injection.from_waypoint_config(model_config)
    current_params = model_lib.restore_params(current_checkpoint / "params", dtype=jnp.bfloat16)
    # Keep the donor on host memory.  Only the selected common action leaves
    # become part of the live model; old tracker parameters are never loaded.
    v10_params = model_lib.restore_params(v10_checkpoint / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)
    merged_params, report = v10_action_weight_transplant.transplant_v10_action_params(current_params, v10_params)
    logging.info(
        "Strict V10 action transplant leaves=%d elements=%d expert=%d projections=%d",
        report.selected_leaves,
        report.selected_elements,
        report.action_expert_leaves,
        report.projection_leaves,
    )
    logging.info("First/last transplanted paths: %s / %s", report.selected_paths[0], report.selected_paths[-1])

    model = model_config.load(merged_params)
    model.eval()
    norm_stats = checkpoints.load_norm_stats(current_checkpoint / "assets", ".")
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
    parser.add_argument("--current-checkpoint", type=Path, default=DEFAULT_CURRENT_CHECKPOINT)
    parser.add_argument("--v10-checkpoint", type=Path, default=DEFAULT_V10_CHECKPOINT)
    parser.add_argument("--port", type=int, default=8062)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument(
        "--disable-token-injection",
        action="store_true",
        help="Keep the memory-decoded hard XY waypoint but feed raw tokens to the V10 action expert.",
    )
    args = parser.parse_args()
    current = args.current_checkpoint.expanduser().resolve()
    v10 = args.v10_checkpoint.expanduser().resolve()
    logging.basicConfig(level=logging.INFO, force=True)
    policy = DeterministicNoisePolicy(
        _load_policy(
            current,
            v10,
            args.num_sampling_steps,
            disable_token_injection=args.disable_token_injection,
        )
    )
    logging.info(
        "Loaded hybrid policy current_memory_waypoint=%s v10_action=%s token_injection=%s",
        current,
        v10,
        not args.disable_token_injection,
    )
    logging.info("server listening on host=%s port=%d", socket.gethostname(), args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
