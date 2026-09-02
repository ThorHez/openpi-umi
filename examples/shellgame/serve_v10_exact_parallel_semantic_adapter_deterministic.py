#!/usr/bin/env python3
# ruff: noqa: E402
"""Serve exact V10 with a fresh or trained parallel semantic adapter."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import logging
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from examples.shellgame import serve_old_tracker_full_absolute_eef as standard_server
from examples.shellgame import v10_exact_parallel_semantic_adapter as exact
from examples.shellgame.serve_old_tracker_full_absolute_eef_deterministic import DeterministicNoisePolicy
from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import nnx_utils

DEFAULT_CHECKPOINT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/"
    "absolute_eef7_v10_repro_nom60_v6preserve30_v9timing10_b12_step1000_6gpu_noprealloc_260827/"
    "1000"
)


ADAPTER_MODES = (
    "fresh_zero",
    "v10_baseline",
    "v10_action_no_memory",
    "semantic_replace",
    "semantic_parallel",
)


def _load_policy(
    checkpoint: Path,
    num_sampling_steps: int,
    *,
    adapter_checkpoint: Path | None = None,
    adapter_mode: str = "fresh_zero",
    semantic_memory_tokens: int = 128,
    semantic_memory_width: int = 64,
    semantic_query_tokens: int = 8,
):
    # Create the ordinary V10 policy first so its model, normalization,
    # transforms, prompt contract and metadata are all canonical.
    train_config = standard_server._build_config(  # noqa: SLF001
        sampling_steps=num_sampling_steps
    )
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        default_prompt=standard_server.PROMPT,
        sample_kwargs={"num_steps": num_sampling_steps},
    )
    if not isinstance(policy._model, exact._v10.OldTrackerFullJointGraspModel):  # noqa: SLF001
        raise TypeError(f"Unexpected V10 model: {type(policy._model).__name__}")  # noqa: SLF001

    _, source_state = nnx.split(policy._model)  # noqa: SLF001
    v10_params = source_state.to_pure_dict()
    if adapter_mode not in ADAPTER_MODES:
        raise ValueError(f"adapter_mode must be one of {ADAPTER_MODES}, got {adapter_mode}")
    enabled = adapter_mode in ("semantic_replace", "semantic_parallel")
    old_strength = (
        1.0
        if adapter_mode in ("fresh_zero", "v10_baseline", "semantic_parallel")
        else 0.0
    )
    adapter_config = dataclasses.replace(
        exact.make_config_from_v10(train_config.model),
        semantic_memory_tokens=semantic_memory_tokens,
        semantic_memory_width=semantic_memory_width,
        semantic_query_tokens=semantic_query_tokens,
        parallel_semantic_adapter_enabled=enabled,
        old_memory_condition_strength=old_strength,
    )
    if adapter_checkpoint is None:
        if adapter_mode not in ("fresh_zero", "v10_baseline", "v10_action_no_memory"):
            raise ValueError(f"{adapter_mode} requires --adapter-checkpoint-dir")
        fresh = adapter_config.create(jax.random.key(260827))
        _, fresh_state = nnx.split(fresh)
        merged, counts = exact.merge_exact_v10_with_fresh_parallel_adapter(
            fresh_state.to_pure_dict(), v10_params
        )
        model = adapter_config.load(merged)
        del fresh, fresh_state, merged
    else:
        params_dir = adapter_checkpoint / "params"
        if not params_dir.is_dir():
            raise FileNotFoundError(params_dir)
        trained_params = _model.restore_params(params_dir, dtype=jnp.bfloat16)
        model = adapter_config.load(trained_params)
        counts = {"v10": 322, "parallel_adapter": 34}
        del trained_params
    model.eval()
    del source_state, v10_params
    gc.collect()

    # Reuse the canonical policy's exact transforms and replace only its model
    # callable.  The newly added branch has an effective gate of exactly zero.
    policy._model = model  # noqa: SLF001
    policy._sample_actions = nnx_utils.module_jit(model.sample_actions)  # noqa: SLF001
    policy._metadata = {  # noqa: SLF001
        **policy.metadata,
        "v10_exact_parallel_semantic_adapter": True,
        "parallel_semantic_adapter_enabled": enabled,
        "parallel_semantic_adapter_mode": adapter_mode,
        "old_memory_condition_strength": old_strength,
        "semantic_memory_tokens": semantic_memory_tokens,
        "semantic_memory_width": semantic_memory_width,
        "semantic_query_tokens": semantic_query_tokens,
        "adapter_checkpoint": str(adapter_checkpoint) if adapter_checkpoint else None,
        "exact_v10_restored_leaves": counts["v10"],
        "fresh_parallel_adapter_leaves": counts["parallel_adapter"],
    }
    return policy, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--adapter-checkpoint-dir", type=Path)
    parser.add_argument("--adapter-mode", choices=ADAPTER_MODES, default="fresh_zero")
    parser.add_argument("--port", type=int, default=8075)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--semantic-memory-tokens", type=int, default=128)
    parser.add_argument("--semantic-memory-width", type=int, default=64)
    parser.add_argument("--semantic-query-tokens", type=int, default=8)
    parser.add_argument(
        "--deterministic-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a per-request diffusion seed; disable for standard closed-loop clients.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    policy, counts = _load_policy(
        args.checkpoint_dir.expanduser().resolve(),
        args.num_sampling_steps,
        adapter_checkpoint=(
            args.adapter_checkpoint_dir.expanduser().resolve()
            if args.adapter_checkpoint_dir is not None
            else None
        ),
        adapter_mode=args.adapter_mode,
        semantic_memory_tokens=args.semantic_memory_tokens,
        semantic_memory_width=args.semantic_memory_width,
        semantic_query_tokens=args.semantic_query_tokens,
    )
    served_policy = DeterministicNoisePolicy(policy) if args.deterministic_noise else policy
    logging.info(
        "Loaded exact V10 semantic adapter mode=%s v10=%s adapter=%s counts=%s deterministic_noise=%s",
        args.adapter_mode,
        args.checkpoint_dir,
        args.adapter_checkpoint_dir,
        counts,
        args.deterministic_noise,
    )
    logging.info("server listening on host=%s port=%d", socket.gethostname(), args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=served_policy,
        host="0.0.0.0",
        port=args.port,
        metadata=served_policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
