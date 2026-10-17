"""Ablate the two residual branches in the query-action adapter.

The trained parameter tree is unchanged.  At inference time only the static
forward definition of ``ActionMemoryCrossAttention`` is replaced, allowing a
paired comparison with identical episodes and diffusion noise:

* full: memory cross-attention + action-token MLP
* mlp_only: action-token MLP, memory update removed
* cross_only: memory cross-attention, MLP update removed
* none: both adapter updates removed
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_mem_fixed_grid_query_action as query_action
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as training_config

import training_cup_eval


MODES = ("full", "mlp_only", "cross_only", "none")


class ActionMemoryCrossAttentionBranchAblation(nn.Module):
    """Parameter-compatible adapter with static branch switches."""

    width: int = 1024
    num_heads: int = 8
    gate_init: float = 1.0
    dtype_mm: str = "bfloat16"
    mode: str = "full"

    @nn.compact
    def __call__(self, action_tokens, memory_tokens):
        if self.mode not in MODES:
            raise ValueError(f"Unknown adapter ablation mode: {self.mode}")
        action_norm = nn.LayerNorm(name="action_ln", dtype=self.dtype_mm)(action_tokens)
        memory_norm = nn.LayerNorm(name="memory_ln", dtype=self.dtype_mm)(memory_tokens)
        cross_update = nn.MultiHeadDotProductAttention(
            name="cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(action_norm, memory_norm)
        gate_delta = self.param("gate_delta", nn.initializers.zeros_init(), (1,), jnp.float32)
        gate = (self.gate_init + jnp.tanh(gate_delta)).astype(cross_update.dtype)
        cross_scale = 1.0 if self.mode in ("full", "cross_only") else 0.0
        conditioned = action_tokens + cross_scale * gate * cross_update

        mlp_input = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(conditioned)
        hidden = nn.Dense(self.width * 2, name="mlp_in", dtype=self.dtype_mm)(mlp_input)
        hidden = nn.gelu(hidden)
        mlp_update = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(hidden)
        mlp_scale = 1.0 if self.mode in ("full", "mlp_only") else 0.0
        return conditioned + mlp_scale * gate * mlp_update


def _set_mode(policy, mode: str) -> None:
    original = policy._model.ActionMemoryCrossAttention.module  # noqa: SLF001
    policy._model.ActionMemoryCrossAttention.module = (  # noqa: SLF001
        ActionMemoryCrossAttentionBranchAblation(
            width=int(original.width),
            num_heads=int(original.num_heads),
            gate_init=float(original.gate_init),
            dtype_mm=str(original.dtype_mm),
            mode=mode,
        )
    )
    policy._sample_actions = nnx_utils.module_jit(policy._model.sample_actions)  # noqa: SLF001


def _sample(policy, evaluator):
    batches = []
    for batch_index, observation, valid_size in evaluator.iter_batches():
        actions = policy._sample_actions(  # noqa: SLF001
            evaluator.sample_rng(batch_index),
            observation,
            **policy._sample_kwargs,  # noqa: SLF001
        )
        batches.append(np.asarray(jax.device_get(actions))[:valid_size])
        logging.info("sampled mode batch %d/%d", batch_index + 1, evaluator.num_batches)
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = training_config.get_config(args.config)
    config = dataclasses.replace(config, exp_name="adapter_branch_ablation")
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=(
            "The shell game has ended. Grasp and lift the cup containing the ball."
        ),
        sample_kwargs={"num_steps": config.shellgame_cup_eval.num_sampling_steps},
    )
    if not isinstance(
        policy._model, query_action.Pi0MemFixedGridQueryAction  # noqa: SLF001
    ):
        raise TypeError(f"Unexpected model type: {type(policy._model).__name__}")  # noqa: SLF001

    evaluator = training_cup_eval.ShellgameCupEvaluator(
        config, config.shellgame_cup_eval
    )
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        mode_actions = {}
        mode_details = {}
        for mode in MODES:
            logging.info("=== adapter mode=%s ===", mode)
            _set_mode(policy, mode)
            batches = _sample(policy, evaluator)
            mode_actions[mode] = np.concatenate(batches, axis=0)[: len(evaluator.selected_episode_ids)]
            evaluator.output_dir = args.output / mode
            metrics = evaluator.summarize(batches, step=1000)
            detail_path = evaluator.output_dir / "step_00001000.json"
            mode_details[mode] = {
                "metrics": metrics,
                "detail_path": str(detail_path.resolve()),
                "detail": json.loads(detail_path.read_text(encoding="utf-8")),
            }

        full_actions = mode_actions["full"]
        full_slots = [
            sample["endpoint_slot"] for sample in mode_details["full"]["detail"]["samples"]
        ]
        summary_modes = {}
        for mode in MODES:
            slots = [
                sample["endpoint_slot"] for sample in mode_details[mode]["detail"]["samples"]
            ]
            delta = mode_actions[mode].astype(np.float32) - full_actions.astype(np.float32)
            summary_modes[mode] = {
                "metrics": mode_details[mode]["metrics"],
                "normalized_action_rmse_vs_full": float(np.sqrt(np.mean(np.square(delta)))),
                "normalized_action_max_abs_vs_full": float(np.max(np.abs(delta))),
                "changed_endpoint_slot_count_vs_full": int(
                    sum(left != right for left, right in zip(full_slots, slots, strict=True))
                ),
                "detail_path": mode_details[mode]["detail_path"],
            }

        summary = {
            "config": config.name,
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "same_balanced_episodes_and_diffusion_noise": True,
            "selected_episode_ids": evaluator.selected_episode_ids.tolist(),
            "modes": summary_modes,
        }
        summary_path = args.output / "adapter_branch_ablation.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"summary_path={summary_path.resolve()}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
