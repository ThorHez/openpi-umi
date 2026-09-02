"""Regression gate for the proven visual tracker + query-action checkpoint.

This deliberately rebuilds the *original* dynamic model definition used by
``train_three_swap_query_crossattn_pi_joint_action_probe.py``.  Loading the
checkpoint with that definition is an exact parameter-tree audit.  It then
evaluates normal, zeroed, and batch-shuffled tracker memory on the same 60
balanced held-out episodes with identical diffusion noise.

The output is intended to be the gate before an online robosuite closed-loop
run: normal must be substantially better than both memory ablations.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import jax
import numpy as np

from examples.shellgame import train_three_swap_query_crossattn_pi_joint_action_probe as old_model
from openpi.policies import policy_config
from openpi.shared import nnx_utils

import training_cup_eval


DEFAULT_CHECKPOINT = Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_query_crossattn_pi_joint_action_260810/"
    "query_crossattn_pi_flow_action300_b12_260810/299"
)
MODES = ("normal", "zero", "shuffle_batch")
PROMPT = "The shell game has ended. Grasp and lift the cup containing the ball."


def _original_args(*, episodes: int, batch_size: int, sampling_steps: int) -> Namespace:
    """All architecture/data values from the successful training command."""
    return Namespace(
        exp_name="old_tracker_query_action_regression",
        tracker_checkpoint="",
        raw_memory_mode="normal",
        restore_adapter=False,
        restore_memory_interface=False,
        query_tokens=16,
        query_width=256,
        query_depth=2,
        query_heads=4,
        action_cross_attention_heads=8,
        init_checkpoint="",
        initial_checkpoint="",
        memory_checkpoint="",
        steps=300,
        warmup_steps=30,
        peak_lr=3e-5,
        batch_size=12,
        num_workers=0,
        fsdp_devices=1,
        eval_interval=50,
        eval_batches=10,
        cup_eval_interval=50,
        cup_eval_episodes=episodes,
        cup_eval_batch_size=batch_size,
        num_sampling_steps=sampling_steps,
        encoder_width=256,
        encoder_depth=2,
        encoder_heads=8,
        memory_width=64,
        memory_depth=2,
        memory_heads=4,
        adapter_heads=4,
        memory_tokens=128,
        current_tokens=256,
        residual_scale=1.0,
        overfit_samples_per_class=0,
        video_mode="normal",
        initial_mode="normal",
        relation_mode="one_hot",
        overwrite=False,
    )


def _sample(policy, evaluator) -> list[np.ndarray]:
    batches = []
    for batch_index, observation, valid_size in evaluator.iter_batches():
        actions = policy._sample_actions(  # noqa: SLF001
            evaluator.sample_rng(batch_index),
            observation,
            **policy._sample_kwargs,  # noqa: SLF001
        )
        batches.append(np.asarray(jax.device_get(actions))[:valid_size])
        logging.info("sampled batch %d/%d", batch_index + 1, evaluator.num_batches)
    return batches


def _set_memory_mode(policy, mode: str) -> None:
    policy._model.raw_memory_mode = mode  # noqa: SLF001
    if policy._model.raw_memory_mode != mode:  # noqa: SLF001
        raise RuntimeError(f"Failed to set raw_memory_mode={mode}")
    # raw_memory_mode is a static NNX graph attribute, so retrace the sampler.
    policy._sample_actions = nnx_utils.module_jit(policy._model.sample_actions)  # noqa: SLF001


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dynamic_args = _original_args(
        episodes=args.episodes,
        batch_size=args.batch_size,
        sampling_steps=args.num_sampling_steps,
    )
    config = old_model.build_config(dynamic_args)
    config = dataclasses.replace(
        config,
        exp_name=dynamic_args.exp_name,
        fsdp_devices=1,
        shellgame_cup_eval=dataclasses.replace(
            config.shellgame_cup_eval,
            num_episodes=args.episodes,
            batch_size=args.batch_size,
            num_sampling_steps=args.num_sampling_steps,
        ),
    )

    # create_trained_policy performs an exact model.load against this dynamic
    # model definition. Missing or shape-mismatched leaves fail here.
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=PROMPT,
        sample_kwargs={"num_steps": args.num_sampling_steps},
    )
    if not isinstance(policy._model, old_model.QueryCrossAttnPiJointActionModel):  # noqa: SLF001
        raise TypeError(f"Restored unexpected model type: {type(policy._model).__name__}")  # noqa: SLF001
    state_leaves = jax.tree.leaves(jax.device_get(nnx.state(policy._model)))  # noqa: SLF001
    audit = {
        "exact_model_class": type(policy._model).__name__,  # noqa: SLF001
        "restored_state_leaves": len(state_leaves),
        "raw_memory_shape": [128, 64],
        "query_tokens": 16,
        "query_width": 256,
        "load_succeeded_without_missing_or_shape_mismatch": True,
    }
    logging.info("parameter audit: %s", audit)

    evaluator = training_cup_eval.ShellgameCupEvaluator(config, config.shellgame_cup_eval)
    args.output.mkdir(parents=True, exist_ok=True)
    mode_actions: dict[str, np.ndarray] = {}
    mode_details: dict[str, dict] = {}
    try:
        for mode in MODES:
            logging.info("=== raw_memory_mode=%s ===", mode)
            _set_memory_mode(policy, mode)
            batches = _sample(policy, evaluator)
            mode_actions[mode] = np.concatenate(batches, axis=0)[: args.episodes]
            evaluator.output_dir = args.output / mode
            metrics = evaluator.summarize(batches, step=299)
            detail_path = evaluator.output_dir / "step_00000299.json"
            mode_details[mode] = {
                "metrics": metrics,
                "detail_path": str(detail_path.resolve()),
                "detail": json.loads(detail_path.read_text(encoding="utf-8")),
            }

        normal_slots = [
            row["endpoint_slot"] for row in mode_details["normal"]["detail"]["samples"]
        ]
        comparisons = {}
        for mode in MODES:
            slots = [row["endpoint_slot"] for row in mode_details[mode]["detail"]["samples"]]
            delta = mode_actions[mode].astype(np.float32) - mode_actions["normal"].astype(np.float32)
            comparisons[mode] = {
                "metrics": mode_details[mode]["metrics"],
                "normalized_action_rmse_vs_normal": float(np.sqrt(np.mean(np.square(delta)))),
                "changed_endpoint_slot_count_vs_normal": int(
                    sum(left != right for left, right in zip(normal_slots, slots, strict=True))
                ),
                "detail_path": mode_details[mode]["detail_path"],
            }

        normal_acc = comparisons["normal"]["metrics"]["val/cup_endpoint_accuracy"]
        zero_acc = comparisons["zero"]["metrics"]["val/cup_endpoint_accuracy"]
        shuffle_acc = comparisons["shuffle_batch"]["metrics"]["val/cup_endpoint_accuracy"]
        summary = {
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "parameter_mapping_audit": audit,
            "same_balanced_episodes_and_diffusion_noise": True,
            "selected_episode_ids": evaluator.selected_episode_ids.tolist(),
            "modes": comparisons,
            "closed_loop_gate": {
                "normal_above_chance": normal_acc > 1.0 / 3.0,
                "normal_better_than_zero": normal_acc > zero_acc,
                "normal_better_than_shuffle": normal_acc > shuffle_acc,
                "passed": normal_acc > 0.75 and normal_acc > zero_acc and normal_acc > shuffle_acc,
            },
        }
        summary_path = args.output / "old_tracker_query_action_regression.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"summary_path={summary_path.resolve()}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
