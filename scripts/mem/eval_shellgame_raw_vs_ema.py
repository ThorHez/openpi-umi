"""Compare raw and EMA parameters with the fixed ShellGame cup evaluator.

This restores the complete training state so that ``state.params`` (raw) and
``state.ema_params`` (EMA) are evaluated with identical episodes and diffusion
noise.  It is intentionally a standalone diagnostic and does not alter the
training pipeline.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
from pathlib import Path

import jax
import orbax.checkpoint as ocp

import train_pi0_mem_compress as trainer
from openpi.shared import array_typing as at
from openpi.training import checkpoints
from openpi.training import config as training_config
from openpi.training import sharding


def _sample_all(psample_actions, evaluator, mesh, state):
    batches = []
    for batch_index, observation, valid_size in evaluator.iter_batches():
        with sharding.set_mesh(mesh):
            actions = psample_actions(evaluator.sample_rng(batch_index), state, observation)
        batches.append(jax.device_get(actions)[:valid_size])
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = training_config.get_config(args.config)
    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        fsdp_devices=1,
        resume=True,
        overwrite=False,
    )

    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    state_shape, state_sharding = trainer.init_train_state(
        config, init_rng, mesh, resume=True
    )
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    if not resuming:
        raise RuntimeError(f"No restorable checkpoint found in {config.checkpoint_dir}")
    # The training checkpoint was written on six devices, while this
    # diagnostic deliberately runs on one free GPU.  Explicit per-leaf target
    # shardings prevent Orbax from trying to reuse the saved six-GPU topology.
    with at.disable_typechecking():
        split_state, split_params = checkpoints._split_params(state_shape)  # noqa: SLF001

    def restore_args(tree):
        return jax.tree.map(
            lambda leaf: ocp.ArrayRestoreArgs(
                sharding=getattr(leaf, "sharding", None) or replicated_sharding
            ),
            tree,
        )

    restored = manager.restore(
        args.step,
        args=ocp.args.Composite(
            train_state=ocp.args.PyTreeRestore(
                item=split_state,
                restore_args=restore_args(split_state),
            ),
            params=ocp.args.PyTreeRestore(
                item={"params": split_params},
                restore_args=restore_args({"params": split_params}),
            ),
        ),
    )
    with at.disable_typechecking():
        state = checkpoints._merge_params(  # noqa: SLF001
            restored["train_state"], restored["params"]
        )
    jax.block_until_ready(state)
    if state.ema_params is None:
        raise RuntimeError("Checkpoint has no EMA parameters; raw-vs-EMA comparison is impossible")

    cup_eval_module = trainer._load_shellgame_cup_eval_module()  # noqa: SLF001
    evaluator = cup_eval_module.ShellgameCupEvaluator(config, config.shellgame_cup_eval)
    psample_actions = jax.jit(
        functools.partial(
            trainer.shellgame_cup_eval_step,
            num_steps=config.shellgame_cup_eval.num_sampling_steps,
        ),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=data_sharding,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    try:
        results = {}
        detail_paths = {}
        for name, eval_state in (
            ("ema", state),
            # Keep the same pytree structure and make the evaluator's EMA
            # branch read the raw parameters.
            ("raw", dataclasses.replace(state, ema_params=state.params)),
        ):
            evaluator.output_dir = args.output / name
            batches = _sample_all(psample_actions, evaluator, mesh, eval_state)
            results[name] = evaluator.summarize(batches, step=args.step)
            detail_paths[name] = evaluator.output_dir / f"step_{args.step:08d}.json"

        details = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in detail_paths.items()
        }
        ema_slots = [sample["endpoint_slot"] for sample in details["ema"]["samples"]]
        raw_slots = [sample["endpoint_slot"] for sample in details["raw"]["samples"]]
        changed = [
            int(episode_id)
            for episode_id, ema_slot, raw_slot in zip(
                evaluator.selected_episode_ids, ema_slots, raw_slots, strict=True
            )
            if ema_slot != raw_slot
        ]
        summary = {
            "config": config.name,
            "experiment": args.exp_name,
            "step": args.step,
            "same_episodes_and_noise": True,
            "ema": results["ema"],
            "raw": results["raw"],
            "raw_vs_ema_changed_slot_count": len(changed),
            "raw_vs_ema_changed_episode_ids": changed,
        }
        summary_path = args.output / f"raw_vs_ema_step_{args.step:08d}.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"summary_path={summary_path}")
    finally:
        evaluator.close()
        manager.close()


if __name__ == "__main__":
    main()
