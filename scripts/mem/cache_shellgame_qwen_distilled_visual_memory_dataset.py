#!/usr/bin/env python3
"""Cache frozen direct-visual MEM tokens for an arbitrary ShellGame dataset.

Unlike the original nominal-only cache helper, this script derives the reveal
and final-cup labels from the selected LeRobot dataset's own ``episodes.jsonl``.
This is required for correction datasets, whose episode indices are local and
must never be looked up in the nominal memory bank.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.training import checkpoints
from openpi.training import sharding
from scripts.mem import cache_shellgame_qwen_distilled_visual_memory as _base
from scripts.mem import train_semantic_memory as _memory_trainer


SLOTS = ("left", "middle", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exp-name", default=_base.DEFAULT_EXP_NAME)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--batches", type=int, default=20, help="Validation batches used only to reconstruct the checkpoint graph")
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _label_table(dataset_root: Path) -> jax.Array:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing episode metadata: {episodes_path}")
    records = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No episode metadata in {episodes_path}")
    size = max(int(record["episode_index"]) for record in records) + 1
    # The inference function consumes column 0 (initial reveal) and column 6
    # (final cup).  Relation/stage columns are deliberately never fed to the
    # student and are placeholders here.
    table = np.full((size, 7), -1, dtype=np.int32)
    for record in records:
        episode = int(record["episode_index"])
        initial = str(record["initial_ball_cup"])
        final = str(record["final_ball_cup"])
        if initial not in SLOTS or final not in SLOTS:
            raise ValueError(f"Invalid cup label in episode {episode}: {initial!r}, {final!r}")
        if table[episode, 0] >= 0:
            raise ValueError(f"Duplicate episode_index={episode}")
        table[episode] = (SLOTS.index(initial), 0, 0, 0, 0, 0, SLOTS.index(final))
    if np.any(table < 0):
        missing = np.flatnonzero(np.any(table < 0, axis=1))[:10].tolist()
        raise ValueError(f"Episode indices are not dense; first missing={missing}")
    return jnp.asarray(table)


def _config(args: argparse.Namespace, dataset_root: Path):
    config = _base.make_memory_config(args)
    child = dataclasses.replace(config.data.datasets[0], repo_id=str(dataset_root))
    return dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, datasets=[child]),
        num_workers=0,
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")

    config = _config(args, dataset_root)
    label_table = _label_table(dataset_root)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader, dataset_size = _base._full_loader(config, data_sharding)  # noqa: SLF001
    num_batches = (dataset_size + args.batch_size - 1) // args.batch_size

    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    if not resuming:
        raise FileNotFoundError(f"No memory checkpoint under {config.checkpoint_dir}")
    _rng, init_rng = jax.random.split(jax.random.key(config.seed))
    state, state_sharding = _memory_trainer.init_train_state(
        config, init_rng, mesh, resume=True
    )
    state = checkpoints.restore_state(manager, state, loader)
    jax.block_until_ready(state)
    ppredict = jax.jit(
        _base.predict_step,
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=replicated,
    )

    rows = []
    loader_iter = iter(loader)
    with sharding.set_mesh(mesh):
        for batch_index in range(num_batches):
            rows.append(jax.device_get(ppredict(label_table, state, next(loader_iter))))
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == num_batches:
                print(f"cached {batch_index + 1}/{num_batches} batches", flush=True)
    stacked = jax.tree.map(lambda *values: np.concatenate(values, axis=0), *rows)
    episode = np.asarray(stacked["episode_index"], dtype=np.int32)
    unique_episode, first_index = np.unique(episode, return_index=True)
    if len(unique_episode) != dataset_size:
        raise ValueError(f"Expected {dataset_size} episodes, got {len(unique_episode)}")
    order = first_index[np.argsort(unique_episode)]
    payload = {key: np.asarray(value)[order] for key, value in stacked.items()}
    episode = payload["episode_index"]
    correct = payload["final_prediction"] == payload["final_label"]
    metadata = {
        "dataset_root": str(dataset_root),
        "source_checkpoint": str(config.checkpoint_dir / "999"),
        "inference_contract": "frames 0..59 + initial reveal slot -> final [128,64] memory",
        "uses_swap_or_final_gt_as_model_input": False,
        "initial_slot_source_in_cache": "dataset GT proxy for validated reveal classifier",
        "episodes": int(len(episode)),
        "final_slot_accuracy": float(np.mean(correct)),
        "correct_episodes": int(np.count_nonzero(correct)),
    }
    dense = np.full(int(np.max(episode)) + 1, -1, dtype=np.int32)
    dense[episode] = np.arange(len(episode), dtype=np.int32)
    if np.any(dense < 0):
        raise ValueError("Memory bank requires dense episode indices")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **payload,
        memory_templates=payload["final_memory"],
        episode_template_index=dense,
        memory_correct=correct,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
