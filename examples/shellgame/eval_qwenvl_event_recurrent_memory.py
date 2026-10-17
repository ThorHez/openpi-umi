"""Feed cached Qwen events through the validated recurrent ShellGame memory.

This is a controlled Stage-Q0/M0 bridge.  The visual history is encoded once,
then five teacher-input conditions reuse the same frozen recurrent updater and
readout: GT/GT, GT/Qwen, Qwen/GT, Qwen/Qwen, and wrong-episode Qwen swaps.
Qwen is never imported in the OpenPI process; only its audited JSONL cache is
read here.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax.training import common_utils
import numpy as np
import optax

from openpi.models import model as _model
from openpi.tasks.shellgame import semantic_memory as _semantic
from openpi.tasks.shellgame import qwenvl_event_adapter as _qwen_adapter
from openpi.training import weight_loaders as _weight_loaders
from openpi.training import sharding
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as _recipe
from scripts.mem import train_semantic_memory as _trainer


DEFAULT_CACHE = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame/"
    "qwenvl_event_cache/qwen3vl_4b_semantic_val_12ep.jsonl"
)
DEFAULT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "shellgame_stage_slot_only_relation_recurrent_probe/"
    "stage_slot_only_random_relation_frozen_memory_1k_260821/500/params"
)
CONDITIONS = ("gt_gt", "gt_qwen", "qwen_gt", "qwen_qwen", "gt_wrong_episode_qwen")

_QWEN_INITIAL = None
_QWEN_RELATIONS = None
_QWEN_INITIAL_VALID = None
_QWEN_RELATIONS_VALID = None
_ANNOTATED = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--exp-name", default="qwen3vl_event_recurrent_eval_12ep_260824")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument(
        "--raw-root",
        default="/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
        "shellgame_absolute_eef_phase_instruction_dataset",
    )
    # Retained for command compatibility; this evaluator creates no checkpoint.
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_qwen_tables(path: str, table_size: int):
    records = _qwen_adapter.load_annotation_records(path)
    initial = np.zeros((table_size,), dtype=np.int32)
    relations = np.zeros((table_size, _recipe.NUM_STAGES), dtype=np.int32)
    initial_valid = np.zeros((table_size,), dtype=np.bool_)
    relations_valid = np.zeros((table_size, _recipe.NUM_STAGES), dtype=np.bool_)
    seen = set()
    for record in records:
        episode = int(record["episode_index"])
        if not 0 <= episode < table_size:
            raise ValueError(f"Cache episode {episode} is outside label table size {table_size}")
        query = str(record["query_key"])
        key = (episode, query)
        if key in seen:
            raise ValueError(f"Duplicate cache record: {key}")
        seen.add(key)
        if not record.get("adapter_valid", False):
            continue
        if query == "reveal":
            initial[episode] = _recipe.SLOTS.index(str(record["prediction"]))
            initial_valid[episode] = True
        elif query.startswith("swap_"):
            stage = int(query.split("_")[1])
            pair = tuple(record["prediction"])
            relations[episode, stage] = _recipe.SWAP_PAIRS.index(pair)
            relations_valid[episode, stage] = True
    annotated = initial_valid & np.all(relations_valid, axis=1)
    if not np.any(annotated):
        raise ValueError(f"No fully valid episode annotations in {path}")
    print(
        f"Loaded Qwen cache: full={int(annotated.sum())}, "
        f"initial={int(initial_valid.sum())}, relation_events={int(relations_valid.sum())}"
    )
    return tuple(jnp.asarray(value) for value in (initial, relations, initial_valid, relations_valid, annotated))


def _masked_mean(values, mask):
    values = values.astype(jnp.float32)
    mask = mask.astype(jnp.float32)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _condition_inputs(name, gt_initial, gt_relations, qwen_initial, qwen_relations):
    if name == "gt_gt":
        return gt_initial, gt_relations
    if name == "gt_qwen":
        return gt_initial, qwen_relations
    if name == "qwen_gt":
        return qwen_initial, gt_relations
    if name == "qwen_qwen":
        return qwen_initial, qwen_relations
    if name == "gt_wrong_episode_qwen":
        return gt_initial, jnp.roll(qwen_relations, 1, axis=0)
    raise ValueError(name)


def qwen_eval_step(config, label_table, rng, state, batch):
    del config
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, _actions = batch
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode_index]
    gt_initial = labels[:, 0]
    gt_relations = labels[:, 1 : 1 + _recipe.NUM_STAGES]
    gt_stages = labels[:, 1 + _recipe.NUM_STAGES :]
    qwen_initial = _QWEN_INITIAL[episode_index]
    qwen_relations = _QWEN_RELATIONS[episode_index]
    initial_valid = _QWEN_INITIAL_VALID[episode_index]
    relation_valid = jnp.all(_QWEN_RELATIONS_VALID[episode_index], axis=1)
    annotated = _ANNOTATED[episode_index]

    processed = _model.preprocess_observation(rng, observation, train=False)
    image = processed.images["base_rgb"]
    history = image[:, : model.history_frames]
    _, history_encoder_out = model.PaliGemma.img(history, train=False)
    history_patches = history_encoder_out["with_posemb"][:, : model.history_frames]

    metrics = {
        "val/qwen_initial_accuracy": _masked_mean(qwen_initial == gt_initial, initial_valid),
        "val/qwen_relation_accuracy": _masked_mean(qwen_relations == gt_relations, _QWEN_RELATIONS_VALID[episode_index]),
        "val/qwen_relation_sequence_accuracy": _masked_mean(
            jnp.all(qwen_relations == gt_relations, axis=1), relation_valid
        ),
        "val/annotated_fraction": jnp.mean(annotated),
    }
    losses = []
    for name in CONDITIONS:
        condition_initial, condition_relations = _condition_inputs(
            name, gt_initial, gt_relations, qwen_initial, qwen_relations
        )
        _joint, stage_logits, _memories, _relation_logits, _relation_ids = (
            model.HistoryThreeSwapVisualRelationMemoryTracker(
                history_patches,
                condition_initial.astype(jnp.int32),
                condition_relations.astype(jnp.int32),
            )
        )
        if name == "gt_gt":
            valid = annotated
        elif name == "gt_qwen":
            valid = relation_valid
        elif name == "qwen_gt":
            valid = initial_valid
        elif name == "qwen_qwen":
            valid = annotated
        else:
            valid = relation_valid & jnp.roll(relation_valid, 1, axis=0)
        stage_predictions = jnp.argmax(stage_logits, axis=-1)
        stage_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(stage_logits.astype(jnp.float32), gt_stages),
            axis=1,
        )
        losses.append(_masked_mean(stage_loss, valid))
        metrics[f"val/{name}/stage_memory_accuracy"] = _masked_mean(
            jnp.mean(stage_predictions == gt_stages, axis=1), valid
        )
        metrics[f"val/{name}/final_memory_accuracy"] = _masked_mean(
            stage_predictions[:, -1] == gt_stages[:, -1], valid
        )
    metrics["val/loss"] = jnp.mean(jnp.stack(losses))
    return metrics


def _annotated_episode_ids(cache_path: str) -> list[int]:
    records = _qwen_adapter.load_annotation_records(cache_path)
    query_sets: dict[int, set[str]] = {}
    for record in records:
        if record.get("adapter_valid", False):
            query_sets.setdefault(int(record["episode_index"]), set()).add(str(record["query_key"]))
    required = {"reveal", "swap_0", "swap_1", "swap_2"}
    return sorted(episode for episode, queries in query_sets.items() if required <= queries)


def _raw_batch(raw_root: pathlib.Path, episode_ids: list[int]):
    videos = []
    for episode in episode_ids:
        path = raw_root / f"episode_{episode:06d}" / "vla_trajectory.npz"
        with np.load(path, allow_pickle=False) as trajectory:
            history = np.asarray(trajectory["third_person_images"][:60], dtype=np.uint8)
        # The semantic memory path reads frames 0..59.  Frame 60 is the
        # policy-current slot and is deliberately duplicated here because it
        # is unused by this diagnostic, matching fixed_prefix_current at row 59.
        videos.append(np.concatenate((history, history[-1:]), axis=0))
    images = np.stack(videos)
    batch = len(episode_ids)
    observation = _model.Observation.from_dict(
        {
            "image": {"base_rgb": images},
            "image_mask": {"base_rgb": np.ones((batch,), dtype=np.bool_)},
            "state": np.zeros((batch, 96), dtype=np.float32),
            "episode_index": np.asarray(episode_ids, dtype=np.int32),
            "frame_index": np.full((batch,), 59, dtype=np.int32),
            "frame_valid_mask": {"base_rgb": np.ones((batch, 61), dtype=np.bool_)},
        }
    )
    actions = np.zeros((batch, 16, 32), dtype=np.float32)
    return observation, actions


def main() -> None:
    global _QWEN_INITIAL, _QWEN_RELATIONS, _QWEN_INITIAL_VALID, _QWEN_RELATIONS_VALID, _ANNOTATED
    args = parse_args()
    base = _recipe.make_train_config()
    gt_table = np.asarray(_recipe.load_episode_label_table(base))
    (
        _QWEN_INITIAL,
        _QWEN_RELATIONS,
        _QWEN_INITIAL_VALID,
        _QWEN_RELATIONS_VALID,
        _ANNOTATED,
    ) = _load_qwen_tables(args.cache, gt_table.shape[0])
    config = dataclasses.replace(
        base,
        name="shellgame_qwenvl_event_recurrent_eval",
        exp_name=args.exp_name,
        weight_loader=_weight_loaders.CheckpointWeightLoader(args.checkpoint),
        num_train_steps=0,
        batch_size=args.batch_size,
        num_workers=0,
        fsdp_devices=args.fsdp_devices,
        eval_batches=args.eval_batches,
        eval_interval=1,
        save_interval=1,
        keep_period=1,
        wandb_enabled=False,
        overwrite=args.overwrite,
    )
    episode_ids = _annotated_episode_ids(args.cache)
    expected = args.batch_size * args.eval_batches
    if len(episode_ids) < expected:
        raise ValueError(f"Need {expected} fully annotated episodes, cache contains {len(episode_ids)}")
    episode_ids = episode_ids[:expected]
    raw_root = pathlib.Path(args.raw_root)

    # Do not construct the LeRobot/HuggingFace dataset here.  Expanding its
    # 775k video rows would create a ~59GB parquet cache for a 12-episode
    # diagnostic.  Build the exact fixed-prefix batches directly from raw NPZ.
    _trainer.init_logging()
    logging.info("Direct raw-prefix Qwen recurrent eval episodes: %s", episode_ids)
    rng = jax.random.key(config.seed)
    mesh = sharding.make_mesh(config.fsdp_devices)
    state, state_sharding = _trainer.init_train_state(config, rng, mesh, resume=False)
    jax.block_until_ready(state)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    peval = jax.jit(
        functools.partial(qwen_eval_step, config, jnp.asarray(gt_table)),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=replicated,
    )
    metrics = []
    for batch_index in range(args.eval_batches):
        start = batch_index * args.batch_size
        host_batch = _raw_batch(raw_root, episode_ids[start : start + args.batch_size])
        device_batch = jax.tree.map(
            lambda value: jax.device_put(jnp.asarray(value), data_sharding),
            host_batch,
        )
        with sharding.set_mesh(mesh):
            metrics.append(peval(jax.random.fold_in(rng, batch_index), state, device_batch))
    reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(metrics)))
    result = {key: float(value) for key, value in reduced.items()}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
