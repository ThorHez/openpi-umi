"""Dedicated ShellGame recipe for supervised semantic-memory pretraining.

The recipe owns the task labels, loss weights, dataset sampling, and source
checkpoint.  The standalone trainer imports this module directly; it is not
registered in the action-policy ``train_mem.py`` path.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from openpi.tasks.shellgame import pi0_mem_semantic_action as _shellgame_model
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_semantic_action as _action_recipe

SLOTS = ("left", "middle", "right")
SWAP_PAIRS = (("left", "middle"), ("left", "right"), ("middle", "right"))
NUM_STAGES = 3


@dataclasses.dataclass(frozen=True)
class ShellGameSemanticMemoryPretrainConfig(_config.TrainConfig):
    """Training and supervision contract for the ShellGame memory module."""

    raw_metadata_root: str = (
        "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_absolute_eef_phase_instruction_dataset"
    )
    initial_loss_weight: float = 1.0
    relation_loss_weight: float = 1.0
    stage_memory_loss_weight: float = 1.0
    memory_train_augmentation: bool = False
    # Keep the episode-disjoint train/validation partition fixed when training
    # seeds are varied for multi-seed comparisons.
    split_seed: int = 42


MODEL_CONFIG = dataclasses.replace(_action_recipe.MODEL_CONFIG)


def make_train_config() -> ShellGameSemanticMemoryPretrainConfig:
    data_config_cls = _action_recipe.data_config_type(_config)
    source_checkpoint = (
        "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
        "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/"
        "absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/"
        "5999/params"
    )
    dataset_root = "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7"
    return ShellGameSemanticMemoryPretrainConfig(
        name="shellgame_semantic_memory_pretrain",
        model=MODEL_CONFIG,
        freeze_filter=MODEL_CONFIG.get_freeze_filter_memory_pretrain(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                data_config_cls(
                    repo_id=dataset_root,
                    assets=_config.AssetsConfig(asset_id=".", assets_dir=dataset_root),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=61,
                    frame_stride=1,
                    video_layout="fixed_prefix_current",
                    fixed_prefix_frames=60,
                    tokenize_prompt=False,
                    # The memory path reads the same fixed 60-frame prefix for
                    # every action frame.  Keep exactly one row per episode.
                    min_frame_index=59,
                    max_frame_index=59,
                )
            ],
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderReinitialize(
            source_checkpoint,
            reinitialize_regex=(
                r".*(HistoryFrame0InitialCupClassifier|"
                r"HistoryThreeSwapVisualRelationMemoryTracker).*"
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=3e-4,
            decay_steps=6_000,
            decay_lr=3e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=6_000,
        batch_size=12,
        num_workers=8,
        fsdp_devices=6,
        log_interval=10,
        save_interval=500,
        keep_period=1_000,
        val_ratio=0.1,
        eval_interval=250,
        eval_batches=20,
        wandb_enabled=False,
    )


def cli() -> ShellGameSemanticMemoryPretrainConfig:
    config = make_train_config()
    return tyro.extras.overridable_config_cli({config.name: (config.name, config)})


def _canonical_swap_pair(swap: list[str], *, source: pathlib.Path) -> int:
    if len(swap) != 2 or swap[0] == swap[1] or any(slot not in SLOTS for slot in swap):
        raise ValueError(f"Invalid swap in {source}: {swap!r}")
    pair = tuple(sorted(swap, key=SLOTS.index))
    try:
        return SWAP_PAIRS.index(pair)
    except ValueError as exc:
        raise ValueError(f"Unknown swap pair in {source}: {swap!r}") from exc


def _apply_swap(slot: str, swap: list[str]) -> str:
    if slot == swap[0]:
        return swap[1]
    if slot == swap[1]:
        return swap[0]
    return slot


def load_episode_label_table(
    config: ShellGameSemanticMemoryPretrainConfig,
) -> jax.Array:
    """Build ``episode_index -> [initial, relations(3), stages(3)]`` labels."""
    if not isinstance(config.data, _config.MultiDataConfigFactory) or len(config.data.datasets) != 1:
        raise ValueError("The ShellGame memory recipe requires exactly one dataset.")
    dataset_root = pathlib.Path(config.data.datasets[0].repo_id).expanduser().resolve()
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    raw_root = pathlib.Path(config.raw_metadata_root).expanduser().resolve()
    if not episodes_path.is_file():
        raise FileNotFoundError(f"LeRobot episode metadata not found: {episodes_path}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw ShellGame metadata root not found: {raw_root}")

    lerobot_records = [
        json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lerobot_records:
        raise ValueError(f"No episodes in {episodes_path}")
    max_episode = max(int(record["episode_index"]) for record in lerobot_records)
    table = np.full((max_episode + 1, 1 + NUM_STAGES * 2), -1, dtype=np.int32)

    for lerobot in lerobot_records:
        episode_index = int(lerobot["episode_index"])
        metadata_path = raw_root / f"episode_{episode_index:06d}" / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing raw metadata for episode_index={episode_index}: {metadata_path}")
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        initial_slot = str(raw["initial_ball_cup"])
        swaps = raw["swaps"]
        if initial_slot not in SLOTS or len(swaps) != NUM_STAGES:
            raise ValueError(
                f"Invalid initial slot or swap count in {metadata_path}: initial={initial_slot!r}, swaps={swaps!r}"
            )
        if initial_slot != str(lerobot["initial_ball_cup"]):
            raise ValueError(f"Initial-slot mismatch for episode_index={episode_index}")

        relation_ids = [_canonical_swap_pair(swap, source=metadata_path) for swap in swaps]
        slot = initial_slot
        stage_slots = []
        for swap in swaps:
            slot = _apply_swap(slot, swap)
            stage_slots.append(SLOTS.index(slot))
        if slot != str(raw["final_ball_cup"]) or slot != str(lerobot["final_ball_cup"]):
            raise ValueError(f"Final-slot mismatch for episode_index={episode_index}")
        if table[episode_index, 0] >= 0:
            raise ValueError(f"Duplicate episode_index={episode_index} in {episodes_path}")
        table[episode_index] = (
            SLOTS.index(initial_slot),
            *relation_ids,
            *stage_slots,
        )

    if np.any(table < 0):
        missing = np.flatnonzero(np.any(table < 0, axis=1))[:10].tolist()
        raise ValueError(f"Episode labels are not dense; first missing indices: {missing}")
    logging.info(
        "Loaded %d ShellGame memory labels: initial=%s, relation_by_stage=%s, slot_by_stage=%s",
        table.shape[0],
        np.bincount(table[:, 0], minlength=len(SLOTS)).tolist(),
        [np.bincount(table[:, 1 + i], minlength=len(SWAP_PAIRS)).tolist() for i in range(NUM_STAGES)],
        [np.bincount(table[:, 1 + NUM_STAGES + i], minlength=len(SLOTS)).tolist() for i in range(NUM_STAGES)],
    )
    return jnp.asarray(table)


def compute_objective(
    config: ShellGameSemanticMemoryPretrainConfig,
    model: _shellgame_model.Pi0MemSemanticAction,
    rng,
    observation,
    label_table,
    *,
    train: bool,
):
    """Compute the teacher-forced memory objective and diagnostic metrics."""
    if observation.episode_index is None:
        raise ValueError("Semantic-memory pretraining requires episode_index in each batch.")
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode_index]
    initial_labels = labels[:, 0]
    relation_labels = labels[:, 1 : 1 + NUM_STAGES]
    stage_labels = labels[:, 1 + NUM_STAGES :]
    outputs = model.compute_memory_pretrain_outputs(
        rng,
        observation,
        initial_slots=initial_labels,
        relation_ids=relation_labels,
        train=train and config.memory_train_augmentation,
    )

    initial_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(outputs["initial_logits"].astype(jnp.float32), initial_labels)
    )
    relation_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(outputs["relation_logits"].astype(jnp.float32), relation_labels)
    )
    stage_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(outputs["stage_logits"].astype(jnp.float32), stage_labels)
    )
    loss = (
        config.initial_loss_weight * initial_loss
        + config.relation_loss_weight * relation_loss
        + config.stage_memory_loss_weight * stage_loss
    )
    return loss, {
        "loss": loss,
        "initial_loss": initial_loss,
        "relation_loss": relation_loss,
        "stage_memory_loss": stage_loss,
        "initial_accuracy": jnp.mean(jnp.argmax(outputs["initial_logits"], axis=-1) == initial_labels),
        "relation_accuracy": jnp.mean(jnp.argmax(outputs["relation_logits"], axis=-1) == relation_labels),
        "stage_memory_accuracy": jnp.mean(jnp.argmax(outputs["stage_logits"], axis=-1) == stage_labels),
        "final_memory_accuracy": jnp.mean(jnp.argmax(outputs["stage_logits"][:, -1], axis=-1) == stage_labels[:, -1]),
    }
