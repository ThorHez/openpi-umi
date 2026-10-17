"""Training recipe for causal six-frame ShellGame semantic event memory."""

from __future__ import annotations

import dataclasses

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from openpi.models import model as _model
from openpi.tasks.shellgame import pi0_mem_semantic_event_memory as _event_model
from openpi.tasks.shellgame import semantic_memory_event
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as _base_recipe

ALIGNED_STARTS = (22, 32, 42)
POSITIVE_OFFSET_RADIUS = 2
CROSS_BOUNDARY_STARTS = (25, 26, 27, 28, 29, 35, 36, 37, 38, 39)
STATIC_OR_PARTIAL_STARTS = (
    0,
    5,
    10,
    14,
    15,
    16,
    17,
    18,
    19,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
)

DEFAULT_WINDOW10_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "shellgame_sliding_window_event_recurrent_memory_probe/"
    "sliding_window_event_gate_500_260821/499/params"
)


@dataclasses.dataclass(frozen=True)
class ShellGameSemanticMemoryPretrainConfig(_config.TrainConfig):
    """Configuration consumed by the shared semantic-memory training loop."""

    raw_metadata_root: str = (
        "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_absolute_eef_phase_instruction_dataset"
    )
    initial_loss_weight: float = 1.0
    event_loss_weight: float = 0.5
    relation_loss_weight: float = 0.5
    stage_memory_loss_weight: float = 1.0
    memory_train_augmentation: bool = False


@dataclasses.dataclass(frozen=True)
class SixFrameEventCheckpointLoader:
    """Restore the validated ten-frame tracker and crop temporal position to six."""

    params_path: str = DEFAULT_WINDOW10_CHECKPOINT

    def load(self, params):
        source = flax.traverse_util.flatten_dict(
            _model.restore_params(self.params_path, restore_type=np.ndarray), sep="/"
        )
        target = flax.traverse_util.flatten_dict(params, sep="/")
        result = {}
        exact = cropped_temporal = 0
        missing = []
        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
                continue
            if (
                candidate is not None
                and key.endswith("relative_temporal_pos_embedding")
                and candidate.ndim == reference.ndim == 4
                and candidate.shape[0] == reference.shape[0] == 1
                and candidate.shape[1] == 10
                and reference.shape[1] == semantic_memory_event.WINDOW_SIZE
                and candidate.shape[2:] == reference.shape[2:]
            ):
                start = (candidate.shape[1] - semantic_memory_event.WINDOW_SIZE) // 2
                result[key] = np.asarray(
                    candidate[:, start : start + semantic_memory_event.WINDOW_SIZE],
                    dtype=np.dtype(reference.dtype),
                )
                cropped_temporal += 1
                continue
            result[key] = reference
            missing.append(key)
        if missing:
            raise ValueError(f"Six-frame event checkpoint restore incomplete: {missing[:8]}")
        print(f"SixFrameEventCheckpointLoader: exact={exact}, cropped_temporal={cropped_temporal}, missing=0")
        return flax.traverse_util.unflatten_dict(result, sep="/")


def _copy_model_config():
    source = _base_recipe.MODEL_CONFIG
    values = {
        field.name: getattr(source, field.name)
        for field in dataclasses.fields(_event_model.Pi0MemSemanticEventMemoryConfig)
        if hasattr(source, field.name)
    }
    values.update(
        num_frames=61,
        history_frames=semantic_memory_event.HISTORY_FRAMES,
        event_window_size=semantic_memory_event.WINDOW_SIZE,
    )
    return _event_model.Pi0MemSemanticEventMemoryConfig(**values)


def make_train_config() -> ShellGameSemanticMemoryPretrainConfig:
    base = _base_recipe.make_train_config()
    model = _copy_model_config()
    return ShellGameSemanticMemoryPretrainConfig(
        name="shellgame_event_semantic_memory_pretrain",
        model=model,
        freeze_filter=model.get_freeze_filter_event_memory_pretrain(),
        data=base.data,
        weight_loader=SixFrameEventCheckpointLoader(),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=50, peak_lr=1e-4, decay_steps=500, decay_lr=1e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=500,
        batch_size=6,
        num_workers=0,
        fsdp_devices=6,
        log_interval=10,
        save_interval=250,
        keep_period=250,
        val_ratio=0.1,
        eval_interval=100,
        eval_batches=20,
        wandb_enabled=False,
        raw_metadata_root=base.raw_metadata_root,
    )


def cli() -> ShellGameSemanticMemoryPretrainConfig:
    config = make_train_config()
    return tyro.extras.overridable_config_cli({config.name: (config.name, config)})


def load_episode_label_table(config):
    return _base_recipe.load_episode_label_table(config)


def _stage_index_for_start(starts):
    stage = jnp.full(starts.shape, -1, dtype=jnp.int32)
    for stage_index, aligned in enumerate(ALIGNED_STARTS):
        stage = jnp.where(jnp.abs(starts - aligned) <= POSITIVE_OFFSET_RADIUS, stage_index, stage)
    return stage


def _sample_training_starts(rng, episode_index, *, train: bool):
    batch = episode_index.shape[0]
    if train:
        positive_rng, cross_rng, static_rng = jax.random.split(rng, 3)
        offsets = jax.random.randint(
            positive_rng,
            (batch, semantic_memory_event.NUM_STAGES),
            minval=-POSITIVE_OFFSET_RADIUS,
            maxval=POSITIVE_OFFSET_RADIUS + 1,
        )
        cross_choice = jax.random.randint(cross_rng, (batch, 2), 0, 5)
        static_choice = jax.random.randint(static_rng, (batch, 3), 0, len(STATIC_OR_PARTIAL_STARTS))
    else:
        episode_index = episode_index.astype(jnp.int32)
        offsets = jnp.stack(
            [((episode_index * (stage + 3) + stage) % 5) - 2 for stage in range(semantic_memory_event.NUM_STAGES)],
            axis=1,
        )
        cross_choice = jnp.stack(((episode_index * 3) % 5, (episode_index * 7 + 1) % 5), axis=1)
        static_choice = jnp.stack(
            [(episode_index * (index + 5) + index) % len(STATIC_OR_PARTIAL_STARTS) for index in range(3)],
            axis=1,
        )
    positives = jnp.asarray(ALIGNED_STARTS, dtype=jnp.int32)[None] + offsets
    first_cross = jnp.asarray(CROSS_BOUNDARY_STARTS[:5])[cross_choice[:, 0]]
    second_cross = jnp.asarray(CROSS_BOUNDARY_STARTS[5:])[cross_choice[:, 1]]
    static = jnp.asarray(STATIC_OR_PARTIAL_STARTS)[static_choice]
    return jnp.concatenate((positives, first_cross[:, None], second_cross[:, None], static), axis=1)


def compute_objective(config, model, rng, observation, label_table, *, train: bool):
    """Use sampled teacher events for train and a full causal scan for eval."""
    if observation.episode_index is None:
        raise ValueError("Event-memory training requires episode_index")
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode_index]
    initial_labels = labels[:, 0]
    relation_labels = labels[:, 1:4]
    stage_labels = labels[:, 4:7]

    if train:
        starts = _sample_training_starts(rng, episode_index, train=True)
        outputs = model.compute_event_memory_outputs(
            rng,
            observation,
            initial_slots=initial_labels,
            window_starts=starts,
            causal_selection=False,
            train=config.memory_train_augmentation,
        )
        event_targets = jnp.concatenate(
            (
                jnp.ones((episode_index.shape[0], 3), dtype=jnp.float32),
                jnp.zeros((episode_index.shape[0], starts.shape[1] - 3), dtype=jnp.float32),
            ),
            axis=1,
        )
        relation_targets = relation_labels
        relation_logits = outputs["relation_logits"][:, :3]
        relation_mask = jnp.ones_like(relation_targets, dtype=jnp.bool_)
    else:
        starts = jnp.tile(
            jnp.arange(semantic_memory_event.NUM_WINDOWS, dtype=jnp.int32)[None],
            (episode_index.shape[0], 1),
        )
        outputs = model.compute_event_memory_outputs(
            rng,
            observation,
            initial_slots=initial_labels,
            window_starts=starts,
            causal_selection=True,
            train=False,
        )
        stage_index = _stage_index_for_start(starts)
        event_targets = (stage_index >= 0).astype(jnp.float32)
        safe_stage = jnp.maximum(stage_index, 0)
        relation_targets = jnp.take_along_axis(relation_labels, safe_stage, axis=1)
        relation_logits = outputs["relation_logits"]
        relation_mask = event_targets > 0

    initial_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(outputs["initial_logits"].astype(jnp.float32), initial_labels)
    )
    positive = event_targets > 0
    event_losses = optax.sigmoid_binary_cross_entropy(outputs["event_logits"], event_targets)
    positive_event_loss = jnp.sum(event_losses * positive) / jnp.maximum(jnp.sum(positive), 1)
    negative_event_loss = jnp.sum(event_losses * ~positive) / jnp.maximum(jnp.sum(~positive), 1)
    event_loss = 0.5 * (positive_event_loss + negative_event_loss)
    relation_losses = optax.softmax_cross_entropy_with_integer_labels(
        relation_logits.astype(jnp.float32), relation_targets
    )
    relation_loss = jnp.sum(relation_losses * relation_mask) / jnp.maximum(jnp.sum(relation_mask), 1)
    stage_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(outputs["stage_logits"].astype(jnp.float32), stage_labels)
    )
    loss = (
        config.initial_loss_weight * initial_loss
        + config.event_loss_weight * event_loss
        + config.relation_loss_weight * relation_loss
        + config.stage_memory_loss_weight * stage_loss
    )

    event_predictions = outputs["event_logits"] > 0
    stage_predictions = jnp.argmax(outputs["stage_logits"], axis=-1)
    selected_relation_ids = jnp.argmax(outputs["selected_relation_logits"], axis=-1)
    valid = outputs["selection_valid"]
    return loss, {
        "loss": loss,
        "initial_loss": initial_loss,
        "event_gate_loss": event_loss,
        "relation_loss": relation_loss,
        "stage_memory_loss": stage_loss,
        "initial_accuracy": jnp.mean(jnp.argmax(outputs["initial_logits"], axis=-1) == initial_labels),
        "event_accuracy": jnp.mean(event_predictions == positive),
        "complete_event_recall": jnp.sum(event_predictions * positive) / jnp.maximum(jnp.sum(positive), 1),
        "no_event_rejection": jnp.sum((~event_predictions) * (~positive)) / jnp.maximum(jnp.sum(~positive), 1),
        "relation_accuracy": jnp.sum((jnp.argmax(relation_logits, axis=-1) == relation_targets) * relation_mask)
        / jnp.maximum(jnp.sum(relation_mask), 1),
        "selected_relation_sequence_accuracy": jnp.mean(
            jnp.all(selected_relation_ids == relation_labels, axis=1) & valid
        ),
        "stage_memory_accuracy": jnp.mean(stage_predictions == stage_labels),
        "final_memory_accuracy": jnp.mean(stage_predictions[:, -1] == stage_labels[:, -1]),
        "final_memory_e2e_accuracy": jnp.mean((stage_predictions[:, -1] == stage_labels[:, -1]) & valid),
        "valid_trigger_count": jnp.mean(valid),
        "mean_trigger_count": jnp.mean(outputs["trigger_count"]),
    }
