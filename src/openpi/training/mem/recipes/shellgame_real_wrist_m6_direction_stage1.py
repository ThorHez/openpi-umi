"""Real ShellGame M6 direction-interface stage 1.

This keeps the deployed M6 raw-memory/query/cross-attention/Pi0.5 path and
trains only the first post-history decision row.  Direction classification is
always measured from the clean action reconstructed from the flow velocity;
its auxiliary loss is opt-in and is disabled for the initial flow-only probe.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq

from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base
from openpi.shared import array_typing as at
from openpi.tasks.shellgame import pi0_mem_semantic_action as _shellgame_model
from openpi.training.mem.recipes import shellgame_real_wrist_m5 as _m5
from openpi.training.mem.recipes import shellgame_real_wrist_m6 as _m6
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders


CONFIG_NAME = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_stage1"
LABELS_PATH = Path("/data2/hzl_workspace_for_pi_mem/labels_merged_306_degap.jsonl")
DECISION_FRAME = _stage2.CURRENT_START_FRAME


def _load_split() -> tuple[list[int], list[int]]:
    root = Path(_stage2.DATASET_ROOT)
    audit = json.loads((root / "conversion_audit.json").read_text(encoding="utf-8"))
    validation = sorted(int(value) for value in audit["validation_episode_ids"])
    validation_set = set(validation)
    training = [episode for episode in range(int(audit["episodes"])) if episode not in validation_set]
    if len(training) != 275 or len(validation) != 31:
        raise ValueError(f"Expected seed-42 split 275/31, got {len(training)}/{len(validation)}")
    return training, validation


def load_final_cups() -> tuple[int, ...]:
    return _m5.load_final_cup_labels(LABELS_PATH)


def _raw_xyz_centroids() -> np.ndarray:
    labels = load_final_cups()
    training, _ = _load_split()
    grouped: list[list[np.ndarray]] = [[], [], []]
    root = Path(_stage2.DATASET_ROOT)
    for episode_id in training:
        path = root / "data" / f"chunk-{episode_id // 1000:03d}" / f"episode_{episode_id:06d}.parquet"
        table = pq.read_table(path, columns=["frame_index", "actions"])
        frames = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)
        rows = np.flatnonzero(frames == DECISION_FRAME)
        if rows.size != 1:
            raise ValueError(f"episode {episode_id} has {rows.size} decision rows")
        action = np.asarray(table.column("actions")[int(rows[0])].as_py(), dtype=np.float32)
        grouped[labels[episode_id]].append(action[..., :3])
    return np.stack([np.mean(np.stack(items), axis=0) for items in grouped]).astype(np.float32)


def normalized_xyz_centroids() -> tuple[tuple[tuple[float, ...], ...], ...]:
    root = Path(_stage2.DATASET_ROOT)
    payload = json.loads((root / "norm_stats.json").read_text(encoding="utf-8"))["norm_stats"]["actions"]
    low = np.asarray(payload["min"], dtype=np.float32)[:3]
    high = np.asarray(payload["max"], dtype=np.float32)[:3]
    centroids = 2.0 * (_raw_xyz_centroids() - low) / np.maximum(high - low, 1e-7) - 1.0
    return tuple(tuple(tuple(float(value) for value in xyz) for xyz in chunk) for chunk in centroids)


@dataclasses.dataclass(frozen=True)
class RealM6DirectionStage1Config(_shellgame_model.Pi0MemSemanticActionConfig):
    final_cups: tuple[int, ...] = ()
    direction_xyz_centroids: tuple[tuple[tuple[float, ...], ...], ...] = ()
    direction_loss_weight: float = 0.0
    direction_temperature: float = 5e-4
    direction_early_stop_metric: str = "val/direction_clean_accuracy"
    direction_early_stop_min_evals: int = 3
    direction_early_stop_patience: int = 3
    direction_early_stop_min_delta: float = 0.02
    direction_early_stop_failure_below: float = 0.45
    direction_early_stop_success_above: float = 0.85

    def create(self, rng: at.KeyArrayLike) -> "RealM6DirectionStage1Model":
        return RealM6DirectionStage1Model(self, rngs=nnx.Rngs(rng))


class RealM6DirectionStage1Model(_shellgame_model.Pi0MemSemanticAction):
    """M6 deployment model with an auxiliary clean-trajectory direction loss."""

    def __init__(self, config: RealM6DirectionStage1Config, rngs: nnx.Rngs):
        if len(config.final_cups) != 306:
            raise ValueError("direction stage 1 requires 306 final-cup labels")
        if np.shape(config.direction_xyz_centroids) != (3, config.action_horizon, 3):
            raise ValueError("direction_xyz_centroids must have shape [3, action_horizon, 3]")
        if config.direction_loss_weight < 0 or config.direction_temperature <= 0:
            raise ValueError("direction loss weight must be nonnegative and temperature positive")
        super().__init__(config, rngs)
        self.final_cups = tuple(int(value) for value in config.final_cups)
        self.direction_xyz_centroids = tuple(
            tuple(tuple(float(value) for value in xyz) for xyz in chunk)
            for chunk in config.direction_xyz_centroids
        )
        self.direction_loss_weight = float(config.direction_loss_weight)
        self.direction_temperature = float(config.direction_temperature)

    def compute_loss_with_memory_aux(self, rng, observation, actions, *, train=False):
        del train
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=False)
        if observation.frame_index is None or observation.episode_index is None:
            raise ValueError("direction stage 1 requires frame_index and episode_index")

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions

        raw_memory, memory_tokens, tracked = self._raw_and_resampled_memory(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_current_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_tokens = self.ActionMemoryCrossAttention(suffix_tokens, memory_tokens)
        input_mask = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
        ar_mask = jnp.concatenate((prefix_ar_mask, suffix_ar_mask), axis=0)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=_base.make_attn_mask(input_mask, ar_mask),
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        squared_error = jnp.square(velocity - target_velocity)

        dim_mask = (
            observation.action_loss_mask[..., None, :]
            if observation.action_loss_mask is not None
            else jnp.asarray(self.action_loss_mask)[None, None, :]
        )
        dimension_weights = jnp.ones((self.action_dim,), dtype=jnp.float32)
        dimension_weights = dimension_weights.at[self.gripper_action_index].set(self.gripper_loss_weight)
        dimension_weights = dimension_weights.at[self.real_action_dim :].set(0.0)
        dim_mask = dim_mask * dimension_weights[None, None, :]
        flow_loss = jnp.sum(squared_error * dim_mask, axis=-1) / jnp.maximum(jnp.sum(dim_mask, axis=-1), 1e-8)

        frame_index = jnp.asarray(observation.frame_index, dtype=jnp.int32)
        future_offsets = 1 + jnp.arange(self.action_horizon, dtype=jnp.int32)
        last_frame = (
            jnp.asarray(observation.episode_length, dtype=jnp.int32) - 1
            if observation.episode_length is not None
            else jnp.full_like(frame_index, self.last_episode_frame)
        )
        temporal_valid = frame_index[..., None] + future_offsets <= last_frame[..., None]
        valid_count = jnp.sum(temporal_valid, axis=-1, keepdims=True)
        flow_loss = flow_loss * temporal_valid.astype(flow_loss.dtype) * (
            self.action_horizon / jnp.maximum(valid_count, 1)
        )

        # For x_t=t*noise+(1-t)*action and u=noise-action, clean action is x_t-t*u.
        predicted_clean_xyz = (x_t - time_expanded * velocity)[..., :3]
        centroids = jnp.asarray(self.direction_xyz_centroids, dtype=jnp.float32)
        mean_squared_distances = jnp.mean(
            jnp.square(predicted_clean_xyz[:, None, :, :] - centroids[None, :, :, :]),
            axis=(-2, -1),
        )
        episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
        label_table = jnp.asarray(self.final_cups, dtype=jnp.int32)
        safe_episode = jnp.clip(episode_index, 0, label_table.shape[0] - 1)
        labels = label_table[safe_episode]
        direction_logits = -mean_squared_distances / self.direction_temperature
        direction_loss = optax.softmax_cross_entropy_with_integer_labels(direction_logits, labels)
        total_loss = (
            flow_loss
            if self.direction_loss_weight == 0.0
            else flow_loss + self.direction_loss_weight * direction_loss[:, None]
        )
        direction_accuracy = jnp.mean(jnp.argmax(direction_logits, axis=-1) == labels)

        return total_loss, {
            "history_mem": raw_memory,
            "encoder_auxes": (),
            "history_class_logits": tracked["joint_logits"],
            "temporal_valid_fraction": jnp.mean(temporal_valid),
            "extra_metrics": {
                "flow_loss_only": jnp.mean(flow_loss),
                "direction_loss": jnp.mean(direction_loss),
                "direction_clean_accuracy": direction_accuracy,
                "direction_weight": jnp.asarray(self.direction_loss_weight, dtype=jnp.float32),
            },
        }


def build_model_config(direction_loss_weight: float, direction_temperature: float) -> RealM6DirectionStage1Config:
    fields = {field.name: getattr(_stage2.MODEL_CONFIG, field.name) for field in dataclasses.fields(_stage2.MODEL_CONFIG)}
    return RealM6DirectionStage1Config(
        **fields,
        final_cups=load_final_cups(),
        direction_xyz_centroids=normalized_xyz_centroids(),
        direction_loss_weight=direction_loss_weight,
        direction_temperature=direction_temperature,
    )


def make_train_config(
    *,
    exp_name: str,
    checkpoint: str = _m6.DEFAULT_M5_CHECKPOINT,
    steps: int = 2_000,
    schedule_steps: int | None = None,
    warmup_steps: int = 100,
    peak_lr: float = 3e-5,
    batch_size: int = 8,
    eval_batch_size: int | None = None,
    num_workers: int = 16,
    fsdp_devices: int = 8,
    eval_interval: int = 50,
    eval_batches: int = 3,
    direction_loss_weight: float = 0.0,
    direction_temperature: float = 5e-4,
    enable_direction_early_stop: bool = True,
    save_interval: int = 5_000,
    resume: bool = False,
    overwrite: bool = False,
) -> Any:
    from openpi.training import config as _config

    model = build_model_config(direction_loss_weight, direction_temperature)
    if not enable_direction_early_stop:
        model = dataclasses.replace(model, direction_early_stop_metric="")
    schedule_steps = steps if schedule_steps is None else schedule_steps
    if schedule_steps < steps:
        raise ValueError("schedule_steps must be >= steps")
    data_cls = _m6.data_config_type(_config)
    params_path = Path(checkpoint)
    if params_path.name != "params":
        params_path /= "params"
    reset_modules = r".*(HistorySemanticJointActionReadout|HistoryRawMemoryQueryResampler|ActionMemoryCrossAttention).*"
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_memory_interface_finetune(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                data_cls(
                    repo_id=_stage2.DATASET_ROOT,
                    assets=_config.AssetsConfig(asset_id=".", assets_dir=_stage2.DATASET_ROOT),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * _stage2.ACTION_DIM + (0.0,) * (32 - _stage2.ACTION_DIM),
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    min_frame_index=DECISION_FRAME,
                    max_frame_index=DECISION_FRAME,
                )
            ],
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderReinitialize(
            params_path=str(params_path), reinitialize_regex=reset_modules
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(warmup_steps, max(steps - 1, 0)),
            peak_lr=peak_lr,
            decay_steps=max(schedule_steps, 2),
            decay_lr=peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=steps,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        seed=42,
        log_interval=10,
        save_interval=min(save_interval, steps),
        keep_period=steps,
        resume=resume,
        val_ratio=0.1,
        eval_interval=min(eval_interval, steps),
        eval_batches=eval_batches,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(enabled=False),
    )


def filter_frame241_balanced_indices(dataset, indices, classifier_config):
    """Downsample each split to equal final-cup counts at exactly frame 241."""
    del classifier_config
    current = dataset
    hf_dataset = None
    sample_indices = None
    while current is not None:
        if sample_indices is None:
            sample_indices = getattr(current, "sample_indices", None)
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            break
        current = getattr(current, "_dataset", None)
    if hf_dataset is None:
        raise ValueError("balanced direction sampler could not find HuggingFace dataset")
    frames = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    episodes = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    if sample_indices is not None:
        mapped = np.asarray(sample_indices, dtype=np.int64)
        frames, episodes = frames[mapped], episodes[mapped]
    selected = np.asarray(indices, dtype=np.int64)
    selected = selected[frames[selected] == DECISION_FRAME]
    labels = np.asarray(load_final_cups(), dtype=np.int64)[episodes[selected]]
    groups = [selected[labels == cup] for cup in range(3)]
    per_class = min(len(group) for group in groups)
    if per_class <= 0:
        raise ValueError("balanced direction sampler found an empty class")
    rng = np.random.default_rng(42 + int(np.min(episodes[selected])))
    balanced = np.concatenate([rng.permutation(group)[:per_class] for group in groups])
    return rng.permutation(balanced).tolist()
