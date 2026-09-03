"""Deterministic M5 semantic-to-action probe for the real ShellGame data.

M5 deliberately does not train the Pi flow policy.  It freezes the complete
241-frame visual memory checkpoint and trains only a small deterministic head
that maps a three-way final-cup semantic vector plus the current 10-D robot
state to the first 16 current-relative EEF targets.

Two semantic sources are supported:

* ``memory`` uses the frozen tracker's predicted final-cup probabilities;
* ``oracle`` uses the dataset's ground-truth final-cup one-hot vector.

The oracle variant isolates the action labels/coordinate contract.  The memory
variant then tests whether the frozen semantic prediction can drive the same
action mapping.  Both variants sample only episode frame 241, so target action
row ``h`` is command frame ``242 + h`` and is anchored at measured frame 241.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
import openpi.shared.nnx_utils as nnx_utils
from openpi.shared import array_typing as at
from openpi.tasks.shellgame import pi0_mem_semantic_action as _shellgame_model
from openpi.tasks.shellgame import semantic_memory
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _stage2


LABELS_PATH = Path("/data2/hzl_workspace_for_pi_mem/labels_merged_306_degap.jsonl")
M5_CURRENT_FRAME = _stage2.CURRENT_START_FRAME
M5_CONFIG_NAME = "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5"


def load_final_cup_labels(path: Path = LABELS_PATH) -> tuple[int, ...]:
    """Load a dense episode_index -> final_cup table with contract checks."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episode_ids = [int(row["episode_id"]) for row in rows]
    if episode_ids != list(range(len(rows))):
        raise ValueError(f"Expected contiguous episode ids in {path}, got {episode_ids[:8]}")
    labels = tuple(int(row["final_cup"]) for row in rows)
    invalid = [(index, label) for index, label in enumerate(labels) if label not in (0, 1, 2)]
    if invalid:
        raise ValueError(f"Invalid final_cup labels in {path}: {invalid[:8]}")
    return labels


@dataclasses.dataclass(frozen=True)
class RealWristM5Config(_shellgame_model.Pi0MemSemanticActionConfig):
    """Model config for the frozen-memory deterministic real-robot M5 probe."""

    semantic_source: str = "memory"
    oracle_final_cups: tuple[int, ...] = ()
    diagnostic_state_dim: int = _stage2.ACTION_DIM

    def create(self, rng: at.KeyArrayLike) -> RealWristM5Model:
        return RealWristM5Model(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_m5(self) -> nnx.filterlib.Filter:
        probe_head = nnx_utils.PathRegex(r".*HistorySemanticJointActionReadout.*")
        return nnx.Not(probe_head)


class RealWristM5Model(_shellgame_model.Pi0MemSemanticAction):
    """Frozen semantic tracker with a deterministic 16x10 EEF action probe."""

    def __init__(self, config: RealWristM5Config, rngs: nnx.Rngs):
        if config.semantic_source not in ("memory", "oracle"):
            raise ValueError(
                f"semantic_source must be 'memory' or 'oracle', got {config.semantic_source!r}"
            )
        if config.semantic_source == "oracle" and not config.oracle_final_cups:
            raise ValueError("oracle semantic source requires oracle_final_cups")
        super().__init__(config, rngs)
        self.semantic_source = config.semantic_source
        # Keep labels as static Python data in the NNX graph. Plain JAX array
        # leaves are not legal Module attributes unless wrapped as Variables.
        self.oracle_final_cups = tuple(int(value) for value in config.oracle_final_cups)
        self.semantic_memory_tokens = int(config.semantic_memory_tokens)
        self.semantic_memory_width = int(config.semantic_memory_width)

        # Replace the checkpoint-compatible simulation head (8-D joint state)
        # with the real robot's complete 10-D episode-first EEF state head.
        self.HistorySemanticJointActionReadout = nnx_bridge.ToNNX(
            _shellgame_model.SemanticJointActionReadout(
                state_dim=config.diagnostic_state_dim,
                hidden_width=256,
                action_horizon=config.action_horizon,
                action_dim=config.action_dim,
            )
        )
        self.HistorySemanticJointActionReadout.lazy_init(
            jnp.zeros((1, semantic_memory.NUM_CUPS), dtype=jnp.float32),
            jnp.zeros((1, config.action_dim), dtype=jnp.float32),
            rngs=rngs,
        )

    def _semantic_input(self, observation: _model.Observation):
        batch_size = observation.state.shape[0]
        if self.semantic_source == "oracle":
            if observation.episode_index is None:
                raise ValueError("M5 oracle mode requires episode_index in every observation")
            episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
            label_table = jnp.asarray(self.oracle_final_cups, dtype=jnp.int32)
            safe_index = jnp.clip(episode_index, 0, label_table.shape[0] - 1)
            final_cup = label_table[safe_index]
            probabilities = jax.nn.one_hot(
                final_cup,
                semantic_memory.NUM_CUPS,
                dtype=jnp.float32,
            )
            raw_memory = jnp.zeros(
                (batch_size, self.semantic_memory_tokens, self.semantic_memory_width),
                dtype=jnp.float32,
            )
            return probabilities, raw_memory

        tracked = self._track_history(observation)
        probabilities = jax.nn.softmax(tracked["stage_logits"][:, -1], axis=-1)
        raw_memory = tracked["stage_memories"][:, -1]
        return probabilities, raw_memory

    def _predict_direct_actions(self, observation: _model.Observation):
        probabilities, raw_memory = self._semantic_input(observation)
        probabilities = jax.lax.stop_gradient(probabilities)
        raw_memory = jax.lax.stop_gradient(raw_memory)
        actions = self.HistorySemanticJointActionReadout(
            probabilities,
            observation.state,
        )
        return actions, probabilities, raw_memory

    def compute_loss_with_memory_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        # The only trainable module consumes semantic probabilities and state;
        # stochastic image augmentation would merely perturb a frozen tracker.
        del train
        observation = _model.preprocess_observation(rng, observation, train=False)
        prediction, probabilities, raw_memory = self._predict_direct_actions(observation)
        squared_error = jnp.square(prediction - actions)
        if observation.action_loss_mask is not None:
            dimension_mask = observation.action_loss_mask[..., None, :]
        else:
            dimension_mask = jnp.asarray(self.action_loss_mask)[None, None, :]
        loss_per_timestep = jnp.sum(
            squared_error * dimension_mask,
            axis=-1,
        ) / jnp.maximum(jnp.sum(dimension_mask, axis=-1), 1e-8)

        semantic_ids = jnp.argmax(probabilities, axis=-1)
        xyz_error = prediction[..., :3] - actions[..., :3]
        extra_metrics = {
            "m5_xyz_rmse_normalized": jnp.sqrt(jnp.mean(jnp.square(xyz_error))),
            "m5_semantic_left_fraction": jnp.mean(semantic_ids == 0),
            "m5_semantic_middle_fraction": jnp.mean(semantic_ids == 1),
            "m5_semantic_right_fraction": jnp.mean(semantic_ids == 2),
            "m5_semantic_entropy": jnp.mean(
                -jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, 1e-8)), axis=-1)
            ),
        }
        return loss_per_timestep, {
            "history_mem": raw_memory,
            "encoder_auxes": (),
            "history_class_logits": probabilities,
            "extra_metrics": extra_metrics,
        }

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        loss, _ = self.compute_loss_with_memory_aux(
            rng,
            observation,
            actions,
            train=train,
        )
        return loss

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 1,
        noise=None,
    ) -> _model.Actions:
        del num_steps, noise
        observation = _model.preprocess_observation(rng, observation, train=False)
        actions, _, _ = self._predict_direct_actions(observation)
        return actions


def build_model_config(semantic_source: str) -> RealWristM5Config:
    """Clone the deployed Stage2 tensor contract into the M5 model type."""
    if semantic_source not in ("memory", "oracle"):
        raise ValueError(f"Unsupported semantic_source={semantic_source!r}")
    base_fields = {
        field.name: getattr(_stage2.MODEL_CONFIG, field.name)
        for field in dataclasses.fields(_stage2.MODEL_CONFIG)
    }
    return RealWristM5Config(
        **base_fields,
        semantic_source=semantic_source,
        oracle_final_cups=(load_final_cup_labels() if semantic_source == "oracle" else ()),
        diagnostic_state_dim=_stage2.ACTION_DIM,
    )


def make_train_config(
    *,
    semantic_source: str,
    exp_name: str,
    checkpoint: str = _stage2.MEMORY_CHECKPOINT,
    steps: int = 1_000,
    warmup_steps: int = 30,
    peak_lr: float = 3e-4,
    batch_size: int = 4,
    num_workers: int = 8,
    fsdp_devices: int = 4,
    eval_interval: int = 50,
    eval_batches: int = 8,
    save_interval: int = 100,
    overwrite: bool = False,
) -> Any:
    """Build an M5 training config without registering a deployment policy."""
    from openpi.training import config as _config

    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0 or batch_size % fsdp_devices != 0:
        raise ValueError("batch_size must be positive and divisible by fsdp_devices")
    model = build_model_config(semantic_source)
    data_cls = _stage2.data_config_type(_config)
    return _config.TrainConfig(
        name=M5_CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_m5(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                data_cls(
                    repo_id=_stage2.DATASET_ROOT,
                    assets=_config.AssetsConfig(
                        asset_id=".",
                        assets_dir=_stage2.DATASET_ROOT,
                    ),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * _stage2.ACTION_DIM
                        + (0.0,) * (32 - _stage2.ACTION_DIM),
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    min_frame_index=M5_CURRENT_FRAME,
                    max_frame_index=M5_CURRENT_FRAME,
                )
            ],
        ),
        weight_loader=_weight_loaders.CheckpointWeightLoaderReinitialize(
            params_path=checkpoint,
            reinitialize_regex=r".*HistorySemanticJointActionReadout.*",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(warmup_steps, max(steps - 1, 0)),
            peak_lr=peak_lr,
            decay_steps=max(steps, 2),
            decay_lr=peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=10.0),
        ema_decay=None,
        num_train_steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        seed=42,
        log_interval=10,
        save_interval=min(save_interval, steps),
        # A complete checkpoint is about 5.2 GiB because it includes the
        # frozen backbone. Retain only the latest intermediate/final M5 state.
        keep_period=steps,
        val_ratio=0.1,
        eval_interval=min(eval_interval, steps),
        eval_batches=eval_batches,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(enabled=False),
    )
