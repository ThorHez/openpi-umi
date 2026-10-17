"""Train a frozen-memory -> absolute-joint action diagnostic.

The observation is exactly raw frames 0..59 (60 frames, stride 1) and the
supervised action chunk is raw joint_pos[60:76].  The already validated fully
visual tracker is frozen: frame 0 predicts the initial ball slot, three visual
swap classifiers update the compact recurrent memory, and the final memory
readout predicts the ball slot.  Only a small deterministic action head is
trained to map that semantic state plus the current robot state to a 16-step
absolute-joint trajectory.  FK cup evaluation therefore isolates whether a
correct memory state can be converted into the demonstrated joint motion.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax
import jax.numpy as jnp

from examples.shellgame import eval_three_swap_fully_visual_relation_memory_probe as _fully
from examples.shellgame import train_three_swap_visual_semantic_readout_memory_probe as _semantic
from examples.shellgame.train_fixed_grid_action60_probe import LEROBOT_ROOT
from examples.shellgame.train_fixed_grid_action60_probe import RAW_DATASET_ROOT
from examples.shellgame.train_fixed_grid_action60_probe import _frame59_only
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer


ACTION_INPUT_FRAMES = 60


class SemanticJointActionReadout(nn.Module):
    """Decode a short normalized joint trajectory from the tracked slot."""

    state_dim: int = 8
    hidden_width: int = 256
    action_horizon: int = 16
    action_dim: int = 32

    @nn.compact
    def __call__(self, final_slot_probabilities, state):
        if final_slot_probabilities.shape[-1] != 3:
            raise ValueError(
                f"Expected three final-slot probabilities, got {final_slot_probabilities.shape}"
            )
        state = state[..., : self.state_dim].astype(jnp.float32)
        features = jnp.concatenate(
            (final_slot_probabilities.astype(jnp.float32), state), axis=-1
        )
        features = nn.Dense(self.hidden_width, name="input_projection")(features)
        features = nn.gelu(features)
        residual = features
        features = nn.Dense(self.hidden_width, name="hidden_0")(features)
        features = nn.gelu(features)
        features = nn.Dense(self.hidden_width, name="hidden_1")(features)
        features = nn.gelu(features + residual)
        flat = nn.Dense(
            self.action_horizon * self.action_dim,
            name="trajectory_output",
        )(features)
        return flat.reshape((-1, self.action_horizon, self.action_dim))


@dataclasses.dataclass(frozen=True)
class FullyVisualJointActionConfig(_fully.FullyVisualRelationMemoryConfig):
    def create(self, rng: at.KeyArrayLike) -> FullyVisualJointActionModel:
        return FullyVisualJointActionModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_joint_action(self) -> nnx.filterlib.Filter:
        action_head = nnx_utils.PathRegex(r".*HistorySemanticJointActionReadout.*")
        return nnx.Not(action_head)


class FullyVisualJointActionModel(_fully.FullyVisualRelationMemoryModel):
    def __init__(self, config: FullyVisualJointActionConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.HistorySemanticJointActionReadout = nnx_bridge.ToNNX(
            SemanticJointActionReadout(
                state_dim=8,
                hidden_width=256,
                action_horizon=config.action_horizon,
                action_dim=config.action_dim,
            )
        )
        self.HistorySemanticJointActionReadout.lazy_init(
            jnp.zeros((1, 3), dtype=jnp.float32),
            jnp.zeros((1, config.action_dim), dtype=jnp.float32),
            rngs=rngs,
        )

    def _track_history(self, observation: _model.Observation):
        image = observation.images["base_rgb"]
        if image.ndim == 4:
            image = image[:, None]
        if image.ndim != 5 or image.shape[1] != ACTION_INPUT_FRAMES:
            raise ValueError(
                f"Joint action probe expects [B,{ACTION_INPUT_FRAMES},H,W,C], got {image.shape}"
            )

        # Preserve the exact frozen pathways that achieved 100% held-out
        # tracking.  The extra frame used by the classification-only probe was
        # never consumed by the tracker, so frames 0..59 are sufficient here.
        _, initial_encoder_out = self.PaliGemma.img(image[:, :1], train=False)
        frame0_features = initial_encoder_out["encoded"]
        initial_logits = self.HistoryFrame0InitialCupClassifier(frame0_features)
        initial_ids = jnp.argmax(initial_logits, axis=-1)

        _, history_encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = history_encoder_out["with_posemb"][:, : _semantic.HISTORY_FRAMES]
        joint_logits, stage_logits, stage_memories, relation_logits, relation_ids = (
            self.HistoryThreeSwapVisualRelationMemoryTracker(history_patches, initial_ids)
        )
        return {
            "joint_logits": joint_logits,
            "stage_logits": stage_logits,
            "stage_memories": stage_memories,
            "initial_logits": initial_logits,
            "initial_ids": initial_ids,
            "relation_logits": relation_logits,
            "relation_ids": relation_ids,
        }

    def _predict_direct_actions(self, observation: _model.Observation):
        tracked = self._track_history(observation)
        final_slot_probabilities = jax.lax.stop_gradient(
            jax.nn.softmax(tracked["stage_logits"][:, -1], axis=-1)
        )
        actions = self.HistorySemanticJointActionReadout(
            final_slot_probabilities,
            observation.state,
        )
        return actions, tracked

    def compute_history_classification(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ):
        observation = _model.preprocess_observation(rng, observation, train=train)
        tracked = self._track_history(observation)
        return tracked["joint_logits"], {
            "history_mem": tracked["stage_memories"].reshape(
                -1,
                tracked["stage_memories"].shape[-2],
                tracked["stage_memories"].shape[-1],
            ),
            "stage_logits": tracked["stage_logits"],
            "initial_logits": tracked["initial_logits"],
            "initial_ids": tracked["initial_ids"],
            "relation_logits": tracked["relation_logits"],
            "relation_ids": tracked["relation_ids"],
            "encoder_auxes": (),
        }

    def compute_loss_with_memory_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        # All frozen visual modules were validated without augmentation.
        del train
        observation = _model.preprocess_observation(rng, observation, train=False)
        prediction, tracked = self._predict_direct_actions(observation)
        squared_error = jnp.square(prediction - actions)
        if observation.action_loss_mask is not None:
            mask = observation.action_loss_mask[..., None, :]
        else:
            mask = jnp.asarray(self.action_loss_mask)[None, None, :]
        loss_per_timestep = jnp.sum(squared_error * mask, axis=-1) / jnp.maximum(
            jnp.sum(mask, axis=-1), 1e-8
        )
        return loss_per_timestep, {
            "history_mem": tracked["stage_memories"][:, -1],
            "encoder_auxes": (),
            "history_class_logits": tracked["joint_logits"],
        }

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        loss, _ = self.compute_loss_with_memory_aux(rng, observation, actions, train=train)
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
        actions, _ = self._predict_direct_actions(observation)
        return actions


@dataclasses.dataclass(frozen=True)
class FullyVisualJointActionCheckpointLoader:
    """Restore the proven visual/memory stack and leave only the new head random."""

    visual_params_path: str
    initial_params_path: str
    memory_params_path: str

    def load(self, params: at.Params) -> at.Params:
        flat = flax.traverse_util.flatten_dict(params, sep="/")
        head_prefix = "HistorySemanticJointActionReadout/"
        base_flat = {key: value for key, value in flat.items() if not key.startswith(head_prefix)}
        head_flat = {key: value for key, value in flat.items() if key.startswith(head_prefix)}
        restored_base = _fully.FullyVisualCombinedCheckpointLoader(
            visual_params_path=self.visual_params_path,
            initial_params_path=self.initial_params_path,
            memory_params_path=self.memory_params_path,
        ).load(flax.traverse_util.unflatten_dict(base_flat, sep="/"))
        restored_flat = flax.traverse_util.flatten_dict(restored_base, sep="/")
        restored_flat.update(head_flat)
        print(
            "FullyVisualJointActionCheckpointLoader: "
            f"restored={len(restored_flat) - len(head_flat)}, random_action_head={len(head_flat)}"
        )
        return flax.traverse_util.unflatten_dict(restored_flat, sep="/")


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    parent = _fully.build_config(args, build_three_swap_labels())
    parent_fields = {
        field.name: getattr(parent.model, field.name) for field in dataclasses.fields(parent.model)
    }
    parent_fields.update(
        num_frames=ACTION_INPUT_FRAMES,
        history_classifier_num_classes=0,
    )
    model = FullyVisualJointActionConfig(**parent_fields)
    return dataclasses.replace(
        parent,
        name="pi0_shellgame_three_swap_fully_visual_joint_action_260810",
        model=model,
        freeze_filter=model.get_freeze_filter_joint_action(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=str(LEROBOT_ROOT),
                    assets=_config.AssetsConfig(asset_id=".", assets_dir=str(LEROBOT_ROOT)),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=ACTION_INPUT_FRAMES,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=FullyVisualJointActionCheckpointLoader(
            visual_params_path=args.init_checkpoint,
            initial_params_path=args.initial_checkpoint,
            memory_params_path=args.memory_checkpoint,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 2),
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=10.0),
        ema_decay=None,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=max(args.steps, 1),
        keep_period=max(args.steps, 1),
        val_ratio=0.1,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        wandb_enabled=False,
        overwrite=args.overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(
            enabled=True,
            raw_dataset_root=RAW_DATASET_ROOT,
            robosuite_root="/data2/hzl_workspace_for_pi_mem/robosuite",
            interval=args.cup_eval_interval,
            num_episodes=args.cup_eval_episodes,
            batch_size=args.cup_eval_batch_size,
            num_sampling_steps=1,
            sample_seed=260810,
            selection_radius=0.06,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=_semantic.DEFAULT_PAIR_CHECKPOINT)
    parser.add_argument("--initial-checkpoint", default=_fully.DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument("--memory-checkpoint", default=_fully.DEFAULT_MEMORY_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--cup-eval-interval", type=int, default=50)
    parser.add_argument("--cup-eval-episodes", type=int, default=24)
    parser.add_argument("--cup-eval-batch-size", type=int, default=6)
    parser.add_argument("--encoder-width", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--memory-width", type=int, default=64)
    parser.add_argument("--memory-depth", type=int, default=2)
    parser.add_argument("--memory-heads", type=int, default=4)
    parser.add_argument("--adapter-heads", type=int, default=4)
    parser.add_argument("--memory-tokens", type=int, default=128)
    parser.add_argument("--current-tokens", type=int, default=256)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--overfit-samples-per-class", type=int, default=0)
    parser.add_argument("--video-mode", choices=("normal",), default="normal")
    parser.add_argument("--initial-mode", choices=("normal",), default="normal")
    parser.add_argument("--relation-mode", choices=("one_hot",), default="one_hot")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _trainer._filter_memory_classifier_frame_range = _frame59_only  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
