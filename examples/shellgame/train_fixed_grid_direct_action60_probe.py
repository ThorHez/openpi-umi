"""Directly decode frame-60:75 joint actions from the optimized memory.

This diagnostic deliberately removes Pi0 flow matching from the experiment.
The input is the same 31-frame, stride-2 clip ending at raw frame 59 used by
``train_fixed_grid_action60_probe.py``.  A small deterministic readout consumes
the current-frame tokens *after* their periodic cross-attention to the frozen
history memory and regresses the normalized 16 x 32 action chunk.  Only the
first eight dimensions (seven absolute joints plus gripper) contribute loss.
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
import jax.numpy as jnp

from examples.shellgame.train_fixed_grid_action60_probe import INTEGRATED_READER_CHECKPOINT
from examples.shellgame.train_fixed_grid_action60_probe import LEROBOT_ROOT
from examples.shellgame.train_fixed_grid_action60_probe import RAW_DATASET_ROOT
from examples.shellgame.train_fixed_grid_action60_probe import _frame59_only
from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import IntegratedCheckpointLoader
from openpi.models import model as _model
from openpi.models import pi0_mem_fixed_grid_temporal as _fixed_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer


class DirectActionReadout(nn.Module):
    """Attention-pool memory-conditioned current tokens into one trajectory."""

    input_width: int = 1152
    width: int = 256
    hidden_width: int = 512
    action_horizon: int = 16
    action_dim: int = 32
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, current_tokens, state):
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(current_tokens)
        x = nn.Dense(self.width, name="token_projection", dtype=self.dtype_mm)(x)
        x = nn.tanh(x)
        scores = nn.Dense(1, name="attention", dtype=jnp.float32)(x.astype(jnp.float32))
        weights = nn.softmax(scores, axis=1)
        pooled = jnp.sum(weights * x.astype(jnp.float32), axis=1)
        pooled = nn.LayerNorm(name="pooled_ln", dtype=jnp.float32)(pooled)

        state_features = nn.Dense(64, name="state_projection", dtype=jnp.float32)(
            state.astype(jnp.float32)
        )
        state_features = nn.tanh(state_features)
        features = jnp.concatenate((pooled, state_features), axis=-1)
        features = nn.Dense(self.hidden_width, name="trajectory_hidden", dtype=jnp.float32)(features)
        features = nn.gelu(features)
        flat = nn.Dense(
            self.action_horizon * self.action_dim,
            name="trajectory_output",
            dtype=jnp.float32,
        )(features)
        return flat.reshape((-1, self.action_horizon, self.action_dim))


@dataclasses.dataclass(frozen=True)
class DirectAction60Config(_fixed_model.Pi0MemFixedGridTemporalConfig):
    def create(self, rng: at.KeyArrayLike) -> DirectAction60Model:
        return DirectAction60Model(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_for_phase(self, phase: str) -> nnx.filterlib.Filter:
        head = nnx_utils.PathRegex(r".*DirectActionReadout.*")
        if phase == "head_only":
            trainable = head
        elif phase == "joint_direct":
            history = nnx_utils.PathRegex(r".*PaliGemma/img/Transformer/FixedGridTemporalHistory_0.*")
            reader = nnx_utils.PathRegex(
                r".*PaliGemma/img/Transformer/encoderblock/"
                r"(?:HistoryLayerNorm_0|HistoryMultiHeadDotProductAttention_0).*"
            )
            trainable = nnx.Any(history, reader, head)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return nnx.Not(trainable)


class DirectAction60Model(_fixed_model.Pi0MemFixedGridTemporal):
    def __init__(self, config: DirectAction60Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.DirectActionReadout = nnx_bridge.ToNNX(
            DirectActionReadout(
                input_width=1152,
                width=config.temporal_width,
                action_horizon=config.action_horizon,
                action_dim=config.action_dim,
                dtype_mm=config.dtype,
            )
        )
        self.DirectActionReadout.lazy_init(
            jnp.zeros((1, 256, 1152), dtype=jnp.bfloat16),
            jnp.zeros((1, config.action_dim), dtype=jnp.float32),
            rngs=rngs,
        )

    def _predict_direct_actions(self, observation: _model.Observation):
        _, _, _, history_mem, encoder_auxes = self._embed_prefix_with_history_mem(observation)
        if not encoder_auxes:
            raise ValueError("Direct action probe requires at least the base_rgb image stream")
        # LeRobot also supplies wrist_rgb.  The proven integrated classifier
        # was trained/read out on base_rgb, which is the first configured
        # stream, so keep this diagnostic on the identical representation.
        current_tokens = encoder_auxes[0]["pre_ln"]
        actions = self.DirectActionReadout(
            current_tokens,
            observation.state[..., : self.action_dim],
        )
        # _embed_prefix concatenates memory from all camera streams along the
        # batch axis.  Surface only the matching base stream to diagnostics.
        base_history_mem = history_mem[: current_tokens.shape[0]]
        return actions, base_history_mem, encoder_auxes

    def compute_loss_with_memory_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        # The reader checkpoint was trained without image augmentation.  Keep
        # this bridge test deterministic so train and task-level FK eval see
        # exactly the same visual distribution.
        del train
        observation = _model.preprocess_observation(rng, observation, train=False)
        prediction, history_mem, encoder_auxes = self._predict_direct_actions(observation)
        squared_error = jnp.square(prediction - actions)
        if observation.action_loss_mask is not None:
            mask = observation.action_loss_mask[..., None, :]
        else:
            mask = jnp.asarray(self.action_loss_mask)[None, None, :]
        loss_per_timestep = jnp.sum(squared_error * mask, axis=-1) / jnp.maximum(
            jnp.sum(mask, axis=-1), 1e-8
        )
        return loss_per_timestep, {
            "history_mem": history_mem,
            "encoder_auxes": encoder_auxes,
            "history_class_logits": None,
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
        del rng, num_steps, noise
        observation = _model.preprocess_observation(None, observation, train=False)
        actions, _, _ = self._predict_direct_actions(observation)
        return actions


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    model = DirectAction60Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        num_frames=31,
        memory_every=4,
        current_frame_index=-1,
        history_memory_tokens=128,
        history_resampler_depth=1,
        history_use_current_condition=False,
        history_gate_fixed=1.0,
        diversity_weight=0.0,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=0,
        temporal_width=256,
        temporal_depth=2,
        temporal_heads=8,
        spatial_pool_factor=2,
    )
    return _config.TrainConfig(
        name=f"pi0_shellgame_fixed_grid_direct_action60_{args.phase}_260808",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_for_phase(args.phase),
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
                    num_frames=31,
                    frame_stride=2,
                )
            ],
        ),
        weight_loader=IntegratedCheckpointLoader(args.init_checkpoint),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 2),
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=args.clip_gradient_norm),
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
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(
            enabled=True,
            raw_dataset_root=RAW_DATASET_ROOT,
            robosuite_root="/data2/hzl_workspace_for_pi_mem/robosuite",
            interval=args.cup_eval_interval,
            num_episodes=args.cup_eval_episodes,
            batch_size=args.cup_eval_batch_size,
            num_sampling_steps=1,
            sample_seed=260808,
            selection_radius=0.06,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("head_only", "joint_direct"), required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=INTEGRATED_READER_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--clip-gradient-norm", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--cup-eval-interval", type=int, default=50)
    parser.add_argument("--cup-eval-episodes", type=int, default=24)
    parser.add_argument("--cup-eval-batch-size", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _trainer._filter_memory_classifier_frame_range = _frame59_only  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
