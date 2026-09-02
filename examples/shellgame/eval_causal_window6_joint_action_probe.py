"""Connect causal six-frame recurrent memory to a proven joint action head.

This is a minimal compatibility/effect probe, not a new action training run.
The causal six-frame tracker supplies final-slot probabilities to the small
deterministic action head previously trained on frame-60 joint trajectories.
Normal, batch-shuffled, and zero-memory modes isolate whether the predicted
joint direction actually depends on the new memory path.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from examples.shellgame import train_sliding_window6_event_recurrent_memory_probe as _six
from examples.shellgame import train_sliding_window_event_recurrent_memory_probe as _window
from examples.shellgame import train_three_swap_fully_visual_joint_action_probe as _old_action
from examples.shellgame.train_fixed_grid_action60_probe import LEROBOT_ROOT
from examples.shellgame.train_fixed_grid_action60_probe import _frame59_only
from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as _memory_recipe
from scripts.mem import train_pi0_mem_compress as _trainer

CAUSAL_CONDITION_INDEX = 4
DEFAULT_CAUSAL_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "shellgame_sliding_window6_event_recurrent_memory_probe/"
    "sliding_window6_event_gate_500_260821/499/params"
)
DEFAULT_ACTION_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_fully_visual_joint_action_260810/"
    "fully_visual_joint_action_head300_b72_260810/299/params"
)
CAUSAL_RAW_EVAL_ROOT = (
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_absolute_eef_phase_instruction_dataset"
)


@dataclasses.dataclass(frozen=True)
class CausalWindow6JointActionConfig(_window.SlidingWindowMemoryConfig):
    def create(self, rng: at.KeyArrayLike) -> CausalWindow6JointActionModel:
        return CausalWindow6JointActionModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_action_head(self) -> nnx.filterlib.Filter:
        action_head = nnx_utils.PathRegex(r".*HistorySemanticJointActionReadout.*")
        return nnx.Not(action_head)


class CausalWindow6JointActionModel(_window.Pi0SlidingWindowMemory):
    """Causal tracker followed by the checkpoint-compatible action head."""

    def __init__(self, config: CausalWindow6JointActionConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.raw_memory_mode = config.raw_memory_mode
        self.HistorySemanticJointActionReadout = nnx_bridge.ToNNX(
            _old_action.SemanticJointActionReadout(
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

    def _track_causal_history(self, observation: _model.Observation):
        image = observation.images.get("base_rgb")
        if image is None or image.ndim != 5 or image.shape[1] != self.history_frames:
            raise ValueError(
                f"Causal joint probe expects base_rgb [B,{self.history_frames},H,W,C], "
                f"got {None if image is None else image.shape}"
            )
        history = image[:, : self.history_frames]
        _, initial_encoder_out = self.PaliGemma.img(history[:, :1], train=False)
        initial_logits = self.HistoryFrame0InitialCupClassifier(initial_encoder_out["encoded"])
        initial_ids = jnp.argmax(initial_logits, axis=-1)

        _, history_encoder_out = self.PaliGemma.img(history, train=False)
        history_patches = history_encoder_out["with_posemb"][:, : self.history_frames]
        starts = jnp.tile(
            jnp.arange(_window.NUM_WINDOWS, dtype=jnp.int32)[None],
            (history.shape[0], 1),
        )
        outputs = self.HistorySlidingWindowRelationMemoryTracker(
            history_patches,
            initial_ids,
            starts,
            evaluate_all_windows=True,
            train=False,
        )
        return {"initial_logits": initial_logits, "initial_ids": initial_ids, **outputs}

    def _predict_direct_actions(self, observation: _model.Observation):
        tracked = self._track_causal_history(observation)
        final_slot_probabilities = jax.nn.softmax(
            tracked["stage_logits"][:, CAUSAL_CONDITION_INDEX, -1],
            axis=-1,
        )
        raw_memory = tracked["stage_memories"][:, CAUSAL_CONDITION_INDEX, -1]
        if self.raw_memory_mode == "shuffle_batch":
            final_slot_probabilities = jnp.roll(final_slot_probabilities, 1, axis=0)
            raw_memory = jnp.roll(raw_memory, 1, axis=0)
        elif self.raw_memory_mode == "zero":
            final_slot_probabilities = jnp.zeros_like(final_slot_probabilities)
            raw_memory = jnp.zeros_like(raw_memory)
        elif self.raw_memory_mode != "normal":
            raise ValueError(f"Unknown raw_memory_mode={self.raw_memory_mode!r}")
        actions = self.HistorySemanticJointActionReadout(
            jax.lax.stop_gradient(final_slot_probabilities),
            observation.state,
        )
        return actions, tracked, raw_memory

    def compute_loss_with_memory_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        del train
        observation = _model.preprocess_observation(rng, observation, train=False)
        prediction, tracked, raw_memory = self._predict_direct_actions(observation)
        squared_error = jnp.square(prediction - actions)
        mask = (
            observation.action_loss_mask[..., None, :]
            if observation.action_loss_mask is not None
            else jnp.asarray(self.action_loss_mask)[None, None, :]
        )
        loss_per_timestep = jnp.sum(squared_error * mask, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1), 1e-8)
        return loss_per_timestep, {
            "history_mem": raw_memory,
            "encoder_auxes": (),
            "history_class_logits": tracked["stage_logits"][:, CAUSAL_CONDITION_INDEX, -1],
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
        actions, _, _ = self._predict_direct_actions(observation)
        return actions


@dataclasses.dataclass(frozen=True)
class CausalMemoryActionHeadCheckpointLoader:
    causal_params_path: str
    action_params_path: str

    def load(self, params: at.Params) -> at.Params:
        target = flax.traverse_util.flatten_dict(params, sep="/")
        causal = flax.traverse_util.flatten_dict(
            _model.restore_params(self.causal_params_path, restore_type=np.ndarray),
            sep="/",
        )
        action = flax.traverse_util.flatten_dict(
            _model.restore_params(self.action_params_path, restore_type=np.ndarray),
            sep="/",
        )
        head_prefix = "HistorySemanticJointActionReadout/"
        restored = {}
        counts = {"causal_memory": 0, "action_head": 0}
        missing = []
        for key, reference in target.items():
            source = action if key.startswith(head_prefix) else causal
            candidate = source.get(key)
            if candidate is None or np.shape(candidate) != np.shape(reference):
                restored[key] = reference
                missing.append(key)
                continue
            restored[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
            counts["action_head" if key.startswith(head_prefix) else "causal_memory"] += 1
        if missing:
            raise ValueError(f"Causal-memory/action-head restore incomplete: {missing[:8]}")
        print(
            "CausalMemoryActionHeadCheckpointLoader: "
            f"causal_memory={counts['causal_memory']}, action_head={counts['action_head']}, missing=0"
        )
        return flax.traverse_util.unflatten_dict(restored, sep="/")


def build_config(args: argparse.Namespace):
    model_values = {
        field.name: getattr(_memory_recipe.MODEL_CONFIG, field.name)
        for field in dataclasses.fields(CausalWindow6JointActionConfig)
    }
    model_values.update(
        num_frames=60,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        history_classifier_num_classes=0,
        raw_memory_mode=args.raw_memory_mode,
    )
    model = CausalWindow6JointActionConfig(**model_values)
    return _config.TrainConfig(
        name="pi0_shellgame_causal_window6_joint_action_probe_260821",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_action_head(),
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
                    num_frames=60,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=CausalMemoryActionHeadCheckpointLoader(
            causal_params_path=args.causal_checkpoint,
            action_params_path=args.action_checkpoint,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=3e-4,
            decay_steps=2,
            decay_lr=3e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=10.0),
        ema_decay=None,
        num_train_steps=0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=1,
        keep_period=1,
        val_ratio=0.1,
        eval_interval=1,
        eval_batches=args.eval_batches,
        wandb_enabled=False,
        overwrite=args.overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=_config.ShellgameCupEvalConfig(
            enabled=True,
            # The original absolute-joint episode directories were removed
            # during disk cleanup.  This 5k set has the same 60-frame visual
            # task and stores measured joint_pos for the identical FK metric.
            raw_dataset_root=CAUSAL_RAW_EVAL_ROOT,
            robosuite_root="/data2/hzl_workspace_for_pi_mem/robosuite",
            interval=1,
            num_episodes=args.cup_eval_episodes,
            batch_size=args.cup_eval_batch_size,
            num_sampling_steps=1,
            sample_seed=260821,
            selection_radius=0.06,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--causal-checkpoint", default=DEFAULT_CAUSAL_CHECKPOINT)
    parser.add_argument("--action-checkpoint", default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument(
        "--raw-memory-mode",
        choices=("normal", "shuffle_batch", "zero"),
        default="normal",
    )
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--cup-eval-episodes", type=int, default=30)
    parser.add_argument("--cup-eval-batch-size", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _six._configure_six_frame_globals()  # noqa: SLF001
    _window.ENABLE_CAUSAL_EVAL_SELECTIONS = True
    _trainer._filter_memory_classifier_frame_range = _frame59_only  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
