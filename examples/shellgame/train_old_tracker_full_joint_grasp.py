"""Train full approach/descend/grasp/lift actions on the proven old tracker.

The visual tracker always consumes raw episode frames 0..59.  A 61st image is
the dynamic current observation at frame ``t`` and is used only by Pi0's
current-image prefix. Targets remain 16-step absolute joint + gripper-width
chunks, enabling closed-loop replanning without shifting the tracker's time
axis.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from examples.shellgame.fixed_prefix_current_video_dataset import FixedPrefixCurrentVideoDataset
from examples.shellgame import train_three_swap_query_crossattn_pi_joint_action_probe as _old
from examples.shellgame.train_fixed_grid_action60_probe import LEROBOT_ROOT
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import config_pi0_mem as _config_pi0_mem
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from scripts.mem import train_pi0_mem_compress as _trainer


OLD_QUERY_ACTION_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_query_crossattn_pi_joint_action_260810/"
    "query_crossattn_pi_flow_action300_b12_260810/299/params"
)
HISTORY_FRAMES = 60
TOTAL_INPUT_FRAMES = 61
LAST_EPISODE_FRAME = 154


@dataclasses.dataclass(frozen=True)
class OldTrackerFullJointGraspConfig(_old.QueryCrossAttnPiJointActionConfig):
    gripper_loss_weight: float = 4.0
    real_action_dim: int = 8
    gripper_action_index: int = 7
    last_episode_frame: int = LAST_EPISODE_FRAME

    def create(self, rng: at.KeyArrayLike) -> "OldTrackerFullJointGraspModel":
        return OldTrackerFullJointGraspModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_full_action(self) -> nnx.filterlib.Filter:
        # Preserve the proven tracker, query resampler, and memory read. Only
        # Pi0.5's action expert and action/time projections are optimized.
        action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm/.*_1.*")
        action_modules = nnx_utils.PathRegex(
            r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"
        )
        return nnx.Not(nnx.Any(action_expert, action_modules))


class OldTrackerFullJointGraspModel(_old.QueryCrossAttnPiJointActionModel):
    """Old tracker on frames 0..59; action prefix reads the final current frame."""

    def __init__(self, config: OldTrackerFullJointGraspConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.gripper_loss_weight = float(config.gripper_loss_weight)
        self.real_action_dim = int(config.real_action_dim)
        self.gripper_action_index = int(config.gripper_action_index)
        self.last_episode_frame = int(config.last_episode_frame)
        if not 0 <= self.gripper_action_index < self.real_action_dim <= self.action_dim:
            raise ValueError(
                "Expected 0 <= gripper_action_index < real_action_dim <= action_dim, got "
                f"{self.gripper_action_index}, {self.real_action_dim}, {self.action_dim}"
            )

    def _track_history(self, observation: _model.Observation):
        images = {}
        for name, image in observation.images.items():
            if image.ndim != 5 or image.shape[1] != TOTAL_INPUT_FRAMES:
                raise ValueError(
                    f"Full-action model expects {name} [B,61,H,W,C], got {image.shape}"
                )
            images[name] = image[:, :HISTORY_FRAMES]
        history_observation = observation.replace(images=images)
        return super()._track_history(history_observation)

    def compute_loss_with_memory_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        del train
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=False
        )
        if observation.frame_index is None:
            raise ValueError("Full-action temporal masking requires frame_index")

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        raw_memory, memory_tokens, tracked = self._raw_and_resampled_memory(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_current_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
        suffix_tokens = self.ActionMemoryCrossAttention(suffix_tokens, memory_tokens)
        input_mask = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
        ar_mask = jnp.concatenate((prefix_ar_mask, suffix_ar_mask), axis=0)
        attn_mask = _old._pi_mem.make_attn_mask(input_mask, ar_mask)  # noqa: SLF001
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        squared_error = jnp.square(velocity - u_t)

        if observation.action_loss_mask is not None:
            dim_mask = observation.action_loss_mask[..., None, :]
        else:
            dim_mask = jnp.asarray(self.action_loss_mask)[None, None, :]
        dimension_weights = jnp.ones((self.action_dim,), dtype=jnp.float32)
        dimension_weights = dimension_weights.at[self.gripper_action_index].set(
            self.gripper_loss_weight
        )
        dimension_weights = dimension_weights.at[self.real_action_dim :].set(0.0)
        dim_mask = dim_mask * dimension_weights[None, None, :]
        loss_per_timestep = jnp.sum(squared_error * dim_mask, axis=-1) / jnp.maximum(
            jnp.sum(dim_mask, axis=-1), 1e-8
        )

        frame_index = jnp.asarray(observation.frame_index, dtype=jnp.int32)
        future_offsets = 1 + jnp.arange(self.action_horizon, dtype=jnp.int32)
        temporal_valid = frame_index[..., None] + future_offsets <= self.last_episode_frame
        valid_count = jnp.sum(temporal_valid, axis=-1, keepdims=True)
        # The trainer averages BxH. Scale each sample so it contributes its
        # mean over real future steps, independent of terminal padding.
        temporal_scale = self.action_horizon / jnp.maximum(valid_count, 1)
        loss_per_timestep = (
            loss_per_timestep * temporal_valid.astype(loss_per_timestep.dtype) * temporal_scale
        )
        return loss_per_timestep, {
            "history_mem": raw_memory,
            "encoder_auxes": (),
            "history_class_logits": tracked["joint_logits"],
            "temporal_valid_fraction": jnp.mean(temporal_valid),
        }


def _find_hf_dataset(dataset):
    current = dataset
    while current is not None:
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            return hf_dataset
        current = getattr(current, "_dataset", None)
    raise ValueError("Could not find underlying hf_dataset for phase-balanced sampling")


def _balanced_full_action_indices(dataset, indices: list[int], _classifier_config) -> list[int]:
    """Return a deterministic 20% mixture of selection/approach/descend/grasp/lift."""
    hf_dataset = _find_hf_dataset(dataset)
    if "frame_index" not in getattr(hf_dataset, "column_names", ()):
        raise ValueError("Full-action sampling requires frame_index")
    selected = np.asarray(indices, dtype=np.int64)
    frame = np.asarray(hf_dataset["frame_index"], dtype=np.int64)[selected]
    group_masks = (
        frame == 59,
        (frame >= 60) & (frame <= 88),
        (frame >= 89) & (frame <= 108),
        (frame >= 109) & (frame <= 118),
        (frame >= 119) & (frame <= 153),
    )
    groups = [selected[mask] for mask in group_masks]
    if any(len(group) == 0 for group in groups):
        raise ValueError(f"Empty full-action phase group: {[len(group) for group in groups]}")
    target_size = max(len(group) for group in groups)
    balanced = []
    for group_index, group in enumerate(groups):
        rng = np.random.default_rng(260810 + group_index + len(indices))
        shuffled = rng.permutation(group)
        repeats, remainder = divmod(target_size, len(shuffled))
        expanded = np.concatenate(
            [np.tile(shuffled, repeats), shuffled[:remainder]], axis=0
        )
        balanced.append(expanded)
    # Interleave phases so deterministic validation batches remain balanced.
    return np.stack(balanced, axis=1).reshape(-1).tolist()


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    parent = _old.build_config(args)
    model_fields = {
        field.name: getattr(parent.model, field.name)
        for field in dataclasses.fields(_old.QueryCrossAttnPiJointActionConfig)
    }
    model_fields.update(num_frames=TOTAL_INPUT_FRAMES)
    model = OldTrackerFullJointGraspConfig(
        **model_fields,
        gripper_loss_weight=args.gripper_loss_weight,
        last_episode_frame=LAST_EPISODE_FRAME,
    )
    data = _config.MultiDataConfigFactory(
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
                num_frames=TOTAL_INPUT_FRAMES,
                frame_stride=1,
            )
        ],
    )
    return dataclasses.replace(
        parent,
        name="pi0_shellgame_old_tracker_full_joint_grasp_260810",
        exp_name=args.exp_name,
        model=model,
        data=data,
        freeze_filter=model.get_freeze_filter_full_action(),
        weight_loader=weight_loaders.CheckpointWeightLoader(args.init_checkpoint),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 2),
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        val_ratio=0.1,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        wandb_enabled=False,
        overwrite=args.overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=dataclasses.replace(parent.shellgame_cup_eval, enabled=False),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=OLD_QUERY_ACTION_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--peak-lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--keep-period", type=int, default=1_000)
    parser.add_argument("--gripper-loss-weight", type=float, default=4.0)
    # Parent dynamic-config fields retained for exact architecture reconstruction.
    parser.add_argument("--tracker-checkpoint", default="")
    parser.add_argument("--raw-memory-mode", default="normal")
    parser.add_argument("--restore-adapter", action="store_true")
    parser.add_argument("--restore-memory-interface", action="store_true")
    parser.add_argument("--query-tokens", type=int, default=16)
    parser.add_argument("--query-width", type=int, default=256)
    parser.add_argument("--query-depth", type=int, default=2)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--action-cross-attention-heads", type=int, default=8)
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument("--memory-checkpoint", default="")
    parser.add_argument("--eval-batches-parent", type=int, default=10)
    parser.add_argument("--cup-eval-interval", type=int, default=250)
    parser.add_argument("--cup-eval-episodes", type=int, default=24)
    parser.add_argument("--cup-eval-batch-size", type=int, default=6)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
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
    parser.add_argument("--video-mode", default="normal")
    parser.add_argument("--initial-mode", default="normal")
    parser.add_argument("--relation-mode", default="one_hot")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    # _old.build_config expects this exact field name.
    args.eval_batches = int(args.eval_batches)
    return args


def main() -> None:
    args = parse_args()
    _config_pi0_mem.VideoFrameDataset = FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _balanced_full_action_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
