"""Directly classify the final ShellGame slot from the full visual history.

This is the minimal controlled baseline for deciding whether the explicit
initial-cup / swap-relation / recurrent-state decomposition is necessary.  It
uses no action loss, no relation labels, no teacher forcing, and no compact
M128 bottleneck:

    frames 0..59
      -> frozen SigLIP patch embeddings
      -> fixed 2x2 pooling (16x16 -> 8x8, K=64)
      -> depth-2 factorized temporal/spatial Transformer
      -> learned-query readout
      -> final left/middle/right logits

The validation step also measures reverse-time, wrong-episode-history, and
reveal-only controls while reusing the same frozen SigLIP features.
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
import numpy as np
import optax

from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.models import siglip_mem_semantic as _semantic_core
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import utils as training_utils
from scripts.mem import train_pi0_mem_compress as _trainer

DATASET_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_absolute_eef_raw7"
)
LABELS_PATH = DATASET_ROOT / "meta" / "episodes.jsonl"
SOURCE_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/"
    "absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/"
    "5999/params"
)
NUM_HISTORY_FRAMES = 60
NUM_PATCHES = 256
NUM_CLASSES = 3
SWAP_START = 20
SWAP_END = 50
ABLATION_MODES = ("normal", "reverse_time", "wrong_episode", "reveal_only")


class DirectFinalSlotClassifier(nn.Module):
    """Encode all 60 frames and read one final three-way slot prediction."""

    num_frames: int = NUM_HISTORY_FRAMES
    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    spatial_pool_factor: int = 2
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens, *, train: bool):
        expected = (self.num_frames, NUM_PATCHES, self.input_width)
        if patch_tokens.ndim != 4 or patch_tokens.shape[1:] != expected:
            raise ValueError(f"Expected [B,{expected}], got {patch_tokens.shape}")

        x = _semantic_core.pool_fixed_grid(
            patch_tokens, pool_factor=self.spatial_pool_factor
        )
        spatial_tokens = x.shape[2]
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(x)
        x = nn.Dense(self.width, name="input_projection", dtype=self.dtype_mm)(x)
        temporal_pos = self.param(
            "temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_frames, 1, self.width),
            x.dtype,
        )
        x = x + temporal_pos
        for block_index in range(self.depth):
            x = _semantic_core.FactorizedSpaceTimeBlock(
                name=f"block_{block_index}",
                width=self.width,
                num_heads=self.num_heads,
                dropout=0.0,
                dtype_mm=self.dtype_mm,
            )(x, train=train)

        flat = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(
            x.reshape(x.shape[0], self.num_frames * spatial_tokens, self.width)
        )
        readout_query = self.param(
            "readout_query",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.width),
            flat.dtype,
        )
        query = jnp.tile(readout_query, (flat.shape[0], 1, 1))
        pooled = nn.MultiHeadDotProductAttention(
            name="readout_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=not train,
            dtype=self.dtype_mm,
        )(query, flat)
        pooled = nn.LayerNorm(name="readout_ln", dtype=self.dtype_mm)(pooled[:, 0])
        logits = nn.Dense(NUM_CLASSES, name="classifier", dtype=jnp.float32)(
            pooled.astype(jnp.float32)
        )
        return logits, pooled


@dataclasses.dataclass(frozen=True)
class DirectFinalSlotProbeConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2

    def create(self, rng: at.KeyArrayLike) -> DirectFinalSlotProbeModel:
        return DirectFinalSlotProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_probe_only(self) -> nnx.filterlib.Filter:
        probe = nnx_utils.PathRegex(r".*HistoryDirectFinalSlotClassifier.*")
        return nnx.Not(probe)


class DirectFinalSlotProbeModel(_base_model.Pi0MemCompress):
    """Frozen Pi0.5 vision backbone plus the direct-history classifier."""

    def __init__(self, config: DirectFinalSlotProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.HistoryDirectFinalSlotClassifier = nnx_bridge.ToNNX(
            DirectFinalSlotClassifier(
                num_frames=NUM_HISTORY_FRAMES,
                input_width=1152,
                width=config.temporal_width,
                depth=config.temporal_depth,
                num_heads=config.temporal_heads,
                spatial_pool_factor=config.spatial_pool_factor,
                dtype_mm=config.dtype,
            )
        )
        self.HistoryDirectFinalSlotClassifier.lazy_init(
            jnp.zeros(
                (1, NUM_HISTORY_FRAMES, NUM_PATCHES, 1152), dtype=jnp.bfloat16
            ),
            train=False,
            rngs=rngs,
        )

    @staticmethod
    def _validate_video(observation: _model.Observation):
        image = observation.images.get("base_rgb")
        if image is None:
            raise ValueError("Direct final-slot probe requires the 'base_rgb' stream")
        if image.ndim != 5 or image.shape[1] != NUM_HISTORY_FRAMES + 1:
            raise ValueError(
                "Direct final-slot probe expects "
                f"[B,{NUM_HISTORY_FRAMES + 1},H,W,C], got {image.shape}"
            )
        return image

    @staticmethod
    def _ablate_history(history_patches, mode: str):
        if mode == "normal":
            return history_patches
        if mode == "reverse_time":
            return history_patches[:, ::-1]
        if mode == "wrong_episode":
            return jnp.roll(history_patches, 1, axis=0)
        if mode == "reveal_only":
            frozen = jnp.repeat(
                history_patches[:, SWAP_START - 1 : SWAP_START],
                NUM_HISTORY_FRAMES - SWAP_START,
                axis=1,
            )
            return history_patches.at[:, SWAP_START:].set(frozen)
        raise ValueError(f"Unknown history ablation mode: {mode!r}")

    def _encode_history_patches(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool,
    ):
        observation = _model.preprocess_observation(rng, observation, train=train)
        image = self._validate_video(observation)
        _, encoder_out = self.PaliGemma.img(image, train=False)
        return encoder_out["with_posemb"][:, :NUM_HISTORY_FRAMES]

    def compute_history_classification(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ):
        history_patches = self._encode_history_patches(
            rng, observation, train=train
        )
        logits, pooled = self.HistoryDirectFinalSlotClassifier(
            history_patches, train=train
        )
        return logits, {"history_mem": pooled[:, None], "encoder_auxes": ()}

    def compute_history_ablation_logits(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ):
        """Evaluate all controls after one shared frozen-SigLIP forward pass."""
        history_patches = self._encode_history_patches(
            rng, observation, train=False
        )
        return {
            mode: self.HistoryDirectFinalSlotClassifier(
                self._ablate_history(history_patches, mode), train=False
            )[0]
            for mode in ABLATION_MODES
        }


@dataclasses.dataclass(frozen=True)
class DirectProbeCheckpointLoader:
    """Restore shared policy weights and initialize only the new probe."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        probe_prefix = "HistoryDirectFinalSlotClassifier/"
        result = {}
        exact = 0
        initialized_probe = []
        initialized_other = []
        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
            else:
                result[key] = reference
                if key.startswith(probe_prefix):
                    initialized_probe.append(key)
                else:
                    initialized_other.append(key)
        print(
            "DirectProbeCheckpointLoader: "
            f"exact={exact}, initialized_probe={len(initialized_probe)}, "
            f"initialized_unused_base={len(initialized_other)}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def direct_ablation_eval_step(
    config,
    rng,
    state: training_utils.TrainState,
    batch,
    *,
    class_labels_by_episode=None,
):
    """Evaluate normal and controlled histories with exact sample counts."""
    del config
    if class_labels_by_episode is None:
        raise ValueError("Direct final-slot evaluation requires episode labels")
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, _actions = batch
    if observation.episode_index is None or observation.frame_index is None:
        raise ValueError("Direct final-slot evaluation requires episode/frame indices")

    logits_by_mode = model.compute_history_ablation_logits(rng, observation)
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    frame_index = jnp.asarray(observation.frame_index, dtype=jnp.int32)
    episode_valid = (episode_index >= 0) & (
        episode_index < class_labels_by_episode.shape[0]
    )
    safe_episode = jnp.clip(episode_index, 0, class_labels_by_episode.shape[0] - 1)
    labels = class_labels_by_episode[safe_episode]
    valid = episode_valid & (labels >= 0) & (frame_index == 59)
    valid_f = valid.astype(jnp.float32)
    valid_count = jnp.sum(valid_f)
    safe_labels = jnp.maximum(labels, 0)

    result = {}
    for mode, logits in logits_by_mode.items():
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, safe_labels)
        loss_sum = jnp.sum(losses * valid_f)
        correct_count = jnp.sum(
            (jnp.argmax(logits, axis=-1) == safe_labels) * valid_f
        )
        result[f"val/{mode}_loss"] = loss_sum / jnp.maximum(valid_count, 1.0)
        result[f"val/{mode}_accuracy"] = correct_count / jnp.maximum(
            valid_count, 1.0
        )

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    normal_logits = logits_by_mode["normal"]
    normal_losses = optax.softmax_cross_entropy_with_integer_labels(
        normal_logits, safe_labels
    )
    normal_loss_sum = jnp.sum(normal_losses * valid_f)
    normal_correct_count = jnp.sum(
        (jnp.argmax(normal_logits, axis=-1) == safe_labels) * valid_f
    )
    result.update(
        {
            "val/loss": normal_loss_sum / jnp.maximum(valid_count, 1.0),
            "val/action_loss": zero,
            "val/diversity_loss": zero,
            "val/history_classifier_loss": normal_loss_sum
            / jnp.maximum(valid_count, 1.0),
            "val/history_classifier_accuracy": normal_correct_count
            / jnp.maximum(valid_count, 1.0),
            "val/history_classifier_valid_fraction": jnp.mean(valid_f),
            "_val/history_classifier_loss_sum": normal_loss_sum,
            "_val/history_classifier_correct_count": normal_correct_count,
            "_val/history_classifier_valid_count": valid_count,
        }
    )
    return result


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    model = DirectFinalSlotProbeConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        max_token_len=256,
        num_frames=61,
        memory_every=0,
        current_frame_index=-1,
        history_memory_tokens=1,
        history_resampler_depth=1,
        history_use_current_condition=False,
        history_gate_fixed=0.0,
        diversity_weight=0.0,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=NUM_CLASSES,
        temporal_width=args.temporal_width,
        temporal_depth=args.temporal_depth,
        temporal_heads=args.temporal_heads,
        spatial_pool_factor=2,
    )
    data_config_cls = _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_AbsoluteEEF7
    return _config.TrainConfig(
        name="pi0_shellgame_three_swap_direct_final_slot_probe_260821",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_probe_only(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                data_config_cls(
                    repo_id=str(DATASET_ROOT),
                    assets=_config.AssetsConfig(
                        asset_id=".", assets_dir=str(DATASET_ROOT)
                    ),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=61,
                    frame_stride=1,
                    video_layout="fixed_prefix_current",
                    fixed_prefix_frames=60,
                    tokenize_prompt=False,
                    min_frame_index=59,
                    max_frame_index=59,
                )
            ],
        ),
        weight_loader=DirectProbeCheckpointLoader(args.init_checkpoint),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 1),
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=10.0),
        ema_decay=None,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=args.save_interval,
        keep_period=args.save_interval,
        val_ratio=0.1,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        wandb_enabled=False,
        overwrite=args.overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(
            enabled=True,
            episodes_metadata_path=str(LABELS_PATH),
            label_key="final_ball_cup",
            classes=("left", "middle", "right"),
            min_frame_index=59,
            max_frame_index=59,
            loss_weight=1.0,
            action_loss_weight=0.0,
            disable_train_augmentation=True,
        ),
    )


def run_self_test() -> None:
    classifier = DirectFinalSlotClassifier(
        num_frames=NUM_HISTORY_FRAMES,
        input_width=16,
        width=16,
        depth=1,
        num_heads=4,
        spatial_pool_factor=2,
        dtype_mm="float32",
    )
    inputs = jax.random.normal(
        jax.random.key(1), (2, NUM_HISTORY_FRAMES, NUM_PATCHES, 16)
    )
    variables = classifier.init(jax.random.key(0), inputs, train=False)
    logits, pooled = classifier.apply(variables, inputs, train=False)
    if logits.shape != (2, NUM_CLASSES) or pooled.shape != (2, 16):
        raise AssertionError(f"Unexpected self-test shapes: {logits.shape}, {pooled.shape}")
    print(f"Direct final-slot self-test passed: logits={logits.shape}, pooled={pooled.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=SOURCE_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--temporal-width", type=int, default=256)
    parser.add_argument("--temporal-depth", type=int, default=2)
    parser.add_argument("--temporal-heads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    _trainer.eval_step = direct_ablation_eval_step
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
