"""Train the paper-style Pi0Mem video encoder from only the final cup label.

This is a deliberately isolated architecture/capacity diagnostic:

    ShellGame frames 0..59
      -> native ``openpi.models.siglip_mem`` encoder
         (per-frame spatial attention + causal temporal attention every N layers)
      -> current-frame tokens only
      -> a small learned-query three-way readout
      -> final_ball_cup cross entropy

There is no action loss, relation label, stage-slot label, recurrent semantic
updater, fixed-grid compressor, or ``history_mem`` resampler.  The language
model and action expert remain frozen; the native video encoder and the new
readout are trainable.

The config class derives from ``Pi0MemCompressConfig`` only so the existing
classification-only trainer can provide episode-heldout splitting, label
loading, checkpointing, and metrics.  ``create()`` instantiates
``NativePi0MemFinalSlotProbe``, whose visual path directly derives from
``Pi0Mem`` and uses ``siglip_mem.py``.  No compressed-memory model is created.
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

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_mem as _native_mem
from openpi.models import pi0_mem_compress as _trainer_config_base
from openpi.shared import array_typing as at
import openpi.shared.download as download
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
DEFAULT_INIT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base/params"

NUM_FRAMES = 60
NUM_CLASSES = 3
SWAP_START = 20
ABLATION_MODES = (
    "normal",
    "reverse_history",
    "wrong_episode_history",
    "reveal_only",
    "last_frame_only",
)


class NativeCurrentTokenFinalSlotHead(nn.Module):
    """Read a final slot from current-frame tokens only.

    History tokens never reach this module, so any history dependence must be
    produced by the causal temporal attention inside ``siglip_mem``.
    """

    input_width: int
    width: int = 256
    num_heads: int = 8
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, current_tokens, *, train: bool):
        if current_tokens.ndim != 3 or current_tokens.shape[-1] != self.input_width:
            raise ValueError(
                f"Expected [B,N,{self.input_width}] current tokens, got "
                f"{current_tokens.shape}"
            )
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(current_tokens)
        x = nn.Dense(self.width, name="input_projection", dtype=self.dtype_mm)(x)
        query = self.param(
            "readout_query",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.width),
            x.dtype,
        )
        query = jnp.tile(query, (x.shape[0], 1, 1))
        pooled = nn.MultiHeadDotProductAttention(
            name="readout_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=not train,
            dtype=self.dtype_mm,
        )(query, x)[:, 0]
        pooled = nn.LayerNorm(name="readout_ln", dtype=self.dtype_mm)(pooled)
        logits = nn.Dense(
            NUM_CLASSES,
            name="classifier",
            dtype=jnp.float32,
            kernel_init=nn.initializers.xavier_uniform(),
        )(pooled.astype(jnp.float32))
        return logits, pooled


@dataclasses.dataclass(frozen=True)
class NativePi0MemFinalSlotProbeConfig(_trainer_config_base.Pi0MemCompressConfig):
    """Trainer-compatible config that creates the native Pi0Mem model."""

    temporal_every: int = 4
    native_head_width: int = 256
    native_head_heads: int = 8

    def create(self, rng: at.KeyArrayLike) -> "NativePi0MemFinalSlotProbe":
        return NativePi0MemFinalSlotProbe(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_native_video_and_head(self) -> nnx.filterlib.Filter:
        """Freeze everything except native SigLIP-MEM and the final readout."""
        trainable = nnx_utils.PathRegex(
            r".*(PaliGemma/img|NativeFinalSlotHead).*"
        )
        return nnx.Not(trainable)


class NativePi0MemFinalSlotProbe(_native_mem.Pi0Mem):
    """Native Pi0Mem visual encoder with a diagnostic final-slot readout."""

    def __init__(self, config: NativePi0MemFinalSlotProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        token_width = _gemma.get_config(config.paligemma_variant).width
        self.NativeFinalSlotHead = nnx_bridge.ToNNX(
            NativeCurrentTokenFinalSlotHead(
                input_width=token_width,
                width=config.native_head_width,
                num_heads=config.native_head_heads,
                dtype_mm=config.dtype,
            )
        )
        self.NativeFinalSlotHead.lazy_init(
            jnp.zeros((1, 256, token_width), dtype=jnp.bfloat16),
            train=False,
            rngs=rngs,
        )

    @staticmethod
    def _base_video(observation: _model.Observation):
        video = observation.images.get("base_rgb")
        if video is None:
            raise ValueError("Native final-slot probe requires the 'base_rgb' stream")
        if video.ndim != 5 or video.shape[1] != NUM_FRAMES:
            raise ValueError(
                f"Expected base_rgb [B,{NUM_FRAMES},H,W,C], got {video.shape}"
            )
        return video

    @staticmethod
    def _ablate_video(video, mode: str):
        if mode == "normal":
            return video

        # Frame 59 remains the recipient episode's current observation.  Only
        # frames 0..58 are changed, so the controls cannot alter the label by
        # replacing the current scene itself.
        history = video[:, :-1]
        current = video[:, -1:]
        if mode == "reverse_history":
            history = history[:, ::-1]
        elif mode == "wrong_episode_history":
            history = jnp.roll(history, 1, axis=0)
        elif mode == "reveal_only":
            frozen = jnp.repeat(
                history[:, SWAP_START - 1 : SWAP_START],
                history.shape[1] - SWAP_START,
                axis=1,
            )
            history = history.at[:, SWAP_START:].set(frozen)
        elif mode == "last_frame_only":
            history = jnp.repeat(current, history.shape[1], axis=1)
        else:
            raise ValueError(f"Unknown ablation mode: {mode!r}")
        return jnp.concatenate([history, current], axis=1)

    def _classify_video(self, video, *, train: bool):
        current_tokens, _encoder_out = self.PaliGemma.img(video, train=train)
        logits, pooled = self.NativeFinalSlotHead(current_tokens, train=train)
        return logits, pooled

    def compute_history_classification(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ):
        observation = _model.preprocess_observation(rng, observation, train=train)
        logits, pooled = self._classify_video(
            self._base_video(observation), train=train
        )
        return logits, {
            "history_mem": pooled[:, None, :],
            "encoder_auxes": (),
        }

    def compute_history_ablation_logits(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ):
        observation = _model.preprocess_observation(rng, observation, train=False)
        video = self._base_video(observation)
        return {
            mode: self._classify_video(
                self._ablate_video(video, mode), train=False
            )[0]
            for mode in ABLATION_MODES
        }


@dataclasses.dataclass(frozen=True)
class NativeProbeCheckpointLoader:
    """Restore every shape-compatible base weight and initialize the new head."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        resolved = download.maybe_download(self.params_path)
        loaded = _model.restore_params(resolved, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        result = {}
        exact = 0
        image_total = 0
        image_exact = 0
        initialized_head = 0
        for key, reference in target.items():
            is_image = key.startswith("PaliGemma/img/")
            image_total += int(is_image)
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
                image_exact += int(is_image)
            else:
                result[key] = reference
                initialized_head += int(key.startswith("NativeFinalSlotHead/"))
        if image_total == 0 or image_exact != image_total:
            raise ValueError(
                "Native SigLIP initialization is incomplete: "
                f"matched {image_exact}/{image_total} image leaves from "
                f"{self.params_path}"
            )
        print(
            "NativeProbeCheckpointLoader: "
            f"exact={exact}/{len(target)}, image={image_exact}/{image_total}, "
            f"initialized_head={initialized_head}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def native_ablation_eval_step(
    config,
    rng,
    state: training_utils.TrainState,
    batch,
    *,
    class_labels_by_episode=None,
):
    """Evaluate history interventions while keeping the current frame fixed."""
    del config
    if class_labels_by_episode is None:
        raise ValueError("Native ablation evaluation requires episode labels")
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, _actions = batch
    logits_by_mode = model.compute_history_ablation_logits(rng, observation)

    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    frame_index = jnp.asarray(observation.frame_index, dtype=jnp.int32)
    episode_valid = (episode_index >= 0) & (
        episode_index < class_labels_by_episode.shape[0]
    )
    safe_episode = jnp.clip(episode_index, 0, class_labels_by_episode.shape[0] - 1)
    labels = class_labels_by_episode[safe_episode]
    valid = episode_valid & (labels >= 0) & (frame_index == NUM_FRAMES - 1)
    valid_f = valid.astype(jnp.float32)
    valid_count = jnp.sum(valid_f)
    safe_labels = jnp.maximum(labels, 0)

    result = {}
    for mode, logits in logits_by_mode.items():
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, safe_labels)
        loss_sum = jnp.sum(losses * valid_f)
        correct = jnp.sum((jnp.argmax(logits, axis=-1) == safe_labels) * valid_f)
        result[f"val/{mode}_loss"] = loss_sum / jnp.maximum(valid_count, 1.0)
        result[f"val/{mode}_accuracy"] = correct / jnp.maximum(valid_count, 1.0)

    normal_logits = logits_by_mode["normal"]
    normal_losses = optax.softmax_cross_entropy_with_integer_labels(
        normal_logits, safe_labels
    )
    normal_loss_sum = jnp.sum(normal_losses * valid_f)
    normal_correct = jnp.sum(
        (jnp.argmax(normal_logits, axis=-1) == safe_labels) * valid_f
    )
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    result.update(
        {
            "val/loss": normal_loss_sum / jnp.maximum(valid_count, 1.0),
            "val/action_loss": zero,
            "val/diversity_loss": zero,
            "val/history_classifier_loss": normal_loss_sum
            / jnp.maximum(valid_count, 1.0),
            "val/history_classifier_accuracy": normal_correct
            / jnp.maximum(valid_count, 1.0),
            "val/history_classifier_valid_fraction": jnp.mean(valid_f),
            "_val/history_classifier_loss_sum": normal_loss_sum,
            "_val/history_classifier_correct_count": normal_correct,
            "_val/history_classifier_valid_count": valid_count,
        }
    )
    return result


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    model = NativePi0MemFinalSlotProbeConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        max_token_len=256,
        num_frames=NUM_FRAMES,
        current_frame_index=-1,
        temporal_every=args.temporal_every,
        history_classifier_num_classes=NUM_CLASSES,
        diversity_weight=0.0,
        native_head_width=args.head_width,
        native_head_heads=args.head_heads,
        siglip_remat_policy=args.remat_policy,
    )
    data_config_cls = _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_AbsoluteEEF7
    return _config.TrainConfig(
        name="pi0_shellgame_native_mem_final_slot_probe_260824",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_native_video_and_head(),
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
                    num_frames=NUM_FRAMES,
                    frame_stride=1,
                    video_layout="sliding",
                    fixed_prefix_frames=0,
                    tokenize_prompt=False,
                    min_frame_index=NUM_FRAMES - 1,
                    max_frame_index=NUM_FRAMES - 1,
                )
            ],
        ),
        weight_loader=NativeProbeCheckpointLoader(args.init_checkpoint),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 1),
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=1 if args.overfit_samples_per_class else 10,
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
            min_frame_index=NUM_FRAMES - 1,
            max_frame_index=NUM_FRAMES - 1,
            loss_weight=1.0,
            action_loss_weight=0.0,
            overfit_samples_per_class=args.overfit_samples_per_class,
            overfit_same_samples_for_validation=(
                args.overfit_samples_per_class > 0
            ),
            disable_train_augmentation=True,
        ),
    )


def run_self_test() -> None:
    head = NativeCurrentTokenFinalSlotHead(
        input_width=32,
        width=16,
        num_heads=4,
        dtype_mm="float32",
    )
    tokens = jax.random.normal(jax.random.key(1), (2, 256, 32))
    variables = head.init(jax.random.key(0), tokens, train=False)
    logits, pooled = head.apply(variables, tokens, train=False)
    if logits.shape != (2, NUM_CLASSES) or pooled.shape != (2, 16):
        raise AssertionError(f"Unexpected shapes: {logits.shape}, {pooled.shape}")
    print(f"Native final-slot head self-test passed: {logits.shape}, {pooled.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--peak-lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=300)
    parser.add_argument("--temporal-every", type=int, default=4)
    parser.add_argument("--head-width", type=int, default=256)
    parser.add_argument("--head-heads", type=int, default=8)
    parser.add_argument(
        "--remat-policy", default="nothing_saveable"
    )
    parser.add_argument("--overfit-samples-per-class", type=int, default=10)
    parser.add_argument("--ablation-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.ablation_eval:
        _trainer.eval_step = native_ablation_eval_step
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
