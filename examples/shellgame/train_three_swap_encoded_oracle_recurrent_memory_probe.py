"""Test whether explicit per-swap motion encoding fixes recurrent tracking.

This is a strict bridge between two preceding diagnostics:

* the frozen segment encoder is restored from the swap-pair probe that reached
  100% held-out accuracy;
* the persistent M=128 updater/readout is restored from the Oracle-initialized
  recurrent probe that stayed at chance.

Only the recurrent updater/readout is optimized.  The true initial cup is
still injected parameter-free and action loss remains disabled, so the only
changed variable is whether a swap is represented by factorized temporal /
spatial attention before it reaches persistent memory.
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

from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import IntegratedCurrentReadout
from examples.shellgame.train_one_swap_temporal_transformer_probe import FactorizedSpaceTimeBlock
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import DATASET_ROOT
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import JOINT_CLASSES
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import multistage_eval_step
from examples.shellgame.train_three_swap_oracle_initial_recurrent_memory_probe import build_initial_slot_lookup
from examples.shellgame.train_three_swap_recurrent_memory_fixed_grid_probe import SharedSegmentMemoryUpdater
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer

DEFAULT_SEGMENT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_pair_fixed_grid_probe_260809/"
    "swap_pair_full600_b18_260809/599/params"
)
DEFAULT_RECURRENT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_oracle_initial_recurrent_memory_260809/"
    "oracle_initial_full600_260809/599/params"
)
NUM_HISTORY_FRAMES = 60
SWAP_START_FRAME = 20
SWAP_END_FRAME = 50
SEGMENT_SIZE = 10
NUM_SWAP_SEGMENTS = 3


class PretrainedSwapSegmentEncoder(nn.Module):
    """Encode one swap clip while retaining its 10 x 64 token grid."""

    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    segment_size: int = SEGMENT_SIZE
    dropout: float = 0.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, segment_tokens):
        if segment_tokens.ndim != 4 or segment_tokens.shape[1:] != (
            self.segment_size,
            64,
            self.input_width,
        ):
            raise ValueError(f"Expected [B,{self.segment_size},64,{self.input_width}], got {segment_tokens.shape}")
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(segment_tokens)
        x = nn.Dense(self.width, name="input_projection", dtype=self.dtype_mm)(x)
        temporal_pos = self.param(
            "relative_temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.segment_size, 1, self.width),
            x.dtype,
        )
        x = x + temporal_pos
        for block_index in range(self.depth):
            x = FactorizedSpaceTimeBlock(
                name=f"block_{block_index}",
                width=self.width,
                num_heads=self.num_heads,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
            )(x, train=False)
        return nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(x)


class ThreeSwapEncodedOracleRecurrentMemoryTracker(nn.Module):
    """Apply a frozen motion encoder before each recurrent memory update."""

    num_frames: int = NUM_HISTORY_FRAMES
    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    spatial_pool_factor: int = 2
    num_memory_tokens: int = 128
    segment_size: int = SEGMENT_SIZE
    oracle_code_scale: float = 1.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens, initial_slots):
        b, t, n, d = patch_tokens.shape
        if (t, n, d) != (self.num_frames, 256, self.input_width):
            raise ValueError(f"Expected [B,{self.num_frames},256,{self.input_width}], got {patch_tokens.shape}")
        if initial_slots.shape != (b,):
            raise ValueError(f"Expected initial slots [B], got {initial_slots.shape}")

        input_grid = int(np.sqrt(n))
        output_grid = input_grid // self.spatial_pool_factor
        clips = patch_tokens[:, SWAP_START_FRAME:SWAP_END_FRAME].reshape(
            b,
            NUM_SWAP_SEGMENTS,
            self.segment_size,
            output_grid,
            self.spatial_pool_factor,
            output_grid,
            self.spatial_pool_factor,
            d,
        )
        clips = jnp.mean(clips, axis=(4, 6)).reshape(b * NUM_SWAP_SEGMENTS, self.segment_size, output_grid**2, d)
        encoded = PretrainedSwapSegmentEncoder(
            name="segment_encoder",
            input_width=self.input_width,
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=self.segment_size,
            dtype_mm=self.dtype_mm,
        )(clips)
        encoded = encoded.reshape(b, NUM_SWAP_SEGMENTS, self.segment_size, output_grid**2, self.width)

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.width),
            encoded.dtype,
        )
        memory = jnp.tile(base_memory, (b, 1, 1))
        oracle_code = jax.nn.one_hot(initial_slots, 3, dtype=memory.dtype)
        memory = memory.at[:, 0, :3].add(self.oracle_code_scale * oracle_code)

        updater = SharedSegmentMemoryUpdater(
            name="shared_segment_memory_updater",
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=self.segment_size,
            dtype_mm=self.dtype_mm,
        )
        endpoint_memories = []
        for segment_index in range(NUM_SWAP_SEGMENTS):
            memory = updater(memory, encoded[:, segment_index])
            endpoint_memories.append(memory)

        memory_batch = jnp.stack(endpoint_memories, axis=1).reshape(
            b * NUM_SWAP_SEGMENTS, self.num_memory_tokens, self.width
        )
        memory_batch = nn.LayerNorm(name="memory_output_ln", dtype=self.dtype_mm)(memory_batch)
        memory_batch = nn.Dense(self.input_width, name="memory_output_projection", dtype=self.dtype_mm)(memory_batch)
        memory_batch = memory_batch - jnp.mean(memory_batch, axis=1, keepdims=True)
        memory_batch = nn.LayerNorm(name="pi0_output_ln", dtype=self.dtype_mm)(memory_batch)
        memory_batch = memory_batch - jnp.mean(memory_batch, axis=1, keepdims=True)

        logits = IntegratedCurrentReadout(
            name="shared_readout",
            input_width=self.input_width,
            width=self.width,
            num_classes=3,
            dtype_mm=self.dtype_mm,
        )(memory_batch)
        stage_logits = logits.reshape(b, NUM_SWAP_SEGMENTS, 3)
        stage_memories = memory_batch.reshape(b, NUM_SWAP_SEGMENTS, self.num_memory_tokens, self.input_width)
        logits_0, logits_1, logits_2 = (
            stage_logits[:, 0],
            stage_logits[:, 1],
            stage_logits[:, 2],
        )
        joint_logits = (logits_0[:, :, None, None] + logits_1[:, None, :, None] + logits_2[:, None, None, :]).reshape(
            b, 27
        )
        return joint_logits, stage_logits, stage_memories


@dataclasses.dataclass(frozen=True)
class EncodedOracleProbeConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2
    endpoint_memory_tokens: int = 128
    oracle_initial_slots: tuple[int, ...] = ()
    oracle_mode: str = "correct"
    video_mode: str = "normal"

    def create(self, rng: at.KeyArrayLike) -> EncodedOracleProbeModel:
        return EncodedOracleProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_recurrent_only(self) -> nnx.filterlib.Filter:
        tracker = nnx_utils.PathRegex(r".*HistoryThreeSwapEncodedOracleRecurrentMemoryTracker.*")
        encoder = nnx_utils.PathRegex(r".*HistoryThreeSwapEncodedOracleRecurrentMemoryTracker/segment_encoder.*")
        trainable = nnx.All(tracker, nnx.Not(encoder))
        return nnx.Not(trainable)


class EncodedOracleProbeModel(_base_model.Pi0MemCompress):
    def __init__(self, config: EncodedOracleProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.oracle_initial_slots = config.oracle_initial_slots
        self.oracle_mode = config.oracle_mode
        self.video_mode = config.video_mode
        self.HistoryThreeSwapEncodedOracleRecurrentMemoryTracker = nnx_bridge.ToNNX(
            ThreeSwapEncodedOracleRecurrentMemoryTracker(
                num_frames=NUM_HISTORY_FRAMES,
                input_width=1152,
                width=config.temporal_width,
                depth=config.temporal_depth,
                num_heads=config.temporal_heads,
                spatial_pool_factor=config.spatial_pool_factor,
                num_memory_tokens=config.endpoint_memory_tokens,
                segment_size=SEGMENT_SIZE,
                dtype_mm=config.dtype,
            )
        )
        fake_tokens = jnp.zeros((1, NUM_HISTORY_FRAMES, 256, 1152), dtype=jnp.bfloat16)
        fake_slots = jnp.zeros((1,), dtype=jnp.int32)
        self.HistoryThreeSwapEncodedOracleRecurrentMemoryTracker.lazy_init(fake_tokens, fake_slots, rngs=rngs)

    def compute_history_classification(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ):
        observation = _model.preprocess_observation(rng, observation, train=train)
        image = observation.images["base_rgb"]
        if image.ndim == 4:
            image = image[:, None]
        if image.ndim != 5 or image.shape[1] != 61:
            raise ValueError(f"Encoded Oracle probe expects [B,61,H,W,C], got {image.shape}")
        if observation.episode_index is None:
            raise ValueError("Encoded Oracle probe requires observation.episode_index")

        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, :NUM_HISTORY_FRAMES]
        if self.video_mode == "shuffle_batch":
            donor = jnp.roll(history_patches[:, SWAP_START_FRAME:SWAP_END_FRAME], 1, axis=0)
            history_patches = history_patches.at[:, SWAP_START_FRAME:SWAP_END_FRAME].set(donor)
        elif self.video_mode == "zero_swaps":
            history_patches = history_patches.at[:, SWAP_START_FRAME:SWAP_END_FRAME].set(0)
        elif self.video_mode != "normal":
            raise ValueError(f"Unknown video_mode={self.video_mode!r}")

        episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
        lookup = jnp.asarray(self.oracle_initial_slots, dtype=jnp.int32)
        safe_episode = jnp.clip(episode_index, 0, lookup.shape[0] - 1)
        initial_slots = lookup[safe_episode]
        if self.oracle_mode == "roll":
            initial_slots = (initial_slots + 1) % 3
        elif self.oracle_mode != "correct":
            raise ValueError(f"Unknown oracle_mode={self.oracle_mode!r}")

        joint_logits, stage_logits, stage_memories = self.HistoryThreeSwapEncodedOracleRecurrentMemoryTracker(
            history_patches, initial_slots
        )
        return joint_logits, {
            "history_mem": stage_memories.reshape(-1, stage_memories.shape[-2], stage_memories.shape[-1]),
            "stage_logits": stage_logits,
            "encoder_auxes": (),
        }


@dataclasses.dataclass(frozen=True)
class EncodedOracleCheckpointLoader:
    """Combine the successful segment encoder and failed recurrent state."""

    segment_params_path: str
    recurrent_params_path: str

    def load(self, params: at.Params) -> at.Params:
        segment_loaded = _model.restore_params(self.segment_params_path, restore_type=np.ndarray)
        recurrent_loaded = _model.restore_params(self.recurrent_params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        segment_source = flax.traverse_util.flatten_dict(segment_loaded, sep="/")
        recurrent_source = flax.traverse_util.flatten_dict(recurrent_loaded, sep="/")

        target_root = "HistoryThreeSwapEncodedOracleRecurrentMemoryTracker/"
        target_encoder_root = target_root + "segment_encoder/"
        source_encoder_root = "HistoryThreeSwapPairVisualClassifier/"
        source_recurrent_root = "HistoryThreeSwapOracleRecurrentMemoryTracker/"
        result = {}
        exact = segment_mapped = recurrent_mapped = 0
        initialized = []
        for key, reference in target.items():
            candidate = segment_source.get(key)
            source_kind = "exact"
            if key.startswith(target_encoder_root):
                relative = key.removeprefix(target_encoder_root)
                candidate = segment_source.get(source_encoder_root + relative)
                source_kind = "segment"
            elif key.startswith(target_root):
                relative = key.removeprefix(target_root)
                candidate = recurrent_source.get(source_recurrent_root + relative)
                source_kind = "recurrent"

            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                if source_kind == "exact":
                    exact += 1
                elif source_kind == "segment":
                    segment_mapped += 1
                else:
                    recurrent_mapped += 1
            else:
                result[key] = reference
                initialized.append(key)

        unexpected = [key for key in initialized if key.startswith(target_root)]
        if unexpected:
            raise ValueError(f"Encoded Oracle restore incomplete: {unexpected[:8]}")
        print(
            "EncodedOracleCheckpointLoader: "
            f"exact={exact}, segment_mapped={segment_mapped}, recurrent_mapped={recurrent_mapped}, "
            f"initialized={len(initialized)}, examples={initialized[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def run_causality_self_test() -> None:
    tracker = ThreeSwapEncodedOracleRecurrentMemoryTracker(
        num_frames=NUM_HISTORY_FRAMES,
        input_width=16,
        width=16,
        depth=2,
        num_heads=4,
        spatial_pool_factor=2,
        num_memory_tokens=8,
        segment_size=SEGMENT_SIZE,
        dtype_mm="float32",
    )
    base = jax.random.normal(jax.random.key(1), (2, NUM_HISTORY_FRAMES, 256, 16))
    slots = jnp.asarray((0, 1), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), base, slots)
    _, reference, _ = tracker.apply(variables, base, slots)
    checks = []
    for start_frame, protected_stages in ((30, 1), (40, 2), (50, 3)):
        changed = base.at[:, start_frame:].set(
            jax.random.normal(jax.random.key(start_frame), base[:, start_frame:].shape)
        )
        _, candidate, _ = tracker.apply(variables, changed, slots)
        checks.append(
            bool(
                np.allclose(
                    np.asarray(reference[:, :protected_stages]),
                    np.asarray(candidate[:, :protected_stages]),
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
    outside = base.at[:, :SWAP_START_FRAME].set(99.0)
    outside = outside.at[:, SWAP_END_FRAME:].set(-99.0)
    _, outside_logits, _ = tracker.apply(variables, outside, slots)
    outside_ignored = np.allclose(np.asarray(reference), np.asarray(outside_logits), rtol=0.0, atol=0.0)
    _, different_oracle, _ = tracker.apply(variables, base, (slots + 1) % 3)
    oracle_effect = not np.allclose(np.asarray(reference), np.asarray(different_oracle))
    if not all(checks) or not outside_ignored or not oracle_effect:
        raise AssertionError(
            f"Encoded Oracle self-test failed: causality={checks}, "
            f"outside={outside_ignored}, oracle_effect={oracle_effect}"
        )
    print(
        f"Encoded Oracle self-test passed: causality={checks}, outside={outside_ignored}, oracle_effect={oracle_effect}"
    )


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    initial_slots = build_initial_slot_lookup()
    model = EncodedOracleProbeConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
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
        history_classifier_num_classes=27,
        temporal_width=256,
        temporal_depth=2,
        temporal_heads=8,
        spatial_pool_factor=2,
        endpoint_memory_tokens=128,
        oracle_initial_slots=initial_slots,
        oracle_mode=args.oracle_mode,
        video_mode=args.video_mode,
    )
    return _config.TrainConfig(
        name="pi0_shellgame_three_swap_encoded_oracle_recurrent_memory_260809",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_recurrent_only(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=str(DATASET_ROOT),
                    assets=_config.AssetsConfig(asset_id=".", assets_dir=str(DATASET_ROOT)),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=61,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=EncodedOracleCheckpointLoader(args.segment_checkpoint, args.recurrent_checkpoint),
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
        save_interval=max(args.steps, 1),
        keep_period=max(args.steps, 1),
        val_ratio=0.1,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        wandb_enabled=False,
        overwrite=args.overwrite,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(
            enabled=True,
            episodes_metadata_path=str(labels_path),
            label_key="swap_track_code",
            classes=JOINT_CLASSES,
            min_frame_index=60,
            max_frame_index=60,
            loss_weight=1.0,
            action_loss_weight=0.0,
            overfit_samples_per_class=args.overfit_samples_per_class,
            overfit_same_samples_for_validation=args.overfit_samples_per_class > 0,
            disable_train_augmentation=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--segment-checkpoint", default=DEFAULT_SEGMENT_CHECKPOINT)
    parser.add_argument("--recurrent-checkpoint", default=DEFAULT_RECURRENT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--overfit-samples-per-class", type=int, default=0)
    parser.add_argument("--oracle-mode", choices=("correct", "roll"), default="correct")
    parser.add_argument("--video-mode", choices=("normal", "shuffle_batch", "zero_swaps"), default="normal")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_causality_self_test()
        return
    _trainer.eval_step = multistage_eval_step
    _trainer.main(build_config(args, build_three_swap_labels()))


if __name__ == "__main__":
    main()
