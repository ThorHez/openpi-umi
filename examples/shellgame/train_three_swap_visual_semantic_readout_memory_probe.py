"""Bridge a proven visual swap readout into compact recurrent memory.

The true initial ball slot is injected from episode metadata. Each contiguous
ten-frame swap clip is processed by the frozen encoder *and attention readout*
from the held-out-100% swap-pair classifier. The classifier linear layer is
discarded: its continuous 256-dimensional pre-classifier embedding is
projected to width 64 and broadcast with the exact 10x64 geometry used by the
successful symbolic-pair compact-memory experiment.

Action loss is disabled. This isolates the missing interface between a proven
visual semantic readout and the proven compact-memory/single-read endpoint.
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

from examples.shellgame import train_three_swap_oracle_memory_token_sweep_probe as _sweep
from examples.shellgame import train_three_swap_oracle_single_history_read_adapter_probe as _single_read
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

DEFAULT_PAIR_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_pair_fixed_grid_probe_260809/"
    "swap_pair_full600_b18_260809/599/params"
)
TOTAL_INPUT_FRAMES = 61
HISTORY_FRAMES = 60
SWAP_SLICES = ((20, 30), (30, 40), (40, 50))
SWAP_SEGMENT_SIZE = 10
SPATIAL_TOKENS = 64


def pool_fixed_grid(patch_tokens: jax.Array) -> jax.Array:
    """Pool a 16x16 patch grid to a topology-preserving 8x8 grid."""
    b, t, n, d = patch_tokens.shape
    input_grid = int(np.sqrt(n))
    pool_factor = 2
    output_grid = input_grid // pool_factor
    if input_grid**2 != n or output_grid**2 != SPATIAL_TOKENS:
        raise ValueError(f"Expected a 16x16 patch grid, got n={n}")
    x = patch_tokens.reshape(b, t, output_grid, pool_factor, output_grid, pool_factor, d)
    return jnp.mean(x, axis=(3, 5)).reshape(b, t, SPATIAL_TOKENS, d)


class PretrainedSwapSemanticEncoder(nn.Module):
    """Return the frozen pair classifier's pre-classifier semantic vector."""

    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    segment_size: int = SWAP_SEGMENT_SIZE
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, segment_tokens):
        expected = (self.segment_size, SPATIAL_TOKENS, self.input_width)
        if segment_tokens.ndim != 4 or segment_tokens.shape[1:] != expected:
            raise ValueError(f"Expected [B,{expected}], got {segment_tokens.shape}")
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
                dropout=0.0,
                dtype_mm=self.dtype_mm,
            )(x, train=False)

        flat = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(x.reshape(x.shape[0], -1, self.width))
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
            deterministic=True,
            dtype=self.dtype_mm,
        )(query, flat)
        return nn.LayerNorm(name="readout_ln", dtype=self.dtype_mm)(pooled[:, 0])


class ThreeSwapVisualSemanticMemoryTracker(nn.Module):
    """Write frozen visual semantic embeddings into compact recurrent memory."""

    num_frames: int = HISTORY_FRAMES
    input_width: int = 1152
    encoder_width: int = 256
    encoder_depth: int = 2
    encoder_heads: int = 8
    memory_width: int = 64
    memory_depth: int = 2
    memory_heads: int = 4
    adapter_heads: int = 4
    num_memory_tokens: int = 128
    num_current_tokens: int = 256
    current_width: int = 1152
    residual_scale: float = 1.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens, initial_slots):
        b, t, n, d = patch_tokens.shape
        if (t, n, d) != (self.num_frames, 256, self.input_width):
            raise ValueError(f"Expected [B,{self.num_frames},256,{self.input_width}], got {patch_tokens.shape}")
        if initial_slots.shape != (b,):
            raise ValueError(f"Expected initial slots [B], got {initial_slots.shape}")

        pooled = pool_fixed_grid(patch_tokens)
        clips = jnp.stack([pooled[:, start:end] for start, end in SWAP_SLICES], axis=1).reshape(
            b * len(SWAP_SLICES),
            SWAP_SEGMENT_SIZE,
            SPATIAL_TOKENS,
            self.input_width,
        )
        semantic = PretrainedSwapSemanticEncoder(
            name="swap_semantic_encoder",
            input_width=self.input_width,
            width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            segment_size=SWAP_SEGMENT_SIZE,
            dtype_mm=self.dtype_mm,
        )(clips)
        semantic = nn.Dense(
            self.memory_width,
            name="semantic_to_memory_projection",
            dtype=jnp.float32,
        )(semantic.astype(jnp.float32))
        semantic = semantic.reshape(b, len(SWAP_SLICES), self.memory_width)
        segment_tokens = jnp.broadcast_to(
            semantic[:, :, None, None, :],
            (
                b,
                len(SWAP_SLICES),
                SWAP_SEGMENT_SIZE,
                SPATIAL_TOKENS,
                self.memory_width,
            ),
        )

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.memory_width),
            jnp.float32,
        )
        memory = jnp.tile(base_memory, (b, 1, 1))
        initial_code = jax.nn.one_hot(initial_slots, 3, dtype=jnp.float32)
        memory = memory.at[:, 0, :3].add(initial_code)

        updater = SharedSegmentMemoryUpdater(
            name="shared_swap_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            segment_size=SWAP_SEGMENT_SIZE,
            dtype_mm="float32",
        )
        adapter = _single_read.SingleHistoryReadAdapter(
            name="shared_history_read_adapter",
            memory_width=self.memory_width,
            current_width=self.current_width,
            num_heads=self.adapter_heads,
            residual_scale=self.residual_scale,
        )
        readout = _sweep.SharedMemoryTokenReadout(name="shared_readout", width=self.current_width)
        base_current_tokens = self.param(
            "base_current_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_current_tokens, self.current_width),
            jnp.float32,
        )
        base_current_tokens = jnp.tile(base_current_tokens, (b, 1, 1))

        stage_logits = []
        stage_memories = []
        for stage_index in range(len(SWAP_SLICES)):
            memory = updater(memory, segment_tokens[:, stage_index])
            stage_memories.append(memory)
            current_tokens = adapter(base_current_tokens, memory)
            stage_logits.append(readout(current_tokens))

        stage_logits = jnp.stack(stage_logits, axis=1)
        stage_memories = jnp.stack(stage_memories, axis=1)
        logits_0, logits_1, logits_2 = stage_logits[:, 0], stage_logits[:, 1], stage_logits[:, 2]
        joint_logits = (logits_0[:, :, None, None] + logits_1[:, None, :, None] + logits_2[:, None, None, :]).reshape(
            b, 27
        )
        return joint_logits, stage_logits, stage_memories


@dataclasses.dataclass(frozen=True)
class VisualSemanticMemoryConfig(_base_model.Pi0MemCompressConfig):
    encoder_width: int = 256
    encoder_depth: int = 2
    encoder_heads: int = 8
    memory_width: int = 64
    memory_depth: int = 2
    memory_heads: int = 4
    adapter_heads: int = 4
    endpoint_memory_tokens: int = 128
    adapter_current_tokens: int = 256
    adapter_residual_scale: float = 1.0
    oracle_initial_slots: tuple[int, ...] = ()
    video_mode: str = "normal"

    def create(self, rng: at.KeyArrayLike) -> VisualSemanticMemoryModel:
        return VisualSemanticMemoryModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_visual_tracker(self) -> nnx.filterlib.Filter:
        tracker = nnx_utils.PathRegex(r".*HistoryThreeSwapVisualSemanticMemoryTracker.*")
        encoder = nnx_utils.PathRegex(r".*HistoryThreeSwapVisualSemanticMemoryTracker/swap_semantic_encoder.*")
        trainable = nnx.All(tracker, nnx.Not(encoder))
        return nnx.Not(trainable)


class VisualSemanticMemoryModel(_base_model.Pi0MemCompress):
    def __init__(self, config: VisualSemanticMemoryConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.oracle_initial_slots = config.oracle_initial_slots
        self.video_mode = config.video_mode
        self.HistoryThreeSwapVisualSemanticMemoryTracker = nnx_bridge.ToNNX(
            ThreeSwapVisualSemanticMemoryTracker(
                num_frames=HISTORY_FRAMES,
                input_width=1152,
                encoder_width=config.encoder_width,
                encoder_depth=config.encoder_depth,
                encoder_heads=config.encoder_heads,
                memory_width=config.memory_width,
                memory_depth=config.memory_depth,
                memory_heads=config.memory_heads,
                adapter_heads=config.adapter_heads,
                num_memory_tokens=config.endpoint_memory_tokens,
                num_current_tokens=config.adapter_current_tokens,
                current_width=1152,
                residual_scale=config.adapter_residual_scale,
                dtype_mm=config.dtype,
            )
        )
        fake_tokens = jnp.zeros((1, HISTORY_FRAMES, 256, 1152), dtype=jnp.bfloat16)
        fake_slots = jnp.zeros((1,), dtype=jnp.int32)
        self.HistoryThreeSwapVisualSemanticMemoryTracker.lazy_init(fake_tokens, fake_slots, rngs=rngs)

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
        if image.ndim != 5 or image.shape[1] != TOTAL_INPUT_FRAMES:
            raise ValueError(f"Visual semantic probe expects [B,{TOTAL_INPUT_FRAMES},H,W,C], got {image.shape}")
        if observation.episode_index is None:
            raise ValueError("Visual semantic probe requires observation.episode_index")

        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, :HISTORY_FRAMES]
        if self.video_mode == "shuffle_swaps":
            donor = jnp.roll(history_patches[:, SWAP_SLICES[0][0] : SWAP_SLICES[-1][1]], 1, axis=0)
            history_patches = history_patches.at[:, SWAP_SLICES[0][0] : SWAP_SLICES[-1][1]].set(donor)
        elif self.video_mode == "zero_swaps":
            history_patches = history_patches.at[:, SWAP_SLICES[0][0] : SWAP_SLICES[-1][1]].set(0)
        elif self.video_mode != "normal":
            raise ValueError(f"Unknown video_mode={self.video_mode!r}")

        episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
        lookup = jnp.asarray(self.oracle_initial_slots, dtype=jnp.int32)
        safe_episode = jnp.clip(episode_index, 0, lookup.shape[0] - 1)
        initial_slots = lookup[safe_episode]
        joint_logits, stage_logits, stage_memories = self.HistoryThreeSwapVisualSemanticMemoryTracker(
            history_patches, initial_slots
        )
        return joint_logits, {
            "history_mem": stage_memories.reshape(-1, stage_memories.shape[-2], stage_memories.shape[-1]),
            "stage_logits": stage_logits,
            "encoder_auxes": (),
        }


@dataclasses.dataclass(frozen=True)
class VisualSemanticCheckpointLoader:
    """Restore the frozen base and complete pretrained semantic encoder."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        tracker_root = "HistoryThreeSwapVisualSemanticMemoryTracker/"
        encoder_root = tracker_root + "swap_semantic_encoder/"
        source_encoder_root = "HistoryThreeSwapPairVisualClassifier/"
        result = {}
        exact_base = mapped_encoder = 0
        initialized_tracker = []
        initialized_base = []

        for key, reference in target.items():
            candidate = source.get(key)
            source_kind = "exact"
            if candidate is None or np.shape(candidate) != np.shape(reference):
                candidate = None
                if key.startswith(encoder_root):
                    relative = key.removeprefix(encoder_root)
                    source_candidate = source.get(source_encoder_root + relative)
                    if source_candidate is not None and np.shape(source_candidate) == np.shape(reference):
                        candidate = source_candidate
                        source_kind = "encoder"

            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                if source_kind == "encoder":
                    mapped_encoder += 1
                else:
                    exact_base += 1
            else:
                result[key] = reference
                if key.startswith(tracker_root):
                    initialized_tracker.append(key)
                else:
                    initialized_base.append(key)

        missing_encoder = [key for key in initialized_tracker if key.startswith(encoder_root)]
        if missing_encoder:
            raise ValueError(f"Frozen semantic encoder restore incomplete: {missing_encoder[:8]}")
        if initialized_base:
            raise ValueError(f"Unexpected frozen base initialization: {initialized_base[:8]}")
        print(
            "VisualSemanticCheckpointLoader: "
            f"exact_base={exact_base}, mapped_encoder={mapped_encoder}, "
            f"initialized_tracker={len(initialized_tracker)}, examples={initialized_tracker[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def run_self_test() -> None:
    tracker = ThreeSwapVisualSemanticMemoryTracker(
        num_frames=HISTORY_FRAMES,
        input_width=16,
        encoder_width=16,
        encoder_depth=2,
        encoder_heads=4,
        memory_width=16,
        memory_depth=2,
        memory_heads=4,
        adapter_heads=4,
        num_memory_tokens=8,
        num_current_tokens=8,
        current_width=32,
        dtype_mm="float32",
    )
    base = jax.random.normal(jax.random.key(1), (2, HISTORY_FRAMES, 256, 16))
    slots = jnp.asarray((0, 1), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), base, slots)
    _, reference, _ = tracker.apply(variables, base, slots)

    causal = []
    swap_effects = []
    for stage_index, (start, end) in enumerate(SWAP_SLICES):
        changed = base.at[:, start:end].set(
            jax.random.normal(jax.random.key(10 + stage_index), base[:, start:end].shape)
        )
        _, candidate, _ = tracker.apply(variables, changed, slots)
        causal.append(
            bool(
                np.allclose(
                    np.asarray(reference[:, :stage_index]),
                    np.asarray(candidate[:, :stage_index]),
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
        swap_effects.append(
            not np.allclose(np.asarray(reference[:, stage_index]), np.asarray(candidate[:, stage_index]))
        )

    outside = base.at[:, : SWAP_SLICES[0][0]].set(99.0)
    outside = outside.at[:, SWAP_SLICES[-1][1] :].set(-99.0)
    _, outside_candidate, _ = tracker.apply(variables, outside, slots)
    outside_ignored = np.allclose(np.asarray(reference), np.asarray(outside_candidate), rtol=0.0, atol=0.0)
    _, initial_candidate, _ = tracker.apply(variables, base, (slots + 1) % 3)
    initial_effect = not np.allclose(np.asarray(reference), np.asarray(initial_candidate))
    if not all(causal) or not all(swap_effects) or not outside_ignored or not initial_effect:
        raise AssertionError(
            "Visual semantic self-test failed: "
            f"causal={causal}, swap={swap_effects}, outside={outside_ignored}, initial={initial_effect}"
        )
    print(
        "Visual semantic self-test passed: "
        f"causal={causal}, swap_effects={swap_effects}, outside_ignored={outside_ignored}, "
        f"initial_effect={initial_effect}"
    )


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    model = VisualSemanticMemoryConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        num_frames=TOTAL_INPUT_FRAMES,
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
        encoder_width=args.encoder_width,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        memory_width=args.memory_width,
        memory_depth=args.memory_depth,
        memory_heads=args.memory_heads,
        adapter_heads=args.adapter_heads,
        endpoint_memory_tokens=args.memory_tokens,
        adapter_current_tokens=args.current_tokens,
        adapter_residual_scale=args.residual_scale,
        oracle_initial_slots=build_initial_slot_lookup(),
        video_mode=args.video_mode,
    )
    return _config.TrainConfig(
        name="pi0_shellgame_three_swap_visual_semantic_readout_memory_260809",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_visual_tracker(),
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
                    num_frames=TOTAL_INPUT_FRAMES,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=VisualSemanticCheckpointLoader(args.init_checkpoint),
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
    parser.add_argument("--init-checkpoint", default=DEFAULT_PAIR_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
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
    parser.add_argument("--video-mode", choices=("normal", "shuffle_swaps", "zero_swaps"), default="normal")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    _trainer.eval_step = multistage_eval_step
    _trainer.main(build_config(args, build_three_swap_labels()))


if __name__ == "__main__":
    main()
