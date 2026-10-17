"""Test shared recurrent segment updates on the three-swap ShellGame.

This is an isolated control experiment.  It keeps the successful fixed-grid
interface (SigLIP patch embeddings, K=64, width=256, depth=2, M=128), but
replaces one-shot temporal aggregation with a recurrent state update whose
weights are shared across five ten-frame segments:

    reveal 0..9 -> reveal 10..19 -> swap_0 20..29
    -> swap_1 30..39 -> swap_2 40..49

The 64 spatial state tokens are updated once per segment.  The same memory
compressor and the same three-way readout are applied after the last three
updates.  Frames 50..59 are deliberately ignored, and Pi0 action loss is zero.
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
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import DATASET_ROOT
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import JOINT_CLASSES
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import multistage_eval_step
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.models import siglip_mem_fixed_grid_temporal as _fixed_siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer

DEFAULT_INIT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_shared_endpoint_joint_aux_260809/"
    "three_swap_shared_endpoint_joint_aux_260809/599/params"
)
NUM_HISTORY_FRAMES = 60
USED_HISTORY_FRAMES = 50
SEGMENT_SIZE = 10
NUM_SEGMENTS = USED_HISTORY_FRAMES // SEGMENT_SIZE
READOUT_SEGMENTS = (2, 3, 4)


class SharedSegmentStateUpdater(nn.Module):
    """Update K spatial state tokens from one segment using shared weights."""

    width: int = 256
    depth: int = 2
    num_heads: int = 8
    segment_size: int = 10
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, state, segment):
        if state.ndim != 3 or segment.ndim != 4:
            raise ValueError(f"Expected state [B,K,D] and segment [B,S,K,D], got {state.shape}, {segment.shape}")
        if segment.shape[1] != self.segment_size:
            raise ValueError(f"Expected segment size {self.segment_size}, got {segment.shape[1]}")
        if state.shape[0] != segment.shape[0] or state.shape[1:] != segment.shape[2:]:
            raise ValueError(f"State/segment mismatch: {state.shape} versus {segment.shape}")

        # Position zero is a persistent state slot.  Positions 1..S are the
        # relative frame positions inside every segment; this embedding and all
        # Transformer weights are reused at every recurrent update.
        x = jnp.concatenate((state[:, None], segment), axis=1)
        relative_pos = self.param(
            "relative_temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.segment_size + 1, 1, self.width),
            x.dtype,
        )
        x = x + relative_pos
        for block_index in range(self.depth):
            x = _fixed_siglip.FactorizedSpaceTimeBlock(
                name=f"temporal_block_{block_index}",
                width=self.width,
                num_heads=self.num_heads,
                dropout=0.0,
                dtype_mm=self.dtype_mm,
            )(x, deterministic=True)

        # Only the dedicated state slot is carried across segment boundaries.
        return nn.LayerNorm(name="state_output_ln", dtype=self.dtype_mm)(x[:, 0])


class ThreeSwapRecurrentSegmentTracker(nn.Module):
    """Track the hidden cup with five shared recurrent segment updates."""

    num_frames: int = NUM_HISTORY_FRAMES
    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    spatial_pool_factor: int = 2
    num_memory_tokens: int = 128
    segment_size: int = SEGMENT_SIZE
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens):
        b, t, n, d = patch_tokens.shape
        if (t, n, d) != (self.num_frames, 256, self.input_width):
            raise ValueError(f"Expected [B,{self.num_frames},256,{self.input_width}], got {patch_tokens.shape}")
        if USED_HISTORY_FRAMES % self.segment_size:
            raise ValueError("Used history length must be divisible by segment size")

        input_grid = int(np.sqrt(n))
        output_grid = input_grid // self.spatial_pool_factor
        x = patch_tokens[:, :USED_HISTORY_FRAMES].reshape(
            b,
            USED_HISTORY_FRAMES,
            output_grid,
            self.spatial_pool_factor,
            output_grid,
            self.spatial_pool_factor,
            d,
        )
        x = jnp.mean(x, axis=(3, 5)).reshape(
            b, USED_HISTORY_FRAMES, output_grid**2, d
        )
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(x)
        x = nn.Dense(self.width, name="input_projection", dtype=self.dtype_mm)(x)

        initial_state = self.param(
            "initial_state",
            nn.initializers.normal(stddev=0.02),
            (1, output_grid**2, self.width),
            x.dtype,
        )
        state = jnp.tile(initial_state, (b, 1, 1))
        updater = SharedSegmentStateUpdater(
            name="shared_segment_updater",
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=self.segment_size,
            dtype_mm=self.dtype_mm,
        )

        endpoint_states = []
        for segment_index in range(NUM_SEGMENTS):
            start = segment_index * self.segment_size
            segment = x[:, start : start + self.segment_size]
            state = updater(state, segment)
            if segment_index in READOUT_SEGMENTS:
                endpoint_states.append(state)

        if len(endpoint_states) != 3:
            raise AssertionError(f"Expected three endpoint states, got {len(endpoint_states)}")
        stacked_states = jnp.stack(endpoint_states, axis=1)
        state_batch = stacked_states.reshape(b * 3, 1, output_grid**2, self.width)
        memories = _fixed_siglip.FinalMemoryCompressor(
            name="shared_final_memory_compressor",
            width=self.width,
            output_width=self.input_width,
            num_memory_tokens=self.num_memory_tokens,
            num_heads=self.num_heads,
            dropout=0.0,
            dtype_mm=self.dtype_mm,
        )(state_batch, deterministic=True)
        logits = IntegratedCurrentReadout(
            name="shared_readout",
            input_width=self.input_width,
            width=self.width,
            num_classes=3,
            dtype_mm=self.dtype_mm,
        )(memories)
        stage_logits = logits.reshape(b, 3, 3)

        logits_0, logits_1, logits_2 = (
            stage_logits[:, 0],
            stage_logits[:, 1],
            stage_logits[:, 2],
        )
        joint_logits = (
            logits_0[:, :, None, None]
            + logits_1[:, None, :, None]
            + logits_2[:, None, None, :]
        ).reshape(b, 27)
        return joint_logits, stage_logits


@dataclasses.dataclass(frozen=True)
class RecurrentSegmentProbeConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2
    endpoint_memory_tokens: int = 128

    def create(self, rng: at.KeyArrayLike) -> RecurrentSegmentProbeModel:
        return RecurrentSegmentProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_tracker_only(self) -> nnx.filterlib.Filter:
        tracker = nnx_utils.PathRegex(r".*HistoryThreeSwapRecurrentSegmentTracker.*")
        return nnx.Not(tracker)


class RecurrentSegmentProbeModel(_base_model.Pi0MemCompress):
    def __init__(self, config: RecurrentSegmentProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.HistoryThreeSwapRecurrentSegmentTracker = nnx_bridge.ToNNX(
            ThreeSwapRecurrentSegmentTracker(
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
        self.HistoryThreeSwapRecurrentSegmentTracker.lazy_init(fake_tokens, rngs=rngs)

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
            raise ValueError(f"Recurrent segment probe expects [B,61,H,W,C], got {image.shape}")

        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, :NUM_HISTORY_FRAMES]
        joint_logits, stage_logits = self.HistoryThreeSwapRecurrentSegmentTracker(
            history_patches
        )
        return joint_logits, {
            "history_mem": stage_logits,
            "encoder_auxes": (),
        }


@dataclasses.dataclass(frozen=True)
class RecurrentSegmentCheckpointLoader:
    """Transplant all compatible one-shot endpoint weights into the recurrent tracker."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        result = {}
        exact = 0
        mapped = 0
        initialized = []
        target_root = "HistoryThreeSwapRecurrentSegmentTracker/"
        source_root = "HistoryThreeSwapSharedEndpointTracker/"

        mappings = (
            (target_root + "input_ln/", source_root + "shared_history/input_ln/"),
            (target_root + "input_projection/", source_root + "shared_history/input_projection/"),
            (
                target_root + "shared_segment_updater/temporal_block_",
                source_root + "shared_history/temporal_block_",
            ),
            (
                target_root + "shared_final_memory_compressor/",
                source_root + "shared_history/final_memory_compressor/",
            ),
            (target_root + "shared_readout/", source_root + "shared_readout/"),
        )

        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
                continue

            candidate = None
            for target_prefix, source_prefix in mappings:
                if key.startswith(target_prefix):
                    candidate = source.get(source_prefix + key.removeprefix(target_prefix))
                    break
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                mapped += 1
            else:
                result[key] = reference
                initialized.append(key)

        tracker_keys = [key for key in target if key.startswith(target_root)]
        allowed_new_suffixes = (
            "initial_state",
            "shared_segment_updater/relative_temporal_pos_embedding",
            "shared_segment_updater/state_output_ln/bias",
            "shared_segment_updater/state_output_ln/scale",
        )
        unexpected = [
            key
            for key in initialized
            if key.startswith(target_root)
            and not any(key.endswith(suffix) for suffix in allowed_new_suffixes)
        ]
        if unexpected:
            raise ValueError(f"Recurrent tracker restore incomplete: {unexpected[:8]}")
        print(
            "RecurrentSegmentCheckpointLoader: "
            f"exact={exact}, mapped={mapped}, tracker={len(tracker_keys)}, "
            f"initialized={len(initialized)}, examples={initialized[:8]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def run_causality_self_test() -> None:
    """Numerically verify that later segments cannot affect earlier readouts."""
    tracker = ThreeSwapRecurrentSegmentTracker(
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
    base = jax.random.normal(jax.random.key(1), (1, NUM_HISTORY_FRAMES, 256, 16))
    variables = tracker.init(jax.random.key(0), base)
    _, reference = tracker.apply(variables, base)

    checks = []
    for start_frame, protected_stages in ((30, 1), (40, 2), (50, 3)):
        changed = base.at[:, start_frame:].set(
            jax.random.normal(
                jax.random.key(start_frame),
                base[:, start_frame:].shape,
            )
        )
        _, candidate = tracker.apply(variables, changed)
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
    if not all(checks):
        raise AssertionError(f"Causality self-test failed: {checks}")
    print(f"Causality self-test passed: {checks}")


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    model = RecurrentSegmentProbeConfig(
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
    )
    return _config.TrainConfig(
        name="pi0_shellgame_three_swap_recurrent_segment_joint_aux_260809",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_tracker_only(),
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
        weight_loader=RecurrentSegmentCheckpointLoader(args.init_checkpoint),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=args.steps,
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=10.0),
        ema_decay=None,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=args.steps,
        keep_period=args.steps,
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
            disable_train_augmentation=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=100)
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
