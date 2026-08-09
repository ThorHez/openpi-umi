"""Test whether the recurrent memory can learn a reusable three-swap operator.

The true ``initial_ball_cup`` is injected as a parameter-free one-hot code in
the persistent M=128 memory.  Reveal frames 0..19 are never shown to the
tracker.  The same depth-2 updater then consumes swap clips 20..29, 30..39,
and 40..49, and one shared readout predicts the hidden-ball slot after every
swap.  Action loss is disabled.

This isolates state transition learning from initial ball/cup perception.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
from examples.shellgame.train_one_swap_history_probe import RAW_DATASET_ROOT
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import DATASET_ROOT
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import JOINT_CLASSES
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import SLOTS
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import multistage_eval_step
from examples.shellgame.train_three_swap_recurrent_memory_fixed_grid_probe import SharedSegmentMemoryUpdater
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer

DEFAULT_INIT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_recurrent_memory_joint_aux_260809/"
    "recurrent_memory_overfit54_260809/299/params"
)
NUM_HISTORY_FRAMES = 60
SWAP_START_FRAME = 20
SWAP_END_FRAME = 50
SEGMENT_SIZE = 10
NUM_SWAP_SEGMENTS = 3


def build_initial_slot_lookup() -> tuple[int, ...]:
    """Return a dense episode_index -> initial slot lookup and validate it."""
    slot_to_index = {slot: index for index, slot in enumerate(SLOTS)}
    records: dict[int, int] = {}
    for metadata_path in sorted(RAW_DATASET_ROOT.glob("episode_*/metadata.json")):
        episode_index = int(metadata_path.parent.name.split("_")[-1])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        slot = str(metadata["initial_ball_cup"])
        if slot not in slot_to_index:
            raise ValueError(f"Invalid initial_ball_cup={slot!r} in {metadata_path}")
        if episode_index in records:
            raise ValueError(f"Duplicate episode_index={episode_index}")
        records[episode_index] = slot_to_index[slot]

    if not records:
        raise ValueError(f"No episode metadata found below {RAW_DATASET_ROOT}")
    expected = set(range(max(records) + 1))
    missing = sorted(expected - records.keys())
    if missing:
        raise ValueError(f"Initial-slot lookup has missing episodes: {missing[:10]}")
    lookup = tuple(records[index] for index in range(len(records)))
    counts = np.bincount(np.asarray(lookup), minlength=3)
    print(f"Oracle initial-slot lookup: episodes={len(lookup)}, counts={counts.tolist()}")
    return lookup


class ThreeSwapOracleInitialRecurrentMemoryTracker(nn.Module):
    """Start from the true initial slot and recurrently consume only swaps."""

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
        x = patch_tokens[:, SWAP_START_FRAME:SWAP_END_FRAME].reshape(
            b,
            SWAP_END_FRAME - SWAP_START_FRAME,
            output_grid,
            self.spatial_pool_factor,
            output_grid,
            self.spatial_pool_factor,
            d,
        )
        x = jnp.mean(x, axis=(3, 5)).reshape(b, SWAP_END_FRAME - SWAP_START_FRAME, output_grid**2, d)
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(x)
        x = nn.Dense(self.width, name="input_projection", dtype=self.dtype_mm)(x)

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.width),
            x.dtype,
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
            start = segment_index * self.segment_size
            memory = updater(memory, x[:, start : start + self.segment_size])
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
        logits_0, logits_1, logits_2 = (stage_logits[:, 0], stage_logits[:, 1], stage_logits[:, 2])
        joint_logits = (logits_0[:, :, None, None] + logits_1[:, None, :, None] + logits_2[:, None, None, :]).reshape(
            b, 27
        )
        return joint_logits, stage_logits, stage_memories


@dataclasses.dataclass(frozen=True)
class OracleInitialProbeConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2
    endpoint_memory_tokens: int = 128
    oracle_initial_slots: tuple[int, ...] = ()
    oracle_mode: str = "correct"
    video_mode: str = "normal"

    def create(self, rng: at.KeyArrayLike) -> OracleInitialProbeModel:
        return OracleInitialProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_tracker_only(self) -> nnx.filterlib.Filter:
        tracker = nnx_utils.PathRegex(r".*HistoryThreeSwapOracleRecurrentMemoryTracker.*")
        return nnx.Not(tracker)


class OracleInitialProbeModel(_base_model.Pi0MemCompress):
    def __init__(self, config: OracleInitialProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.oracle_initial_slots = config.oracle_initial_slots
        self.oracle_mode = config.oracle_mode
        self.video_mode = config.video_mode
        self.HistoryThreeSwapOracleRecurrentMemoryTracker = nnx_bridge.ToNNX(
            ThreeSwapOracleInitialRecurrentMemoryTracker(
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
        self.HistoryThreeSwapOracleRecurrentMemoryTracker.lazy_init(fake_tokens, fake_slots, rngs=rngs)

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
            raise ValueError(f"Oracle probe expects [B,61,H,W,C], got {image.shape}")
        if observation.episode_index is None:
            raise ValueError("Oracle probe requires observation.episode_index")

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

        joint_logits, stage_logits, stage_memories = self.HistoryThreeSwapOracleRecurrentMemoryTracker(
            history_patches, initial_slots
        )
        return joint_logits, {
            "history_mem": stage_memories.reshape(-1, stage_memories.shape[-2], stage_memories.shape[-1]),
            "stage_logits": stage_logits,
            "encoder_auxes": (),
        }


@dataclasses.dataclass(frozen=True)
class OracleInitialCheckpointLoader:
    """Restore the same recurrent updater/readout; rename initial memory only."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        target_root = "HistoryThreeSwapOracleRecurrentMemoryTracker/"
        source_root = "HistoryThreeSwapRecurrentMemoryTracker/"
        result = {}
        exact = 0
        mapped = 0
        initialized = []
        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
                continue
            relative = key.removeprefix(target_root) if key.startswith(target_root) else None
            if relative == "base_memory":
                source_key = source_root + "initial_memory"
            elif relative is not None:
                source_key = source_root + relative
            else:
                source_key = None
            candidate = source.get(source_key) if source_key is not None else None
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                mapped += 1
            else:
                result[key] = reference
                initialized.append(key)

        unexpected = [key for key in initialized if key.startswith(target_root)]
        if unexpected:
            raise ValueError(f"Oracle recurrent restore incomplete: {unexpected[:8]}")
        print(
            "OracleInitialCheckpointLoader: "
            f"exact={exact}, mapped={mapped}, initialized={len(initialized)}, "
            f"examples={initialized[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def run_causality_self_test() -> None:
    tracker = ThreeSwapOracleInitialRecurrentMemoryTracker(
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
    _, different_oracle, _ = tracker.apply(variables, base, (slots + 1) % 3)
    oracle_effect = not np.allclose(np.asarray(reference), np.asarray(different_oracle))
    if not all(checks) or not oracle_effect:
        raise AssertionError(f"Oracle causality self-test failed: causality={checks}, oracle_effect={oracle_effect}")
    print(f"Oracle causality self-test passed: causality={checks}, oracle_effect={oracle_effect}")


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    initial_slots = build_initial_slot_lookup()
    model = OracleInitialProbeConfig(
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
        name="pi0_shellgame_three_swap_oracle_initial_recurrent_memory_260809",
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
        weight_loader=OracleInitialCheckpointLoader(args.init_checkpoint),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            # ``steps=0`` is an evaluation-only run: the trainer skips the
            # loop/checkpoint save and executes its final validation once.
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
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=100)
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
