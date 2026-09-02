"""Test the M128 updater with ground-truth swap-pair inputs.

The true initial ball slot and each of the three true swapped cup pairs are
looked up from episode metadata.  Pair identities are injected parameter-free
as one-hot codes into otherwise-zero 10 x 64 x 256 segment tokens.  The exact
same shared M128 recurrent updater and readout used by the failed visual probe
are then trained to predict the ball slot after every swap.

This removes perception and the visual/updater interface from the experiment.
Action loss is disabled.
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
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import multistage_eval_step
from examples.shellgame.train_three_swap_oracle_initial_recurrent_memory_probe import build_initial_slot_lookup
from examples.shellgame.train_three_swap_pair_fixed_grid_probe import SWAP_PAIRS
from examples.shellgame.train_three_swap_pair_fixed_grid_probe import canonical_swap_pair
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
    "pi0_shellgame_three_swap_oracle_initial_recurrent_memory_260809/"
    "oracle_initial_full600_260809/599/params"
)
NUM_SWAP_SEGMENTS = 3
SEGMENT_SIZE = 10
SPATIAL_TOKENS = 64


def build_swap_pair_lookup() -> tuple[tuple[int, int, int], ...]:
    """Return dense episode_index -> three canonical swap-pair class ids."""
    records: dict[int, tuple[int, int, int]] = {}
    stage_counts = np.zeros((NUM_SWAP_SEGMENTS, len(SWAP_PAIRS)), dtype=np.int64)
    for metadata_path in sorted(RAW_DATASET_ROOT.glob("episode_*/metadata.json")):
        episode_index = int(metadata_path.parent.name.split("_")[-1])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        swaps = metadata["swaps"]
        if len(swaps) != NUM_SWAP_SEGMENTS:
            raise ValueError(f"Expected three swaps in {metadata_path}, got {len(swaps)}")
        pair_ids = tuple(SWAP_PAIRS.index(canonical_swap_pair(swap)) for swap in swaps)
        if episode_index in records:
            raise ValueError(f"Duplicate episode_index={episode_index}")
        records[episode_index] = pair_ids
        for stage, pair_id in enumerate(pair_ids):
            stage_counts[stage, pair_id] += 1

    if not records:
        raise ValueError(f"No metadata found below {RAW_DATASET_ROOT}")
    expected = set(range(max(records) + 1))
    missing = sorted(expected - records.keys())
    if missing:
        raise ValueError(f"Swap-pair lookup has missing episodes: {missing[:10]}")
    lookup = tuple(records[index] for index in range(len(records)))
    print(
        f"Oracle swap-pair lookup: episodes={len(lookup)}, classes={SWAP_PAIRS}, stage_counts={stage_counts.tolist()}"
    )
    return lookup


class ThreeSwapOraclePairRecurrentMemoryTracker(nn.Module):
    """Run the existing recurrent memory on parameter-free pair codes."""

    width: int = 256
    input_width: int = 1152
    depth: int = 2
    num_heads: int = 8
    num_memory_tokens: int = 128
    segment_size: int = SEGMENT_SIZE
    spatial_tokens: int = SPATIAL_TOKENS
    oracle_code_scale: float = 1.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, initial_slots, swap_pair_ids):
        if initial_slots.ndim != 1:
            raise ValueError(f"Expected initial slots [B], got {initial_slots.shape}")
        b = initial_slots.shape[0]
        if swap_pair_ids.shape != (b, NUM_SWAP_SEGMENTS):
            raise ValueError(f"Expected swap pairs [B,3], got {swap_pair_ids.shape}")

        pair_codes = jax.nn.one_hot(swap_pair_ids, len(SWAP_PAIRS), dtype=jnp.dtype(self.dtype_mm))
        segment_tokens = jnp.zeros(
            (b, NUM_SWAP_SEGMENTS, self.segment_size, self.spatial_tokens, self.width),
            dtype=pair_codes.dtype,
        )
        pair_codes = pair_codes[:, :, None, None, :]
        segment_tokens = segment_tokens.at[..., : len(SWAP_PAIRS)].add(self.oracle_code_scale * pair_codes)

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.width),
            segment_tokens.dtype,
        )
        memory = jnp.tile(base_memory, (b, 1, 1))
        initial_code = jax.nn.one_hot(initial_slots, 3, dtype=memory.dtype)
        memory = memory.at[:, 0, :3].add(self.oracle_code_scale * initial_code)

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
            memory = updater(memory, segment_tokens[:, segment_index])
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
class OraclePairProbeConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    endpoint_memory_tokens: int = 128
    oracle_initial_slots: tuple[int, ...] = ()
    oracle_swap_pairs: tuple[tuple[int, int, int], ...] = ()
    pair_mode: str = "correct"

    def create(self, rng: at.KeyArrayLike) -> OraclePairProbeModel:
        return OraclePairProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_tracker_only(self) -> nnx.filterlib.Filter:
        tracker = nnx_utils.PathRegex(r".*HistoryThreeSwapOraclePairRecurrentMemoryTracker.*")
        return nnx.Not(tracker)


class OraclePairProbeModel(_base_model.Pi0MemCompress):
    def __init__(self, config: OraclePairProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.oracle_initial_slots = config.oracle_initial_slots
        self.oracle_swap_pairs = config.oracle_swap_pairs
        self.pair_mode = config.pair_mode
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker = nnx_bridge.ToNNX(
            ThreeSwapOraclePairRecurrentMemoryTracker(
                width=config.temporal_width,
                input_width=1152,
                depth=config.temporal_depth,
                num_heads=config.temporal_heads,
                num_memory_tokens=config.endpoint_memory_tokens,
                segment_size=SEGMENT_SIZE,
                spatial_tokens=SPATIAL_TOKENS,
                dtype_mm=config.dtype,
            )
        )
        fake_slots = jnp.zeros((1,), dtype=jnp.int32)
        fake_pairs = jnp.zeros((1, NUM_SWAP_SEGMENTS), dtype=jnp.int32)
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker.lazy_init(fake_slots, fake_pairs, rngs=rngs)

    def compute_history_classification(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ):
        del rng, train
        if observation.episode_index is None:
            raise ValueError("Oracle pair probe requires observation.episode_index")
        episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
        initial_lookup = jnp.asarray(self.oracle_initial_slots, dtype=jnp.int32)
        pair_lookup = jnp.asarray(self.oracle_swap_pairs, dtype=jnp.int32)
        safe_episode = jnp.clip(episode_index, 0, initial_lookup.shape[0] - 1)
        initial_slots = initial_lookup[safe_episode]
        swap_pairs = pair_lookup[safe_episode]
        if self.pair_mode == "roll":
            swap_pairs = (swap_pairs + 1) % len(SWAP_PAIRS)
        elif self.pair_mode == "shuffle_batch":
            swap_pairs = jnp.roll(swap_pairs, 1, axis=0)
        elif self.pair_mode != "correct":
            raise ValueError(f"Unknown pair_mode={self.pair_mode!r}")

        joint_logits, stage_logits, stage_memories = self.HistoryThreeSwapOraclePairRecurrentMemoryTracker(
            initial_slots, swap_pairs
        )
        return joint_logits, {
            "history_mem": stage_memories.reshape(-1, stage_memories.shape[-2], stage_memories.shape[-1]),
            "stage_logits": stage_logits,
            "encoder_auxes": (),
        }


@dataclasses.dataclass(frozen=True)
class OraclePairCheckpointLoader:
    """Restore every recurrent target leaf from the Oracle visual baseline."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        target_root = "HistoryThreeSwapOraclePairRecurrentMemoryTracker/"
        source_root = "HistoryThreeSwapOracleRecurrentMemoryTracker/"
        result = {}
        exact = mapped = 0
        initialized = []
        for key, reference in target.items():
            candidate = source.get(key)
            source_kind = "exact"
            if key.startswith(target_root):
                relative = key.removeprefix(target_root)
                candidate = source.get(source_root + relative)
                source_kind = "mapped"
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                if source_kind == "exact":
                    exact += 1
                else:
                    mapped += 1
            else:
                result[key] = reference
                initialized.append(key)

        unexpected = [key for key in initialized if key.startswith(target_root)]
        if unexpected:
            raise ValueError(f"Oracle pair recurrent restore incomplete: {unexpected[:8]}")
        print(
            f"OraclePairCheckpointLoader: exact={exact}, mapped={mapped}, "
            f"initialized={len(initialized)}, examples={initialized[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def run_causality_self_test() -> None:
    tracker = ThreeSwapOraclePairRecurrentMemoryTracker(
        width=16,
        input_width=16,
        depth=2,
        num_heads=4,
        num_memory_tokens=8,
        segment_size=SEGMENT_SIZE,
        spatial_tokens=SPATIAL_TOKENS,
        dtype_mm="float32",
    )
    slots = jnp.asarray((0, 1), dtype=jnp.int32)
    pairs = jnp.asarray(((0, 1, 2), (2, 1, 0)), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), slots, pairs)
    _, reference, _ = tracker.apply(variables, slots, pairs)
    checks = []
    for changed_stage in range(NUM_SWAP_SEGMENTS):
        changed = pairs.at[:, changed_stage].set((pairs[:, changed_stage] + 1) % 3)
        _, candidate, _ = tracker.apply(variables, slots, changed)
        checks.append(
            bool(
                np.allclose(
                    np.asarray(reference[:, :changed_stage]),
                    np.asarray(candidate[:, :changed_stage]),
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
    _, different_initial, _ = tracker.apply(variables, (slots + 1) % 3, pairs)
    initial_effect = not np.allclose(np.asarray(reference), np.asarray(different_initial))
    if not all(checks) or not initial_effect:
        raise AssertionError(f"Oracle pair self-test failed: causality={checks}, initial_effect={initial_effect}")
    print(f"Oracle pair self-test passed: causality={checks}, initial_effect={initial_effect}")


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    model = OraclePairProbeConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        # Images are deliberately unused in this Oracle-input diagnostic.
        # Load one frame only so validation is not dominated by video decode.
        num_frames=1,
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
        endpoint_memory_tokens=128,
        oracle_initial_slots=build_initial_slot_lookup(),
        oracle_swap_pairs=build_swap_pair_lookup(),
        pair_mode=args.pair_mode,
    )
    return _config.TrainConfig(
        name="pi0_shellgame_three_swap_oracle_pair_recurrent_memory_260809",
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
                    num_frames=1,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=OraclePairCheckpointLoader(args.init_checkpoint),
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
            overfit_samples_per_class=0,
            disable_train_augmentation=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--pair-mode", choices=("correct", "roll", "shuffle_batch"), default="correct")
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
