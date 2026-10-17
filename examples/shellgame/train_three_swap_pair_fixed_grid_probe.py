"""Test whether K64 visual tokens preserve which pair of cups was swapped.

Each ten-frame swap clip is classified independently as one of the three cup
pairs.  The three clips are folded into the batch dimension and therefore use
one shared depth-2 factorized space-time Transformer and one shared readout.
There is no initial-ball input, recurrent state, memory compression, or action
loss.  A factorized 27-way loss supervises all three clips simultaneously.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
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

from examples.shellgame.train_one_swap_history_probe import RAW_DATASET_ROOT
from examples.shellgame.train_one_swap_temporal_transformer_probe import FactorizedSpaceTimeBlock
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import DATASET_ROOT
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import multistage_eval_step
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
LABELS_PATH = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame/"
    "three_swap_pair_fixed_grid_probe/episode_labels.jsonl"
)
SLOT_ORDER = {"left": 0, "middle": 1, "right": 2}
SWAP_PAIRS = ("left-middle", "left-right", "middle-right")
JOINT_PAIR_CLASSES = tuple("|".join(values) for values in itertools.product(SWAP_PAIRS, repeat=3))
NUM_HISTORY_FRAMES = 60
SWAP_START_FRAME = 20
SWAP_END_FRAME = 50
SEGMENT_SIZE = 10
NUM_SEGMENTS = 3


def canonical_swap_pair(swap: list[str]) -> str:
    if len(swap) != 2 or swap[0] == swap[1]:
        raise ValueError(f"Invalid swap: {swap!r}")
    if any(slot not in SLOT_ORDER for slot in swap):
        raise ValueError(f"Unknown cup slot in swap: {swap!r}")
    first, second = sorted(swap, key=SLOT_ORDER.__getitem__)
    pair = f"{first}-{second}"
    if pair not in SWAP_PAIRS:
        raise ValueError(f"Invalid canonical swap pair: {pair}")
    return pair


def build_swap_pair_labels() -> pathlib.Path:
    """Build and validate the three swap-pair labels for all episodes."""
    with (DATASET_ROOT / "meta/episodes.jsonl").open("r", encoding="utf-8") as handle:
        lerobot_indices = {
            int(record["episode_index"]) for line in handle if line.strip() for record in (json.loads(line),)
        }

    records = []
    stage_counts = np.zeros((NUM_SEGMENTS, len(SWAP_PAIRS)), dtype=np.int64)
    raw_paths = sorted(RAW_DATASET_ROOT.glob("episode_*/metadata.json"))
    if len(raw_paths) != len(lerobot_indices):
        raise ValueError(f"Raw/LeRobot episode count mismatch: {len(raw_paths)} != {len(lerobot_indices)}")
    for raw_path in raw_paths:
        episode_index = int(raw_path.parent.name.split("_")[-1])
        if episode_index not in lerobot_indices:
            raise ValueError(f"Missing LeRobot episode_index={episode_index}")
        metadata = json.loads(raw_path.read_text(encoding="utf-8"))
        swaps = metadata["swaps"]
        if len(swaps) != NUM_SEGMENTS:
            raise ValueError(f"Expected three swaps in {raw_path}, got {len(swaps)}")
        pairs = [canonical_swap_pair(swap) for swap in swaps]
        for stage, pair in enumerate(pairs):
            stage_counts[stage, SWAP_PAIRS.index(pair)] += 1
        records.append(
            {
                "episode_index": episode_index,
                "swap_pair_0": pairs[0],
                "swap_pair_1": pairs[1],
                "swap_pair_2": pairs[2],
                "swap_pair_code": "|".join(pairs),
            }
        )

    expected_indices = list(range(len(records)))
    actual_indices = [record["episode_index"] for record in records]
    if actual_indices != expected_indices:
        raise ValueError("Swap-pair labels are not dense and ordered by episode_index")
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    print(f"Swap-pair labels: episodes={len(records)}, classes={SWAP_PAIRS}, stage_counts={stage_counts.tolist()}")
    return LABELS_PATH


class SharedSwapPairVisualClassifier(nn.Module):
    """Classify three independent swap clips with one shared visual encoder."""

    num_frames: int = NUM_HISTORY_FRAMES
    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    spatial_pool_factor: int = 2
    segment_size: int = SEGMENT_SIZE
    dropout: float = 0.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens, *, train: bool):
        b, t, n, d = patch_tokens.shape
        if (t, n, d) != (self.num_frames, 256, self.input_width):
            raise ValueError(f"Expected [B,{self.num_frames},256,{self.input_width}], got {patch_tokens.shape}")

        input_grid = int(np.sqrt(n))
        output_grid = input_grid // self.spatial_pool_factor
        x = patch_tokens[:, SWAP_START_FRAME:SWAP_END_FRAME].reshape(
            b,
            NUM_SEGMENTS,
            self.segment_size,
            output_grid,
            self.spatial_pool_factor,
            output_grid,
            self.spatial_pool_factor,
            d,
        )
        x = jnp.mean(x, axis=(4, 6)).reshape(b * NUM_SEGMENTS, self.segment_size, output_grid**2, d)
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(x)
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
            )(x, train=train)

        flat = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(x.reshape(b * NUM_SEGMENTS, -1, self.width))
        readout_query = self.param(
            "readout_query",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.width),
            flat.dtype,
        )
        query = jnp.tile(readout_query, (b * NUM_SEGMENTS, 1, 1))
        pooled = nn.MultiHeadDotProductAttention(
            name="readout_attention",
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
            dtype=self.dtype_mm,
        )(query, flat)
        pooled = nn.LayerNorm(name="readout_ln", dtype=self.dtype_mm)(pooled[:, 0])
        logits = nn.Dense(3, name="classifier", dtype=jnp.float32)(pooled.astype(jnp.float32))
        stage_logits = logits.reshape(b, NUM_SEGMENTS, 3)
        logits_0, logits_1, logits_2 = (stage_logits[:, 0], stage_logits[:, 1], stage_logits[:, 2])
        joint_logits = (logits_0[:, :, None, None] + logits_1[:, None, :, None] + logits_2[:, None, None, :]).reshape(
            b, 27
        )
        diagnostic_tokens = flat[:, :8]
        return joint_logits, stage_logits, diagnostic_tokens


@dataclasses.dataclass(frozen=True)
class SwapPairProbeConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2
    video_mode: str = "normal"

    def create(self, rng: at.KeyArrayLike) -> SwapPairProbeModel:
        return SwapPairProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_tracker_only(self) -> nnx.filterlib.Filter:
        tracker = nnx_utils.PathRegex(r".*HistoryThreeSwapPairVisualClassifier.*")
        return nnx.Not(tracker)


class SwapPairProbeModel(_base_model.Pi0MemCompress):
    def __init__(self, config: SwapPairProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.video_mode = config.video_mode
        self.HistoryThreeSwapPairVisualClassifier = nnx_bridge.ToNNX(
            SharedSwapPairVisualClassifier(
                num_frames=NUM_HISTORY_FRAMES,
                input_width=1152,
                width=config.temporal_width,
                depth=config.temporal_depth,
                num_heads=config.temporal_heads,
                spatial_pool_factor=config.spatial_pool_factor,
                segment_size=SEGMENT_SIZE,
                dropout=0.0,
                dtype_mm=config.dtype,
            )
        )
        fake_tokens = jnp.zeros((1, NUM_HISTORY_FRAMES, 256, 1152), dtype=jnp.bfloat16)
        self.HistoryThreeSwapPairVisualClassifier.lazy_init(fake_tokens, train=False, rngs=rngs)

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
            raise ValueError(f"Swap-pair probe expects [B,61,H,W,C], got {image.shape}")
        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, :NUM_HISTORY_FRAMES]
        if self.video_mode == "shuffle_batch":
            donor = jnp.roll(
                history_patches[:, SWAP_START_FRAME:SWAP_END_FRAME], 1, axis=0
            )
            history_patches = history_patches.at[
                :, SWAP_START_FRAME:SWAP_END_FRAME
            ].set(donor)
        elif self.video_mode == "zero_swaps":
            history_patches = history_patches.at[
                :, SWAP_START_FRAME:SWAP_END_FRAME
            ].set(0)
        elif self.video_mode != "normal":
            raise ValueError(f"Unknown video_mode={self.video_mode!r}")
        joint_logits, stage_logits, diagnostic_tokens = self.HistoryThreeSwapPairVisualClassifier(
            history_patches, train=train
        )
        return joint_logits, {
            "history_mem": diagnostic_tokens,
            "stage_logits": stage_logits,
            "encoder_auxes": (),
        }


@dataclasses.dataclass(frozen=True)
class SwapPairCheckpointLoader:
    """Restore the frozen policy and initialize only the new visual probe."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        target_root = "HistoryThreeSwapPairVisualClassifier/"
        result = {}
        exact = 0
        initialized = []
        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
            else:
                result[key] = reference
                initialized.append(key)
        unexpected = [key for key in initialized if not key.startswith(target_root)]
        if unexpected:
            raise ValueError(f"Unexpected non-probe initialization: {unexpected[:8]}")
        tracker_count = sum(key.startswith(target_root) for key in target)
        print(
            "SwapPairCheckpointLoader: "
            f"exact={exact}, initialized_probe={len(initialized)}/{tracker_count}, "
            f"examples={initialized[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def run_isolation_self_test() -> None:
    classifier = SharedSwapPairVisualClassifier(
        num_frames=NUM_HISTORY_FRAMES,
        input_width=16,
        width=16,
        depth=2,
        num_heads=4,
        spatial_pool_factor=2,
        segment_size=SEGMENT_SIZE,
        dtype_mm="float32",
    )
    base = jax.random.normal(jax.random.key(1), (1, NUM_HISTORY_FRAMES, 256, 16))
    variables = classifier.init(jax.random.key(0), base, train=False)
    _, reference, _ = classifier.apply(variables, base, train=False)

    outside = base.at[:, :SWAP_START_FRAME].set(99.0)
    outside = outside.at[:, SWAP_END_FRAME:].set(-99.0)
    _, outside_logits, _ = classifier.apply(variables, outside, train=False)
    outside_ignored = np.allclose(np.asarray(reference), np.asarray(outside_logits), rtol=0.0, atol=0.0)
    segment_checks = []
    for segment in range(NUM_SEGMENTS):
        start = SWAP_START_FRAME + segment * SEGMENT_SIZE
        changed = base.at[:, start : start + SEGMENT_SIZE].set(
            jax.random.normal(jax.random.key(10 + segment), base[:, start : start + SEGMENT_SIZE].shape)
        )
        _, candidate, _ = classifier.apply(variables, changed, train=False)
        unaffected = [stage for stage in range(NUM_SEGMENTS) if stage != segment]
        segment_checks.append(
            bool(
                np.allclose(
                    np.asarray(reference[:, unaffected]),
                    np.asarray(candidate[:, unaffected]),
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
    if not outside_ignored or not all(segment_checks):
        raise AssertionError(f"Isolation test failed: outside={outside_ignored}, segments={segment_checks}")
    print(f"Swap-pair isolation self-test passed: outside={outside_ignored}, segments={segment_checks}")


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    model = SwapPairProbeConfig(
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
        video_mode=args.video_mode,
    )
    return _config.TrainConfig(
        name="pi0_shellgame_three_swap_pair_fixed_grid_probe_260809",
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
        weight_loader=SwapPairCheckpointLoader(args.init_checkpoint),
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
            label_key="swap_pair_code",
            classes=JOINT_PAIR_CLASSES,
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
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--overfit-samples-per-class", type=int, default=0)
    parser.add_argument(
        "--video-mode",
        choices=("normal", "shuffle_batch", "zero_swaps"),
        default="normal",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_isolation_self_test()
        return
    _trainer.eval_step = multistage_eval_step
    _trainer.main(build_config(args, build_swap_pair_labels()))


if __name__ == "__main__":
    main()
