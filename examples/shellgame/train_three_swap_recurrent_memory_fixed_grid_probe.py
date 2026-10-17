"""Test direct M=128 recurrent memory updates on the three-swap ShellGame.

Historical frames use only pretrained SigLIP patch embeddings and fixed 2x2
pooling (K=64).  A persistent M=128, width-256 memory cross-attends one
ten-frame segment at a time.  The same depth-2 updater is reused for five
segments: two reveal segments and three swap segments.  Endpoint memories are
read after swaps 0, 1, and 2 with one shared classifier.  Action loss is zero.

Unlike the preceding K64-state diagnostic, this model carries M=128 tokens
directly and therefore never expands a small recurrent state with the old
one-shot final compressor.
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


class SegmentMemoryUpdateBlock(nn.Module):
    """Cross-attend persistent memory to one segment, then refine memory."""

    width: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, memory, segment_tokens):
        memory_norm = nn.LayerNorm(name="memory_ln", dtype=self.dtype_mm)(memory)
        segment_norm = nn.LayerNorm(name="segment_ln", dtype=self.dtype_mm)(segment_tokens)
        update = nn.MultiHeadDotProductAttention(
            name="cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(memory_norm, segment_norm)
        memory = memory + update

        y = nn.LayerNorm(name="self_ln", dtype=self.dtype_mm)(memory)
        y = nn.MultiHeadDotProductAttention(
            name="self_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(y, y)
        memory = memory + y

        y = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(memory)
        y = nn.Dense(self.width * self.mlp_ratio, name="mlp_in", dtype=self.dtype_mm)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(y)
        return memory + y


class SharedSegmentMemoryUpdater(nn.Module):
    """A depth-2 updater reused at every reveal/swap segment."""

    width: int = 256
    depth: int = 2
    num_heads: int = 8
    segment_size: int = 10
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, memory, segment):
        if segment.ndim != 4 or segment.shape[1] != self.segment_size:
            raise ValueError(f"Expected [B,{self.segment_size},K,D] segment, got {segment.shape}")
        relative_pos = self.param(
            "relative_temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.segment_size, 1, self.width),
            segment.dtype,
        )
        segment_tokens = (segment + relative_pos).reshape(
            segment.shape[0], -1, self.width
        )
        for block_index in range(self.depth):
            memory = SegmentMemoryUpdateBlock(
                name=f"update_block_{block_index}",
                width=self.width,
                num_heads=self.num_heads,
                dtype_mm=self.dtype_mm,
            )(memory, segment_tokens)
        return nn.LayerNorm(name="state_output_ln", dtype=self.dtype_mm)(memory)


class ThreeSwapRecurrentMemoryTracker(nn.Module):
    """Maintain M memory tokens while applying one shared update per segment."""

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

        initial_memory = self.param(
            "initial_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.width),
            x.dtype,
        )
        memory = jnp.tile(initial_memory, (b, 1, 1))
        updater = SharedSegmentMemoryUpdater(
            name="shared_segment_memory_updater",
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=self.segment_size,
            dtype_mm=self.dtype_mm,
        )

        endpoint_memories = []
        for segment_index in range(NUM_SEGMENTS):
            start = segment_index * self.segment_size
            memory = updater(memory, x[:, start : start + self.segment_size])
            if segment_index in READOUT_SEGMENTS:
                endpoint_memories.append(memory)
        memory_batch = jnp.stack(endpoint_memories, axis=1).reshape(
            b * 3, self.num_memory_tokens, self.width
        )

        memory_batch = nn.LayerNorm(name="memory_output_ln", dtype=self.dtype_mm)(
            memory_batch
        )
        memory_batch = nn.Dense(
            self.input_width,
            name="memory_output_projection",
            dtype=self.dtype_mm,
        )(memory_batch)
        memory_batch = memory_batch - jnp.mean(memory_batch, axis=1, keepdims=True)
        memory_batch = nn.LayerNorm(name="pi0_output_ln", dtype=self.dtype_mm)(
            memory_batch
        )
        memory_batch = memory_batch - jnp.mean(memory_batch, axis=1, keepdims=True)

        logits = IntegratedCurrentReadout(
            name="shared_readout",
            input_width=self.input_width,
            width=self.width,
            num_classes=3,
            dtype_mm=self.dtype_mm,
        )(memory_batch)
        stage_logits = logits.reshape(b, 3, 3)
        stage_memories = memory_batch.reshape(
            b, 3, self.num_memory_tokens, self.input_width
        )

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
        return joint_logits, stage_logits, stage_memories


@dataclasses.dataclass(frozen=True)
class RecurrentMemoryProbeConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2
    endpoint_memory_tokens: int = 128

    def create(self, rng: at.KeyArrayLike) -> RecurrentMemoryProbeModel:
        return RecurrentMemoryProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_tracker_only(self) -> nnx.filterlib.Filter:
        tracker = nnx_utils.PathRegex(r".*HistoryThreeSwapRecurrentMemoryTracker.*")
        return nnx.Not(tracker)


class RecurrentMemoryProbeModel(_base_model.Pi0MemCompress):
    def __init__(self, config: RecurrentMemoryProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.HistoryThreeSwapRecurrentMemoryTracker = nnx_bridge.ToNNX(
            ThreeSwapRecurrentMemoryTracker(
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
        self.HistoryThreeSwapRecurrentMemoryTracker.lazy_init(fake_tokens, rngs=rngs)

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
            raise ValueError(f"Recurrent memory probe expects [B,61,H,W,C], got {image.shape}")

        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, :NUM_HISTORY_FRAMES]
        joint_logits, stage_logits, stage_memories = self.HistoryThreeSwapRecurrentMemoryTracker(
            history_patches
        )
        return joint_logits, {
            "history_mem": stage_memories.reshape(
                -1, stage_memories.shape[-2], stage_memories.shape[-1]
            ),
            "stage_logits": stage_logits,
            "encoder_auxes": (),
        }


@dataclasses.dataclass(frozen=True)
class RecurrentMemoryCheckpointLoader:
    """Initialize the recurrent memory from the trained one-shot endpoint model."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        result = {}
        exact = 0
        mapped = 0
        initialized = []
        target_root = "HistoryThreeSwapRecurrentMemoryTracker/"
        source_root = "HistoryThreeSwapSharedEndpointTracker/"
        source_history = source_root + "shared_history/"

        direct_mappings = (
            (target_root + "input_ln/", source_history + "input_ln/"),
            (target_root + "input_projection/", source_history + "input_projection/"),
            (
                target_root + "memory_output_ln/",
                source_history + "final_memory_compressor/output_ln/",
            ),
            (
                target_root + "memory_output_projection/",
                source_history + "final_memory_compressor/output_projection/",
            ),
            (
                target_root + "pi0_output_ln/",
                source_history + "final_memory_compressor/pi0_output_ln/",
            ),
            (target_root + "shared_readout/", source_root + "shared_readout/"),
        )

        def mapped_source_key(key: str) -> str | None:
            if key == target_root + "initial_memory":
                return source_history + "final_memory_compressor/memory_queries"
            for target_prefix, source_prefix in direct_mappings:
                if key.startswith(target_prefix):
                    return source_prefix + key.removeprefix(target_prefix)

            updater_prefix = target_root + "shared_segment_memory_updater/update_block_"
            if key.startswith(updater_prefix):
                remainder = key.removeprefix(updater_prefix)
                block_text, leaf = remainder.split("/", 1)
                block_index = int(block_text)
                compressor = source_history + "final_memory_compressor/"
                block = source_history + f"temporal_block_{block_index}/"
                submodule, subleaf = leaf.split("/", 1)
                if submodule == "memory_ln":
                    return compressor + "query_ln/" + subleaf
                if submodule == "segment_ln":
                    return compressor + "input_ln/" + subleaf
                if submodule == "cross_attention":
                    return compressor + "cross_attention/" + subleaf
                if submodule == "self_ln":
                    return block + "temporal_ln/" + subleaf
                if submodule == "self_attention":
                    return block + "temporal_attn/" + subleaf
                if submodule in ("mlp_ln", "mlp_in", "mlp_out"):
                    return compressor + submodule + "/" + subleaf
            if key.startswith(
                target_root + "shared_segment_memory_updater/state_output_ln/"
            ):
                return source_history + "final_memory_compressor/output_ln/" + key.rsplit("/", 1)[-1]
            return None

        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
                continue
            source_key = mapped_source_key(key)
            candidate = source.get(source_key) if source_key is not None else None
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                mapped += 1
            else:
                result[key] = reference
                initialized.append(key)

        allowed_new = target_root + "shared_segment_memory_updater/relative_temporal_pos_embedding"
        unexpected = [
            key
            for key in initialized
            if key.startswith(target_root) and key != allowed_new
        ]
        if unexpected:
            raise ValueError(f"Recurrent memory restore incomplete: {unexpected[:8]}")
        tracker_count = sum(key.startswith(target_root) for key in target)
        print(
            "RecurrentMemoryCheckpointLoader: "
            f"exact={exact}, mapped={mapped}, tracker={tracker_count}, "
            f"initialized={len(initialized)}, examples={initialized[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def run_causality_self_test() -> None:
    tracker = ThreeSwapRecurrentMemoryTracker(
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
    _, reference, _ = tracker.apply(variables, base)
    checks = []
    for start_frame, protected_stages in ((30, 1), (40, 2), (50, 3)):
        changed = base.at[:, start_frame:].set(
            jax.random.normal(jax.random.key(start_frame), base[:, start_frame:].shape)
        )
        _, candidate, _ = tracker.apply(variables, changed)
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
    model = RecurrentMemoryProbeConfig(
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
        name="pi0_shellgame_three_swap_recurrent_memory_joint_aux_260809",
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
        weight_loader=RecurrentMemoryCheckpointLoader(args.init_checkpoint),
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
