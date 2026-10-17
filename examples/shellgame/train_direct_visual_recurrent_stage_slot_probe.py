"""Train recurrent visual tracking with ball-slot supervision only.

This is the strict no-relation-bottleneck ablation.  Each ten-frame swap clip
is encoded into continuous visual tokens and fed directly to a shared
recurrent memory updater.  The model contains no relation classifier, relation
logits, relation probabilities, relation ids, or teacher-forced relation
input.  Ground-truth initial ball slots initialize memory, and cross-entropy
on the ball slot after each of three swaps is the only task loss.
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
from openpi.models import siglip_mem_semantic as _memory_core
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.tasks.shellgame import pi0_mem_semantic_action as _shellgame_model
from openpi.tasks.shellgame import semantic_memory as _semantic
from openpi.training import optimizer as _optimizer
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as _recipe
from scripts.mem import train_semantic_memory as _trainer


class DirectVisualSegmentEncoder(nn.Module):
    """Preserve the clip token grid while projecting it to memory width."""

    segment_size: int = _semantic.SWAP_SEGMENT_SIZE
    spatial_tokens: int = _semantic.SPATIAL_TOKENS
    input_width: int = 1152
    encoder_width: int = 256
    output_width: int = 64
    depth: int = 2
    num_heads: int = 8
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, segment_tokens, *, train: bool = False):
        expected = (self.segment_size, self.spatial_tokens, self.input_width)
        if segment_tokens.ndim != 4 or segment_tokens.shape[1:] != expected:
            raise ValueError(f"Expected [B,{expected}], got {segment_tokens.shape}")

        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(segment_tokens)
        x = nn.Dense(self.encoder_width, name="input_projection", dtype=self.dtype_mm)(x)
        temporal_position = self.param(
            "relative_temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.segment_size, 1, self.encoder_width),
            x.dtype,
        )
        x = x + temporal_position
        for block_index in range(self.depth):
            x = _memory_core.FactorizedSpaceTimeBlock(
                name=f"block_{block_index}",
                width=self.encoder_width,
                num_heads=self.num_heads,
                dropout=0.0,
                dtype_mm=self.dtype_mm,
            )(x, train=train)
        x = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(x)
        return nn.Dense(self.output_width, name="memory_projection", dtype=jnp.float32)(x.astype(jnp.float32))


class ThreeSwapDirectVisualMemoryTracker(nn.Module):
    """Update compact memory directly from three continuous visual clips."""

    num_frames: int = _semantic.HISTORY_FRAMES
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
    def __call__(self, patch_tokens, initial_slots, *, train: bool = False):
        batch, frames, tokens, width = patch_tokens.shape
        expected = (self.num_frames, 256, self.input_width)
        if (frames, tokens, width) != expected:
            raise ValueError(f"Expected [B,{expected}], got {patch_tokens.shape}")
        if initial_slots.shape != (batch,):
            raise ValueError(f"Expected initial slots [B], got {initial_slots.shape}")

        pooled = _memory_core.pool_fixed_grid(patch_tokens, pool_factor=2)
        if pooled.shape[2] != _semantic.SPATIAL_TOKENS:
            raise ValueError(f"Expected {_semantic.SPATIAL_TOKENS} pooled tokens, got {pooled.shape[2]}")
        clips = jnp.stack(
            [pooled[:, start:end] for start, end in _semantic.SWAP_SLICES],
            axis=1,
        ).reshape(
            batch * len(_semantic.SWAP_SLICES),
            _semantic.SWAP_SEGMENT_SIZE,
            _semantic.SPATIAL_TOKENS,
            self.input_width,
        )
        evidence = DirectVisualSegmentEncoder(
            name="direct_visual_segment_encoder",
            segment_size=_semantic.SWAP_SEGMENT_SIZE,
            spatial_tokens=_semantic.SPATIAL_TOKENS,
            input_width=self.input_width,
            encoder_width=self.encoder_width,
            output_width=self.memory_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            dtype_mm=self.dtype_mm,
        )(clips, train=train)
        evidence = evidence.reshape(
            batch,
            len(_semantic.SWAP_SLICES),
            _semantic.SWAP_SEGMENT_SIZE * _semantic.SPATIAL_TOKENS,
            self.memory_width,
        )

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.memory_width),
            jnp.float32,
        )
        memory = jnp.tile(base_memory, (batch, 1, 1))
        initial_code = jax.nn.one_hot(initial_slots, _semantic.NUM_CUPS, dtype=jnp.float32)
        memory = memory.at[:, 0, : _semantic.NUM_CUPS].add(initial_code)

        _, stage_memories = _memory_core.RecurrentMemoryUpdater(
            name="shared_visual_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            dtype_mm="float32",
        )(memory, evidence)
        adapter = _memory_core.SingleHistoryReadAdapter(
            name="shared_history_read_adapter",
            memory_width=self.memory_width,
            current_width=self.current_width,
            num_heads=self.adapter_heads,
            residual_scale=self.residual_scale,
        )
        readout = _semantic.SharedMemoryTokenReadout(name="shared_readout", width=self.current_width)
        base_current_tokens = self.param(
            "base_current_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_current_tokens, self.current_width),
            jnp.float32,
        )
        base_current_tokens = jnp.tile(base_current_tokens, (batch, 1, 1))

        stage_logits = []
        for stage_index in range(len(_semantic.SWAP_SLICES)):
            current_tokens = adapter(base_current_tokens, stage_memories[:, stage_index])
            stage_logits.append(readout(current_tokens))
        stage_logits = jnp.stack(stage_logits, axis=1)
        logits_0, logits_1, logits_2 = (
            stage_logits[:, 0],
            stage_logits[:, 1],
            stage_logits[:, 2],
        )
        joint_logits = (logits_0[:, :, None, None] + logits_1[:, None, :, None] + logits_2[:, None, None, :]).reshape(
            batch, _semantic.NUM_CUPS ** len(_semantic.SWAP_SLICES)
        )
        return joint_logits, stage_logits, stage_memories


@dataclasses.dataclass(frozen=True)
class DirectVisualMemoryConfig(_shellgame_model.Pi0MemSemanticActionConfig):
    """ShellGame config whose memory path has no relation interface."""

    def create(self, rng: at.KeyArrayLike) -> Pi0DirectVisualMemory:
        return Pi0DirectVisualMemory(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_direct_tracker(self) -> nnx.filterlib.Filter:
        direct_tracker = nnx_utils.PathRegex(r".*HistoryThreeSwapDirectVisualMemoryTracker.*")
        return nnx.Not(direct_tracker)


class Pi0DirectVisualMemory(_base_model.Pi0MemCompress):
    """Minimal policy shell exposing only direct visual recurrent tracking."""

    def __init__(self, config: DirectVisualMemoryConfig, rngs: nnx.Rngs):
        if config.num_frames != config.history_frames + 1:
            raise ValueError(
                "Direct visual memory requires fixed history plus one current frame: "
                f"num_frames={config.num_frames}, history_frames={config.history_frames}"
            )
        super().__init__(config, rngs)
        self.history_frames = int(config.history_frames)
        self.video_mode = config.video_mode
        self.HistoryThreeSwapDirectVisualMemoryTracker = nnx_bridge.ToNNX(
            ThreeSwapDirectVisualMemoryTracker(
                num_frames=config.history_frames,
                input_width=1152,
                encoder_width=config.encoder_width,
                encoder_depth=config.encoder_depth,
                encoder_heads=config.encoder_heads,
                memory_width=config.semantic_memory_width,
                memory_depth=config.semantic_memory_depth,
                memory_heads=config.semantic_memory_heads,
                adapter_heads=config.diagnostic_adapter_heads,
                num_memory_tokens=config.semantic_memory_tokens,
                num_current_tokens=config.diagnostic_current_tokens,
                current_width=1152,
                residual_scale=config.diagnostic_residual_scale,
                dtype_mm=config.dtype,
            )
        )
        self.HistoryThreeSwapDirectVisualMemoryTracker.lazy_init(
            jnp.zeros((1, config.history_frames, 256, 1152), dtype=jnp.bfloat16),
            jnp.zeros((1,), dtype=jnp.int32),
            train=False,
            rngs=rngs,
        )

    def track_history(
        self,
        observation: _model.Observation,
        *,
        initial_slots,
        train: bool = False,
    ):
        image = observation.images.get("base_rgb")
        if image is None:
            raise ValueError("Direct visual memory requires a 'base_rgb' stream")
        expected_frames = self.history_frames + 1
        if image.ndim != 5 or image.shape[1] != expected_frames:
            raise ValueError(f"Expected base_rgb [B,{expected_frames},H,W,C], got {image.shape}")
        history = image[:, : self.history_frames]
        _, history_encoder_out = self.PaliGemma.img(history, train=False)
        history_patches = history_encoder_out["with_posemb"][:, : self.history_frames]
        if self.video_mode == "shuffle_swaps":
            start = _semantic.SWAP_SLICES[0][0]
            end = _semantic.SWAP_SLICES[-1][1]
            history_patches = history_patches.at[:, start:end].set(jnp.roll(history_patches[:, start:end], 1, axis=0))
        elif self.video_mode == "zero_swaps":
            start = _semantic.SWAP_SLICES[0][0]
            end = _semantic.SWAP_SLICES[-1][1]
            history_patches = history_patches.at[:, start:end].set(0)
        elif self.video_mode != "normal":
            raise ValueError(f"Unknown video_mode={self.video_mode!r}")

        joint_logits, stage_logits, stage_memories = self.HistoryThreeSwapDirectVisualMemoryTracker(
            history_patches,
            initial_slots.astype(jnp.int32),
            train=train,
        )
        return {
            "joint_logits": joint_logits,
            "stage_logits": stage_logits,
            "stage_memories": stage_memories,
        }


@dataclasses.dataclass(frozen=True)
class DirectVisualCheckpointLoader:
    """Restore the frozen base policy and randomize the complete direct tracker."""

    params_path: str

    def load(self, params):
        source = flax.traverse_util.flatten_dict(
            _model.restore_params(self.params_path, restore_type=np.ndarray), sep="/"
        )
        target = flax.traverse_util.flatten_dict(params, sep="/")
        tracker_root = "HistoryThreeSwapDirectVisualMemoryTracker/"
        result = {}
        restored_base = randomized_tracker = 0
        missing_base = []
        for key, reference in target.items():
            if key.startswith(tracker_root):
                result[key] = reference
                randomized_tracker += 1
                continue
            candidate = source.get(key)
            if candidate is None or np.shape(candidate) != np.shape(reference):
                result[key] = reference
                missing_base.append(key)
            else:
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                restored_base += 1
        if missing_base:
            raise ValueError(f"Frozen base restore incomplete: {missing_base[:8]}")
        print(
            "DirectVisualCheckpointLoader: "
            f"restored_base={restored_base}, randomized_tracker={randomized_tracker}, "
            "missing_base=0"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def stage_slot_only_objective(
    config,
    model,
    rng,
    observation,
    label_table,
    *,
    train: bool,
):
    """Use only initial/stage ball positions; never read relation labels."""
    if observation.episode_index is None:
        raise ValueError("Direct stage-slot training requires episode_index")
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode_index]
    initial_labels = labels[:, 0]
    stage_labels = labels[:, -_recipe.NUM_STAGES :]
    processed = _model.preprocess_observation(
        rng,
        observation,
        train=train and config.memory_train_augmentation,
    )
    outputs = model.track_history(
        processed,
        initial_slots=initial_labels,
        train=train,
    )
    stage_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(outputs["stage_logits"].astype(jnp.float32), stage_labels)
    )
    metrics = {
        "loss": stage_loss,
        "stage_memory_loss": stage_loss,
        "stage_memory_accuracy": jnp.mean(jnp.argmax(outputs["stage_logits"], axis=-1) == stage_labels),
        "final_memory_accuracy": jnp.mean(jnp.argmax(outputs["stage_logits"][:, -1], axis=-1) == stage_labels[:, -1]),
        "memory_token_variance": jnp.mean(jnp.var(outputs["stage_memories"].astype(jnp.float32), axis=-2)),
    }
    for stage_index in range(_recipe.NUM_STAGES):
        metrics[f"slot_{stage_index}_accuracy"] = jnp.mean(
            jnp.argmax(outputs["stage_logits"][:, stage_index], axis=-1) == stage_labels[:, stage_index]
        )
    return stage_loss, metrics


def _copy_model_config(source) -> DirectVisualMemoryConfig:
    values = {field.name: getattr(source, field.name) for field in dataclasses.fields(DirectVisualMemoryConfig)}
    return DirectVisualMemoryConfig(**values)


def build_config(args: argparse.Namespace):
    base = _recipe.make_train_config()
    model = _copy_model_config(base.model)
    return dataclasses.replace(
        base,
        name="shellgame_direct_visual_recurrent_stage_slot_probe",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_direct_tracker(),
        weight_loader=DirectVisualCheckpointLoader(args.init_checkpoint),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 1),
            decay_lr=args.peak_lr * 0.1,
        ),
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=args.save_interval,
        keep_period=args.save_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        initial_loss_weight=0.0,
        relation_loss_weight=0.0,
        stage_memory_loss_weight=1.0,
        wandb_enabled=False,
        overwrite=args.overwrite,
    )


def run_self_test() -> None:
    tracker = ThreeSwapDirectVisualMemoryTracker(
        num_frames=_semantic.HISTORY_FRAMES,
        input_width=16,
        encoder_width=16,
        encoder_depth=1,
        encoder_heads=4,
        memory_width=16,
        memory_depth=1,
        memory_heads=4,
        adapter_heads=4,
        num_memory_tokens=8,
        num_current_tokens=8,
        current_width=32,
        dtype_mm="float32",
    )
    patches = jax.random.normal(jax.random.key(1), (2, _semantic.HISTORY_FRAMES, 256, 16))
    initial = jnp.asarray((0, 1), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), patches, initial, train=False)
    flat_params = flax.traverse_util.flatten_dict(variables["params"], sep="/")
    forbidden = [key for key in flat_params if "relation" in key.lower()]
    if forbidden:
        raise AssertionError(f"Relation parameters leaked into direct tracker: {forbidden}")
    _, reference_logits, _ = tracker.apply(variables, patches, initial, train=False)
    causal = []
    for stage_index, (start, end) in enumerate(_semantic.SWAP_SLICES):
        changed = patches.at[:, start:end].set(
            jax.random.normal(jax.random.key(10 + stage_index), patches[:, start:end].shape)
        )
        _, changed_logits, _ = tracker.apply(variables, changed, initial, train=False)
        causal.append(
            bool(
                np.allclose(
                    np.asarray(reference_logits[:, :stage_index]),
                    np.asarray(changed_logits[:, :stage_index]),
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
    _, changed_initial_logits, _ = tracker.apply(variables, patches, (initial + 1) % _semantic.NUM_CUPS, train=False)
    initial_effect = not np.allclose(np.asarray(reference_logits), np.asarray(changed_initial_logits))
    if not all(causal) or not initial_effect:
        raise AssertionError(f"Direct tracker self-test failed: causal={causal}, initial_effect={initial_effect}")
    print(f"Direct tracker self-test passed: causal={causal}, initial_effect={initial_effect}, relation_params=0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument(
        "--init-checkpoint",
        default=(
            "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
            "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/"
            "absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/"
            "5999/params"
        ),
    )
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test_only:
        run_self_test()
        return
    _recipe.compute_objective = stage_slot_only_objective
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
