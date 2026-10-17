"""Train recurrent MEM from full-episode replay with ten non-overlapping clips.

Each replay sample supplies only observation frames 0..59.  A per-sample random
offset in [0, 5] shifts the clip grid; the missing tail is zero-padded rather
than reading frame 60+, because those frames already contain target-directed
robot motion.  The resulting window is split into ten disjoint six-frame
clips.  The current student is unrolled across all ten clips in one JAX graph,
so the final cup loss backpropagates through every preceding memory state.  No
event, phase, relation, or Qwen output is an updater input.

Two controlled initialization modes are supported:

* ``warm`` maps the validated Qwen-distilled 10-frame tracker into this
  six-frame tracker (the temporal position embedding is sliced to six steps).
* ``scratch`` restores the identical Pi base but leaves the entire replay
  tracker randomly initialized.
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

from examples.shellgame import train_direct_visual_recurrent_stage_slot_probe as _direct
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

DEFAULT_WARM_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "shellgame_qwen_distilled_direct_visual_recurrent_memory_probe/"
    "qwen_distilled_direct_visual_memory250_260825/999/params"
)
REPLAY_FRAMES = 60
WINDOW_FRAMES = 60
CLIP_SIZE = 6
NUM_CLIPS = WINDOW_FRAMES // CLIP_SIZE
MAX_OFFSET = CLIP_SIZE - 1
SWAP_END_FRAMES = (29, 39, 49)


class ReplayRecurrentMemoryUpdater(nn.Module):
    """Shared transition with an optional carry-biased learned gate."""

    width: int = 64
    depth: int = 2
    num_heads: int = 4
    dtype_mm: str = "float32"
    detach_between_clips: bool = False
    use_carry_gate: bool = False
    carry_gate_bias: float = -2.0

    @nn.compact
    def __call__(
        self,
        memory,
        evidence_steps,
        *,
        gate_multiplier=None,
        gate_override=None,
    ):
        expected_gate_shape = evidence_steps.shape[:2]
        if gate_multiplier is not None and gate_multiplier.shape != expected_gate_shape:
            raise ValueError(f"Expected gate_multiplier {expected_gate_shape}, got {gate_multiplier.shape}")
        if gate_override is not None and gate_override.shape != expected_gate_shape:
            raise ValueError(f"Expected gate_override {expected_gate_shape}, got {gate_override.shape}")
        blocks = tuple(
            _memory_core.MemoryUpdateBlock(
                name=f"update_block_{index}",
                width=self.width,
                num_heads=self.num_heads,
                dtype_mm=self.dtype_mm,
            )
            for index in range(self.depth)
        )
        output_norm = nn.LayerNorm(name="state_output_ln", dtype=self.dtype_mm)
        if self.use_carry_gate:
            gate_memory_norm = nn.LayerNorm(name="gate_memory_ln", dtype=self.dtype_mm)
            gate_evidence_norm = nn.LayerNorm(name="gate_evidence_ln", dtype=self.dtype_mm)
            gate_hidden_layer = nn.Dense(self.width, name="gate_hidden", dtype=self.dtype_mm)
            gate_output_layer = nn.Dense(
                1,
                name="gate_out",
                dtype=self.dtype_mm,
                kernel_init=nn.initializers.zeros_init(),
                bias_init=nn.initializers.constant(self.carry_gate_bias),
            )
        states = []
        gates = []
        for step in range(evidence_steps.shape[1]):
            if step and self.detach_between_clips:
                memory = jax.lax.stop_gradient(memory)
            candidate = memory
            for block in blocks:
                candidate = block(candidate, evidence_steps[:, step])
            candidate = output_norm(candidate)
            if self.use_carry_gate:
                memory_summary = jnp.mean(
                    gate_memory_norm(memory),
                    axis=1,
                )
                evidence_summary = jnp.mean(
                    gate_evidence_norm(evidence_steps[:, step]),
                    axis=1,
                )
                gate_hidden = nn.gelu(gate_hidden_layer(jnp.concatenate([memory_summary, evidence_summary], axis=-1)))
                gate = jax.nn.sigmoid(gate_output_layer(gate_hidden))
            else:
                gate = jnp.ones((memory.shape[0], 1), dtype=memory.dtype)
            if gate_multiplier is not None:
                gate = gate * gate_multiplier[:, step, None].astype(gate.dtype)
            if gate_override is not None:
                gate = gate_override[:, step, None].astype(gate.dtype)
            memory = memory + gate[:, :, None] * (candidate - memory)
            states.append(memory)
            gates.append(gate[:, 0])
        return memory, jnp.stack(states, axis=1), jnp.stack(gates, axis=1)


class ReplayUnrolledVisualMemoryTracker(nn.Module):
    """Encode 10x6 clips and recurrently update persistent compact memory."""

    replay_frames: int = REPLAY_FRAMES
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
    detach_between_clips: bool = False
    clip_order: str = "normal"
    use_carry_gate: bool = False
    carry_gate_bias: float = -2.0

    @nn.compact
    def __call__(
        self,
        patch_tokens,
        initial_slots,
        offsets,
        *,
        train: bool = False,
        gate_multiplier=None,
        gate_override=None,
    ):
        batch, frames, tokens, width = patch_tokens.shape
        expected = (self.replay_frames, 256, self.input_width)
        if (frames, tokens, width) != expected:
            raise ValueError(f"Expected [B,{expected}], got {patch_tokens.shape}")
        if initial_slots.shape != (batch,) or offsets.shape != (batch,):
            raise ValueError(f"Expected initial_slots/offsets [B], got {initial_slots.shape}/{offsets.shape}")

        # Never use post-observation frames.  Frame 60 begins robot_approach
        # and leaks the target cup through expert arm motion.  Zero padding is
        # deliberately applied in patch space so every real frame appears at
        # most once, while offsets still move swaps across clip boundaries.
        padded = jnp.concatenate(
            [
                patch_tokens,
                jnp.zeros((batch, MAX_OFFSET, tokens, width), dtype=patch_tokens.dtype),
            ],
            axis=1,
        )

        def select_window(sample, offset):
            return jax.lax.dynamic_slice_in_dim(sample, offset, WINDOW_FRAMES, axis=0)

        windows = jax.vmap(select_window)(padded, offsets)
        pooled = _memory_core.pool_fixed_grid(windows, pool_factor=2)
        clips = pooled.reshape(batch, NUM_CLIPS, CLIP_SIZE, _semantic.SPATIAL_TOKENS, self.input_width)
        if self.clip_order == "reverse":
            clips = clips[:, ::-1]
        elif self.clip_order == "shuffle_batch":
            clips = jnp.roll(clips, shift=1, axis=0)
        elif self.clip_order != "normal":
            raise ValueError(f"Unknown clip_order={self.clip_order!r}")

        evidence = _direct.DirectVisualSegmentEncoder(
            name="direct_visual_segment_encoder",
            segment_size=CLIP_SIZE,
            spatial_tokens=_semantic.SPATIAL_TOKENS,
            input_width=self.input_width,
            encoder_width=self.encoder_width,
            output_width=self.memory_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            dtype_mm=self.dtype_mm,
        )(
            clips.reshape(
                batch * NUM_CLIPS,
                CLIP_SIZE,
                _semantic.SPATIAL_TOKENS,
                self.input_width,
            ),
            train=train,
        ).reshape(
            batch,
            NUM_CLIPS,
            CLIP_SIZE * _semantic.SPATIAL_TOKENS,
            self.memory_width,
        )

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.memory_width),
            jnp.float32,
        )
        initial_code = jax.nn.one_hot(initial_slots, _semantic.NUM_CUPS, dtype=jnp.float32)
        memory = jnp.tile(base_memory, (batch, 1, 1))
        memory = memory.at[:, 0, : _semantic.NUM_CUPS].add(initial_code)
        _, memories, gates = ReplayRecurrentMemoryUpdater(
            name="shared_visual_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            dtype_mm="float32",
            detach_between_clips=self.detach_between_clips,
            use_carry_gate=self.use_carry_gate,
            carry_gate_bias=self.carry_gate_bias,
        )(
            memory,
            evidence,
            gate_multiplier=gate_multiplier,
            gate_override=gate_override,
        )

        adapter = _memory_core.SingleHistoryReadAdapter(
            name="shared_history_read_adapter",
            memory_width=self.memory_width,
            current_width=self.current_width,
            num_heads=self.adapter_heads,
            residual_scale=self.residual_scale,
        )
        readout = _semantic.SharedMemoryTokenReadout(name="shared_readout", width=self.current_width)
        current = self.param(
            "base_current_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_current_tokens, self.current_width),
            jnp.float32,
        )
        current = jnp.tile(current, (batch, 1, 1))
        logits = jnp.stack(
            [readout(adapter(current, memories[:, step])) for step in range(NUM_CLIPS)],
            axis=1,
        )
        return {
            "clip_logits": logits,
            "memories": memories,
            "evidence": evidence,
            "offsets": offsets,
            "gates": gates,
        }


@dataclasses.dataclass(frozen=True)
class ReplayUnrolledMemoryConfig(_shellgame_model.Pi0MemSemanticActionConfig):
    final_slot_weight: float = 1.0
    intermediate_slot_weight: float = 0.25
    transition_slot_weight: float = 0.0
    hold_slot_weight: float = 0.0
    detach_between_clips: bool = False
    clip_order: str = "normal"
    use_carry_gate: bool = False
    carry_gate_bias: float = -2.0

    def create(self, rng: at.KeyArrayLike) -> Pi0ReplayUnrolledMemory:
        return Pi0ReplayUnrolledMemory(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_replay_tracker(self) -> nnx.filterlib.Filter:
        tracker = nnx_utils.PathRegex(r".*HistoryReplayUnrolledVisualMemoryTracker.*")
        return nnx.Not(tracker)


class Pi0ReplayUnrolledMemory(_base_model.Pi0MemCompress):
    def __init__(self, config: ReplayUnrolledMemoryConfig, rngs: nnx.Rngs):
        if config.history_frames != REPLAY_FRAMES or config.num_frames != REPLAY_FRAMES + 1:
            raise ValueError(f"Replay probe expects history_frames={REPLAY_FRAMES}, num_frames={REPLAY_FRAMES + 1}")
        super().__init__(config, rngs)
        self.history_frames = int(config.history_frames)
        self.HistoryReplayUnrolledVisualMemoryTracker = nnx_bridge.ToNNX(
            ReplayUnrolledVisualMemoryTracker(
                replay_frames=config.history_frames,
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
                detach_between_clips=config.detach_between_clips,
                clip_order=config.clip_order,
                use_carry_gate=config.use_carry_gate,
                carry_gate_bias=config.carry_gate_bias,
            )
        )
        self.HistoryReplayUnrolledVisualMemoryTracker.lazy_init(
            jnp.zeros((1, REPLAY_FRAMES, 256, 1152), dtype=jnp.bfloat16),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            train=False,
            rngs=rngs,
        )

    def track_history(self, observation, *, initial_slots, offsets, train: bool = False):
        image = observation.images.get("base_rgb")
        if image is None or image.ndim != 5 or image.shape[1] != REPLAY_FRAMES + 1:
            raise ValueError(
                f"Expected base_rgb [B,{REPLAY_FRAMES + 1},H,W,C], got {None if image is None else image.shape}"
            )
        history = image[:, :REPLAY_FRAMES]
        _, encoder_out = self.PaliGemma.img(history, train=False)
        patches = encoder_out["with_posemb"][:, :REPLAY_FRAMES]
        return self.HistoryReplayUnrolledVisualMemoryTracker(
            patches,
            initial_slots.astype(jnp.int32),
            offsets.astype(jnp.int32),
            train=train,
        )


@dataclasses.dataclass(frozen=True)
class ReplayCheckpointLoader:
    params_path: str
    init_mode: str

    def load(self, params):
        source = flax.traverse_util.flatten_dict(
            _model.restore_params(self.params_path, restore_type=np.ndarray), sep="/"
        )
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source_roots = (
            "HistoryReplayUnrolledVisualMemoryTracker/",
            "HistoryQwenDistilledDirectVisualMemoryTracker/",
        )
        target_root = "HistoryReplayUnrolledVisualMemoryTracker/"
        result = {}
        counts = {
            "base": 0,
            "warm_tracker": 0,
            "gate_init": 0,
            "scratch_tracker": 0,
            "shape_fallback": 0,
        }
        fallback = []

        for key, reference in target.items():
            if key.startswith(target_root):
                if self.init_mode == "scratch":
                    result[key] = reference
                    counts["scratch_tracker"] += 1
                    continue
                relative = key.removeprefix(target_root)
                candidate = next(
                    (source[root + relative] for root in source_roots if root + relative in source),
                    None,
                )
                if (
                    candidate is not None
                    and relative.endswith("relative_temporal_pos_embedding")
                    and np.shape(candidate)[1] >= CLIP_SIZE
                ):
                    candidate = np.asarray(candidate)[:, :CLIP_SIZE]
                kind = "warm_tracker"
            else:
                candidate = source.get(key)
                kind = "base"

            if candidate is None or np.shape(candidate) != np.shape(reference):
                result[key] = reference
                counts["shape_fallback"] += 1
                fallback.append(key)
            else:
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                counts[kind] += 1

        gate_fallback = [
            key for key in fallback if key.startswith(target_root) and "/shared_visual_memory_updater/gate_" in key
        ]
        counts["gate_init"] = len(gate_fallback)
        tracker_fallback = [key for key in fallback if key.startswith(target_root) and key not in gate_fallback]
        if self.init_mode == "warm" and tracker_fallback:
            raise ValueError(f"Warm checkpoint mapping incomplete ({len(tracker_fallback)}): {tracker_fallback[:8]}")
        print(
            f"ReplayCheckpointLoader(init_mode={self.init_mode}): "
            + ", ".join(f"{key}={value}" for key, value in counts.items())
        )
        if fallback:
            print(f"ReplayCheckpointLoader shape fallbacks ({len(fallback)}): {fallback[:8]}")
        return flax.traverse_util.unflatten_dict(result, sep="/")


def _masked_mean(values, mask):
    mask = mask.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def replay_unroll_objective(config, model, rng, observation, label_table, *, train: bool):
    if observation.episode_index is None:
        raise ValueError("Replay unroll requires episode_index")
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode_index]
    initial = labels[:, 0]
    stage_slots = labels[:, 1 + _recipe.NUM_STAGES :]
    batch = episode_index.shape[0]
    if train:
        offsets = jax.random.randint(rng, (batch,), 0, MAX_OFFSET + 1, dtype=jnp.int32)
    else:
        offsets = jnp.mod(episode_index, MAX_OFFSET + 1).astype(jnp.int32)

    processed = _model.preprocess_observation(rng, observation, train=train and config.memory_train_augmentation)
    outputs = model.track_history(processed, initial_slots=initial, offsets=offsets, train=train)
    clip_ends = jnp.minimum(
        offsets[:, None] + (jnp.arange(NUM_CLIPS, dtype=jnp.int32)[None, :] + 1) * CLIP_SIZE - 1,
        REPLAY_FRAMES - 1,
    )
    completed = jnp.sum(
        clip_ends[:, :, None] >= jnp.asarray(SWAP_END_FRAMES, dtype=jnp.int32)[None, None, :],
        axis=-1,
    )
    state_slots = jnp.concatenate([initial[:, None], stage_slots], axis=1)
    clip_labels = jnp.take_along_axis(state_slots, completed, axis=1)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(
        outputs["clip_logits"].astype(jnp.float32), clip_labels
    )
    final_loss = jnp.mean(token_losses[:, -1])
    intermediate_loss = jnp.mean(token_losses[:, :-1])
    predictions = jnp.argmax(outputs["clip_logits"], axis=-1)
    correct = predictions == clip_labels
    previous_completed = jnp.concatenate([jnp.zeros((batch, 1), dtype=completed.dtype), completed[:, :-1]], axis=1)
    transition_mask = completed > previous_completed
    auxiliary_mask = jnp.arange(NUM_CLIPS)[None, :] < NUM_CLIPS - 1
    transition_aux_mask = transition_mask & auxiliary_mask
    hold_aux_mask = (~transition_mask) & auxiliary_mask
    transition_loss = _masked_mean(token_losses, transition_aux_mask)
    hold_loss = _masked_mean(token_losses, hold_aux_mask)
    if config.model.transition_slot_weight or config.model.hold_slot_weight:
        loss = (
            config.model.final_slot_weight * final_loss
            + config.model.transition_slot_weight * transition_loss
            + config.model.hold_slot_weight * hold_loss
        )
    else:
        loss = config.model.final_slot_weight * final_loss + config.model.intermediate_slot_weight * intermediate_loss
    partial_swap_mask = jnp.any(
        (clip_ends[:, :, None] >= jnp.asarray((20, 30, 40))[None, None, :])
        & (clip_ends[:, :, None] < jnp.asarray(SWAP_END_FRAMES)[None, None, :]),
        axis=-1,
    )
    metrics = {
        "loss": loss,
        "final_slot_loss": final_loss,
        "intermediate_slot_loss": intermediate_loss,
        "transition_slot_loss": transition_loss,
        "hold_slot_loss": hold_loss,
        "clip_slot_accuracy": jnp.mean(correct),
        "final_memory_accuracy": jnp.mean(correct[:, -1]),
        "transition_endpoint_accuracy": _masked_mean(correct, transition_mask),
        "partial_swap_hold_accuracy": _masked_mean(correct, partial_swap_mask),
        "memory_token_variance": jnp.mean(jnp.var(outputs["memories"].astype(jnp.float32), axis=-2)),
        "memory_step_delta": jnp.mean(
            jnp.square(outputs["memories"][:, 1:].astype(jnp.float32) - outputs["memories"][:, :-1].astype(jnp.float32))
        ),
        "mean_offset": jnp.mean(offsets.astype(jnp.float32)),
        "gate_mean": jnp.mean(outputs["gates"].astype(jnp.float32)),
        "transition_gate_mean": _masked_mean(outputs["gates"], transition_aux_mask),
        "hold_gate_mean": _masked_mean(outputs["gates"], hold_aux_mask),
        "final_gate_mean": jnp.mean(outputs["gates"][:, -1]),
    }
    for offset in range(MAX_OFFSET + 1):
        metrics[f"offset_{offset}_final_accuracy"] = _masked_mean(correct[:, -1], offsets == offset)
    return loss, metrics


def _copy_model_config(source, args):
    field_names = {field.name for field in dataclasses.fields(ReplayUnrolledMemoryConfig)}
    values = {name: getattr(source, name) for name in field_names if hasattr(source, name)}
    values.update(
        num_frames=REPLAY_FRAMES + 1,
        history_frames=REPLAY_FRAMES,
        final_slot_weight=args.final_slot_weight,
        intermediate_slot_weight=args.intermediate_slot_weight,
        transition_slot_weight=args.transition_slot_weight,
        hold_slot_weight=args.hold_slot_weight,
        detach_between_clips=args.detach_between_clips,
        clip_order=args.clip_order,
        use_carry_gate=args.carry_gate,
        carry_gate_bias=args.carry_gate_bias,
    )
    return ReplayUnrolledMemoryConfig(**values)


def build_config(args):
    base = _recipe.make_train_config()
    model = _copy_model_config(base.model, args)
    child = dataclasses.replace(
        base.data.datasets[0],
        num_frames=REPLAY_FRAMES + 1,
        frame_stride=1,
        video_layout="fixed_prefix_current",
        fixed_prefix_frames=REPLAY_FRAMES,
        min_frame_index=REPLAY_FRAMES - 1,
        max_frame_index=REPLAY_FRAMES - 1,
    )
    data = dataclasses.replace(base.data, datasets=[child])
    return dataclasses.replace(
        base,
        name="shellgame_replay_unrolled_clip6_memory_probe",
        exp_name=args.exp_name,
        model=model,
        data=data,
        freeze_filter=model.get_freeze_filter_replay_tracker(),
        weight_loader=ReplayCheckpointLoader(args.warm_checkpoint, args.init_mode),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 1),
            decay_lr=args.decay_lr if args.decay_lr is not None else args.peak_lr * 0.1,
        ),
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=args.save_interval,
        keep_period=args.keep_period if args.keep_period is not None else args.save_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        initial_loss_weight=0.0,
        relation_loss_weight=0.0,
        stage_memory_loss_weight=1.0,
        wandb_enabled=False,
        overwrite=args.overwrite,
        resume=args.resume,
    )


def run_self_test(*, use_carry_gate: bool, carry_gate_bias: float):
    tracker = ReplayUnrolledVisualMemoryTracker(
        replay_frames=REPLAY_FRAMES,
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
        use_carry_gate=use_carry_gate,
        carry_gate_bias=carry_gate_bias,
    )
    patches = jax.random.normal(jax.random.key(1), (2, REPLAY_FRAMES, 256, 16))
    initial = jnp.asarray((0, 1), dtype=jnp.int32)
    offsets = jnp.asarray((0, MAX_OFFSET), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), patches, initial, offsets, train=False)
    outputs = tracker.apply(variables, patches, initial, offsets, train=False)
    expected = (2, NUM_CLIPS, 8, 16)
    if outputs["memories"].shape != expected:
        raise AssertionError(f"Unexpected recurrent states: {outputs['memories'].shape}")
    if outputs["gates"].shape != (2, NUM_CLIPS):
        raise AssertionError(f"Unexpected gate shape: {outputs['gates'].shape}")
    selected_real = [list(range(offset, REPLAY_FRAMES)) for offset in (0, MAX_OFFSET)]
    if any(len(frames) != len(set(frames)) for frames in selected_real):
        raise AssertionError("Replay windows contain duplicated real frames")

    def final_logit(input_patches):
        result = tracker.apply(variables, input_patches, initial, offsets, train=False)
        return jnp.sum(result["clip_logits"][:, -1, 0])

    grads = jax.grad(final_logit)(patches)
    first_clip_norm = jnp.linalg.norm(grads[0, :CLIP_SIZE].astype(jnp.float32))
    if not bool(first_clip_norm > 0):
        raise AssertionError("Final loss has no gradient path to the first clip")
    print(
        "Replay-unroll self-test passed: "
        f"states={expected}, offsets=0..{MAX_OFFSET}, first_clip_grad={float(first_clip_norm):.6g}, "
        f"gate_mean={float(jnp.mean(outputs['gates'])):.6g}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-mode", choices=("warm", "scratch"), required=True)
    parser.add_argument("--warm-checkpoint", default=DEFAULT_WARM_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--decay-lr", type=float)
    parser.add_argument("--final-slot-weight", type=float, default=1.0)
    parser.add_argument("--intermediate-slot-weight", type=float, default=0.25)
    parser.add_argument("--transition-slot-weight", type=float, default=0.0)
    parser.add_argument("--hold-slot-weight", type=float, default=0.0)
    parser.add_argument("--carry-gate", action="store_true")
    parser.add_argument("--carry-gate-bias", type=float, default=-2.0)
    parser.add_argument("--detach-between-clips", action="store_true")
    parser.add_argument("--clip-order", choices=("normal", "reverse", "shuffle_batch"), default="normal")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--keep-period", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test_only:
        run_self_test(
            use_carry_gate=args.carry_gate,
            carry_gate_bias=args.carry_gate_bias,
        )
        return
    _recipe.compute_objective = replay_unroll_objective
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
