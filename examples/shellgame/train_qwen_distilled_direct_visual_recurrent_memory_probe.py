"""Distill Qwen ShellGame knowledge into a direct-visual recurrent MEM.

The student inference contract is deliberately strict::

    short continuous image clip -> shallow SigLIP patch tokens
      -> fixed 2x2 spatial pooling -> lightweight factorized encoder
    (previous memory embedding, compressed visual tokens)
      -> shared recurrent updater -> next memory embedding

No relation id, relation probability, relation logit, or Qwen output is an
input to the student updater.  During Stage-1 training only, Qwen's validated
reveal/swap predictions select states from the proven symbolic teacher.  The
student receives dense supervision on every resulting memory embedding.

For the simulator experiment, clean metadata is used as the cached teacher
prediction because Qwen3-VL step375 reached 100% reveal/swap and end-to-end
sequence accuracy on held-out episodes.  This keeps Qwen out of the JAX loop
and is the same offline-teacher-forcing contract used by the action cache.
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

DEFAULT_TEACHER_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "shellgame_stage_slot_only_relation_recurrent_probe/"
    "stage_slot_only_random_relation_frozen_memory_1k_260821/500/params"
)


def _memory_distillation_loss(student_memory, teacher_memory, *, shuffle_targets: bool):
    """Token-aligned cosine plus scale-sensitive MSE memory supervision."""
    student = jnp.asarray(student_memory, dtype=jnp.float32)
    teacher = jax.lax.stop_gradient(jnp.asarray(teacher_memory, dtype=jnp.float32))
    if shuffle_targets:
        teacher = jnp.roll(teacher, shift=1, axis=0)
    student_unit = student / jnp.maximum(jnp.linalg.norm(student, axis=-1, keepdims=True), 1e-6)
    teacher_unit = teacher / jnp.maximum(jnp.linalg.norm(teacher, axis=-1, keepdims=True), 1e-6)
    cosine = jnp.mean(1.0 - jnp.sum(student_unit * teacher_unit, axis=-1))
    mse = jnp.mean(jnp.square(student - teacher))
    return cosine + 0.1 * mse, cosine, mse


class QwenDistilledDirectVisualMemoryTracker(nn.Module):
    """Map compressed visual clips and previous memory to the next memory."""

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
    def __call__(
        self,
        patch_tokens,
        initial_slots,
        *,
        teacher_relation_ids=None,
        train: bool = False,
    ):
        batch, frames, tokens, width = patch_tokens.shape
        expected = (self.num_frames, 256, self.input_width)
        if (frames, tokens, width) != expected:
            raise ValueError(f"Expected [B,{expected}], got {patch_tokens.shape}")
        if initial_slots.shape != (batch,):
            raise ValueError(f"Expected initial_slots [B], got {initial_slots.shape}")
        if teacher_relation_ids is not None and teacher_relation_ids.shape != (
            batch,
            len(_semantic.SWAP_SLICES),
        ):
            raise ValueError(
                f"Expected teacher_relation_ids [B,{len(_semantic.SWAP_SLICES)}], got {teacher_relation_ids.shape}"
            )

        # This is the previously validated low-compute image compression path:
        # patch embedding only, 16x16 -> 8x8 pooling, width-256 depth-2
        # factorized temporal/spatial encoding, then a width-64 projection.
        pooled = _memory_core.pool_fixed_grid(patch_tokens, pool_factor=2)
        clips = jnp.stack(
            [pooled[:, start:end] for start, end in _semantic.SWAP_SLICES],
            axis=1,
        ).reshape(
            batch * len(_semantic.SWAP_SLICES),
            _semantic.SWAP_SEGMENT_SIZE,
            _semantic.SPATIAL_TOKENS,
            self.input_width,
        )
        visual_evidence = _direct.DirectVisualSegmentEncoder(
            name="direct_visual_segment_encoder",
            segment_size=_semantic.SWAP_SEGMENT_SIZE,
            spatial_tokens=_semantic.SPATIAL_TOKENS,
            input_width=self.input_width,
            encoder_width=self.encoder_width,
            output_width=self.memory_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            dtype_mm=self.dtype_mm,
        )(clips, train=train).reshape(
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
        initial_code = jax.nn.one_hot(initial_slots, _semantic.NUM_CUPS, dtype=jnp.float32)
        student_initial = jnp.tile(base_memory, (batch, 1, 1))
        student_initial = student_initial.at[:, 0, : _semantic.NUM_CUPS].add(initial_code)
        _, student_memories = _memory_core.RecurrentMemoryUpdater(
            name="shared_visual_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            dtype_mm="float32",
        )(student_initial, visual_evidence)

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
        current_tokens = jnp.tile(base_current_tokens, (batch, 1, 1))
        stage_logits = jnp.stack(
            [
                readout(adapter(current_tokens, student_memories[:, stage]))
                for stage in range(len(_semantic.SWAP_SLICES))
            ],
            axis=1,
        )

        teacher_memories = None
        if teacher_relation_ids is not None:
            # Teacher-only path.  These codes never enter the student visual
            # encoder or student recurrent updater.
            relation_codes = jax.nn.one_hot(
                teacher_relation_ids.astype(jnp.int32),
                _semantic.NUM_CUPS,
                dtype=jnp.float32,
            )
            teacher_segments = jnp.zeros(
                (
                    batch,
                    len(_semantic.SWAP_SLICES),
                    _semantic.SWAP_SEGMENT_SIZE,
                    _semantic.SPATIAL_TOKENS,
                    self.memory_width,
                ),
                dtype=jnp.float32,
            )
            teacher_segments = teacher_segments.at[..., : _semantic.NUM_CUPS].add(relation_codes[:, :, None, None, :])
            teacher_base_memory = self.param(
                "teacher_base_memory",
                nn.initializers.normal(stddev=0.02),
                (1, self.num_memory_tokens, self.memory_width),
                jnp.float32,
            )
            teacher_initial = jnp.tile(teacher_base_memory, (batch, 1, 1))
            teacher_initial = teacher_initial.at[:, 0, : _semantic.NUM_CUPS].add(initial_code)
            _, teacher_memories = _semantic.ShellGameSwapRecurrentMemoryUpdater(
                name="teacher_swap_memory_updater",
                width=self.memory_width,
                depth=self.memory_depth,
                num_heads=self.memory_heads,
                segment_size=_semantic.SWAP_SEGMENT_SIZE,
                dtype_mm="float32",
            )(teacher_initial, teacher_segments)
            teacher_memories = jax.lax.stop_gradient(teacher_memories)

        return {
            "stage_logits": stage_logits,
            "student_memories": student_memories,
            "teacher_memories": teacher_memories,
            "visual_evidence": visual_evidence,
        }


@dataclasses.dataclass(frozen=True)
class QwenDistilledDirectVisualConfig(_shellgame_model.Pi0MemSemanticActionConfig):
    """Configuration for the isolated Stage-1 distillation probe."""

    memory_distill_weight: float = 1.0
    stage_slot_weight: float = 0.25
    shuffle_teacher_targets: bool = False

    def create(self, rng: at.KeyArrayLike) -> Pi0QwenDistilledDirectVisualMemory:
        return Pi0QwenDistilledDirectVisualMemory(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_qwen_distill(self) -> nnx.filterlib.Filter:
        visual_encoder = nnx_utils.PathRegex(
            r".*HistoryQwenDistilledDirectVisualMemoryTracker/"
            r"direct_visual_segment_encoder.*"
        )
        return nnx.Not(visual_encoder)


class Pi0QwenDistilledDirectVisualMemory(_base_model.Pi0MemCompress):
    """Policy shell exposing the student and training-only teacher state."""

    def __init__(self, config: QwenDistilledDirectVisualConfig, rngs: nnx.Rngs):
        if config.num_frames != config.history_frames + 1:
            raise ValueError("Qwen-distilled visual memory expects 60 history frames plus one current frame")
        super().__init__(config, rngs)
        self.history_frames = int(config.history_frames)
        self.video_mode = config.video_mode
        self.HistoryQwenDistilledDirectVisualMemoryTracker = nnx_bridge.ToNNX(
            QwenDistilledDirectVisualMemoryTracker(
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
        self.HistoryQwenDistilledDirectVisualMemoryTracker.lazy_init(
            jnp.zeros((1, config.history_frames, 256, 1152), dtype=jnp.bfloat16),
            jnp.zeros((1,), dtype=jnp.int32),
            teacher_relation_ids=jnp.zeros((1, len(_semantic.SWAP_SLICES)), dtype=jnp.int32),
            train=False,
            rngs=rngs,
        )

    def track_history(self, observation, *, initial_slots, teacher_relation_ids=None, train: bool = False):
        image = observation.images.get("base_rgb")
        if image is None or image.ndim != 5 or image.shape[1] != self.history_frames + 1:
            raise ValueError(
                f"Expected base_rgb [B,{self.history_frames + 1},H,W,C], got {None if image is None else image.shape}"
            )
        history = image[:, : self.history_frames]
        _, encoder_out = self.PaliGemma.img(history, train=False)
        patches = encoder_out["with_posemb"][:, : self.history_frames]
        if self.video_mode == "shuffle_swaps":
            start, end = _semantic.SWAP_SLICES[0][0], _semantic.SWAP_SLICES[-1][1]
            patches = patches.at[:, start:end].set(jnp.roll(patches[:, start:end], 1, axis=0))
        elif self.video_mode == "zero_swaps":
            start, end = _semantic.SWAP_SLICES[0][0], _semantic.SWAP_SLICES[-1][1]
            patches = patches.at[:, start:end].set(0)
        elif self.video_mode != "normal":
            raise ValueError(f"Unknown video_mode={self.video_mode!r}")
        return self.HistoryQwenDistilledDirectVisualMemoryTracker(
            patches,
            initial_slots.astype(jnp.int32),
            teacher_relation_ids=teacher_relation_ids,
            train=train,
        )


@dataclasses.dataclass(frozen=True)
class QwenDistilledVisualLoader:
    """Copy the proven state basis into both teacher and student updaters."""

    params_path: str

    def load(self, params):
        source = flax.traverse_util.flatten_dict(
            _model.restore_params(self.params_path, restore_type=np.ndarray), sep="/"
        )
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source_root = "HistoryThreeSwapVisualRelationMemoryTracker/"
        target_root = "HistoryQwenDistilledDirectVisualMemoryTracker/"
        encoder_root = target_root + "direct_visual_segment_encoder/"
        result = {}
        counts = {"base": 0, "student_memory": 0, "teacher_memory": 0, "random_visual": 0}
        missing = []

        for key, reference in target.items():
            if key.startswith(encoder_root):
                result[key] = reference
                counts["random_visual"] += 1
                continue

            candidate = None
            kind = "base"
            if key.startswith(target_root):
                relative = key.removeprefix(target_root)
                if relative == "teacher_base_memory":
                    relative = "base_memory"
                    kind = "teacher_memory"
                elif relative.startswith("teacher_swap_memory_updater/"):
                    relative = "shared_swap_memory_updater/" + relative.removeprefix("teacher_swap_memory_updater/")
                    kind = "teacher_memory"
                elif relative.startswith("shared_visual_memory_updater/"):
                    relative = "shared_swap_memory_updater/" + relative.removeprefix("shared_visual_memory_updater/")
                    kind = "student_memory"
                else:
                    kind = "student_memory"
                candidate = source.get(source_root + relative)
            else:
                candidate = source.get(key)

            if candidate is None or np.shape(candidate) != np.shape(reference):
                result[key] = reference
                missing.append(key)
            else:
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                counts[kind] += 1

        if missing:
            raise ValueError(f"Qwen-distilled checkpoint mapping incomplete ({len(missing)}): {missing[:8]}")
        print("QwenDistilledVisualLoader: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        return flax.traverse_util.unflatten_dict(result, sep="/")


def qwen_memory_distillation_objective(config, model, rng, observation, label_table, *, train: bool):
    """Align visual recurrent states to Qwen-selected teacher states."""
    if observation.episode_index is None:
        raise ValueError("Qwen memory distillation requires episode_index")
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode_index]
    initial_labels = labels[:, 0]
    teacher_relations = labels[:, 1 : 1 + _recipe.NUM_STAGES]
    stage_labels = labels[:, 1 + _recipe.NUM_STAGES :]
    processed = _model.preprocess_observation(
        rng,
        observation,
        train=train and config.memory_train_augmentation,
    )
    outputs = model.track_history(
        processed,
        initial_slots=initial_labels,
        teacher_relation_ids=teacher_relations,
        train=train,
    )
    memory_loss, cosine_loss, mse_loss = _memory_distillation_loss(
        outputs["student_memories"],
        outputs["teacher_memories"],
        shuffle_targets=config.model.shuffle_teacher_targets,
    )
    stage_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(outputs["stage_logits"].astype(jnp.float32), stage_labels)
    )
    loss = config.model.memory_distill_weight * memory_loss + config.model.stage_slot_weight * stage_loss
    predictions = jnp.argmax(outputs["stage_logits"], axis=-1)
    metrics = {
        "loss": loss,
        "memory_distill_loss": memory_loss,
        "memory_cosine_loss": cosine_loss,
        "memory_mse_loss": mse_loss,
        "stage_memory_loss": stage_loss,
        "stage_memory_accuracy": jnp.mean(predictions == stage_labels),
        "final_memory_accuracy": jnp.mean(predictions[:, -1] == stage_labels[:, -1]),
        "student_memory_token_variance": jnp.mean(jnp.var(outputs["student_memories"].astype(jnp.float32), axis=-2)),
        "teacher_memory_token_variance": jnp.mean(jnp.var(outputs["teacher_memories"].astype(jnp.float32), axis=-2)),
    }
    for stage in range(_recipe.NUM_STAGES):
        metrics[f"slot_{stage}_accuracy"] = jnp.mean(predictions[:, stage] == stage_labels[:, stage])
    return loss, metrics


def _copy_model_config(source, args) -> QwenDistilledDirectVisualConfig:
    field_names = {field.name for field in dataclasses.fields(QwenDistilledDirectVisualConfig)}
    values = {name: getattr(source, name) for name in field_names if hasattr(source, name)}
    values.update(
        memory_distill_weight=args.memory_distill_weight,
        stage_slot_weight=args.stage_slot_weight,
        shuffle_teacher_targets=args.shuffle_teacher_targets,
    )
    return QwenDistilledDirectVisualConfig(**values)


def build_config(args):
    base = _recipe.make_train_config()
    model = _copy_model_config(base.model, args)
    return dataclasses.replace(
        base,
        name="shellgame_qwen_distilled_direct_visual_recurrent_memory_probe",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_qwen_distill(),
        weight_loader=QwenDistilledVisualLoader(args.teacher_checkpoint),
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


def run_self_test():
    tracker = QwenDistilledDirectVisualMemoryTracker(
        num_frames=60,
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
    patches = jax.random.normal(jax.random.key(1), (2, 60, 256, 16))
    initial = jnp.asarray((0, 1), dtype=jnp.int32)
    relations = jnp.asarray(((0, 1, 2), (2, 1, 0)), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), patches, initial, teacher_relation_ids=relations, train=False)
    outputs = tracker.apply(variables, patches, initial, teacher_relation_ids=relations, train=False)
    expected = (2, 3, 8, 16)
    if outputs["student_memories"].shape != expected or outputs["teacher_memories"].shape != expected:
        raise AssertionError(
            f"Unexpected memories: {outputs['student_memories'].shape}, {outputs['teacher_memories'].shape}"
        )
    flat = flax.traverse_util.flatten_dict(variables["params"], sep="/")
    student_forbidden = [
        key
        for key in flat
        if key.startswith(("direct_visual_segment_encoder/", "shared_visual_memory_updater/"))
        and "relation" in key.lower()
    ]
    if student_forbidden:
        raise AssertionError(f"Relation interface leaked into student: {student_forbidden}")
    print(
        "Qwen direct-visual distillation self-test passed: "
        f"student={expected}, teacher={expected}, student_relation_params=0"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--teacher-checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--decay-lr", type=float)
    parser.add_argument("--memory-distill-weight", type=float, default=1.0)
    parser.add_argument("--stage-slot-weight", type=float, default=0.25)
    parser.add_argument("--shuffle-teacher-targets", action="store_true")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--keep-period", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test_only:
        run_self_test()
        return
    _recipe.compute_objective = qwen_memory_distillation_objective
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
