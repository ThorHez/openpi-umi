"""Train a boundary-aware sliding-window ShellGame recurrent memory probe.

The fixed 20:30, 30:40, and 40:50 clips are replaced by length-10 windows.
During training, complete-event windows are jittered by +/-2 frames and hard
windows spanning two swap phases are explicitly supervised as ``no_event``.
The task retains exactly three top-level losses:

* frame-0 initial-cup loss;
* window relation loss (event gate plus three-way swap identity); and
* recurrent stage-memory loss after each accepted event.

The proven recurrent updater/readout is restored and frozen.  This isolates
whether a learned window gate can protect memory from partial/mixed windows.
Full validation scans all 51 windows and compares aligned, automatic,
automatic-with-cross-windows-masked, and forced-cross-window conditions.
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

WINDOW = _semantic.SWAP_SEGMENT_SIZE
NUM_WINDOWS = _semantic.HISTORY_FRAMES - WINDOW + 1
NUM_STAGES = _recipe.NUM_STAGES
ALIGNED_STARTS = (20, 30, 40)
POSITIVE_OFFSET_RADIUS = 2
MIN_EVENT_SEPARATION = 8

# A length-10 window at these starts contains substantial content from two
# adjacent swap phases.  It must not be allowed to update recurrent memory.
CROSS_BOUNDARY_STARTS = (23, 24, 25, 26, 27, 33, 34, 35, 36, 37)
STRICT_CROSS_STARTS = (24, 25, 26, 34, 35, 36)
STATIC_OR_PARTIAL_STARTS = (0, 5, 10, 13, 14, 15, 16, 17, 43, 44, 45, 46, 47, 48, 49, 50)
CONDITION_NAMES = ("aligned", "automatic", "automatic_no_cross", "forced_cross")
RELATION_CODE_MODE = "probabilities"
ENABLE_CAUSAL_EVAL_SELECTIONS = False

DEFAULT_ANCHORED_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "shellgame_stage_slot_only_relation_recurrent_probe/"
    "stage_slot_only_random_relation_frozen_memory_1k_260821/500/params"
)


def _is_in(values: jax.Array, candidates: tuple[int, ...]) -> jax.Array:
    candidate_array = jnp.asarray(candidates, dtype=jnp.int32)
    return jnp.any(values[..., None] == candidate_array, axis=-1)


def _stage_index_for_start(starts: jax.Array) -> jax.Array:
    """Return stage 0/1/2 for complete windows, or -1 for a negative."""
    stage = jnp.full(starts.shape, -1, dtype=jnp.int32)
    for stage_index, aligned in enumerate(ALIGNED_STARTS):
        valid = jnp.abs(starts - aligned) <= POSITIVE_OFFSET_RADIUS
        stage = jnp.where(valid, stage_index, stage)
    return stage


def _temporal_nms(scores: jax.Array, starts: jax.Array) -> jax.Array:
    """Select three high-scoring separated windows and return their positions."""
    if scores.ndim != 2 or starts.shape != scores.shape:
        raise ValueError(f"Expected matching [B,W] scores/starts, got {scores.shape}/{starts.shape}")
    available = jnp.ones_like(scores, dtype=jnp.bool_)
    selected_positions = []
    for _ in range(NUM_STAGES):
        position = jnp.argmax(jnp.where(available, scores, -jnp.inf), axis=1)
        selected_positions.append(position)
        selected_start = jnp.take_along_axis(starts, position[:, None], axis=1)[:, 0]
        available &= jnp.abs(starts - selected_start[:, None]) >= MIN_EVENT_SEPARATION
    positions = jnp.stack(selected_positions, axis=1)
    selected_starts = jnp.take_along_axis(starts, positions, axis=1)
    order = jnp.argsort(selected_starts, axis=1)
    return jnp.take_along_axis(positions, order, axis=1)


def _store_first_three_triggers(trigger_mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Store the first three causal triggers and count all triggers."""
    batch, num_candidates = trigger_mask.shape
    selected = jnp.zeros((batch, NUM_STAGES), dtype=jnp.int32)
    count = jnp.zeros((batch,), dtype=jnp.int32)
    batch_axis = jnp.arange(batch, dtype=jnp.int32)
    for position in range(num_candidates):
        trigger = trigger_mask[:, position]
        slot = jnp.minimum(count, NUM_STAGES - 1)
        should_store = trigger & (count < NUM_STAGES)
        old_value = selected[batch_axis, slot]
        selected = selected.at[batch_axis, slot].set(jnp.where(should_store, position, old_value))
        count += trigger.astype(jnp.int32)
    return selected, count


def _causal_rising_edge_select(event_logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Trigger once per contiguous positive event region without future scores."""
    high = event_logits > 0.0
    previous_high = jnp.pad(high[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    return _store_first_three_triggers(high & ~previous_high)


def _fixed_six_chunk_select(event_logits: jax.Array, starts: jax.Array, offset: int):
    """Select event-positive windows from one non-overlapping six-frame grid."""
    on_grid = (starts >= offset) & (((starts - offset) % 6) == 0)
    return _store_first_three_triggers(on_grid & (event_logits > 0.0))


class SlidingWindowEventRelationClassifier(nn.Module):
    """Predict event completeness and swap identity from one visual window."""

    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    segment_size: int = WINDOW
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, window_tokens, *, train: bool = False):
        semantic = _memory_core.FactorizedSpaceTimeEncoder(
            name="semantic_encoder",
            input_width=self.input_width,
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=self.segment_size,
            spatial_tokens=_semantic.SPATIAL_TOKENS,
            dtype_mm=self.dtype_mm,
        )(window_tokens, train=train)
        semantic = semantic.astype(jnp.float32)
        event_logits = nn.Dense(1, name="event_classifier", dtype=jnp.float32)(semantic)[..., 0]
        relation_logits = nn.Dense(
            _semantic.NUM_CUPS,
            name="relation_classifier",
            dtype=jnp.float32,
        )(semantic)
        return event_logits, relation_logits


class SlidingWindowRelationMemoryTracker(nn.Module):
    """Encode candidate windows and update memory only from three selections."""

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
        window_starts,
        *,
        evaluate_all_windows: bool = False,
        train: bool = False,
    ):
        batch, frames, tokens, width = patch_tokens.shape
        expected = (self.num_frames, 256, self.input_width)
        if (frames, tokens, width) != expected:
            raise ValueError(f"Expected [B,{expected}], got {patch_tokens.shape}")
        if initial_slots.shape != (batch,):
            raise ValueError(f"Expected initial_slots [B], got {initial_slots.shape}")
        if window_starts.ndim != 2 or window_starts.shape[0] != batch:
            raise ValueError(f"Expected window_starts [B,W], got {window_starts.shape}")

        pooled = _memory_core.pool_fixed_grid(patch_tokens, pool_factor=2)
        frame_indices = window_starts[..., None] + jnp.arange(WINDOW, dtype=jnp.int32)
        batch_indices = jnp.arange(batch, dtype=jnp.int32)[:, None, None]
        windows = pooled[batch_indices, frame_indices]
        num_candidates = windows.shape[1]
        flat_windows = windows.reshape(
            batch * num_candidates,
            WINDOW,
            _semantic.SPATIAL_TOKENS,
            self.input_width,
        )
        event_logits, relation_logits = SlidingWindowEventRelationClassifier(
            name="window_classifier",
            input_width=self.input_width,
            width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            segment_size=WINDOW,
            dtype_mm=self.dtype_mm,
        )(flat_windows, train=train)
        event_logits = event_logits.reshape(batch, num_candidates)
        relation_logits = relation_logits.reshape(batch, num_candidates, _semantic.NUM_CUPS)

        if evaluate_all_windows:
            if num_candidates != NUM_WINDOWS:
                raise ValueError(f"Full scan requires {NUM_WINDOWS} windows, got {num_candidates}")
            event_scores = jax.nn.sigmoid(event_logits)
            automatic = _temporal_nms(event_scores, window_starts)
            cross_mask = _is_in(window_starts, CROSS_BOUNDARY_STARTS)
            automatic_no_cross = _temporal_nms(
                jnp.where(cross_mask, -jnp.inf, event_scores),
                window_starts,
            )
            aligned = jnp.tile(jnp.asarray(ALIGNED_STARTS, dtype=jnp.int32)[None], (batch, 1))
            forced_cross = jnp.tile(jnp.asarray((25, 35, 45), dtype=jnp.int32)[None], (batch, 1))
            selected_positions = jnp.stack(
                (aligned, automatic, automatic_no_cross, forced_cross),
                axis=1,
            )
            selection_counts = jnp.full(
                (batch, selected_positions.shape[1]),
                NUM_STAGES,
                dtype=jnp.int32,
            )
            if ENABLE_CAUSAL_EVAL_SELECTIONS:
                causal_positions, causal_count = _causal_rising_edge_select(event_logits)
                extra_positions = [causal_positions]
                extra_counts = [causal_count]
                for offset in range(6):
                    fixed_positions, fixed_count = _fixed_six_chunk_select(event_logits, window_starts, offset)
                    extra_positions.append(fixed_positions)
                    extra_counts.append(fixed_count)
                selected_positions = jnp.concatenate(
                    (selected_positions, jnp.stack(extra_positions, axis=1)),
                    axis=1,
                )
                selection_counts = jnp.concatenate(
                    (selection_counts, jnp.stack(extra_counts, axis=1)),
                    axis=1,
                )
        else:
            if num_candidates < NUM_STAGES:
                raise ValueError("Training requires the first three candidates to be complete events")
            selected_positions = jnp.tile(
                jnp.arange(NUM_STAGES, dtype=jnp.int32)[None, None],
                (batch, 1, 1),
            )
            selection_counts = jnp.full((batch, 1), NUM_STAGES, dtype=jnp.int32)

        condition_count = selected_positions.shape[1]
        batch_axis = jnp.arange(batch, dtype=jnp.int32)[:, None, None]
        chosen_relation_logits = relation_logits[batch_axis, selected_positions]
        chosen_event_logits = event_logits[batch_axis, selected_positions]
        if RELATION_CODE_MODE == "probabilities":
            relation_codes = jax.nn.softmax(chosen_relation_logits, axis=-1).astype(jnp.float32)
        elif RELATION_CODE_MODE == "one_hot":
            relation_codes = jax.nn.one_hot(
                jnp.argmax(chosen_relation_logits, axis=-1),
                _semantic.NUM_CUPS,
                dtype=jnp.float32,
            )
        else:
            raise ValueError(f"Unknown RELATION_CODE_MODE={RELATION_CODE_MODE!r}")
        relation_codes = relation_codes.reshape(
            batch * condition_count,
            NUM_STAGES,
            _semantic.NUM_CUPS,
        )

        segment_tokens = jnp.zeros(
            (
                batch * condition_count,
                NUM_STAGES,
                WINDOW,
                _semantic.SPATIAL_TOKENS,
                self.memory_width,
            ),
            dtype=jnp.float32,
        )
        segment_tokens = segment_tokens.at[..., : _semantic.NUM_CUPS].add(relation_codes[:, :, None, None, :])
        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.memory_width),
            jnp.float32,
        )
        memory = jnp.tile(base_memory, (batch * condition_count, 1, 1))
        tiled_initial = jnp.repeat(initial_slots, condition_count)
        initial_code = jax.nn.one_hot(tiled_initial, _semantic.NUM_CUPS, dtype=jnp.float32)
        memory = memory.at[:, 0, : _semantic.NUM_CUPS].add(initial_code)
        updater = _semantic.ShellGameSwapRecurrentMemoryUpdater(
            name="shared_swap_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            segment_size=WINDOW,
            dtype_mm="float32",
        )
        _, stage_memories = updater(memory, segment_tokens)
        adapter = _memory_core.SingleHistoryReadAdapter(
            name="shared_history_read_adapter",
            memory_width=self.memory_width,
            current_width=self.current_width,
            num_heads=self.adapter_heads,
            residual_scale=self.residual_scale,
        )
        readout = _semantic.SharedMemoryTokenReadout(
            name="shared_readout",
            width=self.current_width,
        )
        base_current_tokens = self.param(
            "base_current_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_current_tokens, self.current_width),
            jnp.float32,
        )
        current_tokens = jnp.tile(base_current_tokens, (batch * condition_count, 1, 1))
        stage_logits = [
            readout(adapter(current_tokens, stage_memories[:, stage_index])) for stage_index in range(NUM_STAGES)
        ]
        stage_logits = jnp.stack(stage_logits, axis=1).reshape(
            batch,
            condition_count,
            NUM_STAGES,
            _semantic.NUM_CUPS,
        )
        stage_memories = stage_memories.reshape(
            batch,
            condition_count,
            NUM_STAGES,
            self.num_memory_tokens,
            self.memory_width,
        )
        selected_starts = jnp.take_along_axis(
            window_starts[:, None, :],
            selected_positions,
            axis=2,
        )
        return {
            "event_logits": event_logits,
            "relation_logits": relation_logits,
            "selected_positions": selected_positions,
            "selected_starts": selected_starts,
            "selected_event_logits": chosen_event_logits,
            "selected_relation_logits": chosen_relation_logits,
            "selection_counts": selection_counts,
            "selection_valid": selection_counts == NUM_STAGES,
            "stage_logits": stage_logits,
            "stage_memories": stage_memories,
        }


@dataclasses.dataclass(frozen=True)
class SlidingWindowMemoryConfig(_shellgame_model.Pi0MemSemanticActionConfig):
    """Task config for the boundary-aware sliding-window tracker."""

    def create(self, rng: at.KeyArrayLike) -> Pi0SlidingWindowMemory:
        return Pi0SlidingWindowMemory(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_sliding_window(self) -> nnx.filterlib.Filter:
        initial = nnx_utils.PathRegex(r".*HistoryFrame0InitialCupClassifier.*")
        window_classifier = nnx_utils.PathRegex(r".*HistorySlidingWindowRelationMemoryTracker/window_classifier.*")
        return nnx.Not(nnx.Any(initial, window_classifier))


class Pi0SlidingWindowMemory(_base_model.Pi0MemCompress):
    """Minimal policy shell exposing the sliding-window memory path."""

    def __init__(self, config: SlidingWindowMemoryConfig, rngs: nnx.Rngs):
        if config.num_frames not in (config.history_frames, config.history_frames + 1):
            raise ValueError(
                "Sliding-window memory requires 60 history frames, optionally followed by one current frame"
            )
        super().__init__(config, rngs)
        self.history_frames = int(config.history_frames)
        self.HistoryFrame0InitialCupClassifier = nnx_bridge.ToNNX(
            _semantic.FrozenFrame0InitialCupClassifier(input_width=1152)
        )
        self.HistoryFrame0InitialCupClassifier.lazy_init(
            jnp.zeros((1, 256, 1152), dtype=jnp.bfloat16),
            rngs=rngs,
        )
        self.HistorySlidingWindowRelationMemoryTracker = nnx_bridge.ToNNX(
            SlidingWindowRelationMemoryTracker(
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
        self.HistorySlidingWindowRelationMemoryTracker.lazy_init(
            jnp.zeros((1, config.history_frames, 256, 1152), dtype=jnp.bfloat16),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.asarray([[20, 30, 40, 25, 35, 45]], dtype=jnp.int32),
            evaluate_all_windows=False,
            train=False,
            rngs=rngs,
        )

    def compute_window_outputs(
        self,
        rng,
        observation,
        *,
        initial_slots,
        window_starts,
        evaluate_all_windows: bool,
        train: bool,
    ):
        observation = _model.preprocess_observation(rng, observation, train=train)
        image = observation.images.get("base_rgb")
        valid_frame_counts = (self.history_frames, self.history_frames + 1)
        if image is None or image.ndim != 5 or image.shape[1] not in valid_frame_counts:
            raise ValueError(
                f"Expected base_rgb frame count in {valid_frame_counts}, got {None if image is None else image.shape}"
            )
        history = image[:, : self.history_frames]
        _, initial_encoder_out = self.PaliGemma.img(history[:, :1], train=False)
        initial_logits = self.HistoryFrame0InitialCupClassifier(initial_encoder_out["encoded"])
        _, history_encoder_out = self.PaliGemma.img(history, train=False)
        tracker_outputs = self.HistorySlidingWindowRelationMemoryTracker(
            history_encoder_out["with_posemb"][:, : self.history_frames],
            initial_slots.astype(jnp.int32),
            window_starts.astype(jnp.int32),
            evaluate_all_windows=evaluate_all_windows,
            train=train,
        )
        return {"initial_logits": initial_logits, **tracker_outputs}


@dataclasses.dataclass(frozen=True)
class SlidingWindowCheckpointLoader:
    """Restore the proven tracker and add only a random event-gate head."""

    params_path: str

    def load(self, params):
        source = flax.traverse_util.flatten_dict(
            _model.restore_params(self.params_path, restore_type=np.ndarray),
            sep="/",
        )
        target = flax.traverse_util.flatten_dict(params, sep="/")
        target_tracker = "HistorySlidingWindowRelationMemoryTracker/"
        source_tracker = "HistoryThreeSwapVisualRelationMemoryTracker/"
        target_classifier = target_tracker + "window_classifier/"
        source_classifier = source_tracker + "swap_relation_classifier/"
        result = {}
        counts = {"base": 0, "initial": 0, "tracker": 0, "random_event": 0}
        missing = []
        for key, reference in target.items():
            candidate = None
            kind = "base"
            if key.startswith(target_classifier + "event_classifier/"):
                result[key] = reference
                counts["random_event"] += 1
                continue
            if key.startswith(target_classifier):
                relative = key.removeprefix(target_classifier)
                if relative.startswith("relation_classifier/"):
                    relative = "classifier/" + relative.removeprefix("relation_classifier/")
                candidate = source.get(source_classifier + relative)
                kind = "tracker"
            elif key.startswith(target_tracker):
                relative = key.removeprefix(target_tracker)
                candidate = source.get(source_tracker + relative)
                kind = "tracker"
            else:
                candidate = source.get(key)
                if key.startswith("HistoryFrame0InitialCupClassifier/"):
                    kind = "initial"
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                counts[kind] += 1
            else:
                result[key] = reference
                missing.append(key)
        if missing:
            raise ValueError(f"Sliding-window checkpoint restore incomplete: {missing[:8]}")
        print(
            "SlidingWindowCheckpointLoader: "
            + ", ".join(f"{key}={value}" for key, value in counts.items())
            + ", missing=0"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def _sample_training_starts(rng, batch: int, episode_index, *, train: bool):
    """Build three complete positives plus hard-cross/static negatives."""
    if train:
        positive_rng, cross_rng, static_rng = jax.random.split(rng, 3)
        offsets = jax.random.randint(
            positive_rng,
            (batch, NUM_STAGES),
            minval=-POSITIVE_OFFSET_RADIUS,
            maxval=POSITIVE_OFFSET_RADIUS + 1,
        )
        cross_choice = jax.random.randint(cross_rng, (batch, 2), minval=0, maxval=5)
        static_choice = jax.random.randint(
            static_rng,
            (batch, 3),
            minval=0,
            maxval=len(STATIC_OR_PARTIAL_STARTS),
        )
    else:
        episode_index = episode_index.astype(jnp.int32)
        offsets = jnp.stack(
            [((episode_index * (stage + 3) + stage) % 5) - 2 for stage in range(NUM_STAGES)],
            axis=1,
        )
        cross_choice = jnp.stack(((episode_index * 3) % 5, (episode_index * 7 + 1) % 5), axis=1)
        static_choice = jnp.stack(
            [((episode_index * (index + 5) + index) % len(STATIC_OR_PARTIAL_STARTS)) for index in range(3)],
            axis=1,
        )
    positives = jnp.asarray(ALIGNED_STARTS, dtype=jnp.int32)[None] + offsets
    first_cross = jnp.asarray(CROSS_BOUNDARY_STARTS[:5], dtype=jnp.int32)[cross_choice[:, 0]]
    second_cross = jnp.asarray(CROSS_BOUNDARY_STARTS[5:], dtype=jnp.int32)[cross_choice[:, 1]]
    static = jnp.asarray(STATIC_OR_PARTIAL_STARTS, dtype=jnp.int32)[static_choice]
    return jnp.concatenate((positives, first_cross[:, None], second_cross[:, None], static), axis=1)


def sliding_window_objective(config, model, rng, observation, label_table, *, train: bool):
    """Compute initial + relation/event + recurrent stage-memory losses."""
    if observation.episode_index is None:
        raise ValueError("Sliding-window training requires episode_index")
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode_index]
    initial_labels = labels[:, 0]
    relation_labels = labels[:, 1 : 1 + NUM_STAGES]
    stage_labels = labels[:, 1 + NUM_STAGES :]
    starts = _sample_training_starts(rng, episode_index.shape[0], episode_index, train=train)
    outputs = model.compute_window_outputs(
        rng,
        observation,
        initial_slots=initial_labels,
        window_starts=starts,
        evaluate_all_windows=False,
        train=train and config.memory_train_augmentation,
    )
    initial_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            outputs["initial_logits"].astype(jnp.float32),
            initial_labels,
        )
    )
    relation_ce = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            outputs["relation_logits"][:, :NUM_STAGES].astype(jnp.float32),
            relation_labels,
        )
    )
    event_targets = jnp.concatenate(
        (
            jnp.ones((episode_index.shape[0], NUM_STAGES), dtype=jnp.float32),
            jnp.zeros((episode_index.shape[0], starts.shape[1] - NUM_STAGES), dtype=jnp.float32),
        ),
        axis=1,
    )
    event_ce = jnp.mean(optax.sigmoid_binary_cross_entropy(outputs["event_logits"], event_targets))
    relation_loss = 0.5 * (relation_ce + event_ce)
    stage_logits = outputs["stage_logits"][:, 0]
    stage_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(stage_logits.astype(jnp.float32), stage_labels)
    )
    loss = (
        config.initial_loss_weight * initial_loss
        + config.relation_loss_weight * relation_loss
        + config.stage_memory_loss_weight * stage_loss
    )
    event_predictions = outputs["event_logits"] > 0
    metrics = {
        "loss": loss,
        "initial_loss": initial_loss,
        "relation_loss": relation_loss,
        "relation_class_loss": relation_ce,
        "event_gate_loss": event_ce,
        "stage_memory_loss": stage_loss,
        "initial_accuracy": jnp.mean(jnp.argmax(outputs["initial_logits"], axis=-1) == initial_labels),
        "relation_accuracy": jnp.mean(
            jnp.argmax(outputs["relation_logits"][:, :NUM_STAGES], axis=-1) == relation_labels
        ),
        "event_accuracy": jnp.mean(event_predictions == event_targets.astype(jnp.bool_)),
        "complete_event_recall": jnp.mean(event_predictions[:, :NUM_STAGES]),
        "cross_boundary_rejection": jnp.mean(~event_predictions[:, NUM_STAGES : NUM_STAGES + 2]),
        "static_partial_rejection": jnp.mean(~event_predictions[:, NUM_STAGES + 2 :]),
        "stage_memory_accuracy": jnp.mean(jnp.argmax(stage_logits, axis=-1) == stage_labels),
        "final_memory_accuracy": jnp.mean(jnp.argmax(stage_logits[:, -1], axis=-1) == stage_labels[:, -1]),
    }
    for stage_index in range(NUM_STAGES):
        metrics[f"slot_{stage_index}_accuracy"] = jnp.mean(
            jnp.argmax(stage_logits[:, stage_index], axis=-1) == stage_labels[:, stage_index]
        )
    return loss, metrics


def sliding_full_eval_step(config, label_table, rng, state, batch):
    """Scan all windows and compare boundary-aware recurrent conditions."""
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, _actions = batch
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode_index]
    initial_labels = labels[:, 0]
    relation_labels = labels[:, 1 : 1 + NUM_STAGES]
    stage_labels = labels[:, 1 + NUM_STAGES :]
    starts = jnp.tile(jnp.arange(NUM_WINDOWS, dtype=jnp.int32)[None], (episode_index.shape[0], 1))
    outputs = model.compute_window_outputs(
        rng,
        observation,
        initial_slots=initial_labels,
        window_starts=starts,
        evaluate_all_windows=True,
        train=False,
    )

    initial_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(outputs["initial_logits"], initial_labels))
    stage_index = _stage_index_for_start(starts)
    event_targets = stage_index >= 0
    positive_event_loss = jnp.sum(
        optax.sigmoid_binary_cross_entropy(outputs["event_logits"], jnp.ones_like(outputs["event_logits"]))
        * event_targets
    ) / jnp.maximum(jnp.sum(event_targets), 1)
    negative_event_loss = jnp.sum(
        optax.sigmoid_binary_cross_entropy(outputs["event_logits"], jnp.zeros_like(outputs["event_logits"]))
        * (~event_targets)
    ) / jnp.maximum(jnp.sum(~event_targets), 1)
    event_loss = 0.5 * (positive_event_loss + negative_event_loss)
    safe_stage_index = jnp.maximum(stage_index, 0)
    window_relation_targets = jnp.take_along_axis(relation_labels, safe_stage_index, axis=1)
    relation_losses = optax.softmax_cross_entropy_with_integer_labels(
        outputs["relation_logits"],
        window_relation_targets,
    )
    relation_ce = jnp.sum(relation_losses * event_targets) / jnp.maximum(jnp.sum(event_targets), 1)
    relation_loss = 0.5 * (relation_ce + event_loss)
    automatic_stage_logits = outputs["stage_logits"][:, 1]
    stage_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(automatic_stage_logits, stage_labels))
    loss = (
        config.initial_loss_weight * initial_loss
        + config.relation_loss_weight * relation_loss
        + config.stage_memory_loss_weight * stage_loss
    )
    result = {
        "val/loss": loss,
        "val/initial_loss": initial_loss,
        "val/relation_loss": relation_loss,
        "val/relation_class_loss": relation_ce,
        "val/event_gate_loss": event_loss,
        "val/stage_memory_loss": stage_loss,
        "val/initial_accuracy": jnp.mean(jnp.argmax(outputs["initial_logits"], axis=-1) == initial_labels),
        "val/complete_event_recall": jnp.sum((outputs["event_logits"] > 0) * event_targets)
        / jnp.maximum(jnp.sum(event_targets), 1),
        "val/no_event_rejection": jnp.sum((outputs["event_logits"] <= 0) * (~event_targets))
        / jnp.maximum(jnp.sum(~event_targets), 1),
        "val/cross_boundary_rejection": jnp.mean(
            outputs["event_logits"][:, jnp.asarray(STRICT_CROSS_STARTS, dtype=jnp.int32)] <= 0
        ),
        "val/positive_window_relation_accuracy": jnp.sum(
            (jnp.argmax(outputs["relation_logits"], axis=-1) == window_relation_targets) * event_targets
        )
        / jnp.maximum(jnp.sum(event_targets), 1),
    }
    aligned = jnp.asarray(ALIGNED_STARTS, dtype=jnp.int32)[None]
    for condition_index, condition_name in enumerate(CONDITION_NAMES):
        selected_starts = outputs["selected_starts"][:, condition_index]
        selected_relation_ids = jnp.argmax(
            outputs["selected_relation_logits"][:, condition_index],
            axis=-1,
        )
        stage_logits = outputs["stage_logits"][:, condition_index]
        stage_predictions = jnp.argmax(stage_logits, axis=-1)
        relation_correct = selected_relation_ids == relation_labels
        selection_valid = outputs["selection_valid"][:, condition_index]
        valid_count = jnp.maximum(jnp.sum(selection_valid), 1)
        final_correct = stage_predictions[:, -1] == stage_labels[:, -1]
        prefix = f"val/{condition_name}"
        result[f"{prefix}/valid_trigger_count"] = jnp.mean(selection_valid)
        result[f"{prefix}/mean_trigger_count"] = jnp.mean(outputs["selection_counts"][:, condition_index])
        result[f"{prefix}/relation_accuracy"] = (
            jnp.sum(jnp.mean(relation_correct, axis=1) * selection_valid) / valid_count
        )
        result[f"{prefix}/relation_sequence_accuracy"] = (
            jnp.sum(jnp.all(relation_correct, axis=1) * selection_valid) / valid_count
        )
        result[f"{prefix}/relation_sequence_e2e_accuracy"] = jnp.mean(
            jnp.all(relation_correct, axis=1) & selection_valid
        )
        result[f"{prefix}/stage_memory_accuracy"] = (
            jnp.sum(jnp.mean(stage_predictions == stage_labels, axis=1) * selection_valid) / valid_count
        )
        result[f"{prefix}/final_memory_accuracy"] = jnp.sum(final_correct * selection_valid) / valid_count
        result[f"{prefix}/final_memory_e2e_accuracy"] = jnp.mean(final_correct & selection_valid)
        result[f"{prefix}/start_mae"] = (
            jnp.sum(jnp.mean(jnp.abs(selected_starts - aligned).astype(jnp.float32), axis=1) * selection_valid)
            / valid_count
        )
        result[f"{prefix}/selected_cross_fraction"] = (
            jnp.sum(jnp.mean(_is_in(selected_starts, CROSS_BOUNDARY_STARTS), axis=1) * selection_valid) / valid_count
        )
    return result


def _copy_model_config(source) -> SlidingWindowMemoryConfig:
    values = {field.name: getattr(source, field.name) for field in dataclasses.fields(SlidingWindowMemoryConfig)}
    return SlidingWindowMemoryConfig(**values)


def build_config(args: argparse.Namespace):
    base = _recipe.make_train_config()
    model = _copy_model_config(base.model)
    return dataclasses.replace(
        base,
        name="shellgame_sliding_window_event_recurrent_memory_probe",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_sliding_window(),
        weight_loader=SlidingWindowCheckpointLoader(args.init_checkpoint),
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
        initial_loss_weight=1.0,
        relation_loss_weight=1.0,
        stage_memory_loss_weight=1.0,
        memory_train_augmentation=False,
        wandb_enabled=False,
        overwrite=args.overwrite,
    )


def run_self_test() -> None:
    starts = jnp.arange(NUM_WINDOWS, dtype=jnp.int32)[None]
    stage_index = np.asarray(_stage_index_for_start(starts))[0]
    if int(np.sum(stage_index >= 0)) != 15:
        raise AssertionError(f"Expected 15 complete-event windows, got {np.sum(stage_index >= 0)}")
    if np.any(stage_index[np.asarray(STRICT_CROSS_STARTS)] >= 0):
        raise AssertionError("Strict cross-boundary windows must be no_event negatives")

    tracker = SlidingWindowRelationMemoryTracker(
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
    training_starts = jnp.tile(jnp.asarray((20, 30, 40, 25, 35, 45))[None], (2, 1))
    variables = tracker.init(
        jax.random.key(0),
        patches,
        initial,
        training_starts,
        evaluate_all_windows=False,
        train=False,
    )
    training_output = tracker.apply(
        variables,
        patches,
        initial,
        training_starts,
        evaluate_all_windows=False,
        train=False,
    )
    full_starts = jnp.tile(jnp.arange(NUM_WINDOWS, dtype=jnp.int32)[None], (2, 1))
    full_output = tracker.apply(
        variables,
        patches,
        initial,
        full_starts,
        evaluate_all_windows=True,
        train=False,
    )
    if training_output["stage_logits"].shape != (2, 1, 3, 3):
        raise AssertionError(f"Unexpected train stage shape: {training_output['stage_logits'].shape}")
    if full_output["stage_logits"].shape != (2, len(CONDITION_NAMES), 3, 3):
        raise AssertionError(f"Unexpected full stage shape: {full_output['stage_logits'].shape}")
    if not np.array_equal(np.asarray(full_output["selected_starts"][:, 0]), np.tile(ALIGNED_STARTS, (2, 1))):
        raise AssertionError("Aligned full-scan selector is incorrect")
    print("Sliding-window self-test passed: 15 positives, strict boundary negatives, train/full recurrent shapes valid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=DEFAULT_ANCHORED_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test_only:
        run_self_test()
        return
    _recipe.compute_objective = sliding_window_objective
    _trainer.eval_step = sliding_full_eval_step
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
