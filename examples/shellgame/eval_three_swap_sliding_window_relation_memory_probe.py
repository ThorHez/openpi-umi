"""Evaluate zero-shot sliding-window event selection for ShellGame memory.

This is a strict interface control.  It restores the previously validated
initial-state decoder, ten-frame swap-relation decoder, and recurrent memory
without training any new weights.  The only change is how three relation clips
are selected:

* fixed starts 20/30/40 (the current implementation);
* temporal NMS over all 51 ten-frame windows using relation confidence;
* temporal NMS using visual motion energy; or
* temporal NMS using normalized confidence plus motion energy.

The same selectors are evaluated on the original history and on a per-episode
integer shift in [-8, 8].  Ground-truth initial cup is supplied to isolate the
event-boundary question from the already solved frame-0 classifier.
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
import jax
import jax.numpy as jnp

from examples.shellgame import eval_three_swap_fully_visual_relation_memory_probe as _fully
from examples.shellgame import train_three_swap_causal_fixed_grid_probe as _causal
from examples.shellgame import train_three_swap_oracle_initial_recurrent_memory_probe as _oracle_initial
from examples.shellgame import train_three_swap_oracle_pair_recurrent_memory_probe as _oracle_pair
from examples.shellgame import train_three_swap_visual_relation_onehot_memory_probe as _relation
from examples.shellgame import train_three_swap_visual_semantic_readout_memory_probe as _semantic
from examples.shellgame.train_three_swap_oracle_memory_token_sweep_probe import SharedMemoryTokenReadout
from examples.shellgame.train_three_swap_oracle_single_history_read_adapter_probe import SingleHistoryReadAdapter
from examples.shellgame.train_three_swap_recurrent_memory_fixed_grid_probe import SharedSegmentMemoryUpdater
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.training import config as _config
from openpi.training import utils as training_utils
from scripts.mem import train_pi0_mem_compress as _trainer

WINDOW = _semantic.SWAP_SEGMENT_SIZE
NUM_WINDOWS = _semantic.HISTORY_FRAMES - WINDOW + 1
FIXED_STARTS = (20, 30, 40)
NUM_EVENTS = 3
MIN_EVENT_SEPARATION = 8
SHIFT_RADIUS = 8
SELECTORS = ("fixed", "oracle_start", "confidence", "motion", "combined")
PAIR_ENDPOINTS = jnp.asarray(((0, 1), (0, 2), (1, 2)), dtype=jnp.int32)
EEF_DATASET_ROOT = pathlib.Path("/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7")
EEF_RAW_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_absolute_eef_phase_instruction_dataset"
)


def _shift_time_axis(x: jax.Array, shifts: jax.Array) -> jax.Array:
    """Edge-pad a batch of histories after applying an integer time shift."""
    times = jnp.arange(x.shape[1], dtype=jnp.int32)[None, :]
    source = jnp.clip(times - shifts[:, None], 0, x.shape[1] - 1)
    return jax.vmap(lambda sample, indices: sample[indices])(x, source)


def _temporal_nms(scores: jax.Array) -> jax.Array:
    """Select three high-scoring, separated windows and return time order."""
    if scores.ndim != 2 or scores.shape[1] != NUM_WINDOWS:
        raise ValueError(f"Expected scores [B,{NUM_WINDOWS}], got {scores.shape}")
    starts = jnp.arange(NUM_WINDOWS, dtype=jnp.int32)[None, :]
    available = jnp.ones_like(scores, dtype=jnp.bool_)
    selected = []
    for _ in range(NUM_EVENTS):
        index = jnp.argmax(jnp.where(available, scores, -jnp.inf), axis=1)
        selected.append(index)
        available &= jnp.abs(starts - index[:, None]) >= MIN_EVENT_SEPARATION
    first, second, third = selected
    minimum = jnp.minimum(jnp.minimum(first, second), third)
    maximum = jnp.maximum(jnp.maximum(first, second), third)
    middle = first + second + third - minimum - maximum
    return jnp.stack((minimum, middle, maximum), axis=1)


def _standardize(values: jax.Array) -> jax.Array:
    mean = jnp.mean(values, axis=1, keepdims=True)
    std = jnp.std(values, axis=1, keepdims=True)
    return (values - mean) / jnp.maximum(std, 1e-5)


class SlidingWindowVisualRelationMemoryTracker(nn.Module):
    """Scan every ten-frame window, then run the unchanged recurrent memory."""

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
    def __call__(self, patch_tokens, initial_slots, shifts):
        batch, frames, tokens, width = patch_tokens.shape
        expected = (self.num_frames, 256, self.input_width)
        if (frames, tokens, width) != expected:
            raise ValueError(f"Expected [B,{expected}], got {patch_tokens.shape}")
        if initial_slots.shape != (batch,) or shifts.shape != (batch,):
            raise ValueError(f"Expected initial_slots/shifts [B], got {initial_slots.shape}/{shifts.shape}")

        shifted = _shift_time_axis(patch_tokens, shifts)
        pooled = _semantic.pool_fixed_grid(shifted)
        windows = jnp.stack(
            [pooled[:, start : start + WINDOW] for start in range(NUM_WINDOWS)],
            axis=1,
        )
        flat_windows = windows.reshape(
            batch * NUM_WINDOWS,
            WINDOW,
            _semantic.SPATIAL_TOKENS,
            self.input_width,
        )
        relation_logits = _relation.FrozenSwapRelationClassifier(
            name="swap_relation_classifier",
            input_width=self.input_width,
            width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            segment_size=WINDOW,
            dtype_mm=self.dtype_mm,
        )(flat_windows).reshape(batch, NUM_WINDOWS, 3)
        probabilities = jax.nn.softmax(relation_logits.astype(jnp.float32), axis=-1)
        best_index = jnp.argmax(probabilities, axis=-1)
        best = jnp.max(probabilities, axis=-1)
        runner_up = jnp.max(
            jnp.where(
                jnp.arange(3, dtype=jnp.int32) == best_index[..., None],
                -jnp.inf,
                probabilities,
            ),
            axis=-1,
        )
        confidence_scores = best - runner_up

        frame_motion = jnp.mean(
            jnp.square(pooled[:, 1:].astype(jnp.float32) - pooled[:, :-1].astype(jnp.float32)),
            axis=(2, 3),
        )
        motion_scores = jnp.stack(
            [jnp.mean(frame_motion[:, start : start + WINDOW - 1], axis=1) for start in range(NUM_WINDOWS)],
            axis=1,
        )
        combined_scores = _standardize(confidence_scores) + _standardize(motion_scores)

        batch_axis = jnp.arange(batch, dtype=jnp.int32)[:, None]
        fixed_starts = jnp.tile(jnp.asarray(FIXED_STARTS, dtype=jnp.int32)[None], (batch, 1))
        oracle_starts = fixed_starts + shifts[:, None]
        selected_starts = {
            "fixed": fixed_starts,
            "oracle_start": oracle_starts,
            "confidence": _temporal_nms(confidence_scores),
            "motion": _temporal_nms(motion_scores),
            "combined": _temporal_nms(combined_scores),
        }

        updater = SharedSegmentMemoryUpdater(
            name="shared_swap_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            segment_size=WINDOW,
            dtype_mm="float32",
        )
        adapter = SingleHistoryReadAdapter(
            name="shared_history_read_adapter",
            memory_width=self.memory_width,
            current_width=self.current_width,
            num_heads=self.adapter_heads,
            residual_scale=self.residual_scale,
        )
        readout = SharedMemoryTokenReadout(name="shared_readout", width=self.current_width)
        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.memory_width),
            jnp.float32,
        )
        base_current_tokens = self.param(
            "base_current_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_current_tokens, self.current_width),
            jnp.float32,
        )

        outputs = {}
        for selector in SELECTORS:
            starts_for_selector = selected_starts[selector]
            selected_logits = relation_logits[batch_axis, starts_for_selector]
            relation_ids = jnp.argmax(selected_logits, axis=-1)
            relation_codes = jax.nn.one_hot(relation_ids, 3, dtype=jnp.float32)
            evidence = jnp.zeros(
                (
                    batch,
                    NUM_EVENTS,
                    WINDOW,
                    _semantic.SPATIAL_TOKENS,
                    self.memory_width,
                ),
                dtype=jnp.float32,
            )
            evidence = evidence.at[..., :3].add(relation_codes[:, :, None, None, :])

            memory = jnp.tile(base_memory, (batch, 1, 1))
            initial_code = jax.nn.one_hot(initial_slots, 3, dtype=jnp.float32)
            memory = memory.at[:, 0, :3].add(initial_code)
            current_tokens = jnp.tile(base_current_tokens, (batch, 1, 1))
            stage_logits = []
            for stage_index in range(NUM_EVENTS):
                memory = updater(memory, evidence[:, stage_index])
                stage_logits.append(readout(adapter(current_tokens, memory)))
            outputs[selector] = {
                "starts": starts_for_selector,
                "relation_ids": relation_ids,
                "stage_logits": jnp.stack(stage_logits, axis=1),
            }
        return outputs


@dataclasses.dataclass(frozen=True)
class SlidingWindowConfig(_fully.FullyVisualRelationMemoryConfig):
    def create(self, rng: at.KeyArrayLike) -> SlidingWindowModel:
        return SlidingWindowModel(self, rngs=nnx.Rngs(rng))


class SlidingWindowModel(_fully.FullyVisualRelationMemoryModel):
    def __init__(self, config: SlidingWindowConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.HistoryThreeSwapVisualRelationMemoryTracker = nnx_bridge.ToNNX(
            SlidingWindowVisualRelationMemoryTracker(
                num_frames=config.history_frames if hasattr(config, "history_frames") else _semantic.HISTORY_FRAMES,
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
        self.HistoryThreeSwapVisualRelationMemoryTracker.lazy_init(
            jnp.zeros((1, _semantic.HISTORY_FRAMES, 256, 1152), dtype=jnp.bfloat16),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            rngs=rngs,
        )

    def compute_sliding_diagnostics(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        initial_slots: jax.Array,
        shifts: jax.Array,
    ):
        observation = _model.preprocess_observation(rng, observation, train=False)
        image = observation.images["base_rgb"]
        if image.ndim != 5 or image.shape[1] != _semantic.TOTAL_INPUT_FRAMES:
            raise ValueError(f"Expected [B,{_semantic.TOTAL_INPUT_FRAMES},H,W,C], got {image.shape}")
        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, : _semantic.HISTORY_FRAMES]
        normal = self.HistoryThreeSwapVisualRelationMemoryTracker(
            history_patches,
            initial_slots,
            jnp.zeros_like(shifts),
        )
        jittered = self.HistoryThreeSwapVisualRelationMemoryTracker(
            history_patches,
            initial_slots,
            shifts,
        )
        return {"normal": normal, "jittered": jittered}


def _apply_pair(slot: jax.Array, pair_ids: jax.Array) -> jax.Array:
    endpoints = PAIR_ENDPOINTS[pair_ids]
    left, right = endpoints[..., 0], endpoints[..., 1]
    return jnp.where(slot == left, right, jnp.where(slot == right, left, slot))


def sliding_eval_step(
    config,
    rng,
    state: training_utils.TrainState,
    batch,
    *,
    class_labels_by_episode=None,
):
    del class_labels_by_episode
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, _actions = batch
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    safe_episode = jnp.clip(episode_index, 0, len(config.model.oracle_initial_slots) - 1)
    true_initial = jnp.asarray(config.model.oracle_initial_slots, dtype=jnp.int32)[safe_episode]
    true_relations = jnp.asarray(config.model.oracle_swap_pairs, dtype=jnp.int32)[safe_episode]
    shifts = ((safe_episode * 5 + 3) % (2 * SHIFT_RADIUS + 1)) - SHIFT_RADIUS
    conditions = model.compute_sliding_diagnostics(
        rng,
        observation,
        true_initial,
        shifts.astype(jnp.int32),
    )

    true_stages = []
    slot = true_initial
    for stage_index in range(NUM_EVENTS):
        slot = _apply_pair(slot, true_relations[:, stage_index])
        true_stages.append(slot)
    true_stages = jnp.stack(true_stages, axis=1)

    result = {}
    for condition_name, condition in conditions.items():
        expected_starts = jnp.asarray(FIXED_STARTS, dtype=jnp.int32)[None]
        if condition_name == "jittered":
            expected_starts = expected_starts + shifts[:, None]
        for selector, output in condition.items():
            prefix = f"val/{condition_name}/{selector}"
            relation_correct = output["relation_ids"] == true_relations
            stage_predictions = jnp.argmax(output["stage_logits"], axis=-1)
            start_error = jnp.abs(output["starts"] - expected_starts)
            result[f"{prefix}/relation_accuracy"] = jnp.mean(relation_correct)
            result[f"{prefix}/relation_sequence_accuracy"] = jnp.mean(jnp.all(relation_correct, axis=1))
            result[f"{prefix}/stage_accuracy"] = jnp.mean(stage_predictions == true_stages)
            result[f"{prefix}/final_slot_accuracy"] = jnp.mean(stage_predictions[:, -1] == true_stages[:, -1])
            result[f"{prefix}/start_mae"] = jnp.mean(start_error.astype(jnp.float32))
            result[f"{prefix}/start_within_2"] = jnp.mean(start_error <= 2)
    # A scalar required by the generic validation loop; the useful metrics are
    # the selector-specific values above.
    result["val/loss"] = 1.0 - result["val/jittered/combined/final_slot_accuracy"]
    return result


def build_config(args: argparse.Namespace, labels_path: pathlib.Path):
    parent = _fully.build_config(args, labels_path)
    parent_model = parent.model
    parent_fields = {field.name: getattr(parent_model, field.name) for field in dataclasses.fields(parent_model)}
    model = SlidingWindowConfig(**parent_fields)
    data = _config.MultiDataConfigFactory(
        state_pad_dim=96,
        weights=[1.0],
        datasets=[
            _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_AbsoluteEEF7(
                repo_id=str(EEF_DATASET_ROOT),
                assets=_config.AssetsConfig(
                    asset_id=".",
                    assets_dir=str(EEF_DATASET_ROOT),
                ),
                base_config=_config.UmiDataConfig(
                    action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
                    robot_type="ARM=1 G=0 H=0",
                ),
                num_frames=_semantic.TOTAL_INPUT_FRAMES,
                frame_stride=1,
                video_layout="fixed_prefix_current",
                fixed_prefix_frames=_semantic.HISTORY_FRAMES,
                tokenize_prompt=False,
                min_frame_index=_semantic.HISTORY_FRAMES - 1,
                max_frame_index=_semantic.HISTORY_FRAMES - 1,
            )
        ],
    )
    return dataclasses.replace(
        parent,
        name="pi0_shellgame_three_swap_sliding_window_relation_memory_260821",
        model=model,
        data=data,
        freeze_filter=model.get_freeze_filter_fully_visual(),
        shellgame_memory_classifier=dataclasses.replace(
            parent.shellgame_memory_classifier,
            enabled=False,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=_semantic.DEFAULT_PAIR_CHECKPOINT)
    parser.add_argument("--initial-checkpoint", default=_fully.DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument("--memory-checkpoint", default=_fully.DEFAULT_MEMORY_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--peak-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
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
    parser.add_argument("--video-mode", default="normal")
    parser.add_argument("--initial-mode", default="normal")
    parser.add_argument("--relation-mode", default="one_hot")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The validated 260809 modules imported their old joint-data roots by
    # value. Point every lookup/config owner at the current EEF data while
    # leaving all restored model parameters unchanged.
    _causal.DATASET_ROOT = EEF_DATASET_ROOT
    _causal.RAW_DATASET_ROOT = EEF_RAW_ROOT
    _semantic.DATASET_ROOT = EEF_DATASET_ROOT
    _oracle_initial.RAW_DATASET_ROOT = EEF_RAW_ROOT
    _oracle_pair.RAW_DATASET_ROOT = EEF_RAW_ROOT
    _trainer.eval_step = sliding_eval_step
    _trainer.main(build_config(args, _causal.build_three_swap_labels()))


if __name__ == "__main__":
    main()
