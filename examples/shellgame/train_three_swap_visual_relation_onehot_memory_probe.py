"""Feed frozen visual swap predictions into the proven symbolic memory path.

Each ten-frame swap clip is processed by the complete frozen visual relation
classifier that reached 100% held-out accuracy.  Its argmax is converted to the
same parameter-free three-way one-hot code used by the successful symbolic
single-history-read experiment.  The true initial cup slot remains an oracle
input and action loss is disabled.

This is the hard-interface control between visual relation decoding and the
compact recurrent memory.  It tests whether the failure of continuous semantic
embeddings came from dropping the trained relation decoder.
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

from examples.shellgame import train_three_swap_oracle_memory_token_sweep_probe as _sweep
from examples.shellgame import train_three_swap_oracle_pair_recurrent_memory_probe as _oracle_pair
from examples.shellgame import train_three_swap_oracle_single_history_read_adapter_probe as _single_read
from examples.shellgame import train_three_swap_visual_semantic_readout_memory_probe as _semantic
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from examples.shellgame.train_three_swap_recurrent_memory_fixed_grid_probe import SharedSegmentMemoryUpdater
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import utils as training_utils
from scripts.mem import train_pi0_mem_compress as _trainer


class FrozenSwapRelationClassifier(nn.Module):
    """Restore and run the complete pretrained swap-pair classifier."""

    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    segment_size: int = _semantic.SWAP_SEGMENT_SIZE
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, segment_tokens):
        semantic = _semantic.PretrainedSwapSemanticEncoder(
            name="semantic_encoder",
            input_width=self.input_width,
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=self.segment_size,
            dtype_mm=self.dtype_mm,
        )(segment_tokens)
        return nn.Dense(3, name="classifier", dtype=jnp.float32)(semantic.astype(jnp.float32))


class ThreeSwapVisualRelationMemoryTracker(nn.Module):
    """Decode visual relations first, then use the exact symbolic memory input."""

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
    relation_mode: str = "one_hot"
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens, initial_slots):
        b, t, n, d = patch_tokens.shape
        expected = (self.num_frames, 256, self.input_width)
        if (t, n, d) != expected:
            raise ValueError(f"Expected [B,{expected}], got {patch_tokens.shape}")
        if initial_slots.shape != (b,):
            raise ValueError(f"Expected initial slots [B], got {initial_slots.shape}")

        pooled = _semantic.pool_fixed_grid(patch_tokens)
        clips = jnp.stack(
            [pooled[:, start:end] for start, end in _semantic.SWAP_SLICES], axis=1
        ).reshape(
            b * len(_semantic.SWAP_SLICES),
            _semantic.SWAP_SEGMENT_SIZE,
            _semantic.SPATIAL_TOKENS,
            self.input_width,
        )
        relation_logits = FrozenSwapRelationClassifier(
            name="swap_relation_classifier",
            input_width=self.input_width,
            width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            segment_size=_semantic.SWAP_SEGMENT_SIZE,
            dtype_mm=self.dtype_mm,
        )(clips).reshape(b, len(_semantic.SWAP_SLICES), 3)
        relation_ids = jnp.argmax(relation_logits, axis=-1)
        if self.relation_mode == "one_hot":
            relation_codes = jax.nn.one_hot(relation_ids, 3, dtype=jnp.float32)
        elif self.relation_mode == "probabilities":
            relation_codes = jax.nn.softmax(relation_logits, axis=-1).astype(jnp.float32)
        elif self.relation_mode == "logits":
            relation_codes = relation_logits.astype(jnp.float32)
        else:
            raise ValueError(f"Unknown relation_mode={self.relation_mode!r}")

        # This is deliberately identical to the successful symbolic control:
        # only the first three channels contain the explicit relation code.
        segment_tokens = jnp.zeros(
            (
                b,
                len(_semantic.SWAP_SLICES),
                _semantic.SWAP_SEGMENT_SIZE,
                _semantic.SPATIAL_TOKENS,
                self.memory_width,
            ),
            dtype=jnp.float32,
        )
        segment_tokens = segment_tokens.at[..., :3].add(relation_codes[:, :, None, None, :])

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.memory_width),
            jnp.float32,
        )
        memory = jnp.tile(base_memory, (b, 1, 1))
        initial_code = jax.nn.one_hot(initial_slots, 3, dtype=jnp.float32)
        memory = memory.at[:, 0, :3].add(initial_code)

        updater = SharedSegmentMemoryUpdater(
            name="shared_swap_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            segment_size=_semantic.SWAP_SEGMENT_SIZE,
            dtype_mm="float32",
        )
        adapter = _single_read.SingleHistoryReadAdapter(
            name="shared_history_read_adapter",
            memory_width=self.memory_width,
            current_width=self.current_width,
            num_heads=self.adapter_heads,
            residual_scale=self.residual_scale,
        )
        readout = _sweep.SharedMemoryTokenReadout(name="shared_readout", width=self.current_width)
        base_current_tokens = self.param(
            "base_current_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_current_tokens, self.current_width),
            jnp.float32,
        )
        base_current_tokens = jnp.tile(base_current_tokens, (b, 1, 1))

        stage_logits = []
        stage_memories = []
        for stage_index in range(len(_semantic.SWAP_SLICES)):
            memory = updater(memory, segment_tokens[:, stage_index])
            stage_memories.append(memory)
            current_tokens = adapter(base_current_tokens, memory)
            stage_logits.append(readout(current_tokens))

        stage_logits = jnp.stack(stage_logits, axis=1)
        stage_memories = jnp.stack(stage_memories, axis=1)
        logits_0, logits_1, logits_2 = stage_logits[:, 0], stage_logits[:, 1], stage_logits[:, 2]
        joint_logits = (
            logits_0[:, :, None, None]
            + logits_1[:, None, :, None]
            + logits_2[:, None, None, :]
        ).reshape(b, 27)
        return joint_logits, stage_logits, stage_memories, relation_logits, relation_ids


@dataclasses.dataclass(frozen=True)
class VisualRelationMemoryConfig(_semantic.VisualSemanticMemoryConfig):
    relation_mode: str = "one_hot"
    oracle_swap_pairs: tuple[tuple[int, int, int], ...] = ()

    def create(self, rng: at.KeyArrayLike) -> VisualRelationMemoryModel:
        return VisualRelationMemoryModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_visual_relation_tracker(self) -> nnx.filterlib.Filter:
        tracker = nnx_utils.PathRegex(r".*HistoryThreeSwapVisualRelationMemoryTracker.*")
        relation = nnx_utils.PathRegex(
            r".*HistoryThreeSwapVisualRelationMemoryTracker/swap_relation_classifier.*"
        )
        trainable = nnx.All(tracker, nnx.Not(relation))
        return nnx.Not(trainable)


class VisualRelationMemoryModel(_base_model.Pi0MemCompress):
    def __init__(self, config: VisualRelationMemoryConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.oracle_initial_slots = config.oracle_initial_slots
        self.oracle_swap_pairs = config.oracle_swap_pairs
        self.video_mode = config.video_mode
        self.HistoryThreeSwapVisualRelationMemoryTracker = nnx_bridge.ToNNX(
            ThreeSwapVisualRelationMemoryTracker(
                num_frames=_semantic.HISTORY_FRAMES,
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
                relation_mode=config.relation_mode,
                dtype_mm=config.dtype,
            )
        )
        fake_tokens = jnp.zeros((1, _semantic.HISTORY_FRAMES, 256, 1152), dtype=jnp.bfloat16)
        fake_slots = jnp.zeros((1,), dtype=jnp.int32)
        self.HistoryThreeSwapVisualRelationMemoryTracker.lazy_init(fake_tokens, fake_slots, rngs=rngs)

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
        if image.ndim != 5 or image.shape[1] != _semantic.TOTAL_INPUT_FRAMES:
            raise ValueError(
                f"Visual relation probe expects [B,{_semantic.TOTAL_INPUT_FRAMES},H,W,C], got {image.shape}"
            )
        if observation.episode_index is None:
            raise ValueError("Visual relation probe requires observation.episode_index")

        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, : _semantic.HISTORY_FRAMES]
        if self.video_mode == "shuffle_swaps":
            start, end = _semantic.SWAP_SLICES[0][0], _semantic.SWAP_SLICES[-1][1]
            history_patches = history_patches.at[:, start:end].set(
                jnp.roll(history_patches[:, start:end], 1, axis=0)
            )
        elif self.video_mode == "zero_swaps":
            start, end = _semantic.SWAP_SLICES[0][0], _semantic.SWAP_SLICES[-1][1]
            history_patches = history_patches.at[:, start:end].set(0)
        elif self.video_mode != "normal":
            raise ValueError(f"Unknown video_mode={self.video_mode!r}")

        episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
        initial_lookup = jnp.asarray(self.oracle_initial_slots, dtype=jnp.int32)
        safe_episode = jnp.clip(episode_index, 0, initial_lookup.shape[0] - 1)
        initial_slots = initial_lookup[safe_episode]
        joint_logits, stage_logits, stage_memories, relation_logits, relation_ids = (
            self.HistoryThreeSwapVisualRelationMemoryTracker(history_patches, initial_slots)
        )
        return joint_logits, {
            "history_mem": stage_memories.reshape(-1, stage_memories.shape[-2], stage_memories.shape[-1]),
            "stage_logits": stage_logits,
            "relation_logits": relation_logits,
            "relation_ids": relation_ids,
            "encoder_auxes": (),
        }


@dataclasses.dataclass(frozen=True)
class VisualRelationCheckpointLoader:
    """Restore the frozen base and every relation-classifier leaf."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        tracker_root = "HistoryThreeSwapVisualRelationMemoryTracker/"
        relation_root = tracker_root + "swap_relation_classifier/"
        source_root = "HistoryThreeSwapPairVisualClassifier/"
        result = {}
        exact_base = mapped_relation = 0
        initialized_tracker = []
        initialized_base = []

        for key, reference in target.items():
            candidate = source.get(key)
            source_kind = "exact"
            if candidate is None or np.shape(candidate) != np.shape(reference):
                candidate = None
                if key.startswith(relation_root):
                    relative = key.removeprefix(relation_root)
                    if relative.startswith("semantic_encoder/"):
                        relative = relative.removeprefix("semantic_encoder/")
                    source_candidate = source.get(source_root + relative)
                    if source_candidate is not None and np.shape(source_candidate) == np.shape(reference):
                        candidate = source_candidate
                        source_kind = "relation"

            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                if source_kind == "relation":
                    mapped_relation += 1
                else:
                    exact_base += 1
            else:
                result[key] = reference
                if key.startswith(tracker_root):
                    initialized_tracker.append(key)
                else:
                    initialized_base.append(key)

        missing_relation = [key for key in initialized_tracker if key.startswith(relation_root)]
        if missing_relation:
            raise ValueError(f"Frozen relation classifier restore incomplete: {missing_relation[:8]}")
        if initialized_base:
            raise ValueError(f"Unexpected frozen base initialization: {initialized_base[:8]}")
        print(
            "VisualRelationCheckpointLoader: "
            f"exact_base={exact_base}, mapped_relation={mapped_relation}, "
            f"initialized_tracker={len(initialized_tracker)}, examples={initialized_tracker[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class VisualRelationCombinedCheckpointLoader:
    """Combine the frozen visual decoder with a trained symbolic memory."""

    visual_params_path: str
    memory_params_path: str

    def load(self, params: at.Params) -> at.Params:
        visual_loaded = _model.restore_params(self.visual_params_path, restore_type=np.ndarray)
        memory_loaded = _model.restore_params(self.memory_params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        visual_source = flax.traverse_util.flatten_dict(visual_loaded, sep="/")
        memory_source = flax.traverse_util.flatten_dict(memory_loaded, sep="/")
        tracker_root = "HistoryThreeSwapVisualRelationMemoryTracker/"
        relation_root = tracker_root + "swap_relation_classifier/"
        visual_relation_root = "HistoryThreeSwapPairVisualClassifier/"
        memory_root = "HistoryThreeSwapOraclePairRecurrentMemoryTracker/"
        result = {}
        exact_base = mapped_relation = mapped_memory = 0
        missing = []

        for key, reference in target.items():
            candidate = None
            source_kind = ""
            if key.startswith(relation_root):
                relative = key.removeprefix(relation_root)
                if relative.startswith("semantic_encoder/"):
                    relative = relative.removeprefix("semantic_encoder/")
                candidate = visual_source.get(visual_relation_root + relative)
                source_kind = "relation"
            elif key.startswith(tracker_root):
                relative = key.removeprefix(tracker_root)
                if relative.startswith("shared_swap_memory_updater/"):
                    relative = "shared_segment_memory_updater/" + relative.removeprefix(
                        "shared_swap_memory_updater/"
                    )
                candidate = memory_source.get(memory_root + relative)
                source_kind = "memory"
            else:
                candidate = visual_source.get(key)
                source_kind = "base"

            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                if source_kind == "relation":
                    mapped_relation += 1
                elif source_kind == "memory":
                    mapped_memory += 1
                else:
                    exact_base += 1
            else:
                result[key] = reference
                missing.append(key)

        if missing:
            raise ValueError(f"Combined checkpoint restore incomplete: {missing[:8]}")
        print(
            "VisualRelationCombinedCheckpointLoader: "
            f"exact_base={exact_base}, mapped_relation={mapped_relation}, "
            f"mapped_memory={mapped_memory}, missing=0"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


_BASE_EVAL_STEP = _trainer.eval_step


def relation_eval_step(
    config,
    rng,
    state: training_utils.TrainState,
    batch,
    *,
    class_labels_by_episode=None,
):
    """Report ball tracking and frozen visual-relation accuracy together."""
    result = _BASE_EVAL_STEP(
        config,
        rng,
        state,
        batch,
        class_labels_by_episode=class_labels_by_episode,
    )
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, _ = batch
    joint_logits, aux = model.compute_history_classification(rng, observation, train=False)
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    safe_episode = jnp.clip(episode_index, 0, len(config.model.oracle_swap_pairs) - 1)
    true_relations = jnp.asarray(config.model.oracle_swap_pairs, dtype=jnp.int32)[safe_episode]
    for stage in range(3):
        result[f"val/relation_{stage}_accuracy"] = jnp.mean(
            aux["relation_ids"][:, stage] == true_relations[:, stage]
        )

    labels = class_labels_by_episode[safe_episode]
    predictions = jnp.argmax(joint_logits, axis=-1)
    pred_stages = jnp.stack(
        (predictions // 9, (predictions // 3) % 3, predictions % 3), axis=-1
    )
    label_stages = jnp.stack((labels // 9, (labels // 3) % 3, labels % 3), axis=-1)
    for stage in range(3):
        result[f"val/swap_{stage}_accuracy"] = jnp.mean(
            pred_stages[:, stage] == label_stages[:, stage]
        )
    return result


def run_self_test() -> None:
    tracker = ThreeSwapVisualRelationMemoryTracker(
        num_frames=_semantic.HISTORY_FRAMES,
        input_width=16,
        encoder_width=16,
        encoder_depth=2,
        encoder_heads=4,
        memory_width=16,
        memory_depth=2,
        memory_heads=4,
        adapter_heads=4,
        num_memory_tokens=8,
        num_current_tokens=8,
        current_width=32,
        residual_scale=1.0,
        relation_mode="probabilities",
        dtype_mm="float32",
    )
    base = jax.random.normal(jax.random.key(1), (2, _semantic.HISTORY_FRAMES, 256, 16))
    slots = jnp.asarray((0, 1), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), base, slots)
    _, reference, _, relation_reference, _ = tracker.apply(variables, base, slots)

    causal = []
    relation_effects = []
    for stage_index, (start, end) in enumerate(_semantic.SWAP_SLICES):
        changed = base.at[:, start:end].set(
            jax.random.normal(jax.random.key(10 + stage_index), base[:, start:end].shape)
        )
        _, candidate, _, relation_candidate, _ = tracker.apply(variables, changed, slots)
        causal.append(
            bool(
                np.allclose(
                    np.asarray(reference[:, :stage_index]),
                    np.asarray(candidate[:, :stage_index]),
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
        relation_effects.append(
            not np.allclose(
                np.asarray(relation_reference[:, stage_index]),
                np.asarray(relation_candidate[:, stage_index]),
            )
        )

    outside = base.at[:, : _semantic.SWAP_SLICES[0][0]].set(99.0)
    outside = outside.at[:, _semantic.SWAP_SLICES[-1][1] :].set(-99.0)
    _, outside_candidate, _, _, _ = tracker.apply(variables, outside, slots)
    outside_ignored = np.allclose(
        np.asarray(reference), np.asarray(outside_candidate), rtol=0.0, atol=0.0
    )
    _, initial_candidate, _, _, _ = tracker.apply(variables, base, (slots + 1) % 3)
    initial_effect = not np.allclose(np.asarray(reference), np.asarray(initial_candidate))
    if not all(causal) or not all(relation_effects) or not outside_ignored or not initial_effect:
        raise AssertionError(
            "Visual relation self-test failed: "
            f"causal={causal}, relation={relation_effects}, outside={outside_ignored}, "
            f"initial={initial_effect}"
        )
    print(
        "Visual relation self-test passed: "
        f"causal={causal}, relation_effects={relation_effects}, "
        f"outside_ignored={outside_ignored}, initial_effect={initial_effect}"
    )


def build_config(args: argparse.Namespace, labels_path: pathlib.Path):
    base_config = _semantic.build_config(args, labels_path)
    parent_model = base_config.model
    parent_fields = {
        field.name: getattr(parent_model, field.name) for field in dataclasses.fields(parent_model)
    }
    model = VisualRelationMemoryConfig(
        **parent_fields,
        relation_mode=args.relation_mode,
        oracle_swap_pairs=_oracle_pair.build_swap_pair_lookup(),
    )
    weight_loader = VisualRelationCheckpointLoader(args.init_checkpoint)
    if args.memory_checkpoint:
        weight_loader = VisualRelationCombinedCheckpointLoader(
            visual_params_path=args.init_checkpoint,
            memory_params_path=args.memory_checkpoint,
        )
    return dataclasses.replace(
        base_config,
        name="pi0_shellgame_three_swap_visual_relation_onehot_memory_260809",
        model=model,
        freeze_filter=model.get_freeze_filter_visual_relation_tracker(),
        weight_loader=weight_loader,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=_semantic.DEFAULT_PAIR_CHECKPOINT)
    parser.add_argument("--memory-checkpoint", default="")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
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
    parser.add_argument("--video-mode", choices=("normal", "shuffle_swaps", "zero_swaps"), default="normal")
    parser.add_argument("--relation-mode", choices=("one_hot", "probabilities", "logits"), default="one_hot")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    _trainer.eval_step = relation_eval_step
    _trainer.main(build_config(args, build_three_swap_labels()))


if __name__ == "__main__":
    main()
