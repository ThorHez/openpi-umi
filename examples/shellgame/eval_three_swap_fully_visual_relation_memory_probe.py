"""Evaluate fully visual initial-state and three-swap tracking.

One frozen frame-0 classifier predicts the initial ball cup.  The complete
frozen swap-relation classifier predicts each of the three swapped cup pairs.
Those four discrete visual decisions drive the already trained compact
recurrent memory and single-read endpoint.  Episode metadata is used only by
the evaluator to score predictions, never as model input.  Action loss remains
disabled.
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
import jax.numpy as jnp
import numpy as np

from examples.shellgame import train_three_swap_visual_relation_onehot_memory_probe as _relation
from examples.shellgame import train_three_swap_visual_semantic_readout_memory_probe as _semantic
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import utils as training_utils
from scripts.mem import train_pi0_mem_compress as _trainer

DEFAULT_INITIAL_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_frame0_initial_cup_probe_260807/"
    "frame0_initial_cup_linear_260807/299/params"
)
DEFAULT_MEMORY_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_oracle_single_history_read_adapter_260809/"
    "oracle_single_read_for_visual_relation_600_b72_260810/599/params"
)


class FrozenFrame0InitialCupClassifier(nn.Module):
    """Exact spatially ordered linear head from the 100%-accurate probe."""

    input_width: int = 1152

    @nn.compact
    def __call__(self, patch_features):
        if patch_features.ndim != 3 or patch_features.shape[1:] != (256, self.input_width):
            raise ValueError(f"Expected frame-0 patches [B,256,{self.input_width}], got {patch_features.shape}")
        x = nn.LayerNorm(name="initial_ln", dtype=jnp.bfloat16)(patch_features)
        return nn.Dense(3, name="initial_head", dtype=jnp.bfloat16)(
            x.reshape(x.shape[0], -1)
        )


@dataclasses.dataclass(frozen=True)
class FullyVisualRelationMemoryConfig(_relation.VisualRelationMemoryConfig):
    initial_mode: str = "normal"

    def create(self, rng: at.KeyArrayLike) -> FullyVisualRelationMemoryModel:
        return FullyVisualRelationMemoryModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_fully_visual(self) -> nnx.filterlib.Filter:
        tracker = nnx_utils.PathRegex(r".*HistoryThreeSwapVisualRelationMemoryTracker.*")
        relation = nnx_utils.PathRegex(
            r".*HistoryThreeSwapVisualRelationMemoryTracker/swap_relation_classifier.*"
        )
        trainable = nnx.All(tracker, nnx.Not(relation))
        return nnx.Not(trainable)


class FullyVisualRelationMemoryModel(_base_model.Pi0MemCompress):
    def __init__(self, config: FullyVisualRelationMemoryConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.video_mode = config.video_mode
        self.initial_mode = config.initial_mode
        self.HistoryFrame0InitialCupClassifier = nnx_bridge.ToNNX(
            FrozenFrame0InitialCupClassifier(input_width=1152)
        )
        self.HistoryFrame0InitialCupClassifier.lazy_init(
            jnp.zeros((1, 256, 1152), dtype=jnp.bfloat16), rngs=rngs
        )
        self.HistoryThreeSwapVisualRelationMemoryTracker = nnx_bridge.ToNNX(
            _relation.ThreeSwapVisualRelationMemoryTracker(
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
        self.HistoryThreeSwapVisualRelationMemoryTracker.lazy_init(
            jnp.zeros((1, _semantic.HISTORY_FRAMES, 256, 1152), dtype=jnp.bfloat16),
            jnp.zeros((1,), dtype=jnp.int32),
            rngs=rngs,
        )

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
                f"Fully visual probe expects [B,{_semantic.TOTAL_INPUT_FRAMES},H,W,C], got {image.shape}"
            )

        # The memory SigLIP encoder returns fully encoded tokens only for its
        # configured current frame.  Run frame 0 separately so this input is
        # exactly the same as the original 100%-accurate one-frame probe.
        _, initial_encoder_out = self.PaliGemma.img(image[:, :1], train=False)
        frame0_features = initial_encoder_out["encoded"]
        if frame0_features.ndim != 3:
            raise ValueError(f"Expected frame-0 encoded patches [B,256,1152], got {frame0_features.shape}")
        if self.initial_mode == "shuffle_batch":
            frame0_features = jnp.roll(frame0_features, 1, axis=0)
        elif self.initial_mode == "zero":
            frame0_features = jnp.zeros_like(frame0_features)
        elif self.initial_mode != "normal":
            raise ValueError(f"Unknown initial_mode={self.initial_mode!r}")
        initial_logits = self.HistoryFrame0InitialCupClassifier(frame0_features)
        initial_ids = jnp.argmax(initial_logits, axis=-1)

        _, history_encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = history_encoder_out["with_posemb"][:, : _semantic.HISTORY_FRAMES]
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

        joint_logits, stage_logits, stage_memories, relation_logits, relation_ids = (
            self.HistoryThreeSwapVisualRelationMemoryTracker(history_patches, initial_ids)
        )
        return joint_logits, {
            "history_mem": stage_memories.reshape(-1, stage_memories.shape[-2], stage_memories.shape[-1]),
            "stage_logits": stage_logits,
            "initial_logits": initial_logits,
            "initial_ids": initial_ids,
            "relation_logits": relation_logits,
            "relation_ids": relation_ids,
            "encoder_auxes": (),
        }


@dataclasses.dataclass(frozen=True)
class FullyVisualCombinedCheckpointLoader:
    """Restore base, initial decoder, relation decoder, and trained memory."""

    visual_params_path: str
    initial_params_path: str
    memory_params_path: str

    def load(self, params: at.Params) -> at.Params:
        visual_loaded = _model.restore_params(self.visual_params_path, restore_type=np.ndarray)
        initial_loaded = _model.restore_params(self.initial_params_path, restore_type=np.ndarray)
        memory_loaded = _model.restore_params(self.memory_params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        visual_source = flax.traverse_util.flatten_dict(visual_loaded, sep="/")
        initial_source = flax.traverse_util.flatten_dict(initial_loaded, sep="/")
        memory_source = flax.traverse_util.flatten_dict(memory_loaded, sep="/")

        initial_root = "HistoryFrame0InitialCupClassifier/"
        tracker_root = "HistoryThreeSwapVisualRelationMemoryTracker/"
        relation_root = tracker_root + "swap_relation_classifier/"
        visual_relation_root = "HistoryThreeSwapPairVisualClassifier/"
        memory_root = "HistoryThreeSwapOraclePairRecurrentMemoryTracker/"
        initial_name_map = {
            "initial_ln/bias": "HistoryClassifierNorm/bias",
            "initial_ln/scale": "HistoryClassifierNorm/scale",
            "initial_head/bias": "HistoryClassifierHead/bias",
            "initial_head/kernel": "HistoryClassifierHead/kernel",
        }
        result = {}
        counts = {"base": 0, "initial": 0, "relation": 0, "memory": 0}
        missing = []

        for key, reference in target.items():
            candidate = None
            source_kind = "base"
            if key.startswith(initial_root):
                relative = key.removeprefix(initial_root)
                source_key = initial_name_map.get(relative)
                candidate = initial_source.get(source_key) if source_key else None
                source_kind = "initial"
            elif key.startswith(relation_root):
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

            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                counts[source_kind] += 1
            else:
                result[key] = reference
                missing.append(key)

        if missing:
            raise ValueError(f"Fully visual restore incomplete: {missing[:8]}")
        print(
            "FullyVisualCombinedCheckpointLoader: "
            f"base={counts['base']}, initial={counts['initial']}, "
            f"relation={counts['relation']}, memory={counts['memory']}, missing=0"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


_BASE_EVAL_STEP = _trainer.eval_step


def fully_visual_eval_step(
    config,
    rng,
    state: training_utils.TrainState,
    batch,
    *,
    class_labels_by_episode=None,
):
    """Score initial state, visual relations, and recurrent ball tracking."""
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
    safe_episode = jnp.clip(episode_index, 0, len(config.model.oracle_initial_slots) - 1)
    true_initial = jnp.asarray(config.model.oracle_initial_slots, dtype=jnp.int32)[safe_episode]
    true_relations = jnp.asarray(config.model.oracle_swap_pairs, dtype=jnp.int32)[safe_episode]
    result["val/initial_accuracy"] = jnp.mean(aux["initial_ids"] == true_initial)
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


def build_config(args: argparse.Namespace, labels_path: pathlib.Path):
    relation_config = _relation.build_config(args, labels_path)
    parent_model = relation_config.model
    parent_fields = {
        field.name: getattr(parent_model, field.name) for field in dataclasses.fields(parent_model)
    }
    model = FullyVisualRelationMemoryConfig(**parent_fields, initial_mode=args.initial_mode)
    return dataclasses.replace(
        relation_config,
        name="pi0_shellgame_three_swap_fully_visual_relation_memory_260810",
        model=model,
        freeze_filter=model.get_freeze_filter_fully_visual(),
        weight_loader=FullyVisualCombinedCheckpointLoader(
            visual_params_path=args.init_checkpoint,
            initial_params_path=args.initial_checkpoint,
            memory_params_path=args.memory_checkpoint,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=_semantic.DEFAULT_PAIR_CHECKPOINT)
    parser.add_argument("--initial-checkpoint", default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument("--memory-checkpoint", default=DEFAULT_MEMORY_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--peak-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=9)
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
    parser.add_argument("--initial-mode", choices=("normal", "shuffle_batch", "zero"), default="normal")
    parser.add_argument("--relation-mode", choices=("one_hot", "probabilities", "logits"), default="one_hot")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _trainer.eval_step = fully_visual_eval_step
    _trainer.main(build_config(args, build_three_swap_labels()))


if __name__ == "__main__":
    main()
