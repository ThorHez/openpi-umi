"""Train direct visual evidence against a frozen, proven recurrent memory.

The model is identical to ``train_direct_visual_recurrent_stage_slot_probe``:
there is no relation classifier, relation logit/probability/id, or relation
teacher forcing.  This control restores and freezes base memory, the recurrent
updater, adapter, and readout from the successful stage-slot-only relation
checkpoint.  Only the continuous visual segment encoder is random/trainable,
and three stage ball-slot cross-entropies remain the only task loss.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import flax.traverse_util
import numpy as np

from examples.shellgame import train_direct_visual_recurrent_stage_slot_probe as _direct
from openpi.models import model as _model
import openpi.shared.nnx_utils as nnx_utils
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as _recipe
from scripts.mem import train_semantic_memory as _trainer

DEFAULT_ANCHORED_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "shellgame_stage_slot_only_relation_recurrent_probe/"
    "stage_slot_only_random_relation_frozen_memory_1k_260821/500/params"
)


@dataclasses.dataclass(frozen=True)
class FrozenMemoryDirectVisualLoader:
    """Map a successful relation tracker into the relation-free tracker."""

    params_path: str

    def load(self, params):
        source = flax.traverse_util.flatten_dict(
            _model.restore_params(self.params_path, restore_type=np.ndarray), sep="/"
        )
        target = flax.traverse_util.flatten_dict(params, sep="/")
        target_tracker = "HistoryThreeSwapDirectVisualMemoryTracker/"
        target_encoder = target_tracker + "direct_visual_segment_encoder/"
        source_tracker = "HistoryThreeSwapVisualRelationMemoryTracker/"

        result = {}
        counts = {"base": 0, "memory": 0, "random_visual": 0}
        missing_base = []
        missing_memory = []
        for key, reference in target.items():
            if key.startswith(target_encoder):
                result[key] = reference
                counts["random_visual"] += 1
                continue

            candidate = None
            kind = "base"
            if key.startswith(target_tracker):
                relative = key.removeprefix(target_tracker)
                if relative.startswith("shared_visual_memory_updater/"):
                    relative = "shared_swap_memory_updater/" + relative.removeprefix("shared_visual_memory_updater/")
                candidate = source.get(source_tracker + relative)
                kind = "memory"
            else:
                candidate = source.get(key)

            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                counts[kind] += 1
            else:
                result[key] = reference
                if kind == "memory":
                    missing_memory.append(key)
                else:
                    missing_base.append(key)

        if missing_base:
            raise ValueError(f"Frozen base restore incomplete: {missing_base[:8]}")
        if missing_memory:
            raise ValueError(f"Frozen memory restore incomplete: {missing_memory[:8]}")
        print(
            "FrozenMemoryDirectVisualLoader: "
            + ", ".join(f"{key}={value}" for key, value in counts.items())
            + ", missing_base=0, missing_memory=0"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def build_config(args: argparse.Namespace):
    config = _direct.build_config(args)
    visual_interface = nnx_utils.PathRegex(
        r".*HistoryThreeSwapDirectVisualMemoryTracker/"
        r"direct_visual_segment_encoder.*"
    )
    return dataclasses.replace(
        config,
        name="shellgame_direct_visual_frozen_memory_stage_slot_probe",
        freeze_filter=nnx.Not(visual_interface),
        weight_loader=FrozenMemoryDirectVisualLoader(args.memory_checkpoint),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--memory-checkpoint", default=DEFAULT_ANCHORED_CHECKPOINT)
    # Kept because the shared direct config accepts it before this experiment
    # replaces the loader with ``FrozenMemoryDirectVisualLoader``.
    parser.add_argument("--init-checkpoint", default=DEFAULT_ANCHORED_CHECKPOINT)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _recipe.compute_objective = _direct.stage_slot_only_objective
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
