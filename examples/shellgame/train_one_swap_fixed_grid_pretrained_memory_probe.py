"""Probe final memory capacity with a pretrained fixed-grid temporal encoder.

The fixed-grid temporal path is initialized from the one-swap checkpoint that
reached 100% validation accuracy and is frozen.  Only the final 128-token Pi0
memory compressor and its classifier are trained.  This distinguishes a true
memory-capacity failure from failed end-to-end optimization through a randomly
initialized memory bottleneck.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.traverse_util
import flax.nnx as nnx
import numpy as np

from examples.shellgame.train_one_swap_fixed_grid_temporal_memory_probe import (
    build_config as build_end_to_end_config,
)
from examples.shellgame.train_one_swap_fixed_grid_temporal_memory_probe import (
    parse_args,
)
from examples.shellgame.train_one_swap_history_probe import build_one_swap_labels
from examples.shellgame.train_one_swap_history_probe import SOURCE_CHECKPOINT
from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import weight_loaders
from scripts.mem import train_pi0_mem_compress as _trainer


PRETRAINED_TEMPORAL_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_one_swap_fixed_grid_temporal_probe_260808/"
    "one_swap_fixed_grid_k64_260808/499/params"
)


@dataclasses.dataclass(frozen=True)
class PretrainedFixedGridMemoryCheckpointLoader:
    """Restore Pi0 plus the successful fixed-grid temporal probe weights."""

    base_params_path: str
    temporal_params_path: str

    def load(self, params: at.Params) -> at.Params:
        base_params = _model.restore_params(
            self.base_params_path,
            restore_type=np.ndarray,
        )
        temporal_params = _model.restore_params(
            self.temporal_params_path,
            restore_type=np.ndarray,
        )
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(base_params, sep="/")
        flat_temporal = flax.traverse_util.flatten_dict(temporal_params, sep="/")

        source_prefix = "HistoryFixedGridTemporalProbe/"
        target_prefix = "HistoryFixedGridTemporalMemory/"
        shared_roots = (
            "input_ln/",
            "input_projection/",
            "temporal_pos_embedding",
            "temporal_block_0/",
            "temporal_block_1/",
        )
        mapped_count = 0
        for target_key in flat_ref:
            if not target_key.startswith(target_prefix):
                continue
            relative = target_key[len(target_prefix) :]
            if not relative.startswith(shared_roots):
                continue
            source_relative = relative
            if relative.startswith("temporal_block_"):
                source_relative = relative.replace("temporal_block_", "block_", 1)
            source_key = source_prefix + source_relative
            if source_key not in flat_temporal:
                raise KeyError(
                    f"Missing pretrained temporal weight {source_key} for {target_key}"
                )
            flat_loaded[target_key] = flat_temporal[source_key]
            mapped_count += 1

        expected_shared = [
            key
            for key in flat_ref
            if key.startswith(target_prefix)
            and key[len(target_prefix) :].startswith(shared_roots)
        ]
        if mapped_count != len(expected_shared):
            raise ValueError(
                f"Mapped {mapped_count}/{len(expected_shared)} temporal weights"
            )

        loaded = flax.traverse_util.unflatten_dict(flat_loaded, sep="/")
        return weight_loaders._merge_params(
            loaded,
            params,
            missing_regex=(
                r".*(lora|HistoryResampler|HistoryLayerNorm_0|"
                r"HistoryMultiHeadDotProductAttention_0|HistoryOutProj|"
                r"history_memory_gate_logit|HistoryClassifier|"
                r"HistoryFixedGridTemporalMemory/(final_memory_compressor|"
                r"readout_projection|readout_attention|readout_ln|classifier)).*"
            ),
        )


def main() -> None:
    args = parse_args()
    config = build_end_to_end_config(args, build_one_swap_labels())
    trainable_memory = nnx_utils.PathRegex(
        r".*HistoryFixedGridTemporalMemory/(final_memory_compressor|"
        r"readout_projection|readout_attention|readout_ln|classifier).*"
    )
    config = dataclasses.replace(
        config,
        name="pi0_shellgame_one_swap_fixed_grid_pretrained_memory_probe_260808",
        weight_loader=PretrainedFixedGridMemoryCheckpointLoader(
            SOURCE_CHECKPOINT,
            PRETRAINED_TEMPORAL_CHECKPOINT,
        ),
        freeze_filter=nnx.Not(trainable_memory),
    )
    _trainer.main(config)


if __name__ == "__main__":
    main()
