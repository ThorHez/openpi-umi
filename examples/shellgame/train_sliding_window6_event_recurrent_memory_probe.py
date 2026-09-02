"""Six-frame sliding-window control for recurrent ShellGame memory.

This reuses the boundary-aware sliding-window experiment while changing only
the visual window length and its temporal labels.  Each six-frame positive is
fully contained inside one ten-frame swap phase.  Windows crossing adjacent
phases are hard ``no_event`` negatives.  Three selected relation events still
drive the same recurrent updater in chronological order.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.traverse_util
import numpy as np

from examples.shellgame import train_sliding_window_event_recurrent_memory_probe as _window
from openpi.models import model as _model
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as _recipe
from scripts.mem import train_semantic_memory as _trainer

WINDOW = 6
ALIGNED_STARTS = (22, 32, 42)
CROSS_BOUNDARY_STARTS = (25, 26, 27, 28, 29, 35, 36, 37, 38, 39)
STRICT_CROSS_STARTS = (27, 28, 29, 37, 38, 39)
STATIC_OR_PARTIAL_STARTS = (
    0,
    5,
    10,
    14,
    15,
    16,
    17,
    18,
    19,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
)

DEFAULT_WINDOW10_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "shellgame_sliding_window_event_recurrent_memory_probe/"
    "sliding_window_event_gate_500_260821/499/params"
)


def _configure_six_frame_globals() -> None:
    """Set the shared experimental implementation to the six-frame control."""
    _window.WINDOW = WINDOW
    _window.NUM_WINDOWS = _window._semantic.HISTORY_FRAMES - WINDOW + 1  # noqa: SLF001
    _window.ALIGNED_STARTS = ALIGNED_STARTS
    _window.CROSS_BOUNDARY_STARTS = CROSS_BOUNDARY_STARTS
    _window.STRICT_CROSS_STARTS = STRICT_CROSS_STARTS
    _window.STATIC_OR_PARTIAL_STARTS = STATIC_OR_PARTIAL_STARTS


@dataclasses.dataclass(frozen=True)
class SixFrameCheckpointLoader:
    """Restore the ten-frame model, center-cropping temporal embeddings to six."""

    params_path: str

    def load(self, params):
        source = flax.traverse_util.flatten_dict(
            _model.restore_params(self.params_path, restore_type=np.ndarray),
            sep="/",
        )
        target = flax.traverse_util.flatten_dict(params, sep="/")
        result = {}
        exact = cropped_temporal = 0
        missing = []
        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
                continue
            if (
                candidate is not None
                and key.endswith("relative_temporal_pos_embedding")
                and candidate.ndim == reference.ndim == 4
                and candidate.shape[0] == reference.shape[0] == 1
                and candidate.shape[1] == 10
                and reference.shape[1] == WINDOW
                and candidate.shape[2:] == reference.shape[2:]
            ):
                start = (candidate.shape[1] - WINDOW) // 2
                result[key] = np.asarray(
                    candidate[:, start : start + WINDOW],
                    dtype=np.dtype(reference.dtype),
                )
                cropped_temporal += 1
                continue
            result[key] = reference
            missing.append(key)
        if missing:
            raise ValueError(f"Six-frame checkpoint restore incomplete: {missing[:8]}")
        print(f"SixFrameCheckpointLoader: exact={exact}, cropped_temporal={cropped_temporal}, missing=0")
        return flax.traverse_util.unflatten_dict(result, sep="/")


def build_config(args: argparse.Namespace):
    config = _window.build_config(args)
    return dataclasses.replace(
        config,
        name="shellgame_sliding_window6_event_recurrent_memory_probe",
        weight_loader=SixFrameCheckpointLoader(args.init_checkpoint),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=DEFAULT_WINDOW10_CHECKPOINT)
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
    parser.add_argument("--hard-relation-codes", action="store_true")
    return parser.parse_args()


def main() -> None:
    _configure_six_frame_globals()
    args = parse_args()
    _window.RELATION_CODE_MODE = "one_hot" if args.hard_relation_codes else "probabilities"
    if args.self_test_only:
        _window.run_self_test()
        return
    _recipe.compute_objective = _window.sliding_window_objective
    _trainer.eval_step = _window.sliding_full_eval_step
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
