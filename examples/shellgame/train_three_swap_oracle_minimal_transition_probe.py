"""Validate the ShellGame label pipeline with a minimal learned transition cell.

The model receives only the exact initial ball slot and the exact swapped cup
pair for each of three swaps.  A small shared residual MLP updates one recurrent
state per swap and a shared classifier predicts the ball slot after every
update.  Images and action loss are unused.  This isolates the data lookup,
labels, loss, and optimizer from the M128 attention-based memory updater.
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

from examples.shellgame import train_three_swap_oracle_pair_recurrent_memory_probe as _oracle_pair
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import multistage_eval_step
from examples.shellgame.train_three_swap_pair_fixed_grid_probe import SWAP_PAIRS
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
from openpi.training import config as _config
from scripts.mem import train_pi0_mem_compress as _trainer


class ResidualOracleTransitionCell(nn.Module):
    """One shared learned update conditioned on the currently swapped pair."""

    width: int = 64
    depth: int = 2

    @nn.compact
    def __call__(self, state, pair_code):
        x = jnp.concatenate((state, pair_code.astype(state.dtype)), axis=-1)
        x = nn.LayerNorm(name="input_ln", dtype=jnp.float32)(x)
        for layer_index in range(self.depth):
            x = nn.Dense(self.width * 2, name=f"hidden_{layer_index}", dtype=jnp.float32)(x)
            x = nn.gelu(x)
        update = nn.Dense(self.width, name="output", dtype=jnp.float32)(x)
        return nn.LayerNorm(name="state_ln", dtype=jnp.float32)(state + update)


class MinimalOracleTransitionTracker(nn.Module):
    """Apply the same small transition cell to all three Oracle swap pairs."""

    width: int = 64
    depth: int = 2

    @nn.compact
    def __call__(self, initial_slots, swap_pair_ids):
        if initial_slots.ndim != 1:
            raise ValueError(f"Expected initial slots [B], got {initial_slots.shape}")
        batch_size = initial_slots.shape[0]
        expected = (batch_size, _oracle_pair.NUM_SWAP_SEGMENTS)
        if swap_pair_ids.shape != expected:
            raise ValueError(f"Expected swap pairs {expected}, got {swap_pair_ids.shape}")

        initial_code = jax.nn.one_hot(initial_slots, 3, dtype=jnp.float32)
        pair_codes = jax.nn.one_hot(swap_pair_ids, len(SWAP_PAIRS), dtype=jnp.float32)
        state = nn.Dense(self.width, name="initial_projection", dtype=jnp.float32)(initial_code)
        state = nn.tanh(state)

        transition = ResidualOracleTransitionCell(
            name="shared_transition",
            width=self.width,
            depth=self.depth,
        )
        classifier = nn.Dense(3, name="shared_classifier", dtype=jnp.float32)
        stage_logits = []
        stage_states = []
        for stage_index in range(_oracle_pair.NUM_SWAP_SEGMENTS):
            state = transition(state, pair_codes[:, stage_index])
            stage_states.append(state)
            stage_logits.append(classifier(state))

        stage_logits = jnp.stack(stage_logits, axis=1)
        stage_states = jnp.stack(stage_states, axis=1)[:, :, None, :]
        logits_0, logits_1, logits_2 = (
            stage_logits[:, 0],
            stage_logits[:, 1],
            stage_logits[:, 2],
        )
        joint_logits = (logits_0[:, :, None, None] + logits_1[:, None, :, None] + logits_2[:, None, None, :]).reshape(
            batch_size, 27
        )
        return joint_logits, stage_logits, stage_states


@dataclasses.dataclass(frozen=True)
class MinimalOracleTransitionConfig(_oracle_pair.OraclePairProbeConfig):
    transition_width: int = 64
    transition_depth: int = 2

    def create(self, rng: at.KeyArrayLike) -> MinimalOracleTransitionModel:
        return MinimalOracleTransitionModel(self, rngs=nnx.Rngs(rng))


class MinimalOracleTransitionModel(_oracle_pair.OraclePairProbeModel):
    def __init__(self, config: MinimalOracleTransitionConfig, rngs: nnx.Rngs):
        _base_model.Pi0MemCompress.__init__(self, config, rngs)
        self.oracle_initial_slots = config.oracle_initial_slots
        self.oracle_swap_pairs = config.oracle_swap_pairs
        self.pair_mode = config.pair_mode
        # Retain the old attribute name so the parent's compute path and freeze
        # filter remain identical to the M128 Oracle-pair experiment.
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker = nnx_bridge.ToNNX(
            MinimalOracleTransitionTracker(
                width=config.transition_width,
                depth=config.transition_depth,
            )
        )
        fake_slots = jnp.zeros((1,), dtype=jnp.int32)
        fake_pairs = jnp.zeros((1, _oracle_pair.NUM_SWAP_SEGMENTS), dtype=jnp.int32)
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker.lazy_init(fake_slots, fake_pairs, rngs=rngs)


@dataclasses.dataclass(frozen=True)
class MinimalTransitionCheckpointLoader:
    """Load frozen Pi0 leaves exactly and initialize only the tiny tracker."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        tracker_root = "HistoryThreeSwapOraclePairRecurrentMemoryTracker/"
        result = {}
        exact_base = 0
        exact_tracker = 0
        initialized_tracker = []
        initialized_base = []
        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                if key.startswith(tracker_root):
                    exact_tracker += 1
                else:
                    exact_base += 1
            else:
                result[key] = reference
                if key.startswith(tracker_root):
                    initialized_tracker.append(key)
                else:
                    initialized_base.append(key)
        if initialized_base:
            raise ValueError(f"Unexpected randomly initialized frozen leaves: {initialized_base[:8]}")
        print(
            "MinimalTransitionCheckpointLoader: "
            f"exact_base={exact_base}, exact_tracker={exact_tracker}, "
            f"initialized_tracker={len(initialized_tracker)}, "
            f"examples={initialized_tracker[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def run_self_test() -> None:
    tracker = MinimalOracleTransitionTracker(width=16, depth=2)
    slots = jnp.asarray((0, 1), dtype=jnp.int32)
    pairs = jnp.asarray(((0, 1, 2), (2, 1, 0)), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), slots, pairs)
    _, reference, _ = tracker.apply(variables, slots, pairs)
    causal = []
    for changed_stage in range(_oracle_pair.NUM_SWAP_SEGMENTS):
        changed = pairs.at[:, changed_stage].set((pairs[:, changed_stage] + 1) % len(SWAP_PAIRS))
        _, candidate, _ = tracker.apply(variables, slots, changed)
        causal.append(
            bool(
                np.allclose(
                    np.asarray(reference[:, :changed_stage]),
                    np.asarray(candidate[:, :changed_stage]),
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
    _, changed_initial, _ = tracker.apply(variables, (slots + 1) % 3, pairs)
    initial_effect = not np.allclose(np.asarray(reference), np.asarray(changed_initial))
    if not all(causal) or not initial_effect:
        raise AssertionError(f"Minimal transition self-test failed: causal={causal}, initial={initial_effect}")
    print(f"Minimal transition self-test passed: causal={causal}, initial_effect={initial_effect}")


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    base_config = _oracle_pair.build_config(args, labels_path)
    parent_model = base_config.model
    parent_fields = {field.name: getattr(parent_model, field.name) for field in dataclasses.fields(parent_model)}
    model = MinimalOracleTransitionConfig(
        **parent_fields,
        transition_width=args.transition_width,
        transition_depth=args.transition_depth,
    )
    return dataclasses.replace(
        base_config,
        name="pi0_shellgame_three_swap_oracle_minimal_transition_260809",
        model=model,
        freeze_filter=model.get_freeze_filter_tracker_only(),
        weight_loader=MinimalTransitionCheckpointLoader(args.init_checkpoint),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-checkpoint", default=_oracle_pair.DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--peak-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=9)
    parser.add_argument("--transition-width", type=int, default=64)
    parser.add_argument("--transition-depth", type=int, default=2)
    parser.add_argument("--pair-mode", choices=("correct", "roll", "shuffle_batch"), default="correct")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    _trainer.eval_step = multistage_eval_step
    _trainer.main(build_config(args, build_three_swap_labels()))


if __name__ == "__main__":
    main()
