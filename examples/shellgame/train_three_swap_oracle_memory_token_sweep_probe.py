"""Sweep persistent memory-token count with the original recurrent updater.

Ground-truth initial slot and swap-pair inputs remove perception.  Each pair is
expanded to the same 10x64 segment used by preceding Oracle probes.  Persistent
memory tokens are updated by the original depth-2 cross-attention,
self-attention, and MLP updater.  A lightweight shared attention-pooling head
reads every swap endpoint.  The token count is configurable while every other
part of the diagnostic remains fixed.
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
import numpy as np

from examples.shellgame import train_three_swap_oracle_minimal_transition_probe as _minimal
from examples.shellgame import train_three_swap_oracle_pair_recurrent_memory_probe as _oracle_pair
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import multistage_eval_step
from examples.shellgame.train_three_swap_pair_fixed_grid_probe import SWAP_PAIRS
from examples.shellgame.train_three_swap_recurrent_memory_fixed_grid_probe import SharedSegmentMemoryUpdater
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
from openpi.training import config as _config
from scripts.mem import train_pi0_mem_compress as _trainer

SEGMENT_SIZE = 10
SPATIAL_TOKENS = 64


class SharedMemoryTokenReadout(nn.Module):
    """Read one slot prediction from any number of memory tokens."""

    width: int = 64

    @nn.compact
    def __call__(self, memory):
        x = nn.LayerNorm(name="input_ln", dtype=jnp.float32)(memory)
        scores = nn.Dense(1, name="attention", dtype=jnp.float32)(x)
        weights = nn.softmax(scores, axis=1)
        pooled = jnp.sum(weights * x, axis=1)
        pooled = nn.LayerNorm(name="pooled_ln", dtype=jnp.float32)(pooled)
        return nn.Dense(3, name="classifier", dtype=jnp.float32)(pooled)


class OracleMemoryTokenTracker(nn.Module):
    """Apply the original attention updater to a configurable token state."""

    width: int = 64
    depth: int = 2
    num_heads: int = 4
    num_memory_tokens: int = 1

    @nn.compact
    def __call__(self, initial_slots, swap_pair_ids):
        if initial_slots.ndim != 1:
            raise ValueError(f"Expected initial slots [B], got {initial_slots.shape}")
        batch_size = initial_slots.shape[0]
        expected = (batch_size, _oracle_pair.NUM_SWAP_SEGMENTS)
        if swap_pair_ids.shape != expected:
            raise ValueError(f"Expected swap pairs {expected}, got {swap_pair_ids.shape}")

        pair_codes = jax.nn.one_hot(swap_pair_ids, len(SWAP_PAIRS), dtype=jnp.float32)
        segment_tokens = jnp.zeros(
            (
                batch_size,
                _oracle_pair.NUM_SWAP_SEGMENTS,
                SEGMENT_SIZE,
                SPATIAL_TOKENS,
                self.width,
            ),
            dtype=jnp.float32,
        )
        segment_tokens = segment_tokens.at[..., : len(SWAP_PAIRS)].add(pair_codes[:, :, None, None, :])

        initial_code = jax.nn.one_hot(initial_slots, 3, dtype=jnp.float32)
        initial_state = nn.Dense(self.width, name="initial_projection", dtype=jnp.float32)(initial_code)
        initial_state = nn.tanh(initial_state)
        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.width),
            jnp.float32,
        )
        memory = jnp.tile(base_memory, (batch_size, 1, 1))
        memory = memory.at[:, 0].add(initial_state)

        updater = SharedSegmentMemoryUpdater(
            name="shared_segment_memory_updater",
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=SEGMENT_SIZE,
            dtype_mm="float32",
        )
        readout = SharedMemoryTokenReadout(name="shared_readout", width=self.width)
        stage_logits = []
        stage_memories = []
        for stage_index in range(_oracle_pair.NUM_SWAP_SEGMENTS):
            memory = updater(memory, segment_tokens[:, stage_index])
            stage_memories.append(memory)
            stage_logits.append(readout(memory))

        stage_logits = jnp.stack(stage_logits, axis=1)
        stage_memories = jnp.stack(stage_memories, axis=1)
        logits_0, logits_1, logits_2 = (
            stage_logits[:, 0],
            stage_logits[:, 1],
            stage_logits[:, 2],
        )
        joint_logits = (logits_0[:, :, None, None] + logits_1[:, None, :, None] + logits_2[:, None, None, :]).reshape(
            batch_size, 27
        )
        return joint_logits, stage_logits, stage_memories


@dataclasses.dataclass(frozen=True)
class OracleMemoryTokenConfig(_minimal.MinimalOracleTransitionConfig):
    memory_token_width: int = 64
    memory_token_depth: int = 2
    memory_token_heads: int = 4
    diagnostic_memory_tokens: int = 1

    def create(self, rng: at.KeyArrayLike) -> OracleMemoryTokenModel:
        return OracleMemoryTokenModel(self, rngs=nnx.Rngs(rng))


class OracleMemoryTokenModel(_oracle_pair.OraclePairProbeModel):
    def __init__(self, config: OracleMemoryTokenConfig, rngs: nnx.Rngs):
        _base_model.Pi0MemCompress.__init__(self, config, rngs)
        self.oracle_initial_slots = config.oracle_initial_slots
        self.oracle_swap_pairs = config.oracle_swap_pairs
        self.pair_mode = config.pair_mode
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker = nnx_bridge.ToNNX(
            OracleMemoryTokenTracker(
                width=config.memory_token_width,
                depth=config.memory_token_depth,
                num_heads=config.memory_token_heads,
                num_memory_tokens=config.diagnostic_memory_tokens,
            )
        )
        fake_slots = jnp.zeros((1,), dtype=jnp.int32)
        fake_pairs = jnp.zeros((1, _oracle_pair.NUM_SWAP_SEGMENTS), dtype=jnp.int32)
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker.lazy_init(fake_slots, fake_pairs, rngs=rngs)


def run_self_test() -> None:
    for num_tokens in (1, 8):
        tracker = OracleMemoryTokenTracker(width=16, depth=2, num_heads=4, num_memory_tokens=num_tokens)
        slots = jnp.asarray((0, 1), dtype=jnp.int32)
        pairs = jnp.asarray(((0, 1, 2), (2, 1, 0)), dtype=jnp.int32)
        variables = tracker.init(jax.random.key(num_tokens), slots, pairs)
        _, reference, memories = tracker.apply(variables, slots, pairs)
        expected_memory_shape = (2, 3, num_tokens, 16)
        if memories.shape != expected_memory_shape:
            raise AssertionError(f"M={num_tokens}: expected {expected_memory_shape}, got {memories.shape}")
        causal = []
        pair_effects = []
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
            pair_effects.append(
                not np.allclose(np.asarray(reference[:, changed_stage]), np.asarray(candidate[:, changed_stage]))
            )
        _, changed_initial, _ = tracker.apply(variables, (slots + 1) % 3, pairs)
        initial_effect = not np.allclose(np.asarray(reference), np.asarray(changed_initial))
        if not all(causal) or not all(pair_effects) or not initial_effect:
            raise AssertionError(
                f"M={num_tokens} self-test failed: causal={causal}, pair={pair_effects}, initial={initial_effect}"
            )
        print(
            f"M={num_tokens} self-test passed: causal={causal}, "
            f"pair_effects={pair_effects}, initial_effect={initial_effect}"
        )


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    base_config = _minimal.build_config(args, labels_path)
    parent_model = base_config.model
    parent_fields = {field.name: getattr(parent_model, field.name) for field in dataclasses.fields(parent_model)}
    model = OracleMemoryTokenConfig(
        **parent_fields,
        memory_token_width=args.memory_width,
        memory_token_depth=args.memory_depth,
        memory_token_heads=args.memory_heads,
        diagnostic_memory_tokens=args.memory_tokens,
    )
    return dataclasses.replace(
        base_config,
        name="pi0_shellgame_three_swap_oracle_memory_token_sweep_260809",
        model=model,
        freeze_filter=model.get_freeze_filter_tracker_only(),
        weight_loader=_minimal.MinimalTransitionCheckpointLoader(args.init_checkpoint),
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
    parser.add_argument("--memory-tokens", type=int, required=True)
    parser.add_argument("--memory-width", type=int, default=64)
    parser.add_argument("--memory-depth", type=int, default=2)
    parser.add_argument("--memory-heads", type=int, default=4)
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
