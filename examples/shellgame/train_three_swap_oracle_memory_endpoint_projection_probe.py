"""Add the original 1152-D endpoint path to the successful raw-initial M128 probe.

The recurrent state remains M=128, width=64, float32, with sparse raw initial
slot injection and the original depth-2 attention updater.  Only the endpoint
readout changes: memory is normalized, projected from 64 to 1152, normalized
again, and consumed by IntegratedCurrentReadout.  Token-axis mean subtraction
is deliberately omitted because it was already ruled out independently.
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

from examples.shellgame import train_three_swap_oracle_memory_token_raw_initial_probe as _raw
from examples.shellgame import train_three_swap_oracle_memory_token_sweep_probe as _sweep
from examples.shellgame import train_three_swap_oracle_minimal_transition_probe as _minimal
from examples.shellgame import train_three_swap_oracle_pair_recurrent_memory_probe as _oracle_pair
from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import IntegratedCurrentReadout
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import multistage_eval_step
from examples.shellgame.train_three_swap_pair_fixed_grid_probe import SWAP_PAIRS
from examples.shellgame.train_three_swap_recurrent_memory_fixed_grid_probe import SharedSegmentMemoryUpdater
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
from openpi.training import config as _config
from scripts.mem import train_pi0_mem_compress as _trainer


class EndpointProjectionOracleMemoryTracker(nn.Module):
    """Raw-initial M128 tracker with the 64-to-1152 endpoint path restored."""

    width: int = 64
    output_width: int = 1152
    depth: int = 2
    num_heads: int = 4
    num_memory_tokens: int = 128

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
                _sweep.SEGMENT_SIZE,
                _sweep.SPATIAL_TOKENS,
                self.width,
            ),
            dtype=jnp.float32,
        )
        segment_tokens = segment_tokens.at[..., : len(SWAP_PAIRS)].add(pair_codes[:, :, None, None, :])

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.width),
            jnp.float32,
        )
        memory = jnp.tile(base_memory, (batch_size, 1, 1))
        initial_code = jax.nn.one_hot(initial_slots, 3, dtype=jnp.float32)
        memory = memory.at[:, 0, :3].add(initial_code)

        updater = SharedSegmentMemoryUpdater(
            name="shared_segment_memory_updater",
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=_sweep.SEGMENT_SIZE,
            dtype_mm="float32",
        )
        endpoint_memories = []
        for stage_index in range(_oracle_pair.NUM_SWAP_SEGMENTS):
            memory = updater(memory, segment_tokens[:, stage_index])
            endpoint_memories.append(memory)

        memory_batch = jnp.stack(endpoint_memories, axis=1).reshape(
            batch_size * _oracle_pair.NUM_SWAP_SEGMENTS,
            self.num_memory_tokens,
            self.width,
        )
        memory_batch = nn.LayerNorm(name="memory_output_ln", dtype=jnp.float32)(memory_batch)
        memory_batch = nn.Dense(
            self.output_width,
            name="memory_output_projection",
            dtype=jnp.float32,
        )(memory_batch)
        memory_batch = nn.LayerNorm(name="pi0_output_ln", dtype=jnp.float32)(memory_batch)
        logits = IntegratedCurrentReadout(
            name="shared_readout",
            input_width=self.output_width,
            width=self.width,
            num_classes=3,
            dtype_mm="float32",
        )(memory_batch)

        stage_logits = logits.reshape(batch_size, _oracle_pair.NUM_SWAP_SEGMENTS, 3)
        stage_memories = memory_batch.reshape(
            batch_size,
            _oracle_pair.NUM_SWAP_SEGMENTS,
            self.num_memory_tokens,
            self.output_width,
        )
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
class EndpointProjectionOracleMemoryConfig(_raw.RawInitialOracleMemoryTokenConfig):
    endpoint_output_width: int = 1152

    def create(self, rng: at.KeyArrayLike) -> EndpointProjectionOracleMemoryModel:
        return EndpointProjectionOracleMemoryModel(self, rngs=nnx.Rngs(rng))


class EndpointProjectionOracleMemoryModel(_oracle_pair.OraclePairProbeModel):
    def __init__(self, config: EndpointProjectionOracleMemoryConfig, rngs: nnx.Rngs):
        _base_model.Pi0MemCompress.__init__(self, config, rngs)
        self.oracle_initial_slots = config.oracle_initial_slots
        self.oracle_swap_pairs = config.oracle_swap_pairs
        self.pair_mode = config.pair_mode
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker = nnx_bridge.ToNNX(
            EndpointProjectionOracleMemoryTracker(
                width=config.memory_token_width,
                output_width=config.endpoint_output_width,
                depth=config.memory_token_depth,
                num_heads=config.memory_token_heads,
                num_memory_tokens=config.diagnostic_memory_tokens,
            )
        )
        fake_slots = jnp.zeros((1,), dtype=jnp.int32)
        fake_pairs = jnp.zeros((1, _oracle_pair.NUM_SWAP_SEGMENTS), dtype=jnp.int32)
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker.lazy_init(fake_slots, fake_pairs, rngs=rngs)


def run_self_test() -> None:
    tracker = EndpointProjectionOracleMemoryTracker(
        width=16,
        output_width=32,
        depth=2,
        num_heads=4,
        num_memory_tokens=8,
    )
    slots = jnp.asarray((0, 1), dtype=jnp.int32)
    pairs = jnp.asarray(((0, 1, 2), (2, 1, 0)), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), slots, pairs)
    _, reference, memories = tracker.apply(variables, slots, pairs)
    if memories.shape != (2, 3, 8, 32):
        raise AssertionError(f"Unexpected memory shape: {memories.shape}")
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
            f"Endpoint self-test failed: causal={causal}, pair={pair_effects}, initial={initial_effect}"
        )
    print(
        "Endpoint projection self-test passed: "
        f"causal={causal}, pair_effects={pair_effects}, initial_effect={initial_effect}"
    )


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    base_config = _raw.build_config(args, labels_path)
    parent_model = base_config.model
    parent_fields = {field.name: getattr(parent_model, field.name) for field in dataclasses.fields(parent_model)}
    model = EndpointProjectionOracleMemoryConfig(
        **parent_fields,
        endpoint_output_width=args.endpoint_output_width,
    )
    return dataclasses.replace(
        base_config,
        name="pi0_shellgame_three_swap_oracle_memory_endpoint_projection_260809",
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
    parser.add_argument("--memory-tokens", type=int, default=128)
    parser.add_argument("--memory-width", type=int, default=64)
    parser.add_argument("--memory-depth", type=int, default=2)
    parser.add_argument("--memory-heads", type=int, default=4)
    parser.add_argument("--endpoint-output-width", type=int, default=1152)
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
