"""Validate one explicit history-read query for compact M128 memory.

A single learned width-64 query summarizes compact M128 history.  That semantic
token is normalized, projected once to 1152, normalized again, and broadcast as
a fixed-scale residual to 256 Pi0-like current tokens.  This avoids both the
old per-memory-token 64->1152->64 round trip and the failed 256-query by
128-memory double averaging.
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
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import multistage_eval_step
from examples.shellgame.train_three_swap_pair_fixed_grid_probe import SWAP_PAIRS
from examples.shellgame.train_three_swap_recurrent_memory_fixed_grid_probe import SharedSegmentMemoryUpdater
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
from openpi.training import config as _config
from scripts.mem import train_pi0_mem_compress as _trainer


class SingleHistoryReadAdapter(nn.Module):
    """Summarize compact history once and emit one normalized wide residual."""

    memory_width: int = 64
    current_width: int = 1152
    num_heads: int = 4
    residual_scale: float = 1.0

    @nn.compact
    def __call__(self, current_tokens, memory):
        read_query = self.param(
            "read_query",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.memory_width),
            jnp.float32,
        )
        read_query = jnp.tile(read_query, (memory.shape[0], 1, 1))
        query_norm = nn.LayerNorm(name="query_ln", dtype=jnp.float32)(read_query)
        memory_norm = nn.LayerNorm(name="memory_ln", dtype=jnp.float32)(memory)
        attended = nn.MultiHeadDotProductAttention(
            name="cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=jnp.float32,
        )(query_norm, memory_norm)
        semantic = nn.LayerNorm(name="semantic_ln", dtype=jnp.float32)(read_query + attended)
        residual = nn.Dense(
            self.current_width,
            name="residual_projection",
            dtype=jnp.float32,
        )(semantic)
        residual = nn.LayerNorm(name="residual_ln", dtype=jnp.float32)(residual)
        return current_tokens + self.residual_scale * residual


class SingleHistoryReadOracleTracker(nn.Module):
    """Expose raw-initial compact M128 history through one semantic read token."""

    memory_width: int = 64
    current_width: int = 1152
    memory_depth: int = 2
    memory_heads: int = 4
    adapter_heads: int = 4
    num_memory_tokens: int = 128
    num_current_tokens: int = 256
    residual_scale: float = 1.0

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
                self.memory_width,
            ),
            dtype=jnp.float32,
        )
        segment_tokens = segment_tokens.at[..., : len(SWAP_PAIRS)].add(pair_codes[:, :, None, None, :])

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.memory_width),
            jnp.float32,
        )
        memory = jnp.tile(base_memory, (batch_size, 1, 1))
        initial_code = jax.nn.one_hot(initial_slots, 3, dtype=jnp.float32)
        memory = memory.at[:, 0, :3].add(initial_code)

        base_current_tokens = self.param(
            "base_current_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_current_tokens, self.current_width),
            jnp.float32,
        )
        base_current_tokens = jnp.tile(base_current_tokens, (batch_size, 1, 1))

        updater = SharedSegmentMemoryUpdater(
            name="shared_segment_memory_updater",
            width=self.memory_width,
            depth=self.memory_depth,
            num_heads=self.memory_heads,
            segment_size=_sweep.SEGMENT_SIZE,
            dtype_mm="float32",
        )
        adapter = SingleHistoryReadAdapter(
            name="shared_history_read_adapter",
            memory_width=self.memory_width,
            current_width=self.current_width,
            num_heads=self.adapter_heads,
            residual_scale=self.residual_scale,
        )
        readout = _sweep.SharedMemoryTokenReadout(
            name="shared_readout",
            width=self.current_width,
        )

        stage_logits = []
        stage_current_tokens = []
        for stage_index in range(_oracle_pair.NUM_SWAP_SEGMENTS):
            memory = updater(memory, segment_tokens[:, stage_index])
            current_tokens = adapter(base_current_tokens, memory)
            stage_current_tokens.append(current_tokens)
            stage_logits.append(readout(current_tokens))

        stage_logits = jnp.stack(stage_logits, axis=1)
        stage_current_tokens = jnp.stack(stage_current_tokens, axis=1)
        logits_0, logits_1, logits_2 = (
            stage_logits[:, 0],
            stage_logits[:, 1],
            stage_logits[:, 2],
        )
        joint_logits = (logits_0[:, :, None, None] + logits_1[:, None, :, None] + logits_2[:, None, None, :]).reshape(
            batch_size, 27
        )
        return joint_logits, stage_logits, stage_current_tokens


@dataclasses.dataclass(frozen=True)
class SingleHistoryReadOracleConfig(_raw.RawInitialOracleMemoryTokenConfig):
    adapter_current_width: int = 1152
    adapter_current_tokens: int = 256
    adapter_heads: int = 4
    adapter_residual_scale: float = 1.0

    def create(self, rng: at.KeyArrayLike) -> SingleHistoryReadOracleModel:
        return SingleHistoryReadOracleModel(self, rngs=nnx.Rngs(rng))


class SingleHistoryReadOracleModel(_oracle_pair.OraclePairProbeModel):
    def __init__(self, config: SingleHistoryReadOracleConfig, rngs: nnx.Rngs):
        _base_model.Pi0MemCompress.__init__(self, config, rngs)
        self.oracle_initial_slots = config.oracle_initial_slots
        self.oracle_swap_pairs = config.oracle_swap_pairs
        self.pair_mode = config.pair_mode
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker = nnx_bridge.ToNNX(
            SingleHistoryReadOracleTracker(
                memory_width=config.memory_token_width,
                current_width=config.adapter_current_width,
                memory_depth=config.memory_token_depth,
                memory_heads=config.memory_token_heads,
                adapter_heads=config.adapter_heads,
                num_memory_tokens=config.diagnostic_memory_tokens,
                num_current_tokens=config.adapter_current_tokens,
                residual_scale=config.adapter_residual_scale,
            )
        )
        fake_slots = jnp.zeros((1,), dtype=jnp.int32)
        fake_pairs = jnp.zeros((1, _oracle_pair.NUM_SWAP_SEGMENTS), dtype=jnp.int32)
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker.lazy_init(fake_slots, fake_pairs, rngs=rngs)


def run_self_test() -> None:
    tracker = SingleHistoryReadOracleTracker(
        memory_width=16,
        current_width=32,
        memory_depth=2,
        memory_heads=4,
        adapter_heads=4,
        num_memory_tokens=8,
        num_current_tokens=8,
        residual_scale=1.0,
    )
    slots = jnp.asarray((0, 1), dtype=jnp.int32)
    pairs = jnp.asarray(((0, 1, 2), (2, 1, 0)), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), slots, pairs)
    _, reference, tokens = tracker.apply(variables, slots, pairs)
    if tokens.shape != (2, 3, 8, 32):
        raise AssertionError(f"Unexpected current-token shape: {tokens.shape}")
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
            f"Single-read self-test failed: causal={causal}, pair={pair_effects}, initial={initial_effect}"
        )
    print(
        "Single history-read self-test passed: "
        f"causal={causal}, pair_effects={pair_effects}, initial_effect={initial_effect}"
    )


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    base_config = _raw.build_config(args, labels_path)
    parent_model = base_config.model
    parent_fields = {field.name: getattr(parent_model, field.name) for field in dataclasses.fields(parent_model)}
    model = SingleHistoryReadOracleConfig(
        **parent_fields,
        adapter_current_width=args.adapter_current_width,
        adapter_current_tokens=args.adapter_current_tokens,
        adapter_heads=args.adapter_heads,
        adapter_residual_scale=args.adapter_residual_scale,
    )
    return dataclasses.replace(
        base_config,
        name="pi0_shellgame_three_swap_oracle_single_history_read_adapter_260809",
        model=model,
        freeze_filter=model.get_freeze_filter_tracker_only(),
        weight_loader=_minimal.MinimalTransitionCheckpointLoader(args.init_checkpoint),
        resume=args.resume,
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
    parser.add_argument("--adapter-current-width", type=int, default=1152)
    parser.add_argument("--adapter-current-tokens", type=int, default=256)
    parser.add_argument("--adapter-heads", type=int, default=4)
    parser.add_argument("--adapter-residual-scale", type=float, default=1.0)
    parser.add_argument("--pair-mode", choices=("correct", "roll", "shuffle_batch"), default="correct")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
