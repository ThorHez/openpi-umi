"""Insert pair-token cross-attention before the proven minimal transition cell.

This is the next controlled step after the minimal Oracle transition probe.
The exact pair one-hot is expanded into a 10x64 token segment, matching the
temporal/spatial token count used by the M128 updater.  One state-conditioned
query cross-attends to that segment and projects the result back to a 3-D pair
feature.  The width-64 shared residual transition and shared stage classifier
remain identical to the successful minimal probe.  No pair auxiliary loss,
images, or action loss are used.
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
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
from openpi.training import config as _config
from scripts.mem import train_pi0_mem_compress as _trainer

SEGMENT_SIZE = 10
SPATIAL_TOKENS = 64


class OraclePairCrossAttentionExtractor(nn.Module):
    """Extract one three-dimensional pair feature from a dense token segment."""

    width: int = 64
    num_heads: int = 4
    segment_size: int = SEGMENT_SIZE
    spatial_tokens: int = SPATIAL_TOKENS

    @nn.compact
    def __call__(self, state, segment):
        expected = (state.shape[0], self.segment_size, self.spatial_tokens, self.width)
        if segment.shape != expected:
            raise ValueError(f"Expected segment {expected}, got {segment.shape}")

        relative_pos = self.param(
            "relative_temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.segment_size, 1, self.width),
            segment.dtype,
        )
        tokens = (segment + relative_pos).reshape(state.shape[0], -1, self.width)
        query = nn.LayerNorm(name="query_ln", dtype=jnp.float32)(state)[:, None, :]
        tokens = nn.LayerNorm(name="segment_ln", dtype=jnp.float32)(tokens)
        extracted = nn.MultiHeadDotProductAttention(
            name="cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=jnp.float32,
        )(query, tokens)[:, 0]
        extracted = nn.LayerNorm(name="output_ln", dtype=jnp.float32)(extracted)
        return nn.Dense(3, name="pair_projection", dtype=jnp.float32)(extracted)


class CrossAttentionOracleTransitionTracker(nn.Module):
    """Use cross-attended pair features in the proven shared transition MLP."""

    width: int = 64
    depth: int = 2
    num_heads: int = 4

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
        state = nn.Dense(self.width, name="initial_projection", dtype=jnp.float32)(initial_code)
        state = nn.tanh(state)

        extractor = OraclePairCrossAttentionExtractor(
            name="shared_pair_extractor",
            width=self.width,
            num_heads=self.num_heads,
        )
        transition = _minimal.ResidualOracleTransitionCell(
            name="shared_transition",
            width=self.width,
            depth=self.depth,
        )
        classifier = nn.Dense(3, name="shared_classifier", dtype=jnp.float32)
        stage_logits = []
        stage_states = []
        for stage_index in range(_oracle_pair.NUM_SWAP_SEGMENTS):
            pair_feature = extractor(state, segment_tokens[:, stage_index])
            state = transition(state, pair_feature)
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
class CrossAttentionOracleTransitionConfig(_minimal.MinimalOracleTransitionConfig):
    cross_attention_heads: int = 4

    def create(self, rng: at.KeyArrayLike) -> CrossAttentionOracleTransitionModel:
        return CrossAttentionOracleTransitionModel(self, rngs=nnx.Rngs(rng))


class CrossAttentionOracleTransitionModel(_oracle_pair.OraclePairProbeModel):
    def __init__(self, config: CrossAttentionOracleTransitionConfig, rngs: nnx.Rngs):
        _base_model.Pi0MemCompress.__init__(self, config, rngs)
        self.oracle_initial_slots = config.oracle_initial_slots
        self.oracle_swap_pairs = config.oracle_swap_pairs
        self.pair_mode = config.pair_mode
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker = nnx_bridge.ToNNX(
            CrossAttentionOracleTransitionTracker(
                width=config.transition_width,
                depth=config.transition_depth,
                num_heads=config.cross_attention_heads,
            )
        )
        fake_slots = jnp.zeros((1,), dtype=jnp.int32)
        fake_pairs = jnp.zeros((1, _oracle_pair.NUM_SWAP_SEGMENTS), dtype=jnp.int32)
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker.lazy_init(fake_slots, fake_pairs, rngs=rngs)


def run_self_test() -> None:
    tracker = CrossAttentionOracleTransitionTracker(width=16, depth=2, num_heads=4)
    slots = jnp.asarray((0, 1), dtype=jnp.int32)
    pairs = jnp.asarray(((0, 1, 2), (2, 1, 0)), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), slots, pairs)
    _, reference, _ = tracker.apply(variables, slots, pairs)
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
            f"Cross-attention self-test failed: causal={causal}, pair={pair_effects}, initial={initial_effect}"
        )
    print(
        "Cross-attention transition self-test passed: "
        f"causal={causal}, pair_effects={pair_effects}, initial_effect={initial_effect}"
    )


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    base_config = _minimal.build_config(args, labels_path)
    parent_model = base_config.model
    parent_fields = {field.name: getattr(parent_model, field.name) for field in dataclasses.fields(parent_model)}
    model = CrossAttentionOracleTransitionConfig(
        **parent_fields,
        cross_attention_heads=args.cross_attention_heads,
    )
    return dataclasses.replace(
        base_config,
        name="pi0_shellgame_three_swap_oracle_cross_attention_transition_260809",
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
    parser.add_argument("--cross-attention-heads", type=int, default=4)
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
