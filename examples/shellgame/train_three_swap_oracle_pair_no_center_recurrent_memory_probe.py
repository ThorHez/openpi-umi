"""Ablate token-axis mean subtraction in the Oracle-pair M128 probe.

This is identical to ``train_three_swap_oracle_pair_recurrent_memory_probe``
except that it removes the two token-axis mean-subtraction operations after
the memory output projection.  Ground-truth initial slot and swap-pair codes,
the recurrent updater, readout, initialization, labels, and optimizer remain
unchanged.
"""

from __future__ import annotations

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

from examples.shellgame import train_three_swap_oracle_pair_recurrent_memory_probe as _centered
from examples.shellgame.train_one_swap_fixed_grid_integrated_probe import IntegratedCurrentReadout
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import build_three_swap_labels
from examples.shellgame.train_three_swap_causal_fixed_grid_probe import multistage_eval_step
from examples.shellgame.train_three_swap_pair_fixed_grid_probe import SWAP_PAIRS
from examples.shellgame.train_three_swap_recurrent_memory_fixed_grid_probe import SharedSegmentMemoryUpdater
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
from scripts.mem import train_pi0_mem_compress as _trainer


class ThreeSwapOraclePairNoCenterRecurrentMemoryTracker(_centered.ThreeSwapOraclePairRecurrentMemoryTracker):
    """The same tracker with only the two centering operations removed."""

    @nn.compact
    def __call__(self, initial_slots, swap_pair_ids):
        if initial_slots.ndim != 1:
            raise ValueError(f"Expected initial slots [B], got {initial_slots.shape}")
        b = initial_slots.shape[0]
        if swap_pair_ids.shape != (b, _centered.NUM_SWAP_SEGMENTS):
            raise ValueError(f"Expected swap pairs [B,3], got {swap_pair_ids.shape}")

        pair_codes = jax.nn.one_hot(swap_pair_ids, len(SWAP_PAIRS), dtype=jnp.dtype(self.dtype_mm))
        segment_tokens = jnp.zeros(
            (
                b,
                _centered.NUM_SWAP_SEGMENTS,
                self.segment_size,
                self.spatial_tokens,
                self.width,
            ),
            dtype=pair_codes.dtype,
        )
        pair_codes = pair_codes[:, :, None, None, :]
        segment_tokens = segment_tokens.at[..., : len(SWAP_PAIRS)].add(self.oracle_code_scale * pair_codes)

        base_memory = self.param(
            "base_memory",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.width),
            segment_tokens.dtype,
        )
        memory = jnp.tile(base_memory, (b, 1, 1))
        initial_code = jax.nn.one_hot(initial_slots, 3, dtype=memory.dtype)
        memory = memory.at[:, 0, :3].add(self.oracle_code_scale * initial_code)

        updater = SharedSegmentMemoryUpdater(
            name="shared_segment_memory_updater",
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            segment_size=self.segment_size,
            dtype_mm=self.dtype_mm,
        )
        endpoint_memories = []
        for segment_index in range(_centered.NUM_SWAP_SEGMENTS):
            memory = updater(memory, segment_tokens[:, segment_index])
            endpoint_memories.append(memory)

        memory_batch = jnp.stack(endpoint_memories, axis=1).reshape(
            b * _centered.NUM_SWAP_SEGMENTS, self.num_memory_tokens, self.width
        )
        memory_batch = nn.LayerNorm(name="memory_output_ln", dtype=self.dtype_mm)(memory_batch)
        memory_batch = nn.Dense(
            self.input_width,
            name="memory_output_projection",
            dtype=self.dtype_mm,
        )(memory_batch)
        # Controlled ablation: the centered version subtracts the token mean here.
        memory_batch = nn.LayerNorm(name="pi0_output_ln", dtype=self.dtype_mm)(memory_batch)
        # Controlled ablation: the centered version subtracts the token mean again here.

        logits = IntegratedCurrentReadout(
            name="shared_readout",
            input_width=self.input_width,
            width=self.width,
            num_classes=3,
            dtype_mm=self.dtype_mm,
        )(memory_batch)
        stage_logits = logits.reshape(b, _centered.NUM_SWAP_SEGMENTS, 3)
        stage_memories = memory_batch.reshape(
            b,
            _centered.NUM_SWAP_SEGMENTS,
            self.num_memory_tokens,
            self.input_width,
        )
        logits_0, logits_1, logits_2 = (
            stage_logits[:, 0],
            stage_logits[:, 1],
            stage_logits[:, 2],
        )
        joint_logits = (logits_0[:, :, None, None] + logits_1[:, None, :, None] + logits_2[:, None, None, :]).reshape(
            b, 27
        )
        return joint_logits, stage_logits, stage_memories


@dataclasses.dataclass(frozen=True)
class OraclePairNoCenterProbeConfig(_centered.OraclePairProbeConfig):
    def create(self, rng: at.KeyArrayLike) -> OraclePairNoCenterProbeModel:
        return OraclePairNoCenterProbeModel(self, rngs=nnx.Rngs(rng))


class OraclePairNoCenterProbeModel(_centered.OraclePairProbeModel):
    def __init__(self, config: OraclePairNoCenterProbeConfig, rngs: nnx.Rngs):
        _base_model.Pi0MemCompress.__init__(self, config, rngs)
        self.oracle_initial_slots = config.oracle_initial_slots
        self.oracle_swap_pairs = config.oracle_swap_pairs
        self.pair_mode = config.pair_mode
        # Keep the old attribute name so every parameter path is identical.
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker = nnx_bridge.ToNNX(
            ThreeSwapOraclePairNoCenterRecurrentMemoryTracker(
                width=config.temporal_width,
                input_width=1152,
                depth=config.temporal_depth,
                num_heads=config.temporal_heads,
                num_memory_tokens=config.endpoint_memory_tokens,
                segment_size=_centered.SEGMENT_SIZE,
                spatial_tokens=_centered.SPATIAL_TOKENS,
                dtype_mm=config.dtype,
            )
        )
        fake_slots = jnp.zeros((1,), dtype=jnp.int32)
        fake_pairs = jnp.zeros((1, _centered.NUM_SWAP_SEGMENTS), dtype=jnp.int32)
        self.HistoryThreeSwapOraclePairRecurrentMemoryTracker.lazy_init(fake_slots, fake_pairs, rngs=rngs)


@dataclasses.dataclass(frozen=True)
class NoCenterCheckpointLoader:
    """Restore either the original Oracle tracker or a no-center checkpoint."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(loaded, sep="/")
        target_root = "HistoryThreeSwapOraclePairRecurrentMemoryTracker/"
        original_root = "HistoryThreeSwapOracleRecurrentMemoryTracker/"
        result = {}
        exact = mapped = 0
        initialized = []
        for key, reference in target.items():
            candidate = source.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                exact += 1
                continue
            source_key = None
            if key.startswith(target_root):
                source_key = original_root + key.removeprefix(target_root)
            candidate = source.get(source_key) if source_key is not None else None
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                mapped += 1
            else:
                result[key] = reference
                initialized.append(key)

        unexpected = [key for key in initialized if key.startswith(target_root)]
        if unexpected:
            raise ValueError(f"No-center restore incomplete: {unexpected[:8]}")
        print(
            f"NoCenterCheckpointLoader: exact={exact}, mapped={mapped}, "
            f"initialized={len(initialized)}, examples={initialized[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def build_config(args, labels_path):
    centered_config = _centered.build_config(args, labels_path)
    centered_model = centered_config.model
    model_kwargs = {field.name: getattr(centered_model, field.name) for field in dataclasses.fields(centered_model)}
    model = OraclePairNoCenterProbeConfig(**model_kwargs)
    return dataclasses.replace(
        centered_config,
        name="pi0_shellgame_three_swap_oracle_pair_no_center_recurrent_memory_260809",
        model=model,
        freeze_filter=model.get_freeze_filter_tracker_only(),
        weight_loader=NoCenterCheckpointLoader(args.init_checkpoint),
    )


def run_causality_self_test() -> None:
    tracker = ThreeSwapOraclePairNoCenterRecurrentMemoryTracker(
        width=16,
        input_width=16,
        depth=2,
        num_heads=4,
        num_memory_tokens=8,
        segment_size=_centered.SEGMENT_SIZE,
        spatial_tokens=_centered.SPATIAL_TOKENS,
        dtype_mm="float32",
    )
    slots = jnp.asarray((0, 1), dtype=jnp.int32)
    pairs = jnp.asarray(((0, 1, 2), (2, 1, 0)), dtype=jnp.int32)
    variables = tracker.init(jax.random.key(0), slots, pairs)
    _, reference, memory = tracker.apply(variables, slots, pairs)
    changed = pairs.at[:, 2].set((pairs[:, 2] + 1) % 3)
    _, candidate, _ = tracker.apply(variables, slots, changed)
    causal = np.allclose(
        np.asarray(reference[:, :2]),
        np.asarray(candidate[:, :2]),
        rtol=0.0,
        atol=0.0,
    )
    retained_mean = float(jnp.linalg.norm(jnp.mean(memory, axis=2))) > 0.0
    if not causal or not retained_mean:
        raise AssertionError(f"No-center self-test failed: causal={causal}, retained_mean={retained_mean}")
    print(f"No-center self-test passed: causal={causal}, retained_mean={retained_mean}")


def main() -> None:
    args = _centered.parse_args()
    if args.self_test:
        run_causality_self_test()
        return
    _trainer.eval_step = multistage_eval_step
    _trainer.main(build_config(args, build_three_swap_labels()))


if __name__ == "__main__":
    main()
