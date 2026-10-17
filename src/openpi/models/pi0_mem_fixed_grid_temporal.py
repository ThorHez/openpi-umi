"""Pi0/Pi0.5 with a topology-preserving, inexpensive history encoder.

This is an independent model variant.  It keeps the current frame on the
pretrained 27-layer SigLIP path, while historical frames stop after patch
embedding and follow the fixed-grid temporal path implemented in
``siglip_mem_fixed_grid_temporal``.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base
from openpi.models import siglip_mem_fixed_grid_temporal as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils


@dataclasses.dataclass(frozen=True)
class Pi0MemFixedGridTemporalConfig(_base.Pi0MemCompressConfig):
    """Configuration for the fixed-grid temporal history variant."""

    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0MemFixedGridTemporal:
        return Pi0MemFixedGridTemporal(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_history_only(self) -> nnx.filterlib.Filter:
        """Train only the fixed-grid temporal encoder and Pi0 memory readers."""
        history = nnx_utils.PathRegex(
            r".*PaliGemma/img/Transformer/(FixedGridTemporalHistory_0|encoderblock/"
            r"(?:HistoryLayerNorm_0|HistoryMultiHeadDotProductAttention_0)).*"
        )
        return nnx.Not(history)


class Pi0MemFixedGridTemporal(_base.Pi0MemCompress):
    """Pi0 whose history preserves a fixed 8x8 spatial grid across time."""

    def __init__(self, config: Pi0MemFixedGridTemporalConfig, rngs: nnx.Rngs):
        # Deliberately construct the model directly rather than calling the
        # Pi0MemCompress constructor and allocating its discarded resampler.
        _model.BaseModel.__init__(
            self,
            config.action_dim,
            config.action_horizon,
            config.max_token_len,
        )
        self.pi05 = config.pi05
        self.action_loss_mask = config.action_loss_mask
        self.num_frames = config.num_frames

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if config.pi05 else [False, False],
        )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
                memory_every=config.memory_every,
                current_frame_index=config.current_frame_index,
                history_memory_tokens=config.history_memory_tokens,
                temporal_width=config.temporal_width,
                temporal_depth=config.temporal_depth,
                temporal_heads=config.temporal_heads,
                spatial_pool_factor=config.spatial_pool_factor,
                history_gate_init=config.history_gate_init,
                history_gate_fixed=config.history_gate_fixed,
                remat_policy=config.siglip_remat_policy,
            )
        )
        fake_obs = config.fake_obs()
        sample_image = next(iter(fake_obs.images.values()))
        img.lazy_init(sample_image, train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(
                2 * action_expert_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Retain the normal Pi0MemCompress diagnostic classifier contract.
        self.history_classifier_num_classes = config.history_classifier_num_classes
        if self.history_classifier_num_classes > 0:
            vision_width = _siglip.decode_variant("So400m/14")["width"]
            self.HistoryClassifierNorm = nnx.LayerNorm(vision_width, rngs=rngs)
            self.HistoryClassifierHead = nnx.Linear(
                config.history_memory_tokens * vision_width,
                self.history_classifier_num_classes,
                rngs=rngs,
            )

        self.deterministic = True
