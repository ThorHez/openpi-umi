"""Isolated Pi0 capacity experiment with post-compression visual Transformer.

All policy/loss methods are inherited from ``Pi0MemCompress``.  Only model
construction changes the visual module to ``siglip_mem_post_transformer`` so
existing experiments and checkpoints remain untouched.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge

from openpi.models import gemma as _gemma
from openpi.models import pi0_mem_compress as _base
from openpi.models import siglip_mem_post_transformer as _siglip_post
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi0MemPostTransformerConfig(_base.Pi0MemCompressConfig):
    """Pi0MemCompress config with scheduled joint memory/current self-attention."""

    history_joint_start_layer: int = 18

    def create(self, rng: at.KeyArrayLike) -> Pi0MemPostTransformer:
        return Pi0MemPostTransformer(self, rngs=nnx.Rngs(rng))


class Pi0MemPostTransformer(_base.Pi0MemCompress):
    """Pi0MemCompress whose compressed memory traverses SigLIP blocks."""

    def __init__(self, config: Pi0MemPostTransformerConfig, rngs: nnx.Rngs):
        # Deliberately duplicate only construction; all behavior after visual
        # encoding stays inherited from Pi0MemCompress.
        nnx.Module.__init__(self)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.max_token_len = config.max_token_len
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
            _siglip_post.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
                current_frame_index=config.current_frame_index,
                history_memory_tokens=config.history_memory_tokens,
                history_resampler_depth=config.history_resampler_depth,
                history_use_current_condition=config.history_use_current_condition,
                joint_start_layer=config.history_joint_start_layer,
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
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.history_classifier_num_classes = config.history_classifier_num_classes
        if self.history_classifier_num_classes > 0:
            vision_width = _siglip_post.decode_variant("So400m/14")["width"]
            self.HistoryClassifierNorm = nnx.LayerNorm(vision_width, rngs=rngs)
            self.HistoryClassifierHead = nnx.Linear(
                config.history_memory_tokens * vision_width,
                self.history_classifier_num_classes,
                rngs=rngs,
            )
        self.deterministic = True
