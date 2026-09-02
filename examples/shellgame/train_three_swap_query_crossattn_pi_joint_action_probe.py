"""Condition Pi0's action expert through a compact, dedicated memory path.

Controlled follow-up to ``train_three_swap_raw_memory_pi_joint_action_probe``:

* the same frozen visual tracker produces raw memory ``[B, 128, 64]``;
* 16 learned queries resample it into action-width memory tokens;
* action suffix tokens read those tokens through dedicated gated cross-attention;
* history memory is not appended to the visual/language prefix.

No cup label, final-slot probability, or task metadata is passed to the action
model.  The only training objective is Pi0.5 flow matching on absolute joints.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import einops
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from examples.shellgame import train_three_swap_fully_visual_joint_action_probe as _direct
from examples.shellgame import train_three_swap_raw_memory_pi_joint_action_probe as _raw
from examples.shellgame.train_fixed_grid_action60_probe import _frame59_only
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _pi_mem
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from scripts.mem import train_pi0_mem_compress as _trainer


class RawMemoryQueryResampler(nn.Module):
    """Compress 128 raw tokens into a small set of generic learned queries."""

    input_width: int = 64
    width: int = 256
    output_width: int = 1024
    input_tokens: int = 128
    query_tokens: int = 16
    depth: int = 2
    num_heads: int = 4
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, memory):
        if memory.ndim != 3 or memory.shape[1:] != (self.input_tokens, self.input_width):
            raise ValueError(
                f"Expected memory [B,{self.input_tokens},{self.input_width}], got {memory.shape}"
            )
        batch_size = memory.shape[0]
        position = self.param(
            "memory_position_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.input_tokens, self.input_width),
            jnp.float32,
        )
        memory = nn.LayerNorm(name="memory_input_ln", dtype=jnp.float32)(
            memory.astype(jnp.float32) + position
        )
        memory = nn.Dense(self.width, name="memory_projection", dtype=self.dtype_mm)(memory)

        learned_queries = self.param(
            "learned_queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.query_tokens, self.width),
            jnp.float32,
        )
        queries = jnp.tile(learned_queries, (batch_size, 1, 1)).astype(memory.dtype)
        for layer in range(self.depth):
            query_norm = nn.LayerNorm(name=f"query_ln_{layer}", dtype=self.dtype_mm)(queries)
            memory_norm = nn.LayerNorm(name=f"memory_ln_{layer}", dtype=self.dtype_mm)(memory)
            update = nn.MultiHeadDotProductAttention(
                name=f"query_cross_attention_{layer}",
                num_heads=self.num_heads,
                dropout_rate=0.0,
                deterministic=True,
                dtype=self.dtype_mm,
            )(query_norm, memory_norm)
            queries = queries + update
            mlp_input = nn.LayerNorm(name=f"mlp_ln_{layer}", dtype=self.dtype_mm)(queries)
            hidden = nn.Dense(self.width * 4, name=f"mlp_in_{layer}", dtype=self.dtype_mm)(mlp_input)
            hidden = nn.gelu(hidden)
            queries = queries + nn.Dense(
                self.width, name=f"mlp_out_{layer}", dtype=self.dtype_mm
            )(hidden)

        queries = nn.LayerNorm(name="query_output_ln", dtype=self.dtype_mm)(queries)
        queries = nn.Dense(
            self.output_width, name="action_width_projection", dtype=self.dtype_mm
        )(queries)
        return nn.LayerNorm(name="action_width_ln", dtype=self.dtype_mm)(queries)


class ActionMemoryCrossAttention(nn.Module):
    """Give action tokens a direct memory read, with a learned residual gate."""

    width: int = 1024
    num_heads: int = 8
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, action_tokens, memory_tokens):
        if action_tokens.shape[-1] != self.width or memory_tokens.shape[-1] != self.width:
            raise ValueError(
                f"Expected width {self.width}, got {action_tokens.shape} and {memory_tokens.shape}"
            )
        action_norm = nn.LayerNorm(name="action_ln", dtype=self.dtype_mm)(action_tokens)
        memory_norm = nn.LayerNorm(name="memory_ln", dtype=self.dtype_mm)(memory_tokens)
        update = nn.MultiHeadDotProductAttention(
            name="cross_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(action_norm, memory_norm)
        # The effective gate starts at 1.0.  tanh keeps it bounded in [0, 2].
        gate_delta = self.param("gate_delta", nn.initializers.zeros_init(), (1,), jnp.float32)
        gate = (1.0 + jnp.tanh(gate_delta)).astype(update.dtype)
        conditioned = action_tokens + gate * update
        mlp_input = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(conditioned)
        hidden = nn.Dense(self.width * 2, name="mlp_in", dtype=self.dtype_mm)(mlp_input)
        hidden = nn.gelu(hidden)
        mlp_update = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(hidden)
        return conditioned + gate * mlp_update


@dataclasses.dataclass(frozen=True)
class QueryCrossAttnPiJointActionConfig(_raw.RawMemoryPiJointActionConfig):
    query_tokens: int = 16
    query_width: int = 256
    query_depth: int = 2
    query_heads: int = 4
    action_cross_attention_heads: int = 8

    def create(self, rng: at.KeyArrayLike) -> QueryCrossAttnPiJointActionModel:
        return QueryCrossAttnPiJointActionModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_query_crossattn_action(self) -> nnx.filterlib.Filter:
        memory_interface = nnx_utils.PathRegex(
            r".*(HistoryRawMemoryQueryResampler|ActionMemoryCrossAttention).*"
        )
        action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm/.*_1.*")
        action_modules = nnx_utils.PathRegex(
            r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"
        )
        return nnx.Not(nnx.Any(memory_interface, action_expert, action_modules))


class QueryCrossAttnPiJointActionModel(_direct.FullyVisualJointActionModel):
    def __init__(self, config: QueryCrossAttnPiJointActionConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.raw_memory_mode = config.raw_memory_mode
        self.HistoryRawMemoryQueryResampler = nnx_bridge.ToNNX(
            RawMemoryQueryResampler(
                input_width=64,
                width=config.query_width,
                output_width=1024,
                input_tokens=128,
                query_tokens=config.query_tokens,
                depth=config.query_depth,
                num_heads=config.query_heads,
                dtype_mm=config.dtype,
            )
        )
        self.HistoryRawMemoryQueryResampler.lazy_init(
            jnp.zeros((1, 128, 64), dtype=jnp.bfloat16), rngs=rngs
        )
        self.ActionMemoryCrossAttention = nnx_bridge.ToNNX(
            ActionMemoryCrossAttention(
                width=1024,
                num_heads=config.action_cross_attention_heads,
                dtype_mm=config.dtype,
            )
        )
        self.ActionMemoryCrossAttention.lazy_init(
            jnp.zeros((1, self.action_horizon, 1024), dtype=jnp.bfloat16),
            jnp.zeros((1, config.query_tokens, 1024), dtype=jnp.bfloat16),
            rngs=rngs,
        )

    def _raw_and_resampled_memory(self, observation: _model.Observation):
        tracked = self._track_history(observation)
        raw_memory = jax.lax.stop_gradient(tracked["stage_memories"][:, -1])
        if self.raw_memory_mode == "shuffle_batch":
            raw_memory = jnp.roll(raw_memory, 1, axis=0)
        elif self.raw_memory_mode == "zero":
            raw_memory = jnp.zeros_like(raw_memory)
        elif self.raw_memory_mode != "normal":
            raise ValueError(f"Unknown raw_memory_mode={self.raw_memory_mode!r}")
        return raw_memory, self.HistoryRawMemoryQueryResampler(raw_memory), tracked

    def _embed_current_prefix(self, observation: _model.Observation):
        tokens = []
        input_mask = []
        ar_mask: list[bool] = []
        for name, video in observation.images.items():
            image = video[:, -1] if video.ndim == 5 else video
            image_tokens, _ = self.PaliGemma.img(image[:, None], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(observation.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]
        if observation.tokenized_prompt is not None:
            prompt_tokens = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            tokens.append(prompt_tokens)
            input_mask.append(observation.tokenized_prompt_mask)
            ar_mask += [False] * prompt_tokens.shape[1]
        return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1), jnp.asarray(ar_mask)

    def compute_loss_with_memory_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        del train
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=False)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        raw_memory, memory_tokens, tracked = self._raw_and_resampled_memory(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_current_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
        suffix_tokens = self.ActionMemoryCrossAttention(suffix_tokens, memory_tokens)
        input_mask = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
        ar_mask = jnp.concatenate((prefix_ar_mask, suffix_ar_mask), axis=0)
        attn_mask = _pi_mem.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        squared_error = jnp.square(velocity - u_t)
        mask = (
            observation.action_loss_mask[..., None, :]
            if observation.action_loss_mask is not None
            else jnp.asarray(self.action_loss_mask)[None, None, :]
        )
        loss_per_timestep = jnp.sum(squared_error * mask, axis=-1) / jnp.maximum(
            jnp.sum(mask, axis=-1), 1e-8
        )
        return loss_per_timestep, {
            "history_mem": raw_memory,
            "encoder_auxes": (),
            "history_class_logits": tracked["joint_logits"],
        }

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        loss, _ = self.compute_loss_with_memory_aux(rng, observation, actions, train=train)
        return loss

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        _, memory_tokens, _ = self._raw_and_resampled_memory(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_current_prefix(observation)
        prefix_attn_mask = _pi_mem.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_tokens = self.ActionMemoryCrossAttention(suffix_tokens, memory_tokens)
            suffix_attn_mask = _pi_mem.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate((prefix_for_suffix, suffix_attn_mask), axis=-1)
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * velocity, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return actions


@dataclasses.dataclass(frozen=True)
class QueryCrossAttnCheckpointLoader:
    params_path: str
    restore_memory_interface: bool = False

    def load(self, params: at.Params) -> at.Params:
        source = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target_flat = flax.traverse_util.flatten_dict(params, sep="/")
        source_flat = flax.traverse_util.flatten_dict(source, sep="/")
        interface_prefixes = (
            "HistoryRawMemoryQueryResampler/",
            "ActionMemoryCrossAttention/",
        )
        restored = {}
        counts = {"checkpoint": 0, "random_interface": 0}
        missing = []
        for key, reference in target_flat.items():
            is_interface = key.startswith(interface_prefixes)
            if is_interface and not self.restore_memory_interface:
                restored[key] = reference
                counts["random_interface"] += 1
                continue
            candidate = source_flat.get(key)
            if candidate is None or np.shape(candidate) != np.shape(reference):
                missing.append(key)
                restored[key] = reference
            else:
                restored[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                counts["checkpoint"] += 1
        if missing:
            raise ValueError(f"Query-cross-attention restore incomplete: {missing[:8]}")
        print(
            "QueryCrossAttnCheckpointLoader: "
            f"checkpoint={counts['checkpoint']}, random_interface={counts['random_interface']}, missing=0"
        )
        return flax.traverse_util.unflatten_dict(restored, sep="/")


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    parent = _raw.build_config(args)
    base_fields = {
        field.name: getattr(parent.model, field.name)
        for field in dataclasses.fields(_raw.RawMemoryPiJointActionConfig)
    }
    model = QueryCrossAttnPiJointActionConfig(
        **base_fields,
        query_tokens=args.query_tokens,
        query_width=args.query_width,
        query_depth=args.query_depth,
        query_heads=args.query_heads,
        action_cross_attention_heads=args.action_cross_attention_heads,
    )
    return dataclasses.replace(
        parent,
        name="pi0_shellgame_three_swap_query_crossattn_pi_joint_action_260810",
        model=model,
        freeze_filter=model.get_freeze_filter_query_crossattn_action(),
        weight_loader=QueryCrossAttnCheckpointLoader(
            args.tracker_checkpoint,
            restore_memory_interface=args.restore_memory_interface,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--tracker-checkpoint", default=_raw.DEFAULT_TRACKER_CHECKPOINT)
    parser.add_argument(
        "--raw-memory-mode", choices=("normal", "shuffle_batch", "zero"), default="normal"
    )
    parser.add_argument("--restore-adapter", action="store_true")
    parser.add_argument("--restore-memory-interface", action="store_true")
    parser.add_argument("--query-tokens", type=int, default=16)
    parser.add_argument("--query-width", type=int, default=256)
    parser.add_argument("--query-depth", type=int, default=2)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--action-cross-attention-heads", type=int, default=8)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument("--memory-checkpoint", default="")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--peak-lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--cup-eval-interval", type=int, default=50)
    parser.add_argument("--cup-eval-episodes", type=int, default=24)
    parser.add_argument("--cup-eval-batch-size", type=int, default=6)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--encoder-width", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--memory-width", type=int, default=64)
    parser.add_argument("--memory-depth", type=int, default=2)
    parser.add_argument("--memory-heads", type=int, default=4)
    parser.add_argument("--adapter-heads", type=int, default=4)
    parser.add_argument("--memory-tokens", type=int, default=128)
    parser.add_argument("--current-tokens", type=int, default=256)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--overfit-samples-per-class", type=int, default=0)
    parser.add_argument("--video-mode", choices=("normal",), default="normal")
    parser.add_argument("--initial-mode", choices=("normal",), default="normal")
    parser.add_argument("--relation-mode", choices=("one_hot",), default="one_hot")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _trainer._filter_memory_classifier_frame_range = _frame59_only  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
