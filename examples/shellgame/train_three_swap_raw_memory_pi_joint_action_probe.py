"""Condition the native Pi0.5 flow action expert on raw compact memory.

This is the controlled follow-up to the successful semantic joint-action
diagnostic.  It keeps the validated visual tracker frozen but removes the
three-way final-slot probability interface.  The final raw compact memory
``[B, 128, 64]`` is projected into 128 Pi prefix tokens and concatenated with
the current images and prompt.  Pi0.5's native flow-matching action expert then
predicts the normalized absolute-joint chunk for raw frames 60:75.
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
from examples.shellgame.train_fixed_grid_action60_probe import _frame59_only
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _pi_mem
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from scripts.mem import train_pi0_mem_compress as _trainer


DEFAULT_TRACKER_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_three_swap_fully_visual_joint_action_260810/"
    "fully_visual_joint_action_head300_b72_260810/299/params"
)


class RawMemoryToPiPrefix(nn.Module):
    """Token-wise adapter; it has no task-specific cup classifier."""

    input_width: int = 64
    hidden_width: int = 512
    output_width: int = 2048
    num_tokens: int = 128
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, memory):
        if memory.ndim != 3 or memory.shape[1:] != (self.num_tokens, self.input_width):
            raise ValueError(
                f"Expected raw memory [B,{self.num_tokens},{self.input_width}], got {memory.shape}"
            )
        position = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_tokens, self.input_width),
            jnp.float32,
        )
        x = memory.astype(jnp.float32) + position
        x = nn.LayerNorm(name="input_ln", dtype=jnp.float32)(x)
        x = nn.Dense(self.hidden_width, name="hidden", dtype=self.dtype_mm)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_width, name="output", dtype=self.dtype_mm)(x)
        x = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(x)
        return x


@dataclasses.dataclass(frozen=True)
class RawMemoryPiJointActionConfig(_direct.FullyVisualJointActionConfig):
    raw_memory_mode: str = "normal"

    def create(self, rng: at.KeyArrayLike) -> RawMemoryPiJointActionModel:
        return RawMemoryPiJointActionModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_raw_memory_pi_action(self) -> nnx.filterlib.Filter:
        adapter = nnx_utils.PathRegex(r".*HistoryRawMemoryToPiPrefix.*")
        action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm/.*_1.*")
        action_modules = nnx_utils.PathRegex(
            r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"
        )
        return nnx.Not(nnx.Any(adapter, action_expert, action_modules))


class RawMemoryPiJointActionModel(_direct.FullyVisualJointActionModel):
    def __init__(self, config: RawMemoryPiJointActionConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.raw_memory_mode = config.raw_memory_mode
        self.HistoryRawMemoryToPiPrefix = nnx_bridge.ToNNX(
            RawMemoryToPiPrefix(
                input_width=64,
                hidden_width=512,
                output_width=2048,
                num_tokens=128,
                dtype_mm=config.dtype,
            )
        )
        self.HistoryRawMemoryToPiPrefix.lazy_init(
            jnp.zeros((1, 128, 64), dtype=jnp.bfloat16), rngs=rngs
        )

    def _embed_prefix_with_raw_memory(self, observation: _model.Observation):
        tracked = self._track_history(observation)
        raw_memory = jax.lax.stop_gradient(tracked["stage_memories"][:, -1])
        if self.raw_memory_mode == "shuffle_batch":
            raw_memory = jnp.roll(raw_memory, 1, axis=0)
        elif self.raw_memory_mode == "zero":
            raw_memory = jnp.zeros_like(raw_memory)
        elif self.raw_memory_mode != "normal":
            raise ValueError(f"Unknown raw_memory_mode={self.raw_memory_mode!r}")
        memory_tokens = self.HistoryRawMemoryToPiPrefix(raw_memory)

        tokens = []
        input_mask = []
        ar_mask: list[bool] = []
        # Pi0 sees only the current image here.  All temporal information must
        # arrive through raw_memory, avoiding a second competing history path.
        for name, video in observation.images.items():
            image = video[:, -1] if video.ndim == 5 else video
            image_tokens, _ = self.PaliGemma.img(image[:, None], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    observation.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        tokens.append(memory_tokens)
        input_mask.append(jnp.ones(memory_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [False] * memory_tokens.shape[1]

        if observation.tokenized_prompt is not None:
            prompt_tokens = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            tokens.append(prompt_tokens)
            input_mask.append(observation.tokenized_prompt_mask)
            ar_mask += [False] * prompt_tokens.shape[1]

        return (
            jnp.concatenate(tokens, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.asarray(ar_mask),
            raw_memory,
            tracked,
        )

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

        prefix_tokens, prefix_mask, prefix_ar_mask, raw_memory, tracked = (
            self._embed_prefix_with_raw_memory(observation)
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
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
        if observation.action_loss_mask is not None:
            mask = observation.action_loss_mask[..., None, :]
        else:
            mask = jnp.asarray(self.action_loss_mask)[None, None, :]
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
            noise = jax.random.normal(
                rng, (batch_size, self.action_horizon, self.action_dim)
            )

        prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = (
            self._embed_prefix_with_raw_memory(observation)
        )
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
            suffix_attn_mask = _pi_mem.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate(
                (prefix_for_suffix, suffix_attn_mask), axis=-1
            )
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
class RawMemoryPiCheckpointLoader:
    params_path: str
    restore_adapter: bool = False

    def load(self, params: at.Params) -> at.Params:
        source = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target_flat = flax.traverse_util.flatten_dict(params, sep="/")
        source_flat = flax.traverse_util.flatten_dict(source, sep="/")
        adapter_prefix = "HistoryRawMemoryToPiPrefix/"
        restored = {}
        counts = {"checkpoint": 0, "random_adapter": 0}
        missing = []
        for key, reference in target_flat.items():
            if key.startswith(adapter_prefix) and not self.restore_adapter:
                restored[key] = reference
                counts["random_adapter"] += 1
                continue
            candidate = source_flat.get(key)
            if candidate is None or np.shape(candidate) != np.shape(reference):
                missing.append(key)
                restored[key] = reference
            else:
                restored[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                counts["checkpoint"] += 1
        if missing:
            raise ValueError(f"Raw-memory Pi restore incomplete: {missing[:8]}")
        print(
            "RawMemoryPiCheckpointLoader: "
            f"checkpoint={counts['checkpoint']}, random_adapter={counts['random_adapter']}, missing=0"
        )
        return flax.traverse_util.unflatten_dict(restored, sep="/")


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    parent = _direct.build_config(args)
    fields = {
        field.name: getattr(parent.model, field.name) for field in dataclasses.fields(parent.model)
    }
    model = RawMemoryPiJointActionConfig(**fields, raw_memory_mode=args.raw_memory_mode)
    return dataclasses.replace(
        parent,
        name="pi0_shellgame_three_swap_raw_memory_pi_joint_action_260810",
        model=model,
        freeze_filter=model.get_freeze_filter_raw_memory_pi_action(),
        weight_loader=RawMemoryPiCheckpointLoader(
            args.tracker_checkpoint, restore_adapter=args.restore_adapter
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 2),
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        save_interval=max(args.steps, 1),
        keep_period=max(args.steps, 1),
        shellgame_cup_eval=dataclasses.replace(
            parent.shellgame_cup_eval,
            interval=args.cup_eval_interval,
            num_episodes=args.cup_eval_episodes,
            batch_size=args.cup_eval_batch_size,
            num_sampling_steps=args.num_sampling_steps,
            sample_seed=260810,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--tracker-checkpoint", default=DEFAULT_TRACKER_CHECKPOINT)
    parser.add_argument(
        "--raw-memory-mode",
        choices=("normal", "shuffle_batch", "zero"),
        default="normal",
    )
    parser.add_argument("--restore-adapter", action="store_true")
    # Arguments consumed by the parent controlled-data configuration.
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
