import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def infer_gripper_indices(action_dim: int, explicit=None) -> tuple[int, ...]:
    """Infer gripper indices from action_dim unless explicitly provided.

    Priority:
      1) explicit config
      2) if action_dim % 10 == 0 -> use [9, 19, 29, ...]
      3) if action_dim % 7 == 0  -> use [6, 13, 20, ...]
      4) else return empty tuple
    """
    if explicit is not None:
        return tuple(int(i) for i in explicit)

    if action_dim % 10 == 0:
        return tuple(range(9, action_dim, 10))
    if action_dim % 7 == 0:
        return tuple(range(6, action_dim, 7))

    logger.warning(
        "Could not infer gripper indices from action_dim=%d. "
        "Binary gripper head will be disabled unless config.gripper_binary_indices is set.",
        action_dim,
    )
    return ()


def binary_cross_entropy_with_logits(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Stable BCE with logits."""
    return jnp.maximum(logits, 0.0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))


@dataclasses.dataclass(frozen=True)
class Pi0GripperConfig(pi0_config.Pi0Config):
    """Pi0/Pi0.5 config with an additional binary gripper head."""

    gripper_binary_indices: tuple[int, ...] | None = None
    gripper_binary_loss_weight: float = 1.0
    gripper_binary_close_value: float = 0.0
    gripper_binary_open_value: float = 0.085
    gripper_binary_threshold: float | None = None

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        # Store action loss mask if provided (for masking out padded action dimensions)
        # Store as tuple to avoid issues with jax.eval_shape, convert to array when used
        self.action_loss_mask = config.action_loss_mask

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        # Original continuous action head
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # ----- binary gripper head -----
        # Expected optional config fields:
        #   gripper_binary_indices: tuple[int, ...] | None = None
        #   gripper_binary_loss_weight: float = 1.0
        #   gripper_binary_close_value: float = 0.0
        #   gripper_binary_open_value: float = 0.085
        #   gripper_binary_threshold: float | None = None
        explicit_gripper_indices = getattr(config, "gripper_binary_indices", None)
        self.gripper_binary_indices = infer_gripper_indices(config.action_dim, explicit_gripper_indices)
        self.num_grippers = len(self.gripper_binary_indices)

        self.gripper_binary_loss_weight = float(getattr(config, "gripper_binary_loss_weight", 1.0))
        self.gripper_binary_close_value = float(getattr(config, "gripper_binary_close_value", 0.0))
        self.gripper_binary_open_value = float(getattr(config, "gripper_binary_open_value", 0.085))

        threshold = getattr(config, "gripper_binary_threshold", None)
        if threshold is None:
            threshold = 0.5 * (self.gripper_binary_close_value + self.gripper_binary_open_value)
        self.gripper_binary_threshold = float(threshold)

        if self.num_grippers > 0:
            self.gripper_binary_head = nnx.Linear(
                action_expert_config.width,
                self.num_grippers,
                rngs=rngs,
            )
            logger.info(
                "Enable binary gripper head: indices=%s, loss_weight=%.4f, close_value=%.4f, open_value=%.4f, threshold=%.4f",
                self.gripper_binary_indices,
                self.gripper_binary_loss_weight,
                self.gripper_binary_close_value,
                self.gripper_binary_open_value,
                self.gripper_binary_threshold,
            )
        else:
            self.gripper_binary_head = None
            logger.warning("Binary gripper head is disabled because no gripper indices were found.")

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _get_gripper_targets(self, actions: jax.Array) -> jax.Array:
        """Return binary labels for gripper dims.

        Convention:
          closed -> 1
          open   -> 0
        """
        if self.num_grippers == 0:
            raise ValueError("No gripper indices configured for binary gripper head.")

        gripper_values = actions[..., self.gripper_binary_indices]  # (..., ah, ng)
        return (gripper_values <= self.gripper_binary_threshold).astype(jnp.float32)

    def _get_gripper_logits(self, suffix_out: jax.Array) -> jax.Array | None:
        """Predict binary gripper logits from action-token hidden states."""
        if self.gripper_binary_head is None:
            return None
        action_hidden = suffix_out[:, -self.action_horizon :]  # (b, ah, emb)
        return self.gripper_binary_head(action_hidden)  # (b, ah, ng)

    def _overwrite_gripper_with_binary(self, actions: jax.Array, gripper_logits: jax.Array | None) -> jax.Array:
        """Overwrite continuous gripper dims with binary open/close values."""
        if gripper_logits is None or self.num_grippers == 0:
            return actions

        closed = (jax.nn.sigmoid(gripper_logits) > 0.5).astype(actions.dtype)  # 1=closed, 0=open
        binary_values = (
            closed * self.gripper_binary_close_value
            + (1.0 - closed) * self.gripper_binary_open_value
        )  # (b, ah, ng)

        out = actions
        for i, idx in enumerate(self.gripper_binary_indices):
            out = out.at[..., idx].set(binary_values[..., i])
        return out

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        action_hidden = suffix_out[:, -self.action_horizon :]
        v_t = self.action_out_proj(action_hidden)

        # ---------------- continuous action loss ----------------
        squared_error = jnp.square(v_t - u_t)  # shape: (*batch, ah, ad)

        if observation.action_loss_mask is not None:
            # Per-sample mask shape (*batch, ad) -> expand to (*batch, 1, ad)
            mask = observation.action_loss_mask[..., None, :]  # (*batch, 1, ad)
            squared_error_masked = squared_error * mask
            mask_sum = jnp.sum(mask, axis=-1, keepdims=True)
            mask_sum = jnp.maximum(mask_sum, 1e-8)
            cont_loss_per_timestep = jnp.sum(squared_error_masked, axis=-1) / jnp.squeeze(mask_sum, axis=-1)
        elif self.action_loss_mask is not None:
            # Config-level mask
            mask = jnp.asarray(self.action_loss_mask)
            squared_error_masked = squared_error * mask
            cont_loss_per_timestep = jnp.sum(squared_error_masked, axis=-1) / jnp.sum(mask)
        else:
            cont_loss_per_timestep = jnp.mean(squared_error, axis=-1)

        # ---------------- binary gripper BCE loss ----------------
        if self.gripper_binary_head is not None and self.num_grippers > 0:
            gripper_logits = self.gripper_binary_head(action_hidden)  # (*batch, ah, ng)
            gripper_targets = self._get_gripper_targets(actions)      # (*batch, ah, ng)
            gripper_bce = binary_cross_entropy_with_logits(gripper_logits, gripper_targets)

            if observation.action_loss_mask is not None:
                # observation.action_loss_mask: (*batch, ad)
                gripper_mask = observation.action_loss_mask[..., self.gripper_binary_indices]  # (*batch, ng)
                gripper_mask = gripper_mask[..., None, :]  # (*batch, 1, ng)
                gripper_bce = gripper_bce * gripper_mask
                denom = jnp.maximum(jnp.sum(gripper_mask, axis=-1), 1e-8)  # (*batch, 1)
                gripper_loss_per_timestep = jnp.sum(gripper_bce, axis=-1) / denom
            elif self.action_loss_mask is not None:
                full_mask = jnp.asarray(self.action_loss_mask)
                gripper_mask = full_mask[jnp.array(self.gripper_binary_indices)]  # (ng,)
                gripper_bce = gripper_bce * gripper_mask
                denom = jnp.maximum(jnp.sum(gripper_mask), 1e-8)
                gripper_loss_per_timestep = jnp.sum(gripper_bce, axis=-1) / denom
            else:
                gripper_loss_per_timestep = jnp.mean(gripper_bce, axis=-1)

            loss_per_timestep = cont_loss_per_timestep + self.gripper_binary_loss_weight * gripper_loss_per_timestep
        else:
            loss_per_timestep = cont_loss_per_timestep

        return loss_per_timestep

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len)
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

        # Final binary gripper decode at near-zero time, then overwrite gripper dims.
        if self.gripper_binary_head is not None and self.num_grippers > 0:
            final_time = jnp.full((batch_size,), 1e-3, dtype=x_0.dtype)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_0, final_time
            )

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None

            gripper_logits = self._get_gripper_logits(suffix_out)
            x_0 = self._overwrite_gripper_with_binary(x_0, gripper_logits)

        return x_0