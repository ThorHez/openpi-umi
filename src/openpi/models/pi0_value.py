import logging
import dataclasses

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

# pi0.6-style distributional value: normalize to (-1, 0], discretize into 201 bins
VALUE_BINS = 201
VALUE_MIN = -1.0
VALUE_MAX = 0.0


def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@dataclasses.dataclass(frozen=True)
class Pi0ValueConfig(pi0_config.Pi0Config):
    """Prefix-only distributional value config.

    The model outputs per-sample value distribution [B, VALUE_BINS].
    Target follows pi0.6 shaped reward and per-task normalization.
    """

    gamma: float = 1.0
    c_fail_mult: float = 1.0  # C_fail = c_fail_mult * task_max_steps
    task_max_steps: int = 1500  # Maximum steps for the task (used for normalization)

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Value":
        return Pi0Value(self, rngs=nnx.Rngs(rng))


class Pi0Value(_model.BaseModel):
    """Prefix-only distributional value model (no action expert suffix).

    Output: logits [B, 201] for value/return in [VALUE_MIN, VALUE_MAX]=[-1,0].
    Target: normalized Monte-Carlo return R_t0_norm for the current step t0.

    Shaped reward (pi0.6):
      r_t = -1 for t < T
      r_T = 0 if success else -C_fail

    Return:
      R_t = sum_{t'=t}^T r_{t'} = -(T - t) + r_T

    Normalization (per-task):
      R_norm = clip(R / task_max_steps, -1, 0)
    """

    def __init__(self, config: Pi0ValueConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        self.gamma = float(getattr(config, "gamma", 1.0))
        self.c_fail_mult = float(getattr(config, "c_fail_mult", 1.0))
        self.task_max_steps = int(getattr(config, "task_max_steps", 100))

        paligemma_config = _gemma.get_config(config.paligemma_variant)

        # Backbone: we still instantiate as in repo; but we only feed prefix stream.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config],
                embed_dtype=config.dtype,
                adarms=False,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])

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

        width = paligemma_config.width
        self.value_token = nnx.Param(jax.random.normal(rngs(), (1, width)) * 0.02)
        self.value_head = nnx.Linear(width, VALUE_BINS, rngs=rngs)

        self.deterministic = True

    # ---------- bins ----------
    def _value_to_bins(self, values: at.Float[at.Array, "*b"]) -> at.Int[at.Array, "*b"]:
        clipped = jnp.clip(values, VALUE_MIN, VALUE_MAX)
        normalized = (clipped - VALUE_MIN) / (VALUE_MAX - VALUE_MIN + 1e-8)
        bins = jnp.rint(normalized * (VALUE_BINS - 1)).astype(jnp.int32)
        return jnp.clip(bins, 0, VALUE_BINS - 1)

    def _bins_to_value(self, bins: at.Int[at.Array, "*b"]) -> at.Float[at.Array, "*b"]:
        normalized = bins.astype(jnp.float32) / (VALUE_BINS - 1)
        return normalized * (VALUE_MAX - VALUE_MIN) + VALUE_MIN

    # ---------- prefix embedding ----------
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        # images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        # language
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    # ---------- pi0.6 target ----------
    def _extract_episode_info(self, obs: _model.Observation):
        """Adapt these field names to your dataset."""
        # terminal reward or success label at end of episode
        if hasattr(obs, "terminal_reward"):
            terminal_reward = obs.terminal_reward  # [B]
        elif hasattr(obs, "reward"):
            terminal_reward = obs.reward  # [B] (your dataset: only last frame has reward)
        else:
            raise AttributeError("Observation must have `terminal_reward` or `reward`.")

        if not hasattr(obs, "step_index"):
            raise AttributeError("Observation must have `step_index` (current timestep in episode).")
        if not hasattr(obs, "episode_T"):
            raise AttributeError("Observation must have `episode_T` (terminal timestep index).")

        task_max_steps = self.task_max_steps

        return terminal_reward, obs.step_index, obs.episode_T, task_max_steps

    def _paper_return_target_scalar(self, obs: _model.Observation) -> at.Float[at.Array, "b"]:
        """Return normalized scalar target R_norm(t0) in [-1,0] for current step t0."""
        terminal_reward, step_index, episode_T, task_max_steps = self._extract_episode_info(obs)

        # success criterion: >0 => success else failure (adjust if your labels differ)
        success = terminal_reward > 0

        t0 = step_index.astype(jnp.int32)            # [B]
        T = episode_T.astype(jnp.int32)              # [B]
        task_max = float(task_max_steps)             # scalar

        # C_fail per-task (paper: constant; using proportional to max steps is robust)
        C_fail = self.c_fail_mult * task_max
        r_T = jnp.where(success, 0.0, -C_fail)       # [B]

        remaining = (T - t0).astype(jnp.float32)     # [B]
        remaining = jnp.maximum(remaining, 0.0)

        R = -remaining + r_T                         # [B]

        # normalize to [-1,0]
        R_norm = R / (task_max + 1e-8)
        R_norm = jnp.clip(R_norm, -1.0, 0.0)
        return R_norm

    # ---------- pooling ----------
    def _pool_prefix(self, seq: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Pool prefix sequence -> single embedding [B, emb].

        Default: take the last valid token (works well for prefix-only).
        """
        # mask: [B, S] bool
        lengths = jnp.sum(mask, axis=1).astype(jnp.int32)           # [B]
        last_idx = jnp.maximum(lengths - 1, 0)                      # [B]
        return seq[jnp.arange(seq.shape[0]), last_idx, :]           # [B, emb]

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b"]:
        preprocess_rng, _ = jax.random.split(rng, 2)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # prefix forward only
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)


        B = prefix_tokens.shape[0]
        v = jnp.broadcast_to(self.value_token.value[None, :, :], (B, 1, self.value_token.value.shape[-1]))
        v_mask = jnp.ones((B, 1), dtype=jnp.bool_)

        # 让 value token 在一个新 block（可以 attend 到 prefix）
        v_ar = jnp.array([True])  # 新 block，能看之前所有 token

        tokens = jnp.concatenate([prefix_tokens, v], axis=1)      # [B, S+1, emb]
        mask = jnp.concatenate([prefix_mask, v_mask], axis=1)   # [B, S+1]
        ar_mask = jnp.concatenate([prefix_ar_mask, v_ar], axis=0)  # [S+1]

        # full attention within prefix
        attn_mask = make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1

        prefix_out, _ = self.PaliGemma.llm([tokens], mask=attn_mask, positions=positions, adarms_cond=[None])

        v_out = prefix_out[:, -1, :]
        value_logits = self.value_head(v_out)                      # [B, 201]

        # scalar return target from pi0.6 shaped reward
        R_norm = self._paper_return_target_scalar(observation)      # [B] in [-1,0]
        target_bins = self._value_to_bins(R_norm)                   # [B]

        log_probs = jax.nn.log_softmax(value_logits, axis=-1)       # [B, 201]
        target_log_probs = jnp.take_along_axis(log_probs, target_bins[..., None], axis=-1)[..., 0]  # [B]
        loss = -target_log_probs                                    # [B]
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        raise NotImplementedError("Pi0Value predicts values/returns, not actions.")


    # Add these methods inside your Pi0Value class (the prefix-only [B, 201] value model).

    def _value_from_logits_expectation(
        self, value_logits: at.Float[at.Array, "b vb"]
    ) -> at.Float[at.Array, "b"]:
        """Convert distributional logits [B, 201] -> scalar V_norm [B] via expectation."""
        probs = jax.nn.softmax(value_logits, axis=-1)  # [B, 201]
        bin_centers = jnp.linspace(VALUE_MIN, VALUE_MAX, VALUE_BINS, dtype=probs.dtype)  # [-1, 0]
        return jnp.sum(probs * bin_centers[None, :], axis=-1)  # [B]


    def _value_from_logits_map(
        self, value_logits: at.Float[at.Array, "b vb"]
    ) -> at.Float[at.Array, "b"]:
        """Convert logits [B, 201] -> scalar V_norm [B] via MAP bin (argmax)."""
        bins = jnp.argmax(value_logits, axis=-1).astype(jnp.float32)  # [B]
        normalized = bins / (VALUE_BINS - 1.0)
        return normalized * (VALUE_MAX - VALUE_MIN) + VALUE_MIN       # [B]


    def compute_advantage(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
        value_reduce: str = "expectation",  # "expectation" | "map"
        stopgrad_value: bool = True,
    ) -> tuple[at.Float[at.Array, "b"], at.Float[at.Array, "b"], at.Float[at.Array, "b"]]:
        preprocess_rng, _ = jax.random.split(rng, 2)
        obs = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # 1) Return target (normalized)
        R_norm = self._paper_return_target_scalar(obs)  # [B] in [-1, 0]

        # 2) Forward exactly like compute_loss (prefix + value token)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)

        B = prefix_tokens.shape[0]
        v = jnp.broadcast_to(self.value_token.value[None, :, :], (B, 1, self.value_token.value.shape[-1]))
        v_mask = jnp.ones((B, 1), dtype=jnp.bool_)
        v_ar = jnp.array([True])

        tokens = jnp.concatenate([prefix_tokens, v], axis=1)
        mask = jnp.concatenate([prefix_mask, v_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, v_ar], axis=0)

        attn_mask = make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1

        (out_list, _kv) = self.PaliGemma.llm([tokens], mask=attn_mask, positions=positions, adarms_cond=[None])
        prefix_out = out_list[0]
        v_out = prefix_out[:, -1, :]
        value_logits = self.value_head(v_out)  # [B, 201]

        # 3) logits -> scalar V_norm
        if value_reduce == "map":
            V_norm = self._value_from_logits_map(value_logits)
        else:
            V_norm = self._value_from_logits_expectation(value_logits)

        if stopgrad_value:
            V_norm = jax.lax.stop_gradient(V_norm)

        # 4) Advantage (MC): A = R - V
        A_norm = R_norm - V_norm
        return A_norm, R_norm, V_norm

