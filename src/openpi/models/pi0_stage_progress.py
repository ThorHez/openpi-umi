"""Paired-frame stage progress regression on an OpenPI π0.5 backbone."""

import dataclasses
import math

from flax import nnx
import jax.numpy as jnp

from openpi.models import gemma
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask


@dataclasses.dataclass(frozen=True)
class Pi0StageProgressConfig(pi0_config.Pi0Config):
    pi05: bool = True
    discrete_state_input: bool = False
    n_step: int = 50
    positive_ratio: float = 0.3
    stage_count: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.pi05 is not True or self.discrete_state_input is not False:
            raise ValueError("阶段进度估计器要求 π0.5 且关闭离散状态输入")
        if (type(self.n_step) is not int or not 1 <= self.n_step <= 1000
                or type(self.stage_count) is not int or not 1 <= self.stage_count <= 100):
            raise ValueError("阶段进度的优势窗口须为 1–1000 帧，阶段数须为 1–100")
        if (type(self.positive_ratio) not in (int, float) or not math.isfinite(self.positive_ratio)
                or not 0.001 <= self.positive_ratio <= 1):
            raise ValueError("阶段进度的正例比例须为 0.001–1")

    def create(self, rng):
        return Pi0StageProgress(self, rngs=nnx.Rngs(rng))


class Pi0StageProgress(Pi0):
    """Regress progress(current) - progress(reference) from images and task text."""

    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        width = gemma.get_config(config.action_expert_variant).width
        self.stage_progress_fc1 = nnx.Linear(width, width, rngs=rngs)
        self.stage_progress_fc2 = nnx.Linear(width, width, rngs=rngs)
        self.stage_progress_out = nnx.Linear(width, 1, rngs=rngs)

    def predict_progress(self, reference, current):
        # The task prompt is shared by the pair. Keep visual tokens in reference
        # then current order so the same backbone can score both directions.
        reference = dataclasses.replace(reference, tokenized_prompt=None, tokenized_prompt_mask=None)
        reference = self.preprocess_observation(None, reference, train=False)
        current = self.preprocess_observation(None, current, train=False)
        reference_tokens, reference_mask, reference_ar = self.embed_prefix(reference)
        current_tokens, current_mask, current_ar = self.embed_prefix(current)
        prefix = jnp.concatenate((reference_tokens, current_tokens), axis=1)

        # Use the same deterministic action-expert query in training and
        # annotation. Recorded actions never enter the progress estimator.
        batch_size = current.state.shape[0]
        dummy_actions = jnp.zeros((batch_size, self.action_horizon, self.action_dim), dtype=jnp.float32)
        midpoint = jnp.full((batch_size,), 0.5, dtype=jnp.float32)
        suffix, suffix_mask, suffix_ar, condition = self.embed_suffix(current, dummy_actions, midpoint)
        suffix, suffix_mask, suffix_ar = suffix[:, :1], suffix_mask[:, :1], suffix_ar[:1]
        mask = jnp.concatenate((reference_mask, current_mask, suffix_mask), axis=1)
        ar_mask = jnp.concatenate((reference_ar, current_ar, suffix_ar))
        positions = jnp.cumsum(mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix, suffix], mask=make_attn_mask(mask, ar_mask), positions=positions,
            adarms_cond=[None, condition],
        )
        features = suffix_out[:, 0, :].astype(jnp.float32)
        features = nnx.swish(self.stage_progress_fc1(features))
        features = nnx.swish(self.stage_progress_fc2(features))
        return jnp.tanh(self.stage_progress_out(features)[:, 0])
