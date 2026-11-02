"""Exercise the RTC integrator and checkpoint contract without loading weights."""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import pi0_rtc
from openpi.policies import policy as _policy
from openpi.shared import nnx_utils


class _ZeroFlowLLM(nnx.Module):
    def __call__(self, inputs, **_kwargs):
        return (None, None if inputs[1] is None else jnp.zeros_like(inputs[1])), None


class _ToyRTC(pi0_rtc.Pi0RTC):
    def __init__(self):
        self.action_horizon = 4
        self.action_dim = 20
        self.PaliGemma = nnx.Dict(llm=_ZeroFlowLLM())

    def preprocess_observation(self, _rng, observation, **_kwargs):
        return observation

    def embed_prefix(self, observation):
        batch = observation.state.shape[0]
        return jnp.zeros((batch, 1, self.action_dim)), jnp.ones((batch, 1), dtype=bool), jnp.array([False])

    def embed_suffix(self, _observation, actions, _time):
        return actions, jnp.ones(actions.shape[:2], dtype=bool), jnp.array([True, False, False, False]), None

    def action_out_proj(self, value):
        return value


def _observation():
    return _model.Observation(images={}, image_masks={}, state=jnp.zeros((1, 20)))


@pytest.mark.parametrize("schedule", ["ones", "zeros", "linear", "exp"])
def test_prefix_schedules_match_kai0(schedule):
    indices = np.arange(6)
    ramp = np.clip((1 - indices) / 4 + 1, 0, 1)
    expected = {
        "ones": np.ones(6),
        "zeros": (indices < 2).astype(float),
        "linear": ramp,
        "exp": ramp * np.expm1(ramp) / np.expm1(1.0),
    }[schedule]
    expected[indices >= 5] = 0
    actual = jax.jit(lambda start, end: pi0_rtc.get_prefix_weights(start, end, 6, schedule))(2, 5)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_invalid_prefix_schedule_fails():
    with pytest.raises(ValueError, match="Invalid RTC prefix schedule"):
        pi0_rtc.get_prefix_weights(1, 2, 4, "invalid")


def test_jitted_short_narrow_prefix_guides_all_provided_dimensions():
    model = _ToyRTC()
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("prefix_attention_schedule",))
    result = sample(
        jax.random.key(0),
        _observation(),
        num_steps=2,
        noise=jnp.zeros((1, 4, 20)),
        prev_action_chunk=jnp.ones((1, 2, 17)),
        inference_delay=1,
        execute_horizon=2,
        prefix_attention_schedule="ones",
        max_guidance_weight=0.5,
    )
    assert np.all(np.asarray(result[:, :2, :17]) > 0)
    np.testing.assert_array_equal(result[:, 2:], 0)
    np.testing.assert_array_equal(result[..., 17:], 0)


@pytest.mark.parametrize("disabled", [{"enable_rtc": False}, {"max_guidance_weight": 0.0}])
def test_disabled_guidance_matches_ordinary_sampling(disabled):
    model = _ToyRTC()
    noise = jax.random.normal(jax.random.key(1), (1, 4, 20))
    expected = pi0.Pi0.sample_actions(model, jax.random.key(0), _observation(), noise=noise, num_steps=3)
    sample = nnx_utils.module_jit(model.sample_actions)
    result = sample(
        jax.random.key(0),
        _observation(),
        noise=noise,
        num_steps=3,
        prev_action_chunk=jnp.ones((1, 2, 17)),
        mask_prefix_delay=True,
        **disabled,
    )
    np.testing.assert_array_equal(result, expected)


def test_mask_prefix_delay_is_honored_under_jit():
    model = _ToyRTC()
    sample = nnx_utils.module_jit(model.sample_actions)
    kwargs = {
        "noise": jnp.zeros((1, 4, 20)),
        "num_steps": 2,
        "prev_action_chunk": jnp.ones((1, 2, 17)),
        "inference_delay": 1,
        "execute_horizon": 2,
    }
    without = sample(jax.random.key(0), _observation(), mask_prefix_delay=False, **kwargs)
    with_mask = sample(jax.random.key(0), _observation(), mask_prefix_delay=True, **kwargs)
    assert np.all(np.asarray(without[:, 0, :17]) > 0)
    np.testing.assert_array_equal(with_mask[:, 0], 0)
    np.testing.assert_array_equal(with_mask[:, 1:], without[:, 1:])


def test_fixed_shape_prefix_supports_dynamic_valid_width_under_jit():
    model = _ToyRTC()
    sample = nnx_utils.module_jit(model.sample_actions)
    noise = jnp.zeros((1, 4, 20))
    kwargs = {
        "noise": noise,
        "num_steps": 2,
        "prev_action_chunk": jnp.ones((1, 4, 20)),
        "inference_delay": 1,
        "execute_horizon": 2,
    }
    expected = pi0.Pi0.sample_actions(model, jax.random.key(0), _observation(), noise=noise, num_steps=2)
    disabled = sample(jax.random.key(0), _observation(), prev_action_dim=0, **kwargs)
    guided = sample(jax.random.key(0), _observation(), prev_action_dim=17, **kwargs)
    np.testing.assert_array_equal(disabled, expected)
    assert np.all(np.asarray(guided[:, :2, :17]) > 0)
    np.testing.assert_array_equal(guided[:, 2:], 0)
    np.testing.assert_array_equal(guided[..., 17:], 0)


def test_real_policy_keeps_sampling_shapes_fixed_across_shrinking_wire_prefixes(monkeypatch):
    policy = _policy.Policy(_ToyRTC(), sample_kwargs={"num_steps": 2})
    sample_actions = policy._sample_actions  # noqa: SLF001
    recorded_shapes = []
    recorded_keys = []

    def record_sampling(*args, **kwargs):
        recorded_shapes.append(kwargs["prev_action_chunk"].shape)
        recorded_keys.append(set(kwargs))
        return sample_actions(*args, **kwargs)

    monkeypatch.setattr(policy, "_sample_actions", record_sampling)
    observation = {"state": np.zeros(20, dtype=np.float32), "image": {}, "image_mask": {}}
    noise = np.zeros((4, 20), dtype=np.float32)
    first = policy.infer(observation, noise=noise)
    np.testing.assert_array_equal(first["actions"], 0)
    for horizon, dimensions in ((3, 17), (2, 12), (1, 5)):
        result = policy.infer(
            {
                **observation,
                "prev_action_chunk": np.ones((horizon, dimensions), dtype=np.float32),
                "prev_action_chunk_normalized": True,
            },
            noise=noise,
        )
        actions = result["actions"]
        assert np.all(actions[:horizon, :dimensions] > 0)
        np.testing.assert_array_equal(actions[horizon:], 0)
        np.testing.assert_array_equal(actions[:, dimensions:], 0)
    assert recorded_shapes == [(1, 4, 20)] * 4
    assert all(keys == recorded_keys[0] for keys in recorded_keys)


@pytest.mark.parametrize("pi05", [False, True])
def test_rtc_config_keeps_checkpoint_model_type(pi05):
    ordinary = pi0_config.Pi0Config(pi05=pi05, image_keys=("camera",), action_horizon=16)
    rtc = pi0_config.Pi0RTCConfig(pi05=pi05, image_keys=("camera",), action_horizon=16)
    assert rtc.model_type == ordinary.model_type
    assert rtc.inputs_spec()[0].images.keys() == ordinary.inputs_spec()[0].images.keys()
    with pytest.raises(NotImplementedError, match="JAX checkpoint"):
        rtc.load_pytorch(None, "model.safetensors")


@pytest.mark.parametrize("pi05", [False, True])
def test_rtc_accepts_complete_base_config_and_loads_its_parameter_tree(pi05):
    ordinary = pi0_config.Pi0Config(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=pi05,
        action_dim=20,
        action_horizon=4,
        max_token_len=12,
        discrete_state_input=pi05,
        action_loss_mask=(1.0,) * 17 + (0.0,) * 3,
        image_keys=("camera",),
        depth_keys=(),
        allow_all_masked_images=True,
    )
    rtc = pi0_config.Pi0RTCConfig(
        **{field.name: getattr(ordinary, field.name) for field in dataclasses.fields(ordinary)}
    )
    key = jax.random.key(0)
    ordinary_model = nnx.eval_shape(ordinary.create, key)
    rtc_model = nnx.eval_shape(rtc.create, key)
    assert isinstance(rtc_model, pi0_rtc.Pi0RTC)

    def parameter_contract(model):
        return {
            path: (variable.value.shape, variable.value.dtype)
            for path, variable in nnx.state(model, nnx.Param).flat_state().items()
        }

    expected = parameter_contract(ordinary_model)
    assert expected
    assert parameter_contract(rtc_model) == expected
    # Exercise BaseModelConfig.load's structure/shape checks and state restore
    # under eval_shape, avoiding allocation of the shared SigLIP backbone.
    params = nnx.state(ordinary_model, nnx.Param).to_pure_dict()
    loaded = nnx.eval_shape(rtc.load, params)
    assert isinstance(loaded, pi0_rtc.Pi0RTC)
    assert parameter_contract(loaded) == expected
