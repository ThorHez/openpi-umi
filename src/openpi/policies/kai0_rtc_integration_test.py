"""Exercise the kai0 RTC wire and coordinate contracts without model weights."""

import asyncio

import jax.numpy as jnp
import numpy as np
from openpi_client import msgpack_numpy
import pytest

from openpi.policies import policy as policy_module
from openpi.serving import websocket_policy_server


class _ActionModel:
    action_horizon = 8
    action_dim = 4

    def sample_actions(self, _rng, _observation):
        return jnp.arange(32, dtype=jnp.float32).reshape(1, 8, 4)


class _ClientRTCModel(_ActionModel):
    def sample_actions(
        self,
        _rng,
        _observation,
        *,
        prev_action_chunk=None,
        inference_delay=0,
        execute_horizon=None,
        enable_rtc=True,
        mask_prefix_delay=False,
        prefix_attention_schedule="exp",
        max_guidance_weight=0.5,
    ):
        self.received = {
            "prev_action_chunk": prev_action_chunk,
            "inference_delay": inference_delay,
            "execute_horizon": execute_horizon,
            "enable_rtc": enable_rtc,
            "mask_prefix_delay": mask_prefix_delay,
            "prefix_attention_schedule": prefix_attention_schedule,
            "max_guidance_weight": max_guidance_weight,
        }
        return super().sample_actions(_rng, _observation)


@pytest.fixture
def observation():
    return {
        "state": np.arange(4, dtype=np.float32),
        "image": {"camera": np.zeros((1, 1, 3), dtype=np.uint8)},
        "image_mask": {"camera": np.True_},
    }


@pytest.fixture(autouse=True)
def plain_model_calls(monkeypatch):
    # Test Policy's host boundary; model JIT behavior has separate sampler tests.
    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda fn, **_kwargs: fn)


def test_client_options_are_forwarded_before_observation_transforms(observation):
    model = _ClientRTCModel()
    original_chunk = np.ones((3, 2), dtype=np.float32)
    request = {
        **observation,
        "prev_action_chunk": original_chunk,
        "prev_action_chunk_normalized": True,
        "inference_delay": 1,
        "execute_horizon": 3,
        "enable_rtc": False,
        "mask_prefix_delay": True,
        "prefix_attention_schedule": "linear",
        "max_guidance_weight": 0.25,
    }

    def input_transform(inputs):
        assert set(inputs) == set(observation)
        return inputs

    policy = policy_module.Policy(model, transforms=[input_transform])
    policy.infer(request)
    assert policy.supports_client_rtc
    np.testing.assert_array_equal(model.received.pop("prev_action_chunk"), original_chunk[None])
    assert model.received == {
        "inference_delay": 1,
        "execute_horizon": 3,
        "enable_rtc": False,
        "mask_prefix_delay": True,
        "prefix_attention_schedule": "linear",
        "max_guidance_weight": 0.25,
    }
    assert request["prev_action_chunk"] is original_chunk
    assert request["prev_action_chunk_normalized"] is True


def test_physical_prefix_is_transformed_using_current_observation(observation):
    model = _ClientRTCModel()
    seen = []
    physical_chunk = np.full((3, 4), 10, dtype=np.float32)

    def action_transform(obs, actions):
        assert "prev_action_chunk" not in obs
        seen.append(np.asarray(obs["state"]).copy())
        return (actions - obs["state"]) / 2

    policy = policy_module.Policy(model, rtc_action_transform=action_transform)
    policy.infer({**observation, "prev_action_chunk": physical_chunk, "execute_horizon": 3})
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], observation["state"])
    np.testing.assert_array_equal(
        model.received["prev_action_chunk"], ((physical_chunk - observation["state"]) / 2)[None]
    )
    np.testing.assert_array_equal(physical_chunk, np.full((3, 4), 10))


def test_normalized_prefix_bypasses_physical_transform(observation):
    model = _ClientRTCModel()

    def action_transform(_obs, _actions):
        pytest.fail("Normalized model actions must not be normalized a second time")

    policy = policy_module.Policy(model, rtc_action_transform=action_transform)
    policy.infer(
        {
            **observation,
            "prev_action_chunk": [[1.0, 2.0]],
            "prev_action_chunk_normalized": True,
            "execute_horizon": 1,
        }
    )
    np.testing.assert_array_equal(model.received["prev_action_chunk"], [[[1.0, 2.0]]])


def test_sampling_defaults_remain_overridable_by_request(observation):
    model = _ClientRTCModel()
    policy = policy_module.Policy(
        model,
        sample_kwargs={"prefix_attention_schedule": "linear", "max_guidance_weight": 0.25},
    )
    policy.infer({**observation, "prev_action_chunk": [[1.0, 2.0]], "prev_action_chunk_normalized": True})
    assert model.received["prefix_attention_schedule"] == "linear"
    assert model.received["max_guidance_weight"] == 0.25
    policy.infer({**observation, "max_guidance_weight": 0.125})
    assert model.received["max_guidance_weight"] == 0.125


def test_physical_prefix_requires_explicit_action_preprocessing(observation):
    policy = policy_module.Policy(_ClientRTCModel())
    with pytest.raises(ValueError, match=r"(?i)(transform|normaliz)"):
        policy.infer({**observation, "prev_action_chunk": np.ones((8, 4))})


def test_model_actions_preserve_coordinate_space_after_output_transform(observation):
    def output_transform(result):
        # Output adapters may replace dictionaries and convert dimensions/units.
        return {"actions": result["actions"][:, :2] * 2 + 100}

    policy = policy_module.Policy(_ClientRTCModel(), output_transforms=[output_transform])
    result = policy.infer(observation)
    normalized = np.arange(32, dtype=np.float32).reshape(8, 4)
    np.testing.assert_array_equal(result["actions"], normalized[:, :2] * 2 + 100)
    np.testing.assert_array_equal(result["model_actions"], normalized)
    np.testing.assert_array_equal(policy.last_model_actions, normalized)
    result["model_actions"][0, 0] = -100
    np.testing.assert_array_equal(policy.last_model_actions, normalized)


def test_ordinary_inference_keeps_existing_response_contract(observation):
    policy = policy_module.Policy(_ActionModel())
    result = policy.infer(observation)
    assert not policy.supports_client_rtc
    assert "model_actions" not in result
    np.testing.assert_array_equal(result["actions"], np.arange(32).reshape(8, 4))


@pytest.mark.parametrize("field", ["prev_action_chunk", "inference_delay", "enable_rtc"])
def test_ordinary_models_reject_rtc_fields_before_transforms(observation, field):
    def input_transform(_inputs):
        pytest.fail("Unsupported RTC requests must fail before tokenization")

    policy = policy_module.Policy(_ActionModel(), transforms=[input_transform])
    with pytest.raises(ValueError, match=r"(?i)RTC"):
        policy.infer({**observation, field: 1})


def test_client_and_server_guidance_cannot_be_mixed(observation):
    policy = policy_module.Policy(_ClientRTCModel())
    with pytest.raises(ValueError, match=r"(?i)(mix|combin|both|client|server)"):
        policy.infer(
            {**observation, "enable_rtc": True},
            rtc_actions=np.ones((8, 4)),
            rtc_mask=np.ones(8),
        )


@pytest.mark.parametrize(
    "options",
    [
        {"inference_delay": -1},
        {"inference_delay": 4, "execute_horizon": 3},
        {"execute_horizon": 9},
        {"execute_horizon": 0},
        {"inference_delay": 0.5},
        {"max_guidance_weight": -1},
        {"max_guidance_weight": float("nan")},
        {"prefix_attention_schedule": "unknown"},
        {"prev_action_chunk": np.ones((9, 4))},
        {"prev_action_chunk": np.ones((8, 5))},
        {"prev_action_chunk": np.ones(4)},
        {"prev_action_chunk": np.zeros((0, 4))},
        {"prev_action_chunk": np.full((8, 4), np.nan)},
    ],
)
def test_invalid_rtc_requests_fail_at_policy_boundary(observation, options):
    model = _ClientRTCModel()
    policy = policy_module.Policy(model)
    request = {
        **observation,
        "prev_action_chunk": np.ones((8, 4)),
        "prev_action_chunk_normalized": True,
        **options,
    }
    with pytest.raises(ValueError, match=r".+"):
        policy.infer(request)
    assert not hasattr(model, "received")


def test_websocket_payload_round_trip_retains_normalized_prefix(observation):
    model = _ClientRTCModel()
    policy = policy_module.Policy(model)
    request = {
        **observation,
        "prev_action_chunk": np.ones((3, 2), dtype=np.float32),
        "prev_action_chunk_normalized": True,
        "inference_delay": 1,
        "execute_horizon": 3,
        "mask_prefix_delay": True,
        "prefix_attention_schedule": "ones",
        "max_guidance_weight": 0.125,
    }

    class Connection:
        remote_address = ("rtc-controller", 0)

        def __init__(self):
            self.messages = []
            self.reads = 0

        async def send(self, value):
            self.messages.append(value)

        async def recv(self):
            self.reads += 1
            if self.reads == 1:
                return msgpack_numpy.packb(request)
            raise asyncio.CancelledError

    connection = Connection()
    server = websocket_policy_server.WebsocketPolicyServer(policy)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server._handler(connection))  # noqa: SLF001

    assert len(connection.messages) == 2
    response = msgpack_numpy.unpackb(connection.messages[1])
    assert response["actions"].shape == (8, 4)
    np.testing.assert_array_equal(response["model_actions"], np.arange(32).reshape(8, 4))
    np.testing.assert_array_equal(model.received["prev_action_chunk"], request["prev_action_chunk"][None])
    assert model.received["mask_prefix_delay"] is True
    assert model.received["prefix_attention_schedule"] == "ones"
    assert model.received["max_guidance_weight"] == 0.125
