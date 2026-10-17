"""Check ordinary, memory, and RTC policy contracts without loading weights."""

import asyncio

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.policies import policy as policy_module
from openpi.policies import rtc_policy
from openpi.serving import websocket_policy_server


class _ActionModel:
    action_horizon = 8

    def sample_actions(self, _rng, _observation):
        return jnp.arange(self.action_horizon, dtype=jnp.float32).reshape(1, self.action_horizon, 1)


class _MemoryModel(_ActionModel):
    def compute_history_classification(self, _rng, _observation):
        return jnp.array([[2.0, 3.0]]), {"confidence": jnp.array([0.75])}


class _GuidedModel(_ActionModel):
    def sample_actions(self, _rng, _observation, *, rtc_actions=None, rtc_mask=None, rtc_guidance_weight=5.0):
        self.guidance = (rtc_actions, rtc_mask, rtc_guidance_weight)
        return super().sample_actions(_rng, _observation)


@pytest.fixture
def observation():
    return {
        "state": np.zeros(1, dtype=np.float32),
        "image": {"camera": np.zeros((1, 1, 3), dtype=np.uint8)},
        "image_mask": {"camera": np.True_},
    }


@pytest.fixture(autouse=True)
def plain_model_calls(monkeypatch):
    # These model doubles exercise Policy's data boundary, without NNX graphs.
    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda fn: fn)


def test_ordinary_memory_policy_retains_chunk_and_classifier(observation):
    policy = policy_module.Policy(_MemoryModel())
    result = policy.infer(observation)
    np.testing.assert_array_equal(result["actions"], np.arange(8)[:, None])
    memory = policy.infer_memory(observation)
    np.testing.assert_array_equal(memory["joint_logits"], [2.0, 3.0])
    assert memory["confidence"] == 0.75
    with pytest.raises(ValueError, match="only supported by flow-based"):
        rtc_policy.RealTimeChunkingPolicy(policy, rtc_policy.RTCConfig(enabled=True))


def test_guidance_uses_model_actions_before_output_transform(observation):
    model = _GuidedModel()
    policy = policy_module.Policy(
        model,
        output_transforms=[lambda result: {**result, "actions": result["actions"] + 100}],
    )
    result = policy.infer(observation)
    np.testing.assert_array_equal(result["actions"], 100 + np.arange(8)[:, None])
    np.testing.assert_array_equal(policy.last_model_actions, np.arange(8)[:, None])
    assert policy.supports_rtc
    target = np.ones((8, 1), dtype=np.float32)
    mask = np.ones(8, dtype=np.float32)
    policy.infer(observation, rtc_actions=target, rtc_mask=mask, rtc_guidance_weight=2.0)
    np.testing.assert_array_equal(model.guidance[0], target[None])
    np.testing.assert_array_equal(model.guidance[1], mask[None, :, None])
    assert model.guidance[2] == 2.0
    policy.reset()
    assert policy.last_model_actions is None


def test_policy_rejects_incomplete_guidance(observation):
    policy = policy_module.Policy(_GuidedModel())
    with pytest.raises(ValueError, match="provided together"):
        policy.infer(observation, rtc_mask=np.ones(8))


def test_server_clears_policy_state_when_controller_disconnects():
    class InnerPolicy:
        resets = 0
        inferences = 0

        def reset(self):
            self.resets += 1

        def infer(self, _observation):
            self.inferences += 1
            return {"actions": np.zeros((1, 1), dtype=np.float32)}

    class Connection:
        remote_address = ("controller", 0)

        def __init__(self):
            self.messages = []
            self.reads = 0

        async def send(self, value):
            self.messages.append(value)

        async def recv(self):
            self.reads += 1
            if self.reads == 1:
                return websocket_policy_server.msgpack_numpy.Packer().pack({})
            raise asyncio.CancelledError

    policy = InnerPolicy()
    connection = Connection()
    server = websocket_policy_server.WebsocketPolicyServer(policy)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server._handler(connection))
    assert policy.inferences == 1
    assert policy.resets == 2
    assert len(connection.messages) == 2  # Metadata followed by one action response.
