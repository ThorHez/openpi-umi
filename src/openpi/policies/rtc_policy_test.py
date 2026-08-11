import threading

import numpy as np
import pytest

from openpi.policies import rtc_policy


def test_make_soft_mask_matches_equation_5():
    mask = rtc_policy.make_soft_mask(horizon=8, delay=2, execution_horizon=3)

    assert mask.shape == (8,)
    np.testing.assert_array_equal(mask[:2], np.ones(2))
    np.testing.assert_array_equal(mask[5:], np.zeros(3))

    c = (5 - np.arange(2, 5)) / 4
    expected_decay = np.square(c) * np.expm1(c) / np.expm1(1.0)
    np.testing.assert_allclose(mask[2:5], expected_decay, rtol=1e-6)


@pytest.mark.parametrize(
    ("delay", "execution_horizon"),
    [(-1, 3), (4, 3), (3, 6)],
)
def test_make_soft_mask_rejects_invalid_horizons(delay, execution_horizon):
    with pytest.raises(ValueError, match="RTC requires"):
        rtc_policy.make_soft_mask(horizon=8, delay=delay, execution_horizon=execution_horizon)


class _FakeFlowPolicy:
    supports_rtc = True
    action_horizon = 8

    def __init__(self):
        self.metadata = {}
        self.last_model_actions = None
        self.guidance_started = threading.Event()
        self.release_guidance = threading.Event()
        self.received_rtc_actions = None
        self.received_rtc_mask = None

    def infer(self, _obs, *, rtc_actions=None, rtc_mask=None, rtc_guidance_weight=5.0):
        del rtc_guidance_weight
        if rtc_actions is None:
            actions = np.arange(self.action_horizon, dtype=np.float32)[:, None]
        else:
            self.received_rtc_actions = rtc_actions
            self.received_rtc_mask = rtc_mask
            self.guidance_started.set()
            if not self.release_guidance.wait(timeout=2):
                raise TimeoutError("test did not release guided inference")
            actions = 100 + np.arange(self.action_horizon, dtype=np.float32)[:, None]
        self.last_model_actions = actions.copy()
        return {"actions": actions}

    def reset(self):
        self.last_model_actions = None


def test_rtc_defaults_adapt_to_model_horizon():
    policy = rtc_policy.RealTimeChunkingPolicy(_FakeFlowPolicy(), rtc_policy.RTCConfig(enabled=True))  # type: ignore[arg-type]
    try:
        assert policy.metadata["rtc"]["min_execution_horizon"] == 4
        assert policy.metadata["rtc"]["initial_delay_steps"] == 2
    finally:
        policy.reset()


def test_rtc_executes_old_chunk_while_guided_inference_runs():
    inner = _FakeFlowPolicy()
    config = rtc_policy.RTCConfig(
        enabled=True,
        min_execution_horizon=2,
        initial_delay_steps=1,
        delay_buffer_size=3,
        num_steps=5,
        guidance_weight=5,
    )
    policy = rtc_policy.RealTimeChunkingPolicy(inner, config)  # type: ignore[arg-type]
    try:
        np.testing.assert_array_equal(policy.infer({})["actions"], [[0]])
        np.testing.assert_array_equal(policy.infer({})["actions"], [[1]])
        assert inner.guidance_started.wait(timeout=2)

        # This action is consumed while the model is still computing.
        np.testing.assert_array_equal(policy.infer({})["actions"], [[2]])
        inner.release_guidance.set()

        with policy._condition:  # noqa: SLF001
            swapped = policy._condition.wait_for(  # noqa: SLF001
                lambda: (
                    policy._current_result is not None  # noqa: SLF001
                    and policy._current_result["actions"][0, 0] == 100  # noqa: SLF001
                ),
                timeout=2,
            )
        assert swapped

        np.testing.assert_array_equal(inner.received_rtc_actions[:6], np.arange(2, 8)[:, None])
        np.testing.assert_array_equal(inner.received_rtc_actions[6:], np.zeros((2, 1)))
        np.testing.assert_allclose(
            inner.received_rtc_mask,
            rtc_policy.make_soft_mask(horizon=8, delay=1, execution_horizon=2),
        )
        # One tick elapsed during inference, so Algorithm 1 resumes at A_new[1].
        np.testing.assert_array_equal(policy.infer({})["actions"], [[101]])
    finally:
        inner.release_guidance.set()
        policy.reset()
