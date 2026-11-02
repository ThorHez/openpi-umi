"""Deterministic capture-time RTC scheduling without a robot or JAX."""

import copy
import threading
import time

import numpy as np
import pytest

from openpi_client.rtc_controller import RTCController
from openpi_client.rtc_controller import RTCError


class _Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class _Policy:
    def __init__(self, clock):
        self.clock = clock
        self.requests = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.completed = threading.Event()
        self.error = None

    def get_server_metadata(self):
        return {"rtc": {"enabled": True, "mode": "client"}}

    def infer(self, request):
        self.requests.append(copy.deepcopy(request))
        if len(self.requests) == 1:
            self.clock.value += 0.1
            return {"actions": np.arange(16, dtype=float)[:, None]}
        self.started.set()
        self.release.wait(timeout=5)
        self.completed.set()
        if self.error:
            raise self.error
        return {"actions": 100 + np.arange(16, dtype=float)[:, None]}


def _resample(actions, timestamps, targets):
    assert targets[0] >= timestamps[0]
    assert targets[-1] <= timestamps[-1]
    return np.interp(targets, timestamps, actions[:, 0])[:, None]


def _controller(policy, clock, **kwargs):
    return RTCController(
        policy,
        frequency=4,
        action_to_world=lambda actions, obs: actions + obs["_anchor"],
        world_to_action=lambda actions, obs: actions - obs["_anchor"],
        resample_world=_resample,
        replan_steps=2,
        clock=clock,
        **kwargs,
    )


def _await(predicate):
    until = time.monotonic() + 2
    while not predicate() and time.monotonic() < until:
        time.sleep(0.001)
    assert predicate()


def test_initial_plan_discards_observation_age_and_inference_latency():
    clock = _Clock()
    policy = _Policy(clock)
    clock.value = 100.3
    with _controller(policy, clock) as controller:
        plan = controller.start({"_anchor": 5, "state": np.array([1.0])}, 100.0)
        assert plan.timestamps[0] == 100.5
        assert np.all(plan.timestamps > clock() + 0.05)
        np.testing.assert_array_equal(plan.actions[:, 0], np.arange(2, 16) + 5)
        assert "_anchor" not in policy.requests[0]
        assert "prev_action_chunk" not in policy.requests[0]
        # A caller/robot driver mutating its returned plan must not corrupt the
        # retained old trajectory used for future guidance.
        plan.actions[:] = -100
        clock.value = 100.6
        controller.poll({"_anchor": 0}, 100.5)
        assert policy.started.wait(timeout=2)
        assert np.all(policy.requests[1]["prev_action_chunk"] > 0)
        policy.release.set()


def test_inference_overlaps_old_plan_and_rebases_interpolated_world_prefix():
    clock = _Clock()
    policy = _Policy(clock)
    with _controller(policy, clock) as controller:
        initial = controller.start({"_anchor": 0, "state": np.array([1.0])}, 100.0)
        clock.value = 100.5
        new_observation = {"_anchor": 10, "state": np.array([2.0])}
        assert controller.poll(new_observation, 100.3) is None
        assert policy.started.wait(timeout=2)
        request = policy.requests[1]
        assert request["prev_action_chunk_normalized"] is False
        assert request["inference_delay"] == 3  # .2s age + .5s estimate + .05 command latency.
        assert request["execute_horizon"] == 14
        np.testing.assert_allclose(request["prev_action_chunk"][:, 0], 1.2 + np.arange(14) - 10)
        assert controller.inference_active
        assert controller.running
        # The original submitted plan remains valid and unchanged while the
        # worker blocks; poll does not wait for that worker.
        before = time.monotonic()
        assert controller.poll(new_observation, 100.3) is None
        assert time.monotonic() - before < 0.1
        assert initial.timestamps[-1] == 103.75
        new_observation["_anchor"] = 999
        new_observation["state"][0] = 999
        clock.value = 101.1
        policy.release.set()
        _await(lambda: not controller.inference_active or policy.completed.is_set())
        plan = None
        until = time.monotonic() + 2
        while plan is None and time.monotonic() < until:
            plan = controller.poll(new_observation, 101.0)
            if plan is None:
                time.sleep(0.001)
        assert plan is not None
        # Capture-time grid: .30, .55, .80, 1.05, 1.30. Four rows expired,
        # including the initial observation age, not merely the infer RTT.
        assert plan.timestamps[0] == pytest.approx(101.3)
        assert plan.actions[0, 0] == 114  # raw row4 + snapshotted anchor10.
        np.testing.assert_array_equal(request["state"], [2.0])
        assert not controller.inference_active


def test_prediction_must_fit_old_plan_instead_of_clipping_delay():
    clock = _Clock()
    policy = _Policy(clock)
    with _controller(policy, clock, initial_delay_steps=20) as controller:
        controller.start({"_anchor": 0}, 100.0)
        clock.value = 100.5
        with pytest.raises(RTCError, match="cannot cover"):
            controller.poll({"_anchor": 1}, 100.4)
        assert len(policy.requests) == 1
        assert not controller.running


def test_running_out_of_plan_raises_without_repeating_last_action():
    clock = _Clock()
    policy = _Policy(clock)
    with _controller(policy, clock) as controller:
        controller.start({"_anchor": 0}, 100.0)
        clock.value = 103.8
        with pytest.raises(RTCError, match="exhausted"):
            controller.poll({"_anchor": 1}, 103.5)
        assert not controller.running
        assert controller.poll({"_anchor": 1}, 103.6) is None


def test_expired_initial_plan_fails_before_returning_waypoints():
    clock = _Clock()
    policy = _Policy(clock)
    clock.value = 104.0
    with _controller(policy, clock) as controller:
        with pytest.raises(RTCError, match="All RTC waypoints expired"):
            controller.start({"_anchor": 0}, 100.0)
        assert not controller.running


def test_worker_error_invalidates_episode():
    clock = _Clock()
    policy = _Policy(clock)
    policy.error = ValueError("model failed")
    with _controller(policy, clock) as controller:
        controller.start({"_anchor": 0}, 100.0)
        clock.value = 100.5
        controller.poll({"_anchor": 0}, 100.4)
        assert policy.started.wait(timeout=2)
        policy.release.set()
        _await(policy.completed.is_set)
        # completed is set just before infer raises; polling may race its
        # publication to the Future, so wait for an actual Future completion.
        until = time.monotonic() + 2
        surfaced = None
        while time.monotonic() < until:
            try:
                controller.poll({"_anchor": 0}, 100.4)
            except RTCError as error:
                surfaced = error
                break
            time.sleep(0.001)
        else:
            pytest.fail("Worker error was not surfaced")
        assert isinstance(surfaced.__cause__, ValueError)
        assert not controller.running


def test_timeout_stops_publishing_without_waiting_for_worker():
    clock = _Clock()
    policy = _Policy(clock)
    with _controller(policy, clock, max_inference_seconds=0.2) as controller:
        controller.start({"_anchor": 0}, 100.0)
        clock.value = 100.5
        controller.poll({"_anchor": 0}, 100.4)
        assert policy.started.wait(timeout=2)
        clock.value = 100.8
        try:
            with pytest.raises(RTCError, match="exceeded"):
                controller.poll({"_anchor": 0}, 100.7)
            assert not controller.running
            assert controller.poll({"_anchor": 0}, 100.7) is None
        finally:
            policy.release.set()


@pytest.mark.parametrize("method", ["stop", "reset", "close"])
def test_stop_discards_inflight_results_and_protects_transport(method):
    clock = _Clock()
    policy = _Policy(clock)
    controller = _controller(policy, clock)
    try:
        controller.start({"_anchor": 0}, 100.0)
        clock.value = 100.5
        controller.poll({"_anchor": 0}, 100.4)
        assert policy.started.wait(timeout=2)
        getattr(controller, method)()
        assert controller.poll({"_anchor": 0}, 100.4) is None
        with pytest.raises(RTCError, match=r"closed|still using"):
            controller.start({"_anchor": 0}, 100.4)
        policy.release.set()
        _await(policy.completed.is_set)
        assert controller.poll({"_anchor": 0}, 100.4) is None
        assert not controller.running
    finally:
        policy.release.set()
        controller.close()


def test_warmup_is_discarded_and_not_used_as_latency_estimate():
    clock = _Clock()

    class Policy:
        calls = 0

        def infer(self, _observation):
            self.calls += 1
            clock.value += 10 if self.calls == 1 else 0.1
            return {"actions": np.arange(16, dtype=float)[:, None]}

    with _controller(Policy(), clock) as controller:
        controller.warmup({"_anchor": 0})
        assert not controller.running
        assert controller.estimated_inference_seconds == 0.5
        fresh_capture = clock()
        plan = controller.start({"_anchor": 0}, fresh_capture)
        assert np.all(plan.timestamps > clock() + 0.05)
        assert controller.estimated_inference_seconds == 0.5


@pytest.mark.parametrize(
    "metadata", [{}, {"rtc": {"enabled": False, "mode": "client"}}, {"rtc": {"enabled": True, "mode": "server"}}]
)
def test_rejects_policy_without_client_rtc_metadata(metadata):
    clock = _Clock()
    with pytest.raises(ValueError, match="client-mode"):
        _controller(object(), clock, server_metadata=metadata)


def test_real_transport_returning_no_metadata_is_not_treated_as_a_test_double():
    class Policy:
        def get_server_metadata(self):
            return None

    with pytest.raises(ValueError, match="client-mode"):
        _controller(Policy(), _Clock())


def test_future_observation_timestamp_is_rejected():
    clock = _Clock()
    policy = _Policy(clock)
    with _controller(policy, clock) as controller:
        with pytest.raises(RTCError, match="past Unix clock"):
            controller.start({"_anchor": 0}, 100.1)
        assert not policy.requests


def test_observation_older_than_current_plan_is_not_extrapolated():
    clock = _Clock()
    policy = _Policy(clock)
    with _controller(policy, clock) as controller:
        controller.start({"_anchor": 0}, 100.0)
        clock.value = 100.5
        with pytest.raises(RTCError, match="precedes"):
            controller.poll({"_anchor": 0}, 99.9)
        assert len(policy.requests) == 1
