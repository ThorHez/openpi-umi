"""Client-owned RTC execution timing without robot or model dependencies.

Only the caller submits returned plans to hardware. The inference worker owns
the policy transport, while plan alignment uses observation capture times in
the same Unix clock as the robot's waypoint scheduler.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
import math
import time

import numpy as np


class RTCError(RuntimeError):
    """The controller must stop submitting trajectories and hold the robot."""


@dataclass(frozen=True)
class TimedPlan:
    """World-coordinate actions and their absolute controller deadlines."""

    actions: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class _Inference:
    plan: TimedPlan
    elapsed: float


@dataclass(frozen=True)
class _Pending:
    future: Future
    started_at: float


def _public_observation(observation: dict) -> dict:
    return {key: value for key, value in observation.items() if not key.startswith("_")}


class RTCController:
    """Infer replacement chunks while a previously submitted plan executes.

    Call ``warmup``, obtain a fresh observation, then call ``start`` once.
    Submit that returned plan on the caller's thread. Repeated ``poll`` calls
    either return a replacement plan or immediately return ``None`` while the
    old trajectory continues executing. Never use a robot's independent
    dynamic-latency adjustment with these absolute waypoint times.

    ``action_to_world`` decodes ordinary postprocessed policy actions using
    the observation at which they were predicted. ``world_to_action`` rebases
    an old world plan into the *new* observation's raw training action space.
    ``resample_world`` interpolates a world plan onto a new observation grid;
    it must interpolate orientations correctly and must not extrapolate.

    Every RTCError invalidates the active episode. The caller should issue a
    measured-pose hold, since stopping this object cannot cancel hardware
    waypoints that were previously submitted.
    """

    def __init__(
        self,
        policy,
        *,
        frequency: float,
        action_to_world: Callable,
        world_to_action: Callable,
        resample_world: Callable,
        request_observation: Callable = _public_observation,
        replan_steps: int = 4,
        command_latency: float = 0.05,
        max_inference_seconds: float = 2.0,
        initial_delay_steps: int = 2,
        delay_buffer_size: int = 10,
        clock: Callable = time.time,
        server_metadata: dict | None = None,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 0.5,
    ):
        for name, value in (("frequency", frequency), ("max_inference_seconds", max_inference_seconds)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if not np.isfinite(command_latency) or command_latency < 0:
            raise ValueError("command_latency must be finite and nonnegative")
        for name, value, minimum in (
            ("replan_steps", replan_steps, 1),
            ("initial_delay_steps", initial_delay_steps, 0),
            ("delay_buffer_size", delay_buffer_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(name + " has an invalid integer value")
        if prefix_attention_schedule not in ("ones", "zeros", "linear", "exp"):
            raise ValueError("Unknown RTC prefix schedule")
        if not np.isfinite(max_guidance_weight) or max_guidance_weight < 0:
            raise ValueError("max_guidance_weight must be finite and nonnegative")
        metadata_getter = getattr(policy, "get_server_metadata", None)
        has_metadata = server_metadata is not None or callable(metadata_getter)
        if server_metadata is None and callable(metadata_getter):
            server_metadata = metadata_getter()
        if has_metadata:
            rtc = server_metadata.get("rtc") if isinstance(server_metadata, dict) else None
            if not isinstance(rtc, dict) or rtc.get("enabled") is not True or rtc.get("mode") != "client":
                raise ValueError("RTCController requires a client-mode RTC policy server")

        self._policy = policy
        self._frequency = float(frequency)
        self._action_to_world = action_to_world
        self._world_to_action = world_to_action
        self._resample_world = resample_world
        self._request_observation = request_observation
        self._replan_steps = replan_steps
        self._command_latency = float(command_latency)
        self._max_inference_seconds = float(max_inference_seconds)
        self._initial_delay_seconds = initial_delay_steps / self._frequency
        self._delay_seconds = deque([self._initial_delay_seconds], maxlen=delay_buffer_size)
        self._clock = clock
        self._schedule = prefix_attention_schedule
        self._guidance_weight = float(max_guidance_weight)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-rtc-client")
        self._plan = None
        self._pending = None
        self._retired = None
        self._running = False
        self._closed = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def inference_active(self) -> bool:
        return self._pending is not None

    @property
    def estimated_inference_seconds(self) -> float:
        return max(self._delay_seconds)

    def _check_idle_transport(self) -> None:
        if self._closed:
            raise RTCError("RTC controller is closed")
        if self._running:
            raise RTCError("Stop the active RTC episode before starting another")
        if self._retired is not None and not self._retired.done():
            raise RTCError("Previous RTC inference is still using the policy transport")
        self._retired = None

    def warmup(self, observation: dict) -> None:
        """Compile sampling before execution; discard this stale initial plan."""
        self._check_idle_transport()
        request = self._request_observation(copy.deepcopy(observation))
        self._policy.infer(request)

    def start(self, observation: dict, observation_time: float) -> TimedPlan:
        """Synchronously obtain the initial plan from a fresh captured frame."""
        self._check_idle_transport()
        self._validate_capture_time(observation_time)
        snapshot = copy.deepcopy(observation)
        self._delay_seconds.clear()
        self._delay_seconds.append(self._initial_delay_seconds)
        try:
            result = self._infer(snapshot, observation_time, None)
            if result.elapsed > self._max_inference_seconds:
                raise RTCError("Initial RTC inference exceeded max_inference_seconds")
            if len(result.plan.actions) <= self._replan_steps:
                raise RTCError("Model action horizon must exceed replan_steps")
            public_plan = self._trim(result.plan, self._clock())
        except Exception as error:
            self._fail("Initial RTC inference failed", error)
        self._delay_seconds.append(result.elapsed)
        self._plan = result.plan
        self._running = True
        return public_plan

    def poll(self, observation: dict, observation_time: float) -> TimedPlan | None:
        """Poll/start inference without waiting; return only future waypoints."""
        if not self._running:
            return None
        now = self._clock()
        try:
            self._require_coverage(now)
            if self._pending is not None:
                pending = self._pending
                if not pending.future.done():
                    if now - pending.started_at > self._max_inference_seconds:
                        raise RTCError("RTC inference exceeded max_inference_seconds")
                    return None
                result = pending.future.result()
                if result.elapsed > self._max_inference_seconds:
                    raise RTCError("RTC inference exceeded max_inference_seconds")
                public_plan = self._trim(result.plan, now)
                self._plan = result.plan
                self._pending = None
                self._delay_seconds.append(result.elapsed)
                return public_plan

            if now < self._plan.timestamps[0] + self._replan_steps / self._frequency:
                return None
            self._validate_capture_time(observation_time)
            snapshot = copy.deepcopy(observation)
            fields = self._previous_fields(snapshot, observation_time, now)
            self._pending = _Pending(
                self._executor.submit(self._infer, snapshot, observation_time, fields), started_at=now
            )
            return None
        except Exception as error:
            self._fail("RTC replacement inference failed", error)

    def _infer(self, observation: dict, observation_time: float, fields: dict | None) -> _Inference:
        request = self._request_observation(copy.deepcopy(observation))
        if not isinstance(request, dict):
            raise RTCError("request_observation must return a dictionary")
        if fields is not None:
            request.update(fields)
        started = self._clock()
        response = self._policy.infer(request)
        elapsed = self._clock() - started
        if not np.isfinite(elapsed) or elapsed < 0:
            raise RTCError("Controller clock moved backwards during inference")
        actions = np.asarray(response["actions"])
        if actions.ndim != 2 or len(actions) < 2 or actions.shape[1] == 0 or not np.all(np.isfinite(actions)):
            raise RTCError("Policy actions must be a finite action chunk")
        world = np.asarray(self._action_to_world(actions.copy(), observation), dtype=np.float64)
        if world.ndim != 2 or len(world) != len(actions) or world.shape[1] == 0 or not np.all(np.isfinite(world)):
            raise RTCError("action_to_world must retain the finite chunk horizon")
        timestamps = observation_time + np.arange(len(world), dtype=np.float64) / self._frequency
        return _Inference(TimedPlan(world.copy(), timestamps), elapsed)

    def _previous_fields(self, observation: dict, observation_time: float, now: float) -> dict:
        if observation_time < self._plan.timestamps[0] - 1e-7:
            raise RTCError("Observation capture time precedes the current RTC plan")
        available = math.floor((self._plan.timestamps[-1] - observation_time) * self._frequency + 1e-7) + 1
        if available <= 0:
            raise RTCError("No old RTC plan covers the new observation")
        targets = observation_time + np.arange(available, dtype=np.float64) / self._frequency
        world = self._resample_world(self._plan.actions.copy(), self._plan.timestamps.copy(), targets)
        previous = np.asarray(self._world_to_action(world, observation), dtype=np.float32)
        if previous.ndim != 2 or len(previous) != available or not np.all(np.isfinite(previous)):
            raise RTCError("world_to_action must retain the finite previous-prefix horizon")
        predicted_delay = math.ceil(
            max(0.0, now - observation_time + self.estimated_inference_seconds + self._command_latency)
            * self._frequency
            - 1e-9
        )
        if predicted_delay >= available:
            raise RTCError("Old RTC plan cannot cover the estimated observation age and inference latency")
        return {
            "prev_action_chunk": previous,
            "prev_action_chunk_normalized": False,
            "inference_delay": predicted_delay,
            "execute_horizon": available,
            "enable_rtc": True,
            "mask_prefix_delay": False,
            "prefix_attention_schedule": self._schedule,
            "max_guidance_weight": self._guidance_weight,
        }

    def _validate_capture_time(self, observation_time: float) -> None:
        if not np.isfinite(observation_time) or observation_time > self._clock() + 1e-6:
            raise RTCError("Observation capture time must be finite and use the controller's past Unix clock")

    def _require_coverage(self, now: float) -> None:
        if self._plan is None or now + self._command_latency >= self._plan.timestamps[-1]:
            raise RTCError("RTC exhausted its current plan; hold the measured robot pose")

    def _trim(self, plan: TimedPlan, now: float) -> TimedPlan:
        first = int(np.searchsorted(plan.timestamps, now + self._command_latency, side="right"))
        if first >= len(plan.actions):
            raise RTCError("All RTC waypoints expired before inference completed")
        return TimedPlan(plan.actions[first:].copy(), plan.timestamps[first:].copy())

    def _fail(self, message: str, cause: Exception) -> None:
        self.stop()
        if isinstance(cause, RTCError):
            raise cause
        raise RTCError(message) from cause

    def stop(self) -> None:
        """Discard pending results; the caller must separately hold hardware."""
        self._running = False
        self._plan = None
        if self._pending is not None:
            future = self._pending.future
            if not future.cancel():
                self._retired = future
            self._pending = None

    def reset(self) -> None:
        self.stop()
        self._delay_seconds.clear()
        self._delay_seconds.append(self._initial_delay_seconds)

    def close(self) -> None:
        self.stop()
        self._closed = True
        # Never block an emergency stop on inference/network completion.
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
