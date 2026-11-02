"""Real WebSocket + scene factory + JAX RTC sampler, using a tiny test flow.

This tests the complete wire/geometry/timing path. The test flow has no real
checkpoint weights and the returned commands are never sent to a robot.
"""

import asyncio
from contextlib import contextmanager
from queue import Queue
import threading
import time

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from openpi_client.rtc_controller import RTCController
from openpi_client.umi_pose import interpolate_world_actions
from openpi_client.umi_pose import model_actions_to_world_actions
from openpi_client.umi_pose import world_actions_to_model_actions
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import websockets.asyncio.server as websocket_server

from openpi import transforms
from openpi.models import model as model_module
from openpi.models import pi0_config
from openpi.models.pi0_rtc import Pi0RTC
from openpi.policies import elec_meta_config
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize
from openpi.training import config


class _ZeroVelocity(nnx.Module):
    def __call__(self, inputs, **_kwargs):
        return (None, None if inputs[1] is None else jnp.zeros_like(inputs[1])), None


class _SceneTestFlow(Pi0RTC):
    def __init__(self):
        self.action_horizon = 16
        self.action_dim = 32
        self.PaliGemma = nnx.Dict(llm=_ZeroVelocity())

    def preprocess_observation(self, _rng, observation, **_kwargs):
        assert observation.state.shape == (1, 32)
        return observation

    def embed_prefix(self, observation):
        batch = observation.state.shape[0]
        return jnp.zeros((batch, 1, 32)), jnp.ones((batch, 1), dtype=bool), jnp.array([False])

    def embed_suffix(self, _observation, actions, _time):
        return actions, jnp.ones(actions.shape[:2], dtype=bool), jnp.arange(16) == 0, None

    def action_out_proj(self, value):
        return value


@contextmanager
def _serve(policy):
    ready = Queue()
    service = WebsocketPolicyServer(policy, metadata=policy.metadata)

    async def run():
        stop = asyncio.Event()
        try:
            async with websocket_server.serve(
                service._handler,  # noqa: SLF001
                "127.0.0.1",
                0,
                compression=None,
                max_size=None,
            ) as server:
                ready.put((asyncio.get_running_loop(), stop, server.sockets[0].getsockname()[1]))
                await stop.wait()
        except Exception as error:
            ready.put(error)

    thread = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)
    thread.start()
    started = ready.get(timeout=10)
    if isinstance(started, Exception):
        raise started
    loop, stop, port = started
    try:
        yield port
    finally:
        loop.call_soon_threadsafe(stop.set)
        thread.join(timeout=10)
        assert not thread.is_alive()


def _observation(capture, x):
    rotation = np.tile([1, 0, 0, 0, 1, 0], (2, 1)).astype(np.float32)
    return {
        "robot0_eef_pos": np.zeros((2, 3), np.float32),
        "robot0_eef_rot_axis_angle": rotation,
        "robot0_eef_rot_axis_angle_wrt_start": rotation.copy(),
        "robot0_gripper_width": np.full((2, 1), 0.03, np.float32),
        "camera0_rgb": np.zeros((2, 3, 224, 224), np.uint8),
        "prompt": "pick electronic components",
        "_tcp_pose": np.array([x, 0, 0.3, 0, 0, 0]),
        "_observation_time": capture,
    }


def test_scene_websocket_jit_and_async_rebased_replanning(monkeypatch, tmp_path):
    stats = {
        key: transforms.NormStats(
            mean=np.zeros(width),
            std=np.ones(width),
            q01=-np.ones(width),
            q99=np.ones(width),
        )
        for key, width in (("actions", 10), ("state", 16))
    }
    (tmp_path / "assets").mkdir()
    normalize.save(tmp_path / "assets", stats)
    monkeypatch.setattr(model_module, "restore_params", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(pi0_config.Pi0RTCConfig, "load", lambda _config, _params: _SceneTestFlow())
    # Valid 10D identity orientation/gripper, with padded model dimensions.
    noise = np.zeros((1, 16, 32), dtype=np.float32)
    noise[..., 3] = noise[..., 7] = 1
    noise[..., 9] = 0.03
    original = config.get_config(elec_meta_config.ELEC_META_RECIPE)
    policy = policy_config.create_trained_policy(
        elec_meta_config.adapt_elec_meta_config(original),
        tmp_path,
        client_rtc=True,
        sample_kwargs={"num_steps": 3, "noise": noise},
    )
    sampler = policy._sample_actions  # noqa: SLF001
    sampled_prefixes = []

    def sample(*args, **kwargs):
        if kwargs["prev_action_dim"]:
            sampled_prefixes.append(np.asarray(kwargs["prev_action_chunk"]).copy())
        return sampler(*args, **kwargs)

    monkeypatch.setattr(policy, "_sample_actions", sample)
    now = [100.0]
    with _serve(policy) as port:
        client = WebsocketClientPolicy("127.0.0.1", port, inference_timeout=20)
        metadata = client.get_server_metadata()
        assert metadata["inference_profile"] == "elec-meta"
        assert metadata["rtc"]["action_dim"] == 10
        controller = RTCController(
            client,
            frequency=8,
            replan_steps=4,
            command_latency=0.05,
            clock=lambda: now[0],
            action_to_world=lambda actions, obs: model_actions_to_world_actions(
                actions,
                obs["_tcp_pose"],
                pose_repr="relative",
            ),
            world_to_action=lambda world, obs: world_actions_to_model_actions(
                world,
                obs["_tcp_pose"],
                pose_repr="relative",
            ),
            resample_world=interpolate_world_actions,
        )
        try:
            controller.warmup(_observation(99.0, 0.4))
            first = controller.start(_observation(99.9, 0.4), 99.9)
            assert first.actions.shape[1] == 7
            assert np.all(first.timestamps > now[0] + 0.05)
            np.testing.assert_allclose(first.actions[:, 0], 0.4)

            now[0] = 100.55
            fresh = _observation(100.4, 0.5)
            assert controller.poll(fresh, 100.4) is None
            assert controller.inference_active
            deadline = time.monotonic() + 15
            while not sampled_prefixes:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            # The old absolute x=.4 plan is now represented from TCP x=.5.
            np.testing.assert_allclose(sampled_prefixes[0][0, :12, 0], -0.1, atol=1e-6)
            assert sampled_prefixes[0].shape == (1, 16, 32)
            np.testing.assert_array_equal(sampled_prefixes[0][..., 10:], 0)

            now[0] = 100.8
            replacement = None
            while replacement is None:
                assert time.monotonic() < deadline
                replacement = controller.poll(fresh, 100.4)
                time.sleep(0.005)
            # Includes observation age, inference time, and command latency.
            np.testing.assert_allclose(replacement.timestamps[0], 100.9)
            assert np.all(replacement.timestamps > now[0] + 0.05)
            assert np.all(replacement.actions[:4, 0] < 0.5)
            np.testing.assert_allclose(replacement.actions[:, 6], 0.03, atol=1e-6)
        finally:
            controller.close()
            client.close()
