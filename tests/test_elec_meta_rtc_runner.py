"""Exercise the elec_meta scene loop without importing a robot SDK."""

import builtins
import importlib.util
from pathlib import Path
import threading

import numpy as np
from openpi_client.rtc_controller import RTCError
from openpi_client.rtc_controller import TimedPlan
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "examples/umi_arx/elec_meta_rtc.py"
SPEC = importlib.util.spec_from_file_location("elec_meta_rtc_example", SOURCE)
SCENE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCENE)


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def raw_observation(now=10.0):
    return {
        "robot0_eef_pos": np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        "robot0_eef_rot_axis_angle": np.array([[0.0, 0.0, np.pi / 2]] * 2),
        "robot0_gripper_width": np.array([[0.03], [0.04]]),
        "camera0_rgb": np.zeros((2, 224, 224, 3), dtype=np.uint8),
        "timestamp": np.array([now - 0.125, now]),
    }


def test_capture_preparation_uses_current_tcp_and_episode_start():
    raw = raw_observation()
    observation = SCENE.prepare_observation(raw, np.zeros(6), obs_pose_repr="relative", prompt="sort components")
    np.testing.assert_allclose(observation["robot0_eef_pos"], [[-1, 0, 0], [0, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(observation["robot0_eef_rot_axis_angle"], [[1, 0, 0, 0, 1, 0]] * 2, atol=1e-12)
    np.testing.assert_allclose(
        observation["robot0_eef_rot_axis_angle_wrt_start"],
        [[0, -1, 0, 1, 0, 0]] * 2,
        atol=1e-12,
    )
    np.testing.assert_allclose(observation["_tcp_pose"], [1, 1, 0, 0, 0, np.pi / 2])
    assert observation["_observation_time"] == 10.0
    assert observation["camera0_rgb"].shape == (2, 3, 224, 224)
    assert observation["camera0_rgb"].dtype == np.uint8
    assert raw["camera0_rgb"].shape == (2, 224, 224, 3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("robot0_eef_pos", np.zeros((1, 3))),
        ("robot0_eef_rot_axis_angle", np.full((2, 3), np.nan)),
        ("camera0_rgb", np.full((2, 224, 224, 3), 2.0)),
        ("timestamp", np.array([10.0, 9.0])),
    ],
)
def test_invalid_captured_observations_fail_before_inference(field, value):
    with pytest.raises(ValueError, match="must"):
        SCENE.prepare_observation(
            {**raw_observation(), field: value}, np.zeros(6), obs_pose_repr="relative", prompt="sort"
        )


class Environment:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.submitted = []
        self.holds = []
        self.robots = [self]
        self.frequency = 200
        self.measured_pose = np.zeros(6)

    def get_obs(self):
        self.calls.append(("observe", threading.get_ident()))
        return raw_observation(self.clock())

    def exec_actions(self, **kwargs):
        self.calls.append(("execute", threading.get_ident()))
        self.submitted.append(kwargs)

    def get_robot_state(self):
        self.calls.append(("state", threading.get_ident()))
        return [{"ActualTCPPose": self.measured_pose.copy(), "gripper_position": 0.035}]

    def servoL(self, **kwargs):  # noqa: N802
        self.calls.append(("hold", threading.get_ident()))
        self.holds.append(kwargs)

    def start_episode(self, _time):
        self.calls.append(("start_recording", threading.get_ident()))

    def end_episode(self):
        self.calls.append(("end_recording", threading.get_ident()))


class Controller:
    def __init__(self, clock, *, failure=False):
        self.clock = clock
        self.failure = failure
        self.observations = []
        self.stopped = False
        self.poll_count = 0

    def warmup(self, observation):
        self.observations.append(observation)
        self.clock.sleep(1.0)

    def start(self, observation, capture_time):
        self.observations.append(observation)
        assert capture_time == self.clock()
        return TimedPlan(np.zeros((3, 7)), self.clock() + np.array([-0.1, 0.1, 0.2]))

    def poll(self, _observation, _capture_time):
        self.poll_count += 1
        if self.failure:
            raise RTCError("Background inference failed")
        if self.poll_count == 1:
            actions = np.zeros((2, 7))
            actions[:, 0], actions[:, 6] = 0.001, 0.04
            return TimedPlan(actions, self.clock() + np.array([0.2, 0.3]))
        return None

    def stop(self):
        self.stopped = True


def prepare(raw):
    return SCENE.prepare_observation(raw, np.zeros(6), obs_pose_repr="relative", prompt="sort")


def test_runner_drops_stale_rows_and_uses_main_thread_for_execution_and_hold():
    clock = Clock()
    env, controller = Environment(clock), Controller(clock)
    plans = []
    count = SCENE.run_episode(
        env, controller, prepare, frequency=8, max_duration=0.3, clock=clock, sleep=clock.sleep, sink=plans.append
    )
    assert count == len(plans) == len(env.submitted) == 2
    assert controller.observations[0]["_observation_time"] < controller.observations[1]["_observation_time"]
    assert len(env.submitted[0]["actions"]) == 2
    for submitted in env.submitted:
        assert submitted["dynamic_latency"] is False
        assert submitted["compensate_latency"] is False
    assert {ident for _name, ident in env.calls} == {threading.get_ident()}
    np.testing.assert_array_equal(env.holds[0]["pose"], env.measured_pose)
    assert env.holds[0]["gripper_pos"] == 0.035
    assert controller.stopped
    assert [name for name, _ident in env.calls][-1] == "end_recording"


def test_background_failure_stops_and_holds_the_measured_pose():
    clock = Clock()
    env, controller = Environment(clock), Controller(clock, failure=True)
    with pytest.raises(RTCError, match="Background"):
        SCENE.run_episode(env, controller, prepare, frequency=8, max_duration=1, clock=clock, sleep=clock.sleep)
    assert len(env.submitted) == 1
    assert len(env.holds) == 1
    assert controller.stopped


def test_stop_before_start_discards_warmup_and_holds_without_submitting():
    clock = Clock()
    env, controller = Environment(clock), Controller(clock)
    assert (
        SCENE.run_episode(
            env,
            controller,
            prepare,
            frequency=8,
            max_duration=1,
            clock=clock,
            sleep=clock.sleep,
            should_stop=lambda: True,
        )
        == 0
    )
    assert not env.submitted
    assert len(env.holds) == 1
    assert controller.stopped


def test_profile_metadata_is_required_before_creating_controller():
    class OrdinaryPolicy:
        def get_server_metadata(self):
            return {"rtc": {"enabled": True, "mode": "client"}}

    with pytest.raises(ValueError, match="elec-meta"):
        SCENE.make_controller(OrdinaryPolicy(), frequency=8, action_pose_repr="relative")


def test_replay_cli_never_imports_robot_sdk(tmp_path, monkeypatch, capsys):
    path = tmp_path / "capture.npz"
    np.savez(path, **raw_observation())
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("modules.", "peripherals.", "multiprocessing.managers")):
            pytest.fail("Replay must not import a hardware environment")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert (
        SCENE.main(
            [
                "--replay",
                str(path),
                "--mock-policy",
                "--obs-pose-repr",
                "relative",
                "--action-pose-repr",
                "relative",
                "--frequency",
                "20",
                "--prompt",
                "sort",
                "--max-duration",
                "0.06",
                "--command-latency",
                "0",
            ]
        )
        == 0
    )
    assert '"mode": "replay"' in capsys.readouterr().out


def test_cli_requires_explicit_execution_or_replay_before_opening_transport():
    def no_transport(**_kwargs):
        pytest.fail("Invalid CLI must not open a transport")

    with pytest.raises(SystemExit):
        SCENE.main(
            ["--obs-pose-repr", "relative", "--action-pose-repr", "relative", "--frequency", "8", "--prompt", "sort"],
            policy_factory=no_transport,
        )


def test_mock_policy_cannot_be_used_for_hardware():
    with pytest.raises(SystemExit):
        SCENE.main(
            [
                "--execute",
                "--mock-policy",
                "--obs-pose-repr",
                "relative",
                "--action-pose-repr",
                "relative",
                "--frequency",
                "8",
                "--prompt",
                "sort",
            ]
        )


@pytest.mark.parametrize(
    ("actions", "timestamps", "message"),
    [
        ([[0.1, 0, 0, 0, 0, 0, 0.04]], [0.1], "translation speed"),
        ([[0, 0, 0, 0, 0, 0, 0.04], [0.1, 0, 0, 0, 0, 0, 0.04]], [0.1, 0.2], "translation speed"),
        ([[0, 0, 0, 0, 0, 0.1, 0.04]], [0.1], "rotation speed"),
        ([[0, 0, 0, 0, 0, 0, 0.04], [0, 0, 0, 0, 0, 0.1, 0.04]], [0.1, 0.2], "rotation speed"),
        ([[0, 0, 0, 0, 0, 0, -0.001]], [0.1], "gripper width"),
        ([[0, 0, 0, 0, 0, 0, 0.086]], [0.1], "gripper width"),
        ([[0, 0, 0, 0, 0, 0, 0.04]], [-0.1], "deadlines"),
    ],
)
def test_world_command_bounds_reject_first_and_adjacent_segments(actions, timestamps, message):
    with pytest.raises(RTCError, match=message):
        SCENE.validate_world_commands(actions, timestamps, np.zeros(6), now=0)


def test_world_command_rotation_speed_uses_so3_shortest_path_without_mutation():
    actions = np.array([[0.01, 0, 0, 0, 0, -np.pi + 0.01, 0.04]])
    before = actions.copy()
    timestamps = np.array([0.1])
    SCENE.validate_world_commands(actions, timestamps, [0, 0, 0, 0, 0, np.pi - 0.01], now=0)
    np.testing.assert_array_equal(actions, before)
    np.testing.assert_array_equal(timestamps, [0.1])


def test_runner_checks_current_measured_tcp_before_submitting_and_holds_on_speed_violation():
    clock = Clock()
    env, controller = Environment(clock), Controller(clock)
    env.measured_pose[0] = 0.5
    with pytest.raises(RTCError, match="translation speed"):
        SCENE.run_episode(env, controller, prepare, frequency=8, max_duration=0.3, clock=clock, sleep=clock.sleep)
    assert not env.submitted
    np.testing.assert_array_equal(env.holds[0]["pose"], env.measured_pose)
    assert controller.stopped


def test_replay_checks_latest_raw_tcp_instead_of_skipping_speed_bounds():
    clock = Clock()
    env, controller = Environment(clock), Controller(clock)
    env.robots = []
    env.get_robot_state = list
    # The replay raw observation has TCP (1,1,0); the dummy plan jumps to zero.
    with pytest.raises(RTCError, match="translation speed"):
        SCENE.run_episode(env, controller, prepare, frequency=8, max_duration=0.3, clock=clock, sleep=clock.sleep)
    assert not env.submitted
    assert controller.stopped


def test_initial_observation_failure_in_cli_still_holds_and_closes_transport():
    clock = Clock()

    class BrokenCamera(Environment):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_obs(self):
            raise RuntimeError("Camera capture failed")

    class Policy(SCENE._MockPolicy):  # noqa: SLF001
        closed = False

        def close(self):
            self.closed = True

    env, policy = BrokenCamera(clock), Policy()
    transport_options = {}

    def factory(**kwargs):
        transport_options.update(kwargs)
        return policy

    with pytest.raises(RuntimeError, match="Camera capture failed"):
        SCENE.main(
            [
                "--execute",
                "--obs-pose-repr",
                "relative",
                "--action-pose-repr",
                "relative",
                "--frequency",
                "8",
                "--prompt",
                "sort",
            ],
            policy_factory=factory,
            env_factory=lambda _args: env,
            clock=clock,
            sleep=clock.sleep,
        )
    assert len(env.holds) == 1
    assert not env.submitted
    assert policy.closed
    assert transport_options["ping_interval"] is None
    assert transport_options["inference_timeout"] == 120.0
