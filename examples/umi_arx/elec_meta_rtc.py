"""Single-arm elec_meta RTC runner; hardware is opened only with --execute.

Replay accepts an NPZ raw observation (two history frames per key), or a
sequence of such observations with an additional leading sample axis. Sample
selection follows --frequency, with synthetic local capture timestamps. A
single snapshot repeats for local interface checks, without sensor-age claims.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import numpy as np
from openpi_client.rtc_controller import RTCController
from openpi_client.rtc_controller import RTCError
from openpi_client.rtc_controller import TimedPlan
from openpi_client.umi_pose import interpolate_world_actions
from openpi_client.umi_pose import model_actions_to_world_actions
from openpi_client.umi_pose import world_actions_to_model_actions
from scipy.spatial.transform import Rotation


def _finite(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must have finite shape {shape}")
    return value


def prepare_observation(env_obs, episode_start_pose, *, obs_pose_repr, prompt):
    """Convert two captured world TCP samples into the elec_meta input schema."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    position = _finite(env_obs["robot0_eef_pos"], (2, 3), "robot0_eef_pos")
    rotation = _finite(env_obs["robot0_eef_rot_axis_angle"], (2, 3), "robot0_eef_rot_axis_angle")
    gripper = np.asarray(env_obs["robot0_gripper_width"])
    if gripper.shape == (2,):
        gripper = gripper[:, None]
    gripper = _finite(gripper, (2, 1), "robot0_gripper_width")
    timestamps = _finite(env_obs["timestamp"], (2,), "timestamp")
    if timestamps[1] <= timestamps[0]:
        raise ValueError("Observation timestamps must increase")
    world = np.concatenate((position, rotation, gripper), axis=-1)
    current_tcp = world[-1, :6].copy()
    poses = world_actions_to_model_actions(world, current_tcp, pose_repr=obs_pose_repr)
    start_relative = world_actions_to_model_actions(world, episode_start_pose, pose_repr="relative")
    image = np.asarray(env_obs["camera0_rgb"])
    if image.shape != (2, 224, 224, 3):
        raise ValueError("camera0_rgb must contain two THWC RGB 224x224 frames")
    if image.dtype != np.uint8 and (
        image.dtype.kind != "f" or not np.all(np.isfinite(image)) or np.any((image < 0) | (image > 1))
    ):
        raise ValueError("camera0_rgb must be uint8 or finite floats in [0, 1]")
    return {
        "robot0_eef_pos": poses[:, :3],
        "robot0_eef_rot_axis_angle": poses[:, 3:9],
        "robot0_eef_rot_axis_angle_wrt_start": start_relative[:, 3:9],
        "robot0_gripper_width": gripper.copy(),
        "camera0_rgb": np.moveaxis(image, -1, 1).copy(),
        "prompt": prompt,
        "_tcp_pose": current_tcp,
        "_observation_time": float(timestamps[-1]),
    }


def make_controller(policy, *, frequency, action_pose_repr, **options):
    """Require the scene profile before allowing its commands into a runner."""
    metadata = policy.get_server_metadata()
    rtc = metadata.get("rtc", {})
    if metadata.get("inference_profile") != "elec-meta" or any(
        rtc.get(key) != value
        for key, value in {
            "enabled": True,
            "mode": "client",
            "action_horizon": 16,
            "action_dim": 10,
            "model_action_dim": 32,
        }.items()
    ):
        raise ValueError("The server must expose the elec-meta client RTC profile (H=16, action=10, model=32)")
    # Validate the explicit coordinate convention before any hardware setup.
    world_actions_to_model_actions(np.zeros((1, 7)), np.zeros(6), pose_repr=action_pose_repr)
    return RTCController(
        policy,
        frequency=frequency,
        action_to_world=lambda actions, obs: model_actions_to_world_actions(
            actions,
            obs["_tcp_pose"],
            pose_repr=action_pose_repr,
        ),
        world_to_action=lambda world, obs: world_actions_to_model_actions(
            world,
            obs["_tcp_pose"],
            pose_repr=action_pose_repr,
        ),
        resample_world=interpolate_world_actions,
        server_metadata=metadata,
        **options,
    )


def hold_current_pose(env):
    """Replace queued trajectories with the measured TCP pose and gripper."""
    states = env.get_robot_state()
    for robot, state in zip(env.robots, states, strict=True):
        pose = _finite(state["ActualTCPPose"], (6,), "ActualTCPPose")
        gripper = float(state["gripper_position"])
        if not np.isfinite(gripper):
            raise ValueError("Measured gripper position must be finite")
        robot.servoL(pose=pose, gripper_pos=gripper, duration=max(0.1, 1.0 / float(robot.frequency)))


def _validate_limits(max_pos_speed, max_rot_speed, gripper_min, gripper_max):
    if any(not np.isfinite(value) or value <= 0 for value in (max_pos_speed, max_rot_speed)):
        raise ValueError("Position and rotation speed limits must be finite and positive")
    if not np.isfinite(gripper_min) or not np.isfinite(gripper_max) or not 0 <= gripper_min < gripper_max:
        raise ValueError("Gripper bounds must be finite with 0 <= minimum < maximum")


def validate_world_commands(
    actions,
    timestamps,
    current_pose,
    *,
    now,
    max_pos_speed=0.25,
    max_rot_speed=0.6,
    gripper_min=0.0,
    gripper_max=0.085,
):
    """Reject speed/gripper violations without changing commands or deadlines.

    Check the measured pose-to-first-deadline segment as well as every pair of
    successive waypoints. Orientations use shortest-path SO(3) displacement.
    These checks do not validate collision clearance, reachability, or IK.
    """
    _validate_limits(max_pos_speed, max_rot_speed, gripper_min, gripper_max)
    actions = np.asarray(actions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    current_pose = np.asarray(current_pose, dtype=np.float64)
    if (
        actions.ndim != 2
        or actions.shape[1] != 7
        or len(actions) == 0
        or timestamps.shape != (len(actions),)
        or current_pose.shape != (6,)
        or not np.all(np.isfinite(actions))
        or not np.all(np.isfinite(timestamps))
        or not np.all(np.isfinite(current_pose))
        or not np.isfinite(now)
    ):
        raise RTCError("World commands, current TCP pose, and deadlines must be finite with consistent shapes")
    intervals = np.diff(np.concatenate(([now], timestamps)))
    if np.any(intervals <= 0):
        raise RTCError("World command deadlines must increase strictly after the current time")
    if np.any((actions[:, 6] < gripper_min) | (actions[:, 6] > gripper_max)):
        raise RTCError("World commands exceed the configured gripper width bounds")
    positions = np.vstack((current_pose[:3], actions[:, :3]))
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=-1) / intervals
    if np.any(speeds > max_pos_speed + 1e-9):
        raise RTCError("World commands exceed the configured translation speed")
    rotations = Rotation.from_rotvec(np.vstack((current_pose[3:6], actions[:, 3:6])))
    angular_speeds = (rotations[1:] * rotations[:-1].inv()).magnitude() / intervals
    if np.any(angular_speeds > max_rot_speed + 1e-9):
        raise RTCError("World commands exceed the configured rotation speed")


def run_episode(
    env,
    controller,
    prepare,
    *,
    frequency,
    max_duration,
    clock=time.time,
    sleep=time.sleep,
    should_stop=lambda: False,
    sink=None,
    command_latency=0.05,
    max_pos_speed=0.25,
    max_rot_speed=0.6,
    gripper_min=0.0,
    gripper_max=0.085,
):
    """Run on the caller's thread; only inference belongs to the background worker.

    Before submission, enforce finite 7D commands, future deadlines, measured
    TCP-to-first-waypoint and adjacent-waypoint speed limits, and gripper bounds.
    Reject violations without retiming or clipping. Collision clearance and IK
    are not verified here. ``sink`` observes the plans submitted to the environment.
    """
    if not np.isfinite(frequency) or frequency <= 0 or not np.isfinite(max_duration) or max_duration <= 0:
        raise ValueError("frequency and max_duration must be finite and positive")
    if not np.isfinite(command_latency) or command_latency < 0:
        raise ValueError("command_latency must be finite and nonnegative")
    _validate_limits(max_pos_speed, max_rot_speed, gripper_min, gripper_max)
    submitted = 0
    recording = False
    latest_raw = None

    def submit(plan):
        nonlocal submitted
        actions = np.asarray(plan.actions, dtype=np.float64)
        timestamps = np.asarray(plan.timestamps, dtype=np.float64)
        if (
            actions.ndim != 2
            or actions.shape[1] != 7
            or timestamps.shape != (len(actions),)
            or not np.all(np.isfinite(actions))
            or not np.all(np.isfinite(timestamps))
            or np.any(np.diff(timestamps) <= 0)
        ):
            raise RTCError("RTC plan must contain finite 7D actions with increasing deadlines")
        if len(env.robots) == 1:
            states = env.get_robot_state()
            if len(states) != 1:
                raise RTCError("elec_meta requires exactly one measured robot state")
            current_pose = states[0]["ActualTCPPose"]
        elif not env.robots and latest_raw is not None:
            current_pose = np.concatenate(
                (
                    latest_raw["robot0_eef_pos"][-1],
                    latest_raw["robot0_eef_rot_axis_angle"][-1],
                )
            )
        else:
            raise RTCError("elec_meta supports one robot or a replay environment")
        now = clock()
        future = timestamps > now + command_latency
        if not np.any(future):
            raise RTCError("No future RTC waypoints remain at submission")
        public = TimedPlan(actions[future].copy(), timestamps[future].copy())
        validate_world_commands(
            public.actions,
            public.timestamps,
            current_pose,
            now=now,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
        )
        env.exec_actions(
            actions=public.actions,
            timestamps=public.timestamps,
            dynamic_latency=False,
            compensate_latency=False,
        )
        submitted += 1
        if sink is not None:
            sink(public)

    try:
        controller.warmup(prepare(env.get_obs()))
        if should_stop():
            return submitted
        latest_raw = env.get_obs()
        initial = prepare(latest_raw)
        started_at = clock()
        start_recording = getattr(env, "start_episode", None)
        if callable(start_recording):
            start_recording(started_at)
            recording = True
        submit(controller.start(initial, initial["_observation_time"]))
        while not should_stop() and clock() - started_at < max_duration:
            latest_raw = env.get_obs()
            observation = prepare(latest_raw)
            replacement = controller.poll(observation, observation["_observation_time"])
            if replacement is not None:
                submit(replacement)
            sleep(min(1.0 / frequency, max(0.0, max_duration - (clock() - started_at))))
        return submitted
    finally:
        try:
            hold_current_pose(env)
        finally:
            try:
                controller.stop()
            finally:
                if recording:
                    env.end_episode()


class _ReplayEnv:
    def __init__(self, path, frequency, clock):
        with np.load(path, allow_pickle=False) as source:
            self.data = {key: source[key].copy() for key in source.files}
        self.frequency, self.clock = frequency, clock
        self.origin = clock()
        self.snapshot = self.data["robot0_eef_pos"].ndim == 2
        self.count = 1 if self.snapshot else len(self.data["robot0_eef_pos"])
        self.robots = []
        self.plans = []

    def get_obs(self):
        elapsed = max(0.0, self.clock() - self.origin)
        index = min(int(elapsed * self.frequency), self.count - 1)
        result = {key: value.copy() if self.snapshot else value[index].copy() for key, value in self.data.items()}
        # Snapshot replay has no live sensor clock: preserve the two-frame
        # spacing on a synthetic capture grid at the current replay time.
        result["timestamp"] = self.clock() - np.array([1.0 / self.frequency, 0.0])
        return result

    def exec_actions(self, **kwargs):
        self.plans.append(kwargs)

    def get_robot_state(self):
        return []


class _MockPolicy:
    """Deterministic transport exercise, never a trained robot policy."""

    def get_server_metadata(self):
        return {
            "inference_profile": "elec-meta",
            "rtc": {
                "enabled": True,
                "mode": "client",
                "action_horizon": 16,
                "action_dim": 10,
                "model_action_dim": 32,
            },
        }

    def infer(self, observation):
        if "prev_action_chunk" in observation:
            previous = np.asarray(observation["prev_action_chunk"], dtype=np.float64)
            actions = np.concatenate((previous, np.repeat(previous[-1:], 16 - len(previous), axis=0)))
        else:
            actions = np.zeros((16, 10), dtype=np.float64)
            actions[:, 3], actions[:, 7] = 1.0, 1.0
            actions[:, 9] = np.asarray(observation["robot0_gripper_width"])[-1, 0]
        return {"actions": actions}


def main(argv=None, *, policy_factory=None, env_factory=None, clock=time.time, sleep=time.sleep):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", type=Path, help="NPZ observations; no hardware is imported or opened")
    mode.add_argument("--execute", action="store_true", help="Open the actual ARX environment")
    parser.add_argument("--mock-policy", action="store_true", help="Replay-only local interface check")
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--inference-timeout", type=float, default=120.0)
    parser.add_argument("--obs-pose-repr", required=True, choices=("abs", "rel", "relative"))
    parser.add_argument("--action-pose-repr", required=True, choices=("abs", "rel", "relative"))
    parser.add_argument("--frequency", type=float, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--command-latency", type=float, default=0.05)
    parser.add_argument("--max-inference-seconds", type=float, default=2.0)
    parser.add_argument("--initial-delay-steps", type=int, default=2)
    parser.add_argument("--max-pos-speed", type=float, default=0.25, help="Maximum TCP translation speed, m/s")
    parser.add_argument("--max-rot-speed", type=float, default=0.6, help="Maximum TCP angular speed, rad/s")
    parser.add_argument("--gripper-min-width", type=float, default=0.0)
    parser.add_argument("--gripper-max-width", type=float, default=0.085)
    parser.add_argument("--umi-deploy-root", type=Path)
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=8765)
    parser.add_argument("--camera-reorder", type=int, default=0)
    parser.add_argument("--camera-observation-latency", type=float, default=0.17)
    parser.add_argument("--robot-observation-latency", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("data/elec_meta_rtc"))
    args = parser.parse_args(argv)
    if not np.isfinite(args.frequency) or args.frequency <= 0:
        parser.error("--frequency must be finite and positive")
    if not np.isfinite(args.max_duration) or args.max_duration <= 0 or not args.prompt.strip():
        parser.error("--max-duration must be positive and --prompt must be nonempty")
    try:
        _validate_limits(args.max_pos_speed, args.max_rot_speed, args.gripper_min_width, args.gripper_max_width)
    except ValueError as error:
        parser.error(str(error))
    if args.mock_policy and args.execute:
        parser.error("--mock-policy is available only with --replay")
    if args.execute and args.umi_deploy_root is None and env_factory is None:
        parser.error("--execute requires --umi-deploy-root")
    if args.mock_policy:
        policy = _MockPolicy()
    else:
        if policy_factory is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: PLC0415

            policy_factory = WebsocketClientPolicy
        policy = policy_factory(
            host=args.policy_host,
            port=args.policy_port,
            inference_timeout=args.inference_timeout,
            ping_interval=None,
        )
    with ExitStack() as stack:
        close_policy = getattr(policy, "close", None)
        if callable(close_policy):
            stack.callback(close_policy)
        controller = stack.enter_context(
            make_controller(
                policy,
                frequency=args.frequency,
                action_pose_repr=args.action_pose_repr,
                replan_steps=args.replan_steps,
                command_latency=args.command_latency,
                max_inference_seconds=args.max_inference_seconds,
                initial_delay_steps=args.initial_delay_steps,
                clock=clock,
            )
        )
        if args.replay is not None:
            env = _ReplayEnv(args.replay, args.frequency, clock)
        elif env_factory is None:
            from multiprocessing.managers import SharedMemoryManager  # noqa: PLC0415

            sys.path.insert(0, str(args.umi_deploy_root.resolve()))
            from modules.arx5_env import Arx5Env  # noqa: PLC0415

            manager = stack.enter_context(SharedMemoryManager())
            args.output.mkdir(parents=True, exist_ok=True)
            env = stack.enter_context(
                Arx5Env(
                    output_dir=args.output,
                    frequency=args.frequency,
                    shm_manager=manager,
                    robots_config=[
                        {
                            "robot_type": "arx5",
                            "robot_ip": args.robot_ip,
                            "robot_port": args.robot_port,
                            "robot_obs_latency": args.robot_observation_latency,
                            "robot_action_latency": args.command_latency,
                        }
                    ],
                    camera_reorder=[args.camera_reorder],
                    obs_image_resolution=(224, 224),
                    obs_float32=True,
                    camera_obs_horizon=2,
                    robot_obs_horizon=2,
                    init_joints=False,
                    camera_obs_latency=args.camera_observation_latency,
                    enable_multi_cam_vis=False,
                )
            )
        else:
            env = stack.enter_context(env_factory(args))
        episode_start = None

        def prepare(observation):
            nonlocal episode_start
            if episode_start is None:
                episode_start = np.concatenate(
                    (
                        observation["robot0_eef_pos"][-1],
                        observation["robot0_eef_rot_axis_angle"][-1],
                    )
                )
            return prepare_observation(
                observation,
                episode_start,
                obs_pose_repr=args.obs_pose_repr,
                prompt=args.prompt,
            )

        count = run_episode(
            env,
            controller,
            prepare,
            frequency=args.frequency,
            max_duration=args.max_duration,
            clock=clock,
            sleep=sleep,
            command_latency=args.command_latency,
            max_pos_speed=args.max_pos_speed,
            max_rot_speed=args.max_rot_speed,
            gripper_min=args.gripper_min_width,
            gripper_max=args.gripper_max_width,
        )
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "replay",
                "plans_submitted": count,
                "hardware_commands": args.execute,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
