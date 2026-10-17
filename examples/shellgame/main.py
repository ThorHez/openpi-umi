from __future__ import annotations

import dataclasses
import json
import logging
import math
import pathlib
import sys
from argparse import Namespace
from collections import deque

import numpy as np
from PIL import Image
import tyro


@dataclasses.dataclass
class Args:
    # Policy server.
    host: str = "127.0.0.1"
    port: int = 8000
    replan_steps: int = 5

    # robosuite checkout. Defaults to the sibling workspace used in this thread.
    robosuite_root: str = "../robosuite"

    # ShellGame episode generation parameters. Keep these aligned with the
    # training data generation command unless intentionally testing OOD.
    num_trials: int = 20
    # Start index in the deterministic episode schedule. This permits exact
    # non-overlapping parallel shards while preserving a single seeded run.
    trial_start: int = 0
    seed: int = 0
    env: str = "ShellGame"
    robots: str = "Panda"
    camera: str = "opponentview"
    wrist_camera: str = "robot0_eye_in_hand"
    width: int = 512
    height: int = 512
    gpu_id: int = -1
    fps: int = 30
    image_rotation: int = 180
    initial_ball_cup: str = "random"
    min_swaps: int = 3
    max_swaps: int = 3
    reveal_frames: int = 10
    cover_frames: int = 10
    swap_frames: int = 10
    settle_frames: int = 10
    scripted_observation: bool = True
    control_during_scripted_observation: bool = True
    cup_spacing: float = 0.11
    cup_position_noise: float = 0.012
    cup_position_cross_noise: float = 0.006
    cup_outer_radius: float = 0.032
    cup_inner_radius: float = 0.024
    cup_half_height: float = 0.045
    cup_handle_radius: float = 0.0
    cup_handle_half_height: float = 0.0
    cup_color: str = "random"
    ball_radius: float = 0.014
    layout_axis: str = "y"
    reveal_ball_y_offset: float = -0.09
    reveal_cup_y_offset: float = -0.035
    reveal_cup_lift_height: float = 0.12
    swap_arc_offset: float = 0.12
    lift_retreat_offset: float = -0.08
    observe_eef_x: float = 0.20
    observe_eef_y: float = 0.0
    observe_eef_z: float = 1.05
    control_observe_orientation: bool = False
    observe_eef_roll: float = math.pi
    observe_eef_pitch: float = 0.0
    observe_eef_yaw: float = 0.0
    observe_eef_frames: int = 0

    # Policy input layout from pi0_mem_compress_evan_shellgame_openpi_umi_success_260703.
    num_frames: int = 32
    frame_stride: int = 5
    resize_size: int = 224
    action_horizon: int = 16
    action_dim: int = 10
    # history: Pi0Mem input with indexed historical frames.
    # single_frame: standard Pi0/Pi0.5 input with only the current two views.
    # mme_framesamp: MME-VLA current observation plus an explicit history buffer.
    policy_input_mode: str = "history"
    # pose10: legacy relative target pose + measured gripper width.
    # raw7: native robosuite controller command, passed directly to env.step().
    action_mode: str = "pose10"
    task: str = "Watch the shell game and lift the cup hiding the ball."
    phase_instructions: bool = False
    observe_task: str = "Observe the ball moving under a cup and remember which cup contains it."
    grasp_task: str = "Grasp and lift the cup containing the ball."
    observation_position_frame: str = "absolute"
    action_pose_frame: str = "current"
    rot6d_convention: str = "openpi"
    # Native OSC command convention. ``raw7`` checkpoints generated with
    # ``--osc-input-type absolute`` must use the same controller semantics at
    # evaluation time; otherwise absolute world poses are interpreted as
    # deltas and the rollout is invalid.
    osc_input_type: str = "delta"

    # Policy-control rollout.
    max_policy_steps: int = 120
    lift_success_height: float = 0.08
    # Cup-selection metric: after the scripted swaps/settle phase, ignore the
    # first N rollout frames, then vote for the cup underneath the end effector
    # during the following window. Frames farther than the XY radius from every
    # cup are ignored instead of being forced into a choice.
    cup_selection_skip_frames: int = 10
    cup_selection_window_frames: int = 30
    cup_selection_xy_radius: float = 0.06
    gripper_deadband: float = 0.004
    default_gripper_action: float = -1.0
    zero_action_policy: bool = False
    physics_debug: bool = False
    physics_debug_window: int = 30

    # Output.
    video_out_path: str = "data/shellgame/videos"
    save_videos: bool = True


def _import_shellgame_tools(robosuite_root: str):
    root = pathlib.Path(robosuite_root).expanduser().resolve()
    scripts = root / "robosuite" / "scripts"
    for path in (root, scripts):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import render_shellgame_episode as shell  # type: ignore

    return shell


def _episode_namespace(args: Args, *, seed: int, initial_ball_cup: str, num_swaps: int) -> Namespace:
    return Namespace(
        env=args.env,
        robots=args.robots,
        osc_input_type=args.osc_input_type,
        arm_controller_type="OSC_POSE",
        import_module=None,
        output="",
        camera=args.camera,
        wrist_camera=args.wrist_camera,
        no_wrist_camera=False,
        save_wrist_frames=False,
        image_rotation=args.image_rotation,
        width=args.width,
        height=args.height,
        gpu_id=args.gpu_id,
        seed=seed,
        fps=args.fps,
        initial_ball_cup=initial_ball_cup,
        num_swaps=num_swaps,
        reveal_frames=args.reveal_frames,
        cover_frames=args.cover_frames,
        swap_frames=args.swap_frames,
        settle_frames=args.settle_frames,
        lift_frames=10,
        cup_spacing=args.cup_spacing,
        cup_position_noise=args.cup_position_noise,
        cup_position_cross_noise=args.cup_position_cross_noise,
        cup_outer_radius=args.cup_outer_radius,
        cup_inner_radius=args.cup_inner_radius,
        cup_half_height=args.cup_half_height,
        cup_handle_radius=args.cup_handle_radius,
        cup_handle_half_height=args.cup_handle_half_height,
        cup_color=args.cup_color,
        ball_radius=args.ball_radius,
        layout_axis=args.layout_axis,
        reveal_ball_y_offset=args.reveal_ball_y_offset,
        reveal_cup_y_offset=args.reveal_cup_y_offset,
        reveal_cup_lift_height=args.reveal_cup_lift_height,
        swap_arc_offset=args.swap_arc_offset,
        lift_height=0.20,
        lift_retreat_offset=args.lift_retreat_offset,
        robot_reveal=True,
        attach_cup_to_gripper=False,
        robot_approach_frames=30,
        robot_descend_frames=20,
        robot_grasp_frames=10,
        robot_lift_frames=35,
        robot_hover_height=0.22,
        robot_grasp_z_offset=0.0,
        observe_eef_x=args.observe_eef_x,
        observe_eef_y=args.observe_eef_y,
        observe_eef_z=args.observe_eef_z,
        control_observe_orientation=args.control_observe_orientation,
        observe_eef_roll=args.observe_eef_roll,
        observe_eef_pitch=args.observe_eef_pitch,
        observe_eef_yaw=args.observe_eef_yaw,
        observe_eef_frames=args.observe_eef_frames,
        language_instruction=args.task,
        no_lift=False,
        no_video=True,
    )


def _resize_uint8(image: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[:2] == (size, size):
        # Camera observations may be backed by a renderer-owned readback
        # buffer.  History must own its pixels because the renderer can reuse
        # that storage on the next observation.
        return np.array(arr, dtype=np.uint8, order="C", copy=True)
    resized = Image.fromarray(arr).resize((size, size), Image.BICUBIC)
    return np.array(resized, dtype=np.uint8, order="C", copy=True)


def _quat_to_rot6d(quat_xyzw: np.ndarray, convention: str) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2mat

    mat = quat2mat(np.asarray(quat_xyzw, dtype=np.float64))
    if convention == "openpi":
        return mat[:2, :].reshape(6).astype(np.float32)
    if convention == "shellgame_legacy":
        return mat[:, :2].reshape(6).astype(np.float32)
    raise ValueError(f"Unknown rot6d convention: {convention}")


def _rot6d_to_matrix(rot6d: np.ndarray, convention: str) -> np.ndarray:
    if convention == "openpi":
        a = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
        x = a[0]
        y = a[1]
        x = x / max(np.linalg.norm(x), 1e-8)
        y = y - x * np.dot(x, y)
        y = y / max(np.linalg.norm(y), 1e-8)
        z = np.cross(x, y)
        return np.stack([x, y, z], axis=0)
    if convention != "shellgame_legacy":
        raise ValueError(f"Unknown rot6d convention: {convention}")

    a = np.asarray(rot6d, dtype=np.float64).reshape(3, 2)
    x = a[:, 0]
    y = a[:, 1]
    x = x / max(np.linalg.norm(x), 1e-8)
    y = y - x * np.dot(x, y)
    y = y / max(np.linalg.norm(y), 1e-8)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def _relative_rot6d_to_quat(base_quat_xyzw: np.ndarray, rel_rot6d: np.ndarray, convention: str) -> np.ndarray:
    from robosuite.utils.transform_utils import mat2quat, quat2mat

    target_mat = quat2mat(np.asarray(base_quat_xyzw, dtype=np.float64)) @ _rot6d_to_matrix(rel_rot6d, convention)
    return mat2quat(target_mat).astype(np.float32)


def _relative_pose10_to_world(
    shell,
    env,
    target10: np.ndarray,
    *,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    from robosuite.utils.transform_utils import mat2quat, quat2mat

    base_mat = np.eye(4, dtype=np.float64)
    base_mat[:3, :3] = quat2mat(np.asarray(base_quat, dtype=np.float64))
    base_mat[:3, 3] = np.asarray(base_pos, dtype=np.float64)

    rel_mat = np.eye(4, dtype=np.float64)
    rel_mat[:3, :3] = _rot6d_to_matrix(target10[3:9], convention)
    rel_mat[:3, 3] = np.asarray(target10[:3], dtype=np.float64)

    target_mat = base_mat @ rel_mat
    return target_mat[:3, 3].astype(np.float32), mat2quat(target_mat[:3, :3]).astype(np.float32)


def _gripper_width(gripper_state: np.ndarray) -> float:
    g = np.asarray(gripper_state, dtype=np.float32).reshape(-1)
    if g.size >= 2:
        return float(abs(g[0] - g[1]))
    if g.size == 1:
        return float(abs(g[0]))
    return 0.0


def _append_observation(
    shell,
    env,
    ep_args: Namespace,
    wrist_camera_name: str | None,
    history: list[dict],
    replay: list[np.ndarray],
    *,
    resize_size: int,
):
    obs = env._get_observations(force_update=True)
    base = shell.image_from_obs(obs, ep_args.camera, image_rotation=ep_args.image_rotation)
    wrist = None
    if wrist_camera_name is not None:
        wrist = shell.optional_image_from_obs(obs, wrist_camera_name, image_rotation=0)
    if wrist is None:
        raise RuntimeError("ShellGame wrist camera is unavailable; training/eval expects wrist frames.")

    history.append(
        {
            "base": _resize_uint8(base, resize_size),
            "wrist": _resize_uint8(wrist, resize_size),
            "eef_pos": np.asarray(shell.get_eef_pos(env), dtype=np.float32),
            "eef_quat": np.asarray(shell.get_eef_quat(env), dtype=np.float32),
            "gripper_width": _gripper_width(shell.obs_vector(obs, "robot0_gripper_qpos")),
        }
    )
    # Keep video frames independent from both the observation dictionary and
    # the EGL readback buffer.  Without an explicit copy, delayed video writes
    # can encode renderer-reused memory instead of the frame seen here.
    replay.append(np.array(base, dtype=np.uint8, order="C", copy=True))


def _zero_shellgame_object_velocities(env) -> None:
    """Clear free-joint velocities after scripted teleporting of cups / ball."""
    joint_names = []
    for cup in getattr(env, "cups", {}).values():
        joint_names.extend(getattr(cup, "joints", ()))
    ball = getattr(env, "ball", None)
    if ball is not None:
        joint_names.extend(getattr(ball, "joints", ()))

    for joint_name in joint_names:
        try:
            joint_id = env.sim.model.joint_name2id(joint_name)
            qvel_addr = int(env.sim.model.jnt_dofadr[joint_id])
            qvel_dim = 6 if int(env.sim.model.jnt_type[joint_id]) == 0 else 1
            env.sim.data.qvel[qvel_addr : qvel_addr + qvel_dim] = 0.0
        except Exception:
            logging.debug("Failed to clear qvel for joint %s", joint_name, exc_info=True)


def _zero_env_action(env, gripper_action: float) -> np.ndarray:
    action_low, action_high = env.action_spec
    action = np.zeros_like(action_low)
    if action.shape[0] >= 1:
        action[-1] = gripper_action
    return np.clip(action, action_low, action_high)


def _cup_positions(shell, env) -> dict[str, np.ndarray]:
    return {
        cup: np.asarray(shell.body_pos(env, env.cup_body_ids[cup]), dtype=np.float32)
        for cup in shell.CUP_NAMES
    }


def _window(history: list[dict], *, num_frames: int, frame_stride: int) -> list[dict]:
    cur = len(history) - 1
    indices = [cur - (num_frames - 1 - i) * frame_stride for i in range(num_frames)]
    indices = [max(0, min(cur, idx)) for idx in indices]
    return [history[idx] for idx in indices]


def _policy_action_dim(args: Args) -> int:
    if args.action_mode == "pose10":
        return args.action_dim
    if args.action_mode == "raw7":
        return 7
    raise ValueError(f"Unknown action_mode={args.action_mode!r}; expected 'pose10' or 'raw7'")


def _policy_input(
    history: list[dict],
    start_eef_pos: np.ndarray,
    *,
    args: Args,
    prompt: str | None = None,
) -> dict:
    cur = history[-1]
    if args.observation_position_frame == "absolute":
        observation_eef_pos = np.asarray(cur["eef_pos"], dtype=np.float32)
    elif args.observation_position_frame == "episode_start":
        observation_eef_pos = (cur["eef_pos"] - start_eef_pos).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown observation_position_frame={args.observation_position_frame!r}; "
            "expected 'absolute' or 'episode_start'"
        )
    rot6d = _quat_to_rot6d(cur["eef_quat"], args.rot6d_convention)
    identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    width = np.array([cur["gripper_width"]], dtype=np.float32)

    element = {
        "robot0_eef_pos": np.stack([observation_eef_pos, np.zeros(3, dtype=np.float32)], axis=0),
        "robot0_eef_rot_axis_angle": np.stack([rot6d, identity_rot6d], axis=0),
        "robot0_gripper_width": np.stack([width, width], axis=0),
        "actions": np.zeros((args.action_horizon, _policy_action_dim(args)), dtype=np.float32),
        "prompt": args.task if prompt is None else prompt,
    }
    if args.policy_input_mode == "history":
        frames = _window(history, num_frames=args.num_frames, frame_stride=args.frame_stride)
        for i, frame in enumerate(frames):
            element[f"left_wrist_0_rgb_0_{i}"] = frame["wrist"]
            element[f"left_wrist_0_rgb_1_{i}"] = frame["base"]
    elif args.policy_input_mode == "single_frame":
        element["left_wrist_0_rgb_0"] = cur["wrist"]
        element["left_wrist_0_rgb_1"] = cur["base"]
    elif args.policy_input_mode == "mme_framesamp":
        element = {
            "observation/image": cur["base"],
            "observation/wrist_image": cur["wrist"],
            "observation/state": np.concatenate(
                [
                    element["robot0_eef_pos"].reshape(-1),
                    element["robot0_eef_rot_axis_angle"].reshape(-1),
                    element["robot0_gripper_width"].reshape(-1),
                ]
            ).astype(np.float32),
            "prompt": element["prompt"],
        }
    else:
        raise ValueError(
            f"Unknown policy_input_mode={args.policy_input_mode!r}; "
            "expected 'history', 'single_frame', or 'mme_framesamp'"
        )
    return element


def _sync_mme_framesamp_buffer(
    client,
    history: list[dict],
    start_eef_pos: np.ndarray,
    *,
    args: Args,
) -> None:
    """Send every observation not yet seen by the online MME memory buffer."""

    sent = int(getattr(client, "_shellgame_history_sent", 0))
    if sent >= len(history):
        return
    new_frames = history[sent:]
    states = [
        _policy_input([frame], start_eef_pos, args=args)["observation/state"]
        for frame in new_frames
    ]
    response = client.add_buffer(
        {
            "add_buffer": True,
            "images": np.stack([frame["base"] for frame in new_frames])[:, None, ...],
            "state": np.stack(states).astype(np.float32),
            "exec_start_idx": 0,
        }
    )
    if not response.get("add_buffer_finished", False):
        raise RuntimeError(f"MME-VLA server rejected history buffer: {response}")
    client._shellgame_history_sent = len(history)


def _target_action_to_env_action(
    shell,
    env,
    target10: np.ndarray,
    *,
    start_eef_pos: np.ndarray,
    start_eef_quat: np.ndarray,
    plan_base_pos: np.ndarray | None,
    plan_base_quat: np.ndarray | None,
    last_gripper_action: float,
    deadband: float,
    args: Args,
) -> tuple[np.ndarray, float]:
    target10 = np.asarray(target10, dtype=np.float32).reshape(-1)
    action_low, action_high = env.action_spec
    if args.action_pose_frame == "current":
        if plan_base_pos is None or plan_base_quat is None:
            plan_base_pos = shell.get_eef_pos(env)
            plan_base_quat = shell.get_eef_quat(env)
        target_pos, target_quat = _relative_pose10_to_world(
            shell,
            env,
            target10,
            base_pos=plan_base_pos,
            base_quat=plan_base_quat,
            convention=args.rot6d_convention,
        )
    elif args.action_pose_frame == "episode_start":
        target_pos = start_eef_pos + target10[:3]
        target_quat = _relative_rot6d_to_quat(start_eef_quat, target10[3:9], args.rot6d_convention)
    else:
        raise ValueError(f"Unknown action_pose_frame={args.action_pose_frame!r}; expected 'episode_start' or 'current'")

    obs = env._get_observations(force_update=True)
    cur_width = _gripper_width(shell.obs_vector(obs, "robot0_gripper_qpos"))
    target_width = float(target10[9]) if target10.size > 9 else cur_width
    gripper_action = last_gripper_action
    if target_width < cur_width - deadband:
        gripper_action = 1.0
    elif target_width > cur_width + deadband:
        gripper_action = -1.0

    action = shell.make_robot_action(
        env,
        target_pos=target_pos,
        target_quat=target_quat,
        gripper_action=gripper_action,
    )
    return np.clip(action, action_low, action_high), gripper_action


def _raw_action_to_env_action(env, raw_action: np.ndarray) -> tuple[np.ndarray, float]:
    """Pass a native robosuite controller action through without pose reinterpretation."""
    action_low, action_high = env.action_spec
    action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if action.shape != action_low.shape or action_high.shape != action_low.shape:
        raise RuntimeError(
            "raw7 policy output must match env.action_spec exactly; "
            f"got action={action.shape}, low={action_low.shape}, high={action_high.shape}"
        )
    action = np.clip(action, action_low, action_high)
    return action, float(action[-1])


def _policy_env_action(
    shell,
    env,
    history: list[dict],
    start_eef_pos: np.ndarray,
    start_eef_quat: np.ndarray,
    action_plan: deque,
    gripper_action: float,
    *,
    client,
    args: Args,
    prompt: str | None = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    if args.zero_action_policy:
        return _zero_env_action(env, args.default_gripper_action), gripper_action, None

    if not action_plan:
        element = _policy_input(history, start_eef_pos, args=args, prompt=prompt)
        if client is None:
            raise RuntimeError("Policy client is required unless zero_action_policy is enabled.")
        if args.policy_input_mode == "mme_framesamp":
            _sync_mme_framesamp_buffer(client, history, start_eef_pos, args=args)
        action_dim = _policy_action_dim(args)
        if args.action_mode == "pose10":
            plan_base_pos = np.asarray(shell.get_eef_pos(env), dtype=np.float32)
            plan_base_quat = np.asarray(shell.get_eef_quat(env), dtype=np.float32)
        else:
            plan_base_pos = None
            plan_base_quat = None
        actions = np.asarray(client.infer(element)["actions"], dtype=np.float32)
        valid_shape = actions.ndim == 2 and actions.shape[-1] >= action_dim
        if args.action_mode in {"raw7", "joint8"}:
            # Fail loudly if the server loaded a policy with different action
            # semantics; truncating pose or joint actions would move the robot
            # with unrelated values.
            valid_shape = actions.ndim == 2 and actions.shape[-1] == action_dim
        if not valid_shape:
            raise RuntimeError(f"Policy returned unexpected actions shape {actions.shape}")
        for a in actions[: args.replan_steps, :action_dim]:
            action_plan.append((a, plan_base_pos, plan_base_quat))

    policy_action, plan_base_pos, plan_base_quat = action_plan.popleft()
    if args.action_mode == "raw7":
        env_action, gripper_action = _raw_action_to_env_action(env, policy_action)
    else:
        env_action, gripper_action = _target_action_to_env_action(
            shell,
            env,
            policy_action,
            start_eef_pos=start_eef_pos,
            start_eef_quat=start_eef_quat,
            plan_base_pos=plan_base_pos,
            plan_base_quat=plan_base_quat,
            last_gripper_action=gripper_action,
            deadband=args.gripper_deadband,
            args=args,
        )
    return env_action, gripper_action, policy_action


def _run_scripted_observation(
    shell,
    env,
    ep_args: Namespace,
    args: Args,
    history: list[dict],
    replay: list[np.ndarray],
    *,
    client=None,
):
    rng = np.random.default_rng(ep_args.seed)
    swaps = shell.sample_swaps(rng, ep_args.num_swaps)
    wrist_camera_name = shell.resolve_wrist_camera_name(env, ep_args.wrist_camera)
    control_policy = args.control_during_scripted_observation
    action_plan: deque = deque()
    gripper_action = args.default_gripper_action
    start_eef_pos: np.ndarray | None = None
    start_eef_quat: np.ndarray | None = None
    active_prompt: str | None = None
    remaining_scripted_frames = (
        ep_args.reveal_frames
        + ep_args.cover_frames
        + len(swaps) * ep_args.swap_frames
        + ep_args.settle_frames
    )

    def step_policy_from_current_observation() -> None:
        nonlocal gripper_action, start_eef_pos, start_eef_quat, active_prompt, remaining_scripted_frames
        remaining_scripted_frames = max(0, remaining_scripted_frames - 1)
        if not control_policy:
            return
        prompt = None
        if args.phase_instructions:
            # The raw-action converter aligns observation[i] with action[i+1]
            # and shifts its instruction likewise. Therefore the final
            # scripted observation must already request the first grasp action.
            prompt = args.grasp_task if remaining_scripted_frames == 0 else args.observe_task
        if prompt != active_prompt:
            # Never execute actions cached under the previous instruction.
            action_plan.clear()
            active_prompt = prompt
            logging.info(
                "policy prompt switched to %r (scripted_frames_remaining=%d)",
                prompt,
                remaining_scripted_frames,
            )
        if start_eef_pos is None or start_eef_quat is None:
            start_eef_pos = np.asarray(history[0]["eef_pos"], dtype=np.float32)
            start_eef_quat = np.asarray(history[0]["eef_quat"], dtype=np.float32)
        env_action, gripper_action, _ = _policy_env_action(
            shell,
            env,
            history,
            start_eef_pos,
            start_eef_quat,
            action_plan,
            gripper_action,
            client=client,
            args=args,
            prompt=prompt,
        )
        env.step(env_action)

    env.reset()
    shell.move_to_observation_pose(env, ep_args)
    cup_slots = {name: name for name in shell.CUP_NAMES}
    target_cup = ep_args.initial_ball_cup

    center_xy = {cup_name: shell.slot_xy(env, slot_name) for cup_name, slot_name in cup_slots.items()}
    lifted_target_cup = {target_cup: ep_args.reveal_cup_lift_height}
    reveal_cup_xy = dict(center_xy)
    reveal_ball_xy = shell.slot_xy(env, cup_slots[target_cup], offset=ep_args.reveal_ball_y_offset)
    for _ in range(ep_args.reveal_frames):
        env.set_shellgame_positions(reveal_cup_xy, target_cup, reveal_ball_xy, lifted_target_cup, forward=True)
        _zero_shellgame_object_velocities(env)
        _append_observation(shell, env, ep_args, wrist_camera_name, history, replay, resize_size=args.resize_size)
        step_policy_from_current_observation()

    cover_ball_return_fraction = 0.4
    for i in range(ep_args.cover_frames):
        t = 1.0 if ep_args.cover_frames <= 1 else i / (ep_args.cover_frames - 1)
        cup_xy = dict(center_xy)
        if t <= cover_ball_return_fraction:
            local_t = t / cover_ball_return_fraction
            ball_xy = (1.0 - local_t) * reveal_ball_xy + local_t * center_xy[target_cup]
            cup_z_offsets = lifted_target_cup
        else:
            local_t = (t - cover_ball_return_fraction) / (1.0 - cover_ball_return_fraction)
            ball_xy = center_xy[target_cup]
            cup_z_offsets = {target_cup: ep_args.reveal_cup_lift_height * (1.0 - local_t)}
        env.set_shellgame_positions(cup_xy, target_cup, ball_xy, cup_z_offsets, forward=True)
        _zero_shellgame_object_velocities(env)
        _append_observation(shell, env, ep_args, wrist_camera_name, history, replay, resize_size=args.resize_size)
        step_policy_from_current_observation()

    for swap_idx, (slot_a, slot_b) in enumerate(swaps):
        cup_a = shell.cup_in_slot(cup_slots, slot_a)
        cup_b = shell.cup_in_slot(cup_slots, slot_b)
        start_xy = {cup_name: shell.slot_xy(env, slot_name) for cup_name, slot_name in cup_slots.items()}
        end_slots = dict(cup_slots)
        end_slots[cup_a], end_slots[cup_b] = slot_b, slot_a
        end_xy = {cup_name: shell.slot_xy(env, slot_name) for cup_name, slot_name in end_slots.items()}

        for i in range(ep_args.swap_frames):
            t = 1.0 if ep_args.swap_frames <= 1 else (i + 1) / ep_args.swap_frames
            cup_xy = dict(start_xy)
            cup_xy[cup_a] = shell.interpolate_xy(
                start_xy[cup_a], end_xy[cup_a], t, side_offset=ep_args.swap_arc_offset, layout_axis=ep_args.layout_axis
            )
            cup_xy[cup_b] = shell.interpolate_xy(
                start_xy[cup_b], end_xy[cup_b], t, side_offset=-ep_args.swap_arc_offset, layout_axis=ep_args.layout_axis
            )
            ball_xy = cup_xy[target_cup]
            env.set_shellgame_positions(cup_xy, target_cup, ball_xy, forward=True)
            _zero_shellgame_object_velocities(env)
            _append_observation(shell, env, ep_args, wrist_camera_name, history, replay, resize_size=args.resize_size)
            step_policy_from_current_observation()

        cup_slots = end_slots

    settle_xy = {cup_name: shell.slot_xy(env, slot_name) for cup_name, slot_name in cup_slots.items()}
    settle_cup_pos = None
    for i in range(ep_args.settle_frames):
        env.set_shellgame_positions(settle_xy, target_cup, settle_xy[target_cup], forward=True)
        _zero_shellgame_object_velocities(env)
        if i == ep_args.settle_frames - 1:
            settle_cup_pos = _cup_positions(shell, env)
        _append_observation(shell, env, ep_args, wrist_camera_name, history, replay, resize_size=args.resize_size)
        step_policy_from_current_observation()

    if control_policy:
        _append_observation(shell, env, ep_args, wrist_camera_name, history, replay, resize_size=args.resize_size)

    if settle_cup_pos is None:
        settle_cup_pos = _cup_positions(shell, env)
    return {
        "target_cup": target_cup,
        "final_ball_cup": cup_slots[target_cup],
        "settle_xy": settle_xy,
        "settle_cup_pos": settle_cup_pos,
        "wrist_camera_name": wrist_camera_name,
        "swaps": swaps,
    }


def _initialize_direct_policy_rollout(shell, env, ep_args: Namespace, args: Args, history: list[dict], replay: list[np.ndarray]):
    wrist_camera_name = shell.resolve_wrist_camera_name(env, ep_args.wrist_camera)

    env.reset()
    _zero_shellgame_object_velocities(env)
    _append_observation(shell, env, ep_args, wrist_camera_name, history, replay, resize_size=args.resize_size)

    settle_xy = {cup_name: shell.slot_xy(env, cup_name) for cup_name in shell.CUP_NAMES}
    settle_cup_pos = _cup_positions(shell, env)
    target_cup = ep_args.initial_ball_cup
    return {
        "target_cup": target_cup,
        "final_ball_cup": target_cup,
        "settle_xy": settle_xy,
        "settle_cup_pos": settle_cup_pos,
        "wrist_camera_name": wrist_camera_name,
        "swaps": [],
    }


def _success(shell, env, target_cup: str, settle_cup_pos: dict[str, np.ndarray], lift_height: float) -> tuple[bool, dict]:
    cup_pos = _cup_positions(shell, env)
    lifts = {cup: float(cup_pos[cup][2] - settle_cup_pos[cup][2]) for cup in shell.CUP_NAMES}
    target_lift = lifts[target_cup]
    max_other = max(v for cup, v in lifts.items() if cup != target_cup)
    ok = target_lift >= lift_height and target_lift >= max_other + 0.02
    return ok, {"lifts": lifts, "target_lift": target_lift, "max_other_lift": max_other}


def _as_list(value) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _model_name(model, kind: str, idx: int) -> str:
    try:
        if kind == "body":
            return model.body_id2name(int(idx)) or f"body:{int(idx)}"
        if kind == "geom":
            return model.geom_id2name(int(idx)) or f"geom:{int(idx)}"
    except Exception:
        pass
    return f"{kind}:{int(idx)}"


def _object_qvel(env, obj) -> list[float]:
    joints = getattr(obj, "joints", ())
    if not joints:
        return []
    try:
        joint_id = env.sim.model.joint_name2id(joints[0])
        qvel_addr = int(env.sim.model.jnt_dofadr[joint_id])
        qvel_dim = 6 if int(env.sim.model.jnt_type[joint_id]) == 0 else 1
        return _as_list(env.sim.data.qvel[qvel_addr : qvel_addr + qvel_dim])
    except Exception:
        return []


def _physics_snapshot(shell, env, *, step: int, env_action: np.ndarray, target10, gripper_action: float, stats: dict) -> dict:
    model = env.sim.model
    object_body_ids = set(int(v) for v in env.cup_body_ids.values())
    if getattr(env, "ball_body_id", None) is not None:
        object_body_ids.add(int(env.ball_body_id))

    contacts = []
    for contact_idx in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[contact_idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if body1 not in object_body_ids and body2 not in object_body_ids:
            continue
        contacts.append(
            {
                "geom1": _model_name(model, "geom", geom1),
                "geom2": _model_name(model, "geom", geom2),
                "body1": _model_name(model, "body", body1),
                "body2": _model_name(model, "body", body2),
                "dist": float(contact.dist),
                "pos": _as_list(contact.pos),
            }
        )

    cup_pos = _cup_positions(shell, env)
    return {
        "step": int(step),
        "env_action": _as_list(env_action),
        "target10": None if target10 is None else _as_list(target10),
        "gripper_action": float(gripper_action),
        "eef_pos": _as_list(shell.get_eef_pos(env)),
        "cup_pos": {cup: _as_list(pos) for cup, pos in cup_pos.items()},
        "cup_qvel": {cup: _object_qvel(env, env.cups[cup]) for cup in shell.CUP_NAMES},
        "ball_pos": _as_list(shell.body_pos(env, env.ball_body_id)),
        "ball_qvel": _object_qvel(env, env.ball),
        "contacts": contacts,
        "success_stats": stats,
    }


def _gripper_contacted_cups(contacts: list[dict]) -> set[str]:
    """Return cup identities directly contacting either gripper finger/body."""
    touched: set[str] = set()
    for contact in contacts:
        bodies = (str(contact["body1"]), str(contact["body2"]))
        if not any("gripper" in body for body in bodies):
            continue
        for cup in ("left", "middle", "right"):
            if f"{cup}_cup_root" in bodies:
                touched.add(cup)
    return touched


def _current_gripper_contacted_cups(env) -> set[str]:
    """Collect direct gripper/cup contacts without enabling full debug traces."""

    model = env.sim.model
    contacts = []
    for contact_idx in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[contact_idx]
        body1 = int(model.geom_bodyid[int(contact.geom1)])
        body2 = int(model.geom_bodyid[int(contact.geom2)])
        contacts.append(
            {
                "body1": _model_name(model, "body", body1),
                "body2": _model_name(model, "body", body2),
            }
        )
    return _gripper_contacted_cups(contacts)


def _write_physics_debug(video_dir: pathlib.Path, trial: int, payload: dict) -> None:
    path = video_dir / "physics_debug" / f"trial_{trial:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("trial=%d physics_debug=%s", trial, path)


def _save_video(path: pathlib.Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, frames, fps=fps)
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        logging.warning("Failed to save video %s: %s", path, exc)


def eval_shellgame(args: Args) -> None:
    from openpi_client import websocket_client_policy as _websocket_client_policy

    shell = _import_shellgame_tools(args.robosuite_root)
    rng = np.random.default_rng(args.seed)
    if args.zero_action_policy:
        client = None
    elif args.policy_input_mode == "mme_framesamp":
        client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(args.host, args.port)
    else:
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    video_dir = pathlib.Path(args.video_out_path)
    successes = 0
    cup_selection_correct = 0
    cup_selection_decisions = 0
    correct_selection_and_contact = 0
    target_cup_contacts = 0
    any_cup_contacts = 0
    target_cup_lifts = 0
    any_cup_lifts = 0
    episode_results: list[dict] = []

    if args.phase_instructions:
        logging.info("phase instructions enabled: observe=%r grasp=%r", args.observe_task, args.grasp_task)

    if args.trial_start < 0:
        raise ValueError(f"trial_start must be non-negative, got {args.trial_start}")

    for schedule_trial in range(args.trial_start + args.num_trials):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        initial = str(rng.choice(shell.CUP_NAMES)) if args.initial_ball_cup == "random" else args.initial_ball_cup
        num_swaps = int(rng.integers(args.min_swaps, args.max_swaps + 1)) if args.scripted_observation else 0
        if schedule_trial < args.trial_start:
            continue
        trial = schedule_trial
        local_trial = schedule_trial - args.trial_start
        ep_args = _episode_namespace(args, seed=episode_seed, initial_ball_cup=initial, num_swaps=num_swaps)

        env = shell.make_env(ep_args)
        history: list[dict] = []
        replay: list[np.ndarray] = []
        physics_trace: deque[dict] = deque(maxlen=max(1, int(args.physics_debug_window)))
        try:
            if args.policy_input_mode == "mme_framesamp" and client is not None:
                response = client.reset()
                if not response.get("reset_finished", False):
                    raise RuntimeError(f"MME-VLA server reset failed: {response}")
                client._shellgame_history_sent = 0
            if args.scripted_observation:
                meta = _run_scripted_observation(shell, env, ep_args, args, history, replay, client=client)
            else:
                meta = _initialize_direct_policy_rollout(shell, env, ep_args, args, history, replay)

            start_eef_pos = np.asarray(history[0]["eef_pos"], dtype=np.float32)
            start_eef_quat = np.asarray(history[0]["eef_quat"], dtype=np.float32)
            action_plan: deque[np.ndarray] = deque()
            gripper_action = args.default_gripper_action
            selection_votes = {cup: 0 for cup in shell.CUP_NAMES}
            selection_distance_sums = {cup: 0.0 for cup in shell.CUP_NAMES}
            selection_window_end = args.cup_selection_skip_frames + args.cup_selection_window_frames
            gripper_contacted_cups: set[str] = set()

            for _step in range(args.max_policy_steps):
                target10_debug = None
                env_action, gripper_action, target10_debug = _policy_env_action(
                    shell,
                    env,
                    history,
                    start_eef_pos,
                    start_eef_quat,
                    action_plan,
                    gripper_action,
                    client=client,
                    args=args,
                    prompt=args.grasp_task if args.phase_instructions else None,
                )
                env.step(env_action)
                gripper_contacted_cups.update(_current_gripper_contacted_cups(env))
                _append_observation(
                    shell,
                    env,
                    ep_args,
                    meta["wrist_camera_name"],
                    history,
                    replay,
                    resize_size=args.resize_size,
                )

                if args.cup_selection_skip_frames <= _step < selection_window_end:
                    eef_xy = np.asarray(history[-1]["eef_pos"][:2], dtype=np.float32)
                    distances = {
                        cup: float(
                            np.linalg.norm(
                                eef_xy - np.asarray(meta["settle_cup_pos"][cup][:2], dtype=np.float32)
                            )
                        )
                        for cup in shell.CUP_NAMES
                    }
                    nearest_cup = min(distances, key=distances.get)
                    if distances[nearest_cup] <= args.cup_selection_xy_radius:
                        selection_votes[nearest_cup] += 1
                        selection_distance_sums[nearest_cup] += distances[nearest_cup]

                ok, stats = _success(shell, env, meta["target_cup"], meta["settle_cup_pos"], args.lift_success_height)
                if args.physics_debug:
                    snapshot = _physics_snapshot(
                        shell,
                        env,
                        step=_step,
                        env_action=env_action,
                        target10=target10_debug,
                        gripper_action=gripper_action,
                        stats=stats,
                    )
                    physics_trace.append(snapshot)
                if ok:
                    successes += 1
                    logging.info("trial=%d success target=%s final_slot=%s stats=%s", trial, meta["target_cup"], meta["final_ball_cup"], stats)
                    break
            else:
                ok, stats = _success(shell, env, meta["target_cup"], meta["settle_cup_pos"], args.lift_success_height)
                successes += int(ok)
                logging.info("trial=%d success=%s target=%s final_slot=%s stats=%s", trial, ok, meta["target_cup"], meta["final_ball_cup"], stats)

            max_votes = max(selection_votes.values(), default=0)
            selected_cup = None
            if max_votes > 0:
                candidates = [cup for cup, votes in selection_votes.items() if votes == max_votes]
                selected_cup = min(
                    candidates,
                    key=lambda cup: selection_distance_sums[cup] / selection_votes[cup],
                )
                cup_selection_decisions += 1
                cup_selection_correct += int(selected_cup == meta["target_cup"])
            logging.info(
                "trial=%d cup_selection=%s target=%s correct=%s votes=%s",
                trial,
                selected_cup if selected_cup is not None else "no_decision",
                meta["target_cup"],
                selected_cup == meta["target_cup"],
                selection_votes,
            )

            selected_correct = selected_cup == meta["target_cup"]
            target_contact = meta["target_cup"] in gripper_contacted_cups
            any_contact = bool(gripper_contacted_cups)
            target_lift = float(stats["lifts"][meta["target_cup"]]) >= args.lift_success_height
            any_lift = max(stats["lifts"].values()) >= args.lift_success_height
            correct_selection_and_contact += int(selected_correct and target_contact)
            target_cup_contacts += int(target_contact)
            any_cup_contacts += int(any_contact)
            target_cup_lifts += int(target_lift)
            any_cup_lifts += int(any_lift)
            episode_results.append(
                {
                    "trial": trial,
                    "episode_seed": episode_seed,
                    "target_cup": meta["target_cup"],
                    "selected_cup": selected_cup,
                    "cup_selection_correct": selected_correct,
                    "gripper_contacted_cups": sorted(gripper_contacted_cups),
                    "target_cup_contact": target_contact,
                    "any_cup_contact": any_contact,
                    "correct_selection_and_contact": selected_correct and target_contact,
                    "target_lift_success": target_lift,
                    "any_cup_lift_success": any_lift,
                    "strict_success": bool(ok),
                    "lifts": stats["lifts"],
                }
            )

            if args.physics_debug:
                _write_physics_debug(
                    video_dir,
                    trial,
                    {
                        "trial": trial,
                        "episode_seed": episode_seed,
                        "initial_ball_cup": initial,
                        "num_swaps": num_swaps,
                        "target_cup": meta["target_cup"],
                        "final_ball_cup": meta["final_ball_cup"],
                        "success": bool(ok),
                        "selected_cup": selected_cup,
                        "cup_selection_correct": selected_cup == meta["target_cup"],
                        "gripper_contacted_cups": sorted(gripper_contacted_cups),
                        "any_cup_contact": bool(gripper_contacted_cups),
                        "target_cup_contact": meta["target_cup"] in gripper_contacted_cups,
                        "selected_cup_contact": selected_cup in gripper_contacted_cups,
                        "correct_selection_and_contact": (
                            selected_cup == meta["target_cup"]
                            and meta["target_cup"] in gripper_contacted_cups
                        ),
                        "cup_selection_votes": selection_votes,
                        "final_stats": stats,
                        "zero_action_policy": bool(args.zero_action_policy),
                        "trace": list(physics_trace),
                    },
                )

            if args.save_videos:
                suffix = "success" if ok else "failure"
                _save_video(video_dir / f"trial_{trial:04d}_{suffix}.mp4", replay, args.fps)
        finally:
            env.close()

        logging.info(
            "running success rate: %d/%d = %.1f%% | cup selection accuracy: %d/%d = %.1f%% "
            "(decisions=%d/%d)",
            successes,
            local_trial + 1,
            successes / (local_trial + 1) * 100.0,
            cup_selection_correct,
            local_trial + 1,
            cup_selection_correct / (local_trial + 1) * 100.0,
            cup_selection_decisions,
            local_trial + 1,
        )

    logging.info(
        "final success rate: %d/%d = %.1f%% | cup selection accuracy: %d/%d = %.1f%% "
        "(decisions=%d/%d)",
        successes,
        args.num_trials,
        successes / max(args.num_trials, 1) * 100.0,
        cup_selection_correct,
        args.num_trials,
        cup_selection_correct / max(args.num_trials, 1) * 100.0,
        cup_selection_decisions,
        args.num_trials,
    )
    result = {
        "protocol": dataclasses.asdict(args),
        "num_trials": args.num_trials,
        "strict_success": successes,
        "cup_selection_correct": cup_selection_correct,
        "cup_selection_decisions": cup_selection_decisions,
        "correct_selection_and_contact": correct_selection_and_contact,
        "target_cup_contact": target_cup_contacts,
        "any_cup_contact": any_cup_contacts,
        "target_cup_lift_success": target_cup_lifts,
        "any_cup_lift_success": any_cup_lifts,
        "episodes": episode_results,
    }
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logging.info("wrote aggregate result: %s", video_dir / "result.json")

def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    eval_shellgame(tyro.cli(Args))


if __name__ == "__main__":
    main()
