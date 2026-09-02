"""Paired V10 / fine-tune / Oracle handoff diagnostic for absolute EEF7.

Each condition and episode runs in a fresh MuJoCo/EGL process.  Diffusion
noise is explicitly seeded per (episode seed, global query index), so the V10
prefix is bitwise controlled across ``v10_full``, ``v10_to_ft499``, and
``v10_to_oracle``.  The handoff happens on the replan=8 boundary at policy
step 80.

The Oracle suffix exactly matches the V10 on-policy correction writer:
10 recenter, 30 descend, 15 grasp, and then the same cosine lift.  The cup
crosses the 80 mm success threshold at step 154, so the rollout stops at 155
steps before this driver's renderer readback becomes unstable.  Success at the
original 150-step budget is reported separately.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import math
import multiprocessing as mp
import pathlib

import main as base
import main_absolute_eef_fixed_history as fixed_eef
import numpy as np
import tyro


CUP_NAMES = ("left", "middle", "right")
CONDITIONS = (
    "v10_full",
    "ft499_full",
    "v10_to_ft499",
    "v10_to_oracle",
    "recorded_v10_to_ft499",
    "recorded_v10_to_ft499_to_v10",
)
NOISE_SEED_KEY = "__openpi_deterministic_noise_seed__"
_RENDERER_REFRESH_INTERVAL = 64


@dataclasses.dataclass
class Args(base.Args):
    num_trials: int = 20
    seed: int = 260813
    video_out_path: str = "evaluation/shellgame/v10_ft_oracle_handoff_paired"
    physics_debug: bool = True
    physics_debug_window: int = 155
    websocket_reconnect_interval: int = 4
    renderer_refresh_interval: int = 64

    secondary_port: int = 8141
    episode_indices: str = "0,1,2,3,4,7,9,12,16,17"
    conditions: str = ",".join(CONDITIONS)
    switch_step: int = 80
    second_switch_step: int = 96
    original_budget_steps: int = 150
    deterministic_sample_salt: int = 260820
    recorded_prefix_trace_root: str = (
        "evaluation/shellgame/"
        "eef7_v10_ft499_oracle_step80_paired10_seed260813_260820/v10_full"
    )

    oracle_recenter_steps: int = 10
    oracle_descend_steps: int = 30
    oracle_grasp_steps: int = 15
    oracle_lift_steps: int = 40
    oracle_hover_height: float = 0.05
    oracle_lift_height: float = 0.20


_base_policy_env_action = base._policy_env_action  # noqa: SLF001
_base_run_scripted_observation = base._run_scripted_observation  # noqa: SLF001
_base_physics_snapshot = base._physics_snapshot  # noqa: SLF001
_base_append_observation = base._append_observation  # noqa: SLF001


def _cosine_progress(index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    linear = index / (count - 1)
    return 0.5 - 0.5 * math.cos(math.pi * linear)


def _append_observation_orientation_stable(*args, **kwargs):
    """Refresh EGL periodically and correct an intermittent 180-degree readback.

    This robosuite / MuJoCo build starts returning striped pixels after roughly
    180 observations from one offscreen context.  Recreating only the renderer
    leaves the simulator state and controller untouched, while bounding every
    context to substantially fewer readbacks.
    """
    env = args[1]
    history = args[4]
    refresh_interval = _RENDERER_REFRESH_INTERVAL
    if history and len(history) % refresh_interval == 0:
        from robosuite.utils.binding_utils import MjRenderContextOffscreen

        old_context = env.sim._render_context_offscreen  # noqa: SLF001
        env.sim._render_context_offscreen = None  # noqa: SLF001
        del old_context
        gc.collect()
        MjRenderContextOffscreen(
            env.sim,
            device_id=int(env.render_gpu_device_id),
        )
        env.sim._render_context_offscreen.vopt.geomgroup[0] = (  # noqa: SLF001
            1 if env.render_collision_mesh else 0
        )
        env.sim._render_context_offscreen.vopt.geomgroup[1] = (  # noqa: SLF001
            1 if env.render_visual_mesh else 0
        )
        env._paired_renderer_refresh_count = (  # noqa: SLF001
            int(getattr(env, "_paired_renderer_refresh_count", 0)) + 1
        )
        logging.info(
            "Refreshed EGL offscreen context before observation=%d count=%d",
            len(history),
            env._paired_renderer_refresh_count,  # noqa: SLF001
        )
    _base_append_observation(*args, **kwargs)
    replay = args[5]
    if len(history) < 2:
        return
    corrected_keys = []
    for key in ("base", "wrist"):
        previous = np.asarray(history[-2][key], dtype=np.int16)
        current = np.asarray(history[-1][key], dtype=np.uint8)
        rotated = np.ascontiguousarray(np.rot90(current, 2))
        direct_delta = float(np.mean(np.abs(current.astype(np.int16) - previous)))
        rotated_delta = float(np.mean(np.abs(rotated.astype(np.int16) - previous)))
        if rotated_delta + 20.0 < direct_delta:
            history[-1][key] = rotated
            corrected_keys.append(key)
            if key == "base":
                replay[-1] = np.ascontiguousarray(np.rot90(replay[-1], 2))
    if corrected_keys:
        counts = getattr(env, "_paired_orientation_counts", {"base": 0, "wrist": 0})
        for key in corrected_keys:
            counts[key] += 1
        env._paired_orientation_counts = counts  # noqa: SLF001
        if counts["base"] + counts["wrist"] <= 4:
            logging.warning("Corrected 180-degree readback flip keys=%s", corrected_keys)


def _run_scripted_observation_with_state(*args, **kwargs):
    meta = _base_run_scripted_observation(*args, **kwargs)
    env = args[1]
    ep_args = args[2]
    history = args[4]
    env._paired_target_cup = meta["target_cup"]  # noqa: SLF001
    env._paired_state = {  # noqa: SLF001
        "policy_step": -1,
        "canonical_quat": np.asarray(history[0]["eef_quat"], dtype=np.float64),
        "grasp_z_offset": float(ep_args.robot_grasp_z_offset),
        "oracle_reference": None,
        "last_event": None,
    }
    return meta


def _initialize_oracle_reference(shell, env, args: Args, state: dict) -> dict:
    target_cup = env._paired_target_cup  # noqa: SLF001
    cup = np.asarray(base._cup_positions(shell, env)[target_cup], dtype=np.float64)  # noqa: SLF001
    measured = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
    grasp_z = float(env.cup_handle_grasp_z() + state["grasp_z_offset"])
    return {
        "switch_eef": measured,
        "target_xy": cup[:2].copy(),
        "hover_pos": np.array([cup[0], cup[1], grasp_z + args.oracle_hover_height]),
        "grasp_pos": np.array([cup[0], cup[1], grasp_z]),
        "lift_pos": np.array([cup[0], cup[1], grasp_z + args.oracle_lift_height]),
    }


def _oracle_reference(args: Args, state: dict, suffix_step: int) -> tuple[str, np.ndarray, bool]:
    ref = state["oracle_reference"]
    recenter_end = args.oracle_recenter_steps
    descend_end = recenter_end + args.oracle_descend_steps
    grasp_end = descend_end + args.oracle_grasp_steps
    lift_end = grasp_end + args.oracle_lift_steps
    if suffix_step < recenter_end:
        progress = _cosine_progress(suffix_step, args.oracle_recenter_steps)
        target = (1.0 - progress) * ref["switch_eef"] + progress * ref["hover_pos"]
        return "recenter", target, False
    if suffix_step < descend_end:
        index = suffix_step - recenter_end
        progress = _cosine_progress(index, args.oracle_descend_steps)
        target = (1.0 - progress) * ref["hover_pos"] + progress * ref["grasp_pos"]
        return "descend", target, False
    if suffix_step < grasp_end:
        return "grasp", ref["grasp_pos"].copy(), True
    index = min(suffix_step - grasp_end, args.oracle_lift_steps - 1)
    progress = _cosine_progress(index, args.oracle_lift_steps)
    target = (1.0 - progress) * ref["grasp_pos"] + progress * ref["lift_pos"]
    stage = "lift" if suffix_step < lift_end else "lift_hold"
    return stage, target, True


def _paired_policy_env_action(
    shell,
    env,
    history,
    start_eef_pos,
    start_eef_quat,
    action_plan,
    gripper_action,
    *,
    client,
    args: Args,
    prompt=None,
):
    state = env._paired_state  # noqa: SLF001
    state["policy_step"] += 1
    policy_step = int(state["policy_step"])
    condition = args._active_condition

    if policy_step == args.switch_step:
        state["oracle_reference"] = _initialize_oracle_reference(shell, env, args, state)

    use_oracle = condition == "v10_to_oracle" and policy_step >= args.switch_step
    use_recorded_prefix = (
        condition in {"recorded_v10_to_ft499", "recorded_v10_to_ft499_to_v10"}
        and policy_step < args.switch_step
    )
    if use_recorded_prefix:
        action_plan.clear()
        env_action = np.asarray(args._recorded_prefix_actions[policy_step], dtype=np.float32)
        next_gripper = float(env_action[-1])
        policy_action = env_action.copy()
        source = "recorded_v10"
        query_index = None
        stage = "recorded_prefix"
    elif use_oracle:
        action_plan.clear()
        suffix_step = policy_step - args.switch_step
        stage, target_pos, close = _oracle_reference(args, state, suffix_step)
        env_action = np.asarray(
            shell.make_robot_action(
                env,
                target_pos=target_pos,
                target_quat=state["canonical_quat"],
                gripper_action=1.0 if close else -1.0,
            ),
            dtype=np.float32,
        )
        next_gripper = 1.0 if close else -1.0
        policy_action = env_action.copy()
        source = "oracle"
        query_index = None
    else:
        env_action, next_gripper, policy_action = _base_policy_env_action(
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
        client_event = getattr(client, "_last_event", {})
        source = client_event.get("route", "unknown")
        query_index = client_event.get("query_index")
        stage = "model"

    reference = None
    if policy_step >= args.switch_step:
        if state["oracle_reference"] is None:
            state["oracle_reference"] = _initialize_oracle_reference(shell, env, args, state)
        ref_stage, ref_pos, ref_close = _oracle_reference(
            args, state, policy_step - args.switch_step
        )
        command = np.asarray(env_action, dtype=np.float64)
        reference = {
            "stage": ref_stage,
            "target_pos": ref_pos.tolist(),
            "close": bool(ref_close),
            "command_xy_error_mm": float(np.linalg.norm(command[:2] - ref_pos[:2]) * 1_000.0),
            "command_z_error_mm": float(abs(command[2] - ref_pos[2]) * 1_000.0),
            "gripper_mismatch": bool((float(command[-1]) > 0.0) != ref_close),
        }

    state["last_event"] = {
        "condition": condition,
        "policy_step": policy_step,
        "control_source": source,
        "query_index": query_index,
        "oracle_stage": stage if use_oracle else None,
        "oracle_reference": reference,
    }
    return env_action, next_gripper, policy_action


def _physics_snapshot_with_pairing(*args, **kwargs):
    payload = _base_physics_snapshot(*args, **kwargs)
    shell = args[0]
    env = args[1]
    state = getattr(env, "_paired_state", None)
    if state is None:
        return payload
    target_cup = env._paired_target_cup  # noqa: SLF001
    target = np.asarray(base._cup_positions(shell, env)[target_cup], dtype=np.float64)  # noqa: SLF001
    measured = np.asarray(shell.get_eef_pos(env), dtype=np.float64)
    payload["paired"] = state["last_event"]
    payload["target_cup"] = target_cup
    payload["target_cup_pos"] = target.tolist()
    payload["eef_target_xy_error_mm"] = float(np.linalg.norm(measured[:2] - target[:2]) * 1_000.0)
    payload["eef_target_dz_mm"] = float((measured[2] - target[2]) * 1_000.0)
    payload["orientation_stabilization_counts"] = getattr(
        env, "_paired_orientation_counts", {"base": 0, "wrist": 0}
    )
    payload["renderer_refresh_count"] = int(
        getattr(env, "_paired_renderer_refresh_count", 0)
    )
    return payload


def _episode_specs(args: Args) -> list[dict]:
    requested = sorted({int(item.strip()) for item in args.episode_indices.split(",") if item.strip()})
    if not requested or requested[0] < 0 or requested[-1] >= args.num_trials:
        raise ValueError("--episode-indices must be non-empty and inside [0, num-trials)")
    rng = np.random.default_rng(args.seed)
    all_specs = []
    for episode_index in range(args.num_trials):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        initial = str(rng.choice(CUP_NAMES)) if args.initial_ball_cup == "random" else args.initial_ball_cup
        num_swaps = int(rng.integers(args.min_swaps, args.max_swaps + 1))
        all_specs.append(
            {
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "initial_ball_cup": initial,
                "num_swaps": num_swaps,
            }
        )
    return [all_specs[index] for index in requested]


def _run_episode(args: Args, *, condition: str, spec: dict) -> None:
    from openpi_client import websocket_client_policy

    global _RENDERER_REFRESH_INTERVAL  # noqa: PLW0603
    _RENDERER_REFRESH_INTERVAL = int(args.renderer_refresh_interval)

    episode_index = int(spec["episode_index"])
    episode_dir = pathlib.Path(args.video_out_path) / condition / f"episode_{episode_index:04d}"
    child_args = dataclasses.replace(
        args,
        num_trials=1,
        seed=0,
        initial_ball_cup=spec["initial_ball_cup"],
        min_swaps=spec["num_swaps"],
        max_swaps=spec["num_swaps"],
        video_out_path=str(episode_dir),
    )
    child_args._active_condition = condition
    if condition in {"recorded_v10_to_ft499", "recorded_v10_to_ft499_to_v10"}:
        trace_path = (
            pathlib.Path(args.recorded_prefix_trace_root).expanduser().resolve()
            / f"episode_{episode_index:04d}"
            / "physics_debug"
            / "trial_0000.json"
        )
        trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
        prefix = [row for row in trace_payload["trace"] if int(row["step"]) < args.switch_step]
        if [int(row["step"]) for row in prefix] != list(range(args.switch_step)):
            raise RuntimeError(f"{trace_path}: incomplete recorded prefix")
        child_args._recorded_prefix_actions = np.asarray(
            [row["env_action"] for row in prefix], dtype=np.float32
        )

    original_episode_namespace = base._episode_namespace  # noqa: SLF001

    def episode_namespace(current_args, *, seed, initial_ball_cup, num_swaps):
        del seed
        return original_episode_namespace(
            current_args,
            seed=spec["episode_seed"],
            initial_ball_cup=initial_ball_cup,
            num_swaps=num_swaps,
        )

    base._episode_namespace = episode_namespace  # noqa: SLF001
    base._policy_input = fixed_eef._fixed_history_policy_input  # noqa: SLF001
    # Long EGL rollouts intermittently return a frame rotated by 180 degrees.
    # This continuity check is required for valid post-step-120 observations;
    # the rollout budget is separately capped before true striped corruption.
    base._append_observation = _append_observation_orientation_stable  # noqa: SLF001
    base._run_scripted_observation = _run_scripted_observation_with_state  # noqa: SLF001
    base._policy_env_action = _paired_policy_env_action  # noqa: SLF001
    base._physics_snapshot = _physics_snapshot_with_pairing  # noqa: SLF001

    original_client = websocket_client_policy.WebsocketClientPolicy

    class PairedWebsocketClient:
        def __init__(self, host="0.0.0.0", port=None, api_key=None):
            self._host = host
            self._api_key = api_key
            self._ports = {"v10": int(port), "ft499": int(args.secondary_port)}
            required = (
                {"ft499"}
                if condition in {"ft499_full", "recorded_v10_to_ft499"}
                else {"v10"}
            )
            if condition == "v10_to_ft499":
                required = {"v10", "ft499"}
            if condition == "recorded_v10_to_ft499_to_v10":
                required = {"v10", "ft499"}
            self._clients = {
                route: original_client(host, self._ports[route], api_key)
                for route in required
            }
            self._route_calls = dict.fromkeys(required, 0)
            # Preserve the same suffix diffusion seeds as a normal V10 prefix,
            # which would have made switch_step/replan_steps policy queries.
            self._calls = (
                args.switch_step // args.replan_steps
                if condition in {"recorded_v10_to_ft499", "recorded_v10_to_ft499_to_v10"}
                else 0
            )
            self._last_event = {}

        def _route(self) -> str:
            if condition in {"ft499_full", "recorded_v10_to_ft499"}:
                return "ft499"
            if condition == "recorded_v10_to_ft499_to_v10":
                if self._calls * args.replan_steps < args.second_switch_step:
                    return "ft499"
                return "v10"
            if condition == "v10_to_ft499" and self._calls * args.replan_steps >= args.switch_step:
                return "ft499"
            return "v10"

        def _reconnect(self, route: str) -> None:
            self._clients[route]._ws.close()  # noqa: SLF001
            self._clients[route] = original_client(
                self._host, self._ports[route], self._api_key
            )

        def infer(self, observation):
            route = self._route()
            route_calls = int(self._route_calls[route])
            if route_calls and route_calls % args.websocket_reconnect_interval == 0:
                self._reconnect(route)
            sample_seed = int(
                np.random.SeedSequence(
                    [args.deterministic_sample_salt, int(spec["episode_seed"]), self._calls]
                ).generate_state(1, dtype=np.uint32)[0]
            ) & 0x7FFF_FFFF
            query_index = self._calls
            self._calls += 1
            self._route_calls[route] = route_calls + 1
            self._last_event = {
                "route": route,
                "query_index": query_index,
                "sample_seed": sample_seed,
            }
            return self._clients[route].infer({**observation, NOISE_SEED_KEY: sample_seed})

        def get_server_metadata(self):
            return next(iter(self._clients.values())).get_server_metadata()

        def reset(self):
            for client in self._clients.values():
                client.reset()

    websocket_client_policy.WebsocketClientPolicy = PairedWebsocketClient
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info(
        "paired handoff condition=%s episode=%d seed=%d switch=%d",
        condition,
        episode_index,
        spec["episode_seed"],
        args.switch_step,
    )
    base.eval_shellgame(child_args)


def _is_success_stats(stats: dict, threshold: float) -> bool:
    return (
        float(stats["target_lift"]) >= threshold
        and float(stats["target_lift"]) >= float(stats["max_other_lift"]) + 0.02
    )


def _read_result(root: pathlib.Path, condition: str, spec: dict, args: Args) -> dict:
    episode_index = int(spec["episode_index"])
    episode_dir = root / condition / f"episode_{episode_index:04d}"
    trace_path = episode_dir / "physics_debug" / "trial_0000.json"
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    videos = sorted(episode_dir.glob("trial_0000_*.mp4"))
    trace = payload["trace"]
    first_success_step = next(
        (
            int(step["step"])
            for step in trace
            if _is_success_stats(step["success_stats"], args.lift_success_height)
        ),
        None,
    )
    success_at_original_budget = first_success_step is not None and first_success_step < args.original_budget_steps
    after_switch = [step for step in trace if int(step["step"]) >= args.switch_step]
    close_steps = [step for step in trace if float(step["env_action"][-1]) > 0.0]
    first_close = close_steps[0] if close_steps else None
    oracle_errors = [
        step["paired"]["oracle_reference"]
        for step in after_switch
        if step.get("paired") and step["paired"].get("oracle_reference") is not None
    ]
    # ``switch_step`` is the first action produced by the suffix controller.
    # The state presented to that controller is therefore the snapshot after
    # the preceding action, not the already-updated snapshot at switch_step.
    switch_snapshot = next(
        (step for step in trace if int(step["step"]) == args.switch_step - 1),
        None,
    )
    sources = {}
    for step in trace:
        event = step.get("paired") or {}
        source = event.get("control_source")
        if source is not None:
            sources[source] = sources.get(source, 0) + 1
    return {
        **spec,
        "condition": condition,
        "target_cup": payload["target_cup"],
        "final_ball_cup": payload["final_ball_cup"],
        "selected_cup": payload["selected_cup"],
        "cup_selection_correct": bool(payload["cup_selection_correct"]),
        "success_at_150": bool(success_at_original_budget),
        "success_at_extended_budget": bool(payload["success"]),
        "first_success_step": first_success_step,
        "final_stats": payload["final_stats"],
        "trace_steps": len(trace),
        "control_source_steps": sources,
        "switch_xy_error_mm": None if switch_snapshot is None else switch_snapshot["eef_target_xy_error_mm"],
        "switch_dz_mm": None if switch_snapshot is None else switch_snapshot["eef_target_dz_mm"],
        "post_switch_mean_xy_error_mm": (
            None if not after_switch else float(np.mean([step["eef_target_xy_error_mm"] for step in after_switch]))
        ),
        "post_switch_min_xy_error_mm": (
            None if not after_switch else float(np.min([step["eef_target_xy_error_mm"] for step in after_switch]))
        ),
        "first_close_step": None if first_close is None else int(first_close["step"]),
        "first_close_xy_error_mm": None if first_close is None else first_close["eef_target_xy_error_mm"],
        "first_close_dz_mm": None if first_close is None else first_close["eef_target_dz_mm"],
        "mean_oracle_command_xy_error_mm": (
            None if not oracle_errors else float(np.mean([item["command_xy_error_mm"] for item in oracle_errors]))
        ),
        "mean_oracle_command_z_error_mm": (
            None if not oracle_errors else float(np.mean([item["command_z_error_mm"] for item in oracle_errors]))
        ),
        "oracle_gripper_mismatch_rate": (
            None if not oracle_errors else float(np.mean([item["gripper_mismatch"] for item in oracle_errors]))
        ),
        "orientation_stabilization_counts": (
            {"base": 0, "wrist": 0}
            if not trace
            else trace[-1].get("orientation_stabilization_counts", {"base": 0, "wrist": 0})
        ),
        "video": str(videos[0]) if videos else None,
        "physics_trace": str(trace_path),
    }


def _mean_optional(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return None if not values else float(np.mean(values))


def _aggregate(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "selection_correct": sum(row["cup_selection_correct"] for row in rows),
        "successes_at_150": sum(row["success_at_150"] for row in rows),
        "successes_at_extended_budget": sum(
            row["success_at_extended_budget"] for row in rows
        ),
        "mean_switch_xy_error_mm": _mean_optional(rows, "switch_xy_error_mm"),
        "mean_switch_dz_mm": _mean_optional(rows, "switch_dz_mm"),
        "mean_post_switch_xy_error_mm": _mean_optional(rows, "post_switch_mean_xy_error_mm"),
        "mean_first_close_xy_error_mm": _mean_optional(rows, "first_close_xy_error_mm"),
        "mean_first_close_dz_mm": _mean_optional(rows, "first_close_dz_mm"),
        "mean_oracle_command_xy_error_mm": _mean_optional(rows, "mean_oracle_command_xy_error_mm"),
        "mean_oracle_command_z_error_mm": _mean_optional(rows, "mean_oracle_command_z_error_mm"),
        "mean_oracle_gripper_mismatch_rate": _mean_optional(rows, "oracle_gripper_mismatch_rate"),
    }


def main(args: Args) -> None:
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    if not conditions or any(item not in CONDITIONS for item in conditions):
        raise ValueError(f"--conditions must contain only {CONDITIONS}")
    if args.replan_steps != 8 or args.switch_step % args.replan_steps:
        raise ValueError("This paired diagnostic requires replan=8 and an aligned switch step")
    if args.second_switch_step % args.replan_steps:
        raise ValueError("second-switch-step must align with replan_steps")
    if args.second_switch_step <= args.switch_step:
        raise ValueError("second-switch-step must be later than switch-step")
    min_oracle_lift_steps = 20
    min_oracle_budget = (
        args.switch_step
        + args.oracle_recenter_steps
        + args.oracle_descend_steps
        + args.oracle_grasp_steps
        + min_oracle_lift_steps
    )
    if args.max_policy_steps < min_oracle_budget:
        raise ValueError(
            "max-policy-steps must reach the Oracle 80 mm lift threshold; "
            f"need at least {min_oracle_budget}"
        )
    if args.physics_debug_window < args.max_policy_steps:
        raise ValueError("physics-debug-window must retain the complete rollout")
    if args.num_frames != fixed_eef.TOTAL_FRAMES or args.frame_stride != 1:
        raise ValueError("This diagnostic requires 61 frames with stride 1")
    if args.action_mode != "raw7" or args.action_dim != 7 or args.osc_input_type != "absolute":
        raise ValueError("This diagnostic requires absolute raw7 actions")
    if args.websocket_reconnect_interval <= 0:
        raise ValueError("websocket-reconnect-interval must be positive")
    if args.renderer_refresh_interval <= 0:
        raise ValueError("renderer-refresh-interval must be positive")
    if any(
        condition in {"recorded_v10_to_ft499", "recorded_v10_to_ft499_to_v10"}
        for condition in conditions
    ):
        trace_root = pathlib.Path(args.recorded_prefix_trace_root).expanduser().resolve()
        if not trace_root.is_dir():
            raise FileNotFoundError(trace_root)

    root = pathlib.Path(args.video_out_path)
    root.mkdir(parents=True, exist_ok=True)
    specs = _episode_specs(args)
    context = mp.get_context("spawn")
    results = []
    for condition in conditions:
        for spec in specs:
            episode_dir = root / condition / f"episode_{int(spec['episode_index']):04d}"
            if episode_dir.exists():
                raise FileExistsError(f"Refusing to overwrite {episode_dir}")
            process = context.Process(
                target=_run_episode,
                kwargs={"args": args, "condition": condition, "spec": spec},
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"condition={condition} episode={spec['episode_index']} exited {process.exitcode}"
                )
            result = _read_result(root, condition, spec, args)
            results.append(result)
            logging.info(
                "aggregate condition=%s completed=%d/%d success@150=%s success@extended=%s selection=%s",
                condition,
                sum(item["condition"] == condition for item in results),
                len(specs),
                result["success_at_150"],
                result["success_at_extended_budget"],
                result["cup_selection_correct"],
            )

    by_condition = {
        condition: {
            "overall": _aggregate([row for row in results if row["condition"] == condition]),
            "by_target_cup": {
                cup: _aggregate(
                    [
                        row
                        for row in results
                        if row["condition"] == condition and row["target_cup"] == cup
                    ]
                )
                for cup in CUP_NAMES
            },
        }
        for condition in conditions
    }
    summary = {
        "experiment": "deterministic V10/FT499/Oracle step-80 handoff paired diagnostic",
        "seed": args.seed,
        "episode_indices": [spec["episode_index"] for spec in specs],
        "conditions": conditions,
        "switch_step": args.switch_step,
        "replan_steps": args.replan_steps,
        "max_policy_steps": args.max_policy_steps,
        "extended_budget_steps": args.max_policy_steps,
        "original_budget_steps": args.original_budget_steps,
        "deterministic_sample_salt": args.deterministic_sample_salt,
        "oracle_suffix_steps": {
            "recenter": args.oracle_recenter_steps,
            "descend": args.oracle_descend_steps,
            "grasp": args.oracle_grasp_steps,
            "lift": args.oracle_lift_steps,
        },
        "by_condition": by_condition,
        "results": results,
    }
    result_path = root / "result.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("paired handoff result=%s", result_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
